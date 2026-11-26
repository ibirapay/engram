//! Recall orchestration: the keyword, vector and graph legs, and their fusion.
//!
//! The graph leg reads the SQLite native graph (`graph.backend: sqlite`); the keyword
//! leg reads the persistent keyword index; the vector leg reads Qdrant.

use engram_config::Config;
use engram_graph::GraphClient;
use engram_models::OpenAiCompatibleClient;
use engram_retrieval::index::KeywordIndex;
use engram_retrieval::{bm25, rrf, rrf_weighted};
use engram_store::{Memory, load, load_file};
use engram_tenant::{GraphScope, Tenant};
use engram_vector::{Corpus, QdrantClient, Scope};
use serde::Serialize;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Mutex, OnceLock},
    time::Duration,
};

#[derive(Debug, Serialize)]
pub struct ResultItem {
    pub file: String,
    pub name: String,
    pub description: String,
    pub sources: Vec<String>,
    /// The facts that led to THIS memory.
    ///
    /// Facts are kept per memory. Flattening everything into one global list throws
    /// away the attribution: a caller could see twelve facts and not know which
    /// memory any of them came from.
    pub facts: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct Output {
    pub query: String,
    pub results: Vec<ResultItem>,
    /// Every fact from the graph leg, deduped — retained for callers that just
    /// want context. Per-record attribution lives on each [`ResultItem`].
    pub facts: Vec<String>,
    /// Related memories (1 hop). Kept in the response contract; the SQLite graph does
    /// not produce them, so it is empty.
    pub neighbours: Vec<String>,
    pub legs: HashMap<String, String>,
}

/// What the graph leg produced.
struct GraphLeg {
    files: Vec<String>,
    /// file -> facts, so attribution survives fusion.
    facts_by_file: HashMap<String, Vec<String>>,
    /// Facts not attributable to a single file (the native token leg).
    loose_facts: Vec<String>,
    /// 1-hop related memory names, when the backend reports them.
    neighbours: Vec<String>,
}

impl GraphLeg {
    fn empty() -> Self {
        Self {
            files: Vec::new(),
            facts_by_file: HashMap::new(),
            loose_facts: Vec::new(),
            neighbours: Vec::new(),
        }
    }

    /// All facts, deduped, for `Output::facts`.
    fn all_facts(&self) -> Vec<String> {
        let mut facts: Vec<String> = self
            .facts_by_file
            .values()
            .flatten()
            .chain(self.loose_facts.iter())
            .cloned()
            .collect();
        facts.sort();
        facts.dedup();
        facts
    }
}

pub async fn recall(
    config_path: &Path,
    tenant: &Tenant,
    slug: &str,
    query: &str,
    k: usize,
) -> Result<Output, String> {
    let config = Config::load(config_path).map_err(|error| error.to_string())?;
    recall_with_config(config, config_path, tenant, slug, query, k).await
}

/// Recall for the prompt hook: the cheap legs only.
///
/// Fast mode runs the keyword index and the vector leg, plus ONE graph fact query,
/// so a prompt stays well inside its 2.5s budget.
pub async fn recall_fast(
    config_path: &Path,
    tenant: &Tenant,
    slug: &str,
    query: &str,
    k: usize,
) -> Result<Output, String> {
    let config = Config::load(config_path).map_err(|error| error.to_string())?;
    recall_with_backend(config, config_path, tenant, slug, query, k, Mode::Fast).await
}

/// Which legs to run.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    /// Graph facts, keyword and semantic legs.
    Full,
    /// Local + vector legs plus a single graph fact query.
    Fast,
}

async fn recall_with_config(
    config: Config,
    config_path: &Path,
    tenant: &Tenant,
    slug: &str,
    query: &str,
    k: usize,
) -> Result<Output, String> {
    recall_with_backend(config, config_path, tenant, slug, query, k, Mode::Full).await
}

async fn recall_with_backend(
    config: Config,
    config_path: &Path,
    tenant: &Tenant,
    slug: &str,
    query: &str,
    k: usize,
    mode: Mode,
) -> Result<Output, String> {
    // Authorise before reading anything. `graph_scope` refuses a slug this
    // tenant does not own, which is the one place operator input (`--slug`)
    // could ask one agent's process to read another agent's store. Doing it here
    // rather than inside each leg means a refusal cannot be bypassed by a leg
    // that forgot to check — and the markdown store below is read straight off
    // disk, where no database filter would have helped.
    let scope = tenant
        .graph_scope(slug)
        .map_err(|error| error.to_string())?;
    let store = engram_paths::store_dir(config_path, slug);
    // The store must be readable even when no leg reads it whole: a wrong slug or an
    // unmounted store is an error, not "no matches". Only the directory is opened.
    std::fs::read_dir(&store).map_err(|error| {
        format!(
            "could not read the memory store at {}: {error}",
            store.display()
        )
    })?;
    let mut legs = HashMap::new();
    let backend = graph_backend(&config);
    let keyword = keyword_leg(config_path, &store, slug, query, k * 2, &mut legs)?;
    let vector = {
        // In fast mode the vector leg gets its OWN, tighter deadline.
        //
        // The hook wraps this entire call in `recall.inject.timeout_ms`, so a
        // slow embedding endpoint used to consume the whole budget and the
        // prompt got NOTHING — including the local BM25 result that was already
        // computed in ~66ms from markdown on disk. The slowest leg starving the
        // fastest is the wrong failure: incomplete beats empty.
        //
        // This is not hypothetical. A tenanted host indexes one store per
        // tenant, so the embedding endpoint sees far more load than it did when
        // the daemon owned a single store — and a queued request behind three
        // indexing passes blew the 2.5s budget on every prompt.
        let leg_budget = (mode == Mode::Fast)
            .then(|| Duration::from_millis(config.recall.inject.timeout_ms.max(250) * 2 / 5));
        vector_leg(&config, tenant, query, slug, k * 2, &mut legs, leg_budget).await
    };
    let graph_limit = k * 2;
    let graph = match mode {
        Mode::Full => {
            native_leg(
                &config,
                config_path,
                &backend,
                &scope,
                query,
                graph_limit,
                &mut legs,
            )
            .await
        }
        Mode::Fast => fast_graph_leg(&config, config_path, &scope, query, &mut legs).await,
    };
    let ranked: Vec<String> = {
        // Weights and the RRF constant come from config; they were hardcoded to
        // equal weighting and 60.0 while engram.yaml documented both. The order
        // of the rankings below must match the weight lookup.
        let weight_of = |leg: &str| {
            config
                .recall
                .hybrid
                .weights
                .get(leg)
                .copied()
                .unwrap_or(1.0)
        };
        rrf_weighted(
            &[keyword.clone(), vector.clone(), graph.files.clone()],
            &[
                weight_of("keyword"),
                weight_of("vector"),
                weight_of("graph"),
            ],
            k,
            config.recall.hybrid.k_rrf,
        )
        .into_iter()
        .map(|hit| hit.file)
        .collect()
    };
    // Only the files that survived fusion are read: `load_file` refuses any name that
    // is not a bare memory file name, so a hostile vector payload or graph node cannot
    // point this at a path outside the store.
    let mut by_file: HashMap<String, Memory> = HashMap::new();
    for file in &ranked {
        if by_file.contains_key(file) {
            continue;
        }
        if let Ok(Some(memory)) = load_file(&store, file) {
            by_file.insert(file.clone(), memory);
        }
    }
    // A graph hit for a file that is not on disk in this slug's store is dropped.
    // Say so, rather than silently returning fewer results than the leg found:
    // with the wrong slug this produced zero results and a healthy-looking report.
    let dropped = ranked
        .iter()
        .filter(|file| !by_file.contains_key(*file))
        .count();
    if dropped > 0 {
        legs.insert(
            "store".into(),
            format!(
                "{dropped} hit(s) had no file in {} — wrong slug?",
                store.display()
            ),
        );
    }
    let results = ranked
        .into_iter()
        .filter_map(|file| {
            let memory = by_file.get(&file)?;
            let mut sources = Vec::new();
            if keyword.contains(&file) {
                sources.push("keyword".into());
            }
            if vector.contains(&file) {
                sources.push("vector".into());
            }
            if graph.files.contains(&file) {
                sources.push("graph".into());
            }
            Some(ResultItem {
                file: memory.file.clone(),
                name: memory.name.clone(),
                description: memory.description.clone(),
                sources,
                facts: graph.facts_by_file.get(&file).cloned().unwrap_or_default(),
            })
        })
        .collect();
    Ok(Output {
        query: query.into(),
        results,
        facts: graph.all_facts(),
        neighbours: graph.neighbours.clone(),
        legs,
    })
}

/// The keyword leg: BM25 over the persistent SQLite index, falling back to scanning
/// the store when the index cannot be opened or written (read-only disk, locked by a
/// long writer). Same ranking either way; only the cost differs.
fn keyword_leg(
    config_path: &Path,
    store: &Path,
    slug: &str,
    query: &str,
    limit: usize,
    legs: &mut HashMap<String, String>,
) -> Result<Vec<String>, String> {
    let indexed = KeywordIndex::open(&engram_paths::keyword_index_path(config_path, slug))
        .and_then(|mut index| {
            index.sync(store)?;
            index.rank(query, limit)
        });
    match indexed {
        Ok(hits) => {
            legs.insert("keyword".into(), "ok".into());
            Ok(hits.into_iter().map(|hit| hit.file).collect())
        }
        Err(error) => {
            let memories = load(store).map_err(|error| {
                format!(
                    "could not read the memory store at {}: {error}",
                    store.display()
                )
            })?;
            legs.insert(
                "keyword".into(),
                format!("ok (scanned the store: keyword index unavailable: {error})"),
            );
            Ok(bm25(&memories, query, limit)
                .into_iter()
                .map(|hit| hit.file)
                .collect())
        }
    }
}

fn graph_backend(config: &Config) -> String {
    std::env::var("ENGRAM_GRAPH_BACKEND")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| config.graph.backend.clone())
}

async fn vector_leg(
    config: &Config,
    tenant: &Tenant,
    query: &str,
    slug: &str,
    k: usize,
    legs: &mut HashMap<String, String>,
    budget: Option<Duration>,
) -> Vec<String> {
    if !config.vector_store.enabled {
        legs.insert("vector".into(), "disabled".into());
        return Vec::new();
    }
    let result = async {
        let embeddings = OpenAiCompatibleClient::new(config.embed_endpoint())?
            .with_api_key(config.embed.api_key.present())
            .with_timeout(config.embed_timeout_seconds());
        // Query prefix, not the document prefix: an asymmetric model indexed with
        // one and queried with neither searches a different part of the space.
        let vector = embeddings
            .embedding(&config.embed.model, &config.query_text(query))
            .await?;
        // Scope to the active embedding space always, and to the slug when
        // configured: a reindex after a model change leaves both spaces in the
        // collection at once, and vectors from two models cannot be compared.
        //
        // The TENANT boundary is the collection, not a filter clause — see
        // QdrantClient::for_tenant. `scope_to_slug: false` therefore widens the
        // search across this tenant's own stores and no further.
        let space = config.embedding_space_id();
        let scope = match config.recall.scope_to_slug {
            true => Scope::slug(&space, slug),
            false => Scope::space(&space),
        };
        let hits = QdrantClient::for_tenant(config, tenant, Corpus::Memory)
            .search(vector, k, scope)
            .await?;
        Ok::<_, Box<dyn std::error::Error>>(hits)
    };
    // Unbounded when no budget is given (an explicit CLI/MCP call may wait);
    // bounded in fast mode, where a prompt may not.
    let result = match budget {
        Some(budget) => tokio::time::timeout(budget, result).await,
        None => Ok(result.await),
    };
    match result {
        Ok(Ok(hits)) => {
            legs.insert("vector".into(), "ok".into());
            hits.into_iter().map(|hit| hit.file).collect()
        }
        Ok(Err(error)) => {
            legs.insert("vector".into(), error.to_string());
            Vec::new()
        }
        Err(_) => {
            legs.insert(
                "vector".into(),
                format!(
                    "timed out after {}ms; the local legs still ran",
                    budget.unwrap_or_default().as_millis()
                ),
            );
            Vec::new()
        }
    }
}

/// Entity-ish words for the graph lookup: 4+ chars, case-insensitively deduped.
/// The cap is generous on purpose: at 6, prose prompts spent it on filler words
/// ("that", "when") and the real entity further in was never looked up. Every
/// token goes in one UNWIND query, so cost is flat in token count. Mirrors
/// `memory_recall.graph_facts`.
fn graph_query_tokens(query: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .filter(|word| word.len() >= 4 && seen.insert(word.to_lowercase()))
        // Same function-word filter the BM25 leg uses: "esta"/"como" reached
        // the fact query and matched "restarts" through a substring.
        .filter(|word| !engram_retrieval::is_stopword(word))
        .take(32)
        .map(str::to_string)
        .collect()
}

async fn native_leg(
    config: &Config,
    config_path: &Path,
    backend: &str,
    scope: &GraphScope,
    query: &str,
    k: usize,
    legs: &mut HashMap<String, String>,
) -> GraphLeg {
    if engram_config::Graph::is_legacy(backend) {
        legs.insert(
            "graph".into(),
            engram_config::Graph::legacy_message(backend),
        );
        return GraphLeg::empty();
    }
    let client = match GraphClient::connect(config_path) {
        Ok(client) => client,
        Err(error) => {
            legs.insert("graph".into(), format!("native: {error}"));
            return GraphLeg::empty();
        }
    };
    let tokens = graph_query_tokens(query);

    // Each sub-leg's failure is recorded. These were all `unwrap_or_default()`
    // followed by an unconditional legs["graph"] = "ok", so an auth failure, a
    // schema error or a dead database all reported as a healthy empty result.
    let mut problems = Vec::new();
    let mut leg = GraphLeg::empty();

    match client.native_facts_for_tokens(scope, &tokens, 6).await {
        Ok(facts) => leg.loose_facts = facts,
        Err(error) => problems.push(format!("facts: {error}")),
    }

    let keyword = match client.native_keyword_files(scope, query, k).await {
        Ok(hits) => hits,
        Err(error) => {
            problems.push(format!("keyword: {error}"));
            Vec::new()
        }
    };

    let semantic = match embed_query(config, query).await {
        Ok(vector) => match client.native_semantic_files(scope, &vector, k).await {
            Ok(hits) => hits,
            Err(error) => {
                problems.push(format!("semantic: {error}"));
                Vec::new()
            }
        },
        Err(error) => {
            problems.push(format!("embedding: {error}"));
            Vec::new()
        }
    };

    // Keep each hit's own facts rather than melting them into one list.
    for hit in keyword.iter().chain(semantic.iter()) {
        let entry = leg.facts_by_file.entry(hit.file.clone()).or_default();
        for fact in &hit.facts {
            if !entry.contains(fact) {
                entry.push(fact.clone());
            }
        }
    }
    leg.files = rrf(
        &[
            keyword.iter().map(|hit| hit.file.clone()).collect(),
            semantic.iter().map(|hit| hit.file.clone()).collect(),
        ],
        k,
        config.recall.hybrid.k_rrf,
    )
    .into_iter()
    .map(|hit| hit.file)
    .collect();

    legs.insert(
        "graph".into(),
        if problems.is_empty() {
            "ok".into()
        } else {
            format!("native degraded — {}", problems.join("; "))
        },
    );
    leg
}

/// One graph query for facts about the query's tokens, no ranking, no child
/// process: one fact query against the SQLite graph.
async fn fast_graph_leg(
    config: &Config,
    config_path: &Path,
    scope: &GraphScope,
    query: &str,
    legs: &mut HashMap<String, String>,
) -> GraphLeg {
    let mut leg = GraphLeg::empty();
    let tokens = graph_query_tokens(query);
    if tokens.is_empty() {
        legs.insert("graph".into(), "fast: no usable query tokens".into());
        return leg;
    }
    let backend = graph_backend(config);
    if engram_config::Graph::is_legacy(&backend) {
        legs.insert(
            "graph".into(),
            engram_config::Graph::legacy_message(&backend),
        );
        return leg;
    }
    let client = match GraphClient::connect(config_path) {
        Ok(client) => client,
        Err(error) => {
            legs.insert("graph".into(), format!("fast: {error}"));
            return leg;
        }
    };
    let facts = client
        .native_facts_for_tokens(scope, &tokens, config.recall.inject.max_facts.max(1))
        .await;
    match facts {
        Ok(mut facts) => {
            facts.sort();
            facts.dedup();
            leg.loose_facts = facts;
            legs.insert("graph".into(), "fast facts".into());
        }
        Err(error) => {
            legs.insert("graph".into(), format!("fast: {error}"));
        }
    }
    leg
}

async fn embed_query(config: &Config, query: &str) -> Result<Vec<f32>, String> {
    OpenAiCompatibleClient::new(config.embed_endpoint())
        .map_err(|error| error.to_string())?
        .with_api_key(config.embed.api_key.present())
        .with_timeout(config.embed_timeout_seconds())
        .embedding(&config.embed.model, &config.query_text(query))
        .await
        .map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_tokens_reach_an_entity_past_the_filler() {
        let tokens = graph_query_tokens(
            "I'm noticing that when i start a chat qdrant recall and neo4j recall isn't working",
        );
        assert!(tokens.iter().any(|t| t == "neo4j"), "{tokens:?}");
        assert_eq!(
            tokens.iter().filter(|t| *t == "recall").count(),
            1,
            "{tokens:?}"
        );
    }

    fn config(yaml: &str) -> Config {
        serde_yaml::from_str(yaml).unwrap()
    }

    /// The pre-tenancy identity (the single legacy group), which most tests use.
    fn legacy_tenant() -> Tenant {
        Tenant::resolve(&config("backend: ollama\n"), None).unwrap()
    }

    #[test]
    fn facts_stay_attributed_to_their_memory() {
        let mut leg = GraphLeg::empty();
        leg.files = vec!["a.md".into(), "b.md".into()];
        leg.facts_by_file
            .insert("a.md".into(), vec!["a1".into(), "shared".into()]);
        leg.facts_by_file
            .insert("b.md".into(), vec!["shared".into()]);
        leg.loose_facts = vec!["token-fact".into()];

        assert_eq!(leg.facts_by_file["a.md"], vec!["a1", "shared"]);
        // the global list is deduped across sources
        assert_eq!(leg.all_facts(), vec!["a1", "shared", "token-fact"]);
    }

    /// Fast mode is what the prompt hook uses: keyword and vector legs plus one graph
    /// fact query, inside the 2.5s per-prompt budget.
    #[tokio::test]
    async fn fast_recall_uses_the_local_legs_and_the_sqlite_graph() {
        let dir = std::env::temp_dir().join(format!(
            "engram-hybrid-fast-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let store = dir.join("projects/-test/memory");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(
            store.join("reference_qdrant.md"),
            "---\nname: qdrant-notes\ndescription: the vector store\nmetadata:\n  type: reference\n---\nQdrant holds the embedding collection for recall.\n",
        )
        .unwrap();
        let config_path = dir.join("engram.yaml");
        std::fs::write(
            &config_path,
            "graph: {backend: sqlite}\nvector_store: {enabled: false}\n",
        )
        .unwrap();

        let started = std::time::Instant::now();
        let output = recall_fast(
            &config_path,
            &legacy_tenant(),
            "-test",
            "qdrant embedding collection",
            4,
        )
        .await
        .expect("fast recall must not fail when only the graph is unavailable");
        let elapsed = started.elapsed();

        assert!(
            elapsed < Duration::from_millis(2500),
            "fast recall took {elapsed:?}, over the per-prompt budget"
        );
        assert_eq!(
            output.results.first().map(|hit| hit.file.as_str()),
            Some("reference_qdrant.md"),
            "the local BM25 leg must run in the fast path"
        );
        assert_eq!(output.legs["keyword"], "ok");
        assert!(
            output.legs["graph"].starts_with("fast"),
            "expected the single-query fact leg, got {:?}",
            output.legs["graph"]
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A config still on `native` or `graphiti_compat` must not break recall: the graph
    /// leg says how to migrate and the keyword leg still answers.
    #[tokio::test]
    async fn a_legacy_graph_backend_degrades_with_the_migration_message() {
        let dir = std::env::temp_dir().join(format!(
            "engram-hybrid-legacy-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let store = dir.join("projects/-test/memory");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(
            store.join("reference_qdrant.md"),
            "---\nname: qdrant-notes\ndescription: the vector store\n---\nQdrant holds the embedding collection.\n",
        )
        .unwrap();
        let config_path = dir.join("engram.yaml");
        std::fs::write(
            &config_path,
            "graph: {backend: native}\nvector_store: {enabled: false}\n",
        )
        .unwrap();
        let output = recall_fast(
            &config_path,
            &legacy_tenant(),
            "-test",
            "qdrant embedding collection",
            4,
        )
        .await
        .expect("a legacy graph backend must not fail recall");
        assert!(
            output.legs["graph"].contains("engram-graph-to-sqlite"),
            "the leg must say how to migrate, got {:?}",
            output.legs["graph"]
        );
        assert_eq!(output.legs["keyword"], "ok");
        assert_eq!(
            output.results.first().map(|hit| hit.file.as_str()),
            Some("reference_qdrant.md")
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A slow embedding endpoint must not cost the prompt its local results.
    ///
    /// The hook wraps the whole recall in `recall.inject.timeout_ms`, so a
    /// queued embedding request used to consume the entire budget and inject
    /// NOTHING — discarding a BM25 result already computed in ~66ms from
    /// markdown on disk. Found on a live host: a tenanted install indexes one
    /// store per tenant, so the embedding endpoint sees far more load than when
    /// the daemon owned a single store, and every prompt timed out at exactly
    /// 2.515s with zero results while a direct embedding call took 0.6s.
    #[tokio::test]
    async fn a_stalled_vector_leg_does_not_starve_the_local_legs() {
        let dir = std::env::temp_dir().join(format!(
            "engram-hybrid-legbudget-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let store = dir.join("projects/-test/memory");
        std::fs::create_dir_all(&store).unwrap();
        std::fs::write(
            store.join("reference_chateau.md"),
            "---\nname: chateau-host-memory\ndescription: the host ledger\nmetadata:\n  type: reference\n---\n\
             chateau free output is unreliable; the real 251 GiB ledger lives here.\n",
        )
        .unwrap();
        // A vector endpoint that accepts the connection and never answers — the
        // shape of a saturated embedder, not of a refused one.
        let dead = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = dead.local_addr().unwrap().port();
        tokio::spawn(async move {
            // Accept and hold, so the client waits rather than erroring.
            let mut held = Vec::new();
            while let Ok((socket, _)) = dead.accept().await {
                held.push(socket);
            }
        });
        let config_path = dir.join("engram.yaml");
        std::fs::write(
            &config_path,
            format!(
                "graph: {{backend: sqlite}}\n\
                 vector_store: {{enabled: true, url: 'http://127.0.0.1:{port}'}}\n\
                 embed: {{provider: llama_cpp, url: 'http://127.0.0.1:{port}/v1', model: m, dim: 1024}}\n\
                 recall: {{inject: {{timeout_ms: 1000}}}}\n"
            ),
        )
        .unwrap();

        let tenant = legacy_tenant();
        let started = std::time::Instant::now();
        let output = recall_fast(&config_path, &tenant, "-test", "chateau host ledger", 4)
            .await
            .expect("fast recall must survive a stalled vector leg");
        let elapsed = started.elapsed();

        // The local result survives...
        assert_eq!(
            output.results.first().map(|hit| hit.file.as_str()),
            Some("reference_chateau.md"),
            "the BM25 result was discarded because the vector leg was slow: {:?}",
            output.legs
        );
        // ...the vector leg reports why, rather than looking healthy...
        assert!(
            output.legs["vector"].contains("timed out"),
            "expected an explicit vector timeout, got {:?}",
            output.legs["vector"]
        );
        // ...and the leg's own deadline is a FRACTION of the prompt budget, so
        // there is time left to do the local work and return.
        assert!(
            elapsed < Duration::from_millis(1000),
            "the vector leg consumed the whole prompt budget: {elapsed:?}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every guard on the one path by which an agent's output reaches disk.
    ///
    /// Containment, subtree confinement, extension, and redaction BEFORE the
    /// bytes land — the last one matters most, because a vault is synced: a
    /// credential written there has left the box whatever the index holds.
    #[test]
    fn agent_writes_are_guarded_four_ways() {
        let dir = std::env::temp_dir().join(format!(
            "engram-hybrid-write-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        let vault = dir.join("vault");
        std::fs::create_dir_all(vault.join("Runbooks")).unwrap();
        std::fs::create_dir_all(vault.join("_agent")).unwrap();
        std::fs::write(vault.join("Runbooks/DNS.md"), "# curated\n").unwrap();
        let config: Config = serde_yaml::from_str(&format!(
            "tenants:\n  t:\n    slugs: ['-t']\n    vault: {}\n",
            vault.display()
        ))
        .unwrap();
        config.validate().unwrap();
        let tenant = Tenant::resolve(&config, Some("t")).unwrap();

        // 1. the happy path, including a directory that does not exist yet
        let report = wiki_write(
            &tenant,
            "_agent/findings/dns.md",
            "# Finding\n\nResolver drops queries under load.\n",
            WriteMode::Create,
        )
        .unwrap();
        assert!(report.contains("_agent/findings/dns.md"), "{report}");
        let written = std::fs::read_to_string(vault.join("_agent/findings/dns.md")).unwrap();
        assert!(written.contains("Resolver drops queries"));

        // 2. create refuses to clobber; append adds; replace overwrites
        assert!(
            wiki_write(&tenant, "_agent/findings/dns.md", "x", WriteMode::Create).is_err(),
            "create must not silently discard an earlier note"
        );
        wiki_write(
            &tenant,
            "_agent/findings/dns.md",
            "More detail.\n",
            WriteMode::Append,
        )
        .unwrap();
        let appended = std::fs::read_to_string(vault.join("_agent/findings/dns.md")).unwrap();
        assert!(appended.contains("Resolver drops") && appended.contains("More detail"));
        wiki_write(
            &tenant,
            "_agent/findings/dns.md",
            "# Fresh\n",
            WriteMode::Replace,
        )
        .unwrap();
        assert!(
            !std::fs::read_to_string(vault.join("_agent/findings/dns.md"))
                .unwrap()
                .contains("Resolver drops")
        );

        // 3. the curated vault, traversal, and non-markdown are all refused
        for (path, why) in [
            ("Runbooks/DNS.md", "curated page"),
            ("_agent/../Runbooks/DNS.md", "traversal out of the subtree"),
            ("../outside.md", "traversal out of the vault"),
            ("/etc/cron.d/x.md", "absolute path"),
            ("_agent/payload.sh", "non-markdown"),
        ] {
            assert!(
                wiki_write(&tenant, path, "x", WriteMode::Replace).is_err(),
                "{why} was allowed: {path}"
            );
        }
        // the curated page is untouched by any of that
        assert_eq!(
            std::fs::read_to_string(vault.join("Runbooks/DNS.md")).unwrap(),
            "# curated\n"
        );

        // 4. a credential is redacted ON DISK, not merely in the index
        let secret = "ghp_abcdefghijklmnopqrstuvwxyz0123456789";
        let report = wiki_write(
            &tenant,
            "_agent/creds.md",
            &format!("token={secret}\n"),
            WriteMode::Create,
        )
        .unwrap();
        let on_disk = std::fs::read_to_string(vault.join("_agent/creds.md")).unwrap();
        assert!(
            !on_disk.contains(secret),
            "the credential reached the vault: {on_disk}"
        );
        assert!(on_disk.contains(engram_secrets::REDACTION), "{on_disk}");
        assert!(
            report.contains("redacted"),
            "the caller must be told: {report}"
        );

        // and a runaway write is capped
        assert!(
            wiki_write(
                &tenant,
                "_agent/big.md",
                &"x".repeat(MAX_WRITE_BYTES + 1),
                WriteMode::Replace
            )
            .is_err()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `wiki_fetch` takes a path from the MODEL, so containment is enforced
    /// there and not only at the MCP boundary.
    ///
    /// A path argument is the one input an agent fully controls, and the vault
    /// sits next to other tenants' vaults and the config itself. Traversal is
    /// refused on shape before anything is opened, and a symlink out of the
    /// vault is refused after resolution.
    #[test]
    fn wiki_fetch_refuses_to_leave_the_vault() {
        let dir = std::env::temp_dir().join(format!(
            "engram-hybrid-fetch-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        let vault = dir.join("vault");
        std::fs::create_dir_all(vault.join("Runbooks")).unwrap();
        std::fs::write(vault.join("Runbooks/DNS.md"), "# DNS\n\nResolver notes.\n").unwrap();
        std::fs::write(
            dir.join("secret.md"),
            "# other tenant\n\nNOT-FOR-THIS-AGENT\n",
        )
        .unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(dir.join("secret.md"), vault.join("leak.md")).unwrap();

        let config: Config = serde_yaml::from_str(&format!(
            "tenants:\n  t:\n    slugs: ['-t']\n    vault: {}\n",
            vault.display()
        ))
        .unwrap();
        config.validate().unwrap();
        let tenant = Tenant::resolve(&config, Some("t")).unwrap();

        // its own page reads
        let page = wiki_fetch(&tenant, "Runbooks/DNS.md", None, 8000).unwrap();
        assert!(page.contains("Resolver notes"));

        for bad in [
            "../secret.md",
            "/etc/passwd",
            "Runbooks/../../secret.md",
            "leak.md",
        ] {
            let error =
                wiki_fetch(&tenant, bad, None, 8000).expect_err(&format!("{bad} was not refused"));
            assert!(
                !error.contains("NOT-FOR-THIS-AGENT"),
                "the refusal leaked the content: {error}"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The keyword leg caches each file's chunks, so the property that matters is
    /// that the cache follows the vault: an edited file's NEW text is served (the
    /// `(mtime, size)` key invalidated the stale entry) and a deleted file stops
    /// appearing (the entry was evicted). A steady-state vault reusing cached
    /// chunks is what the other wiki tests already exercise; this one drives the
    /// invalidate and evict branches that only a changing vault reaches.
    ///
    /// The edit changes the byte length, so the `size` half of the key fires
    /// deterministically — no dependence on the filesystem's coarse mtime clock,
    /// which is exactly the same-second-same-size edit the cache is documented not
    /// to catch and which no test should hinge on.
    #[tokio::test]
    async fn keyword_leg_cache_refreshes_on_edit_and_delete() {
        let dir = std::env::temp_dir().join(format!(
            "engram-hybrid-wikicache-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        let vault = dir.join("vault");
        std::fs::create_dir_all(vault.join("Notes")).unwrap();
        let page = vault.join("Notes/A.md");
        std::fs::write(&page, "# Alpha\n\nThe BEFORE marker lives here.\n").unwrap();

        // Written literally (not re-serialized) so the test needs no Serialize on
        // Config; vector disabled makes this a pure keyword-leg search.
        let yaml = format!(
            "vector_store:\n  enabled: false\ntenants:\n  t:\n    slugs: ['-t']\n    vault: {}\n",
            vault.display()
        );
        let config: Config = serde_yaml::from_str(&yaml).unwrap();
        config.validate().unwrap();
        let tenant = Tenant::resolve(&config, Some("t")).unwrap();
        let config_path = dir.join("engram.yaml");
        std::fs::write(&config_path, &yaml).unwrap();

        // Finds the original content.
        let hit = wiki_search(&config_path, &tenant, "before marker", 5)
            .await
            .unwrap();
        assert!(
            hit.results.iter().any(|r| r.text.contains("BEFORE")),
            "first search did not find the original page"
        );

        // Rewrite with new, longer content: the differing byte length changes the
        // cache key, so the stale entry is dropped and the page re-parsed.
        std::fs::write(
            &page,
            "# Alpha\n\nThe AFTER marker lives here now, extended.\n",
        )
        .unwrap();
        let hit = wiki_search(&config_path, &tenant, "after marker extended", 5)
            .await
            .unwrap();
        assert!(
            hit.results.iter().any(|r| r.text.contains("AFTER")),
            "edited content was not served — cache did not invalidate"
        );
        assert!(
            !hit.results.iter().any(|r| r.text.contains("BEFORE")),
            "stale content still served after edit"
        );

        // Delete it: the evict branch must drop it from the cache.
        std::fs::remove_file(&page).unwrap();
        let hit = wiki_search(&config_path, &tenant, "after marker extended", 5)
            .await
            .unwrap();
        assert!(
            hit.results.is_empty(),
            "deleted page still returned from cache: {:?}",
            hit.results
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// The whole point of the feature, end to end: one identity's recall must not
    /// surface another's memory, even when the two stores hold a file of the same
    /// name saying different things.
    ///
    /// Colliding filenames on purpose. Node and point identity used to be the
    /// bare filename, so `Runbook.md` in two stores merged into one record — and
    /// the markdown store is read straight off DISK, where no database filter
    /// would have helped. This asserts the refusal happens before any read, and
    /// that the other tenant's text appears nowhere in the response.
    #[tokio::test]
    async fn one_tenants_recall_cannot_reach_another_tenants_store() {
        let dir = std::env::temp_dir().join(format!(
            "engram-hybrid-isolation-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&dir).ok();
        for (slug, secret) in [("-work", "WORK-ONLY-PAYROLL"), ("-home", "HOME-ONLY-NAS")] {
            let store = dir.join(format!("projects/{slug}/memory"));
            std::fs::create_dir_all(&store).unwrap();
            // the same filename in both stores, saying different things
            std::fs::write(
                store.join("Runbook.md"),
                format!(
                    "---\nname: runbook\ndescription: the runbook\nmetadata:\n  type: reference\n---\n\
                     Restore procedure for {secret} systems.\n"
                ),
            )
            .unwrap();
        }
        let config_path = dir.join("engram.yaml");
        std::fs::write(
            &config_path,
            "vector_store: {enabled: false}\ngraph: {backend: native}\n\
             tenants:\n  work:\n    slugs: ['-work']\n  homelab:\n    slugs: ['-home']\n",
        )
        .unwrap();
        let config = Config::load(&config_path).unwrap();
        let work = Tenant::resolve(&config, Some("work")).unwrap();

        // Its own store: found, with its own content.
        let mine = recall(&config_path, &work, "-work", "restore procedure", 4)
            .await
            .expect("a tenant must be able to read its own store");
        assert_eq!(
            mine.results.first().map(|hit| hit.file.as_str()),
            Some("Runbook.md")
        );

        // The other tenant's store: refused outright, not filtered afterwards.
        let error = recall(&config_path, &work, "-home", "restore procedure", 4)
            .await
            .expect_err("reading another identity's store must fail");
        assert!(
            error.contains("does not own"),
            "the refusal must name the cause: {error}"
        );

        // And the other tenant's text is nowhere in the successful response —
        // the check that would catch a leak through any leg, not just the store.
        let rendered = serde_json::to_string(&mine).unwrap();
        assert!(
            !rendered.contains("HOME-ONLY-NAS"),
            "the other identity's content reached the response: {rendered}"
        );
        assert!(
            rendered.contains("Runbook.md"),
            "guard the guard — the response must be non-empty: {rendered}"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}

// ---------------------------------------------------------------------------
// Wiki recall.
//
// A SEPARATE entry point from `recall`, deliberately. Memories are atomic facts
// and wiki chunks are fragments of long documents; fusing them into one ranking
// would let a 40-chunk page outvote every memory in the store, and the prompt
// hook injects from `recall` on every single prompt. Wiki is retrieved when
// asked for — which is also why `wiki_fetch` exists: search finds the section,
// fetch returns the surrounding document within a budget.
// ---------------------------------------------------------------------------

/// One wiki search result.
#[derive(Debug, Serialize)]
pub struct WikiHit {
    pub path: String,
    pub title: String,
    /// `Page > Section > Subsection`.
    pub breadcrumb: String,
    pub chunk_index: usize,
    pub text: String,
    pub sources: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct WikiOutput {
    pub query: String,
    pub results: Vec<WikiHit>,
    pub legs: HashMap<String, String>,
}

/// A chunk read off disk, for the local keyword leg.
#[derive(Clone)]
struct LocalChunk {
    path: String,
    title: String,
    breadcrumb: String,
    index: usize,
    text: String,
}

/// One file's parsed+chunked form, held between queries.
///
/// `mtime`/`size` are the freshness key: a `stat` rather than a content hash,
/// precisely because reading the content is the cost being cached away. The one
/// edit this cannot see is a same-second, same-byte-length rewrite — and the
/// keyword leg is best-effort and local-first, so a rare staleness there just
/// means the SHA-accurate vector leg (the authoritative wiki index) carries the
/// hit instead.
struct CachedFile {
    mtime: i64,
    size: u64,
    chunks: Vec<LocalChunk>,
}

/// Process-global, per-vault cache of parsed chunks, so `local_chunks` re-reads
/// and re-chunks only files that changed since the last query rather than the
/// whole vault every time. Keyed by canonical vault root in the outer map so two
/// vaults (or two tenants) can never share entries — correctness holds even
/// though in practice one `engram-mcp` process serves exactly one tenant for its
/// whole life. Only that long-lived server calls `wiki_search`, so this is where
/// the caching pays off; a one-shot caller would simply build it once and exit.
fn wiki_cache() -> &'static Mutex<HashMap<PathBuf, HashMap<String, CachedFile>>> {
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, HashMap<String, CachedFile>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

impl engram_retrieval::Indexable for LocalChunk {
    fn id(&self) -> String {
        format!("{}#{}", self.path, self.index)
    }
    fn text(&self) -> String {
        // The breadcrumb is searchable too: "DNS > Failover" should match a
        // query for "dns failover" even when neither word is in the prose.
        format!("{} {} {}", self.title, self.breadcrumb, self.text)
    }
}

/// Search one tenant's vault: keyword over the files, vectors over the chunks,
/// fused with RRF.
///
/// The keyword leg needs no index: it walks the vault and chunks it in process,
/// so it works the moment a file is saved and keeps working when Qdrant or the
/// embedder is down — the same local-first posture the memory keyword leg has.
/// It does not re-chunk the whole vault per query, though: [`local_chunks`]
/// caches each file's chunks under its `(mtime, size)` and re-parses only what
/// changed, so a large vault costs a walk plus the parse of its edits, not a
/// full re-read every time.
pub async fn wiki_search(
    config_path: &Path,
    tenant: &Tenant,
    query: &str,
    k: usize,
) -> Result<WikiOutput, String> {
    let config = Config::load(config_path).map_err(|error| error.to_string())?;
    // Say so plainly rather than returning an empty result with a puzzling leg
    // status: an unconfigured vault is a setup step, not a search that found
    // nothing, and the two look identical to a caller.
    if tenant.vault().is_none() {
        return Err(format!(
            "tenant '{}' has no vault configured; add `vault: /path/to/vault` under it \
             in engram.yaml",
            tenant.label()
        ));
    }
    let mut legs = HashMap::new();

    let local = local_chunks(tenant, &mut legs);
    let keyword: Vec<String> = engram_retrieval::rank(&local, query, k * 3)
        .into_iter()
        .map(|hit| hit.file)
        .collect();
    if !local.is_empty() {
        legs.insert("keyword".into(), "ok".into());
    }

    let mut by_id: HashMap<String, WikiHit> = local
        .iter()
        .map(|chunk| {
            (
                format!("{}#{}", chunk.path, chunk.index),
                WikiHit {
                    path: chunk.path.clone(),
                    title: chunk.title.clone(),
                    breadcrumb: chunk.breadcrumb.clone(),
                    chunk_index: chunk.index,
                    text: chunk.text.clone(),
                    sources: Vec::new(),
                },
            )
        })
        .collect();

    let vector = wiki_vector_leg(&config, tenant, query, k * 3, &mut by_id, &mut legs).await;

    let ranked = rrf(
        &[keyword.clone(), vector.clone()],
        k,
        config.recall.hybrid.k_rrf,
    );
    let results = ranked
        .into_iter()
        .filter_map(|hit| {
            let mut found = by_id.remove(&hit.file)?;
            if keyword.contains(&hit.file) {
                found.sources.push("keyword".into());
            }
            if vector.contains(&hit.file) {
                found.sources.push("vector".into());
            }
            Some(found)
        })
        .collect();
    Ok(WikiOutput {
        query: query.into(),
        results,
        legs,
    })
}

/// Walk the vault and return its chunks for the keyword leg, re-reading and
/// re-chunking only the files that changed since the last query.
///
/// The walk still runs every call — it is cheap (readdir + stat) and it is how
/// new, deleted and renamed files are noticed. What it no longer does is
/// `read_to_string` + parse + chunk every page on every query: each file's
/// chunks are cached under its `(mtime, size)` in [`wiki_cache`], so a
/// steady-state vault produces its chunks from memory and only a genuinely
/// edited page pays the parse cost. A deleted file is evicted at the end of the
/// walk so its chunks stop being served.
fn local_chunks(tenant: &Tenant, legs: &mut HashMap<String, String>) -> Vec<LocalChunk> {
    let root = match tenant.vault_root() {
        Ok(root) => root,
        Err(error) => {
            legs.insert("keyword".into(), error.to_string());
            return Vec::new();
        }
    };
    let paths = match engram_wiki::walk(&root) {
        Ok(paths) => paths,
        Err(error) => {
            legs.insert("keyword".into(), error.to_string());
            return Vec::new();
        }
    };
    let params = engram_wiki::ChunkParams::default();

    // One short critical section for the whole walk. `local_chunks` is sync (no
    // `.await` anywhere below) and the MCP server drives it single-threaded, so
    // holding the std `Mutex` across the fs reads neither parks the async
    // executor nor contends — and it keeps "only one re-chunk of a given vault
    // at a time" true if a second caller ever appears.
    let mut guard = wiki_cache()
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let files = guard.entry(root.clone()).or_default();

    let present: HashSet<&String> = paths.iter().collect();
    files.retain(|path, _| present.contains(path));

    let mut chunks = Vec::new();
    for path in &paths {
        let absolute = engram_wiki::absolute(&root, path);
        let Ok(metadata) = std::fs::metadata(&absolute) else {
            continue; // vanished between walk and stat — skip, as a read failure would
        };
        let size = metadata.len();
        let mtime = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|age| age.as_secs() as i64)
            .unwrap_or_default();

        if let Some(cached) = files
            .get(path)
            .filter(|cached| cached.mtime == mtime && cached.size == size)
        {
            chunks.extend(cached.chunks.iter().cloned());
            continue;
        }

        let Ok(raw) = std::fs::read_to_string(&absolute) else {
            continue;
        };
        let document = engram_wiki::parse(path, &raw, mtime);
        let file_chunks: Vec<LocalChunk> = engram_wiki::chunk(&document, params)
            .into_iter()
            .map(|chunk| LocalChunk {
                path: path.clone(),
                title: document.title.clone(),
                breadcrumb: chunk.breadcrumb(),
                index: chunk.index,
                text: chunk.text,
            })
            .collect();
        chunks.extend(file_chunks.iter().cloned());
        files.insert(
            path.clone(),
            CachedFile {
                mtime,
                size,
                chunks: file_chunks,
            },
        );
    }
    chunks
}

async fn wiki_vector_leg(
    config: &Config,
    tenant: &Tenant,
    query: &str,
    k: usize,
    by_id: &mut HashMap<String, WikiHit>,
    legs: &mut HashMap<String, String>,
) -> Vec<String> {
    if !config.vector_store.enabled {
        legs.insert("vector".into(), "disabled".into());
        return Vec::new();
    }
    let space = config.embedding_space_id();
    let result = async {
        let embeddings = OpenAiCompatibleClient::new(config.embed_endpoint())?
            .with_api_key(config.embed.api_key.present())
            .with_timeout(config.embed_timeout_seconds());
        let vector = embeddings
            .embedding(&config.embed.model, &config.query_text(query))
            .await?;
        let hits = QdrantClient::for_tenant(config, tenant, Corpus::Wiki)
            .search_chunks(vector, k, Scope::space(&space))
            .await?;
        Ok::<_, Box<dyn std::error::Error>>(hits)
    }
    .await;
    match result {
        Ok(hits) => {
            legs.insert("vector".into(), "ok".into());
            hits.into_iter()
                .map(|hit| {
                    let id = format!("{}#{}", hit.path, hit.chunk_index);
                    // A chunk may be in Qdrant and not on disk — the vault is
                    // unmounted, or indexing ran before a delete. The stored
                    // text means the hit is still answerable.
                    by_id.entry(id.clone()).or_insert_with(|| WikiHit {
                        path: hit.path,
                        title: hit.title,
                        breadcrumb: hit.breadcrumb,
                        chunk_index: hit.chunk_index,
                        text: hit.text,
                        sources: Vec::new(),
                    });
                    id
                })
                .collect()
        }
        Err(error) => {
            legs.insert("vector".into(), error.to_string());
            Vec::new()
        }
    }
}

/// A whole document, or one of its sections, within a character budget.
///
/// This is the half that makes the feature useful: search locates the section,
/// this returns the context around it. Returning a truncated fragment is what
/// the memory path already does badly for long documents, so the budget cuts at
/// a section boundary where it can and says what it dropped.
pub fn wiki_fetch(
    tenant: &Tenant,
    path: &str,
    heading: Option<&str>,
    max_chars: usize,
) -> Result<String, String> {
    // Through the tenant, so `..` and a symlink out of the vault are refused
    // before anything is opened.
    let resolved = tenant
        .resolve_in_vault(Path::new(path))
        .map_err(|error| error.to_string())?;
    let raw = std::fs::read_to_string(&resolved).map_err(|error| format!("{path}: {error}"))?;
    let document = engram_wiki::parse(path, &raw, 0);
    let params = engram_wiki::ChunkParams::default();
    let chunks = engram_wiki::chunk(&document, params);

    let wanted: Vec<&engram_wiki::Chunk> = match heading {
        None => chunks.iter().collect(),
        Some(heading) => {
            let matching: Vec<&engram_wiki::Chunk> = chunks
                .iter()
                .filter(|chunk| {
                    chunk
                        .heading_path
                        .iter()
                        .any(|part| part.eq_ignore_ascii_case(heading.trim()))
                })
                .collect();
            if matching.is_empty() {
                return Err(format!(
                    "no section matching {heading:?} in {path}; sections are: {}",
                    chunks
                        .iter()
                        .map(|chunk| chunk.breadcrumb())
                        .collect::<Vec<_>>()
                        .join(" | ")
                ));
            }
            matching
        }
    };

    // Redact on the READ path too. The vault is hand-authored/imported and never
    // passed a save-time guard, and this text is handed straight to the model over
    // MCP — the same boundary the write and index paths redact at. (`path` is a
    // vault-relative path, not content, so it is left as-is.)
    let (safe_title, _) = engram_secrets::redact(&document.title);
    let mut out = format!("# {safe_title}\n_{}_\n\n", path);
    let mut dropped = 0;
    for chunk in wanted {
        let (safe_crumb, _) = engram_secrets::redact(&chunk.breadcrumb());
        let (safe_text, _) = engram_secrets::redact(&chunk.text);
        let piece = format!("## {safe_crumb}\n\n{safe_text}\n\n");
        if out.len() + piece.len() > max_chars && !out.is_empty() {
            dropped += 1;
            continue;
        }
        out.push_str(&piece);
    }
    if dropped > 0 {
        out.push_str(&format!(
            "_[{dropped} further section(s) omitted to stay within {max_chars} characters — \
             fetch a specific heading to read them]_\n"
        ));
    }
    Ok(out)
}

/// How an agent write behaves when the file already exists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum WriteMode {
    /// Refuse if the path exists. The default, so an agent cannot silently
    /// discard a note it or another session wrote earlier.
    Create,
    /// Append, creating the file if absent. The natural mode for a running log.
    Append,
    /// Overwrite. Explicit, because it is the only mode that destroys content.
    Replace,
}

impl WriteMode {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "create" => Ok(Self::Create),
            "append" => Ok(Self::Append),
            "replace" => Ok(Self::Replace),
            other => Err(format!(
                "unknown mode {other:?}; use create, append or replace"
            )),
        }
    }
}

/// Largest single write. A guard against a looping agent filling the disk, not
/// a judgement about document length — a page longer than this should be
/// several pages anyway.
const MAX_WRITE_BYTES: usize = 1 << 20;

/// Write a note into the agent-writable subtree of a tenant's vault.
///
/// Four independent guards, because this is the only path by which an agent's
/// output reaches the operator's disk:
///
/// 1. **Containment** — the path must resolve inside the vault, symlinks and
///    `..` included, so one identity cannot write into another's vault.
/// 2. **Subtree** — and inside `agent_subtree` specifically, so a curated page
///    cannot be touched. Separate from (1) on purpose: a future change to one
///    must not silently widen the other.
/// 3. **Extension** — `.md` only. Obsidian renders nothing else, and without
///    this an agent could drop a script or a binary into a directory a human
///    browses and a sync client propagates.
/// 4. **Redaction** — through the shared detector, BEFORE the bytes hit disk.
///    Not merely before indexing: a credential written to a synced vault has
///    left the box regardless of what the index holds.
///
/// The write itself is temp file + fsync + rename, the discipline the config
/// saver already uses, so a crash leaves either the old file or the new one and
/// never a half-written page for Obsidian to render.
pub fn wiki_write(
    tenant: &Tenant,
    path: &str,
    content: &str,
    mode: WriteMode,
) -> Result<String, String> {
    use std::io::Write as _;

    if content.len() > MAX_WRITE_BYTES {
        return Err(format!(
            "content is {} bytes; the limit is {MAX_WRITE_BYTES}",
            content.len()
        ));
    }
    let relative = path.trim();
    if !relative.to_ascii_lowercase().ends_with(".md") {
        return Err(format!(
            "{relative:?} must end in .md — the vault holds markdown, and a sync \
             client propagates whatever is in it"
        ));
    }
    let target = tenant
        .resolve_for_agent_write(Path::new(relative))
        .map_err(|error| error.to_string())?;

    let (safe, redacted) = engram_secrets::redact(content);
    if target.exists() && mode == WriteMode::Create {
        return Err(format!(
            "{relative} already exists; use mode=append to add to it or \
             mode=replace to overwrite it"
        ));
    }
    let parent = target
        .parent()
        .ok_or_else(|| format!("{relative} has no parent directory"))?;
    std::fs::create_dir_all(parent).map_err(|error| format!("{relative}: {error}"))?;

    if mode == WriteMode::Append && target.exists() {
        // Append must NOT read-modify-rename the whole file: two concurrent appends
        // would each read the old content and the last rename would silently discard
        // the other's addition. Open with O_APPEND, under which each write() syscall
        // seeks to EOF atomically, so no append is ever lost. Separator and body are
        // concatenated into ONE write_all so they are not split by a concurrent
        // append; a note large enough for the kernel to split one write_all into
        // several syscalls could still interleave with another concurrent large
        // append (no data loss, only ordering) — fine for hand-sized notes. The
        // separator is decided from a cheap last-byte read — a stale read only adds or
        // omits one blank line, it can never lose data the way the whole-file replace did.
        let ends_with_newline = {
            use std::io::{Read, Seek, SeekFrom};
            std::fs::File::open(&target)
                .and_then(|mut f| {
                    if f.seek(SeekFrom::End(0))? == 0 {
                        return Ok(true); // empty file: no separator wanted
                    }
                    f.seek(SeekFrom::End(-1))?;
                    let mut last = [0u8; 1];
                    f.read_exact(&mut last)?;
                    Ok(last[0] == b'\n')
                })
                .unwrap_or(true)
        };
        let separator: &str = if ends_with_newline { "" } else { "\n" };
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&target)
            .map_err(|error| format!("{relative}: {error}"))?;
        let payload = format!("{separator}{safe}");
        file.write_all(payload.as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("{relative}: {error}"))?;
        if let Ok(dir) = std::fs::File::open(parent) {
            dir.sync_all().ok();
        }
    } else {
        // Create (refused above if the path exists), Replace, or appending to a file
        // that does not exist yet: write the whole body to a uniquely named temp file
        // in the destination directory and atomically rename it into place. The temp
        // name carries a per-process monotonic counter as well as the pid, so two
        // concurrent writes to the same target in one process cannot collide on it.
        static WRITE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let temporary = parent.join(format!(
            ".engram-write-{}-{}-{}.tmp",
            std::process::id(),
            WRITE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            target
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default()
        ));
        let mut file =
            std::fs::File::create(&temporary).map_err(|error| format!("{relative}: {error}"))?;
        let result = file
            .write_all(safe.as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("{relative}: {error}"));
        if let Err(error) = result {
            std::fs::remove_file(&temporary).ok();
            return Err(error);
        }
        drop(file);
        std::fs::rename(&temporary, &target).map_err(|error| {
            std::fs::remove_file(&temporary).ok();
            format!("{relative}: {error}")
        })?;
        // fsync the DIRECTORY too, or the rename itself can be lost on power loss
        // while the file contents survive — the same reason the config saver does it.
        if let Ok(dir) = std::fs::File::open(parent) {
            dir.sync_all().ok();
        }
    }

    let note = match redacted {
        0 => String::new(),
        1 => " (1 credential redacted)".into(),
        many => format!(" ({many} credentials redacted)"),
    };
    Ok(format!(
        "wrote {} bytes to {relative}{note}; it will be indexed on the next wiki pass",
        safe.len()
    ))
}
