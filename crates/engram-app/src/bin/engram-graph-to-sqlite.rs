//! Copy a store's native graph from Neo4j into the SQLite file (`graph.backend: sqlite`).
//!
//! Read-only on Neo4j. The scope's rows in the SQLite file are REPLACED, so the
//! command is safe to re-run. It reads facts and triples through their memory, so the
//! retired orphans Neo4j keeps are not carried over, and it skips the entity and
//! mention nodes that no query ever read.
//!
//!   engram-graph-to-sqlite [--config PATH] [--slug S] [--tenant T] [--db PATH] [--dry-run]

use clap::Parser;
use engram_graph::GraphClient;
use std::{path::PathBuf, process::ExitCode};

#[derive(Parser)]
struct Args {
    #[arg(long, env = "ENGRAM_CONFIG")]
    config: Option<PathBuf>,
    #[arg(long, allow_hyphen_values = true)]
    slug: Option<String>,
    #[arg(long, env = "ENGRAM_TENANT")]
    tenant: Option<String>,
    /// Destination file (default: <engram home>/index/graph.db).
    #[arg(long)]
    db: Option<PathBuf>,
    /// Read and count, write nothing.
    #[arg(long)]
    dry_run: bool,
    /// After copying, run the same reads against Neo4j and SQLite and compare them.
    #[arg(long)]
    verify: bool,
}

const PAGE: usize = 500;

#[tokio::main]
async fn main() -> ExitCode {
    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("engram-graph-to-sqlite: {error}");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<(), String> {
    let config_path = engram_paths::config_path(args.config);
    let engram_tenant::Resolved {
        config,
        tenant,
        slug,
    } = engram_tenant::resolve_for_cli(
        &config_path,
        args.tenant.as_deref(),
        args.slug.as_deref(),
        engram_tenant::Derive::Environment,
    )?;
    let scope = tenant
        .graph_scope(&slug)
        .map_err(|error| error.to_string())?;
    let creds = config.graph.credentials();
    if creds.password.is_empty() {
        return Err(
            "no Neo4j password: set NEO4J_PASSWORD, graph.neo4j_password, or graph/.env".into(),
        );
    }
    let neo4j = GraphClient::from_credentials(&creds).map_err(|error| error.to_string())?;

    let memories = neo4j
        .export_native_memories(&scope)
        .await
        .map_err(|error| error.to_string())?;
    let (expected_facts, expected_embedded) = neo4j
        .count_native_facts(&scope)
        .await
        .map_err(|error| error.to_string())?;
    let mut facts = Vec::new();
    let mut skip = 0;
    loop {
        let page = neo4j
            .export_native_facts(&scope, skip, PAGE)
            .await
            .map_err(|error| error.to_string())?;
        let got = page.len();
        facts.extend(page);
        if got < PAGE {
            break;
        }
        skip += PAGE;
    }
    let triples = neo4j
        .export_native_triples(&scope)
        .await
        .map_err(|error| error.to_string())?;
    println!(
        "neo4j  {}/{}: {} memories, {} facts ({} embedded), {} triples",
        scope.tenant(),
        scope.slug(),
        memories.len(),
        facts.len(),
        facts.iter().filter(|fact| fact.embedding.is_some()).count(),
        triples.len()
    );
    if i64::try_from(facts.len()).unwrap_or(-1) != expected_facts {
        return Err(format!(
            "read {} facts but Neo4j counts {expected_facts}: refusing to write a partial copy",
            facts.len()
        ));
    }
    if args.dry_run {
        println!("dry run: nothing written");
        return Ok(());
    }
    let path = args
        .db
        .unwrap_or_else(|| engram_paths::graph_db_path(&config_path));
    let sqlite =
        engram_graph::sqlite::SqliteGraph::open(&path).map_err(|error| error.to_string())?;
    let counts = sqlite
        .import(&scope, &memories, &facts, &triples)
        .map_err(|error| error.to_string())?;
    println!(
        "sqlite {}: {} memories, {} facts ({} embedded), {} triples",
        path.display(),
        counts.memories,
        counts.facts,
        counts.facts_embedded,
        counts.triples
    );
    if counts.facts != expected_facts || counts.facts_embedded != expected_embedded {
        return Err(format!(
            "copy does not match Neo4j: facts {}/{expected_facts}, embedded {}/{expected_embedded}",
            counts.facts, counts.facts_embedded
        ));
    }
    if args.verify {
        verify(&neo4j, &path, &scope, &facts).await?;
    }
    Ok(())
}

/// Prompts that exercise the reads: names, compound ids, accents, filler words.
const PROMPTS: &[&str] = &[
    "Como está o timer do engram e o backend de llm?",
    "preciso revisar o deploy do ibirapay no dedi",
    "qual a regra de merge dos PRs do cipher staging",
    "strata diskcache fila e releitura de contexto",
    "wallet android f-droid release 1.2.2",
    "o que sabemos sobre peerswap no gateway",
    "gateway dns ipv6 resolver unbound",
    "moises fatura energia mineracao hash",
    "robosats coordinator lightning canal",
    "vaultwarden senha rotacao amantikir",
    "pdv relay dashboard quadlet systemd",
    "google ads campanha jaleco luiza",
];

fn tokens(query: &str) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .filter(|word| word.len() >= 4 && seen.insert(word.to_lowercase()))
        .map(str::to_string)
        .collect()
}

async fn verify(
    neo4j: &GraphClient,
    path: &std::path::Path,
    scope: &engram_tenant::GraphScope,
    facts: &[engram_graph::sqlite::ImportFact],
) -> Result<(), String> {
    let local = GraphClient::sqlite(path).map_err(|error| error.to_string())?;
    let mut problems = Vec::new();
    for prompt in PROMPTS {
        let names = tokens(prompt);
        let (a, b) = (
            neo4j.native_facts_for_tokens(scope, &names, 6).await,
            local.native_facts_for_tokens(scope, &names, 6).await,
        );
        let (a, b) = (a.map_err(|e| e.to_string())?, b.map_err(|e| e.to_string())?);
        let mut left = a.clone();
        let mut right = b.clone();
        left.sort();
        right.sort();
        println!(
            "facts   {:>2}/{:>2} same-set={} | {prompt}",
            a.len(),
            b.len(),
            left == right
        );
        if left != right {
            problems.push(format!(
                "facts differ for {prompt:?}:\n  neo4j  {a:?}\n  sqlite {b:?}"
            ));
        }
        let (a, b) = (
            neo4j.native_keyword_files(scope, prompt, 12).await,
            local.native_keyword_files(scope, prompt, 12).await,
        );
        let (a, b) = (a.map_err(|e| e.to_string())?, b.map_err(|e| e.to_string())?);
        // Neo4j orders ties arbitrarily; compare (file, score) as sets.
        let key = |hits: &[engram_graph::RecallHit]| {
            let mut v: Vec<(String, u64)> = hits
                .iter()
                .map(|h| (h.file.clone(), h.score as u64))
                .collect();
            v.sort();
            v
        };
        let top_scores = |hits: &[engram_graph::RecallHit]| {
            hits.iter().map(|h| h.score as u64).collect::<Vec<_>>()
        };
        println!(
            "keyword {:>2}/{:>2} same-scores={} | {prompt}",
            a.len(),
            b.len(),
            top_scores(&a) == top_scores(&b)
        );
        if top_scores(&a) != top_scores(&b) {
            problems.push(format!("keyword scores differ for {prompt:?}"));
        } else if key(&a) != key(&b)
            && a.iter().map(|h| h.score as u64).min() != b.iter().map(|h| h.score as u64).min()
        {
            problems.push(format!("keyword files differ for {prompt:?}"));
        }
    }
    // Semantic: use stored fact embeddings as query vectors.
    let queries: Vec<&Vec<f32>> = facts
        .iter()
        .filter_map(|fact| fact.embedding.as_ref())
        .step_by((facts.len() / 8).max(1))
        .take(8)
        .collect();
    for vector in queries {
        let (a, b) = (
            neo4j.native_semantic_files(scope, vector, 8).await,
            local.native_semantic_files(scope, vector, 8).await,
        );
        let (a, b) = (a.map_err(|e| e.to_string())?, b.map_err(|e| e.to_string())?);
        let same_files = a.iter().map(|h| &h.file).collect::<Vec<_>>()
            == b.iter().map(|h| &h.file).collect::<Vec<_>>();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x.score - y.score).abs())
            .fold(0.0f64, f64::max);
        println!(
            "semantic {:>2}/{:>2} same-order={same_files} max-score-diff={worst:.2e}",
            a.len(),
            b.len()
        );
        if !same_files || worst > 1e-3 {
            problems.push(format!(
                "semantic differs (same_order={same_files}, diff={worst:.2e})"
            ));
        }
    }
    if problems.is_empty() {
        println!("verify: neo4j and sqlite agree on every read");
        Ok(())
    } else {
        Err(format!("verify failed:\n{}", problems.join("\n")))
    }
}
