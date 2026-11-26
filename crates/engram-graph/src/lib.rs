//! The native graph: facts and triples extracted from the memory store, scoped by
//! tenant and slug, in a local SQLite file (see [`sqlite`]).
//!
//! `graph.backend` used to select between Graphiti (Python, Neo4j) and a Rust writer
//! that put the same data into Neo4j. Both are gone; [`neo4j`] remains only as the
//! read-only source of the one-time copy into SQLite.

pub mod neo4j;
pub mod sqlite;

use engram_tenant::GraphScope;
use serde::Serialize;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct RecallHit {
    pub file: String,
    pub facts: Vec<String>,
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NativeTriple {
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f64,
    pub temporal: String,
}

pub const RELATION_TAXONOMY: &[&str] = &[
    "belongs_to",
    "conflicts_with",
    "connects_to",
    "depends_on",
    "hosts",
    "implements",
    "owns",
    "provides",
    "runs_on",
    "supersedes",
    "uses",
];

pub fn is_valid_relation(relation: &str) -> bool {
    RELATION_TAXONOMY.contains(&relation)
}

/// Triple lifecycle states. Only `Active` is eligible for recall.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TripleStatus {
    /// Believed true now.
    Active,
    /// Below the confidence floor — stored, never recalled.
    Quarantined,
    /// Explicitly a past state. Recall must not present it as current.
    Historical,
}

impl TripleStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Quarantined => "quarantined",
            Self::Historical => "historical",
        }
    }

    /// Whether this state carries a `valid_until`, i.e. is closed off in time.
    pub fn is_closed(self) -> bool {
        !matches!(self, Self::Active)
    }
}

/// Confidence below this is kept but never recalled.
pub const CONFIDENCE_FLOOR: f64 = 0.7;

/// Decide a triple's lifecycle state.
///
/// `temporal: "formerly"` used to be stored as `active` with no `valid_until`,
/// and no recall predicate ever read `temporal` — so a memory saying a service
/// *used to* run somewhere was returned as though it still did. A past state is
/// now closed off at write time, which is the only place that information exists.
pub fn triple_status(triple: &NativeTriple) -> TripleStatus {
    if triple.confidence < CONFIDENCE_FLOOR {
        return TripleStatus::Quarantined;
    }
    match triple.temporal.trim().to_lowercase().as_str() {
        "formerly" | "historical" | "past" => TripleStatus::Historical,
        _ => TripleStatus::Active,
    }
}

/// A stable identity for a triple within one memory.
pub fn triple_key(triple: &NativeTriple) -> String {
    format!(
        "{}|{}|{}",
        triple.subject.to_lowercase(),
        triple.relation,
        triple.object.to_lowercase()
    )
}

/// Which earlier claims a batch of triples retires.
///
/// For `(A, supersedes, B)`, the claim being replaced is the one about **B** — the
/// superseded object. The previous query instead looked for another triple with the
/// same `subject` and the relation `supersedes`, which is a different thing
/// entirely: it never matched the `uses`/`runs_on` claim actually being replaced,
/// so the old claim stayed active alongside the new one.
pub fn supersession_targets(triples: &[NativeTriple]) -> Vec<String> {
    let mut targets: Vec<String> = triples
        .iter()
        .filter(|triple| {
            triple.relation == "supersedes" && triple_status(triple) == TripleStatus::Active
        })
        .map(|triple| triple.object.trim().to_lowercase())
        .filter(|object| !object.is_empty())
        .collect();
    targets.sort();
    targets.dedup();
    targets
}

#[derive(Debug, Error)]
pub enum GraphError {
    #[error("invalid Neo4j URI: {0}")]
    Uri(String),
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    #[error("graph database error: {0}")]
    Database(String),
}

/// The graph store the recall legs and the sync talk to. Cheap to clone.
#[derive(Clone)]
pub struct GraphClient {
    db: sqlite::SqliteGraph,
}

impl GraphClient {
    pub fn open(path: &std::path::Path) -> Result<Self, GraphError> {
        Ok(Self {
            db: sqlite::SqliteGraph::open(path)?,
        })
    }

    /// The graph file next to the config (`<engram home>/index/graph.db`).
    pub fn connect(config_path: &std::path::Path) -> Result<Self, GraphError> {
        Self::open(&engram_paths::graph_db_path(config_path))
    }

    pub fn sqlite_graph(&self) -> &sqlite::SqliteGraph {
        &self.db
    }

    pub async fn native_facts_for_tokens(
        &self,
        scope: &GraphScope,
        tokens: &[String],
        limit: usize,
    ) -> Result<Vec<String>, GraphError> {
        self.db.facts_for_tokens(scope, tokens, limit)
    }

    /// Keyword recall over a memory's own text, its facts and its active triples.
    /// Substring matching on tokens of four characters or more.
    pub async fn native_keyword_files(
        &self,
        scope: &GraphScope,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RecallHit>, GraphError> {
        let tokens = query
            .split(|ch: char| !ch.is_alphanumeric())
            .filter(|word| word.len() >= 4)
            .map(|word| word.to_lowercase())
            .collect::<Vec<_>>();
        self.db.keyword_files(scope, &tokens, limit)
    }

    pub async fn native_semantic_files(
        &self,
        scope: &GraphScope,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<RecallHit>, GraphError> {
        self.db.semantic_files(scope, vector, limit)
    }

    pub async fn upsert_native_memory(
        &self,
        scope: &GraphScope,
        file: &str,
        name: &str,
        description: &str,
        body: &str,
        source_mtime: i64,
    ) -> Result<(), GraphError> {
        self.db
            .upsert_memory(scope, file, name, description, body, source_mtime)
    }

    /// Bring the supersession ordering key up to date for the whole store. Call once,
    /// before syncing any memory.
    pub async fn set_native_memory_mtimes(
        &self,
        scope: &GraphScope,
        memories: &[(String, i64)],
    ) -> Result<(), GraphError> {
        self.db.set_memory_mtimes(scope, memories)
    }

    pub async fn native_memory_is_current(
        &self,
        scope: &GraphScope,
        file: &str,
        sha: &str,
        space: &str,
    ) -> Result<bool, GraphError> {
        self.db.memory_is_current(scope, file, sha, space)
    }

    pub async fn replace_native_facts(
        &self,
        scope: &GraphScope,
        file: &str,
        facts: &[String],
    ) -> Result<(), GraphError> {
        self.db.replace_facts(scope, file, facts)
    }

    pub async fn replace_native_triples(
        &self,
        scope: &GraphScope,
        file: &str,
        triples: &[NativeTriple],
    ) -> Result<(), GraphError> {
        self.db.replace_triples(scope, file, triples)
    }

    pub async fn set_native_fact_embeddings(
        &self,
        scope: &GraphScope,
        file: &str,
        points: &[(&str, Vec<f32>)],
    ) -> Result<(), GraphError> {
        self.db.set_fact_embeddings(scope, file, points)
    }

    /// Record that `file` is fully synced. Must be the LAST write of a sync.
    pub async fn mark_native_memory_current(
        &self,
        scope: &GraphScope,
        file: &str,
        sha: &str,
        space: &str,
    ) -> Result<(), GraphError> {
        self.db.mark_memory_current(scope, file, sha, space)
    }

    /// Every memory file the graph holds for a slug.
    pub async fn native_memory_files(&self, scope: &GraphScope) -> Result<Vec<String>, GraphError> {
        self.db.memory_files(scope)
    }

    /// Remove graph data for memories that are no longer in the store.
    pub async fn prune_missing_memories(
        &self,
        scope: &GraphScope,
        files: &[String],
    ) -> Result<(), GraphError> {
        self.db.prune_missing(scope, files)
    }

    /// Files that already hold a commit marker or facts.
    pub async fn native_files(&self, scope: &GraphScope) -> Result<Vec<String>, GraphError> {
        self.db.native_files(scope)
    }
}
