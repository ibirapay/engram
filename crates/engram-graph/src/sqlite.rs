//! The native graph in SQLite.
//!
//! Same data model and same query semantics as the Neo4j-backed native path in
//! [`crate::GraphClient`] (`EngramMemory` / `EngramFact` / `EngramTriple`, scoped by
//! `tenant` + `slug`), minus two things that nothing ever read:
//!
//! * `EngramEntity` nodes and their `MENTIONS` / `SUBJECT` / `OBJECT` edges. On the
//!   reference install that was 38,897 nodes and ~121,000 relationships written on
//!   every sync and traversed by no query.
//! * The legacy Graphiti surface (`Entity`, `Episodic`, `RELATES_TO`) and the tenancy
//!   migration, which exist only to read the old graph.
//!
//! The hot read, [`SqliteGraph::facts_for_tokens`], is what the per-prompt hook
//! runs. On Neo4j it is one Cypher statement with a regex per triple per token
//! (~500 ms on the reference install); here the substring prefilter is an `instr`
//! over a pre-lowercased column and only the survivors get the word-boundary check.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::{Arc, Mutex};

use engram_tenant::GraphScope;
use rusqlite::{Connection, OptionalExtension, params};

use crate::{GraphError, NativeTriple, RecallHit, is_valid_relation, supersession_targets};
use crate::{triple_key, triple_status};

fn db(error: impl std::fmt::Display) -> GraphError {
    GraphError::Database(format!("sqlite: {error}"))
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .and_then(|age| i64::try_from(age.as_secs()).ok())
        .unwrap_or_default()
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY,
    tenant TEXT NOT NULL, slug TEXT NOT NULL, file TEXT NOT NULL,
    name TEXT, description TEXT, body TEXT,
    search_l TEXT NOT NULL DEFAULT '',
    source_mtime INTEGER,
    sha TEXT, embedding_space TEXT, triple_version INTEGER, synced_at INTEGER,
    updated_at INTEGER,
    UNIQUE (tenant, slug, file)
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tenant TEXT NOT NULL, slug TEXT NOT NULL, memory_file TEXT NOT NULL,
    text TEXT NOT NULL, legacy_name TEXT,
    subject TEXT, object TEXT, subject_l TEXT, object_l TEXT,
    fact_l TEXT NOT NULL DEFAULT '',
    embedding BLOB,
    valid_until INTEGER,
    created_at INTEGER, updated_at INTEGER,
    UNIQUE (tenant, slug, memory_file, text)
);
CREATE INDEX IF NOT EXISTS facts_memory ON facts (memory_id);
CREATE INDEX IF NOT EXISTS facts_subject ON facts (tenant, slug, subject_l);
CREATE INDEX IF NOT EXISTS facts_object ON facts (tenant, slug, object_l);
CREATE TABLE IF NOT EXISTS triples (
    id INTEGER PRIMARY KEY,
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    tenant TEXT NOT NULL, slug TEXT NOT NULL, memory_file TEXT NOT NULL,
    key TEXT NOT NULL,
    subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
    subject_l TEXT NOT NULL, object_l TEXT NOT NULL, triple_l TEXT NOT NULL,
    confidence REAL, temporal TEXT,
    status TEXT NOT NULL,
    valid_until INTEGER,
    created_at INTEGER, updated_at INTEGER,
    UNIQUE (tenant, slug, memory_file, key)
);
CREATE INDEX IF NOT EXISTS triples_memory ON triples (memory_id);
CREATE INDEX IF NOT EXISTS triples_object ON triples (tenant, slug, object_l);
";

/// The graph database. Cheap to clone; every clone shares one connection.
#[derive(Clone)]
pub struct SqliteGraph {
    conn: Arc<Mutex<Connection>>,
}

/// A memory as the exporter reads it out of Neo4j and hands to [`SqliteGraph::import`].
#[derive(Clone, Debug, Default)]
pub struct ImportMemory {
    pub file: String,
    pub name: Option<String>,
    pub description: Option<String>,
    pub body: Option<String>,
    pub source_mtime: Option<i64>,
    pub sha: Option<String>,
    pub embedding_space: Option<String>,
    pub triple_version: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct ImportFact {
    pub memory_file: String,
    pub text: String,
    pub legacy_name: Option<String>,
    pub subject: Option<String>,
    pub object: Option<String>,
    pub embedding: Option<Vec<f32>>,
    pub valid_until: Option<i64>,
}

#[derive(Clone, Debug, Default)]
pub struct ImportTriple {
    pub memory_file: String,
    pub key: String,
    pub subject: String,
    pub relation: String,
    pub object: String,
    pub confidence: f64,
    pub temporal: String,
    pub status: String,
    pub valid_until: Option<i64>,
}

/// Row counts of one store, for the exporter's verification.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub memories: i64,
    pub facts: i64,
    pub facts_embedded: i64,
    pub triples: i64,
}

fn embedding_bytes(vector: &[f32]) -> Vec<u8> {
    vector
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn embedding_from(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Neo4j's `vector.similarity.cosine`: the cosine mapped from [-1, 1] to [0, 1].
/// Kept so `score > 0` and anything reading the score mean what they always meant.
fn similarity(left: &[f32], right: &[f32]) -> Option<f64> {
    if left.len() != right.len() || left.is_empty() {
        return None;
    }
    let (mut dot, mut norm_l, mut norm_r) = (0.0f64, 0.0f64, 0.0f64);
    for (a, b) in left.iter().zip(right) {
        let (a, b) = (f64::from(*a), f64::from(*b));
        dot += a * b;
        norm_l += a * a;
        norm_r += b * b;
    }
    if norm_l == 0.0 || norm_r == 0.0 {
        return None;
    }
    Some((1.0 + dot / (norm_l.sqrt() * norm_r.sqrt())) / 2.0)
}

/// `needle` occurs in `haystack` delimited by non-alphanumerics (or the ends): the
/// whole-word rule the Cypher expressed as a regex. Both arguments are lowercase.
fn has_word(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return false;
    }
    haystack.match_indices(needle).any(|(start, _)| {
        let before = haystack[..start].chars().next_back();
        let after = haystack[start + needle.len()..].chars().next();
        !before.is_some_and(char::is_alphanumeric) && !after.is_some_and(char::is_alphanumeric)
    })
}

impl SqliteGraph {
    pub fn open(path: &Path) -> Result<Self, GraphError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(db)?;
        }
        Self::init(Connection::open(path).map_err(db)?)
    }

    pub fn open_in_memory() -> Result<Self, GraphError> {
        Self::init(Connection::open_in_memory().map_err(db)?)
    }

    fn init(conn: Connection) -> Result<Self, GraphError> {
        conn.busy_timeout(std::time::Duration::from_millis(5000))
            .map_err(db)?;
        conn.query_row("PRAGMA journal_mode = WAL", [], |_| Ok(()))
            .optional()
            .map_err(db)?;
        conn.execute_batch("PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;")
            .map_err(db)?;
        conn.execute_batch(SCHEMA).map_err(db)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    fn with<T>(
        &self,
        run: impl FnOnce(&mut Connection) -> Result<T, rusqlite::Error>,
    ) -> Result<T, GraphError> {
        let mut conn = self
            .conn
            .lock()
            .map_err(|_| GraphError::Database("sqlite: connection lock poisoned".into()))?;
        run(&mut conn).map_err(db)
    }

    pub fn counts(&self) -> Result<Counts, GraphError> {
        self.with(|conn| {
            conn.query_row(
                "SELECT (SELECT COUNT(*) FROM memories), (SELECT COUNT(*) FROM facts),
                        (SELECT COUNT(*) FROM facts WHERE embedding IS NOT NULL),
                        (SELECT COUNT(*) FROM triples)",
                [],
                |row| {
                    Ok(Counts {
                        memories: row.get(0)?,
                        facts: row.get(1)?,
                        facts_embedded: row.get(2)?,
                        triples: row.get(3)?,
                    })
                },
            )
        })
    }

    // ---- reads -----------------------------------------------------------------

    /// Facts about the query's tokens, ranked. Mirrors `NATIVE_FACTS_FOR_TOKENS`:
    /// whole-word matches only, an exact subject/object match (2.0) outranks a
    /// partial one (1.0), at most 3 facts per token per branch picked by score and
    /// text, and facts supported by more tokens first.
    pub fn facts_for_tokens(
        &self,
        scope: &GraphScope,
        tokens: &[String],
        limit: usize,
    ) -> Result<Vec<String>, GraphError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let (tenant, slug) = (scope.tenant(), scope.slug());
        self.with(|conn| {
            let mut triples = conn.prepare_cached(
                "SELECT subject, relation, object, subject_l, object_l FROM triples
                 WHERE tenant = ?1 AND slug = ?2 AND status = 'active' AND valid_until IS NULL
                   AND (instr(subject_l, ?3) > 0 OR instr(object_l, ?3) > 0)",
            )?;
            let mut facts = conn.prepare_cached(
                "SELECT text FROM facts
                 WHERE tenant = ?1 AND slug = ?2 AND valid_until IS NULL
                   AND (subject_l = ?3 OR object_l = ?3)
                 ORDER BY text LIMIT 3",
            )?;
            // (fact, score as bits) rows across all tokens, one set per token: the
            // UNION inside the CALL dedupes a row within a token, not across them.
            let mut rows: Vec<(String, u64)> = Vec::new();
            for token in tokens {
                let name = token.to_lowercase();
                let mut best: BTreeMap<String, f64> = BTreeMap::new();
                let found = triples.query_map(params![tenant, slug, name], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                    ))
                })?;
                for entry in found {
                    let (subject, relation, object, subject_l, object_l) = entry?;
                    if !(has_word(&subject_l, &name) || has_word(&object_l, &name)) {
                        continue;
                    }
                    let score = if subject_l == name || object_l == name {
                        2.0
                    } else {
                        1.0
                    };
                    let slot = best
                        .entry(format!("{subject} {relation} {object}"))
                        .or_insert(0.0);
                    if score > *slot {
                        *slot = score;
                    }
                }
                let mut ranked: Vec<(String, f64)> = best.into_iter().collect();
                ranked.sort_by(|left, right| {
                    right
                        .1
                        .total_cmp(&left.1)
                        .then_with(|| left.0.cmp(&right.0))
                });
                ranked.truncate(3);
                let mut token_rows: HashSet<(String, u64)> = ranked
                    .into_iter()
                    .map(|(fact, score)| (fact, score.to_bits()))
                    .collect();
                let from_facts =
                    facts.query_map(params![tenant, slug, name], |row| row.get::<_, String>(0))?;
                for fact in from_facts {
                    token_rows.insert((fact?, 2.0f64.to_bits()));
                }
                rows.extend(token_rows);
            }
            let mut by_fact: HashMap<String, (f64, usize)> = HashMap::new();
            for (fact, bits) in rows {
                let entry = by_fact.entry(fact).or_insert((0.0, 0));
                entry.0 = entry.0.max(f64::from_bits(bits));
                entry.1 += 1;
            }
            let mut ordered: Vec<(String, f64, usize)> = by_fact
                .into_iter()
                .map(|(fact, (score, support))| (fact, score, support))
                .collect();
            ordered.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| right.2.cmp(&left.2))
                    .then_with(|| left.0.cmp(&right.0))
            });
            ordered.truncate(limit);
            Ok(ordered.into_iter().map(|(fact, _, _)| fact).collect())
        })
    }

    /// Mirrors `NATIVE_KEYWORD_FILES`: a memory scores one point per token found in
    /// its own text, plus one per matching live fact, plus one per matching active
    /// triple. `tokens` are already lowercase.
    pub fn keyword_files(
        &self,
        scope: &GraphScope,
        tokens: &[String],
        limit: usize,
    ) -> Result<Vec<RecallHit>, GraphError> {
        if tokens.is_empty() {
            return Ok(Vec::new());
        }
        let (tenant, slug) = (scope.tenant(), scope.slug());
        self.with(|conn| {
            #[derive(Default)]
            struct Acc {
                score: usize,
                facts: Vec<String>,
            }
            let mut files: BTreeMap<String, Acc> = BTreeMap::new();
            {
                let mut stmt = conn.prepare_cached(
                    "SELECT file FROM memories
                     WHERE tenant = ?1 AND slug = ?2 AND instr(search_l, ?3) > 0",
                )?;
                for token in tokens {
                    for file in
                        stmt.query_map(params![tenant, slug, token], |row| row.get::<_, String>(0))?
                    {
                        files.entry(file?).or_default().score += 1;
                    }
                }
            }
            let any = |column: &str| {
                (0..tokens.len())
                    .map(|index| format!("instr({column}, ?{}) > 0", index + 3))
                    .collect::<Vec<_>>()
                    .join(" OR ")
            };
            let mut bind: Vec<&dyn rusqlite::ToSql> = vec![&tenant, &slug];
            bind.extend(tokens.iter().map(|token| token as &dyn rusqlite::ToSql));
            {
                let sql = format!(
                    "SELECT memory_file, text FROM facts
                     WHERE tenant = ?1 AND slug = ?2 AND valid_until IS NULL AND ({})",
                    any("fact_l")
                );
                let mut stmt = conn.prepare(&sql)?;
                for row in stmt.query_map(bind.as_slice(), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })? {
                    let (file, text) = row?;
                    let acc = files.entry(file).or_default();
                    acc.score += 1;
                    acc.facts.push(text);
                }
            }
            {
                let sql = format!(
                    "SELECT memory_file, subject || ' ' || relation || ' ' || object FROM triples
                     WHERE tenant = ?1 AND slug = ?2 AND status = 'active'
                       AND valid_until IS NULL AND ({})",
                    any("triple_l")
                );
                let mut stmt = conn.prepare(&sql)?;
                for row in stmt.query_map(bind.as_slice(), |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })? {
                    let (file, text) = row?;
                    let acc = files.entry(file).or_default();
                    acc.score += 1;
                    if !acc.facts.contains(&text) {
                        acc.facts.push(text);
                    }
                }
            }
            let mut hits: Vec<RecallHit> = files
                .into_iter()
                .filter(|(_, acc)| acc.score > 0)
                .map(|(file, acc)| RecallHit {
                    file,
                    facts: acc.facts,
                    score: acc.score as f64,
                })
                .collect();
            hits.sort_by(|left, right| {
                right
                    .score
                    .total_cmp(&left.score)
                    .then_with(|| left.file.cmp(&right.file))
            });
            hits.truncate(limit);
            Ok(hits)
        })
    }

    /// Mirrors `NATIVE_SEMANTIC_FILES`: best fact similarity per memory.
    pub fn semantic_files(
        &self,
        scope: &GraphScope,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<RecallHit>, GraphError> {
        let (tenant, slug) = (scope.tenant(), scope.slug());
        self.with(|conn| {
            let mut stmt = conn.prepare_cached(
                "SELECT memory_file, text, embedding FROM facts
                 WHERE tenant = ?1 AND slug = ?2 AND valid_until IS NULL
                   AND embedding IS NOT NULL",
            )?;
            let mut best: BTreeMap<String, (f64, Vec<String>)> = BTreeMap::new();
            for row in stmt.query_map(params![tenant, slug], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Vec<u8>>(2)?,
                ))
            })? {
                let (file, text, blob) = row?;
                let Some(score) = similarity(&embedding_from(&blob), vector) else {
                    continue;
                };
                if score <= 0.0 {
                    continue;
                }
                let entry = best.entry(file).or_insert((0.0, Vec::new()));
                entry.0 = entry.0.max(score);
                if !entry.1.contains(&text) {
                    entry.1.push(text);
                }
            }
            let mut hits: Vec<RecallHit> = best
                .into_iter()
                .map(|(file, (score, facts))| RecallHit { file, facts, score })
                .collect();
            hits.sort_by(|left, right| {
                right
                    .score
                    .total_cmp(&left.score)
                    .then_with(|| left.file.cmp(&right.file))
            });
            hits.truncate(limit);
            Ok(hits)
        })
    }

    pub fn memory_files(&self, scope: &GraphScope) -> Result<Vec<String>, GraphError> {
        self.with(|conn| {
            let mut stmt =
                conn.prepare_cached("SELECT file FROM memories WHERE tenant = ?1 AND slug = ?2")?;
            stmt.query_map(params![scope.tenant(), scope.slug()], |row| row.get(0))?
                .collect()
        })
    }

    /// Files that already hold a commit marker or facts (see `NATIVE_FILES`).
    pub fn native_files(&self, scope: &GraphScope) -> Result<Vec<String>, GraphError> {
        self.with(|conn| {
            let mut stmt = conn.prepare_cached(
                "SELECT m.file FROM memories m WHERE m.tenant = ?1 AND m.slug = ?2
                   AND (m.sha IS NOT NULL
                        OR EXISTS (SELECT 1 FROM facts f WHERE f.memory_id = m.id))",
            )?;
            stmt.query_map(params![scope.tenant(), scope.slug()], |row| row.get(0))?
                .collect()
        })
    }

    pub fn memory_is_current(
        &self,
        scope: &GraphScope,
        file: &str,
        sha: &str,
        space: &str,
    ) -> Result<bool, GraphError> {
        self.with(|conn| {
            conn.query_row(
                "SELECT COUNT(*) > 0 FROM memories
                 WHERE tenant = ?1 AND slug = ?2 AND file = ?3 AND sha = ?4
                   AND embedding_space = ?5 AND triple_version = 2",
                params![scope.tenant(), scope.slug(), file, sha, space],
                |row| row.get(0),
            )
        })
    }

    // ---- writes ----------------------------------------------------------------

    /// Opens a sync generation and clears the commit marker (see `UPSERT_NATIVE_MEMORY`).
    pub fn upsert_memory(
        &self,
        scope: &GraphScope,
        file: &str,
        name: &str,
        description: &str,
        body: &str,
        source_mtime: i64,
    ) -> Result<(), GraphError> {
        let search = format!("{name} {description} {body}").to_lowercase();
        self.with(|conn| {
            conn.execute(
                "INSERT INTO memories
                    (tenant, slug, file, name, description, body, search_l, source_mtime, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                 ON CONFLICT (tenant, slug, file) DO UPDATE SET
                    name = excluded.name, description = excluded.description,
                    body = excluded.body, search_l = excluded.search_l,
                    source_mtime = excluded.source_mtime, updated_at = excluded.updated_at,
                    sha = NULL, embedding_space = NULL,
                    triple_version = NULL, synced_at = NULL",
                params![
                    scope.tenant(),
                    scope.slug(),
                    file,
                    name,
                    description,
                    body,
                    search,
                    source_mtime,
                    now()
                ],
            )
            .map(|_| ())
        })
    }

    pub fn set_memory_mtimes(
        &self,
        scope: &GraphScope,
        memories: &[(String, i64)],
    ) -> Result<(), GraphError> {
        self.with(|conn| {
            let tx = conn.transaction()?;
            {
                let mut stmt = tx.prepare_cached(
                    "UPDATE memories SET source_mtime = ?4
                     WHERE tenant = ?1 AND slug = ?2 AND file = ?3
                       AND COALESCE(source_mtime, 0) <> ?4",
                )?;
                for (file, mtime) in memories {
                    stmt.execute(params![scope.tenant(), scope.slug(), file, mtime])?;
                }
            }
            tx.commit()
        })
    }

    /// Replace the memory's live facts with `facts`: the ones it no longer states
    /// are removed, the rest are (re)opened. A memory with no row is a no-op.
    pub fn replace_facts(
        &self,
        scope: &GraphScope,
        file: &str,
        facts: &[String],
    ) -> Result<(), GraphError> {
        self.with(|conn| {
            let tx = conn.transaction()?;
            let Some(memory_id) = tx
                .query_row(
                    "SELECT id FROM memories WHERE tenant = ?1 AND slug = ?2 AND file = ?3",
                    params![scope.tenant(), scope.slug(), file],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
            else {
                return Ok(());
            };
            let wanted: HashSet<&str> = facts.iter().map(String::as_str).collect();
            let existing: Vec<(i64, String)> = {
                let mut stmt = tx.prepare("SELECT id, text FROM facts WHERE memory_id = ?1")?;
                stmt.query_map(params![memory_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<_, _>>()?
            };
            for (id, text) in existing {
                if !wanted.contains(text.as_str()) {
                    tx.execute("DELETE FROM facts WHERE id = ?1", params![id])?;
                }
            }
            {
                let mut insert = tx.prepare_cached(
                    "INSERT INTO facts
                        (memory_id, tenant, slug, memory_file, text, fact_l, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)
                     ON CONFLICT (tenant, slug, memory_file, text) DO UPDATE SET
                        valid_until = NULL, updated_at = excluded.updated_at",
                )?;
                for fact in facts {
                    insert.execute(params![
                        memory_id,
                        scope.tenant(),
                        scope.slug(),
                        file,
                        fact,
                        format!("{fact} ").to_lowercase(),
                        now()
                    ])?;
                }
            }
            tx.commit()
        })
    }

    /// Retire the triples the memory no longer asserts, write the ones it does, then
    /// apply supersession (see `APPLY_SUPERSESSION` for why the order is by source
    /// mtime with the file name breaking ties).
    pub fn replace_triples(
        &self,
        scope: &GraphScope,
        file: &str,
        triples: &[NativeTriple],
    ) -> Result<(), GraphError> {
        let valid: Vec<&NativeTriple> = triples
            .iter()
            .filter(|triple| is_valid_relation(&triple.relation))
            .collect();
        let targets = supersession_targets(triples);
        self.with(|conn| {
            let tx = conn.transaction()?;
            let Some(memory_id) = tx
                .query_row(
                    "SELECT id FROM memories WHERE tenant = ?1 AND slug = ?2 AND file = ?3",
                    params![scope.tenant(), scope.slug(), file],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
            else {
                return Ok(());
            };
            let keys: HashSet<String> = valid.iter().map(|triple| triple_key(triple)).collect();
            let existing: Vec<(i64, String)> = {
                let mut stmt = tx.prepare("SELECT id, key FROM triples WHERE memory_id = ?1")?;
                stmt.query_map(params![memory_id], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect::<Result<_, _>>()?
            };
            for (id, key) in existing {
                if !keys.contains(&key) {
                    tx.execute("DELETE FROM triples WHERE id = ?1", params![id])?;
                }
            }
            if valid.is_empty() {
                return tx.commit();
            }
            {
                let mut write = tx.prepare_cached(
                    "INSERT INTO triples
                        (memory_id, tenant, slug, memory_file, key, subject, relation, object,
                         subject_l, object_l, triple_l, confidence, temporal, status,
                         valid_until, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                             CASE WHEN ?15 THEN ?16 ELSE NULL END, ?16, ?16)
                     ON CONFLICT (tenant, slug, memory_file, key) DO UPDATE SET
                        subject = excluded.subject, relation = excluded.relation,
                        object = excluded.object, subject_l = excluded.subject_l,
                        object_l = excluded.object_l, triple_l = excluded.triple_l,
                        confidence = excluded.confidence, temporal = excluded.temporal,
                        status = excluded.status,
                        valid_until = CASE WHEN ?15
                            THEN COALESCE(triples.valid_until, ?16) ELSE NULL END,
                        updated_at = excluded.updated_at",
                )?;
                for triple in &valid {
                    let status = triple_status(triple);
                    let (subject_l, object_l) =
                        (triple.subject.to_lowercase(), triple.object.to_lowercase());
                    write.execute(params![
                        memory_id,
                        scope.tenant(),
                        scope.slug(),
                        file,
                        triple_key(triple),
                        triple.subject,
                        triple.relation,
                        triple.object,
                        subject_l,
                        object_l,
                        format!("{} {} {}", triple.subject, triple.relation, triple.object)
                            .to_lowercase(),
                        triple.confidence,
                        triple.temporal,
                        status.as_str(),
                        status.is_closed(),
                        now()
                    ])?;
                }
            }
            if !targets.is_empty() {
                let mut supersede = tx.prepare_cached(
                    "UPDATE triples SET valid_until = ?5, status = 'superseded'
                     WHERE id IN (
                        SELECT prior.id
                        FROM memories sm
                        JOIN triples sup ON sup.memory_id = sm.id
                        JOIN memories pm ON pm.tenant = sm.tenant AND pm.slug = sm.slug
                        JOIN triples prior ON prior.memory_id = pm.id
                        WHERE sm.tenant = ?1 AND sm.slug = ?2 AND sm.file = ?3
                          AND sup.relation = 'supersedes' AND sup.object_l = ?4
                          AND prior.valid_until IS NULL AND prior.relation <> 'supersedes'
                          AND prior.object_l = sup.object_l
                          AND pm.file <> sm.file
                          AND (COALESCE(pm.source_mtime, 0) < COALESCE(sm.source_mtime, 0)
                               OR (COALESCE(pm.source_mtime, 0) = COALESCE(sm.source_mtime, 0)
                                   AND pm.file < sm.file)))",
                )?;
                for target in &targets {
                    supersede.execute(params![
                        scope.tenant(),
                        scope.slug(),
                        file,
                        target.to_lowercase(),
                        now()
                    ])?;
                }
            }
            tx.commit()
        })
    }

    pub fn set_fact_embeddings(
        &self,
        scope: &GraphScope,
        file: &str,
        points: &[(&str, Vec<f32>)],
    ) -> Result<(), GraphError> {
        self.with(|conn| {
            let tx = conn.transaction()?;
            {
                let mut stmt = tx.prepare_cached(
                    "UPDATE facts SET embedding = ?5, updated_at = ?6
                     WHERE tenant = ?1 AND slug = ?2 AND memory_file = ?3 AND text = ?4",
                )?;
                for (text, vector) in points {
                    stmt.execute(params![
                        scope.tenant(),
                        scope.slug(),
                        file,
                        text,
                        embedding_bytes(vector),
                        now()
                    ])?;
                }
            }
            tx.commit()
        })
    }

    /// The last write of a sync: only now is the memory "current".
    pub fn mark_memory_current(
        &self,
        scope: &GraphScope,
        file: &str,
        sha: &str,
        space: &str,
    ) -> Result<(), GraphError> {
        self.with(|conn| {
            conn.execute(
                "UPDATE memories SET sha = ?4, embedding_space = ?5, triple_version = 2,
                        synced_at = ?6
                 WHERE tenant = ?1 AND slug = ?2 AND file = ?3",
                params![scope.tenant(), scope.slug(), file, sha, space, now()],
            )
            .map(|_| ())
        })
    }

    /// Drop memories (and, by cascade, their facts and triples) not in `files`.
    pub fn prune_missing(&self, scope: &GraphScope, files: &[String]) -> Result<(), GraphError> {
        let keep: HashSet<&str> = files.iter().map(String::as_str).collect();
        self.with(|conn| {
            let tx = conn.transaction()?;
            let stored: Vec<(i64, String)> = {
                let mut stmt =
                    tx.prepare("SELECT id, file FROM memories WHERE tenant = ?1 AND slug = ?2")?;
                stmt.query_map(params![scope.tenant(), scope.slug()], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })?
                .collect::<Result<_, _>>()?
            };
            for (id, file) in stored {
                if !keep.contains(file.as_str()) {
                    tx.execute("DELETE FROM memories WHERE id = ?1", params![id])?;
                }
            }
            tx.commit()
        })
    }

    /// Bulk-load a store exported from Neo4j, replacing whatever this scope held.
    pub fn import(
        &self,
        scope: &GraphScope,
        memories: &[ImportMemory],
        facts: &[ImportFact],
        triples: &[ImportTriple],
    ) -> Result<Counts, GraphError> {
        self.with(|conn| {
            let tx = conn.transaction()?;
            tx.execute(
                "DELETE FROM memories WHERE tenant = ?1 AND slug = ?2",
                params![scope.tenant(), scope.slug()],
            )?;
            let mut ids: HashMap<&str, i64> = HashMap::new();
            {
                let mut insert = tx.prepare_cached(
                    "INSERT INTO memories
                        (tenant, slug, file, name, description, body, search_l, source_mtime,
                         sha, embedding_space, triple_version, synced_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?12)
                     RETURNING id",
                )?;
                for memory in memories {
                    let search = format!(
                        "{} {} {}",
                        memory.name.as_deref().unwrap_or(""),
                        memory.description.as_deref().unwrap_or(""),
                        memory.body.as_deref().unwrap_or("")
                    )
                    .to_lowercase();
                    let id: i64 = insert.query_row(
                        params![
                            scope.tenant(),
                            scope.slug(),
                            memory.file,
                            memory.name,
                            memory.description,
                            memory.body,
                            search,
                            memory.source_mtime,
                            memory.sha,
                            memory.embedding_space,
                            memory.triple_version,
                            now()
                        ],
                        |row| row.get(0),
                    )?;
                    ids.insert(memory.file.as_str(), id);
                }
            }
            {
                let mut insert = tx.prepare_cached(
                    "INSERT OR IGNORE INTO facts
                        (memory_id, tenant, slug, memory_file, text, legacy_name, subject, object,
                         subject_l, object_l, fact_l, embedding, valid_until, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?14)",
                )?;
                for fact in facts {
                    let Some(memory_id) = ids.get(fact.memory_file.as_str()) else {
                        continue;
                    };
                    insert.execute(params![
                        memory_id,
                        scope.tenant(),
                        scope.slug(),
                        fact.memory_file,
                        fact.text,
                        fact.legacy_name,
                        fact.subject,
                        fact.object,
                        fact.subject.as_deref().map(str::to_lowercase),
                        fact.object.as_deref().map(str::to_lowercase),
                        format!(
                            "{} {}",
                            fact.text,
                            fact.legacy_name.as_deref().unwrap_or("")
                        )
                        .to_lowercase(),
                        fact.embedding.as_deref().map(embedding_bytes),
                        fact.valid_until,
                        now()
                    ])?;
                }
            }
            {
                let mut insert = tx.prepare_cached(
                    "INSERT OR IGNORE INTO triples
                        (memory_id, tenant, slug, memory_file, key, subject, relation, object,
                         subject_l, object_l, triple_l, confidence, temporal, status,
                         valid_until, created_at, updated_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15,
                             ?16, ?16)",
                )?;
                for triple in triples {
                    let Some(memory_id) = ids.get(triple.memory_file.as_str()) else {
                        continue;
                    };
                    insert.execute(params![
                        memory_id,
                        scope.tenant(),
                        scope.slug(),
                        triple.memory_file,
                        triple.key,
                        triple.subject,
                        triple.relation,
                        triple.object,
                        triple.subject.to_lowercase(),
                        triple.object.to_lowercase(),
                        format!("{} {} {}", triple.subject, triple.relation, triple.object)
                            .to_lowercase(),
                        triple.confidence,
                        triple.temporal,
                        triple.status,
                        triple.valid_until,
                        now()
                    ])?;
                }
            }
            tx.commit()
        })?;
        self.counts()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn scope() -> GraphScope {
        GraphScope::unchecked("t", "s")
    }

    fn triple(subject: &str, relation: &str, object: &str) -> NativeTriple {
        NativeTriple {
            subject: subject.into(),
            relation: relation.into(),
            object: object.into(),
            confidence: 0.95,
            temporal: "current".into(),
        }
    }

    fn graph_with(memories: &[(&str, i64)]) -> Result<SqliteGraph, GraphError> {
        let graph = SqliteGraph::open_in_memory()?;
        for (file, mtime) in memories {
            graph.upsert_memory(&scope(), file, file, "", "corpo", *mtime)?;
        }
        Ok(graph)
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn a_token_matches_whole_words_only() -> TestResult {
        // "esta" must not find "restarts": bare substring matching was the original bug.
        let graph = graph_with(&[("a.md", 1)])?;
        graph.replace_triples(
            &scope(),
            "a.md",
            &[
                triple("restarts", "uses", "systemd"),
                triple("esta", "uses", "tool"),
            ],
        )?;
        let facts = graph.facts_for_tokens(&scope(), &strings(&["esta"]), 10)?;
        assert_eq!(facts, vec!["esta uses tool"]);
        Ok(())
    }

    #[test]
    fn an_exact_match_outranks_a_partial_one_and_support_breaks_ties() -> TestResult {
        let graph = graph_with(&[("a.md", 1)])?;
        graph.replace_triples(
            &scope(),
            "a.md",
            &[
                triple("pagcoin-gateway", "uses", "tailscale"),
                triple("gateway", "hosts", "lnd"),
                triple("lnd", "runs_on", "pagcoin-gateway"),
            ],
        )?;
        let facts = graph.facts_for_tokens(&scope(), &strings(&["gateway"]), 10)?;
        assert_eq!(facts[0], "gateway hosts lnd");
        assert_eq!(facts.len(), 3);
        // Two tokens that hit the same fact: it is supported twice and ranks first
        // among equals.
        let facts = graph.facts_for_tokens(&scope(), &strings(&["lnd", "gateway"]), 10)?;
        assert_eq!(facts[0], "gateway hosts lnd");
        Ok(())
    }

    #[test]
    fn at_most_three_triple_facts_per_token_and_the_limit_applies() -> TestResult {
        let graph = graph_with(&[("a.md", 1)])?;
        let many: Vec<NativeTriple> = (0..6)
            .map(|n| triple("hub", "uses", &format!("servico{n}")))
            .collect();
        graph.replace_triples(&scope(), "a.md", &many)?;
        assert_eq!(
            graph
                .facts_for_tokens(&scope(), &strings(&["hub"]), 10)?
                .len(),
            3
        );
        assert_eq!(
            graph
                .facts_for_tokens(&scope(), &strings(&["hub"]), 2)?
                .len(),
            2
        );
        assert!(graph.facts_for_tokens(&scope(), &[], 5)?.is_empty());
        Ok(())
    }

    #[test]
    fn closed_and_other_scopes_never_surface() -> TestResult {
        let graph = graph_with(&[("a.md", 1)])?;
        graph.replace_triples(
            &scope(),
            "a.md",
            &[
                NativeTriple {
                    temporal: "formerly".into(),
                    ..triple("antigo", "uses", "coisa")
                },
                NativeTriple {
                    confidence: 0.1,
                    ..triple("duvidoso", "uses", "coisa")
                },
                triple("vivo", "uses", "coisa"),
            ],
        )?;
        assert_eq!(
            graph.facts_for_tokens(&scope(), &strings(&["coisa"]), 10)?,
            vec!["vivo uses coisa"]
        );
        let other = GraphScope::unchecked("t", "outro");
        assert!(
            graph
                .facts_for_tokens(&other, &strings(&["vivo"]), 10)?
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn keyword_files_scores_text_facts_and_triples() -> TestResult {
        let graph = SqliteGraph::open_in_memory()?;
        graph.upsert_memory(
            &scope(),
            "a.md",
            "gateway dns",
            "resolver",
            "unbound local",
            1,
        )?;
        graph.upsert_memory(&scope(), "b.md", "outra", "nota", "nada relevante", 1)?;
        graph.replace_facts(&scope(), "a.md", &strings(&["o resolver usa unbound"]))?;
        graph.replace_triples(&scope(), "a.md", &[triple("unbound", "runs_on", "gateway")])?;
        graph.replace_facts(&scope(), "b.md", &strings(&["fato sem relacao"]))?;
        let hits = graph.keyword_files(&scope(), &strings(&["unbound"]), 5)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file, "a.md");
        // memory text + fact + triple.
        assert_eq!(hits[0].score, 3.0);
        assert!(
            hits[0]
                .facts
                .contains(&"o resolver usa unbound".to_string())
        );
        assert!(
            hits[0]
                .facts
                .contains(&"unbound runs_on gateway".to_string())
        );
        Ok(())
    }

    #[test]
    fn semantic_files_rank_by_best_fact_per_memory() -> TestResult {
        let graph = graph_with(&[("perto.md", 1), ("longe.md", 1)])?;
        graph.replace_facts(
            &scope(),
            "perto.md",
            &strings(&["fato perto", "fato quase"]),
        )?;
        graph.replace_facts(&scope(), "longe.md", &strings(&["fato longe"]))?;
        graph.set_fact_embeddings(
            &scope(),
            "perto.md",
            &[
                ("fato perto", vec![1.0, 0.0]),
                ("fato quase", vec![0.7, 0.7]),
            ],
        )?;
        graph.set_fact_embeddings(&scope(), "longe.md", &[("fato longe", vec![-0.2, 1.0])])?;
        let hits = graph.semantic_files(&scope(), &[1.0, 0.0], 5)?;
        assert_eq!(
            hits.iter().map(|hit| hit.file.as_str()).collect::<Vec<_>>(),
            vec!["perto.md", "longe.md"]
        );
        assert!(
            (hits[0].score - 1.0).abs() < 1e-9,
            "identical vectors map to 1.0"
        );
        assert_eq!(hits[0].facts.len(), 2);
        Ok(())
    }

    #[test]
    fn replace_facts_retires_what_the_memory_no_longer_says() -> TestResult {
        let graph = graph_with(&[("a.md", 1)])?;
        graph.replace_facts(&scope(), "a.md", &strings(&["um", "dois"]))?;
        graph.set_fact_embeddings(&scope(), "a.md", &[("dois", vec![1.0, 0.0])])?;
        graph.replace_facts(&scope(), "a.md", &strings(&["dois", "tres"]))?;
        assert_eq!(graph.counts()?.facts, 2);
        // The survivor keeps its embedding: re-asserting a fact must not cost a re-embed.
        assert_eq!(graph.counts()?.facts_embedded, 1);
        // A memory that does not exist is a no-op, not an error.
        graph.replace_facts(&scope(), "ausente.md", &strings(&["x"]))?;
        assert_eq!(graph.counts()?.facts, 2);
        Ok(())
    }

    #[test]
    fn supersession_follows_source_mtime_then_file_name() -> TestResult {
        // old.md claims `api uses sqlite`; new.md (newer) says `postgres supersedes sqlite`.
        let graph = graph_with(&[("old.md", 10), ("new.md", 20)])?;
        graph.replace_triples(&scope(), "old.md", &[triple("api", "uses", "sqlite")])?;
        graph.replace_triples(
            &scope(),
            "new.md",
            &[triple("postgres", "supersedes", "sqlite")],
        )?;
        assert!(
            graph
                .facts_for_tokens(&scope(), &strings(&["api"]), 5)?
                .is_empty()
        );

        // The reverse order of mtimes must NOT retire the claim.
        let graph = graph_with(&[("old.md", 30), ("new.md", 20)])?;
        graph.replace_triples(&scope(), "old.md", &[triple("api", "uses", "sqlite")])?;
        graph.replace_triples(
            &scope(),
            "new.md",
            &[triple("postgres", "supersedes", "sqlite")],
        )?;
        assert_eq!(
            graph.facts_for_tokens(&scope(), &strings(&["api"]), 5)?,
            vec!["api uses sqlite"]
        );

        // Equal mtimes: the file name decides, in one direction only.
        let graph = graph_with(&[("a.md", 5), ("b.md", 5)])?;
        graph.replace_triples(&scope(), "a.md", &[triple("x", "supersedes", "y")])?;
        graph.replace_triples(&scope(), "b.md", &[triple("z", "supersedes", "y")])?;
        graph.replace_triples(
            &scope(),
            "a.md",
            &[triple("x", "supersedes", "y"), triple("p", "uses", "y")],
        )?;
        graph.replace_triples(
            &scope(),
            "b.md",
            &[triple("z", "supersedes", "y"), triple("q", "uses", "y")],
        )?;
        let live = graph.facts_for_tokens(&scope(), &strings(&["y"]), 10)?;
        assert!(
            live.contains(&"q uses y".to_string()),
            "b.md is later by name, so a.md's claim goes: {live:?}"
        );
        assert!(!live.contains(&"p uses y".to_string()), "{live:?}");
        Ok(())
    }

    #[test]
    fn a_memory_never_retires_its_own_siblings() -> TestResult {
        let graph = graph_with(&[("a.md", 5)])?;
        graph.replace_triples(
            &scope(),
            "a.md",
            &[
                triple("novo", "supersedes", "velho"),
                triple("app", "uses", "velho"),
            ],
        )?;
        assert_eq!(
            graph.facts_for_tokens(&scope(), &strings(&["app"]), 5)?,
            vec!["app uses velho"]
        );
        Ok(())
    }

    #[test]
    fn retired_triples_leave_when_the_memory_stops_asserting_them() -> TestResult {
        let graph = graph_with(&[("a.md", 1)])?;
        graph.replace_triples(
            &scope(),
            "a.md",
            &[triple("a", "uses", "b"), triple("c", "uses", "d")],
        )?;
        graph.replace_triples(&scope(), "a.md", &[triple("c", "uses", "d")])?;
        assert_eq!(graph.counts()?.triples, 1);
        graph.replace_triples(&scope(), "a.md", &[])?;
        assert_eq!(graph.counts()?.triples, 0);
        Ok(())
    }

    #[test]
    fn the_commit_marker_is_only_valid_after_the_last_write() -> TestResult {
        let graph = graph_with(&[("a.md", 1)])?;
        assert!(!graph.memory_is_current(&scope(), "a.md", "sha1", "space")?);
        graph.mark_memory_current(&scope(), "a.md", "sha1", "space")?;
        assert!(graph.memory_is_current(&scope(), "a.md", "sha1", "space")?);
        assert!(!graph.memory_is_current(&scope(), "a.md", "sha2", "space")?);
        assert!(!graph.memory_is_current(&scope(), "a.md", "sha1", "other")?);
        // Opening a new generation clears the marker (an interrupted sync stays retryable).
        graph.upsert_memory(&scope(), "a.md", "a.md", "", "novo corpo", 2)?;
        assert!(!graph.memory_is_current(&scope(), "a.md", "sha1", "space")?);
        Ok(())
    }

    #[test]
    fn pruning_a_memory_takes_its_facts_and_triples_with_it() -> TestResult {
        let graph = graph_with(&[("a.md", 1), ("b.md", 1)])?;
        graph.replace_facts(&scope(), "a.md", &strings(&["fato a"]))?;
        graph.replace_triples(&scope(), "a.md", &[triple("a", "uses", "b")])?;
        graph.replace_facts(&scope(), "b.md", &strings(&["fato b"]))?;
        graph.prune_missing(&scope(), &strings(&["b.md"]))?;
        assert_eq!(graph.memory_files(&scope())?, vec!["b.md"]);
        assert_eq!(
            graph.counts()?,
            Counts {
                memories: 1,
                facts: 1,
                facts_embedded: 0,
                triples: 0
            }
        );
        Ok(())
    }

    #[test]
    fn native_files_are_those_with_a_marker_or_facts() -> TestResult {
        let graph = graph_with(&[("vazio.md", 1), ("fatos.md", 1), ("marcado.md", 1)])?;
        graph.replace_facts(&scope(), "fatos.md", &strings(&["x"]))?;
        graph.mark_memory_current(&scope(), "marcado.md", "s", "e")?;
        let mut files = graph.native_files(&scope())?;
        files.sort();
        assert_eq!(files, vec!["fatos.md", "marcado.md"]);
        Ok(())
    }

    #[test]
    fn import_replaces_the_scope_and_skips_orphans() -> TestResult {
        let graph = graph_with(&[("velho.md", 1)])?;
        let counts = graph.import(
            &scope(),
            &[ImportMemory {
                file: "a.md".into(),
                name: Some("a".into()),
                body: Some("corpo".into()),
                ..ImportMemory::default()
            }],
            &[
                ImportFact {
                    memory_file: "a.md".into(),
                    text: "fato".into(),
                    subject: Some("Gateway".into()),
                    embedding: Some(vec![1.0, 0.0]),
                    ..ImportFact::default()
                },
                ImportFact {
                    memory_file: "sem-dono.md".into(),
                    text: "orfao".into(),
                    ..ImportFact::default()
                },
            ],
            &[],
        )?;
        assert_eq!(
            counts,
            Counts {
                memories: 1,
                facts: 1,
                facts_embedded: 1,
                triples: 0
            }
        );
        // Branch B of the facts query: a fact whose subject IS the token.
        assert_eq!(
            graph.facts_for_tokens(&scope(), &strings(&["gateway"]), 5)?,
            vec!["fato"]
        );
        Ok(())
    }

    #[test]
    fn a_superseded_legacy_fact_is_history_not_recall() -> TestResult {
        let graph = SqliteGraph::open_in_memory()?;
        let fact = |text: &str, valid_until: Option<i64>| ImportFact {
            memory_file: "a.md".into(),
            text: text.into(),
            subject: Some("gateway".into()),
            embedding: Some(vec![1.0, 0.0]),
            valid_until,
            ..ImportFact::default()
        };
        graph.import(
            &scope(),
            &[ImportMemory {
                file: "a.md".into(),
                ..ImportMemory::default()
            }],
            &[fact("vivo", None), fact("antigo", Some(1))],
            &[],
        )?;
        assert_eq!(
            graph.facts_for_tokens(&scope(), &strings(&["gateway"]), 5)?,
            vec!["vivo"]
        );
        let hits = graph.keyword_files(&scope(), &strings(&["antigo"]), 5)?;
        assert!(hits.is_empty(), "{hits:?}");
        let hits = graph.semantic_files(&scope(), &[1.0, 0.0], 5)?;
        assert_eq!(hits[0].facts, vec!["vivo"]);
        Ok(())
    }

    #[test]
    fn has_word_needs_delimiters_on_both_sides() {
        assert!(has_word("pdv-relay", "pdv"));
        assert!(has_word("o gateway", "gateway"));
        assert!(!has_word("restarts", "esta"));
        assert!(!has_word("gateways", "gateway"));
        assert!(has_word("a nó b", "nó"));
        assert!(!has_word("anó", "nó"));
    }
}
