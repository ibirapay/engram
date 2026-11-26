//! A persistent keyword index over the memory store.
//!
//! [`rank`](super::rank) rebuilds its statistics from scratch on every call: it opens
//! and tokenizes every memory file. In the per-prompt hook that is ~1,100 files read
//! before any network call, and it was most of the hook's ~870 ms. This keeps the
//! same statistics (document length, per-term count and field weight) in SQLite, so a
//! prompt reads only the postings of its own query terms.
//!
//! It is deliberately NOT SQLite FTS5. The ranking here was tuned over several rounds
//! (compound-token ids, field boosts, stopwords, acronyms), and FTS5 has its own
//! tokenizer and its own bm25. This index reuses the same `tokenize`, the same field
//! weights and the same [`term_score`](super::term_score), so it returns exactly what
//! `rank` returns, bit for bit; a test holds that line.
//!
//! The index is derived data: the Markdown store stays the source of truth, and a
//! stale or unreadable index is rebuilt or bypassed, never trusted.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use engram_store::{FileStamp, StoreError, list_files, load_file};
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use thiserror::Error;

use super::{Hit, Ranked, STOPWORDS, term_score, tokenize};

#[derive(Debug, Error)]
pub enum IndexError {
    #[error("keyword index database: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("{0}")]
    Store(#[from] StoreError),
    #[error("keyword index path: {0}")]
    Io(#[from] std::io::Error),
}

/// What a [`KeywordIndex::sync`] did.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SyncStats {
    pub added: usize,
    pub updated: usize,
    pub removed: usize,
    pub unchanged: usize,
    /// The whole index was dropped first because the tokenizer changed.
    pub rebuilt: bool,
}

pub struct KeywordIndex {
    conn: Connection,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS docs (
    id INTEGER PRIMARY KEY,
    file TEXT NOT NULL UNIQUE,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    len INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS postings (
    term TEXT NOT NULL,
    doc INTEGER NOT NULL,
    count INTEGER NOT NULL,
    weight REAL NOT NULL,
    PRIMARY KEY (term, doc)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS postings_doc ON postings (doc);
";

/// Identifies the tokenizer. The indexed terms are only valid for the code that made
/// them, so an edit to `tokenize` or `STOPWORDS` changes this and the next sync
/// rebuilds the index instead of ranking old terms against new queries.
fn fingerprint() -> String {
    const PROBE: &str = "Nó DAS NAS sim API-1 pdv_relay robosats-gateway mineração é à \
        127.0.0.1:6333 Moisés 5 xy";
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |text: &str| {
        for byte in text.bytes().chain(std::iter::once(0)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    };
    for word in STOPWORDS {
        feed(word);
    }
    for token in tokenize(PROBE) {
        feed(&token);
    }
    format!("v1-{hash:016x}")
}

impl KeywordIndex {
    /// Open (creating if needed) the index at `path`.
    pub fn open(path: &Path) -> Result<Self, IndexError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        Self::init(conn)
    }

    /// An index that lives in memory only (tests).
    pub fn open_in_memory() -> Result<Self, IndexError> {
        Self::init(Connection::open_in_memory()?)
    }

    fn init(conn: Connection) -> Result<Self, IndexError> {
        conn.busy_timeout(std::time::Duration::from_millis(3000))?;
        // WAL: the daemon may be re-indexing while a prompt hook reads.
        conn.query_row("PRAGMA journal_mode = WAL", [], |_| Ok(()))
            .optional()?;
        conn.execute_batch("PRAGMA synchronous = NORMAL;")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self { conn })
    }

    /// Bring the index in line with the store directory: add new files, re-read
    /// changed ones (by mtime and size), drop deleted ones. Nothing is opened for a
    /// file that has not changed, so a steady-state sync is a directory listing.
    pub fn sync(&mut self, dir: &Path) -> Result<SyncStats, IndexError> {
        let stamps = list_files(dir)?;
        let mut stats = SyncStats::default();

        let stored: Option<String> = self
            .conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'fingerprint'",
                [],
                |row| row.get(0),
            )
            .optional()?;
        let wanted = fingerprint();
        let stale = stored.as_deref() != Some(wanted.as_str());

        let mut known: HashMap<String, (i64, i64, i64)> = HashMap::new();
        if !stale {
            let mut stmt = self
                .conn
                .prepare("SELECT file, id, mtime_ns, size FROM docs")?;
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                ))
            })?;
            for row in rows {
                let (file, id, mtime, size) = row?;
                known.insert(file, (id, mtime, size));
            }
        }

        let present: HashSet<&str> = stamps.iter().map(|stamp| stamp.file.as_str()).collect();
        let removed: Vec<i64> = known
            .iter()
            .filter(|(file, _)| !present.contains(file.as_str()))
            .map(|(_, (id, _, _))| *id)
            .collect();
        let changed: Vec<&FileStamp> = stamps
            .iter()
            .filter(|stamp| match known.get(&stamp.file) {
                Some((_, mtime, size)) => *mtime != stamp.mtime_ns || *size != stamp.size,
                None => true,
            })
            .collect();
        stats.unchanged = stamps.len() - changed.len();
        if !stale && removed.is_empty() && changed.is_empty() {
            return Ok(stats);
        }

        let tx = self
            .conn
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        if stale {
            tx.execute_batch("DELETE FROM postings; DELETE FROM docs;")?;
            tx.execute(
                "INSERT INTO meta (key, value) VALUES ('fingerprint', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                params![wanted],
            )?;
            stats.rebuilt = true;
        }
        for id in &removed {
            tx.execute("DELETE FROM postings WHERE doc = ?1", params![id])?;
            tx.execute("DELETE FROM docs WHERE id = ?1", params![id])?;
            stats.removed += 1;
        }
        for stamp in changed {
            // Gone between the listing and now: drop it from the index too.
            let Some(memory) = load_file(dir, &stamp.file)? else {
                if let Some((id, _, _)) = known.get(&stamp.file) {
                    tx.execute("DELETE FROM postings WHERE doc = ?1", params![id])?;
                    tx.execute("DELETE FROM docs WHERE id = ?1", params![id])?;
                    stats.removed += 1;
                }
                continue;
            };
            let ranked = Ranked::new(&memory);
            let existed = known.contains_key(&stamp.file);
            let id: i64 = tx.query_row(
                "INSERT INTO docs (file, mtime_ns, size, len) VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(file) DO UPDATE SET
                     mtime_ns = excluded.mtime_ns, size = excluded.size, len = excluded.len
                 RETURNING id",
                params![
                    stamp.file,
                    stamp.mtime_ns,
                    stamp.size,
                    i64::try_from(ranked.len).unwrap_or(i64::MAX)
                ],
                |row| row.get(0),
            )?;
            tx.execute("DELETE FROM postings WHERE doc = ?1", params![id])?;
            {
                let mut insert = tx.prepare_cached(
                    "INSERT INTO postings (term, doc, count, weight) VALUES (?1, ?2, ?3, ?4)",
                )?;
                for (term, count) in &ranked.counts {
                    let weight = ranked.weight.get(term).copied().unwrap_or(1.0);
                    insert.execute(params![
                        term,
                        id,
                        i64::try_from(*count).unwrap_or(i64::MAX),
                        weight
                    ])?;
                }
            }
            if existed {
                stats.updated += 1;
            } else {
                stats.added += 1;
            }
        }
        tx.commit()?;
        Ok(stats)
    }

    /// Same contract and same scores as [`rank`](super::rank) over the indexed store.
    pub fn rank(&self, query: &str, limit: usize) -> Result<Vec<Hit>, IndexError> {
        let mut terms: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();
        for term in tokenize(query) {
            if seen.insert(term.clone()) {
                terms.push(term);
            }
        }
        let (total, length_sum): (i64, i64) = self.conn.query_row(
            "SELECT COUNT(*), COALESCE(SUM(len), 0) FROM docs",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if terms.is_empty() || total == 0 {
            return Ok(Vec::new());
        }
        let total = total as f64;
        let average = length_sum as f64 / total;
        let mut scores: HashMap<i64, (String, f64)> = HashMap::new();
        let mut stmt = self.conn.prepare_cached(
            "SELECT p.doc, d.file, d.len, p.count, p.weight
             FROM postings p JOIN docs d ON d.id = p.doc WHERE p.term = ?1",
        )?;
        for term in &terms {
            let rows: Vec<(i64, String, i64, i64, f64)> = stmt
                .query_map(params![term], |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                })?
                .collect::<Result<_, _>>()?;
            let docs = rows.len() as f64;
            for (doc, file, len, count, weight) in rows {
                let entry = scores.entry(doc).or_insert((file, 0.0));
                entry.1 += term_score(count as f64, docs, total, weight, len as f64, average);
            }
        }
        let mut hits: Vec<Hit> = scores
            .into_values()
            .map(|(file, score)| Hit { file, score })
            .filter(|hit| hit.score > 0.0)
            .collect();
        hits.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.file.cmp(&right.file))
        });
        hits.truncate(limit);
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bm25;
    use std::fs;
    use std::path::PathBuf;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Result<Self, std::io::Error> {
            let dir = std::env::temp_dir().join(format!(
                "engram-keyword-index-{tag}-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|age| age.as_nanos())
                    .unwrap_or_default()
            ));
            fs::create_dir_all(&dir)?;
            Ok(Self(dir))
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    const TOPICS: [&str; 8] = [
        "gateway dns ipv6 resolver unbound",
        "mineração moisés energia fatura hash",
        "robosats coordinator lightning nó canal",
        "strata diskcache fila releitura contexto",
        "DAS simples nacional guia pagamento",
        "wallet android f-droid release assinatura",
        "pdv-relay deploy dedi systemd quadlet",
        "engram timer daemon curate harvest backend",
    ];

    fn write(dir: &Path, n: usize) -> Result<(), std::io::Error> {
        let topic = TOPICS[n % TOPICS.len()];
        let other = TOPICS[(n * 3 + 1) % TOPICS.len()];
        let text = format!(
            "---\nname: memoria-{n}-{}\ndescription: {topic}\nmetadata:\n  type: project\n---\n\
             {other} {topic} nota {n} {}\n",
            topic.split(' ').next().unwrap_or("x"),
            "repetido ".repeat(n % 5)
        );
        fs::write(dir.join(format!("memoria_{n}.md")), text)
    }

    fn corpus(tag: &str, count: usize) -> Result<TempDir, std::io::Error> {
        let dir = TempDir::new(tag)?;
        for n in 0..count {
            write(&dir.0, n)?;
        }
        // Derived indexes must never be ranked.
        fs::write(
            dir.0.join("MEMORY.md"),
            "# index\n- gateway dns mineração\n",
        )?;
        Ok(dir)
    }

    #[test]
    fn index_matches_the_in_memory_ranking_bit_for_bit() -> TestResult {
        let dir = corpus("parity", 60)?;
        let mut index = KeywordIndex::open_in_memory()?;
        index.sync(&dir.0)?;
        let memories = engram_store::load(&dir.0)?;
        for query in [
            "gateway dns",
            "mineracao moises energia",
            "como esta o timer do engram e o backend",
            "DAS simples",
            "pdv-relay deploy",
            "robosats nó lightning",
            "zzz inexistente",
            "",
            "dns dns dns",
        ] {
            let expected = bm25(&memories, query, 10);
            let got = index.rank(query, 10)?;
            assert_eq!(got.len(), expected.len(), "len for {query:?}");
            for (left, right) in got.iter().zip(&expected) {
                assert_eq!(left.file, right.file, "order for {query:?}");
                assert_eq!(
                    left.score.to_bits(),
                    right.score.to_bits(),
                    "score for {query:?} {}",
                    left.file
                );
            }
        }
        Ok(())
    }

    #[test]
    fn steady_state_sync_reads_no_file() -> TestResult {
        let dir = corpus("steady", 10)?;
        let mut index = KeywordIndex::open_in_memory()?;
        let first = index.sync(&dir.0)?;
        assert_eq!(first.added, 10);
        assert!(
            first.rebuilt,
            "a fresh database starts with a fingerprint write"
        );
        let second = index.sync(&dir.0)?;
        assert_eq!(
            second,
            SyncStats {
                unchanged: 10,
                ..SyncStats::default()
            }
        );
        Ok(())
    }

    #[test]
    fn sync_follows_edits_additions_and_deletions() -> TestResult {
        let dir = corpus("edits", 12)?;
        let mut index = KeywordIndex::open_in_memory()?;
        index.sync(&dir.0)?;
        assert!(index.rank("unicornio", 5)?.is_empty());

        fs::write(
            dir.0.join("memoria_3.md"),
            "---\nname: unicornio\ndescription: criatura rara\n---\ncorpo qualquer com unicornio\n",
        )?;
        write(&dir.0, 99)?;
        fs::remove_file(dir.0.join("memoria_5.md"))?;
        let stats = index.sync(&dir.0)?;
        assert_eq!((stats.updated, stats.added, stats.removed), (1, 1, 1));
        let hits = index.rank("unicornio", 5)?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file, "memoria_3.md");

        let memories = engram_store::load(&dir.0)?;
        let expected = bm25(&memories, "gateway dns mineração", 20);
        let got = index.rank("gateway dns mineração", 20)?;
        assert_eq!(
            got.iter().map(|hit| hit.file.as_str()).collect::<Vec<_>>(),
            expected
                .iter()
                .map(|hit| hit.file.as_str())
                .collect::<Vec<_>>()
        );
        assert!(got.iter().all(|hit| hit.file != "memoria_5.md"));
        Ok(())
    }

    #[test]
    fn a_changed_tokenizer_rebuilds_instead_of_trusting_old_terms() -> TestResult {
        let dir = corpus("fingerprint", 6)?;
        let mut index = KeywordIndex::open_in_memory()?;
        index.sync(&dir.0)?;
        index.conn.execute(
            "UPDATE meta SET value = 'old-tokenizer' WHERE key = 'fingerprint'",
            [],
        )?;
        let stats = index.sync(&dir.0)?;
        assert!(stats.rebuilt);
        assert_eq!(stats.added, 6);
        assert!(!index.rank("gateway dns", 3)?.is_empty());
        Ok(())
    }

    #[test]
    fn the_index_persists_across_opens() -> TestResult {
        let dir = corpus("persist", 8)?;
        let db = TempDir::new("persist-db")?;
        let path = db.0.join("nested").join("keyword.db");
        {
            let mut index = KeywordIndex::open(&path)?;
            index.sync(&dir.0)?;
        }
        let mut again = KeywordIndex::open(&path)?;
        let stats = again.sync(&dir.0)?;
        assert_eq!(stats.unchanged, 8);
        assert!(!stats.rebuilt);
        assert!(!again.rank("engram timer", 3)?.is_empty());
        Ok(())
    }
}
