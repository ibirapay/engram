use std::{fs, path::Path};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct Memory {
    pub file: String,
    pub name: String,
    pub description: String,
    pub memory_type: String,
    pub body: String,
    /// Source file mtime, seconds since the epoch.
    ///
    /// The operator's own ordering of events, which nothing else in the graph
    /// records: `created_at`/`updated_at` on a node are *ingestion* times, so
    /// anything that reasons about which claim came first from those is really
    /// reasoning about the order we happened to sync in. Supersession needs the
    /// difference (see `APPLY_SUPERSESSION` in engram-graph).
    pub source_mtime: i64,
}

#[derive(Debug, Error)]
pub enum StoreError {
    #[error("could not read memory store: {0}")]
    Read(#[from] std::io::Error),
}

/// One memory file as the directory lists it: enough to tell whether a cached index
/// of it is still current, without opening it.
#[derive(Clone, Debug, PartialEq)]
pub struct FileStamp {
    pub file: String,
    /// Modification time in nanoseconds since the epoch (0 when unreadable).
    pub mtime_ns: i64,
    pub size: i64,
}

/// The memory files in `dir` (`*.md`, minus the derived `MEMORY*.md` indexes), sorted
/// by name. Reads the directory and stats each file; opens none of them.
pub fn list_files(dir: impl AsRef<Path>) -> Result<Vec<FileStamp>, StoreError> {
    let mut stamps = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("md") || is_index(&path) {
            continue;
        }
        let Some(file) = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
        else {
            continue;
        };
        let meta = entry.metadata()?;
        stamps.push(FileStamp {
            file,
            mtime_ns: meta
                .modified()
                .ok()
                .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
                .and_then(|age| i64::try_from(age.as_nanos()).ok())
                .unwrap_or_default(),
            size: i64::try_from(meta.len()).unwrap_or_default(),
        });
    }
    stamps.sort_by(|left, right| left.file.cmp(&right.file));
    Ok(stamps)
}

/// A bare memory file name: no path separators, `.md`, not a derived index. Anything
/// else is refused, because the names that reach [`load_file`] can come from a vector
/// payload or a graph node, and `dir.join("../x")` would read outside the store.
pub fn is_memory_file_name(file: &str) -> bool {
    !file.is_empty()
        && file.ends_with(".md")
        && !file.contains(['/', '\\', '\0'])
        && file != "."
        && file != ".."
        && !is_index(Path::new(file))
}

/// Read ONE memory from `dir` by file name. `Ok(None)` when the name is not a memory
/// file name or the file is gone.
pub fn load_file(dir: impl AsRef<Path>, file: &str) -> Result<Option<Memory>, StoreError> {
    if !is_memory_file_name(file) {
        return Ok(None);
    }
    let path = dir.as_ref().join(file);
    let meta = match fs::metadata(&path) {
        Ok(meta) if meta.is_file() => meta,
        Ok(_) => return Ok(None),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let source_mtime = meta
        .modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|age| age.as_secs() as i64)
        .unwrap_or_default();
    let raw = match fs::read_to_string(&path) {
        Ok(raw) => raw,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    Ok(Some(build(file, &raw, source_mtime)))
}

pub fn load(dir: impl AsRef<Path>) -> Result<Vec<Memory>, StoreError> {
    let mut memories = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("md") || is_index(&path) {
            continue;
        }
        let source_mtime = entry
            .metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|age| age.as_secs() as i64)
            .unwrap_or_default();
        let raw = fs::read_to_string(&path)?;
        let file = path
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        memories.push(build(&file, &raw, source_mtime));
    }
    memories.sort_by(|left, right| left.file.cmp(&right.file));
    Ok(memories)
}

fn build(file: &str, raw: &str, source_mtime: i64) -> Memory {
    let (meta, body) = parse(raw);
    let fallback = file.trim_end_matches(".md").replace('_', " ");
    Memory {
        file: file.to_string(),
        name: meta
            .get("name")
            .cloned()
            .unwrap_or_else(|| fallback.clone()),
        description: meta.get("description").cloned().unwrap_or_default(),
        memory_type: meta
            .get("type")
            .cloned()
            .unwrap_or_else(|| "reference".into()),
        body,
        source_mtime,
    }
}

/// The store's own derived indexes — `MEMORY.md`, `MEMORY_FULL.md` — are lists of
/// every memory's name and description. They are not memories. Only `MEMORY.md` was
/// ever excluded, so the full index was read as one more document: every topic word
/// in the store exists in it (91 KB of them), which lets it win a BM25 ranking, and
/// it drags up the average document length that every real memory is normalised
/// against. Derived, regenerable, and named `MEMORY*` by both index writers.
fn is_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| name.starts_with("MEMORY") && name.ends_with(".md"))
}

/// Parse a memory's YAML frontmatter into a flat key/value map plus its body.
///
/// The canonical memory format puts the type one level down:
///
/// ```yaml
/// ---
/// name: gateway-dns
/// description: ...
/// metadata:
///   type: project
/// ---
/// ```
///
/// The previous line-based reader skipped every indented line outright, so
/// `type` was never found and EVERY memory fell back to `"reference"` — which then
/// became the `type` payload in Qdrant, making type filters over a Rust-built index
/// wrong for the whole store. Keys under `metadata` are flattened to the top level
/// here (and the nested value wins, since that is where the writers put it).
pub fn parse(raw: &str) -> (std::collections::BTreeMap<String, String>, String) {
    let mut meta = std::collections::BTreeMap::new();
    // Tolerate CRLF and a UTF-8 BOM: memories arrive by import and hand-editing,
    // not only from our own writers.
    let normalized = raw.trim_start_matches('\u{feff}').replace("\r\n", "\n");
    let Some(rest) = normalized.strip_prefix("---\n") else {
        return (meta, raw.to_string());
    };
    let Some((frontmatter, body)) = rest.split_once("\n---\n") else {
        return (meta, raw.to_string());
    };
    let mut nested: Vec<(String, String)> = Vec::new();
    for line in frontmatter.lines() {
        let indented = line.starts_with([' ', '\t']);
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim().to_string();
        let value = scalar(value);
        if key.is_empty() {
            continue;
        }
        if indented {
            // One level of nesting is all the format uses; deeper structure is not
            // addressable by a flat map and is left to the Python readers.
            nested.push((key, value));
        } else if !value.is_empty() {
            meta.insert(key, value);
        }
    }
    // Nested keys last: `metadata.type` is the canonical location, so it must beat
    // any stray top-level `type`.
    for (key, value) in nested {
        if !value.is_empty() {
            meta.insert(key, value);
        }
    }
    (meta, body.to_string())
}

fn scalar(value: &str) -> String {
    let value = value.trim();
    // Strip a matching pair of quotes only, so an apostrophe inside a bare value
    // survives.
    for quote in ['"', '\''] {
        if value.len() >= 2 && value.starts_with(quote) && value.ends_with(quote) {
            return value[1..value.len() - 1].to_string();
        }
    }
    value.to_string()
}

#[cfg(test)]
mod tests {
    #[test]
    fn load_file_refuses_names_that_leave_the_store() {
        let dir = std::env::temp_dir().join(format!("engram-store-lf-{}", std::process::id()));
        let inner = dir.join("memory");
        assert!(std::fs::create_dir_all(&inner).is_ok());
        assert!(std::fs::write(dir.join("secret.md"), "---\nname: x\n---\nsegredo\n").is_ok());
        assert!(std::fs::write(inner.join("ok.md"), "---\nname: ok\n---\ncorpo\n").is_ok());
        for bad in [
            "../secret.md",
            "/etc/passwd",
            "a/b.md",
            "..",
            "MEMORY.md",
            "",
            "x.txt",
        ] {
            assert!(
                matches!(load_file(&inner, bad), Ok(None)),
                "{bad:?} must be refused"
            );
        }
        assert!(matches!(load_file(&inner, "missing.md"), Ok(None)));
        assert!(matches!(load_file(&inner, "ok.md"), Ok(Some(m)) if m.name == "ok"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    use super::*;

    /// The derived index is not a memory. `MEMORY_FULL.md` — 91 KB of every
    /// memory's name and description — was being ranked as if it were one, and it
    /// raised the average document length every real memory is normalised against.
    #[test]
    fn skips_every_derived_index_file() {
        for name in ["MEMORY.md", "MEMORY_FULL.md"] {
            assert!(
                is_index(Path::new(name)),
                "{name} must not load as a memory"
            );
        }
        for name in ["memory_notes.md", "reference_memory_audit.md"] {
            assert!(!is_index(Path::new(name)), "{name} is a memory");
        }
    }

    #[test]
    fn parses_canonical_frontmatter() {
        let (meta, body) =
            parse("---\nname: Gateway\ndescription: DNS\nmetadata:\n  type: reference\n---\nBody");
        assert_eq!(meta["name"], "Gateway");
        assert_eq!(body, "Body");
    }

    /// The regression this file existed to have: the canonical nested type.
    /// The old test fed exactly this shape and then asserted only `name`.
    #[test]
    fn reads_the_nested_metadata_type() {
        for (yaml, want) in [
            ("metadata:\n  type: project", "project"),
            (
                "metadata:\n  node_type: memory\n  type: feedback",
                "feedback",
            ),
            ("metadata:\n\ttype: user", "user"),
            ("metadata:\n  type: \"reference\"", "reference"),
            // a top-level type still works, and the nested one wins over it
            ("type: reference\nmetadata:\n  type: project", "project"),
        ] {
            let raw = format!("---\nname: n\ndescription: d\n{yaml}\n---\nBody");
            let (meta, body) = parse(&raw);
            assert_eq!(
                meta.get("type").map(String::as_str),
                Some(want),
                "failed for {yaml:?}"
            );
            assert_eq!(body, "Body");
        }
    }

    #[test]
    fn a_memory_with_no_type_is_still_a_reference() {
        let (meta, _) = parse("---\nname: n\ndescription: d\n---\nBody");
        assert_eq!(meta.get("type"), None, "load() supplies the default");
    }

    #[test]
    fn tolerates_crlf_and_a_bom() {
        let (meta, body) = parse(
            "\u{feff}---\r\nname: Gateway\r\nmetadata:\r\n  type: project\r\n---\r\nBody\r\n",
        );
        assert_eq!(meta["name"], "Gateway");
        assert_eq!(meta["type"], "project");
        assert_eq!(body.trim(), "Body");
    }

    #[test]
    fn a_file_with_no_frontmatter_is_all_body() {
        let (meta, body) = parse("just prose\nmore prose\n");
        assert!(meta.is_empty());
        assert_eq!(body, "just prose\nmore prose\n");
    }
}
