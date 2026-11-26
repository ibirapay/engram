//! A READ-ONLY reader for the old Neo4j native graph, kept for one job: copying a
//! store into the SQLite graph (`engram-graph-to-sqlite`). Nothing else reads Neo4j.
//! Delete this module once every install has been migrated.

use engram_tenant::GraphScope;
use reqwest::Client;
use serde::Deserialize;

use crate::GraphError;
use crate::sqlite::{ImportFact, ImportMemory, ImportTriple};

#[derive(Clone)]
pub struct Neo4jReader {
    endpoint: String,
    user: String,
    password: String,
    client: Client,
}

#[derive(Deserialize)]
struct Response {
    results: Vec<ResultSet>,
    errors: Vec<NeoError>,
}
#[derive(Deserialize)]
struct ResultSet {
    data: Vec<Row>,
}
#[derive(Deserialize)]
struct Row {
    row: Vec<serde_json::Value>,
}
#[derive(Deserialize)]
struct NeoError {
    message: String,
}

// ---------------------------------------------------------------------------
// Export to the SQLite backend (read-only). Only what recall can reach: facts and
// triples are read THROUGH their memory, so the orphans that retirement left behind
// (nodes whose HAS_FACT / HAS_TRIPLE edge was deleted) are not carried over.
// ---------------------------------------------------------------------------

const EXPORT_NATIVE_MEMORIES: &str = "\
MATCH (m:EngramMemory {tenant: $tenant, slug: $slug}) \
RETURN m.file, m.name, m.description, m.body, m.source_mtime, m.sha, \
       m.embedding_space, m.native_triple_version ORDER BY m.file";

const EXPORT_NATIVE_FACTS: &str = "\
MATCH (m:EngramMemory {tenant: $tenant, slug: $slug})-[:HAS_FACT]->(f:EngramFact) \
RETURN m.file, f.text, f.legacy_name, f.subject, f.object, f.embedding, \
       CASE WHEN f.valid_until IS NULL THEN null ELSE f.valid_until.epochSeconds END \
ORDER BY m.file, f.text SKIP $skip LIMIT $limit";

const EXPORT_NATIVE_TRIPLES: &str = "\
MATCH (m:EngramMemory {tenant: $tenant, slug: $slug})-[:HAS_TRIPLE]->(t:EngramTriple) \
RETURN m.file, t.key, t.subject, t.relation, t.object, t.confidence, t.temporal, t.status, \
       CASE WHEN t.valid_until IS NULL THEN null ELSE t.valid_until.epochSeconds END \
ORDER BY m.file, t.key";

const COUNT_NATIVE_FACTS: &str = "\
MATCH (m:EngramMemory {tenant: $tenant, slug: $slug})-[:HAS_FACT]->(f:EngramFact) \
RETURN count(f), count(f.embedding)";

impl Neo4jReader {
    /// Build a reader, optionally with an explicit HTTP(S) transaction endpoint.
    ///
    /// Required for a non-loopback Neo4j: basic-auth credentials must not cross a
    /// network in plaintext, and the derived endpoint is always `http://`.
    pub fn with_endpoint(
        uri: &str,
        database: &str,
        user: impl Into<String>,
        password: impl Into<String>,
        http_url: Option<&str>,
    ) -> Result<Self, GraphError> {
        let endpoint = match http_url.map(str::trim).filter(|url| !url.is_empty()) {
            Some(url) => explicit_endpoint(url, database)?,
            None => http_endpoint(uri, database)?,
        };
        Ok(Self {
            endpoint,
            user: user.into(),
            password: password.into(),
            client: Client::new(),
        })
    }

    /// Build from the resolved config credentials.
    pub fn from_credentials(creds: &engram_config::GraphCredentials) -> Result<Self, GraphError> {
        Self::with_endpoint(
            &creds.uri,
            &creds.database,
            creds.user.clone(),
            creds.password.expose().to_string(),
            creds.http_url.as_deref(),
        )
    }

    /// Every memory of the scope, as the SQLite importer wants it.
    pub async fn export_native_memories(
        &self,
        scope: &GraphScope,
    ) -> Result<Vec<ImportMemory>, GraphError> {
        let response = self
            .query(
                EXPORT_NATIVE_MEMORIES,
                serde_json::json!({"slug": scope.slug(), "tenant": scope.tenant()}),
            )
            .await?;
        let text = |row: &[serde_json::Value], at: usize| {
            row.get(at)
                .and_then(|value| value.as_str())
                .map(str::to_string)
        };
        Ok(response
            .results
            .into_iter()
            .flat_map(|set| set.data)
            .filter_map(|data| {
                let row = data.row;
                Some(ImportMemory {
                    file: text(&row, 0)?,
                    name: text(&row, 1),
                    description: text(&row, 2),
                    body: text(&row, 3),
                    source_mtime: row.get(4).and_then(|value| value.as_i64()),
                    sha: text(&row, 5),
                    embedding_space: text(&row, 6),
                    triple_version: row.get(7).and_then(|value| value.as_i64()),
                })
            })
            .collect())
    }

    /// One page of the scope's facts (embeddings are large: ask in pages).
    pub async fn export_native_facts(
        &self,
        scope: &GraphScope,
        skip: usize,
        limit: usize,
    ) -> Result<Vec<ImportFact>, GraphError> {
        let response = self
            .query(
                EXPORT_NATIVE_FACTS,
                serde_json::json!({"slug": scope.slug(), "tenant": scope.tenant(),
                    "skip": skip, "limit": limit}),
            )
            .await?;
        let text = |row: &[serde_json::Value], at: usize| {
            row.get(at)
                .and_then(|value| value.as_str())
                .map(str::to_string)
        };
        Ok(response
            .results
            .into_iter()
            .flat_map(|set| set.data)
            .filter_map(|data| {
                let row = data.row;
                Some(ImportFact {
                    memory_file: text(&row, 0)?,
                    text: text(&row, 1)?,
                    legacy_name: text(&row, 2),
                    subject: text(&row, 3),
                    object: text(&row, 4),
                    embedding: row.get(5).and_then(|value| value.as_array()).map(|values| {
                        values
                            .iter()
                            .map(|value| value.as_f64().unwrap_or_default() as f32)
                            .collect()
                    }),
                    valid_until: row.get(6).and_then(|value| value.as_i64()),
                })
            })
            .collect())
    }

    pub async fn export_native_triples(
        &self,
        scope: &GraphScope,
    ) -> Result<Vec<ImportTriple>, GraphError> {
        let response = self
            .query(
                EXPORT_NATIVE_TRIPLES,
                serde_json::json!({"slug": scope.slug(), "tenant": scope.tenant()}),
            )
            .await?;
        let text = |row: &[serde_json::Value], at: usize| {
            row.get(at)
                .and_then(|value| value.as_str())
                .map(str::to_string)
        };
        Ok(response
            .results
            .into_iter()
            .flat_map(|set| set.data)
            .filter_map(|data| {
                let row = data.row;
                Some(ImportTriple {
                    memory_file: text(&row, 0)?,
                    key: text(&row, 1)?,
                    subject: text(&row, 2)?,
                    relation: text(&row, 3)?,
                    object: text(&row, 4)?,
                    confidence: row
                        .get(5)
                        .and_then(|value| value.as_f64())
                        .unwrap_or_default(),
                    temporal: text(&row, 6).unwrap_or_default(),
                    status: text(&row, 7).unwrap_or_else(|| "active".into()),
                    valid_until: row.get(8).and_then(|value| value.as_i64()),
                })
            })
            .collect())
    }

    /// `(facts, facts with an embedding)` reachable from the scope's memories.
    pub async fn count_native_facts(&self, scope: &GraphScope) -> Result<(i64, i64), GraphError> {
        let response = self
            .query(
                COUNT_NATIVE_FACTS,
                serde_json::json!({"slug": scope.slug(), "tenant": scope.tenant()}),
            )
            .await?;
        let row = response
            .results
            .into_iter()
            .flat_map(|set| set.data)
            .next()
            .map(|data| data.row)
            .unwrap_or_default();
        let at = |index: usize| row.get(index).and_then(|value| value.as_i64()).unwrap_or(0);
        Ok((at(0), at(1)))
    }

    async fn query(
        &self,
        statement: &str,
        parameters: serde_json::Value,
    ) -> Result<Response, GraphError> {
        let response = self
            .client
            .post(&self.endpoint)
            .basic_auth(&self.user, Some(&self.password))
            .json(&serde_json::json!({"statements": [{"statement": statement, "parameters": parameters}]}))
            .send()
            .await?
            .error_for_status()?
            .json::<Response>()
            .await?;
        if let Some(error) = response.errors.first() {
            return Err(GraphError::Database(error.message.clone()));
        }
        Ok(response)
    }
}

pub fn http_endpoint(uri: &str, database: &str) -> Result<String, GraphError> {
    let trimmed = uri.trim();
    let authority = trimmed
        .split("://")
        .nth(1)
        .ok_or_else(|| GraphError::Uri("expected bolt://host:port".into()))?;
    let host = authority
        .split('/')
        .next()
        .unwrap_or_default()
        .split(':')
        .next()
        .unwrap_or_default();
    if !matches!(host, "127.0.0.1" | "localhost" | "::1") {
        return Err(GraphError::Uri(format!(
            "remote Neo4j ({host}) requires an explicit HTTPS endpoint — set graph.neo4j_http_url"
        )));
    }
    Ok(format!("http://{host}:7474/db/{database}/tx/commit"))
}

/// Validate an operator-supplied transaction endpoint.
///
/// Plain `http` is allowed for any host: this deployment reaches a remote Neo4j
/// over the tailnet (WireGuard-encrypted at the wire), where plaintext HTTP
/// carries no credential exposure. An operator putting Neo4j on a network they
/// do not trust should set an `https://` endpoint instead.
pub fn explicit_endpoint(url: &str, database: &str) -> Result<String, GraphError> {
    let trimmed = url.trim().trim_end_matches('/');
    let (scheme, _) = trimmed
        .split_once("://")
        .ok_or_else(|| GraphError::Uri("neo4j_http_url must be an absolute HTTP(S) URL".into()))?;
    match scheme {
        "https" | "http" => {}
        other => return Err(GraphError::Uri(format!("unsupported scheme {other}"))),
    }
    // Accept either a bare base URL or a full transaction endpoint.
    if trimmed.contains("/tx/commit") {
        return Ok(trimmed.to_string());
    }
    Ok(format!("{trimmed}/db/{database}/tx/commit"))
}

#[cfg(test)]
mod tests {
    use super::{explicit_endpoint, http_endpoint};

    #[test]
    fn loopback_only_without_an_explicit_endpoint() {
        assert_eq!(
            http_endpoint("bolt://127.0.0.1:7687", "neo4j").unwrap(),
            "http://127.0.0.1:7474/db/neo4j/tx/commit"
        );
        let error = http_endpoint("bolt://10.0.0.9:7687", "neo4j").unwrap_err();
        assert!(
            error.to_string().contains("graph.neo4j_http_url"),
            "the error must name the setting that fixes it: {error}"
        );
    }

    /// The remote path the old error message promised but never implemented.
    #[test]
    fn an_explicit_endpoint_enables_remote_neo4j() {
        assert_eq!(
            explicit_endpoint("https://neo.example:7473", "memories").unwrap(),
            "https://neo.example:7473/db/memories/tx/commit"
        );
        // a full endpoint is passed through
        assert_eq!(
            explicit_endpoint("https://neo.example:7473/db/x/tx/commit", "ignored").unwrap(),
            "https://neo.example:7473/db/x/tx/commit"
        );
        // http is allowed: the remote endpoint we actually deploy lives on the
        // tailnet, which is encrypted at the wire (see fn docs).
        assert!(explicit_endpoint("http://127.0.0.1:7474", "neo4j").is_ok());
        assert_eq!(
            explicit_endpoint("http://100.64.0.4:7474", "neo4j").unwrap(),
            "http://100.64.0.4:7474/db/neo4j/tx/commit"
        );
        assert!(explicit_endpoint("neo.example", "neo4j").is_err());
    }
}
