//! Copy a store's native graph from Neo4j into the SQLite file (`graph.backend: sqlite`).
//!
//! Read-only on Neo4j. The scope's rows in the SQLite file are REPLACED, so the
//! command is safe to re-run. It reads facts and triples through their memory, so the
//! retired orphans Neo4j keeps are not carried over, and it skips the entity and
//! mention nodes that no query ever read.
//!
//!   engram-graph-to-sqlite [--config PATH] [--slug S] [--tenant T] [--db PATH] [--dry-run]
//!
//! The copy is checked by counting: memories, facts, embedded facts and triples read
//! from Neo4j must equal what the SQLite file holds afterwards, or the command fails.

use clap::Parser;
use engram_graph::neo4j::Neo4jReader;
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
    let neo4j = Neo4jReader::from_credentials(&creds).map_err(|error| error.to_string())?;

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
    Ok(())
}
