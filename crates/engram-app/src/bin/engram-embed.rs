//! Always-on CPU embedding server for engram's recall hot path (127.0.0.1:11435).
//!
//! Drop-in replacement for `~/.claude/engram-embed-server.py`: same model (fastembed
//! nomic-embed-text-v1.5, 768-dim, mean pooling, normalised), same routes, same JSON,
//! so neither the Rust engine (`embed.url: .../v1`) nor the Python callers change.
//!
//!   GET  /health          -> {"ok": true, "model": M, "dim": 768}
//!   POST /api/embed       {"input": str | [str]} -> {"embeddings": [[..]], "model": M}   (Ollama shape)
//!   POST /v1/embeddings   {"input": str | [str]} -> {"object": "list", "data": [..]}     (OpenAI shape)
//!
//! Why it was replaced: the Python server grew to 10.9 GB RSS and tipped main into a
//! global OOM on 2026-10-24 that also killed the local chat model. The cause was not
//! Python itself: it truncated at 8192 tokens, one long memory body made ONNX allocate
//! gigabytes of attention activations, and its arena never gave them back (reproduced:
//! one 8192-token input took it from 1.2 to 12.3 GB). Hence `--max-length` below.
//! Vectors below the cutoff match the Python server at cosine 1.000000; Python's were
//! unnormalised, which the Cosine-distance Qdrant collection ignores.
#![forbid(unsafe_code)]

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State, rejection::BytesRejection},
    http::StatusCode,
    routing::{get, post},
};
use clap::Parser;
use fastembed::{EmbeddingModel, ModelInfo, TextEmbedding, TextInitOptions};
use serde_json::{Value, json};
use std::{
    net::SocketAddr,
    path::PathBuf,
    process::ExitCode,
    sync::{Arc, Mutex},
};

/// A memory description, not a document — same cap as the Python server.
const MAX_BODY: usize = 1 << 20;
const DEFAULT_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";

#[derive(Parser)]
struct Args {
    #[arg(long, env = "ENGRAM_EMBED_HOST", default_value = "127.0.0.1")]
    host: String,
    #[arg(long, env = "ENGRAM_EMBED_PORT", default_value_t = 11435)]
    port: u16,
    /// Resolved by engram-paths when omitted; only `embed.fastembed_model` is read.
    #[arg(long, env = "ENGRAM_CONFIG")]
    config: Option<PathBuf>,
    /// Where the ONNX model is downloaded once and kept (not /tmp: wiped on reboot).
    #[arg(long, env = "FASTEMBED_CACHE_DIR")]
    cache_dir: Option<PathBuf>,
    /// Token truncation. 2048 = the model's max_position_embeddings. Python fastembed
    /// used the tokenizer's model_max_length (8192), and attention memory grows with
    /// its square; measured on main over all 789 memories: 512 -> 0.48 GB peak RSS,
    /// 2048 -> 2.6 GB, 8192 -> over 6 GB on the first long file (OOM-killed in a cap).
    #[arg(long, env = "ENGRAM_EMBED_MAX_LENGTH", default_value_t = 2048)]
    max_length: usize,
}

struct App {
    model_name: String,
    dim: usize,
    model: Mutex<TextEmbedding>,
}

type Reply = (StatusCode, Json<Value>);

fn error(code: StatusCode, message: impl Into<String>) -> Reply {
    (code, Json(json!({ "error": message.into() })))
}

/// `embed.fastembed_model` from engram.yaml, the key the Python server read. A
/// missing file or key means the default; a malformed file is an error, because
/// silently serving a different model than the index was built with corrupts recall.
fn configured_model(config: Option<PathBuf>) -> Result<String, String> {
    let path = engram_paths::config_path(config);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(DEFAULT_MODEL.into()),
        Err(e) => return Err(format!("read {}: {e}", path.display())),
    };
    let yaml: serde_yaml::Value =
        serde_yaml::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    Ok(yaml
        .get("embed")
        .and_then(|embed| embed.get("fastembed_model"))
        .and_then(serde_yaml::Value::as_str)
        .filter(|name| !name.trim().is_empty())
        .unwrap_or(DEFAULT_MODEL)
        .to_string())
}

/// Resolve the Hugging Face code ("nomic-ai/nomic-embed-text-v1.5"), the name Python
/// fastembed takes. Several fastembed models can share one code (the quantized
/// `...V15Q` does), and the list is built from a HashMap whose order changes per
/// process, so first-match would pick a different model on each start. When the
/// code is shared, take the unquantized `onnx/model.onnx` that Python loads.
/// Unknown or still-ambiguous names are fatal: no fallback model.
fn resolve_model(model_name: &str) -> Result<ModelInfo<EmbeddingModel>, String> {
    let mut matches: Vec<_> = TextEmbedding::list_supported_models()
        .into_iter()
        .filter(|info| info.model_code.eq_ignore_ascii_case(model_name))
        .collect();
    if matches.len() > 1 {
        matches.retain(|info| info.model_file == "onnx/model.onnx");
    }
    match matches.len() {
        0 => Err(format!("fastembed has no model named {model_name:?}")),
        1 => matches
            .pop()
            .ok_or_else(|| format!("fastembed has no model named {model_name:?}")),
        n => Err(format!("{model_name:?} matches {n} fastembed models")),
    }
}

fn load(
    model_name: &str,
    cache_dir: Option<PathBuf>,
    max_length: usize,
) -> Result<(TextEmbedding, usize), String> {
    let info = resolve_model(model_name)?;
    let mut options = TextInitOptions::new(info.model.clone())
        .with_max_length(max_length)
        .with_show_download_progress(false);
    if let Some(dir) = cache_dir {
        options = options.with_cache_dir(dir);
    }
    let mut model =
        TextEmbedding::try_new(options).map_err(|e| format!("load {model_name}: {e}"))?;
    // Warm-up doubles as the dimension probe, exactly like the Python server.
    let dim = model
        .embed(["warmup"], None)
        .map_err(|e| format!("warm-up embed: {e}"))?
        .first()
        .map(Vec::len)
        .ok_or("warm-up embed returned no vector")?;
    if dim != info.dim {
        return Err(format!(
            "{model_name}: got {dim}-dim vectors, expected {}",
            info.dim
        ));
    }
    Ok((model, dim))
}

/// `input` may be one string or a batch (Ollama accepts both); non-string batch
/// items are stringified, as the Python server's `str(t)` did.
fn texts(body: Result<Bytes, BytesRejection>) -> Result<Vec<String>, Reply> {
    let body = body.map_err(|rejection| error(rejection.status(), rejection.body_text()))?;
    if body.is_empty() {
        return Err(error(StatusCode::PAYLOAD_TOO_LARGE, "bad content-length"));
    }
    let request: Value = serde_json::from_slice(&body)
        .map_err(|e| error(StatusCode::BAD_REQUEST, format!("bad request: {e}")))?;
    match request.get("input") {
        Some(Value::String(text)) => Ok(vec![text.clone()]),
        Some(Value::Array(items)) => Ok(items
            .iter()
            .map(|item| match item {
                Value::String(text) => text.clone(),
                other => other.to_string(),
            })
            .collect()),
        _ => Err(error(
            StatusCode::BAD_REQUEST,
            "input must be a string or list",
        )),
    }
}

async fn embed(app: Arc<App>, texts: Vec<String>) -> Result<Vec<Vec<f32>>, Reply> {
    let joined = tokio::task::spawn_blocking(move || {
        let mut model = app
            .model
            .lock()
            .map_err(|_| "embedding model lock poisoned".to_string())?;
        model.embed(texts, None).map_err(|e| e.to_string())
    })
    .await;
    match joined {
        Ok(Ok(vectors)) => Ok(vectors),
        Ok(Err(message)) => Err(error(StatusCode::INTERNAL_SERVER_ERROR, message)),
        Err(e) => Err(error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("embed task: {e}"),
        )),
    }
}

async fn health(State(app): State<Arc<App>>) -> Reply {
    (
        StatusCode::OK,
        Json(json!({ "ok": true, "model": app.model_name, "dim": app.dim })),
    )
}

async fn ollama(State(app): State<Arc<App>>, body: Result<Bytes, BytesRejection>) -> Reply {
    let texts = match texts(body) {
        Ok(texts) => texts,
        Err(reply) => return reply,
    };
    let model_name = app.model_name.clone();
    match embed(app, texts).await {
        Ok(vectors) => (
            StatusCode::OK,
            Json(json!({ "embeddings": vectors, "model": model_name })),
        ),
        Err(reply) => reply,
    }
}

async fn openai(State(app): State<Arc<App>>, body: Result<Bytes, BytesRejection>) -> Reply {
    let texts = match texts(body) {
        Ok(texts) => texts,
        Err(reply) => return reply,
    };
    let model_name = app.model_name.clone();
    match embed(app, texts).await {
        Ok(vectors) => {
            let data: Vec<Value> = vectors
                .into_iter()
                .enumerate()
                .map(|(index, embedding)| {
                    json!({ "object": "embedding", "index": index, "embedding": embedding })
                })
                .collect();
            (
                StatusCode::OK,
                Json(json!({ "object": "list", "model": model_name, "data": data })),
            )
        }
        Err(reply) => reply,
    }
}

async fn not_found() -> Reply {
    error(StatusCode::NOT_FOUND, "not found")
}

async fn run(args: Args) -> Result<(), String> {
    let model_name = configured_model(args.config)?;
    let (model, dim) = load(&model_name, args.cache_dir, args.max_length)?;
    let addr: SocketAddr = format!("{}:{}", args.host, args.port)
        .parse()
        .map_err(|e| format!("bad bind address {}:{}: {e}", args.host, args.port))?;
    let app = Arc::new(App {
        model_name,
        dim,
        model: Mutex::new(model),
    });
    let router = Router::new()
        .route("/", get(health))
        .route("/health", get(health))
        .route("/api/embed", post(ollama))
        .route("/v1/embeddings", post(openai))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(MAX_BODY))
        .with_state(Arc::clone(&app));
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| format!("bind {addr}: {e}"))?;
    println!(
        "engram-embed: {} dim={} on http://{addr}",
        app.model_name, app.dim
    );
    axum::serve(listener, router)
        .await
        .map_err(|e| format!("serve: {e}"))
}

#[tokio::main]
async fn main() -> ExitCode {
    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("engram-embed: {message}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Result<Vec<String>, StatusCode> {
        texts(Ok(Bytes::from(body.to_string()))).map_err(|(code, _)| code)
    }

    #[test]
    fn input_shapes_match_the_python_server() -> Result<(), String> {
        assert_eq!(parse(r#"{"input":"a"}"#), Ok(vec!["a".to_string()]));
        assert_eq!(
            parse(r#"{"input":["a",7]}"#),
            Ok(vec!["a".to_string(), "7".to_string()])
        );
        assert_eq!(parse(r#"{"input":{}}"#), Err(StatusCode::BAD_REQUEST));
        assert_eq!(parse(r#"{"model":"x"}"#), Err(StatusCode::BAD_REQUEST));
        assert_eq!(parse("not json"), Err(StatusCode::BAD_REQUEST));
        assert_eq!(parse(""), Err(StatusCode::PAYLOAD_TOO_LARGE));
        Ok(())
    }

    /// The quantized NomicEmbedTextV15Q shares the HF code; fastembed's model list
    /// comes from a HashMap, so a first-match lookup picked either model at random
    /// per process (cosine ~0.94 against the index). Python loads onnx/model.onnx.
    #[test]
    fn default_name_resolves_to_the_unquantized_model() -> Result<(), String> {
        let info = resolve_model(DEFAULT_MODEL)?;
        assert_eq!(info.model, EmbeddingModel::NomicEmbedTextV15);
        assert_eq!(info.model_file, "onnx/model.onnx");
        Ok(())
    }

    #[test]
    fn missing_config_means_default_model() -> Result<(), String> {
        let model = configured_model(Some(PathBuf::from("/nonexistent/engram.yaml")))?;
        assert_eq!(model, DEFAULT_MODEL);
        Ok(())
    }
}
