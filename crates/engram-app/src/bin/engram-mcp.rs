use clap::Parser;
use engram_hybrid::recall;
use serde_json::{Value, json};
use std::{
    io::{BufRead, Write},
    path::{Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long, env = "ENGRAM_CONFIG")]
    config: Option<PathBuf>,
    /// Default store for calls that do not name one. Resolved from the environment
    /// and the session's working directory when omitted.
    // allow_hyphen_values because EVERY engram slug starts with '-' (it is a
    // path with separators replaced), so clap read `--slug -home-alice` as a
    // missing value followed by an unknown flag. The documented invocation was
    // unusable as typed.
    #[arg(long, allow_hyphen_values = true)]
    slug: Option<String>,
    /// The agent identity this server serves, fixed for the life of the process.
    /// Required on an install that defines tenants.
    #[arg(long, env = "ENGRAM_TENANT")]
    tenant: Option<String>,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();
    let config = engram_paths::config_path(args.config);
    // An MCP server is started by the client inside the project directory, so the
    // cwd-derived store is the right default; "-root" was correct on one machine.
    //
    // Resolved once, here, and fatally: a server that started without knowing
    // its identity would have to decide per call, and the only safe answer then
    // is to refuse every call — better to not come up at all, where the client
    // surfaces the error to the operator.
    // Resolve the IDENTITY here, fatally; defer the STORE to the tools that
    // need one.
    //
    // Resolving both up front was wrong: a tenant owning several memory stores
    // with none matching the launch directory made the whole server refuse to
    // start, taking recall down with it. A slug failure now surfaces only on
    // the tool that needs a slug.
    //
    // The identity still follows the launch directory's store when --tenant is
    // absent, so one global registration serves every identity on the host.
    let derived = engram_paths::resolve_slug_in_cwd(None);
    let tenant = match engram_config::Config::load(&config)
        .map_err(|e| e.to_string())
        .and_then(|loaded| {
            engram_tenant::resolve_tenant(
                &loaded,
                args.tenant.as_deref(),
                &derived,
                args.slug.as_deref(),
            )
            .map_err(|e| e.to_string())
        }) {
        Ok(tenant) => tenant,
        Err(error) => {
            eprintln!("engram-mcp: {error}");
            std::process::exit(2);
        }
    };
    let default_slug = args.slug.clone().unwrap_or(derived);
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines().map_while(Result::ok) {
        let Ok(request) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let Some(id) = request.get("id") else {
            continue;
        };
        let response = match request
            .get("method")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "initialize" => ok(
                id,
                json!({"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"engram-rust","version":"0.1.0"}}),
            ),
            "tools/list" => ok(
                id,
                json!({"tools":[
                    {"name":"memory_recall","description":"Primary memory recall. Uses the configured graph backend; Graphiti compatibility preserves Graphiti ordering exactly.","inputSchema":{"type":"object","properties":{"q":{"type":"string"},"slug":{"type":"string"},"k":{"type":"integer","minimum":1,"maximum":20}},"required":["q"]}},
                    {"name":"memory_recall_hybrid","description":"Alias for memory_recall, retained for existing callers.","inputSchema":{"type":"object","properties":{"q":{"type":"string"},"slug":{"type":"string"},"k":{"type":"integer","minimum":1,"maximum":20}},"required":["q"]}}
                ]}),
            ),
            "tools/call" => {
                call(
                    id,
                    request.get("params").unwrap_or(&Value::Null),
                    &config,
                    &tenant,
                    &default_slug,
                )
                .await
            }
            _ => err(id, -32601, "method not found"),
        };
        let _ = writeln!(stdout, "{response}");
        let _ = stdout.flush();
    }
}

/// Note what is NOT a tool argument: the tenant.
///
/// It is fixed for the life of the process, from `--tenant` on the command line
/// the client launched. Exposing it in a tool schema would let the model choose
/// its own identity and read another agent's memories simply by asking — the
/// one thing the tenant boundary exists to prevent. `slug` IS exposed, and is
/// therefore validated against the tenant on every call.
async fn call(
    id: &Value,
    params: &Value,
    config: &Path,
    tenant: &engram_tenant::Tenant,
    default_slug: &str,
) -> Value {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let arguments = params.get("arguments").unwrap_or(&Value::Null);
    match name {
        "memory_recall" | "memory_recall_hybrid" => {}
        _ => return err(id, -32602, "unknown tool"),
    }
    let Some(query) = arguments
        .get("q")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
    else {
        return err(id, -32602, "q is required");
    };
    // The model supplied this, so it is checked rather than trusted: a slug
    // belonging to another identity is refused, and the refusal is returned to
    // the model as a tool error so it can correct itself.
    let requested = arguments
        .get("slug")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|slug| !slug.is_empty());
    let slug = match tenant.choose_slug(requested, default_slug) {
        Ok(slug) => slug,
        Err(error) => {
            return ok(
                id,
                json!({"content":[{"type":"text","text":error.to_string()}],"isError":true}),
            );
        }
    };
    let k = arguments
        .get("k")
        .and_then(Value::as_u64)
        .unwrap_or(6)
        .clamp(1, 20) as usize;
    match recall(config, tenant, &slug, query, k).await {
        Ok(result) => ok(
            id,
            json!({"content":[{"type":"text","text":serde_json::to_string(&result).unwrap()}]}),
        ),
        Err(error) => ok(
            id,
            json!({"content":[{"type":"text","text":error}],"isError":true}),
        ),
    }
}
fn ok(id: &Value, result: Value) -> Value {
    json!({"jsonrpc":"2.0","id":id,"result":result})
}
fn err(id: &Value, code: i32, message: &str) -> Value {
    json!({"jsonrpc":"2.0","id":id,"error":{"code":code,"message":message}})
}
