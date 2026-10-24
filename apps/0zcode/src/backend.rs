//! 0zcode Rust backend — thin HTTP edge that links to the SENTINEL core engine.
//!
//! Architecture:
//!
//! ```text
//! 0zcode TUI / clients
//!        │
//!        ▼  :8100 (this process, pure Rust std)
//!  0zcode-backend
//!        │  proxy /v1/chat, /v1/chat/stream, /v1/models, tools, …
//!        ▼  :8000
//!  SENTINEL core (Python / packages/sentinel/serving)
//! ```
//!
//! Run:
//!   0zcode serve
//!   0zcode serve --port 8100 --core http://127.0.0.1:8000
//!
//! Env:
//!   ZEROZ_BACKEND_PORT   default 8100
//!   ZEROZ_AERIS_URL      core engine (default http://127.0.0.1:8000)
//!   ZEROZ_AERIS_KEY      optional X-API-Key forwarded upstream

use crate::config::Config;
use crate::error::{Error, Result};
use crate::llm;
use crate::VERSION;
use std::env;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread;
use std::time::Duration;

const DEFAULT_PORT: u16 = 8100;
const DEFAULT_CORE: &str = "http://127.0.0.1:8000";
const MAX_BODY: usize = 2 * 1024 * 1024; // 2 MiB

/// CLI entry: `0zcode serve [options]`
pub fn run_serve(args: &[String]) -> Result<()> {
    let mut port = env::var("ZEROZ_BACKEND_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let mut core = env::var("ZEROZ_AERIS_URL").unwrap_or_else(|_| DEFAULT_CORE.to_string());

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--port" | "-p" => {
                i += 1;
                port = args
                    .get(i)
                    .ok_or_else(|| Error::Http("--port needs a value".into()))?
                    .parse()
                    .map_err(|_| Error::Http("bad --port".into()))?;
            }
            "--core" | "--aeris" => {
                i += 1;
                core = args
                    .get(i)
                    .ok_or_else(|| Error::Http("--core needs a url".into()))?
                    .clone();
            }
            "--help" | "-h" => {
                print_help();
                return Ok(());
            }
            other => {
                return Err(Error::Http(format!("unknown serve arg: {other}")));
            }
        }
        i += 1;
    }

    let cfg = Config::from_env();
    let core_ep = llm::endpoint(&core).map_err(|e| Error::Llm(e.to_string()))?;

    eprintln!("0zcode-backend {VERSION}");
    eprintln!("  listen : {port}");
    eprintln!("  core   : {} ({}:{})", core, core_ep.host, core_ep.port);
    eprintln!("  link   : Rust edge → SENTINEL core engine");
    eprintln!("  stop   : Ctrl+C");

    let listener = TcpListener::bind(("0.0.0.0", port))
        .map_err(|e| Error::Io(format!("bind :{port}: {e}")))?;
    listener
        .set_nonblocking(false)
        .map_err(|e| Error::Io(e.to_string()))?;

    for conn in listener.incoming() {
        match conn {
            Ok(stream) => {
                let core_host = core_ep.host.clone();
                let core_port = core_ep.port;
                let cfg = cfg.clone();
                let core_url = core.clone();
                thread::spawn(move || {
                    if let Err(e) = handle_client(stream, &core_host, core_port, &core_url, &cfg) {
                        eprintln!("[backend] client error: {e}");
                    }
                });
            }
            Err(e) => eprintln!("[backend] accept: {e}"),
        }
    }
    Ok(())
}

fn print_help() {
    println!(
        "0zcode serve — Rust backend linked to SENTINEL core engine

Usage:
  0zcode serve [--port 8100] [--core http://127.0.0.1:8000]

Env:
  ZEROZ_BACKEND_PORT   listen port (default 8100)
  ZEROZ_AERIS_URL      core engine URL (default http://127.0.0.1:8000)
  ZEROZ_AERIS_KEY      forwarded as X-API-Key to core

Endpoints (this process):
  GET  /                 info
  GET  /healthz          local liveness
  GET  /readyz           probes core /readyz
  GET  /v1/models        proxy → core
  POST /v1/chat          proxy → core
  POST /v1/chat/stream   proxy → core (SSE)
  *    /v1/*             proxy → core (tools, rag, buddy, …)

Point 0zcode TUI at this backend:
  export ZEROZ_AERIS_URL=http://127.0.0.1:8100
"
    );
}

fn handle_client(
    mut stream: TcpStream,
    core_host: &str,
    core_port: u16,
    core_url: &str,
    cfg: &Config,
) -> Result<()> {
    stream
        .set_read_timeout(Some(Duration::from_secs(120)))
        .ok();
    stream
        .set_write_timeout(Some(Duration::from_secs(120)))
        .ok();

    let mut buf = Vec::with_capacity(8192);
    let mut tmp = [0u8; 4096];
    loop {
        let n = stream.read(&mut tmp).map_err(|e| Error::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
        if buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > 64 * 1024 {
            return write_simple(&mut stream, 400, "text/plain", "headers too large");
        }
    }
    if buf.is_empty() {
        return Ok(());
    }

    let text = String::from_utf8_lossy(&buf);
    let (header_part, body_start) = match text.split_once("\r\n\r\n") {
        Some((h, b)) => (h, b.as_bytes().to_vec()),
        None => (text.as_ref(), Vec::new()),
    };

    let mut lines = header_part.lines();
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let path = parts.next().unwrap_or("/");

    let mut content_length: usize = 0;
    let mut content_type = String::new();
    let mut api_key: Option<String> = None;
    for line in lines {
        let lower = line.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("content-length:") {
            content_length = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = lower.strip_prefix("content-type:") {
            content_type = v.trim().to_string();
        } else if let Some(v) = line.split_once(':') {
            if v.0.eq_ignore_ascii_case("x-api-key") {
                api_key = Some(v.1.trim().to_string());
            }
        }
    }

    let mut body = body_start;
    while body.len() < content_length && body.len() < MAX_BODY {
        let n = stream.read(&mut tmp).map_err(|e| Error::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    if body.len() > content_length {
        body.truncate(content_length);
    }

    match (method, path) {
        ("GET", "/") => {
            let body = format!(
                r#"{{"service":"0zcode-backend","version":"{VERSION}","core":"{core_url}","link":"rust→sentinel"}}"#
            );
            return write_simple(&mut stream, 200, "application/json", &body);
        }
        ("GET", "/healthz") | ("GET", "/health") => {
            return write_simple(
                &mut stream,
                200,
                "application/json",
                r#"{"status":"ok","service":"0zcode-backend"}"#,
            );
        }
        ("GET", "/readyz") => {
            return proxy_readyz(&mut stream, core_host, core_port, cfg, api_key.as_deref());
        }
        _ => {}
    }

    if path.starts_with("/v1/")
        || path == "/readyz"
        || path == "/healthz"
        || path == "/metrics"
        || path == "/product"
        || path.starts_with("/ag-ui")
    {
        return proxy(
            &mut stream,
            method,
            path,
            &body,
            &content_type,
            core_host,
            core_port,
            cfg,
            api_key.as_deref(),
        );
    }

    write_simple(
        &mut stream,
        404,
        "application/json",
        r#"{"error":"not found — try /v1/chat or /readyz"}"#,
    )
}

fn proxy_readyz(
    client: &mut TcpStream,
    core_host: &str,
    core_port: u16,
    cfg: &Config,
    api_key: Option<&str>,
) -> Result<()> {
    match upstream_request("GET", "/readyz", b"", "", core_host, core_port, cfg, api_key) {
        Ok((status, _headers, body)) => {
            let code = if status == 200 { 200 } else { 503 };
            let msg = if body.is_empty() {
                if code == 200 {
                    r#"{"status":"ready","core":"up"}"#.to_string()
                } else {
                    r#"{"status":"not_ready","core":"down_or_loading"}"#.to_string()
                }
            } else {
                body
            };
            write_simple(client, code, "application/json", &msg)
        }
        Err(_) => write_simple(
            client,
            503,
            "application/json",
            r#"{"status":"not_ready","core":"unreachable"}"#,
        ),
    }
}

fn proxy(
    client: &mut TcpStream,
    method: &str,
    path: &str,
    body: &[u8],
    content_type: &str,
    core_host: &str,
    core_port: u16,
    cfg: &Config,
    api_key: Option<&str>,
) -> Result<()> {
    let (status, headers, upstream_body) = upstream_request(
        method,
        path,
        body,
        content_type,
        core_host,
        core_port,
        cfg,
        api_key,
    )?;

    let mut ct = "application/json".to_string();
    for h in &headers {
        if h.to_ascii_lowercase().starts_with("content-type:") {
            if let Some((_, v)) = h.split_once(':') {
                ct = v.trim().to_string();
            }
        }
    }

    let is_stream = path.contains("stream") || ct.contains("event-stream");
    if is_stream {
        ct = "text/event-stream".into();
    }

    let status_text = match status {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        503 => "Service Unavailable",
        _ if (200..300).contains(&status) => "OK",
        _ => "Error",
    };

    let mut out = format!(
        "HTTP/1.1 {status} {status_text}\r\n\
Content-Type: {ct}\r\n\
Content-Length: {}\r\n\
Access-Control-Allow-Origin: *\r\n\
Connection: close\r\n\
X-0zcode-Backend: {VERSION}\r\n\
X-0zcode-Core: {core_host}:{core_port}\r\n\r\n",
        upstream_body.len()
    );
    out.push_str(&upstream_body);
    client
        .write_all(out.as_bytes())
        .map_err(|e| Error::Io(e.to_string()))?;
    Ok(())
}

fn upstream_request(
    method: &str,
    path: &str,
    body: &[u8],
    content_type: &str,
    core_host: &str,
    core_port: u16,
    cfg: &Config,
    api_key: Option<&str>,
) -> Result<(u16, Vec<String>, String)> {
    let mut stream = TcpStream::connect((core_host, core_port))
        .map_err(|e| Error::Http(format!("core {core_host}:{core_port} unreachable: {e}")))?;
    stream
        .set_read_timeout(Some(cfg.aeris_timeout.max(Duration::from_secs(30))))
        .ok();
    stream
        .set_write_timeout(Some(cfg.aeris_timeout.max(Duration::from_secs(30))))
        .ok();

    let host_header = format!("{core_host}:{core_port}");
    let ct = if content_type.is_empty() {
        "application/json"
    } else {
        content_type
    };

    let key_header = api_key
        .or(cfg.aeris_key.as_deref())
        .map(|k| format!("X-API-Key: {k}\r\n"))
        .unwrap_or_default();

    let req = format!(
        "{method} {path} HTTP/1.0\r\n\
Host: {host_header}\r\n\
User-Agent: 0zcode-backend/{VERSION}\r\n\
Accept: application/json, text/event-stream, */*\r\n\
Content-Type: {ct}\r\n\
Content-Length: {}\r\n\
{key_header}\
Connection: close\r\n\r\n",
        body.len()
    );

    stream
        .write_all(req.as_bytes())
        .map_err(|e| Error::Http(e.to_string()))?;
    if !body.is_empty() {
        stream
            .write_all(body)
            .map_err(|e| Error::Http(e.to_string()))?;
    }

    let mut buf = Vec::with_capacity(32_768);
    stream
        .read_to_end(&mut buf)
        .map_err(|e| Error::Http(e.to_string()))?;
    let text = String::from_utf8_lossy(&buf);

    let (header_part, body_part) = text.split_once("\r\n\r\n").unwrap_or((text.as_ref(), ""));
    let mut lines = header_part.lines();
    let status_line = lines.next().unwrap_or("HTTP/1.0 502 Bad Gateway");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(502);
    let headers: Vec<String> = lines.map(|s| s.to_string()).collect();
    Ok((status, headers, body_part.to_string()))
}

fn write_simple(stream: &mut TcpStream, code: u16, content_type: &str, body: &str) -> Result<()> {
    let status = match code {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        503 => "Service Unavailable",
        _ => "OK",
    };
    let msg = format!(
        "HTTP/1.1 {code} {status}\r\n\
Content-Type: {content_type}\r\n\
Content-Length: {}\r\n\
Access-Control-Allow-Origin: *\r\n\
Connection: close\r\n\
X-0zcode-Backend: {VERSION}\r\n\r\n\
{body}",
        body.len()
    );
    stream
        .write_all(msg.as_bytes())
        .map_err(|e| Error::Io(e.to_string()))?;
    Ok(())
}
