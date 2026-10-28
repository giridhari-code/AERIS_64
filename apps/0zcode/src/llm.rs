//! AERIS-64 / SENTINEL client — std only (no TLS, no serde).
//!
//! Talks to the Python server in `packages/sentinel/serving/server.py`:
//!
//! | call            | endpoint              | body → reply                                   |
//! |-----------------|-----------------------|------------------------------------------------|
//! | [`chat`]        | `POST /v1/chat`       | `{prompt,session_id,…}` → `{"text": …}`        |
//! | [`stream`]      | `POST /v1/chat/stream`| SSE if present; else auto-fallback to `/v1/chat` |
//! | [`ping`]        | `GET  /v1/models`     | `{"data":[{"id":…,"ready":…}]}`                |
//!
//! Core is **model-only** (tools/RAG/buddy are local to 0zcode, not the Python API).
//! Production servers may require `ZEROZ_AERIS_KEY` → `X-API-Key`.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::http::json_string;
use crate::json_util::string_field;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Server-side limit of `ChatRequest.prompt`.
pub const MAX_PROMPT_CHARS: usize = 4000;
const POLL: Duration = Duration::from_millis(200);
const MAX_BODY: usize = 1 << 20;

fn llm_err<T>(m: impl Into<String>) -> Result<T> {
    Err(Error::Llm(m.into()))
}

// ───────────────────────── addressing ─────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
}

impl Endpoint {
    pub fn base(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

/// Reduce any configured URL (`http://h:8000`, `…/chat`, `…/v1/chat/stream`) to host + port.
pub fn endpoint(url: &str) -> Result<Endpoint> {
    let u = url.trim();
    if u.starts_with("https://") {
        return llm_err("https:// is not supported by this std-only client — use http://127.0.0.1:8000 or a local proxy");
    }
    let u = u.strip_prefix("http://").unwrap_or(u);
    let hostport = u.split(['/', '?', '#']).next().unwrap_or("");
    if hostport.is_empty() {
        return llm_err("empty aeris_url (expected http://127.0.0.1:8000)");
    }
    let (host, port) = match hostport.rsplit_once(':') {
        Some((h, p)) => match p.parse::<u16>() {
            Ok(port) => (h, port),
            Err(_) => return llm_err(format!("bad port in aeris_url: {url}")),
        },
        None => (hostport, 80),
    };
    Ok(Endpoint {
        host: host.to_string(),
        port,
    })
}

/// `0z<hex>` — matches the server's `^[A-Za-z0-9_-]{8,64}$` rule.
pub fn new_session_id() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!(
        "0z{:x}{:x}{:x}",
        t,
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    )
}

// ───────────────────────── HTTP framing ─────────────────────────

/// Incremental `Transfer-Encoding: chunked` decoder (or pass-through).
#[derive(Debug)]
struct Decoder {
    chunked: bool,
    state: St,
    line: Vec<u8>,
    left: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum St {
    Size,
    Data,
    Crlf(u8),
    Done,
}

impl Decoder {
    fn new(chunked: bool) -> Self {
        Self {
            chunked,
            state: St::Size,
            line: Vec::new(),
            left: 0,
        }
    }

    fn finished(&self) -> bool {
        self.chunked && self.state == St::Done
    }

    fn feed(&mut self, input: &[u8], out: &mut Vec<u8>) {
        if !self.chunked {
            out.extend_from_slice(input);
            return;
        }
        let mut i = 0;
        while i < input.len() {
            match self.state {
                St::Size => {
                    let b = input[i];
                    i += 1;
                    if b == b'\n' {
                        let txt = String::from_utf8_lossy(&self.line).to_string();
                        let hex = txt.trim().split(';').next().unwrap_or("").trim();
                        self.left = usize::from_str_radix(hex, 16).unwrap_or(0);
                        self.line.clear();
                        self.state = if self.left == 0 { St::Done } else { St::Data };
                    } else {
                        self.line.push(b);
                    }
                }
                St::Data => {
                    let n = self.left.min(input.len() - i);
                    out.extend_from_slice(&input[i..i + n]);
                    i += n;
                    self.left -= n;
                    if self.left == 0 {
                        self.state = St::Crlf(2);
                    }
                }
                St::Crlf(n) => {
                    i += 1;
                    self.state = if n <= 1 { St::Size } else { St::Crlf(n - 1) };
                }
                St::Done => return,
            }
        }
    }
}

struct Conn {
    sock: TcpStream,
    head: Vec<u8>,
    status: u16,
    dec: Option<Decoder>,
    eof: bool,
    pending: Vec<u8>,
}

fn is_timeout(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut)
}

impl Conn {
    fn open(
        cfg: &Config,
        ep: &Endpoint,
        method: &str,
        path: &str,
        body: Option<&str>,
    ) -> Result<Conn> {
        let addrs = (ep.host.as_str(), ep.port)
            .to_socket_addrs()
            .map_err(|e| Error::Llm(format!("cannot resolve {}: {e}", ep.host)))?;
        let mut sock = None;
        for a in addrs {
            if let Ok(s) = TcpStream::connect_timeout(&a, Duration::from_secs(3)) {
                sock = Some(s);
                break;
            }
        }
        let Some(sock) = sock else {
            return llm_err(format!(
                "not reachable at {} (AERIS offline)",
                ep.base()
            ));
        };
        sock.set_read_timeout(Some(POLL)).ok();
        sock.set_write_timeout(Some(Duration::from_secs(10))).ok();
        sock.set_nodelay(true).ok();

        let key = cfg
            .aeris_key
            .as_deref()
            .map(|k| format!("X-API-Key: {k}\r\n"))
            .unwrap_or_default();
        let (ctype, len) = match body {
            Some(b) => (
                "Content-Type: application/json\r\n".to_string(),
                format!("Content-Length: {}\r\n", b.len()),
            ),
            None => (String::new(), String::new()),
        };
        let req = format!(
            "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nUser-Agent: {}\r\nAccept: */*\r\n{key}{ctype}{len}Connection: close\r\n\r\n{}",
            ep.host,
            ep.port,
            cfg.user_agent,
            body.unwrap_or("")
        );
        let mut c = Conn {
            sock,
            head: Vec::new(),
            status: 0,
            dec: None,
            eof: false,
            pending: Vec::new(),
        };
        c.sock
            .write_all(req.as_bytes())
            .map_err(|e| Error::Llm(format!("send failed: {e}")))?;
        Ok(c)
    }

    /// One socket read. `Ok(false)` = peer closed. Honours `cancel` and `idle`;
    /// a poll timeout is not an error until `since` is older than `idle`.
    fn pump(&mut self, cancel: &AtomicBool, idle: Duration, since: &mut Instant) -> Result<bool> {
        let mut buf = [0u8; 4096];
        loop {
            if cancel.load(Ordering::Relaxed) {
                return llm_err("interrupted");
            }
            match self.sock.read(&mut buf) {
                Ok(0) => {
                    self.eof = true;
                    return Ok(false);
                }
                Ok(n) => {
                    *since = Instant::now();
                    self.absorb(&buf[..n])?;
                    return Ok(true);
                }
                Err(e) if is_timeout(&e) => {
                    if since.elapsed() > idle {
                        return llm_err(format!(
                            "no data for {}s — model busy? raise aeris_timeout_secs",
                            idle.as_secs()
                        ));
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => {}
                Err(e) => return llm_err(format!("read failed: {e}")),
            }
        }
    }

    /// Read until the status line and headers are parsed.
    fn read_status(&mut self, cancel: &AtomicBool, idle: Duration) -> Result<u16> {
        let mut since = Instant::now();
        while self.dec.is_none() {
            if !self.pump(cancel, idle, &mut since)? {
                return llm_err("connection closed before a response arrived");
            }
        }
        Ok(self.status)
    }

    /// Next slice of decoded body bytes; `Ok(None)` at end of body.
    fn read_body(&mut self, cancel: &AtomicBool, idle: Duration) -> Result<Option<Vec<u8>>> {
        let mut since = Instant::now();
        loop {
            if !self.pending.is_empty() {
                return Ok(Some(std::mem::take(&mut self.pending)));
            }
            if self.eof || self.dec.as_ref().is_some_and(Decoder::finished) {
                return Ok(None);
            }
            self.pump(cancel, idle, &mut since)?;
        }
    }

    /// Feed raw socket bytes: parse headers first, then decode the body.
    fn absorb(&mut self, bytes: &[u8]) -> Result<()> {
        if let Some(dec) = self.dec.as_mut() {
            dec.feed(bytes, &mut self.pending);
            return Ok(());
        }
        self.head.extend_from_slice(bytes);
        let Some(pos) = find(&self.head, b"\r\n\r\n") else {
            if self.head.len() > 64 * 1024 {
                return llm_err("oversized response header");
            }
            return Ok(());
        };
        let head = String::from_utf8_lossy(&self.head[..pos]).to_string();
        let rest = self.head[pos + 4..].to_vec();
        let mut lines = head.lines();
        let status_line = lines.next().unwrap_or("");
        self.status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or_else(|| Error::Llm(format!("bad status line: {status_line}")))?;
        let chunked = lines.any(|l| {
            let l = l.to_ascii_lowercase();
            l.starts_with("transfer-encoding:") && l.contains("chunked")
        });
        let mut dec = Decoder::new(chunked);
        dec.feed(&rest, &mut self.pending);
        self.dec = Some(dec);
        self.head.clear();
        Ok(())
    }

    fn read_all(&mut self, cancel: &AtomicBool, idle: Duration) -> Result<String> {
        let mut out = Vec::new();
        while let Some(b) = self.read_body(cancel, idle)? {
            out.extend_from_slice(&b);
            if out.len() > MAX_BODY {
                break;
            }
        }
        Ok(String::from_utf8_lossy(&out).to_string())
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Friendly message for a non-2xx answer from the server.
fn status_error(status: u16, body: &str, base: &str) -> Error {
    let detail = string_field(body, "detail").unwrap_or_default();
    let msg = match status {
        401 | 403 => "unauthorized — the server wants a key: set ZEROZ_AERIS_KEY (X-API-Key) or aeris_key in config.toml".to_string(),
        404 => format!("{base} answered 404 — is this an AERIS-64 server? (expects /v1/chat)"),
        422 => "request rejected by the server (bad prompt / parameters)".to_string(),
        429 => "rate limited by the server — wait a minute".to_string(),
        503 if detail.is_empty() => "model not loaded — set NEUROFIELD_CHECKPOINT and restart the server".to_string(),
        s if detail.is_empty() => format!("HTTP {s}"),
        s => format!("HTTP {s}: {detail}"),
    };
    if (status == 503 || status == 422) && !detail.is_empty() {
        return Error::Llm(format!("{msg} — {detail}"));
    }
    Error::Llm(msg)
}

fn chat_body(cfg: &Config, session: &str, prompt: &str, stream: bool) -> String {
    let p: String = prompt.trim().chars().take(MAX_PROMPT_CHARS).collect();
    format!(
        r#"{{"prompt":{},"session_id":{},"max_new_tokens":{},"temperature":0.7,"top_p":0.9,"stream":{}}}"#,
        json_string(&p),
        json_string(session),
        cfg.aeris_max_tokens,
        stream
    )
}

fn link(cfg: &Config) -> Result<Endpoint> {
    match cfg.aeris_url.as_deref() {
        Some(u) => endpoint(u),
        None => llm_err("not linked — set ZEROZ_AERIS_URL=http://127.0.0.1:8000"),
    }
}

// ───────────────────────── public calls ─────────────────────────

/// `GET /v1/models` → model id. Errors if the server is up but has no model.
pub fn ping(cfg: &Config) -> Result<String> {
    let ep = link(cfg)?;
    let never = AtomicBool::new(false);
    let idle = Duration::from_secs(3);
    let mut c = Conn::open(cfg, &ep, "GET", "/v1/models", None)?;
    let status = c.read_status(&never, idle)?;
    let body = c.read_all(&never, idle)?;
    if !(200..300).contains(&status) {
        return Err(status_error(status, &body, &ep.base()));
    }
    if body.replace(' ', "").contains("\"ready\":false") {
        return llm_err("server is up but the model is not loaded (set NEUROFIELD_CHECKPOINT)");
    }
    Ok(string_field(&body, "id").unwrap_or_else(|| "model".into()))
}

/// Human-readable link diagnosis for `/llm` and boot messages.
pub fn diagnose(cfg: &Config) -> String {
    let url = match cfg.aeris_url.as_deref() {
        None => {
            return "LLM not linked.\n  fix: export ZEROZ_AERIS_URL=http://127.0.0.1:8000\n       or aeris_url in ~/.config/0zcode/config.toml".into();
        }
        Some(u) => u,
    };
    if cfg.offline {
        return format!(
            "LLM linked to {url} but ZEROZ_OFFLINE=1 — chat disabled.\n  fix: unset ZEROZ_OFFLINE"
        );
    }
    match crate::aeris_boot::probe_status(url) {
        crate::aeris_boot::AerisStatus::Ready => {
            let model = ping(cfg).unwrap_or_else(|_| "model".into());
            format!(
                "LLM online\n  url: {url}\n  model: {model}\n  stream: {}\n  max_tokens: {}\n  timeout: {}s",
                cfg.aeris_stream,
                cfg.aeris_max_tokens,
                cfg.aeris_timeout.as_secs()
            )
        }
        crate::aeris_boot::AerisStatus::ServerOnly => format!(
            "LLM server up at {url} but model NOT loaded\n  fix: export NEUROFIELD_CHECKPOINT=/path/to/folder/with/model.safetensors\n       then restart AERIS (scripts/zeroz/run_aeris.sh) or 0zcode (auto-start)"
        ),
        crate::aeris_boot::AerisStatus::Down => format!(
            "LLM not reachable at {url}\n  fix:\n    1) export ZEROZ_AERIS_URL=http://127.0.0.1:8000\n    2) start server:  cd AERIS_64 && PYTHONPATH=packages uvicorn sentinel.api.app:create_app --factory --host 127.0.0.1 --port 8000\n       or:  bash scripts/zeroz/run_aeris.sh\n    3) set NEUROFIELD_CHECKPOINT to your trained model folder\n    4) 0zcode /llm   # re-check\n  note: only http:// supported (not https) in this client"
        ),
    }
}

/// One-shot reply via `POST /v1/chat` (model-only SENTINEL).
pub fn chat(cfg: &Config, session: &str, prompt: &str, cancel: &AtomicBool) -> Result<String> {
    let ep = link(cfg)?;
    let body = chat_body(cfg, session, prompt, false);
    let mut c = Conn::open(cfg, &ep, "POST", "/v1/chat", Some(&body))?;
    let status = c.read_status(cancel, cfg.aeris_timeout)?;
    let resp = c.read_all(cancel, cfg.aeris_timeout)?;
    if !(200..300).contains(&status) {
        return Err(status_error(status, &resp, &ep.base()));
    }
    match string_field(&resp, "text") {
        Some(t) => Ok(t.trim().to_string()),
        None => Err(Error::Parse("AERIS reply has no \"text\" field".into())),
    }
}

/// Live reply via `POST /v1/chat/stream` (SSE). `on_token` gets each decoded
/// piece as it is generated; the return value is the server's final text.
pub fn stream(
    cfg: &Config,
    session: &str,
    prompt: &str,
    cancel: &AtomicBool,
    on_token: &mut dyn FnMut(&str),
) -> Result<String> {
    let ep = link(cfg)?;
    let body = chat_body(cfg, session, prompt, true);
    let mut c = Conn::open(cfg, &ep, "POST", "/v1/chat/stream", Some(&body))?;
    let status = c.read_status(cancel, cfg.aeris_timeout)?;
    // Model-only SENTINEL has no SSE route — fall back to POST /v1/chat.
    if status == 404 || status == 405 || status == 410 || status == 501 {
        let _ = c.read_all(cancel, cfg.aeris_timeout);
        let text = chat(cfg, session, prompt, cancel)?;
        if !text.is_empty() {
            on_token(&text);
        }
        return Ok(text);
    }
    if !(200..300).contains(&status) {
        let resp = c.read_all(cancel, cfg.aeris_timeout)?;
        return Err(status_error(status, &resp, &ep.base()));
    }
    let mut acc = String::new();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(bytes) = c.read_body(cancel, cfg.aeris_timeout)? {
        buf.extend_from_slice(&bytes);
        while let Some(nl) = buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = buf.drain(..=nl).collect();
            let line = String::from_utf8_lossy(&line);
            let Some(data) = line.trim().strip_prefix("data:") else {
                continue;
            };
            match sse_event(data.trim()) {
                Sse::Token(t) => {
                    if !t.is_empty() {
                        acc.push_str(&t);
                        on_token(&t);
                    }
                }
                Sse::Done(text) => return Ok(text.unwrap_or(acc).trim().to_string()),
                Sse::Error(d) => return llm_err(d),
                Sse::End => return finish_stream(acc),
                Sse::Other => {}
            }
        }
    }
    finish_stream(acc)
}

fn finish_stream(acc: String) -> Result<String> {
    if acc.trim().is_empty() {
        llm_err("stream ended without any output")
    } else {
        Ok(acc.trim().to_string())
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Sse {
    Token(String),
    Done(Option<String>),
    Error(String),
    End,
    Other,
}

fn sse_event(data: &str) -> Sse {
    if data == "[DONE]" {
        return Sse::End;
    }
    // token events are `{"token": "...", "id": n}`; everything else has `"event"`
    if data.starts_with("{\"token\"") {
        return Sse::Token(string_field(data, "token").unwrap_or_default());
    }
    match string_field(data, "event").as_deref() {
        Some("done") => Sse::Done(string_field(data, "text")),
        Some("error") => Sse::Error(string_field(data, "detail").unwrap_or_else(|| "server error".into())),
        _ => Sse::Other,
    }
}

// ───────────────────────── tests ─────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_strips_paths_and_defaults() {
        let e = endpoint("http://127.0.0.1:8000/chat").unwrap();
        assert_eq!((e.host.as_str(), e.port), ("127.0.0.1", 8000));
        assert_eq!(endpoint("http://h:9/v1/chat/stream?x=1").unwrap().port, 9);
        assert_eq!(endpoint("localhost").unwrap().port, 80);
        assert_eq!(endpoint("localhost:8000").unwrap().base(), "http://localhost:8000");
        assert!(endpoint("https://x").is_err());
        assert!(endpoint("http://h:abc").is_err());
        assert!(endpoint("").is_err());
    }

    #[test]
    fn session_ids_satisfy_server_rule_and_differ() {
        let a = new_session_id();
        let b = new_session_id();
        assert_ne!(a, b);
        assert!((8..=64).contains(&a.len()), "{a}");
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-'));
    }

    #[test]
    fn chunked_decoder_handles_arbitrary_splits() {
        let wire = b"5\r\nhello\r\n6;ext=1\r\n world\r\n0\r\n\r\n";
        for step in 1..wire.len() {
            let mut d = Decoder::new(true);
            let mut out = Vec::new();
            for part in wire.chunks(step) {
                d.feed(part, &mut out);
            }
            assert_eq!(out, b"hello world", "step {step}");
            assert!(d.finished(), "step {step}");
        }
        let mut raw = Decoder::new(false);
        let mut out = Vec::new();
        raw.feed(b"abc", &mut out);
        assert_eq!(out, b"abc");
        assert!(!raw.finished());
    }

    #[test]
    fn sse_events_parse() {
        assert_eq!(sse_event(r#"{"token": "He", "id": 7}"#), Sse::Token("He".into()));
        assert_eq!(sse_event(r#"{"token": "\n", "id": 7}"#), Sse::Token("\n".into()));
        assert_eq!(
            sse_event(r#"{"event": "done", "latency_ms": 3.2, "text": "Hello"}"#),
            Sse::Done(Some("Hello".into()))
        );
        assert_eq!(
            sse_event(r#"{"event": "error", "detail": "internal error"}"#),
            Sse::Error("internal error".into())
        );
        assert_eq!(sse_event(r#"{"session_id": "x", "event": "start"}"#), Sse::Other);
        assert_eq!(sse_event("[DONE]"), Sse::End);
    }

    #[test]
    fn status_errors_are_actionable() {
        let m = |s, b| status_error(s, b, "http://h:1").to_string();
        assert!(m(401, "{}").contains("ZEROZ_AERIS_KEY"));
        assert!(m(404, "{}").contains("/v1/chat"));
        assert!(m(503, "{}").contains("NEUROFIELD_CHECKPOINT"));
        assert!(m(500, r#"{"detail": "boom"}"#).contains("boom"));
    }

    #[test]
    fn request_body_is_valid_and_clamped() {
        let cfg = Config::default();
        let b = chat_body(&cfg, "0zabc12345", "say \"hi\"\nनमस्ते", true);
        assert!(b.contains(r#""prompt":"say \"hi\"\nनमस्ते""#), "{b}");
        assert!(b.contains(r#""session_id":"0zabc12345""#));
        assert!(b.contains(r#""max_new_tokens":256"#) && b.ends_with(r#""stream":true}"#));
        let long = "x".repeat(9000);
        let b = chat_body(&cfg, "s", &long, false);
        assert!(b.len() < MAX_PROMPT_CHARS + 200);
    }
}
