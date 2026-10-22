//! Minimal HTTP/1.0 client (stdlib only).

use crate::config::Config;
use crate::error::{Error, Result};
use std::io::{Read, Write};
use std::net::TcpStream;

pub fn get_text(host: &str, path: &str, cfg: &Config) -> Result<String> {
    let mut stream = TcpStream::connect((host, 80)).map_err(|e| Error::Http(e.to_string()))?;
    stream.set_read_timeout(Some(cfg.http_timeout)).ok();
    stream.set_write_timeout(Some(cfg.http_timeout)).ok();

    let req = format!(
        "GET {path} HTTP/1.0\r\n\
Host: {host}\r\n\
User-Agent: {}\r\n\
Accept: application/json,text/plain,*/*\r\n\
Connection: close\r\n\r\n",
        cfg.user_agent
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| Error::Http(e.to_string()))?;

    let mut buf = Vec::with_capacity(16_384);
    stream
        .read_to_end(&mut buf)
        .map_err(|e| Error::Http(e.to_string()))?;
    let text = String::from_utf8_lossy(&buf);
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_else(|| text.to_string());

    // Basic status check
    if let Some(status_line) = text.lines().next() {
        if status_line.contains(" 4") || status_line.contains(" 5") {
            return Err(Error::Http(format!(
                "bad status from {host}: {status_line}"
            )));
        }
    }
    Ok(body)
}

pub fn url_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push_str("%20"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// POST a chat message to an AERIS / SENTINEL compatible endpoint.
/// Expects OpenAI-style or simple JSON `{ "message": "..." }` / `{ "prompt": "..." }`.
/// Returns the first text content found in the response body.
pub fn aeris_chat(base_url: &str, message: &str, cfg: &Config) -> Result<String> {
    let url = base_url.trim_end_matches('/');
    // Support both full path and host-only
    let (host, port, path) = parse_url(url)?;
    let body = format!(
        r#"{{"message":{},"prompt":{},"stream":false}}"#,
        json_string(message),
        json_string(message)
    );
    let mut stream =
        TcpStream::connect((host.as_str(), port)).map_err(|e| Error::Http(e.to_string()))?;
    stream.set_read_timeout(Some(cfg.http_timeout)).ok();
    stream.set_write_timeout(Some(cfg.http_timeout)).ok();

    let req = format!(
        "POST {path} HTTP/1.0\r\n\
Host: {host}\r\n\
User-Agent: {}\r\n\
Content-Type: application/json\r\n\
Content-Length: {}\r\n\
Accept: application/json,text/plain,*/*\r\n\
Connection: close\r\n\r\n\
{body}",
        cfg.user_agent,
        body.len(),
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| Error::Http(e.to_string()))?;

    let mut buf = Vec::with_capacity(32_768);
    stream
        .read_to_end(&mut buf)
        .map_err(|e| Error::Http(e.to_string()))?;
    let text = String::from_utf8_lossy(&buf);
    let resp_body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_else(|| text.to_string());

    if let Some(status_line) = text.lines().next() {
        if status_line.contains(" 4") || status_line.contains(" 5") {
            return Err(Error::Http(format!("AERIS bad status: {status_line}")));
        }
    }

    // Best-effort extract of common reply fields
    for key in ["content", "text", "reply", "response", "message", "output"] {
        if let Some(v) = extract_json_string(&resp_body, key) {
            if !v.trim().is_empty() {
                return Ok(v);
            }
        }
    }
    // Fallback: return truncated body if it looks like plain text
    let trimmed = resp_body.trim();
    if !trimmed.is_empty() && !trimmed.starts_with('{') {
        return Ok(trimmed.chars().take(cfg.max_output_chars).collect());
    }
    Err(Error::Http(
        "AERIS returned no extractable text (check endpoint schema)".into(),
    ))
}

fn parse_url(url: &str) -> Result<(String, u16, String)> {
    let u = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .unwrap_or(url);
    let (hostport, path) = match u.split_once('/') {
        Some((h, p)) => (h, format!("/{p}")),
        None => (u, "/chat".to_string()),
    };
    let (host, port) = if let Some((h, p)) = hostport.split_once(':') {
        (
            h.to_string(),
            p.parse::<u16>()
                .map_err(|_| Error::Http(format!("bad port in {url}")))?,
        )
    } else {
        (hostport.to_string(), 80)
    };
    Ok((host, port, path))
}

fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn extract_json_string(body: &str, key: &str) -> Option<String> {
    // Very small extractor: "key": "value" or "key":"value"
    let patterns = [
        format!("\"{key}\":\""),
        format!("\"{key}\": \""),
    ];
    for pat in patterns {
        if let Some(rest) = body.split(&pat).nth(1) {
            let mut out = String::new();
            let mut chars = rest.chars().peekable();
            while let Some(c) = chars.next() {
                match c {
                    '"' => break,
                    '\\' => {
                        if let Some(n) = chars.next() {
                            match n {
                                'n' => out.push('\n'),
                                't' => out.push('\t'),
                                'r' => out.push('\r'),
                                '"' => out.push('"'),
                                '\\' => out.push('\\'),
                                _ => out.push(n),
                            }
                        }
                    }
                    c => out.push(c),
                }
            }
            return Some(out);
        }
    }
    None
}
