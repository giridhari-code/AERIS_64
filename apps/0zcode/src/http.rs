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

/// JSON string literal (quotes + escapes) for hand-built request bodies.
pub(crate) fn json_string(s: &str) -> String {
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
