//! Minimal HTTP/1.0 client (stdlib only).

use crate::config::Config;
use crate::error::{Error, Result};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

pub fn get_text(host: &str, path: &str, cfg: &Config) -> Result<String> {
    get_text_port(host, 80, path, cfg)
}

/// GET with explicit port (needed for local MCP servers / non-80).
pub fn get_text_port(host: &str, port: u16, path: &str, cfg: &Config) -> Result<String> {
    let mut stream =
        TcpStream::connect((host, port)).map_err(|e| Error::Http(e.to_string()))?;
    stream.set_read_timeout(Some(cfg.http_timeout)).ok();
    stream.set_write_timeout(Some(cfg.http_timeout)).ok();

    let host_header = if port == 80 {
        host.to_string()
    } else {
        format!("{host}:{port}")
    };
    let req = format!(
        "GET {path} HTTP/1.0\r\n\
Host: {host_header}\r\n\
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
    parse_http_response(&String::from_utf8_lossy(&buf), host)
}

pub fn post_json(host: &str, port: u16, path: &str, body: &str, cfg: &Config) -> Result<String> {
    let mut stream =
        TcpStream::connect((host, port)).map_err(|e| Error::Http(e.to_string()))?;
    stream.set_read_timeout(Some(cfg.http_timeout)).ok();
    stream.set_write_timeout(Some(cfg.http_timeout)).ok();

    let host_header = if port == 80 {
        host.to_string()
    } else {
        format!("{host}:{port}")
    };
    let req = format!(
        "POST {path} HTTP/1.0\r\n\
Host: {host_header}\r\n\
User-Agent: {}\r\n\
Content-Type: application/json\r\n\
Accept: application/json\r\n\
Content-Length: {}\r\n\
Connection: close\r\n\r\n\
{body}",
        cfg.user_agent,
        body.len()
    );
    stream
        .write_all(req.as_bytes())
        .map_err(|e| Error::Http(e.to_string()))?;

    let mut buf = Vec::with_capacity(16_384);
    stream
        .read_to_end(&mut buf)
        .map_err(|e| Error::Http(e.to_string()))?;
    parse_http_response(&String::from_utf8_lossy(&buf), host)
}

fn parse_http_response(text: &str, host: &str) -> Result<String> {
    let body = text
        .split_once("\r\n\r\n")
        .map(|(_, b)| b.to_string())
        .unwrap_or_else(|| text.to_string());
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

/// Parse `host` or `host:port` from authority section.
pub fn split_host_port(authority: &str, default_port: u16) -> (String, u16) {
    if let Some((h, p)) = authority.rsplit_once(':') {
        if let Ok(port) = p.parse::<u16>() {
            return (h.to_string(), port);
        }
    }
    (authority.to_string(), default_port)
}

#[allow(dead_code)]
pub fn post_json_https(url: &str, body: &str, timeout: Duration) -> Result<String> {
    use std::process::Command;
    let secs = timeout.as_secs().clamp(1, 60).to_string();
    let out = Command::new("curl")
        .args([
            "-sS",
            "-L",
            "--max-time",
            &secs,
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-H",
            "Accept: application/json",
            "-d",
            body,
            url,
        ])
        .output()
        .map_err(|e| Error::Http(format!("https POST needs curl ({e})")))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Http(format!("curl failed: {err}")));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}
