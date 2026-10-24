//! User-configured MCP tools — **MCP v2** (JSON-RPC 2.0) + legacy GET.
//!
//! Config: `~/.config/0zcode/mcp_tools.conf` or `ZEROZ_MCP_CONFIG`
//!
//! ```text
//! name | triggers | https://host/path              # legacy GET ?q=
//! name | triggers | http://127.0.0.1:PORT/mcp | mcp-v2
//! name | triggers | http://host/mcp | mcp-v2 | remote_tool_name
//! ```
//!
//! MCP v2 wire: POST JSON-RPC `tools/list` · `tools/call`
//! Policy: HTTP(S) only. Approval gate on invoke (`ZEROZ_APPROVAL=auto` in CI).

use crate::config::Config;
use crate::error::{Error, Result};
use crate::http::{
    get_text_port, json_string, post_json, post_json_https, split_host_port, url_encode,
};
use crate::json_util::string_field;
use crate::tools::ToolOutput;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum McpProtocol {
    LegacyGet,
    McpV2,
}

impl McpProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            McpProtocol::LegacyGet => "legacy",
            McpProtocol::McpV2 => "mcp-v2",
        }
    }

    fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "mcp-v2" | "mcpv2" | "v2" | "jsonrpc" | "json-rpc" => McpProtocol::McpV2,
            _ => McpProtocol::LegacyGet,
        }
    }
}

#[derive(Debug, Clone)]
pub struct McpTool {
    pub name: String,
    pub triggers: Vec<String>,
    pub url: String,
    pub protocol: McpProtocol,
    pub remote_name: String,
}

fn conf_path() -> PathBuf {
    if let Ok(p) = env::var("ZEROZ_MCP_CONFIG") {
        return PathBuf::from(p);
    }
    let home = env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".config/0zcode/mcp_tools.conf")
}

pub fn conf_path_display() -> String {
    conf_path().display().to_string()
}

pub fn load() -> Result<Vec<McpTool>> {
    let path = conf_path();
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let text = fs::read_to_string(&path)?;
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(t) = parse_line(line) {
            out.push(t);
        }
    }
    Ok(out)
}

fn parse_line(line: &str) -> Option<McpTool> {
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    if parts.len() < 3 {
        return None;
    }
    let name = parts[0].to_string();
    if name.is_empty() || !valid_name(&name) {
        return None;
    }
    let triggers: Vec<String> = parts[1]
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    let url = parts[2].trim().to_string();
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return None;
    }
    if triggers.is_empty() {
        return None;
    }
    let mut protocol = McpProtocol::LegacyGet;
    let mut remote_name = name.clone();
    if parts.len() >= 4 {
        protocol = McpProtocol::parse(parts[3]);
    }
    if parts.len() >= 5 && !parts[4].is_empty() {
        remote_name = parts[4].to_string();
    }
    Some(McpTool {
        name,
        triggers,
        url,
        protocol,
        remote_name,
    })
}

fn valid_name(n: &str) -> bool {
    !n.is_empty()
        && n.len() <= 40
        && n
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn save_all(tools: &[McpTool]) -> Result<()> {
    let path = conf_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut body = String::from(
        "# 0zcode MCP tools\n# name | triggers | url [| mcp-v2 [| remote_name]]\n",
    );
    for t in tools {
        match t.protocol {
            McpProtocol::McpV2 if t.remote_name != t.name => {
                body.push_str(&format!(
                    "{} | {} | {} | mcp-v2 | {}\n",
                    t.name,
                    t.triggers.join(","),
                    t.url,
                    t.remote_name
                ));
            }
            McpProtocol::McpV2 => {
                body.push_str(&format!(
                    "{} | {} | {} | mcp-v2\n",
                    t.name,
                    t.triggers.join(","),
                    t.url
                ));
            }
            McpProtocol::LegacyGet => {
                body.push_str(&format!(
                    "{} | {} | {}\n",
                    t.name,
                    t.triggers.join(","),
                    t.url
                ));
            }
        }
    }
    fs::write(&path, body)?;
    Ok(())
}

pub fn list_text() -> String {
    match load() {
        Ok(tools) if tools.is_empty() => format!(
            "No registered MCP endpoints.\nConfig: {}\n\n\
Register (MCP JSON-RPC v2):\n  /mcp add <id> <keywords> <https://host/path> mcp-v2 [<remote_tool>]\n\
Register (legacy HTTP GET):\n  /mcp add <id> <keywords> <https://host/path>",
            conf_path_display()
        ),
        Ok(tools) => {
            let mut out = format!("User MCP tools ({})\nFile: {}\n\n", tools.len(), conf_path_display());
            for t in &tools {
                out.push_str(&format!(
                    "• {} [{}] triggers=[{}] {}\n",
                    t.name,
                    t.protocol.as_str(),
                    t.triggers.join(", "),
                    t.url
                ));
            }
            out.push_str("\n/mcp add|remove|test|discover|list");
            out
        }
        Err(e) => format!("Could not load MCP config: {e}"),
    }
}

pub fn add(name: &str, triggers: &str, url: &str) -> Result<String> {
    add_with_protocol(name, triggers, url, McpProtocol::LegacyGet, None)
}

pub fn add_with_protocol(
    name: &str,
    triggers: &str,
    url: &str,
    protocol: McpProtocol,
    remote_name: Option<&str>,
) -> Result<String> {
    let name = name.trim();
    if !valid_name(name) {
        return Err(Error::Tool("name: letters/digits/_/- only, max 40".into()));
    }
    let url = url.trim();
    if !url.starts_with("http://") && !url.starts_with("https://") {
        return Err(Error::Tool("url must start with http:// or https://".into()));
    }
    let triggers: Vec<String> = triggers
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    if triggers.is_empty() {
        return Err(Error::Tool("need at least one trigger keyword".into()));
    }
    let remote = remote_name
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(name)
        .to_string();
    let mut tools = load()?;
    tools.retain(|t| t.name != name);
    tools.push(McpTool {
        name: name.to_string(),
        triggers,
        url: url.to_string(),
        protocol,
        remote_name: remote,
    });
    save_all(&tools)?;
    Ok(format!(
        "Saved `{name}` [{}] → {}",
        protocol.as_str(),
        conf_path_display()
    ))
}

pub fn remove(name: &str) -> Result<String> {
    let name = name.trim();
    let mut tools = load()?;
    let before = tools.len();
    tools.retain(|t| t.name != name);
    if tools.len() == before {
        return Err(Error::Tool(format!("no tool named `{name}`")));
    }
    save_all(&tools)?;
    Ok(format!("Removed `{name}`"))
}

pub fn match_tools(user_text: &str) -> Vec<McpTool> {
    let lower = user_text.to_lowercase();
    load()
        .unwrap_or_default()
        .into_iter()
        .filter(|t| t.triggers.iter().any(|tr| lower.contains(tr)))
        .collect()
}

pub fn invoke(tool: &McpTool, query: &str, cfg: &Config) -> Result<ToolOutput> {
    crate::approval::require(
        crate::approval::ActionClass::Mcp,
        &format!("MCP tool `{}` → {}", tool.name, tool.url),
    )
    .map_err(Error::Tool)?;
    match tool.protocol {
        McpProtocol::LegacyGet => invoke_legacy(tool, query, cfg),
        McpProtocol::McpV2 => invoke_v2_call(tool, query, cfg),
    }
}

fn invoke_legacy(tool: &McpTool, query: &str, cfg: &Config) -> Result<ToolOutput> {
    let sep = if tool.url.contains('?') { "&" } else { "?" };
    let full = format!(
        "{}{}q={}&text={}",
        tool.url,
        sep,
        url_encode(query),
        url_encode(query)
    );
    let body = fetch_get(&full, cfg)?;
    let summary = extract_text(&body, cfg);
    Ok(ToolOutput {
        name: "mcp",
        body: format!(
            "│ tool: {} [legacy]\n│ {}\n│ source: {}",
            tool.name, summary, tool.url
        ),
    })
}

fn invoke_v2_call(tool: &McpTool, query: &str, cfg: &Config) -> Result<ToolOutput> {
    let rpc = format!(
        r#"{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":{},"arguments":{{"query":{},"text":{}}}}}}}"#,
        json_string(&tool.remote_name),
        json_string(query),
        json_string(query)
    );
    let body = fetch_post(&tool.url, &rpc, cfg)?;
    if let Some(err) = rpc_error(&body) {
        return Err(Error::Tool(format!("MCP v2 error: {err}")));
    }
    let summary = extract_mcp_result(&body).unwrap_or_else(|| extract_text(&body, cfg));
    Ok(ToolOutput {
        name: "mcp",
        body: format!(
            "│ tool: {} [mcp-v2 → {}]\n│ {}\n│ source: {}",
            tool.name, tool.remote_name, summary, tool.url
        ),
    })
}

/// Real `tools/list` against an mcp-v2 endpoint.
pub fn discover(tool_name: &str, cfg: &Config) -> String {
    let tools = match load() {
        Ok(t) => t,
        Err(e) => return e.to_string(),
    };
    let Some(tool) = tools.into_iter().find(|t| t.name == tool_name) else {
        return format!("no tool `{tool_name}` — /mcp list");
    };
    if tool.protocol != McpProtocol::McpV2 {
        return format!("`{}` is not mcp-v2 — re-add with mcp-v2 flag", tool.name);
    }
    let rpc = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#;
    match fetch_post(&tool.url, rpc, cfg) {
        Ok(body) => {
            if let Some(err) = rpc_error(&body) {
                return format!("tools/list error: {err}");
            }
            format!("MCP v2 tools/list (`{}`)\n{}", tool.name, body.chars().take(2000).collect::<String>())
        }
        Err(e) => format!("discover failed: {e}"),
    }
}

fn extract_text(body: &str, cfg: &Config) -> String {
    string_field(body, "summary")
        .or_else(|| string_field(body, "result"))
        .or_else(|| string_field(body, "text"))
        .or_else(|| string_field(body, "content"))
        .unwrap_or_else(|| body.chars().take(cfg.max_output_chars.min(2000)).collect())
}

fn extract_mcp_result(body: &str) -> Option<String> {
    string_field(body, "text")
        .or_else(|| string_field(body, "content"))
        .or_else(|| {
            body.find("\"result\"").and_then(|idx| {
                let slice = &body[idx..];
                string_field(slice, "text").or_else(|| string_field(slice, "content"))
            })
        })
}

fn rpc_error(body: &str) -> Option<String> {
    if !body.contains("\"error\"") {
        return None;
    }
    string_field(body, "message").or_else(|| Some(body.chars().take(300).collect()))
}

fn fetch_get(url: &str, cfg: &Config) -> Result<String> {
    if let Some(rest) = url.strip_prefix("http://") {
        let (auth, path) = match rest.split_once('/') {
            Some((h, p)) => (h, format!("/{p}")),
            None => (rest, "/".into()),
        };
        let (host, port) = split_host_port(auth, 80);
        return get_text_port(&host, port, &path, cfg);
    }
    if url.starts_with("https://") {
        return fetch_https_curl_get(url, cfg.http_timeout);
    }
    Err(Error::Tool("only http:// or https://".into()))
}

fn fetch_post(url: &str, json_body: &str, cfg: &Config) -> Result<String> {
    if let Some(rest) = url.strip_prefix("http://") {
        let (auth, path) = match rest.split_once('/') {
            Some((h, p)) => (h, format!("/{p}")),
            None => (rest, "/".into()),
        };
        let (host, port) = split_host_port(auth, 80);
        return post_json(&host, port, &path, json_body, cfg);
    }
    if url.starts_with("https://") {
        return post_json_https(url, json_body, cfg.http_timeout);
    }
    Err(Error::Tool("only http:// or https://".into()))
}

fn fetch_https_curl_get(url: &str, timeout: Duration) -> Result<String> {
    let secs = timeout.as_secs().clamp(1, 60).to_string();
    let out = Command::new("curl")
        .args([
            "-sS",
            "-L",
            "--max-time",
            &secs,
            "-H",
            "Accept: application/json,text/plain,*/*",
            url,
        ])
        .output()
        .map_err(|e| Error::Http(format!("https needs curl ({e})")))?;
    if !out.status.success() {
        return Err(Error::Http(format!(
            "curl failed: {}",
            String::from_utf8_lossy(&out.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&out.stdout).to_string())
}

pub fn handle_command(line: &str) -> Option<String> {
    let line = line.trim();
    let lower = line.to_lowercase();
    if lower == "/mcp" || lower == "/mcp list" || lower == "mcp list" {
        return Some(list_text());
    }
    if lower.starts_with("/mcp discover ") {
        let name = line["/mcp discover ".len()..].trim();
        return Some(discover(name, &Config::from_env()));
    }
    if lower.starts_with("/mcp add ") || line.starts_with("/mcp add ") {
        return Some(parse_add(line.strip_prefix("/mcp add ").unwrap_or("")));
    }
    if let Some(rest) = lower
        .strip_prefix("/mcp remove ")
        .or_else(|| lower.strip_prefix("/mcp rm "))
    {
        return Some(match remove(rest.trim()) {
            Ok(s) => s,
            Err(e) => e.to_string(),
        });
    }
    if lower.starts_with("/mcp test ") {
        let rest = line["/mcp test ".len()..].trim();
        let mut parts = rest.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("");
        let q = parts.next().unwrap_or("test");
        return Some(test_tool(name, q));
    }
    if lower == "/mcp" || lower.starts_with("/mcp ") {
        return Some(
            "MCP (v2)\n\
  /mcp list\n\
  /mcp add <name> <triggers> <url> [mcp-v2] [remote_name]\n\
  /mcp remove|test|discover <name>\n\
File: "
                .to_string()
                + &conf_path_display(),
        );
    }
    None
}

fn parse_add(raw: &str) -> String {
    let tokens: Vec<&str> = raw.split_whitespace().collect();
    if tokens.len() < 3 {
        return "usage: /mcp add <name> <triggers> <url> [mcp-v2] [remote_name]".into();
    }
    let name = tokens[0];
    let url_i = tokens
        .iter()
        .position(|t| t.starts_with("http://") || t.starts_with("https://"));
    let Some(ui) = url_i else {
        return "url must be http:// or https://".into();
    };
    if ui < 2 {
        return "usage: /mcp add <name> <triggers> <url> [mcp-v2]".into();
    }
    let triggers = tokens[1..ui].join(" ");
    let url = tokens[ui];
    let mut protocol = McpProtocol::LegacyGet;
    let mut remote: Option<&str> = None;
    if let Some(p) = tokens.get(ui + 1) {
        if McpProtocol::parse(p) == McpProtocol::McpV2 {
            protocol = McpProtocol::McpV2;
            remote = tokens.get(ui + 2).copied();
        }
    }
    match add_with_protocol(name, &triggers, url, protocol, remote) {
        Ok(s) => s,
        Err(e) => e.to_string(),
    }
}

fn test_tool(name: &str, query: &str) -> String {
    let cfg = Config::from_env();
    let tools = match load() {
        Ok(t) => t,
        Err(e) => return e.to_string(),
    };
    let Some(tool) = tools.into_iter().find(|t| t.name == name) else {
        return format!("no tool `{name}`");
    };
    match invoke(&tool, query, &cfg) {
        Ok(o) => o.render(),
        Err(e) => format!("test failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::Duration;

    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn parse_v2_line() {
        let t = parse_line("mail | email | http://127.0.0.1:9/mcp | mcp-v2 | get_inbox").unwrap();
        assert_eq!(t.protocol, McpProtocol::McpV2);
        assert_eq!(t.remote_name, "get_inbox");
    }

    /// Integration: loopback MCP JSON-RPC server exercising tools/list + tools/call.
    #[test]
    fn mcp_v2_integration_tools_call() {
        let _g = LOCK.lock().unwrap();
        env::set_var("ZEROZ_APPROVAL", "auto");

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().unwrap().port();
        let stop = Arc::new(Mutex::new(false));
        let stop2 = stop.clone();

        let server = thread::spawn(move || {
            listener.set_nonblocking(false).ok();
            // handle a few requests then exit
            for _ in 0..4 {
                if *stop2.lock().unwrap() {
                    break;
                }
                let Ok((mut stream, _)) = listener.accept() else {
                    continue;
                };
                stream.set_read_timeout(Some(Duration::from_secs(2))).ok();
                let mut buf = [0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]);
                let body = req.split("\r\n\r\n").nth(1).unwrap_or("");
                let resp_body = if body.contains("tools/list") {
                    r#"{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"status_report","description":"Operational status summary"}]}}"#.to_string()
                } else if body.contains("tools/call") {
                    let q = body
                        .find("\"query\"")
                        .and_then(|i| {
                            let s = &body[i..];
                            let start = s.find(':')?;
                            let s = s[start + 1..].trim();
                            let start = s.find('"')? + 1;
                            let end = s[start..].find('"')? + start;
                            Some(&s[start..end])
                        })
                        .unwrap_or("");
                    // Production-shaped MCP tools/call result (text content block)
                    format!(
                        r#"{{"jsonrpc":"2.0","id":1,"result":{{"content":[{{"type":"text","text":"status=ok query={q}"}}],"isError":false}}}}"#
                    )
                } else {
                    r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32601,"message":"Method not found"}}"#.to_string()
                };
                let resp = format!(
                    "HTTP/1.0 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    resp_body.len(),
                    resp_body
                );
                let _ = stream.write_all(resp.as_bytes());
            }
        });

        thread::sleep(Duration::from_millis(50));

        let dir = env::temp_dir().join(format!("0zcode_mcp_v2_it_{port}"));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let conf = dir.join("mcp_tools.conf");
        env::set_var("ZEROZ_MCP_CONFIG", &conf);

        let url = format!("http://127.0.0.1:{port}/mcp");
        add_with_protocol(
            "ops_status",
            "status,ops,health",
            &url,
            McpProtocol::McpV2,
            Some("status_report"),
        )
        .unwrap();

        let tools = load().unwrap();
        assert_eq!(tools[0].protocol, McpProtocol::McpV2);
        assert_eq!(tools[0].remote_name, "status_report");

        let cfg = Config::from_env();
        let out = invoke(&tools[0], "region=ap-south-1", &cfg).expect("tools/call");
        let rendered = out.render();
        assert!(
            rendered.contains("status=ok") && rendered.contains("region=ap-south-1"),
            "unexpected tools/call body: {rendered}"
        );

        let listed = discover("ops_status", &cfg);
        assert!(
            listed.contains("status_report"),
            "tools/list missing remote tool: {listed}"
        );

        *stop.lock().unwrap() = true;
        let _ = TcpStream::connect(("127.0.0.1", port));
        let _ = server.join();

        remove("ops_status").ok();
        env::remove_var("ZEROZ_MCP_CONFIG");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn legacy_roundtrip_config() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join("0zcode_mcp_legacy");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        env::set_var("ZEROZ_MCP_CONFIG", dir.join("c.conf"));
        add("email", "email,inbox", "https://example.com/mail").unwrap();
        assert_eq!(load().unwrap()[0].protocol, McpProtocol::LegacyGet);
        remove("email").unwrap();
        env::remove_var("ZEROZ_MCP_CONFIG");
    }
}
