//! User-configured MCP-style HTTP tools.
//!
//! File: `~/.config/0zcode/mcp_tools.conf`
//! Line format (one tool per line):
//!   name | trigger1,trigger2 | https://endpoint.example/path
//!
//! REPL:
//!   /mcp list
//!   /mcp add <name> <trigger,trigger> <url>
//!   /mcp remove <name>
//!   /mcp test <name> [query]
//!
//! Policy: only HTTP(S) GET to configured URLs. No shell, no code gen.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::http::{get_text, url_encode};
use crate::json_util::string_field;
use crate::tools::ToolOutput;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct McpTool {
    pub name: String,
    pub triggers: Vec<String>,
    pub url: String,
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
    Some(McpTool {
        name,
        triggers,
        url,
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
        "# 0zcode user MCP tools\n# name | trigger1,trigger2 | https://endpoint\n# Created/updated by /mcp add\n",
    );
    for t in tools {
        body.push_str(&format!(
            "{} | {} | {}\n",
            t.name,
            t.triggers.join(","),
            t.url
        ));
    }
    fs::write(&path, body)?;
    Ok(())
}

pub fn list_text() -> String {
    match load() {
        Ok(tools) if tools.is_empty() => format!(
            "No user MCP tools yet.\n\
File: {}\n\n\
Add one:\n  /mcp add email email,inbox,mail https://your.worker/email/today\n\n\
Then ask: \"aaj ka email kya hai\"",
            conf_path_display()
        ),
        Ok(tools) => {
            let mut out = format!("User MCP tools ({})\nFile: {}\n\n", tools.len(), conf_path_display());
            for t in tools {
                out.push_str(&format!(
                    "• {}  triggers=[{}]  {}\n",
                    t.name,
                    t.triggers.join(", "),
                    t.url
                ));
            }
            out.push_str("\n/mcp add|remove|test  ·  built-in tools: /tools");
            out
        }
        Err(e) => format!("Could not load MCP config: {e}"),
    }
}

pub fn add(name: &str, triggers: &str, url: &str) -> Result<String> {
    let name = name.trim();
    if !valid_name(name) {
        return Err(Error::Tool(
            "name: letters/digits/_/- only, max 40".into(),
        ));
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
    let mut tools = load().unwrap_or_default();
    tools.retain(|t| t.name != name);
    tools.push(McpTool {
        name: name.to_string(),
        triggers,
        url: url.to_string(),
    });
    save_all(&tools)?;
    Ok(format!(
        "Saved MCP tool `{name}` → {}\nTriggers fire when your message contains those words.",
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

/// GET url with ?q=encoded query (and &text=). HTTPS uses curl if available.
pub fn invoke(tool: &McpTool, query: &str, cfg: &Config) -> Result<ToolOutput> {
    let sep = if tool.url.contains('?') { "&" } else { "?" };
    let full = format!(
        "{}{}q={}&text={}",
        tool.url,
        sep,
        url_encode(query),
        url_encode(query)
    );
    let body = fetch_url(&full, cfg)?;
    let summary = string_field(&body, "summary")
        .or_else(|| string_field(&body, "result"))
        .or_else(|| string_field(&body, "text"))
        .unwrap_or_else(|| body.chars().take(cfg.max_output_chars.min(2000)).collect());
    Ok(ToolOutput {
        name: "mcp",
        body: format!(
            "│ tool: {}\n│ {}\n│ source: {}",
            tool.name, summary, tool.url
        ),
    })
}

fn fetch_url(url: &str, cfg: &Config) -> Result<String> {
    if let Some(rest) = url.strip_prefix("http://") {
        let (host, path) = match rest.split_once('/') {
            Some((h, p)) => (h, format!("/{p}")),
            None => (rest, "/".into()),
        };
        // strip host:port
        let host_only = host.split(':').next().unwrap_or(host);
        return get_text(host_only, &path, cfg);
    }
    if url.starts_with("https://") {
        return fetch_https_curl(url, cfg.http_timeout);
    }
    Err(Error::Tool("only http:// or https:// URLs allowed".into()))
}

fn fetch_https_curl(url: &str, timeout: Duration) -> Result<String> {
    // Fixed argv only — not a user shell string.
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
        .map_err(|e| {
            Error::Http(format!(
                "https needs `curl` on PATH ({e}). Or use http:// endpoint."
            ))
        })?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return Err(Error::Http(format!(
            "curl failed: {}",
            err.chars().take(200).collect::<String>()
        )));
    }
    String::from_utf8(out.stdout).map_err(|e| Error::Parse(e.to_string()))
}

/// Handle `/mcp …` meta commands. Returns Some(response) if consumed.
pub fn handle_command(line: &str) -> Option<String> {
    let line = line.trim();
    let lower = line.to_lowercase();
    if lower == "/mcp" || lower == "/mcp list" || lower == "mcp list" {
        return Some(list_text());
    }
    if let Some(rest) = lower
        .strip_prefix("/mcp add ")
        .or_else(|| line.strip_prefix("/mcp add "))
    {
        // name triggers url — triggers may have commas; url is last token starting with http
        let raw = if let Some(r) = line.strip_prefix("/mcp add ") {
            r
        } else {
            rest
        };
        return Some(parse_add(raw));
    }
    if let Some(name) = lower
        .strip_prefix("/mcp remove ")
        .or_else(|| lower.strip_prefix("/mcp rm "))
    {
        return Some(match remove(name.trim()) {
            Ok(s) => s,
            Err(e) => e.to_string(),
        });
    }
    if lower.starts_with("/mcp test ") {
        let rest = line["/mcp test ".len()..].trim();
        let mut sp = rest.splitn(2, char::is_whitespace);
        let name = sp.next().unwrap_or("");
        let q = sp.next().unwrap_or("test");
        return Some(test_tool(name, q));
    }
    if lower == "/mcp" || lower.starts_with("/mcp ") {
        return Some(
            "MCP user tools\n\
  /mcp list\n\
  /mcp add <name> <trig1,trig2> <https://url>\n\
  /mcp remove <name>\n\
  /mcp test <name> [query]\n\
  /mcp              (built-in schema JSON via agent /mcp alone still works for builtins)\n\
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
        return "usage: /mcp add <name> <trigger,trigger> <https://url>".into();
    }
    let name = tokens[0];
    // last token that looks like url
    let url_i = tokens.iter().rposition(|t| t.starts_with("http://") || t.starts_with("https://"));
    let Some(ui) = url_i else {
        return "url must be http:// or https://".into();
    };
    if ui < 2 {
        return "usage: /mcp add <name> <triggers> <url>".into();
    }
    let triggers = tokens[1..ui].join(" ");
    let url = tokens[ui];
    match add(name, &triggers, url) {
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
        return format!("no tool `{name}` — /mcp list");
    };
    match invoke(&tool, query, &cfg) {
        Ok(o) => o.render(),
        Err(e) => format!("test failed: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn parse_and_roundtrip() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join("0zcode_mcp_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mcp_tools.conf");
        env::set_var("ZEROZ_MCP_CONFIG", &path);
        add("email", "email,inbox", "https://example.com/mail").unwrap();
        let tools = load().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "email");
        assert!(match_tools("check my inbox").iter().any(|t| t.name == "email"));
        remove("email").unwrap();
        assert!(load().unwrap().is_empty());
        env::remove_var("ZEROZ_MCP_CONFIG");
    }
}
