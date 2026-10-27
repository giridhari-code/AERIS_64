//! Plugins — features & integrations registry (company-standard).
//!
//! Built-in plugins ship with 0zcode; user plugins live in
//! `{workspace}/.0zcode/plugins/*.conf` (name=… enabled=1 kind=…).
//!
//! Commands: /plugins · /plugins enable <name> · /plugins disable <name>

use crate::workspace;
use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Plugin {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub builtin: bool,
    pub detail: String,
}

fn user_dir() -> PathBuf {
    workspace::root().join(".0zcode").join("plugins")
}

fn builtins() -> Vec<Plugin> {
    vec![
        Plugin {
            name: "search".into(),
            kind: "tool".into(),
            enabled: true,
            builtin: true,
            detail: "web search + page fetch".into(),
        },
        Plugin {
            name: "workspace".into(),
            kind: "integration".into(),
            enabled: true,
            builtin: true,
            detail: "sandboxed list/read/glob/grep/edit/write/shell".into(),
        },
        Plugin {
            name: "mcp".into(),
            kind: "integration".into(),
            enabled: true,
            builtin: true,
            detail: "user MCP HTTP tools".into(),
        },
        Plugin {
            name: "aeris".into(),
            kind: "llm".into(),
            enabled: true,
            builtin: true,
            detail: "AERIS-64 model link".into(),
        },
        Plugin {
            name: "subagents".into(),
            kind: "agent".into(),
            enabled: true,
            builtin: true,
            detail: "explore/edit/test/search workers".into(),
        },
        Plugin {
            name: "teams".into(),
            kind: "agent".into(),
            enabled: true,
            builtin: true,
            detail: "multi-agent coordination".into(),
        },
        Plugin {
            name: "lsp".into(),
            kind: "devtools".into(),
            enabled: true,
            builtin: true,
            detail: "cargo/check diagnostics".into(),
        },
        Plugin {
            name: "computer".into(),
            kind: "devtools".into(),
            enabled: false,
            builtin: true,
            detail: "computer-use surface (gated)".into(),
        },
    ]
}

fn load_user() -> Vec<Plugin> {
    let dir = user_dir();
    let Ok(rd) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("conf") {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let mut name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("plugin")
            .to_string();
        let mut kind = "custom".to_string();
        let mut enabled = true;
        let mut detail = String::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            if let Some((k, v)) = line.split_once('=') {
                match k.trim() {
                    "name" => name = v.trim().to_string(),
                    "kind" => kind = v.trim().to_string(),
                    "enabled" => {
                        enabled = matches!(v.trim(), "1" | "true" | "yes");
                    }
                    "detail" => detail = v.trim().to_string(),
                    _ => {}
                }
            }
        }
        out.push(Plugin {
            name,
            kind,
            enabled,
            builtin: false,
            detail,
        });
    }
    out
}

fn disabled_set() -> Vec<String> {
    env::var("ZEROZ_PLUGINS_DISABLE")
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| !s.is_empty())
        .collect()
}

pub fn list() -> Vec<Plugin> {
    let disabled = disabled_set();
    let mut all = builtins();
    all.extend(load_user());
    for p in &mut all {
        if disabled.iter().any(|d| d == &p.name.to_lowercase()) {
            p.enabled = false;
        }
    }
    all
}

pub fn is_enabled(name: &str) -> bool {
    list()
        .into_iter()
        .find(|p| p.name.eq_ignore_ascii_case(name))
        .map(|p| p.enabled)
        .unwrap_or(true)
}

fn write_override(name: &str, enabled: bool) -> Result<(), String> {
    let dir = user_dir();
    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let path = dir.join(format!("{name}.conf"));
    let mut f = fs::File::create(&path).map_err(|e| e.to_string())?;
    writeln!(f, "name={name}").map_err(|e| e.to_string())?;
    writeln!(f, "enabled={}", if enabled { "1" } else { "0" }).map_err(|e| e.to_string())?;
    writeln!(f, "kind=override").map_err(|e| e.to_string())?;
    Ok(())
}

pub fn list_text() -> String {
    let plugs = list();
    let mut out = format!("Plugins ({} registered)\n\n", plugs.len());
    for p in &plugs {
        out.push_str(&format!(
            "  {}  {:<12}  {:<12}  {}{}\n",
            if p.enabled { "✔" } else { "·" },
            p.name,
            p.kind,
            p.detail,
            if p.builtin { "" } else { " [user]" }
        ));
    }
    out.push_str("\n/plugins enable <name> · /plugins disable <name>\n");
    out.push_str("Env: ZEROZ_PLUGINS_DISABLE=computer,lsp\n");
    out
}

pub fn handle_command(line: &str) -> String {
    let t = line.trim();
    let lower = t.to_lowercase();
    if lower == "/plugins" || lower == "/plugin" || lower == "plugins" {
        return list_text();
    }
    if let Some(name) = lower.strip_prefix("/plugins enable ") {
        let name = name.trim();
        return match write_override(name, true) {
            Ok(()) => format!("Enabled plugin `{name}`.\n\n{}", list_text()),
            Err(e) => format!("plugin error: {e}"),
        };
    }
    if let Some(name) = lower.strip_prefix("/plugins disable ") {
        let name = name.trim();
        return match write_override(name, false) {
            Ok(()) => format!("Disabled plugin `{name}`.\n\n{}", list_text()),
            Err(e) => format!("plugin error: {e}"),
        };
    }
    "Usage: /plugins · /plugins enable <name> · /plugins disable <name>".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_include_aeris() {
        assert!(builtins().iter().any(|p| p.name == "aeris"));
    }
}
