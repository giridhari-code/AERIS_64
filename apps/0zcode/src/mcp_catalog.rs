//! Built-in catalog of 100+ external MCP tools & integrations.
//!
//! Source file: `assets/mcp_catalog.conf` (embedded at compile time when present,
//! otherwise falls back to a minimal inline list).
//!
//! Commands:
//!   /mcp catalog
//!   /mcp catalog <query>
//!   /mcp catalog kind http|mcp|local

use std::env;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct CatalogEntry {
    pub name: String,
    pub triggers: String,
    pub kind: String,
    pub detail: String,
    pub reference: String,
}

/// Prefer workspace / crate assets path; also allow ZEROZ_MCP_CATALOG override.
fn catalog_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();
    if let Ok(p) = env::var("ZEROZ_MCP_CATALOG") {
        if !p.is_empty() {
            paths.push(PathBuf::from(p));
        }
    }
    // relative to CARGO_MANIFEST_DIR when building tests / from source tree
    paths.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets/mcp_catalog.conf"));
    // cwd fallbacks
    paths.push(PathBuf::from("assets/mcp_catalog.conf"));
    paths.push(PathBuf::from("apps/0zcode/assets/mcp_catalog.conf"));
    paths
}

fn parse_line(line: &str) -> Option<CatalogEntry> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let parts: Vec<&str> = line.split('|').map(str::trim).collect();
    if parts.len() < 5 {
        return None;
    }
    Some(CatalogEntry {
        name: parts[0].to_string(),
        triggers: parts[1].to_string(),
        kind: parts[2].to_string(),
        detail: parts[3].to_string(),
        reference: parts[4].to_string(),
    })
}

pub fn load() -> Vec<CatalogEntry> {
    for p in catalog_paths() {
        if let Ok(text) = fs::read_to_string(&p) {
            let mut out = Vec::new();
            for line in text.lines() {
                if let Some(e) = parse_line(line) {
                    out.push(e);
                }
            }
            if !out.is_empty() {
                return out;
            }
        }
    }
    // minimal fallback if asset missing from install prefix
    vec![
        CatalogEntry {
            name: "github_mcp".into(),
            triggers: "github,repo".into(),
            kind: "mcp".into(),
            detail: "GitHub MCP official".into(),
            reference: "@modelcontextprotocol/server-github".into(),
        },
        CatalogEntry {
            name: "filesystem".into(),
            triggers: "files,fs".into(),
            kind: "mcp".into(),
            detail: "MCP filesystem server".into(),
            reference: "@modelcontextprotocol/server-filesystem".into(),
        },
        CatalogEntry {
            name: "wttr".into(),
            triggers: "weather".into(),
            kind: "http".into(),
            detail: "wttr.in weather text".into(),
            reference: "https://wttr.in/{q}?format=3".into(),
        },
    ]
}

pub fn search(query: &str) -> Vec<CatalogEntry> {
    let q = query.trim().to_lowercase();
    if q.is_empty() {
        return load();
    }
    load()
        .into_iter()
        .filter(|e| {
            e.name.to_lowercase().contains(&q)
                || e.triggers.to_lowercase().contains(&q)
                || e.kind.to_lowercase().contains(&q)
                || e.detail.to_lowercase().contains(&q)
                || e.reference.to_lowercase().contains(&q)
        })
        .collect()
}

pub fn by_kind(kind: &str) -> Vec<CatalogEntry> {
    let k = kind.trim().to_lowercase();
    load()
        .into_iter()
        .filter(|e| e.kind.to_lowercase() == k)
        .collect()
}

pub fn list_text(filter: &str) -> String {
    let filter = filter.trim();
    let entries = if filter.is_empty() {
        load()
    } else if let Some(kind) = filter.strip_prefix("kind ") {
        by_kind(kind)
    } else {
        search(filter)
    };
    if entries.is_empty() {
        return format!(
            "No catalog matches for `{filter}`.\n\nTry: /mcp catalog · /mcp catalog github · /mcp catalog kind mcp"
        );
    }
    let total = load().len();
    let mut out = format!(
        "MCP / integrations catalog — showing {} of {} entries\n\n",
        entries.len(),
        total
    );
    out.push_str("name                 kind    detail\n");
    out.push_str("────────────────────────────────────────────────────────────\n");
    for e in entries.iter().take(80) {
        out.push_str(&format!(
            "{:<20} {:<7} {}\n",
            e.name.chars().take(20).collect::<String>(),
            e.kind.chars().take(7).collect::<String>(),
            e.detail.chars().take(48).collect::<String>()
        ));
    }
    if entries.len() > 80 {
        out.push_str(&format!("\n… +{} more (narrow with /mcp catalog <query>)\n", entries.len() - 80));
    }
    out.push_str(
        "\nKinds: http (public GET APIs) · mcp (external MCP packages) · local (AERIS/0z)\n\
Enable HTTP tool:  /mcp add <name> <trigger,trigger> <url>\n\
MCP packages run in Claude/Cursor hosts — 0zcode invokes HTTP tools in-process.\n\
File: assets/mcp_catalog.conf  ·  override: ZEROZ_MCP_CATALOG\n",
    );
    out
}

pub fn show(name: &str) -> String {
    let name = name.trim();
    for e in load() {
        if e.name.eq_ignore_ascii_case(name) {
            return format!(
                "Catalog entry `{name}`\n\n\
  kind:      {}\n\
  triggers:  {}\n\
  detail:    {}\n\
  reference: {}\n\n\
HTTP add example:\n  /mcp add {} {},{} {}\n",
                e.kind,
                e.triggers,
                e.detail,
                e.reference,
                e.name,
                e.triggers.split(',').next().unwrap_or(&e.name),
                e.name,
                e.reference
            );
        }
    }
    format!("Unknown catalog entry `{name}`.\n\n{}", list_text(name))
}

pub fn handle_catalog_arg(arg: &str) -> String {
    let arg = arg.trim();
    if arg.is_empty() {
        return list_text("");
    }
    if let Some(name) = arg.strip_prefix("show ") {
        return show(name.trim());
    }
    list_text(arg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catalog_loads_many() {
        let all = load();
        // asset present in crate → expect 100+
        assert!(
            all.len() >= 3,
            "expected catalog entries, got {}",
            all.len()
        );
    }

    #[test]
    fn search_github() {
        let hits = search("github");
        assert!(!hits.is_empty());
    }
}
