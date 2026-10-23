//! Skills + ToolSearch — discover and run packaged skills / connected tools.

use crate::error::{Error, Result};
use crate::mcp_store;
use crate::tools::ToolOutput;
use crate::workspace;
use std::fs;
use std::path::PathBuf;

fn skills_dir() -> PathBuf {
    if let Ok(p) = std::env::var("ZEROZ_SKILLS_DIR") {
        return PathBuf::from(p);
    }
    workspace::root().join(".0zcode").join("skills")
}

/// ToolSearch — find connected tools (builtin keywords + MCP + skills).
pub fn tool_search(query: &str) -> ToolOutput {
    let q = query.trim().to_lowercase();
    let mut hits: Vec<String> = Vec::new();

    let builtins = [
        ("WebSearch", "internet search"),
        ("WebFetch", "fetch website content"),
        ("TaskCreate", "create task"),
        ("TaskList", "list tasks"),
        ("TeamCreate", "create agent team"),
        ("ListAgents", "list subagents/teams"),
        ("Workflow", "coordinate multi-step"),
        ("ws.list", "list workspace files"),
        ("ws.read", "read file"),
        ("ws.grep", "search in files"),
        ("calc", "calculator"),
        ("weather", "weather forecast"),
        ("time", "current time"),
        ("EnterPlanMode", "planning mode"),
        ("CronCreate", "schedule job"),
        ("Skill", "run a skill from .0zcode/skills"),
        ("ListMcpResources", "list MCP tools as resources"),
        ("ReadMcpResource", "read one MCP tool config"),
    ];
    for (name, desc) in builtins {
        if q.is_empty() || name.to_lowercase().contains(&q) || desc.contains(&q) {
            hits.push(format!("│ {name:<20} {desc}"));
        }
    }
    if let Ok(tools) = mcp_store::load() {
        for t in tools {
            if q.is_empty()
                || t.name.to_lowercase().contains(&q)
                || t.triggers.iter().any(|tr| tr.contains(&q))
            {
                hits.push(format!(
                    "│ mcp:{:<16} triggers={} url={}",
                    t.name,
                    t.triggers.join(","),
                    t.url
                ));
            }
        }
    }
    if let Ok(rd) = fs::read_dir(skills_dir()) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if name.ends_with(".md") || name.ends_with(".txt") || name.ends_with(".skill") {
                if q.is_empty() || name.to_lowercase().contains(&q) {
                    hits.push(format!("│ skill:{name}"));
                }
            }
        }
    }
    if hits.is_empty() {
        hits.push("│ (no matches)".into());
    }
    ToolOutput {
        name: "ToolSearch",
        body: format!("│ ToolSearch `{query}`\n{}", hits.join("\n")),
    }
}

/// Skill — run/read a skill file (markdown instructions in workspace).
pub fn run_skill(name: &str) -> Result<ToolOutput> {
    let name = name.trim().trim_matches('"');
    if name.is_empty() {
        return Err(Error::Tool(
            "Skill: usage Skill <name>  (files in .0zcode/skills/)".into(),
        ));
    }
    let dir = skills_dir();
    let candidates = [
        dir.join(name),
        dir.join(format!("{name}.md")),
        dir.join(format!("{name}.txt")),
        dir.join(format!("{name}.skill")),
    ];
    for p in &candidates {
        if p.is_file() {
            let body = fs::read_to_string(p).map_err(|e| Error::Tool(e.to_string()))?;
            let short: String = body.chars().take(4000).collect();
            return Ok(ToolOutput {
                name: "Skill",
                body: format!("│ skill {}\n│ {}\n{}", p.display(), "—".repeat(20), short),
            });
        }
    }
    Err(Error::Tool(format!(
        "Skill `{name}` not found under {}",
        dir.display()
    )))
}

pub fn list_mcp_resources() -> ToolOutput {
    let tools = mcp_store::load().unwrap_or_default();
    if tools.is_empty() {
        return ToolOutput {
            name: "ListMcpResources",
            body: format!(
                "│ (no MCP tools)\n│ conf: {}\n│ /mcp add <name> <triggers> <https://url>",
                mcp_store::conf_path_display()
            ),
        };
    }
    let mut body = format!("│ {} MCP resource(s)\n", tools.len());
    for t in tools {
        body.push_str(&format!(
            "│ {}  triggers={}  {}\n",
            t.name,
            t.triggers.join(","),
            t.url
        ));
    }
    ToolOutput {
        name: "ListMcpResources",
        body,
    }
}

pub fn read_mcp_resource(name: &str) -> Result<ToolOutput> {
    let name = name.trim();
    let tools = mcp_store::load()?;
    let t = tools
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| Error::Tool(format!("ReadMcpResource: no tool `{name}`")))?;
    Ok(ToolOutput {
        name: "ReadMcpResource",
        body: format!(
            "│ name: {}\n│ triggers: {}\n│ url: {}\n│ conf: {}",
            t.name,
            t.triggers.join(", "),
            t.url,
            mcp_store::conf_path_display()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_finds_web() {
        let o = tool_search("web");
        assert!(o.body.to_lowercase().contains("websearch") || o.body.contains("WebSearch"));
    }
}
