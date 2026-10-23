//! Agent teams — coordinate subagents (OpenCode-style), company-standard.
//! Store: `{workspace}/.0zcode/teams.jsonl`

use crate::error::{Error, Result};
use crate::subagents::Kind;
use crate::tools::ToolOutput;
use crate::workspace;
use std::fs::{self};
use std::io::{BufRead, BufReader, Write};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Team {
    pub id: String,
    pub name: String,
    pub members: Vec<String>, // subagent kinds
    pub active: bool,
}

fn path() -> std::path::PathBuf {
    workspace::root().join(".0zcode").join("teams.jsonl")
}

fn now_id() -> String {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("team{t}")
}

fn field(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let i = json.find(&pat)?;
    let rest = &json[i + pat.len()..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                out.push(n);
            }
        } else if c == '"' {
            break;
        } else {
            out.push(c);
        }
    }
    Some(out)
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn load() -> Result<Vec<Team>> {
    let p = path();
    if !p.is_file() {
        return Ok(Vec::new());
    }
    let f = fs::File::open(&p).map_err(|e| Error::Tool(e.to_string()))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines().flatten() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let id = field(line, "id").unwrap_or_default();
        let name = field(line, "name").unwrap_or_default();
        let members = field(line, "members")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        let active = field(line, "active").map(|a| a == "1" || a == "true").unwrap_or(true);
        if !id.is_empty() {
            out.push(Team {
                id,
                name,
                members,
                active,
            });
        }
    }
    Ok(out)
}

fn save(teams: &[Team]) -> Result<()> {
    let p = path();
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Tool(e.to_string()))?;
    }
    let mut f = fs::File::create(&p).map_err(|e| Error::Tool(e.to_string()))?;
    for t in teams {
        writeln!(
            f,
            "{{\"id\":\"{}\",\"name\":\"{}\",\"members\":\"{}\",\"active\":\"{}\"}}",
            esc(&t.id),
            esc(&t.name),
            esc(&t.members.join(",")),
            if t.active { "1" } else { "0" }
        )
        .map_err(|e| Error::Tool(e.to_string()))?;
    }
    Ok(())
}

pub fn create(name: &str, members: &str) -> Result<ToolOutput> {
    let name = name.trim();
    if name.is_empty() {
        return Err(Error::Tool("TeamCreate: empty name".into()));
    }
    let members: Vec<String> = members
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .filter(|s| Kind::parse(s).is_some())
        .collect();
    if members.is_empty() {
        return Err(Error::Tool(
            "TeamCreate: members must be explore,edit,test,search,research".into(),
        ));
    }
    let mut teams = load()?;
    let t = Team {
        id: now_id(),
        name: name.to_string(),
        members: members.clone(),
        active: true,
    };
    let id = t.id.clone();
    teams.push(t);
    save(&teams)?;
    Ok(ToolOutput {
        name: "TeamCreate",
        body: format!(
            "│ team {id}\n│ name: {name}\n│ members: {}\n│ active: true",
            members.join(", ")
        ),
    })
}

pub fn delete(id: &str) -> Result<ToolOutput> {
    let mut teams = load()?;
    let before = teams.len();
    teams.retain(|t| t.id != id && t.name != id);
    if teams.len() == before {
        return Err(Error::Tool(format!("TeamDelete: no team `{id}`")));
    }
    save(&teams)?;
    Ok(ToolOutput {
        name: "TeamDelete",
        body: format!("│ deleted team {id}"),
    })
}

pub fn list_agents() -> ToolOutput {
    ToolOutput {
        name: "ListAgents",
        body: format!(
            "│ subagents\n│ {}\n│ teams below\n{}",
            crate::subagents::catalog(),
            list_teams_body()
        ),
    }
}

fn list_teams_body() -> String {
    match load() {
        Ok(teams) if teams.is_empty() => "│ (no teams)".into(),
        Ok(teams) => {
            let mut b = String::new();
            for t in teams {
                b.push_str(&format!(
                    "│ {}  {}  [{}]  {}\n",
                    t.id,
                    t.name,
                    t.members.join(","),
                    if t.active { "active" } else { "stopped" }
                ));
            }
            b
        }
        Err(e) => format!("│ teams error: {e}"),
    }
}

pub fn send_message(target: &str, message: &str) -> Result<ToolOutput> {
    let target = target.trim();
    let message = message.trim();
    if target.is_empty() || message.is_empty() {
        return Err(Error::Tool(
            "SendMessage: usage SendMessage <agent|team> <text>".into(),
        ));
    }
    // Log to workspace inbox; actual dispatch is via /agent
    let inbox = workspace::root().join(".0zcode").join("inbox.jsonl");
    if let Some(parent) = inbox.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Tool(e.to_string()))?;
    }
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&inbox)
        .map_err(|e| Error::Tool(e.to_string()))?;
    writeln!(
        f,
        "{{\"to\":\"{}\",\"msg\":\"{}\"}}",
        esc(target),
        esc(message)
    )
    .map_err(|e| Error::Tool(e.to_string()))?;
    Ok(ToolOutput {
        name: "SendMessage",
        body: format!(
            "│ queued for `{target}`\n│ {message}\n│ run: /agent {target}: {message}"
        ),
    })
}

pub fn workflow(steps: &str) -> Result<ToolOutput> {
    // steps: explore|test|search separated by | or ,
    let parts: Vec<&str> = steps
        .split(|c| c == '|' || c == ',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.is_empty() {
        return Err(Error::Tool(
            "Workflow: usage Workflow explore|test|search".into(),
        ));
    }
    let mut body = String::from("│ workflow plan\n");
    for (i, p) in parts.iter().enumerate() {
        let kind = Kind::parse(p).map(|k| k.as_str()).unwrap_or(*p);
        body.push_str(&format!(
            "│ {}. /agent {kind}: step from workflow\n",
            i + 1
        ));
    }
    body.push_str("│ execute each /agent step in order (ZEROZ_AGENT=1)");
    Ok(ToolOutput {
        name: "Workflow",
        body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_agents_nonempty() {
        let o = list_agents();
        assert!(o.body.contains("subagents") || o.body.contains("explore"));
    }
}
