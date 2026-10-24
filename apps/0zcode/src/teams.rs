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
    body.push_str("\n│ parallel: /team run explore,test,search | <goal>");
    Ok(ToolOutput {
        name: "Workflow",
        body,
    })
}

/// Resolve members from a team id/name or explicit comma list.
fn resolve_members(spec: &str) -> Result<Vec<Kind>> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Err(Error::Tool(
            "TeamRun: need team id/name or members (explore,test,search)".into(),
        ));
    }
    // Prefer explicit kind list when every token parses.
    let tokens: Vec<&str> = spec
        .split(|c| c == ',' || c == '|')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if !tokens.is_empty() && tokens.iter().all(|t| Kind::parse(t).is_some()) {
        return Ok(tokens.iter().filter_map(|t| Kind::parse(t)).collect());
    }
    // Otherwise look up a saved team.
    let teams = load()?;
    let team = teams
        .into_iter()
        .find(|t| t.id == spec || t.name.eq_ignore_ascii_case(spec));
    let Some(team) = team else {
        return Err(Error::Tool(format!(
            "TeamRun: unknown team `{spec}` — create with TeamCreate name | explore,test"
        )));
    };
    if !team.active {
        return Err(Error::Tool(format!(
            "TeamRun: team `{}` is stopped",
            team.name
        )));
    }
    let kinds: Vec<Kind> = team
        .members
        .iter()
        .filter_map(|m| Kind::parse(m))
        .collect();
    if kinds.is_empty() {
        return Err(Error::Tool("TeamRun: team has no valid members".into()));
    }
    Ok(kinds)
}

/// Run multiple subagents **in parallel** on the same goal (company-standard).
///
/// Usage:
///   /team run explore,test,search | map workspace
///   TeamRun myteam | investigate flaky tests
///   /parallel explore,research | latest AI news context
///
/// Requires `ZEROZ_AGENT=1`. One Human Approval covers the whole team run.
pub fn run_parallel(spec: &str, goal: &str) -> Result<ToolOutput> {
    let goal = goal.trim();
    if goal.is_empty() {
        return Err(Error::Tool(
            "TeamRun: usage /team run <members|team> | <goal>".into(),
        ));
    }
    if !crate::harness::agent_enabled() {
        return Err(Error::Tool(
            "TeamRun: export ZEROZ_AGENT=1 to enable multi-agent runs".into(),
        ));
    }
    let kinds = resolve_members(spec)?;
    crate::approval::require(
        crate::approval::ActionClass::Agent,
        &format!(
            "parallel team [{}] · {}",
            kinds
                .iter()
                .map(|k| k.as_str())
                .collect::<Vec<_>>()
                .join(","),
            goal.chars().take(80).collect::<String>()
        ),
    )
    .map_err(Error::Tool)?;

    // Cap parallelism — company default 4 concurrent bots.
    let max = std::env::var("ZEROZ_TEAM_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4)
        .clamp(1, 8);
    let kinds: Vec<Kind> = kinds.into_iter().take(max).collect();

    let goal_owned = goal.to_string();
    let mut handles = Vec::new();
    for kind in kinds {
        let g = goal_owned.clone();
        handles.push(std::thread::spawn(move || {
            let label = kind.as_str().to_string();
            match crate::harness::run_kind(kind, &g, false) {
                Ok(out) => (label, true, out.body),
                Err(e) => (label, false, format!("error: {e}")),
            }
        }));
    }

    let mut body = format!(
        "│ multi-agent parallel · {} bots · goal: {}\n│ workspace {}\n",
        handles.len(),
        goal.chars().take(100).collect::<String>(),
        workspace::root_display()
    );
    let mut ok = 0usize;
    let mut fail = 0usize;
    for h in handles {
        match h.join() {
            Ok((label, success, report)) => {
                if success {
                    ok += 1;
                } else {
                    fail += 1;
                }
                body.push_str(&format!(
                    "│\n│ ── bot: {label} · {}\n",
                    if success { "ok" } else { "failed" }
                ));
                for line in report.lines().take(16) {
                    body.push_str(&format!("│ {line}\n"));
                }
            }
            Err(_) => {
                fail += 1;
                body.push_str("│\n│ ── bot: (thread panicked)\n");
            }
        }
    }
    body.push_str(&format!(
        "│\n│ summary: {ok} ok · {fail} failed · parallel join complete"
    ));
    Ok(ToolOutput {
        name: "TeamRun",
        body,
    })
}

/// Parse `/team run members|goal` or `TeamRun members | goal`.
pub fn parse_run_line(msg: &str) -> Option<(String, String)> {
    let t = msg.trim();
    let lower = t.to_lowercase();
    let rest = if lower.starts_with("/team run ") {
        t["/team run ".len()..].trim()
    } else if lower.starts_with("teamrun ") {
        t["teamrun ".len()..].trim()
    } else if lower.starts_with("team run ") {
        t["team run ".len()..].trim()
    } else if lower.starts_with("/parallel ") {
        t["/parallel ".len()..].trim()
    } else if lower.starts_with("parallel ") {
        t["parallel ".len()..].trim()
    } else {
        return None;
    };
    if let Some((spec, goal)) = rest.split_once('|') {
        return Some((spec.trim().to_string(), goal.trim().to_string()));
    }
    // fallback: first token = team/members blob, rest = goal
    let mut parts = rest.splitn(2, char::is_whitespace);
    let spec = parts.next()?.trim().to_string();
    let goal = parts.next().unwrap_or("").trim().to_string();
    if spec.is_empty() {
        return None;
    }
    Some((spec, goal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_agents_nonempty() {
        let o = list_agents();
        assert!(o.body.contains("subagents") || o.body.contains("explore"));
    }

    #[test]
    fn parse_run_line_pipe() {
        let (s, g) = parse_run_line("/team run explore,test | map repo").unwrap();
        assert_eq!(s, "explore,test");
        assert!(g.contains("map"));
    }
}
