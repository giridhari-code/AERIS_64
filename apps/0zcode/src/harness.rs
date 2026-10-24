//! Multi-step agent harness (OpenCode-style), company-standard.
//!
//! Gates:
//!   ZEROZ_AGENT=1   — enable `/agent` loop
//!   ZEROZ_WRITE=1   — allow ws.write / ws.mkdir
//!   ZEROZ_SHELL=1   — allowlisted shell
//!   ZEROZ_AGENT_MAX_STEPS — default 8, clamp 1..=20
//!
//! Protocol (deterministic + optional LLM later):
//!   TOOL list [path]
//!   TOOL read <path>
//!   TOOL grep <pattern> [path]
//!   TOOL write <path> | <body one line or \n escaped>
//!   TOOL mkdir <path>
//!   TOOL shell <allowlisted>
//!   DONE <summary>
//!
//! Without LLM: built-in planner for explore / write-note style goals.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::tools::ToolOutput;
use crate::workspace;
use std::env;

#[derive(Debug, Clone)]
pub struct Step {
    pub n: usize,
    pub action: String,
    pub result: String,
}

pub fn agent_enabled() -> bool {
    env::var("ZEROZ_AGENT").ok().as_deref() == Some("1")
}

fn max_steps() -> usize {
    env::var("ZEROZ_AGENT_MAX_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
        .clamp(1, 20)
}

/// Entry: `/agent <goal>` or `agent: <goal>`.
pub fn run(goal: &str, _cfg: &Config) -> Result<ToolOutput> {
    if !agent_enabled() {
        return Err(Error::Tool(
            "agent loop disabled — export ZEROZ_AGENT=1 (and ZEROZ_WRITE=1 if writes needed)"
                .into(),
        ));
    }
    let goal = goal.trim();
    if goal.is_empty() {
        return Err(Error::Tool(
            "usage: /agent <goal>   e.g. /agent explore this project".into(),
        ));
    }

    let mut steps: Vec<Step> = Vec::new();
    let mut transcript = String::new();
    let kind = crate::subagents::infer(goal);
    let plan = crate::subagents::plan(kind, goal);
    transcript.push_str(&format!("subagent={}\n", kind.as_str()));

    for (i, action) in plan.iter().enumerate() {
        if i >= max_steps() {
            break;
        }
        let n = i + 1;
        let result = execute_action(action)?;
        let summary = result.body.chars().take(400).collect::<String>();
        transcript.push_str(&format!("## step {n}: {action}\n{summary}\n\n"));
        steps.push(Step {
            n,
            action: action.clone(),
            result: summary,
        });
        if action.starts_with("DONE") {
            break;
        }
    }

    let mut body = format!(
        "│ agent harness · {} steps · max {}\n│ goal: {}\n│ workspace {}\n",
        steps.len(),
        max_steps(),
        goal.chars().take(120).collect::<String>(),
        workspace::root_display()
    );
    for s in &steps {
        body.push_str(&format!("│\n│ [{}/{}] {}\n", s.n, steps.len(), s.action));
        for line in s.result.lines().take(8) {
            let l = line.trim_start_matches('│').trim();
            body.push_str(&format!("│   {l}\n"));
        }
    }
    body.push_str("│\n│ done — company sandbox (write/shell gated by env)");
    let _ = transcript;
    Ok(ToolOutput {
        name: "agent",
        body,
    })
}

// Planner lives in `subagents::plan` — harness only executes TOOL steps.

fn execute_action(action: &str) -> Result<ToolOutput> {
    let a = action.trim();
    if a.starts_with("DONE") {
        return Ok(ToolOutput {
            name: "agent.done",
            body: format!("│ {a}"),
        });
    }
    let Some(rest) = a.strip_prefix("TOOL ") else {
        return Err(Error::Tool(format!("bad action: {a}")));
    };
    let rest = rest.trim();
    if rest == "list" || rest.starts_with("list ") {
        let path = rest.strip_prefix("list").unwrap_or("").trim();
        return workspace::list(if path.is_empty() { "." } else { path });
    }
    if let Some(path) = rest.strip_prefix("read ") {
        return workspace::read(path.trim());
    }
    if let Some(rest) = rest.strip_prefix("grep ") {
        let mut sp = rest.splitn(2, char::is_whitespace);
        let pat = sp.next().unwrap_or("");
        let path = sp.next().unwrap_or(".");
        return workspace::grep(pat, path);
    }
    if let Some(rest) = rest.strip_prefix("write ") {
        // write path | body
        let (path, body) = if let Some((p, b)) = rest.split_once('|') {
            (p.trim(), b.trim().replace("\\n", "\n"))
        } else {
            let mut sp = rest.splitn(2, char::is_whitespace);
            (
                sp.next().unwrap_or("").trim(),
                sp.next().unwrap_or("").to_string(),
            )
        };
        return workspace::write_file(path, &body);
    }
    if let Some(path) = rest.strip_prefix("mkdir ") {
        return workspace::mkdir(path.trim());
    }
    if let Some(rest) = rest.strip_prefix("edit ") {
        // edit path | old | new
        let parts: Vec<&str> = rest.splitn(3, '|').map(str::trim).collect();
        if parts.len() < 3 {
            return Err(Error::Tool("usage: TOOL edit path | old | new".into()));
        }
        return workspace::search_replace(parts[0], parts[1], parts[2]);
    }
    if let Some(cmd) = rest.strip_prefix("shell ") {
        return workspace::shell_allowlisted(cmd.trim());
    }
    Err(Error::Tool(format!("unknown TOOL: {rest}")))
}

/// Parse user `/agent ...` line → goal string.
pub fn parse_goal(msg: &str) -> Option<String> {
    let t = msg.trim();
    let lower = t.to_lowercase();
    if lower.starts_with("/agent ") {
        return Some(t[7..].trim().to_string());
    }
    if lower == "/agent" {
        return Some(String::new());
    }
    if lower.starts_with("agent: ") {
        return Some(t[7..].trim().to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn plan_explore_has_list() {
        let kind = crate::subagents::infer("explore this project");
        let p = crate::subagents::plan(kind, "explore this project");
        assert!(p.iter().any(|s| s.contains("list")));
    }

    #[test]
    fn agent_disabled_errors() {
        let _g = LOCK.lock().unwrap();
        env::remove_var("ZEROZ_AGENT");
        let r = run("explore", &Config::default());
        assert!(r.is_err());
    }
}
