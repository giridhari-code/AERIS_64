//! Model-driven agent loop (Claude Code style).
//!
//!   model reply -> parse ONE action -> run tool -> send result back -> repeat
//!
//! Unlike `harness` (fixed plan from `subagents::plan`), here the model decides
//! every step. Same safety rails: workspace sandbox, `approval` gate on
//! write/shell, shell allowlist, step limit.
//!
//! Enable:   export ZEROZ_CODING=1 ZEROZ_AGENT=1 ZEROZ_WRITE=1 ZEROZ_SHELL=1
//! Use:      /code <goal>        e.g.  /code fix the failing test in src/tools.rs
//!
//! Protocol (one action per model message):
//!   TOOL list [path]
//!   TOOL read <path> [start_line] [max_lines]
//!   TOOL grep <pattern> [path]
//!   TOOL glob <pattern> [path]
//!   TOOL fetch <url>
//!   TOOL search <query>
//!   TOOL edit <path>            then a block:  <<<<  old  ====  new  >>>>
//!   TOOL write <path>           then a block:  <<<<  content  >>>>
//!   TOOL mkdir <path>
//!   TOOL shell <allowlisted command>
//!   DONE <summary>
//!
//! Block syntax (not `a | b`) so code containing `|` / `||` is safe.

use crate::config::Config;
use crate::error::{Error, Result};
use crate::tools::ToolOutput;
use crate::workspace;
use std::env;
use std::sync::atomic::AtomicBool;

const SYSTEM: &str = r#"You are a coding agent inside a project folder. Reply with EXACTLY ONE action per message and nothing else.
Actions:
TOOL list [path]
TOOL read <path> [start_line] [max_lines]
TOOL grep <pattern> [path]
TOOL glob <pattern> [path]   (e.g. glob src/**/*.rs)
TOOL fetch <url>
TOOL search <query>
TOOL edit <path>
<<<<
exact old text
====
new text
>>>>
TOOL write <path>
<<<<
file content
>>>>
TOOL shell <command>   (cargo check, cargo test, git status ...)
DONE <short summary>
Rules: read a file before editing it. The old text in edit must match exactly and be unique (include nearby lines). After changing code run cargo check or cargo test. Say DONE when finished."#;

/// Longest tool result sent back to the model (server prompt cap is 4000 chars).
const RESULT_CHARS: usize = 2800;

#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    Tool {
        name: String,
        arg: String,
        body: Option<String>,
    },
    Done(String),
    Invalid(String),
}

pub fn enabled() -> bool {
    env::var("ZEROZ_AGENT").ok().as_deref() == Some("1")
}

fn max_steps() -> usize {
    env::var("ZEROZ_AGENT_MAX_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12)
        .clamp(1, 40)
}

fn loop_max_tokens() -> u32 {
    env::var("ZEROZ_CODE_MAX_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024u32)
        .clamp(64, 8192)
}

/// `/code <goal>` or `code: <goal>`.
pub fn parse_goal(msg: &str) -> Option<String> {
    let t = msg.trim();
    let lower = t.to_lowercase();
    if lower == "/code" {
        return Some(String::new());
    }
    if lower.starts_with("/code ") {
        return Some(t[6..].trim().to_string());
    }
    if lower.starts_with("code: ") {
        return Some(t[6..].trim().to_string());
    }
    None
}

/// Find the first `TOOL ...` or `DONE ...` line in a model reply (small models
/// often add chatter before it). A `<<<< ... >>>>` block after a TOOL line is
/// its body.
pub fn parse_action(reply: &str) -> Action {
    let lines: Vec<&str> = reply.lines().collect();
    for (i, raw) in lines.iter().enumerate() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("DONE") {
            let s = rest.trim().trim_start_matches(':').trim();
            return Action::Done(s.to_string());
        }
        if let Some(rest) = line.strip_prefix("TOOL ") {
            let rest = rest.trim();
            let (name, arg) = match rest.split_once(char::is_whitespace) {
                Some((n, a)) => (n.to_string(), a.trim().to_string()),
                None => (rest.to_string(), String::new()),
            };
            if name.is_empty() {
                return Action::Invalid("empty tool name".into());
            }
            let body = block_after(&lines[i + 1..]);
            return Action::Tool { name, arg, body };
        }
    }
    Action::Invalid("no TOOL or DONE line found".into())
}

fn block_after(rest: &[&str]) -> Option<String> {
    let start = rest.iter().position(|l| l.trim() == "<<<<")?;
    let mut out: Vec<&str> = Vec::new();
    for l in rest[start + 1..].iter() {
        if l.trim() == ">>>>" {
            break;
        }
        out.push(*l);
    }
    Some(out.join("\n"))
}

fn exec_edit(path: &str, body: Option<&str>) -> Result<ToolOutput> {
    if path.is_empty() {
        return Err(Error::Tool("edit: missing path".into()));
    }
    let body = body.ok_or_else(|| {
        Error::Tool("edit needs a block: <<<< old ==== new >>>>".into())
    })?;
    let (old, new) = if let Some((o, n)) = body.split_once("\n====\n") {
        (o, n)
    } else if let Some(o) = body.strip_suffix("\n====") {
        (o, "")
    } else {
        return Err(Error::Tool(
            "edit block must have a line `====` between old and new text".into(),
        ));
    };
    workspace::search_replace(path, old, new)
}

fn execute(name: &str, arg: &str, body: Option<&str>, cfg: &Config) -> Result<ToolOutput> {
    match name {
        "write" => {
            if arg.is_empty() {
                return Err(Error::Tool("write: missing path".into()));
            }
            let content = body.ok_or_else(|| {
                Error::Tool("write needs a block: <<<< content >>>>".into())
            })?;
            workspace::write_file(arg, &format!("{content}\n"))
        }
        "edit" => exec_edit(arg, body),
        "read" => {
            let mut it = arg.split_whitespace();
            let path = it.next().unwrap_or("");
            if path.is_empty() {
                return Err(Error::Tool("read: missing path".into()));
            }
            let start: Option<usize> = it.next().and_then(|s| s.parse().ok());
            let limit: Option<usize> = it.next().and_then(|s| s.parse().ok());
            if start.is_none() && limit.is_none() {
                workspace::read(path)
            } else {
                workspace::read_range(path, start.unwrap_or(1), limit.unwrap_or(200))
            }
        }
        "glob" => {
            let mut it = arg.splitn(2, char::is_whitespace);
            let pat = it.next().unwrap_or("");
            let path = it.next().unwrap_or(".").trim();
            workspace::glob(pat, path)
        }
        "fetch" => {
            if cfg.offline {
                return Err(Error::Offline("fetch disabled (offline)".into()));
            }
            crate::tools::web_fetch(arg, cfg)
        }
        "search" => {
            if cfg.offline {
                return Err(Error::Offline("search disabled (offline)".into()));
            }
            crate::tools::web_search(arg, cfg)
        }
        "list" | "grep" | "mkdir" | "shell" => {
            crate::harness::execute_action(&format!("TOOL {name} {arg}"))
        }
        other => Err(Error::Tool(format!(
            "unknown tool `{other}` (use: list read grep glob edit write mkdir shell fetch search)"
        ))),
    }
}

fn clamp(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let mut t: String = s.chars().take(max).collect();
    t.push_str("\n... (truncated)");
    t
}

fn plain(o: &ToolOutput) -> String {
    o.body
        .lines()
        .map(|l| {
            l.strip_prefix("│ ")
                .or_else(|| l.strip_prefix('│'))
                .unwrap_or(l)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Run the loop for `goal`. Needs ZEROZ_CODING=1 and ZEROZ_AGENT=1.
pub fn run(goal: &str, cfg: &Config) -> Result<ToolOutput> {
    if !crate::policy::coding_enabled() {
        return Err(Error::Policy(
            "coding surface is closed - export ZEROZ_CODING=1 to enable /code".into(),
        ));
    }
    if !enabled() {
        return Err(Error::Tool(
            "agent loop disabled - export ZEROZ_AGENT=1 (ZEROZ_WRITE=1 / ZEROZ_SHELL=1 for edits and cargo)"
                .into(),
        ));
    }
    let goal = goal.trim();
    if goal.is_empty() {
        return Err(Error::Tool(
            "usage: /code <goal>   e.g. /code fix the failing test in src/tools.rs".into(),
        ));
    }
    crate::approval::require(
        crate::approval::ActionClass::Agent,
        &format!(
            "code loop - goal: {}",
            goal.chars().take(100).collect::<String>()
        ),
    )
    .map_err(Error::Tool)?;

    let mut lcfg = cfg.clone();
    lcfg.aeris_max_tokens = loop_max_tokens();
    let session = crate::llm::new_session_id();
    let never = AtomicBool::new(false);

    let mut prompt = format!("{SYSTEM}\n\nGOAL: {goal}");
    let mut log: Vec<String> = Vec::new();
    let mut bad_in_row = 0usize;
    let mut last_key = String::new();
    let mut repeats = 0usize;
    let mut finished: Option<String> = None;
    let mut steps_done = 0usize;
    let limit = max_steps();

    for step in 1..=limit {
        steps_done = step;
        let reply = crate::llm::chat(&lcfg, &session, &prompt, &never)?;
        match parse_action(&reply) {
            Action::Done(summary) => {
                finished = Some(summary);
                break;
            }
            Action::Invalid(why) => {
                bad_in_row += 1;
                log.push(format!("[{step}] (bad reply: {why})"));
                if bad_in_row >= 3 {
                    log.push("stopped: model keeps breaking the action format".into());
                    break;
                }
                prompt = format!(
                    "FORMAT ERROR: {why}. Reply with ONE line: TOOL <name> <args>  or  DONE <summary>."
                );
            }
            Action::Tool { name, arg, body } => {
                bad_in_row = 0;
                let key = format!("{name} {arg} {}", body.as_deref().unwrap_or(""));
                if key == last_key {
                    repeats += 1;
                } else {
                    repeats = 0;
                    last_key = key;
                }
                if repeats >= 2 {
                    log.push(format!("[{step}] TOOL {name} {arg}"));
                    log.push("stopped: model repeats the same action".into());
                    break;
                }
                let result = match execute(&name, &arg, body.as_deref(), &lcfg) {
                    Ok(o) => plain(&o),
                    Err(e) => format!("ERROR: {e}"),
                };
                log.push(format!("[{step}] TOOL {name} {arg}"));
                for l in result.lines().take(4) {
                    log.push(format!("      {l}"));
                }
                prompt = format!(
                    "RESULT of step {step}:\n{}\nNext action (one TOOL or DONE)?",
                    clamp(&result, RESULT_CHARS)
                );
            }
        }
    }

    let mut out = format!(
        "│ code loop · {steps_done}/{limit} steps\n│ goal: {}\n│ workspace {}\n",
        goal.chars().take(120).collect::<String>(),
        workspace::root_display()
    );
    for l in &log {
        out.push_str(&format!("│ {l}\n"));
    }
    match finished {
        Some(s) if s.is_empty() => out.push_str("│ DONE"),
        Some(s) => out.push_str(&format!("│ DONE {s}")),
        None => out.push_str("│ not finished (step limit or format errors)"),
    }
    Ok(ToolOutput {
        name: "code",
        body: out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_done() {
        assert_eq!(
            parse_action("DONE: fixed it"),
            Action::Done("fixed it".into())
        );
    }

    #[test]
    fn parses_simple_tool() {
        assert_eq!(
            parse_action("TOOL read src/main.rs 10 50"),
            Action::Tool {
                name: "read".into(),
                arg: "src/main.rs 10 50".into(),
                body: None
            }
        );
    }

    #[test]
    fn skips_chatter_before_tool() {
        match parse_action("Sure, let me look.\nTOOL grep TODO src") {
            Action::Tool { name, arg, .. } => {
                assert_eq!(name, "grep");
                assert_eq!(arg, "TODO src");
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn parses_edit_block_with_pipes() {
        let reply = "TOOL edit src/a.rs\n<<<<\nlet f = |x| x || y;\n====\nlet f = |x| x && y;\n>>>>";
        match parse_action(reply) {
            Action::Tool { name, arg, body } => {
                assert_eq!(name, "edit");
                assert_eq!(arg, "src/a.rs");
                assert_eq!(
                    body.as_deref(),
                    Some("let f = |x| x || y;\n====\nlet f = |x| x && y;")
                );
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[test]
    fn invalid_when_no_action() {
        assert!(matches!(parse_action("hello there"), Action::Invalid(_)));
    }

    #[test]
    fn goal_parsing() {
        assert_eq!(parse_goal("/code fix bug"), Some("fix bug".into()));
        assert_eq!(parse_goal("code: add test"), Some("add test".into()));
        assert_eq!(parse_goal("hello"), None);
    }

    #[test]
    fn clamp_truncates() {
        let s = "a".repeat(50);
        assert!(clamp(&s, 10).contains("truncated"));
        assert_eq!(clamp("abc", 10), "abc");
    }
}
