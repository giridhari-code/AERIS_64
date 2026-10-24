//! Human Approval — gate sensitive actions until a person confirms.
//!
//! Company-standard (Claude Code / OpenCode style):
//!   write · shell · agent · computer · external MCP
//!
//! Policy (`ZEROZ_APPROVAL`):
//!   ask   — prompt on each sensitive class (default for write/shell/agent/computer)
//!   auto  — allow when env capability already enabled (no prompt)
//!   deny  — always refuse sensitive actions
//!
//! Session memory after **a** (always): class is allowed for the rest of the process.
//! Commands: /approval · /approval auto|ask|deny · /approval reset

use std::env;
use std::io::{self, IsTerminal, Write};
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionClass {
    Write,
    Shell,
    Agent,
    Computer,
    Mcp,
    Network,
}

impl ActionClass {
    pub fn as_str(self) -> &'static str {
        match self {
            ActionClass::Write => "write",
            ActionClass::Shell => "shell",
            ActionClass::Agent => "agent",
            ActionClass::Computer => "computer",
            ActionClass::Mcp => "mcp",
            ActionClass::Network => "network",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "write" | "edit" | "mkdir" => Some(Self::Write),
            "shell" | "bash" | "exec" => Some(Self::Shell),
            "agent" | "harness" => Some(Self::Agent),
            "computer" => Some(Self::Computer),
            "mcp" | "plugin" => Some(Self::Mcp),
            "network" | "net" | "http" => Some(Self::Network),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    Ask,
    Auto,
    Deny,
}

impl Policy {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "ask" | "prompt" | "human" => Some(Self::Ask),
            "auto" | "allow" | "yes" => Some(Self::Auto),
            "deny" | "never" | "no" => Some(Self::Deny),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Policy::Ask => "ask",
            Policy::Auto => "auto",
            Policy::Deny => "deny",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
}

#[derive(Default)]
struct State {
    /// Process-level policy override (from /approval or env at first read).
    policy: Option<Policy>,
    /// Classes approved with "always" this session.
    always: Vec<&'static str>,
}

static STATE: Mutex<State> = Mutex::new(State {
    policy: None,
    always: Vec::new(),
});

fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn policy() -> Policy {
    if let Some(p) = state().policy {
        return p;
    }
    if let Ok(s) = env::var("ZEROZ_APPROVAL") {
        if let Some(p) = Policy::parse(&s) {
            return p;
        }
    }
    // Safe default: ask a human for high-impact actions.
    Policy::Ask
}

pub fn set_policy(p: Policy) {
    state().policy = Some(p);
}

pub fn reset_session() {
    let mut g = state();
    g.always.clear();
}

fn session_always(class: ActionClass) -> bool {
    state()
        .always
        .iter()
        .any(|k| *k == class.as_str())
}

fn remember_always(class: ActionClass) {
    let mut g = state();
    let k = class.as_str();
    if !g.always.iter().any(|x| *x == k) {
        g.always.push(k);
    }
}

/// Whether this class is treated as requiring human approval under `ask`.
pub fn is_sensitive(class: ActionClass) -> bool {
    matches!(
        class,
        ActionClass::Write
            | ActionClass::Shell
            | ActionClass::Agent
            | ActionClass::Computer
            | ActionClass::Mcp
    )
}

/// Core gate. Returns Ok(()) if the action may proceed.
pub fn require(class: ActionClass, summary: &str) -> Result<(), String> {
    if session_always(class) {
        return Ok(());
    }
    match policy() {
        Policy::Auto => Ok(()),
        Policy::Deny => Err(format!(
            "Human Approval: `{}` denied by policy (ZEROZ_APPROVAL=deny).\n  {}",
            class.as_str(),
            summary
        )),
        Policy::Ask => {
            if !is_sensitive(class) {
                // Network etc. only gated when explicitly sensitive elsewhere.
                return Ok(());
            }
            prompt_human(class, summary)
        }
    }
}

fn prompt_human(class: ActionClass, summary: &str) -> Result<(), String> {
    let stdin = io::stdin();
    if !stdin.is_terminal() || !io::stdout().is_terminal() {
        return Err(format!(
            "Human Approval required for `{}` but no TTY.\n  {}\n\
Set ZEROZ_APPROVAL=auto to skip prompts, or run interactively and answer y/a/n.",
            class.as_str(),
            summary
        ));
    }
    let mut out = io::stdout();
    let _ = writeln!(
        out,
        "\n  ⚠  Human Approval — {}\n  {}\n  [y] once  ·  [a] always this session  ·  [n] deny",
        class.as_str(),
        summary
    );
    let _ = write!(out, "  approve? ");
    let _ = out.flush();

    let mut line = String::new();
    if stdin.read_line(&mut line).is_err() {
        return Err("Human Approval: failed to read response".into());
    }
    let ans = line.trim().to_lowercase();
    match ans.as_str() {
        "y" | "yes" => Ok(()),
        "a" | "always" => {
            remember_always(class);
            Ok(())
        }
        "n" | "no" | "" => Err(format!(
            "Human Approval: `{}` denied by user.\n  {}",
            class.as_str(),
            summary
        )),
        other => Err(format!(
            "Human Approval: unknown answer `{other}` (use y / a / n)"
        )),
    }
}

pub fn status_text() -> String {
    let p = policy();
    let always = state().always.clone();
    let mut out = String::from("Human Approval\n\n");
    out.push_str(&format!("  policy:   {} (ZEROZ_APPROVAL / /approval)\n", p.as_str()));
    out.push_str("  gated:    write · shell · agent · computer · mcp\n");
    if always.is_empty() {
        out.push_str("  session:  (no always-allow yet)\n");
    } else {
        out.push_str(&format!("  session:  always → {}\n", always.join(", ")));
    }
    out.push_str(
        "\nCommands:\n\
  /approval              status\n\
  /approval ask|auto|deny\n\
  /approval reset        clear session always-allow\n\
\nPrompts: [y] once · [a] always this session · [n] deny\n",
    );
    out
}

pub fn handle_command(line: &str) -> String {
    let t = line.trim();
    let lower = t.to_lowercase();
    if lower == "/approval" || lower == "/approve" || lower == "approval" {
        return status_text();
    }
    if lower == "/approval reset" || lower == "/approve reset" {
        reset_session();
        return format!("Session always-allow cleared.\n\n{}", status_text());
    }
    if let Some(rest) = lower
        .strip_prefix("/approval ")
        .or_else(|| lower.strip_prefix("/approve "))
    {
        if let Some(p) = Policy::parse(rest) {
            set_policy(p);
            return format!("Human Approval policy → {}.\n\n{}", p.as_str(), status_text());
        }
        return format!("Unknown policy `{rest}`.\n\n{}", status_text());
    }
    status_text()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_policy_blocks() {
        set_policy(Policy::Deny);
        reset_session();
        let r = require(ActionClass::Write, "write notes/x.md");
        assert!(r.is_err());
        set_policy(Policy::Auto);
    }

    #[test]
    fn auto_policy_allows() {
        set_policy(Policy::Auto);
        reset_session();
        assert!(require(ActionClass::Shell, "cargo test").is_ok());
    }

    #[test]
    fn always_skips_second_check() {
        set_policy(Policy::Deny);
        reset_session();
        remember_always(ActionClass::Write);
        assert!(require(ActionClass::Write, "write x").is_ok());
        reset_session();
        set_policy(Policy::Auto);
    }
}
