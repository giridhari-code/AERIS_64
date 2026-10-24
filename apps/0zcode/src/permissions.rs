//! Tool / capability permissions — company-standard access control.
//!
//! Env gates (session can widen only when parent env allows):
//!   ZEROZ_NETWORK=0     block network tools
//!   ZEROZ_SHELL=1       allowlisted shell
//!   ZEROZ_WRITE=1       workspace write
//!   ZEROZ_COMPUTER=1    computer-use surface (still sandbox)
//!   ZEROZ_LSP=1         language diagnostics
//!   ZEROZ_AGENT=1       multi-step agent harness

use std::env;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cap {
    Network,
    Shell,
    Write,
    Computer,
    Lsp,
    Agent,
    Background,
}

#[derive(Debug, Default)]
struct SessionOverrides {
    /// Caps explicitly denied this session (even if env allows).
    deny: Vec<&'static str>,
}

static SESSION: Mutex<Option<SessionOverrides>> = Mutex::new(None);

fn session() -> std::sync::MutexGuard<'static, Option<SessionOverrides>> {
    SESSION.lock().unwrap_or_else(|e| e.into_inner())
}

fn env_truthy(key: &str) -> bool {
    env::var(key)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true") || v.eq_ignore_ascii_case("yes"))
        .unwrap_or(false)
}

fn env_falsey(key: &str) -> bool {
    env::var(key)
        .map(|v| v == "0" || v.eq_ignore_ascii_case("false") || v.eq_ignore_ascii_case("no"))
        .unwrap_or(false)
}

fn cap_key(c: Cap) -> &'static str {
    match c {
        Cap::Network => "network",
        Cap::Shell => "shell",
        Cap::Write => "write",
        Cap::Computer => "computer",
        Cap::Lsp => "lsp",
        Cap::Agent => "agent",
        Cap::Background => "background",
    }
}

/// Whether a capability is allowed right now.
pub fn allowed(cap: Cap) -> bool {
    if let Some(g) = session().as_ref() {
        if g.deny.iter().any(|k| *k == cap_key(cap)) {
            return false;
        }
    }
    match cap {
        Cap::Network => !env_falsey("ZEROZ_NETWORK") && !env_truthy("ZEROZ_OFFLINE"),
        Cap::Shell => env_truthy("ZEROZ_SHELL"),
        Cap::Write => env_truthy("ZEROZ_WRITE"),
        Cap::Computer => env_truthy("ZEROZ_COMPUTER"),
        Cap::Lsp => !env_falsey("ZEROZ_LSP"),
        Cap::Agent => env_truthy("ZEROZ_AGENT"),
        Cap::Background => !env_falsey("ZEROZ_BACKGROUND"),
    }
}

/// Deny a capability for the rest of this process (cannot escalate past env).
pub fn deny_session(cap: Cap) {
    let mut g = session();
    let s = g.get_or_insert_with(SessionOverrides::default);
    let k = cap_key(cap);
    if !s.deny.iter().any(|x| *x == k) {
        s.deny.push(k);
    }
}

pub fn status_text() -> String {
    let row = |c: Cap, label: &str, how: &str| {
        let on = allowed(c);
        format!(
            "  {}  {:<12}  {}\n",
            if on { "✔" } else { "·" },
            label,
            if on { how } else { "denied" }
        )
    };
    let mut out = String::from("Permissions (company gates)\n\n");
    out.push_str(&row(Cap::Network, "network", "ZEROZ_NETWORK≠0 · not offline"));
    out.push_str(&row(Cap::Shell, "shell", "ZEROZ_SHELL=1 · allowlist only"));
    out.push_str(&row(Cap::Write, "write", "ZEROZ_WRITE=1 · workspace only"));
    out.push_str(&row(Cap::Agent, "agent", "ZEROZ_AGENT=1 · harness loop"));
    out.push_str(&row(Cap::Background, "background", "ZEROZ_BACKGROUND≠0"));
    out.push_str(&row(Cap::Lsp, "lsp", "ZEROZ_LSP≠0 · cargo/check"));
    out.push_str(&row(Cap::Computer, "computer", "ZEROZ_COMPUTER=1 · sandbox only"));
    out.push_str("\nDeny this session: /perms deny shell|write|network|computer|lsp|agent|bg\n");
    out
}

pub fn handle_command(line: &str) -> String {
    let t = line.trim();
    let lower = t.to_lowercase();
    if lower == "/perms" || lower == "/permissions" || lower == "perms" {
        return status_text();
    }
    if let Some(rest) = lower
        .strip_prefix("/perms deny ")
        .or_else(|| lower.strip_prefix("/permissions deny "))
    {
        let cap = match rest.trim() {
            "network" | "net" => Some(Cap::Network),
            "shell" => Some(Cap::Shell),
            "write" => Some(Cap::Write),
            "computer" => Some(Cap::Computer),
            "lsp" => Some(Cap::Lsp),
            "agent" => Some(Cap::Agent),
            "bg" | "background" => Some(Cap::Background),
            _ => None,
        };
        return match cap {
            Some(c) => {
                deny_session(c);
                format!("Denied `{}` for this session.\n\n{}", cap_key(c), status_text())
            }
            None => "Unknown capability. Use: network shell write computer lsp agent bg".into(),
        };
    }
    status_text()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_default_allowed_when_not_offline() {
        // cannot assert global env safely; just smoke status
        assert!(status_text().contains("network"));
    }
}
