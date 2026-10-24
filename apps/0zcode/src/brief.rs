//! Session brief — operator handoff from receipts + grounding + buddy.
//!
//! Coding agents leave a chat scroll; ops needs a **one-screen shift brief**:
//! what ran, what failed, evidence ratio, companion, chain tip.
//!
//! `/brief` · `/handoff` (alias)

use crate::buddy;
use crate::grounding;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn receipts_path() -> PathBuf {
    if let Ok(p) = env::var("ZEROZ_RECEIPTS_FILE") {
        if !p.is_empty() {
            return PathBuf::from(p);
        }
    }
    let home = env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".local/state/0zcode/receipts.jsonl")
}

fn now_ist_line() -> String {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let secs = (ms / 1000) as i64 + 5 * 3600 + 30 * 60;
    let s = secs.rem_euclid(86400) as u32;
    let h = s / 3600;
    let m = (s % 3600) / 60;
    format!("{h:02}:{m:02} IST")
}

fn scan_receipts() -> (u32, u32, Vec<String>) {
    let Ok(text) = fs::read_to_string(receipts_path()) else {
        return (0, 0, Vec::new());
    };
    let mut ok = 0u32;
    let mut fail = 0u32;
    let mut tools: Vec<String> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.contains("\"ok\":\"true\"") || line.contains("\"ok\":true") {
            ok += 1;
        } else {
            fail += 1;
        }
        if let Some(t) = field(line, "tool") {
            if !tools.iter().any(|x| x == &t) {
                tools.push(t);
            }
        }
    }
    tools.truncate(12);
    (ok, fail, tools)
}

fn field(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let i = line.find(&pat)?;
    let s = &line[i + pat.len()..];
    let end = s.find('"')?;
    Some(s[..end].to_string())
}

fn tip() -> String {
    let Ok(text) = fs::read_to_string(receipts_path()) else {
        return "—".into();
    };
    for line in text.lines().rev() {
        if let Some(h) = field(line, "hash") {
            return if h.len() >= 8 {
                h[..8].to_string()
            } else {
                h
            };
        }
    }
    "—".into()
}

/// Build handoff text.
pub fn render() -> String {
    let (ok, fail, tools) = scan_receipts();
    let total = ok + fail;
    let g = grounding::handle_command("/grounding").unwrap_or_default();
    // parse evidence ratio line if present
    let ratio_line = g
        .lines()
        .find(|l| l.contains("evidence ratio"))
        .unwrap_or("  evidence ratio:  n/a")
        .trim();

    let buddy_line = {
        let c = buddy::status_chip();
        if c.is_empty() {
            "buddy: off".into()
        } else {
            format!("buddy: {c}")
        }
    };

    let tool_line = if tools.is_empty() {
        "tools seen: (none in receipts yet)".into()
    } else {
        format!("tools seen: {}", tools.join(", "))
    };

    let offline = env::var("ZEROZ_OFFLINE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    format!(
        "Session brief · {}\n\
────────────────────────\n\
receipts: {ok} ok · {fail} fail · {total} total\n\
{ratio_line}\n\
receipt tip: #{}\n\
{tool_line}\n\
{buddy_line}\n\
network: {}\n\
────────────────────────\n\
Handoff: run /receipts verify · /grounding · continue with same ZEROZ_* env.\n\
/brief is local ops status — not a model summary.",
        now_ist_line(),
        tip(),
        if offline {
            "OFFLINE (ZEROZ_OFFLINE)"
        } else {
            "online allowed"
        }
    )
}

pub fn handle_command(line: &str) -> Option<String> {
    let lower = line.trim().to_lowercase();
    if lower == "/brief"
        || lower == "/handoff"
        || lower == "brief"
        || lower == "handoff"
        || lower == "/brief help"
    {
        if lower == "/brief help" {
            return Some(
                "Session brief\n\
  /brief     one-screen handoff (receipts + grounding + buddy)\n\
  /handoff   alias\n\
Built from local evidence files — no LLM call."
                    .into(),
            );
        }
        return Some(render());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn brief_renders_without_panic() {
        let t = render();
        assert!(t.contains("Session brief"));
        assert!(t.contains("receipts:"));
    }
}
