//! Grounding contract — separate *evidence-backed* from *model-only* answers.
//!
//! Most products blend tool output and LLM prose without a clear boundary.
//! 0zcode labels every substantive turn so operators know what is proven.
//!
//! Footer (when enabled):
//!   ── grounding · tools: weather, calc · plane: evidence · tip #a1b2c3d4
//!   ── grounding · plane: model-only · AERIS · no tool evidence this turn
//!
//! Commands: `/grounding` · `/grounding off` is env-only (ZEROZ_GROUNDING=0)
//! Session counters live in-process (not durable).

use std::env;
use std::sync::atomic::{AtomicU64, Ordering};

static TURNS_EVIDENCE: AtomicU64 = AtomicU64::new(0);
static TURNS_MODEL: AtomicU64 = AtomicU64::new(0);
static TURNS_REFUSED: AtomicU64 = AtomicU64::new(0);
static TURNS_META: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plane {
    /// One or more tools / MCP produced the body.
    Evidence,
    /// Free-form model reply, no tool payload this turn.
    ModelOnly,
    /// Policy refusal.
    Refused,
    /// Slash / system meta (help, receipts, buddy, …).
    Meta,
}

impl Plane {
    fn as_str(self) -> &'static str {
        match self {
            Plane::Evidence => "evidence",
            Plane::ModelOnly => "model-only",
            Plane::Refused => "refused",
            Plane::Meta => "meta",
        }
    }
}

fn enabled() -> bool {
    !env::var("ZEROZ_GROUNDING")
        .map(|v| v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

fn tip_hash() -> Option<String> {
    // Best-effort: read last receipt hash from receipts file without full parse API.
    let path = if let Ok(p) = env::var("ZEROZ_RECEIPTS_FILE") {
        std::path::PathBuf::from(p)
    } else {
        let home = env::var("HOME").unwrap_or_else(|_| ".".into());
        std::path::PathBuf::from(home).join(".local/state/0zcode/receipts.jsonl")
    };
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?;
    // "hash":"...."
    let key = "\"hash\":\"";
    let i = line.find(key)?;
    let s = &line[i + key.len()..];
    let end = s.find('"')?;
    let h = &s[..end];
    if h.len() >= 8 {
        Some(h[..8].to_string())
    } else {
        Some(h.to_string())
    }
}

fn footer(plane: Plane, tools: &[&str]) -> String {
    let mut line = format!("── grounding · plane: {}", plane.as_str());
    if !tools.is_empty() {
        line.push_str(" · tools: ");
        line.push_str(&tools.join(", "));
    }
    if plane == Plane::Evidence {
        if let Some(tip) = tip_hash() {
            line.push_str(" · tip #");
            line.push_str(&tip);
        }
    }
    if plane == Plane::ModelOnly {
        line.push_str(" · no tool evidence this turn");
    }
    line
}

/// Attach grounding footer and bump session counters.
pub fn seal(text: String, plane: Plane, tools: &[&str]) -> String {
    match plane {
        Plane::Evidence => {
            TURNS_EVIDENCE.fetch_add(1, Ordering::Relaxed);
        }
        Plane::ModelOnly => {
            TURNS_MODEL.fetch_add(1, Ordering::Relaxed);
        }
        Plane::Refused => {
            TURNS_REFUSED.fetch_add(1, Ordering::Relaxed);
        }
        Plane::Meta => {
            TURNS_META.fetch_add(1, Ordering::Relaxed);
        }
    }
    if !enabled() || text.trim().is_empty() {
        return text;
    }
    // Meta commands stay clean unless user asked for noise
    if plane == Plane::Meta {
        return text;
    }
    format!("{}\n\n{}", text.trim_end(), footer(plane, tools))
}

pub fn handle_command(line: &str) -> Option<String> {
    let lower = line.trim().to_lowercase();
    if lower != "/grounding" && lower != "/grounding status" && !lower.starts_with("/grounding ")
    {
        return None;
    }
    if lower == "/grounding help" || lower.starts_with("/grounding help") {
        return Some(
            "Grounding contract\n\
  Every non-meta answer is labeled:\n\
    plane: evidence     — tool/MCP body\n\
    plane: model-only   — AERIS prose, no tools this turn\n\
    plane: refused      — policy block\n\
  evidence lines may include receipt tip #hash\n\n\
  /grounding          session counters\n\
  ZEROZ_GROUNDING=0   disable footers"
                .into(),
        );
    }
    let e = TURNS_EVIDENCE.load(Ordering::Relaxed);
    let m = TURNS_MODEL.load(Ordering::Relaxed);
    let r = TURNS_REFUSED.load(Ordering::Relaxed);
    let meta = TURNS_META.load(Ordering::Relaxed);
    let total = e + m + r;
    let pct = if total == 0 {
        0
    } else {
        (e * 100) / total
    };
    let tip = tip_hash().unwrap_or_else(|| "—".into());
    Some(format!(
        "Grounding (this process)\n\
  evidence turns:  {e}\n\
  model-only:      {m}\n\
  refused:         {r}\n\
  meta:            {meta}\n\
  evidence ratio:  {pct}% of substantive turns\n\
  receipt tip:     #{tip}\n\
  footers:         {}\n\
/grounding help",
        if enabled() { "on" } else { "off (ZEROZ_GROUNDING=0)" }
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_evidence_adds_footer() {
        std::env::remove_var("ZEROZ_GROUNDING");
        let t = seal("Delhi 32°C".into(), Plane::Evidence, &["weather"]);
        assert!(t.contains("plane: evidence"));
        assert!(t.contains("weather"));
    }

    #[test]
    fn seal_model_marks_no_evidence() {
        std::env::remove_var("ZEROZ_GROUNDING");
        let t = seal("Hello".into(), Plane::ModelOnly, &[]);
        assert!(t.contains("model-only"));
        assert!(t.contains("no tool evidence"));
    }
}
