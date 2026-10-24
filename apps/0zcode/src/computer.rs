//! Computer Use — gated desktop/browser actions (company-standard).
//!
//! Full OS control is intentionally **not** implemented. This module exposes a
//! clear status surface and a narrow, workspace-bound capability checklist so
//! product integrations can be enabled later without weakening policy.
//!
//! Gate: ZEROZ_COMPUTER=1
//! Commands: /computer · /computer status

use crate::permissions::{self, Cap};
use crate::workspace;

pub fn status_text() -> String {
    let enabled = permissions::allowed(Cap::Computer);
    let mut out = String::from("Computer Use (company policy)\n\n");
    out.push_str(&format!(
        "  status:  {}\n",
        if enabled {
            "gate open (ZEROZ_COMPUTER=1)"
        } else {
            "closed — export ZEROZ_COMPUTER=1 to acknowledge"
        }
    ));
    out.push_str(&format!("  workspace: {}\n", workspace::root_display()));
    out.push_str(
        "\nSupported later (not active in this build):\n\
  · screenshot capture (user-approved)\n\
  · focused window metadata\n\
  · accessibility click/type within allowlisted apps\n\
\n\
Still refused by policy:\n\
  · arbitrary shell / sudo\n\
  · credential store access\n\
  · code generation / silent file overwrite outside workspace\n\
\n\
Today use: workspace tools (/ws), agent harness (ZEROZ_AGENT=1), MCP integrations.\n",
    );
    out
}

pub fn handle_command(line: &str) -> String {
    let lower = line.trim().to_lowercase();
    if lower == "/computer" || lower == "/computer status" || lower == "computer" {
        return status_text();
    }
    "Usage: /computer · /computer status\n\nComputer Use is gated and not fully enabled in this build.".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_mentions_gate() {
        assert!(status_text().contains("ZEROZ_COMPUTER"));
    }
}
