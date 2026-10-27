//! Local sandbox security surface — status, path policy helpers, shell line checks.
//!
//! Execution still lives in `workspace` (fs + allowlisted shell). This module is the
//! company-standard security view and shared validators so path/shell policy stays one place.

use crate::permissions::{self, Cap};
use crate::workspace;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Characters that must never appear in an allowlisted shell line (no shell metacharacters).
const SHELL_META: &[char] = &[
    '|', '&', ';', '`', '$', '(', ')', '<', '>', '\n', '\r', '\0',
];

/// Default workspace when `ZEROZ_WORKSPACE` is unset: XDG data dir, not cwd.
pub fn default_workspace_root() -> PathBuf {
    if let Ok(p) = env::var("ZEROZ_WORKSPACE") {
        let pb = PathBuf::from(p.trim());
        if !pb.as_os_str().is_empty() {
            return pb;
        }
    }
    let base = env::var("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    base.join(".local/share/0zcode/workspace")
}

/// Ensure workspace layout exists (`notes/`, `work/`, `.meta/`) with best-effort private dirs.
pub fn ensure_layout(root: &Path) -> std::io::Result<()> {
    fs::create_dir_all(root)?;
    for sub in ["notes", "work", ".meta"] {
        fs::create_dir_all(root.join(sub))?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(root, fs::Permissions::from_mode(0o700));
        for sub in ["notes", "work", ".meta"] {
            let p = root.join(sub);
            let _ = fs::set_permissions(&p, fs::Permissions::from_mode(0o700));
        }
    }
    Ok(())
}

/// Reject shell lines that contain metacharacters or empty input.
pub fn shell_line_safe(line: &str) -> Result<(), String> {
    let line = line.trim();
    if line.is_empty() {
        return Err("empty command".into());
    }
    if line.chars().any(|c| SHELL_META.contains(&c)) {
        return Err(
            "shell metacharacters refused (| & ; ` $ () <> ) — allowlist only, no pipes"
                .into(),
        );
    }
    if line.contains("sudo ") || line.starts_with("sudo") || line.contains(" doas ") {
        return Err("privilege escalation refused".into());
    }
    Ok(())
}

/// Honest sandbox status for `/sandbox` and ops.
pub fn status_text() -> String {
    let root = workspace::root();
    let _ = ensure_layout(&root);

    let row = |on: bool, label: &str, detail: &str| {
        format!(
            "  {}  {:<14}  {}\n",
            if on { "✔" } else { "·" },
            label,
            detail
        )
    };

    let mut out = String::from("Local sandbox (user machine)\n\n");
    out.push_str(&format!("  root           {}\n", root.display()));
    out.push_str(&format!(
        "  pinned         {}\n",
        if env::var("ZEROZ_WORKSPACE").is_ok() {
            "ZEROZ_WORKSPACE"
        } else {
            "default ~/.local/share/0zcode/workspace"
        }
    ));
    out.push('\n');
    out.push_str(&row(true, "ws.read", "list / read / grep under root"));
    out.push_str(&row(
        permissions::allowed(Cap::Write),
        "ws.write",
        if permissions::allowed(Cap::Write) {
            "ZEROZ_WRITE=1 + approval"
        } else {
            "off — export ZEROZ_WRITE=1"
        },
    ));
    out.push_str(&row(
        permissions::allowed(Cap::Shell),
        "shell",
        if permissions::allowed(Cap::Shell) {
            "ZEROZ_SHELL=1 · allowlist · no metacharacters"
        } else {
            "off — export ZEROZ_SHELL=1"
        },
    ));
    out.push_str(&row(
        permissions::allowed(Cap::Network),
        "network",
        if permissions::allowed(Cap::Network) {
            "HTTP tools"
        } else {
            "denied"
        },
    ));
    out.push_str(&row(
        permissions::allowed(Cap::Computer),
        "computer",
        if permissions::allowed(Cap::Computer) {
            "gated surface"
        } else {
            "off"
        },
    ));
    out.push_str(
        "\nSecurity: path jail · symlink/.. escape refused · shell argv allowlist · caps fail-closed.\n",
    );
    out.push_str("Not a VM: do not set ZEROZ_WORKSPACE=/ or disable approvals casually.\n");
    out.push_str("Commands: /sandbox · /ws · /perms · /approval\n");
    out
}

pub fn handle_command(line: &str) -> String {
    let lower = line.trim().to_lowercase();
    if lower == "/sandbox"
        || lower == "/sandbox status"
        || lower == "sandbox"
        || lower == "/jail"
    {
        return status_text();
    }
    "Usage: /sandbox · /sandbox status\n\nLocal sandbox status (workspace root, caps, shell gate).".into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_metacharacters() {
        assert!(shell_line_safe("ls | cat").is_err());
        assert!(shell_line_safe("echo $(whoami)").is_err());
        assert!(shell_line_safe("ls && rm -rf /").is_err());
        assert!(shell_line_safe("pwd").is_ok());
        assert!(shell_line_safe("ls -la").is_ok());
    }

    #[test]
    fn status_mentions_root_and_caps() {
        let s = status_text();
        assert!(s.contains("Local sandbox"));
        assert!(s.contains("ws.read"));
        assert!(s.contains("shell"));
    }
}
