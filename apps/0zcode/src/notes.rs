//! Sandboxed notes — only under 0zcode data dir. Not a general filesystem tool.

use crate::error::{Error, Result};
use crate::tools::ToolOutput;
use std::env;
use std::fs;
use std::path::PathBuf;

fn notes_dir() -> Result<PathBuf> {
    let base = if let Ok(p) = env::var("ZEROZ_NOTES_DIR") {
        PathBuf::from(p)
    } else {
        let home = env::var("HOME").map_err(|_| Error::Io("HOME not set".into()))?;
        PathBuf::from(home).join(".local/share/0zcode/notes")
    };
    fs::create_dir_all(&base)?;
    Ok(base)
}

fn safe_name(title: &str) -> Result<String> {
    let t = title
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .chars()
        .take(64)
        .collect::<String>();
    if t.is_empty() {
        return Err(Error::Tool("empty note title".into()));
    }
    Ok(t)
}

pub fn handle_notes(msg: &str) -> Result<ToolOutput> {
    let lower = msg.to_lowercase();
    let dir = notes_dir()?;

    if lower.contains("list") || lower.trim() == "notes" || lower == "note list" {
        let mut names = Vec::new();
        if let Ok(rd) = fs::read_dir(&dir) {
            for e in rd.flatten() {
                if e.path().extension().and_then(|s| s.to_str()) == Some("txt") {
                    names.push(e.path().file_stem().unwrap().to_string_lossy().to_string());
                }
            }
        }
        names.sort();
        let body = if names.is_empty() {
            "│ (no notes yet)\n│ try: note save groceries milk".into()
        } else {
            format!("│ {}\n│ dir: {}", names.join(", "), dir.display())
        };
        return Ok(ToolOutput {
            name: "notes",
            body,
        });
    }

    if lower.starts_with("note save ") || lower.starts_with("notes save ") || lower.starts_with("note:") {
        let rest = msg
            .replacen("note save ", "", 1)
            .replacen("notes save ", "", 1)
            .replacen("Note save ", "", 1)
            .replacen("note:", "", 1)
            .trim()
            .to_string();
        let mut parts = rest.splitn(2, char::is_whitespace);
        let title = parts.next().unwrap_or("note");
        let content = parts.next().unwrap_or("").trim();
        let name = safe_name(title)?;
        let path = dir.join(format!("{name}.txt"));
        fs::write(&path, content)?;
        return Ok(ToolOutput {
            name: "notes",
            body: format!("│ saved `{name}`\n│ {}", path.display()),
        });
    }

    if lower.starts_with("note read ") || lower.starts_with("notes read ") {
        let title = msg
            .split_whitespace()
            .skip(2)
            .collect::<Vec<_>>()
            .join(" ");
        let name = safe_name(&title)?;
        let path = dir.join(format!("{name}.txt"));
        let content = fs::read_to_string(&path).map_err(|_| Error::Tool(format!("no note `{name}`")))?;
        return Ok(ToolOutput {
            name: "notes",
            body: format!("│ {name}\n│ {content}"),
        });
    }

    Err(Error::Tool(
        "usage: note list · note save <title> <text> · note read <title>".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn save_and_list() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join("0zcode_notes_test");
        let _ = fs::remove_dir_all(&dir);
        env::set_var("ZEROZ_NOTES_DIR", &dir);
        let o = handle_notes("note save testhello world").unwrap();
        assert!(o.body.contains("saved"));
        let o2 = handle_notes("note list").unwrap();
        assert!(o2.body.contains("testhello"));
        env::remove_var("ZEROZ_NOTES_DIR");
    }
}
