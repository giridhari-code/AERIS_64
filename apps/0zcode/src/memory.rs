//! Project memory — instructions and durable context (company-standard).
//!
//! Files (under workspace):
//!   .0zcode/MEMORY.md     — durable notes the agent should remember
//!   AGENTS.md / CLAUDE.md — optional project instructions (read-only merge)
//!
//! Commands: /memory · /memory add <note> · /memory clear

use crate::workspace;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

fn memory_path() -> PathBuf {
    workspace::root().join(".0zcode").join("MEMORY.md")
}

fn agents_paths() -> Vec<PathBuf> {
    let root = workspace::root();
    vec![
        root.join("AGENTS.md"),
        root.join("CLAUDE.md"),
        root.join(".0zcode").join("AGENTS.md"),
    ]
}

fn read_if(path: &PathBuf) -> Option<String> {
    fs::read_to_string(path).ok().map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Full memory blob for display.
pub fn load_display() -> String {
    let mut parts: Vec<String> = Vec::new();
    for p in agents_paths() {
        if let Some(body) = read_if(&p) {
            parts.push(format!(
                "## {}\n\n{}",
                p.file_name().and_then(|n| n.to_str()).unwrap_or("instructions"),
                body
            ));
        }
    }
    let mp = memory_path();
    if let Some(body) = read_if(&mp) {
        parts.push(format!("## MEMORY.md\n\n{body}"));
    }
    if parts.is_empty() {
        return "No project memory yet.\n\nAdd notes: /memory add <text>\n\
Or create AGENTS.md / .0zcode/MEMORY.md in the workspace.".into();
    }
    parts.join("\n\n---\n\n")
}

/// Compact context injected into LLM / agent prompts (capped).
pub fn context_for_prompt() -> String {
    let mut out = String::new();
    for p in agents_paths() {
        if let Some(body) = read_if(&p) {
            let clip: String = body.chars().take(2000).collect();
            out.push_str(&clip);
            out.push('\n');
        }
    }
    if let Some(body) = read_if(&memory_path()) {
        let clip: String = body.chars().take(3000).collect();
        out.push_str("Project memory:\n");
        out.push_str(&clip);
    }
    out
}

pub fn append(note: &str) -> Result<String, String> {
    let note = note.trim();
    if note.is_empty() {
        return Err("empty note".into());
    }
    let path = memory_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| e.to_string())?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    writeln!(f, "\n- ({ts}) {note}").map_err(|e| e.to_string())?;
    Ok(format!("Remembered → {}", path.display()))
}

pub fn clear() -> Result<String, String> {
    let path = memory_path();
    if path.is_file() {
        fs::write(&path, "").map_err(|e| e.to_string())?;
        Ok(format!("Cleared {}", path.display()))
    } else {
        Ok("Nothing to clear.".into())
    }
}

pub fn handle_command(line: &str) -> String {
    let t = line.trim();
    let lower = t.to_lowercase();
    if lower == "/memory" || lower == "memory" {
        return load_display();
    }
    if lower.starts_with("/memory add ") || t.starts_with("/memory add ") {
        let note = t.get("/memory add ".len()..).unwrap_or("").trim();
        return append(note).unwrap_or_else(|e| format!("memory error: {e}"));
    }
    if lower == "/memory clear" {
        return clear().unwrap_or_else(|e| format!("memory error: {e}"));
    }
    "Usage: /memory · /memory add <note> · /memory clear".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn append_and_load() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("0z_mem_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        env::set_var("ZEROZ_WORKSPACE", &dir);
        let _ = append("ship company standard");
        let d = load_display();
        assert!(d.contains("ship company standard"));
        let _ = clear();
        env::remove_var("ZEROZ_WORKSPACE");
        let _ = fs::remove_dir_all(&dir);
    }
}
