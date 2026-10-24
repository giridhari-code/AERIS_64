//! Project rules — company-standard policy files the agent must respect.
//!
//! Location: `{workspace}/.0zcode/rules/*.md` (also `.txt`)
//! Optional single file: `{workspace}/.0zcode/RULES.md`
//!
//! Commands: /rules · /rules show <name>

use crate::workspace;
use std::fs;
use std::path::PathBuf;

fn rules_dir() -> PathBuf {
    workspace::root().join(".0zcode").join("rules")
}

fn rules_root_file() -> PathBuf {
    workspace::root().join(".0zcode").join("RULES.md")
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub name: String,
    pub body: String,
}

pub fn load_all() -> Vec<Rule> {
    let mut out = Vec::new();
    let root = rules_root_file();
    if let Ok(body) = fs::read_to_string(&root) {
        let body = body.trim().to_string();
        if !body.is_empty() {
            out.push(Rule {
                name: "RULES.md".into(),
                body,
            });
        }
    }
    let dir = rules_dir();
    if let Ok(rd) = fs::read_dir(&dir) {
        let mut files: Vec<_> = rd.flatten().collect();
        files.sort_by_key(|e| e.file_name());
        for e in files {
            let path = e.path();
            let ext = path
                .extension()
                .and_then(|x| x.to_str())
                .unwrap_or("")
                .to_lowercase();
            if ext != "md" && ext != "txt" {
                continue;
            }
            if let Ok(body) = fs::read_to_string(&path) {
                let body = body.trim().to_string();
                if body.is_empty() {
                    continue;
                }
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("rule")
                    .to_string();
                out.push(Rule { name, body });
            }
        }
    }
    out
}

/// Compact rules text for prompt injection (capped).
pub fn context_for_prompt() -> String {
    let rules = load_all();
    if rules.is_empty() {
        return String::new();
    }
    let mut out = String::from("Project rules (must follow):\n");
    for r in rules {
        let clip: String = r.body.chars().take(1500).collect();
        out.push_str(&format!("### {}\n{}\n\n", r.name, clip));
    }
    out
}

pub fn list_text() -> String {
    let rules = load_all();
    if rules.is_empty() {
        return "No rules configured.\n\nAdd markdown under `.0zcode/rules/` or `.0zcode/RULES.md`.\n\
Example: .0zcode/rules/security.md".into();
    }
    let mut out = format!("Rules ({} files)\n\n", rules.len());
    for r in &rules {
        let preview: String = r.body.chars().take(120).collect();
        out.push_str(&format!("• {} — {}…\n", r.name, preview.replace('\n', " ")));
    }
    out.push_str("\nShow one: /rules show <name>\n");
    out
}

pub fn handle_command(line: &str) -> String {
    let t = line.trim();
    let lower = t.to_lowercase();
    if lower == "/rules" || lower == "rules" {
        return list_text();
    }
    if let Some(name) = lower.strip_prefix("/rules show ") {
        let name = name.trim();
        for r in load_all() {
            if r.name.eq_ignore_ascii_case(name)
                || r.name
                    .trim_end_matches(".md")
                    .eq_ignore_ascii_case(name)
            {
                return format!("# {}\n\n{}", r.name, r.body);
            }
        }
        return format!("Rule `{name}` not found.\n\n{}", list_text());
    }
    "Usage: /rules · /rules show <name>".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn loads_rules_dir() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("0z_rules_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join(".0zcode/rules")).unwrap();
        fs::write(
            dir.join(".0zcode/rules/security.md"),
            "Never exfiltrate secrets.",
        )
        .unwrap();
        env::set_var("ZEROZ_WORKSPACE", &dir);
        let all = load_all();
        assert_eq!(all.len(), 1);
        assert!(all[0].body.contains("secrets"));
        env::remove_var("ZEROZ_WORKSPACE");
        let _ = fs::remove_dir_all(&dir);
    }
}
