//! Lightweight diagnostics — company-standard "LSP-like" surface without a full
//! language server dependency (std only).
//!
//! Prefers `cargo check` when a Cargo.toml is in the workspace; otherwise a
//! simple heuristic scan for TODO/FIXME and obvious syntax markers.
//!
//! Commands: /lsp · /lsp check [path]
//! Gate: ZEROZ_LSP≠0 (default on)

use crate::permissions::{self, Cap};
use crate::workspace;
use std::fs;
use std::io::Read;
use std::path::Path;
use std::process::Command;

pub fn diagnose(target: &str) -> String {
    if !permissions::allowed(Cap::Lsp) {
        return "LSP diagnostics denied — set ZEROZ_LSP=1 or unset ZEROZ_LSP=0".into();
    }
    let root = workspace::root();
    let rel = target.trim();
    let path = if rel.is_empty() || rel == "." {
        root.clone()
    } else {
        root.join(rel)
    };

    if root.join("Cargo.toml").is_file() {
        if let Some(out) = cargo_check(&root) {
            return out;
        }
    }
    heuristic_scan(&path)
}

fn cargo_check(root: &Path) -> Option<String> {
    let output = Command::new("cargo")
        .args(["check", "--message-format=short", "-q"])
        .current_dir(root)
        .output()
        .ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut body = String::from("LSP · cargo check\n\n");
    if !stderr.trim().is_empty() {
        for line in stderr.lines().take(40) {
            body.push_str(line);
            body.push('\n');
        }
    }
    if !stdout.trim().is_empty() {
        for line in stdout.lines().take(20) {
            body.push_str(line);
            body.push('\n');
        }
    }
    if output.status.success() && stderr.trim().is_empty() && stdout.trim().is_empty() {
        body.push_str("✔ no diagnostics\n");
    } else if output.status.success() {
        body.push_str("\n✔ cargo check exited 0\n");
    } else {
        body.push_str("\n· cargo check reported issues\n");
    }
    Some(body)
}

fn heuristic_scan(path: &Path) -> String {
    let mut hits: Vec<String> = Vec::new();
    scan_path(path, 0, &mut hits);
    if hits.is_empty() {
        return format!(
            "LSP · heuristic scan\npath: {}\n\n✔ no TODO/FIXME markers found\n",
            path.display()
        );
    }
    let mut out = format!(
        "LSP · heuristic scan\npath: {}\nhits: {}\n\n",
        path.display(),
        hits.len()
    );
    for h in hits.iter().take(40) {
        out.push_str(h);
        out.push('\n');
    }
    out
}

fn scan_path(path: &Path, depth: usize, hits: &mut Vec<String>) {
    if depth > 4 || hits.len() >= 40 {
        return;
    }
    if path.is_file() {
        scan_file(path, hits);
        return;
    }
    let Ok(rd) = fs::read_dir(path) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if name.starts_with('.') || name == "target" || name == "node_modules" {
            continue;
        }
        if p.is_dir() {
            scan_path(&p, depth + 1, hits);
        } else if is_source(name) {
            scan_file(&p, hits);
        }
    }
}

fn is_source(name: &str) -> bool {
    name.ends_with(".rs")
        || name.ends_with(".ts")
        || name.ends_with(".js")
        || name.ends_with(".py")
        || name.ends_with(".go")
        || name.ends_with(".md")
}

fn scan_file(path: &Path, hits: &mut Vec<String>) {
    let mut f = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let mut buf = String::new();
    if f.read_to_string(&mut buf).is_err() {
        return;
    }
    for (i, line) in buf.lines().enumerate() {
        let lower = line.to_lowercase();
        if lower.contains("todo") || lower.contains("fixme") || lower.contains("unimplemented!")
        {
            hits.push(format!(
                "{}:{}: {}",
                path.display(),
                i + 1,
                line.trim()
            ));
            if hits.len() >= 40 {
                return;
            }
        }
    }
}

pub fn handle_command(line: &str) -> String {
    let t = line.trim();
    let lower = t.to_lowercase();
    if lower == "/lsp" || lower == "lsp" {
        return diagnose(".");
    }
    if let Some(path) = lower.strip_prefix("/lsp check ") {
        return diagnose(path.trim());
    }
    if lower == "/lsp check" {
        return diagnose(".");
    }
    "Usage: /lsp · /lsp check [path]".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn heuristic_finds_todo() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("0z_lsp_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("x.rs"), "fn main() { /* TODO: ship */ }\n").unwrap();
        env::set_var("ZEROZ_WORKSPACE", &dir);
        let out = diagnose(".");
        assert!(out.contains("TODO") || out.contains("todo"));
        env::remove_var("ZEROZ_WORKSPACE");
        let _ = fs::remove_dir_all(&dir);
    }
}
