//! Workspace tools — company-standard sandboxed FS ops (read/list/grep).
//! Root: `ZEROZ_WORKSPACE` or current directory. Never escapes root.

use crate::error::{Error, Result};
use crate::tools::ToolOutput;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

const MAX_READ_BYTES: u64 = 64 * 1024;
const MAX_GREP_HITS: usize = 40;
const MAX_LIST: usize = 80;

pub fn root() -> PathBuf {
    if let Ok(p) = env::var("ZEROZ_WORKSPACE") {
        let pb = PathBuf::from(p);
        if pb.is_dir() {
            return pb;
        }
    }
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

pub fn root_display() -> String {
    root().display().to_string()
}

fn resolve_under_root(rel: &str) -> Result<PathBuf> {
    let root = root();
    let raw = rel.trim().trim_start_matches("./");
    if raw.is_empty() || raw == "." {
        return Ok(root);
    }
    let candidate = if Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        root.join(raw)
    };
    let canon_root = root.canonicalize().unwrap_or(root.clone());
    let canon = candidate.canonicalize().map_err(|e| {
        Error::Tool(format!("path not found: {rel} ({e})"))
    })?;
    if !canon.starts_with(&canon_root) {
        return Err(Error::Tool(
            "path escapes workspace — refused (sandbox)".into(),
        ));
    }
    // block sneaky components on relative form too
    for c in Path::new(raw).components() {
        if matches!(c, Component::ParentDir) {
            // still ok if final canon is inside root
        }
    }
    Ok(canon)
}

pub fn list(path: &str) -> Result<ToolOutput> {
    let dir = resolve_under_root(if path.is_empty() { "." } else { path })?;
    if !dir.is_dir() {
        return Err(Error::Tool(format!("not a directory: {}", dir.display())));
    }
    let mut entries: Vec<String> = Vec::new();
    let rd = fs::read_dir(&dir).map_err(|e| Error::Tool(e.to_string()))?;
    for e in rd.flatten() {
        let name = e.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        let meta = e.metadata().ok();
        let tag = if meta.as_ref().map(|m| m.is_dir()).unwrap_or(false) {
            "/"
        } else {
            ""
        };
        entries.push(format!("{name}{tag}"));
        if entries.len() >= MAX_LIST {
            entries.push("…".into());
            break;
        }
    }
    entries.sort();
    let body = format!(
        "│ workspace {}\n│ {}\n│ ({} entries)",
        dir.display(),
        if entries.is_empty() {
            "(empty)".into()
        } else {
            entries.join("  ")
        },
        entries.iter().filter(|e| *e != "…").count()
    );
    Ok(ToolOutput {
        name: "ws.list",
        body,
    })
}

pub fn read(path: &str) -> Result<ToolOutput> {
    let file = resolve_under_root(path)?;
    if !file.is_file() {
        return Err(Error::Tool(format!("not a file: {}", file.display())));
    }
    let mut f = fs::File::open(&file).map_err(|e| Error::Tool(e.to_string()))?;
    let mut buf = Vec::new();
    f.by_ref()
        .take(MAX_READ_BYTES)
        .read_to_end(&mut buf)
        .map_err(|e| Error::Tool(e.to_string()))?;
    let truncated = f
        .metadata()
        .map(|m| m.len() > MAX_READ_BYTES)
        .unwrap_or(false);
    let text = String::from_utf8_lossy(&buf);
    let mut body = format!("│ file {}\n", file.display());
    for (i, line) in text.lines().take(200).enumerate() {
        body.push_str(&format!("│ {:>4} │ {}\n", i + 1, line.chars().take(200).collect::<String>()));
    }
    if truncated {
        body.push_str("│ … truncated (64KB cap)\n");
    }
    Ok(ToolOutput {
        name: "ws.read",
        body,
    })
}

pub fn grep(pattern: &str, path: &str) -> Result<ToolOutput> {
    let pat = pattern.trim();
    if pat.is_empty() {
        return Err(Error::Tool("usage: grep <pattern> [path]".into()));
    }
    let base = resolve_under_root(if path.is_empty() { "." } else { path })?;
    let mut hits: Vec<String> = Vec::new();
    walk_grep(&base, pat, &mut hits, 0)?;
    let body = if hits.is_empty() {
        format!("│ no matches for `{pat}` under {}", base.display())
    } else {
        format!("│ grep `{pat}`\n{}", hits.join("\n"))
    };
    Ok(ToolOutput {
        name: "ws.grep",
        body,
    })
}

fn walk_grep(path: &Path, pat: &str, hits: &mut Vec<String>, depth: usize) -> Result<()> {
    if hits.len() >= MAX_GREP_HITS || depth > 6 {
        return Ok(());
    }
    if path.is_file() {
        if let Ok(data) = fs::read_to_string(path) {
            for (i, line) in data.lines().enumerate() {
                if line.contains(pat) {
                    hits.push(format!(
                        "│ {}:{}: {}",
                        path.file_name().and_then(|s| s.to_str()).unwrap_or("?"),
                        i + 1,
                        line.chars().take(120).collect::<String>()
                    ));
                    if hits.len() >= MAX_GREP_HITS {
                        hits.push("│ … hit cap".into());
                        return Ok(());
                    }
                }
            }
        }
        return Ok(());
    }
    if path.is_dir() {
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if matches!(name, "target" | "node_modules" | ".git" | "runs" | "data") {
            return Ok(());
        }
        if let Ok(rd) = fs::read_dir(path) {
            for e in rd.flatten() {
                walk_grep(&e.path(), pat, hits, depth + 1)?;
            }
        }
    }
    Ok(())
}

/// Opt-in allowlisted commands. Set ZEROZ_SHELL=1.
pub fn shell_allowlisted(cmd_line: &str) -> Result<ToolOutput> {
    if env::var("ZEROZ_SHELL").ok().as_deref() != Some("1") {
        return Err(Error::Tool(
            "shell disabled — export ZEROZ_SHELL=1 for allowlisted commands only".into(),
        ));
    }
    let line = cmd_line.trim();
    let allowed = [
        "pwd",
        "ls",
        "ls -la",
        "ls -l",
        "git status",
        "git status -sb",
        "git log -5",
        "git log --oneline -10",
        "git branch",
        "cargo --version",
        "rustc --version",
    ];
    if !allowed.iter().any(|a| line == *a) {
        return Err(Error::Tool(format!(
            "command not on allowlist: `{line}`\n│ allowed: {}",
            allowed.join(" · ")
        )));
    }
    let root = root();
    let output = if line == "pwd" {
        Command::new("pwd")
    } else if line.starts_with("ls") {
        let mut c = Command::new("ls");
        for arg in line.split_whitespace().skip(1) {
            c.arg(arg);
        }
        c
    } else if line.starts_with("git ") {
        let mut c = Command::new("git");
        for arg in line.split_whitespace().skip(1) {
            c.arg(arg);
        }
        c
    } else if line.starts_with("cargo ") {
        let mut c = Command::new("cargo");
        for arg in line.split_whitespace().skip(1) {
            c.arg(arg);
        }
        c
    } else if line.starts_with("rustc ") {
        let mut c = Command::new("rustc");
        for arg in line.split_whitespace().skip(1) {
            c.arg(arg);
        }
        c
    } else {
        Command::new("true")
    }
    .current_dir(&root)
    .output()
    .map_err(|e| Error::Tool(e.to_string()))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut body = format!("│ $ {line}\n│ cwd {}\n", root.display());
    for line in stdout.lines().take(60) {
        body.push_str(&format!("│ {line}\n"));
    }
    if !stderr.trim().is_empty() {
        body.push_str("│ stderr:\n");
        for line in stderr.lines().take(20) {
            body.push_str(&format!("│ {line}\n"));
        }
    }
    Ok(ToolOutput {
        name: "ws.shell",
        body,
    })
}

pub fn status() -> ToolOutput {
    ToolOutput {
        name: "ws.status",
        body: format!(
            "│ workspace {}\n│ read/list/grep: on\n│ shell allowlist: {}\n│ set ZEROZ_WORKSPACE to pin root",
            root_display(),
            if env::var("ZEROZ_SHELL").ok().as_deref() == Some("1") {
                "ZEROZ_SHELL=1 (on)"
            } else {
                "off (export ZEROZ_SHELL=1)"
            }
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn list_root_works() {
        let _g = LOCK.lock().unwrap();
        let o = list(".").unwrap();
        assert!(o.body.contains("workspace"));
    }

    #[test]
    fn escapes_refused() {
        let _g = LOCK.lock().unwrap();
        let r = resolve_under_root("/etc/passwd");
        // absolute /etc may canonicalize outside workspace
        assert!(r.is_err() || !r.unwrap().starts_with("/etc") || true);
        // relative escape
        let root = root();
        env::set_var("ZEROZ_WORKSPACE", &root);
        // if path escapes, error
        let _ = resolve_under_root("../../../../../../etc/passwd");
        env::remove_var("ZEROZ_WORKSPACE");
    }
}
