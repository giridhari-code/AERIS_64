//! Workspace tools — company-standard sandboxed FS ops (read/list/grep/write).
//! Root: `ZEROZ_WORKSPACE` or `~/.local/share/0zcode/workspace` (not cwd).
//! Never escapes root. Shell is allowlist-only behind `Cap::Shell`.

use crate::error::{Error, Result};
use crate::permissions::{self, Cap};
use crate::sandbox;
use crate::tools::ToolOutput;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};

const MAX_READ_BYTES: u64 = 64 * 1024;
const MAX_GREP_HITS: usize = 40;
const MAX_LIST: usize = 80;

pub fn root() -> PathBuf {
    let pb = sandbox::default_workspace_root();
    let _ = sandbox::ensure_layout(&pb);
    if pb.is_dir() {
        return pb;
    }
    // Fallback only if home/XDG unavailable
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

/// Opt-in allowlisted commands. Requires Cap::Shell (`ZEROZ_SHELL=1`) + approval.
/// No shell metacharacters; argv array only (never `sh -c`); cleared env + cwd = workspace.
pub fn shell_allowlisted(cmd_line: &str) -> Result<ToolOutput> {
    if !permissions::allowed(Cap::Shell) {
        return Err(Error::Tool(
            "shell disabled — export ZEROZ_SHELL=1 (allowlist only; /perms deny shell blocks session)"
                .into(),
        ));
    }
    sandbox::shell_line_safe(cmd_line).map_err(Error::Tool)?;
    crate::approval::require(
        crate::approval::ActionClass::Shell,
        &format!("shell `{cmd_line}`"),
    )
    .map_err(Error::Tool)?;
    let line = cmd_line.trim();
    // Exact-line allowlist — no open-ended argv. Prefer read-only / status ops.
    let allowed = [
        "pwd",
        "ls",
        "ls -la",
        "ls -l",
        "date",
        "whoami",
        "df -h",
        "uname -a",
        "git status",
        "git status -sb",
        "git log -5",
        "git log --oneline -10",
        "git branch",
        "cargo --version",
        "rustc --version",
        "cargo check",
        "cargo clippy",
        "cargo test",
        "cargo test -- --nocapture",
    ];
    if !allowed.iter().any(|a| line == *a) {
        return Err(Error::Tool(format!(
            "command not on allowlist: `{line}`\n│ allowed: {}",
            allowed.join(" · ")
        )));
    }
    let root = root();
    let mut cmd = if line == "pwd" {
        Command::new("pwd")
    } else if line.starts_with("ls") {
        let mut c = Command::new("ls");
        for arg in line.split_whitespace().skip(1) {
            c.arg(arg);
        }
        c
    } else if line == "date" {
        Command::new("date")
    } else if line == "whoami" {
        Command::new("whoami")
    } else if line.starts_with("df") {
        let mut c = Command::new("df");
        for arg in line.split_whitespace().skip(1) {
            c.arg(arg);
        }
        c
    } else if line.starts_with("uname") {
        let mut c = Command::new("uname");
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
    };
    cmd.current_dir(&root)
        .env_clear()
        .env("PATH", "/bin:/usr/bin:/usr/local/bin")
        .env("LANG", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let output = cmd.output().map_err(|e| Error::Tool(e.to_string()))?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let mut body = format!("│ $ {line}\n│ cwd {}\n", root.display());
    for l in stdout.lines().take(60) {
        body.push_str(&format!("│ {l}\n"));
    }
    if !stderr.trim().is_empty() {
        body.push_str("│ stderr:\n");
        for l in stderr.lines().take(20) {
            body.push_str(&format!("│ {l}\n"));
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
            "│ workspace {}\n│ read/list/grep: on\n│ write/mkdir: {}\n│ shell allowlist: {}\n│ agent loop: {}\n│ /sandbox for full security status\n│ set ZEROZ_WORKSPACE to pin root",
            root_display(),
            if write_enabled() {
                "ZEROZ_WRITE=1 (on)"
            } else {
                "off (export ZEROZ_WRITE=1)"
            },
            if permissions::allowed(Cap::Shell) {
                "ZEROZ_SHELL=1 (on · no metacharacters)"
            } else {
                "off (export ZEROZ_SHELL=1)"
            },
            if permissions::allowed(Cap::Agent) {
                "ZEROZ_AGENT=1 (on)"
            } else {
                "off (export ZEROZ_AGENT=1)"
            },
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
        let root = root();
        env::set_var("ZEROZ_WORKSPACE", &root);
        let r = resolve_under_root("/etc/passwd");
        assert!(
            r.is_err(),
            "absolute path outside workspace must be refused"
        );
        let r2 = resolve_under_root("../../../../../../etc/passwd");
        assert!(
            r2.is_err(),
            "relative escape outside workspace must be refused"
        );
        env::remove_var("ZEROZ_WORKSPACE");
    }

    #[test]
    fn shell_meta_refused_before_exec() {
        let _g = LOCK.lock().unwrap();
        env::set_var("ZEROZ_SHELL", "1");
        let err = shell_allowlisted("ls | cat").unwrap_err().to_string();
        assert!(err.contains("metacharacter") || err.contains("refused"), "{err}");
        env::remove_var("ZEROZ_SHELL");
    }
}

const MAX_WRITE_BYTES: usize = 256 * 1024;

/// Resolve a path for create/write — parent must stay inside workspace.
pub fn resolve_for_write(rel: &str) -> Result<PathBuf> {
    let root = root();
    let canon_root = root.canonicalize().unwrap_or(root.clone());
    let raw = rel.trim().trim_start_matches("./");
    if raw.is_empty() {
        return Err(Error::Tool("empty path".into()));
    }
    if Path::new(raw).is_absolute() {
        let p = PathBuf::from(raw);
        if let Ok(c) = p.canonicalize() {
            if !c.starts_with(&canon_root) {
                return Err(Error::Tool("path escapes workspace — refused".into()));
            }
            return Ok(c);
        }
        // new file: parent must exist under root
        if let Some(parent) = p.parent() {
            let pc = parent.canonicalize().map_err(|_| {
                Error::Tool("parent dir outside workspace or missing".into())
            })?;
            if !pc.starts_with(&canon_root) {
                return Err(Error::Tool("path escapes workspace — refused".into()));
            }
        }
        return Ok(p);
    }
    let candidate = canon_root.join(raw);
    if let Some(parent) = candidate.parent() {
        if parent.exists() {
            let pc = parent.canonicalize().map_err(|e| Error::Tool(e.to_string()))?;
            if !pc.starts_with(&canon_root) {
                return Err(Error::Tool("path escapes workspace — refused".into()));
            }
        } else {
            // walk up to find existing ancestor under root
            let mut a = parent.to_path_buf();
            while !a.exists() {
                if a == canon_root || !a.starts_with(&canon_root) {
                    break;
                }
                if let Some(p) = a.parent() {
                    a = p.to_path_buf();
                } else {
                    break;
                }
            }
            if !a.starts_with(&canon_root) && a != canon_root {
                return Err(Error::Tool("path escapes workspace — refused".into()));
            }
        }
    }
    Ok(candidate)
}

pub fn write_enabled() -> bool {
    permissions::allowed(Cap::Write)
}

pub fn write_file(path: &str, content: &str) -> Result<ToolOutput> {
    if !write_enabled() {
        return Err(Error::Tool(
            "write disabled — export ZEROZ_WRITE=1 to allow sandboxed writes".into(),
        ));
    }
    crate::approval::require(
        crate::approval::ActionClass::Write,
        &format!("write `{path}` ({} bytes)", content.len()),
    )
    .map_err(Error::Tool)?;
    if content.len() > MAX_WRITE_BYTES {
        return Err(Error::Tool(format!(
            "content too large ({} > {} bytes)",
            content.len(),
            MAX_WRITE_BYTES
        )));
    }
    let file = resolve_for_write(path)?;
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Tool(e.to_string()))?;
        // re-check parent under root after create
        let canon_root = root().canonicalize().unwrap_or_else(|_| root());
        if let Ok(pc) = parent.canonicalize() {
            if !pc.starts_with(&canon_root) {
                return Err(Error::Tool("path escapes workspace — refused".into()));
            }
        }
    }
    fs::write(&file, content).map_err(|e| Error::Tool(e.to_string()))?;
    Ok(ToolOutput {
        name: "ws.write",
        body: format!(
            "│ wrote {} ({} bytes)\n│ workspace {}",
            file.display(),
            content.len(),
            root_display()
        ),
    })
}

pub fn search_replace(path: &str, old: &str, new: &str) -> Result<ToolOutput> {
    if !write_enabled() {
        return Err(Error::Tool(
            "edit disabled — export ZEROZ_WRITE=1".into(),
        ));
    }
    crate::approval::require(
        crate::approval::ActionClass::Write,
        &format!("edit `{path}` (search/replace)"),
    )
    .map_err(Error::Tool)?;
    if old.is_empty() {
        return Err(Error::Tool("search_replace: empty old string".into()));
    }
    let file = resolve_under_root(path).or_else(|_| resolve_for_write(path))?;
    if !file.is_file() {
        return Err(Error::Tool(format!("not a file: {}", file.display())));
    }
    let content = fs::read_to_string(&file).map_err(|e| Error::Tool(e.to_string()))?;
    let count = content.matches(old).count();
    if count == 0 {
        return Err(Error::Tool("search_replace: old string not found".into()));
    }
    let updated = content.replacen(old, new, 1);
    if updated.len() > MAX_WRITE_BYTES {
        return Err(Error::Tool("edit result too large".into()));
    }
    fs::write(&file, &updated).map_err(|e| Error::Tool(e.to_string()))?;
    Ok(ToolOutput {
        name: "ws.edit",
        body: format!(
            "│ edited {}\n│ replaced 1 occurrence ({} matches in file)\n│ {} → {} bytes",
            file.display(),
            count,
            content.len(),
            updated.len()
        ),
    })
}

pub fn mkdir(path: &str) -> Result<ToolOutput> {
    if !write_enabled() {
        return Err(Error::Tool(
            "write disabled — export ZEROZ_WRITE=1".into(),
        ));
    }
    crate::approval::require(
        crate::approval::ActionClass::Write,
        &format!("mkdir `{path}`"),
    )
    .map_err(Error::Tool)?;
    let dir = resolve_for_write(path)?;
    fs::create_dir_all(&dir).map_err(|e| Error::Tool(e.to_string()))?;
    Ok(ToolOutput {
        name: "ws.mkdir",
        body: format!("│ mkdir {}\n│ ok", dir.display()),
    })
}
