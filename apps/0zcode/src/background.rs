//! Background tasks — run long work without blocking the TUI (company-standard).
//!
//! Store: `{workspace}/.0zcode/background/<id>.json`
//! Commands: /bg · /bg run <label> · /bg status <id> · /bg cancel <id>
//!
//! Note: jobs are local process threads recording status; not a cluster queue.

use crate::permissions::{self, Cap};
use crate::workspace;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

fn dir() -> PathBuf {
    workspace::root().join(".0zcode").join("background")
}

fn now_id() -> String {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("bg{t}")
}

fn write_status(id: &str, status: &str, detail: &str) {
    let dir = dir();
    let _ = fs::create_dir_all(&dir);
    let path = dir.join(format!("{id}.json"));
    let body = format!(
        "{{\"id\":\"{}\",\"status\":\"{}\",\"detail\":\"{}\"}}\n",
        id,
        status,
        detail.replace('\\', "\\\\").replace('"', "\\\"")
    );
    let _ = fs::write(path, body);
}

pub fn list_text() -> String {
    let dir = dir();
    let Ok(rd) = fs::read_dir(&dir) else {
        return "No background tasks.\n\nStart: /bg run <label>\n".into();
    };
    let mut rows = Vec::new();
    for e in rd.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        if let Ok(text) = fs::read_to_string(&path) {
            rows.push(text.trim().to_string());
        }
    }
    if rows.is_empty() {
        return "No background tasks.\n\nStart: /bg run <label>\n".into();
    }
    rows.sort();
    let mut out = format!("Background tasks ({})\n\n", rows.len());
    for r in rows.iter().rev().take(20) {
        out.push_str("  ");
        out.push_str(r);
        out.push('\n');
    }
    out.push_str("\n/bg status <id> · /bg cancel <id>\n");
    out
}

/// Spawn a lightweight background marker job (demo-safe, no shell by default).
pub fn spawn(label: &str) -> Result<String, String> {
    if !permissions::allowed(Cap::Background) {
        return Err("background denied — set ZEROZ_BACKGROUND≠0".into());
    }
    let label = label.trim();
    if label.is_empty() {
        return Err("usage: /bg run <label>".into());
    }
    let id = now_id();
    write_status(&id, "running", label);
    let id_t = id.clone();
    let label_t = label.to_string();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel_flag = cancel.clone();
    // store cancel flag path
    let flag_path = dir().join(format!("{id}.cancel"));
    thread::spawn(move || {
        for _ in 0..10 {
            if cancel_flag.load(Ordering::Relaxed) || flag_path.is_file() {
                write_status(&id_t, "cancelled", &label_t);
                let _ = fs::remove_file(&flag_path);
                return;
            }
            thread::sleep(std::time::Duration::from_millis(200));
        }
        write_status(&id_t, "done", &format!("completed: {label_t}"));
    });
    Ok(format!(
        "Background task `{id}` started.\nlabel: {label}\n\n{}",
        list_text()
    ))
}

pub fn cancel(id: &str) -> String {
    let id = id.trim();
    let dir = dir();
    let flag = dir.join(format!("{id}.cancel"));
    let _ = fs::create_dir_all(&dir);
    let _ = fs::File::create(&flag).and_then(|mut f| writeln!(f, "1"));
    write_status(id, "cancelling", "cancel requested");
    format!("Cancel requested for `{id}`.\n\n{}", list_text())
}

pub fn status(id: &str) -> String {
    let path = dir().join(format!("{}.json", id.trim()));
    match fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => format!("Unknown id `{}`.\n\n{}", id, list_text()),
    }
}

pub fn handle_command(line: &str) -> String {
    let t = line.trim();
    let lower = t.to_lowercase();
    if lower == "/bg" || lower == "/background" {
        return list_text();
    }
    if let Some(label) = lower
        .strip_prefix("/bg run ")
        .or_else(|| lower.strip_prefix("/background run "))
    {
        let label = t
            .get(line.len().saturating_sub(label.len())..)
            .map(|_| {
                // recover original case label
                if let Some(i) = lower.find("/bg run ") {
                    t[i + 8..].trim()
                } else if let Some(i) = lower.find("/background run ") {
                    t[i + 16..].trim()
                } else {
                    label.trim()
                }
            })
            .unwrap_or(label.trim());
        return spawn(label).unwrap_or_else(|e| format!("bg error: {e}"));
    }
    if let Some(id) = lower
        .strip_prefix("/bg status ")
        .or_else(|| lower.strip_prefix("/background status "))
    {
        return status(id.trim());
    }
    if let Some(id) = lower
        .strip_prefix("/bg cancel ")
        .or_else(|| lower.strip_prefix("/background cancel "))
    {
        return cancel(id.trim());
    }
    "Usage: /bg · /bg run <label> · /bg status <id> · /bg cancel <id>".into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn spawn_lists() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("0z_bg_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        env::set_var("ZEROZ_WORKSPACE", &dir);
        env::remove_var("ZEROZ_BACKGROUND");
        let r = spawn("unit-test-job").unwrap();
        assert!(r.contains("Background task"));
        env::remove_var("ZEROZ_WORKSPACE");
        let _ = fs::remove_dir_all(&dir);
    }
}
