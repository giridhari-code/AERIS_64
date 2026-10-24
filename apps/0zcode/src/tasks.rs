//! Task management tools — company-standard local store.
//! File: `{workspace}/.0zcode/tasks.jsonl` (one JSON object per line, hand-rolled).

use crate::error::{Error, Result};
use crate::tools::ToolOutput;
use crate::workspace;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Task {
    pub id: String,
    pub title: String,
    pub status: String, // open | running | done | stopped
    pub detail: String,
}

fn store_path() -> std::path::PathBuf {
    workspace::root().join(".0zcode").join("tasks.jsonl")
}

fn now_id() -> String {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("t{t}")
}

fn load_all() -> Result<Vec<Task>> {
    let path = store_path();
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let f = fs::File::open(&path).map_err(|e| Error::Tool(e.to_string()))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines().flatten() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(t) = parse_line(line) {
            out.push(t);
        }
    }
    Ok(out)
}

fn parse_line(line: &str) -> Option<Task> {
    // {"id":"...","title":"...","status":"...","detail":"..."}
    let id = field(line, "id")?;
    let title = field(line, "title").unwrap_or_default();
    let status = field(line, "status").unwrap_or_else(|| "open".into());
    let detail = field(line, "detail").unwrap_or_default();
    Some(Task {
        id,
        title,
        status,
        detail,
    })
}

fn field(json: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let i = json.find(&pat)?;
    let rest = &json[i + pat.len()..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                out.push(n);
            }
        } else if c == '"' {
            break;
        } else {
            out.push(c);
        }
    }
    Some(out)
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn save_all(tasks: &[Task]) -> Result<()> {
    let path = store_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Tool(e.to_string()))?;
    }
    let mut f = fs::File::create(&path).map_err(|e| Error::Tool(e.to_string()))?;
    for t in tasks {
        writeln!(
            f,
            "{{\"id\":\"{}\",\"title\":\"{}\",\"status\":\"{}\",\"detail\":\"{}\"}}",
            esc(&t.id),
            esc(&t.title),
            esc(&t.status),
            esc(&t.detail)
        )
        .map_err(|e| Error::Tool(e.to_string()))?;
    }
    Ok(())
}

pub fn create(title: &str, detail: &str) -> Result<ToolOutput> {
    let title = title.trim();
    if title.is_empty() {
        return Err(Error::Tool("TaskCreate: empty title".into()));
    }
    let mut tasks = load_all()?;
    let t = Task {
        id: now_id(),
        title: title.to_string(),
        status: "open".into(),
        detail: detail.trim().to_string(),
    };
    let id = t.id.clone();
    tasks.push(t);
    save_all(&tasks)?;
    Ok(ToolOutput {
        name: "TaskCreate",
        body: format!("│ created {id}\n│ title: {title}\n│ status: open"),
    })
}

pub fn get(id: &str) -> Result<ToolOutput> {
    let tasks = load_all()?;
    let t = tasks
        .iter()
        .find(|t| t.id == id)
        .ok_or_else(|| Error::Tool(format!("TaskGet: no task `{id}`")))?;
    Ok(ToolOutput {
        name: "TaskGet",
        body: format!(
            "│ {}\n│ title: {}\n│ status: {}\n│ detail: {}",
            t.id, t.title, t.status, t.detail
        ),
    })
}

pub fn list() -> Result<ToolOutput> {
    let tasks = load_all()?;
    if tasks.is_empty() {
        return Ok(ToolOutput {
            name: "TaskList",
            body: "│ (no tasks) — TaskCreate <title>".into(),
        });
    }
    let mut body = format!("│ {} task(s)\n", tasks.len());
    for t in tasks {
        body.push_str(&format!("│ {}  [{}]  {}\n", t.id, t.status, t.title));
    }
    Ok(ToolOutput {
        name: "TaskList",
        body,
    })
}

pub fn update(id: &str, status: Option<&str>, detail: Option<&str>) -> Result<ToolOutput> {
    let mut tasks = load_all()?;
    let t = tasks
        .iter_mut()
        .find(|t| t.id == id)
        .ok_or_else(|| Error::Tool(format!("TaskUpdate: no task `{id}`")))?;
    if let Some(s) = status {
        let s = s.trim().to_lowercase();
        if !matches!(s.as_str(), "open" | "running" | "done" | "stopped") {
            return Err(Error::Tool(
                "status must be open|running|done|stopped".into(),
            ));
        }
        t.status = s;
    }
    if let Some(d) = detail {
        t.detail = d.to_string();
    }
    let snap = t.clone();
    save_all(&tasks)?;
    Ok(ToolOutput {
        name: "TaskUpdate",
        body: format!(
            "│ updated {}\n│ status: {}\n│ detail: {}",
            snap.id, snap.status, snap.detail
        ),
    })
}

pub fn stop(id: &str) -> Result<ToolOutput> {
    update(id, Some("stopped"), None).map(|mut o| {
        o.name = "TaskStop";
        o
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn create_list_stop() {
        let _g = LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join("0zcode_tasks_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        std::env::set_var("ZEROZ_WORKSPACE", &dir);
        let c = create("demo task", "detail").unwrap();
        assert!(c.body.contains("created"));
        let l = list().unwrap();
        assert!(l.body.contains("demo task"));
        std::env::remove_var("ZEROZ_WORKSPACE");
    }
}
