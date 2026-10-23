//! Cron-style scheduled tasks — local file store (no background daemon in-process).
//! File: `{workspace}/.0zcode/cron.jsonl`
//! Format: id | expr | command | enabled

use crate::error::{Error, Result};
use crate::tools::ToolOutput;
use crate::workspace;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub expr: String,
    pub command: String,
    pub enabled: bool,
}

fn path() -> std::path::PathBuf {
    workspace::root().join(".0zcode").join("cron.jsonl")
}

fn now_id() -> String {
    let t = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    format!("cron{t}")
}

fn load() -> Result<Vec<Job>> {
    let p = path();
    if !p.is_file() {
        return Ok(Vec::new());
    }
    let f = fs::File::open(&p).map_err(|e| Error::Tool(e.to_string()))?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines().flatten() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // id|expr|command|1
        let parts: Vec<&str> = line.splitn(4, '|').collect();
        if parts.len() < 3 {
            continue;
        }
        out.push(Job {
            id: parts[0].trim().to_string(),
            expr: parts[1].trim().to_string(),
            command: parts[2].trim().to_string(),
            enabled: parts.get(3).map(|s| s.trim() != "0").unwrap_or(true),
        });
    }
    Ok(out)
}

fn save(jobs: &[Job]) -> Result<()> {
    let p = path();
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).map_err(|e| Error::Tool(e.to_string()))?;
    }
    let mut f = fs::File::create(&p).map_err(|e| Error::Tool(e.to_string()))?;
    writeln!(f, "# 0zcode cron — id|expr|command|enabled").ok();
    for j in jobs {
        writeln!(
            f,
            "{}|{}|{}|{}",
            j.id,
            j.expr,
            j.command,
            if j.enabled { "1" } else { "0" }
        )
        .map_err(|e| Error::Tool(e.to_string()))?;
    }
    Ok(())
}

pub fn create(expr: &str, command: &str) -> Result<ToolOutput> {
    let expr = expr.trim();
    let command = command.trim();
    if expr.is_empty() || command.is_empty() {
        return Err(Error::Tool(
            "CronCreate: usage CronCreate <expr> | <command>\n  expr e.g. daily, hourly, @reboot, 0 9 * * *".into(),
        ));
    }
    let mut jobs = load()?;
    let j = Job {
        id: now_id(),
        expr: expr.to_string(),
        command: command.to_string(),
        enabled: true,
    };
    let id = j.id.clone();
    jobs.push(j);
    save(&jobs)?;
    Ok(ToolOutput {
        name: "CronCreate",
        body: format!(
            "│ scheduled {id}\n│ expr: {expr}\n│ command: {command}\n│ note: stored in .0zcode/cron.jsonl — run via external cron or 0zcode CronList"
        ),
    })
}

pub fn delete(id: &str) -> Result<ToolOutput> {
    let mut jobs = load()?;
    let before = jobs.len();
    jobs.retain(|j| j.id != id);
    if jobs.len() == before {
        return Err(Error::Tool(format!("CronDelete: no job `{id}`")));
    }
    save(&jobs)?;
    Ok(ToolOutput {
        name: "CronDelete",
        body: format!("│ deleted {id}"),
    })
}

pub fn list() -> Result<ToolOutput> {
    let jobs = load()?;
    if jobs.is_empty() {
        return Ok(ToolOutput {
            name: "CronList",
            body: "│ (no cron jobs) — CronCreate daily | TaskList".into(),
        });
    }
    let mut body = format!("│ {} job(s)\n", jobs.len());
    for j in jobs {
        body.push_str(&format!(
            "│ {}  [{}]  {}  →  {}\n",
            j.id,
            if j.enabled { "on" } else { "off" },
            j.expr,
            j.command
        ));
    }
    Ok(ToolOutput {
        name: "CronList",
        body,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn create_list() {
        let _g = LOCK.lock().unwrap();
        let dir = std::env::temp_dir().join("0zcode_cron_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        std::env::set_var("ZEROZ_WORKSPACE", &dir);
        let _ = create("daily", "TaskList").unwrap();
        let l = list().unwrap();
        assert!(l.body.contains("daily"));
        std::env::remove_var("ZEROZ_WORKSPACE");
    }
}
