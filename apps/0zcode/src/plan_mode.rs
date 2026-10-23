//! Planning mode + AskUserQuestion — company-standard session flags.

use crate::error::{Error, Result};
use crate::tools::ToolOutput;
use crate::workspace;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};

static PLAN_MODE: AtomicBool = AtomicBool::new(false);

fn plan_file() -> std::path::PathBuf {
    workspace::root().join(".0zcode").join("plan.md")
}

pub fn is_active() -> bool {
    PLAN_MODE.load(Ordering::Relaxed)
}

pub fn enter() -> ToolOutput {
    PLAN_MODE.store(true, Ordering::Relaxed);
    ToolOutput {
        name: "EnterPlanMode",
        body: "│ plan mode ON\n│ outline steps, then ExitPlanMode to review\n│ plan file: .0zcode/plan.md".into(),
    }
}

pub fn exit_review() -> Result<ToolOutput> {
    PLAN_MODE.store(false, Ordering::Relaxed);
    let path = plan_file();
    let content = if path.is_file() {
        fs::read_to_string(&path).unwrap_or_default()
    } else {
        "(no plan.md yet — write steps while in plan mode)".into()
    };
    Ok(ToolOutput {
        name: "ExitPlanMode",
        body: format!(
            "│ plan mode OFF — review\n│ {}\n│ —\n{}",
            path.display(),
            content.chars().take(3000).collect::<String>()
        ),
    })
}

pub fn append_plan_line(line: &str) -> Result<()> {
    if !is_active() {
        return Ok(());
    }
    let path = plan_file();
    if let Some(p) = path.parent() {
        fs::create_dir_all(p).map_err(|e| Error::Tool(e.to_string()))?;
    }
    use std::io::Write;
    let mut f = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| Error::Tool(e.to_string()))?;
    writeln!(f, "- {line}").map_err(|e| Error::Tool(e.to_string()))?;
    Ok(())
}

pub fn ask_user(question: &str) -> ToolOutput {
    let q = question.trim();
    ToolOutput {
        name: "AskUserQuestion",
        body: format!(
            "│ question for you\n│ {q}\n│ —\n│ reply in the next message (0zcode does not block the TTY)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enter_exit() {
        let _ = enter();
        assert!(is_active());
        let _ = exit_review();
        assert!(!is_active());
    }
}
