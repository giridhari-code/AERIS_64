//! Custom slash commands — user-defined shortcuts (company-standard).
//!
//! Location: `{workspace}/.0zcode/commands/<name>.md`
//! First line optional title; body is the prompt/template.
//! `$ARGS` / `{{args}}` replaced with trailing arguments.
//!
//! Commands: /commands · /commands run <name> [args] · /cmd <name> [args]

use crate::workspace;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct CustomCommand {
    pub name: String,
    pub title: String,
    pub body: String,
}

fn dir() -> PathBuf {
    workspace::root().join(".0zcode").join("commands")
}

pub fn load_all() -> Vec<CustomCommand> {
    let dir = dir();
    let Ok(rd) = fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let path = e.path();
        if path.extension().and_then(|x| x.to_str()) != Some("md") {
            continue;
        }
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("cmd")
            .to_string();
        let Ok(text) = fs::read_to_string(&path) else {
            continue;
        };
        let text = text.trim().to_string();
        if text.is_empty() {
            continue;
        }
        let (title, body) = if let Some((first, rest)) = text.split_once('\n') {
            if first.starts_with('#') {
                (first.trim_start_matches('#').trim().to_string(), rest.trim().to_string())
            } else {
                (name.clone(), text)
            }
        } else {
            (name.clone(), text)
        };
        out.push(CustomCommand { name, title, body });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

pub fn expand(name: &str, args: &str) -> Option<String> {
    let cmd = load_all()
        .into_iter()
        .find(|c| c.name.eq_ignore_ascii_case(name))?;
    let mut body = cmd.body;
    body = body.replace("$ARGS", args);
    body = body.replace("{{args}}", args);
    body = body.replace("$args", args);
    if !args.is_empty() && !body.contains(args) {
        // if template had no placeholder, append args
        if !body.contains("$ARGS") && !body.contains("{{args}}") {
            body.push(' ');
            body.push_str(args);
        }
    }
    Some(body.trim().to_string())
}

pub fn list_text() -> String {
    let cmds = load_all();
    if cmds.is_empty() {
        return "No custom commands.\n\nAdd markdown files under `.0zcode/commands/`.\n\
Example `.0zcode/commands/standup.md`:\n\
  # Daily standup\n\
  Summarize workspace activity for $ARGS\n\n\
Run: /cmd standup today".into();
    }
    let mut out = format!("Custom commands ({})\n\n", cmds.len());
    for c in &cmds {
        out.push_str(&format!("  /cmd {:<14}  {}\n", c.name, c.title));
    }
    out.push_str("\nRun: /cmd <name> [args] · list: /commands\n");
    out
}

pub fn handle_command(line: &str) -> Option<String> {
    let t = line.trim();
    let lower = t.to_lowercase();
    if lower == "/commands" || lower == "/cmds" {
        return Some(list_text());
    }
    if let Some(rest) = lower
        .strip_prefix("/commands run ")
        .or_else(|| lower.strip_prefix("/cmd "))
        .or_else(|| lower.strip_prefix("/command "))
    {
        let rest_orig = if t.contains(' ') {
            if lower.starts_with("/commands run ") {
                t["/commands run ".len()..].trim()
            } else if lower.starts_with("/cmd ") {
                t["/cmd ".len()..].trim()
            } else {
                t["/command ".len()..].trim()
            }
        } else {
            rest
        };
        let mut parts = rest_orig.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").trim();
        let args = parts.next().unwrap_or("").trim();
        if name.is_empty() {
            return Some("Usage: /cmd <name> [args]".into());
        }
        return match expand(name, args) {
            Some(body) => Some(format!("# command `{name}`\n\n{body}")),
            None => Some(format!(
                "Unknown command `{name}`.\n\n{}",
                list_text()
            )),
        };
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::sync::Mutex;
    static LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn expands_args() {
        let _g = LOCK.lock().unwrap();
        let dir = env::temp_dir().join(format!("0z_cmd_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join(".0zcode/commands")).unwrap();
        fs::write(
            dir.join(".0zcode/commands/hello.md"),
            "# Hello\nSay hi to $ARGS\n",
        )
        .unwrap();
        env::set_var("ZEROZ_WORKSPACE", &dir);
        let body = expand("hello", "Grok").unwrap();
        assert!(body.contains("Grok"));
        env::remove_var("ZEROZ_WORKSPACE");
        let _ = fs::remove_dir_all(&dir);
    }
}
