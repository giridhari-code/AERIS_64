//! Session persist / resume (line-oriented, no serde).

use crate::ui::state::{Message, MessageKind, ToolStatus};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;
use std::time::Duration;

pub fn session_dir() -> PathBuf {
    if let Ok(p) = std::env::var("ZEROZ_SESSION_DIR") {
        return PathBuf::from(p);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".local/share/0zcode/sessions")
}

pub fn last_session_path() -> PathBuf {
    session_dir().join("last.jsonl")
}

pub fn ensure_dir() -> io::Result<()> {
    fs::create_dir_all(session_dir())
}

/// Append one message as a single JSON-ish line (manual).
pub fn append_message(path: &PathBuf, msg: &Message) -> io::Result<()> {
    ensure_dir()?;
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    let line = encode(msg);
    writeln!(f, "{line}")?;
    Ok(())
}

pub fn save_all(path: &PathBuf, messages: &[Message]) -> io::Result<()> {
    ensure_dir()?;
    let mut f = File::create(path)?;
    for m in messages {
        writeln!(f, "{}", encode(m))?;
    }
    Ok(())
}

pub fn load_all(path: &PathBuf) -> io::Result<Vec<Message>> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let f = File::open(path)?;
    let mut out = Vec::new();
    for line in BufReader::new(f).lines() {
        let line = line?;
        if let Some(m) = decode(&line) {
            out.push(m);
        }
    }
    Ok(out)
}

fn encode(m: &Message) -> String {
    let (kind, extra) = match &m.kind {
        MessageKind::User => ("user", String::new()),
        MessageKind::Assistant => ("assistant", String::new()),
        MessageKind::System => ("system", String::new()),
        MessageKind::Error => ("error", String::new()),
        MessageKind::Activity { summary } => ("activity", escape(summary)),
        MessageKind::Thought { duration } => {
            ("thought", duration.as_millis().to_string())
        }
        MessageKind::Tool { name, status } => {
            let st = match status {
                ToolStatus::Running => "running",
                ToolStatus::Done => "done",
                ToolStatus::Failed => "failed",
            };
            ("tool", format!("{}|{}", escape(name), st))
        }
    };
    format!(
        "{{\"k\":\"{}\",\"x\":\"{}\",\"t\":\"{}\"}}",
        kind,
        extra,
        escape(&m.text)
    )
}

fn decode(line: &str) -> Option<Message> {
    let k = extract(line, "\"k\":\"")?;
    let x = extract(line, "\"x\":\"").unwrap_or_default();
    let t = extract(line, "\"t\":\"").unwrap_or_default();
    let text = unescape(&t);
    let kind = match k.as_str() {
        "user" => MessageKind::User,
        "assistant" => MessageKind::Assistant,
        "system" => MessageKind::System,
        "error" => MessageKind::Error,
        "activity" => MessageKind::Activity {
            summary: unescape(&x),
        },
        "thought" => {
            let ms: u64 = x.parse().unwrap_or(0);
            MessageKind::Thought {
                duration: Duration::from_millis(ms),
            }
        }
        "tool" => {
            let mut parts = x.splitn(2, '|');
            let name = unescape(parts.next().unwrap_or("tool"));
            let st = match parts.next().unwrap_or("done") {
                "running" => ToolStatus::Running,
                "failed" => ToolStatus::Failed,
                _ => ToolStatus::Done,
            };
            MessageKind::Tool { name, status: st }
        }
        _ => return None,
    };
    Some(Message { kind, text })
}

fn extract(line: &str, key: &str) -> Option<String> {
    let i = line.find(key)? + key.len();
    let rest = &line[i..];
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                out.push(match n {
                    'n' => '\n',
                    't' => '\t',
                    '"' => '"',
                    '\\' => '\\',
                    other => other,
                });
            }
        } else if c == '"' {
            break;
        } else {
            out.push(c);
        }
    }
    Some(out)
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(n) = chars.next() {
                out.push(match n {
                    'n' => '\n',
                    't' => '\t',
                    '"' => '"',
                    '\\' => '\\',
                    other => other,
                });
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_user() {
        let m = Message::user("hello");
        let line = encode(&m);
        let back = decode(&line).unwrap();
        assert_eq!(back.text, "hello");
    }
}
