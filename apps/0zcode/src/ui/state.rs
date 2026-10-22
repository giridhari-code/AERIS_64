//! UI state + channel messages between agent worker and render loop.

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Done,
    Failed,
}

#[derive(Debug, Clone)]
pub enum MessageKind {
    User,
    Assistant,
    Activity { summary: String },
    Thought { duration: Duration },
    Tool {
        name: String,
        status: ToolStatus,
    },
    System,
    Error,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub kind: MessageKind,
    pub text: String,
}

impl Message {
    pub fn user(t: impl Into<String>) -> Self {
        Self {
            kind: MessageKind::User,
            text: t.into(),
        }
    }
    pub fn assistant(t: impl Into<String>) -> Self {
        Self {
            kind: MessageKind::Assistant,
            text: t.into(),
        }
    }
    pub fn activity(s: impl Into<String>) -> Self {
        Self {
            kind: MessageKind::Activity {
                summary: s.into(),
            },
            text: String::new(),
        }
    }
    pub fn thought(d: Duration) -> Self {
        Self {
            kind: MessageKind::Thought { duration: d },
            text: String::new(),
        }
    }
    pub fn tool(name: impl Into<String>, body: impl Into<String>, status: ToolStatus) -> Self {
        Self {
            kind: MessageKind::Tool {
                name: name.into(),
                status,
            },
            text: body.into(),
        }
    }
    pub fn system(t: impl Into<String>) -> Self {
        Self {
            kind: MessageKind::System,
            text: t.into(),
        }
    }
    pub fn error(t: impl Into<String>) -> Self {
        Self {
            kind: MessageKind::Error,
            text: t.into(),
        }
    }
}

/// Events from UI → agent worker
#[derive(Debug, Clone)]
pub enum UiEvent {
    /// User submit with generation id (stale results ignored after cancel).
    Submit { line: String, gen: u64 },
    /// Bump generation so in-flight work is ignored when it finishes.
    Cancel,
    Quit,
}

/// Updates from agent worker → UI
#[derive(Debug, Clone)]
pub enum AgentUpdate {
    ToolStarted { name: String, gen: u64 },
    ToolFinished {
        name: String,
        body: String,
        status: ToolStatus,
        elapsed: Duration,
        gen: u64,
    },
    Assistant { text: String, gen: u64 },
    Error { text: String, gen: u64 },
    Refused { text: String, gen: u64 },
    Idle { gen: u64 },
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub messages: Vec<Message>,
    pub input: String,
    pub cursor: usize,
    pub scroll: u16,
    pub model_label: String,
    pub cwd: String,
    pub version: String,
    pub working: bool,
    pub status_line: String,
    pub mode_hint: String,
    pub started: Instant,
    /// Pending network tool permission: (tool_hint, full user line).
    pub pending_permission: Option<(String, String)>,
    /// Network tools allowed for this session after "allow always".
    pub network_allowed_always: bool,
    /// Monotonic id; cancel bumps this so stale AgentUpdate are dropped.
    pub request_gen: u64,
}

impl AppState {
    pub fn new(model: String, cwd: String, version: String) -> Self {
        Self {
            messages: Vec::new(),
            input: String::new(),
            cursor: 0,
            scroll: 0,
            model_label: model,
            cwd,
            version,
            working: false,
            status_line: "ready".into(),
            mode_hint: "tools-first · coding disabled".into(),
            started: Instant::now(),
            pending_permission: None,
            network_allowed_always: false,
            request_gen: 0,
        }
    }

    pub fn push(&mut self, m: Message) {
        self.messages.push(m);
        if self.messages.len() > 300 {
            let n = self.messages.len() - 300;
            self.messages.drain(0..n);
        }
    }

    pub fn clear_transcript(&mut self) {
        self.messages.clear();
        self.scroll = 0;
    }

    pub fn insert_char(&mut self, c: char) {
        if self.cursor > self.input.len() {
            self.cursor = self.input.len();
        }
        self.input.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    pub fn backspace(&mut self) {
        if self.cursor == 0 {
            return;
        }
        let prev = self.input[..self.cursor]
            .chars()
            .next_back()
            .map(|c| c.len_utf8())
            .unwrap_or(1);
        let start = self.cursor - prev;
        self.input.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    pub fn take_input(&mut self) -> String {
        let s = std::mem::take(&mut self.input);
        self.cursor = 0;
        s.trim().to_string()
    }

    pub fn apply_update(&mut self, u: AgentUpdate) {
        let gen = match &u {
            AgentUpdate::ToolStarted { gen, .. }
            | AgentUpdate::ToolFinished { gen, .. }
            | AgentUpdate::Assistant { gen, .. }
            | AgentUpdate::Error { gen, .. }
            | AgentUpdate::Refused { gen, .. }
            | AgentUpdate::Idle { gen } => *gen,
        };
        if gen != self.request_gen {
            return; // cancelled or superseded
        }
        match u {
            AgentUpdate::ToolStarted { name, .. } => {
                self.working = true;
                self.status_line = format!("running {name}");
                self.push(Message::tool(name, String::new(), ToolStatus::Running));
            }
            AgentUpdate::ToolFinished {
                name,
                body,
                status,
                elapsed,
                ..
            } => {
                if let Some(last) = self.messages.last_mut() {
                    if matches!(
                        &last.kind,
                        MessageKind::Tool {
                            status: ToolStatus::Running,
                            ..
                        }
                    ) {
                        last.kind = MessageKind::Tool { name, status };
                        last.text = body;
                        self.push(Message::thought(elapsed));
                        return;
                    }
                }
                self.push(Message::tool(name, body, status));
                self.push(Message::thought(elapsed));
            }
            AgentUpdate::Assistant { text, .. } => {
                self.push(Message::assistant(text));
                self.working = false;
                self.status_line = "ready".into();
            }
            AgentUpdate::Error { text, .. } | AgentUpdate::Refused { text, .. } => {
                self.push(Message::error(text));
                self.working = false;
                self.status_line = "ready".into();
            }
            AgentUpdate::Idle { .. } => {
                self.working = false;
                self.status_line = "ready".into();
            }
        }
    }

    pub fn cancel_in_flight(&mut self) {
        self.request_gen = self.request_gen.wrapping_add(1);
        self.working = false;
        self.status_line = "cancelled".into();
        // mark running tools failed
        for m in self.messages.iter_mut().rev() {
            if let MessageKind::Tool { status, .. } = &mut m.kind {
                if *status == ToolStatus::Running {
                    *status = ToolStatus::Failed;
                    if m.text.is_empty() {
                        m.text = "cancelled".into();
                    }
                    break;
                }
            }
        }
    }
}

pub fn explored_label(n: usize) -> String {
    match n {
        0 => "Explored: nothing".into(),
        1 => "Explored: 1 tool".into(),
        n => format!("Explored: {n} tools"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plural() {
        assert_eq!(explored_label(1), "Explored: 1 tool");
        assert_eq!(explored_label(3), "Explored: 3 tools");
    }

    #[test]
    fn cancel_drops_stale() {
        let mut s = AppState::new("m".into(), "/tmp".into(), "1".into());
        s.request_gen = 1;
        s.apply_update(AgentUpdate::Assistant {
            text: "old".into(),
            gen: 0,
        });
        assert!(!s.messages.iter().any(|m| m.text == "old"));
        s.apply_update(AgentUpdate::Assistant {
            text: "new".into(),
            gen: 1,
        });
        assert!(s.messages.iter().any(|m| m.text == "new"));
    }
}
