//! In-session conversation history (last N turns).

use std::fmt::Write as _;

#[derive(Debug, Clone)]
pub struct Turn {
    pub role: &'static str,
    pub text: String,
}

#[derive(Debug, Default)]
pub struct History {
    turns: Vec<Turn>,
    cap: usize,
}

impl History {
    pub fn new(cap: usize) -> Self {
        Self {
            turns: Vec::new(),
            cap: cap.clamp(4, 200),
        }
    }

    pub fn push(&mut self, role: &'static str, text: &str) {
        let t = text.trim();
        if t.is_empty() || t.starts_with("__") {
            return;
        }
        self.turns.push(Turn {
            role,
            text: t.chars().take(2000).collect(),
        });
        while self.turns.len() > self.cap {
            self.turns.remove(0);
        }
    }

    pub fn clear(&mut self) {
        self.turns.clear();
    }

    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    pub fn render(&self) -> String {
        if self.turns.is_empty() {
            return "No history yet this session.".into();
        }
        let mut out = String::new();
        let _ = writeln!(out, "session history ({} turns)", self.turns.len());
        for (i, t) in self.turns.iter().enumerate() {
            let preview: String = t.text.chars().take(120).collect();
            let _ = writeln!(out, "{}. [{}] {}", i + 1, t.role, preview);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn caps_turns() {
        let mut h = History::new(4);
        for i in 0..10 {
            h.push("user", &format!("msg{i}"));
        }
        assert_eq!(h.len(), 4);
    }
}
