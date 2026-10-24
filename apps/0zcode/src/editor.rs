//! Single-line editor state: cursor, readline-style shortcuts, input history
//! (persisted across sessions) and the slash-command menu. Pure logic — no I/O
//! besides the optional history file — so it is fully unit-testable.

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

/// Slash commands shown in the menu: `(command, description)`.
pub const COMMANDS: &[(&str, &str)] = &[
    ("/help", "commands and shortcuts"),
    ("/tools", "built-in tools"),
    ("/plugins", "features & integrations"),
    ("/team", "multi-agent teams"),
    ("/parallel", "run bots in parallel"),
    ("/mcp", "user-defined tools"),
    ("/rules", "project rules"),
    ("/memory", "project memory / notes"),
    ("/perms", "tool permissions"),
    ("/approval", "human approval policy"),
    ("/commands", "custom slash commands"),
    ("/cmd", "run custom command"),
    ("/bg", "background tasks"),
    ("/lsp", "diagnostics (cargo/heuristic)"),
    ("/computer", "computer-use status"),
    ("/history", "turns in this session"),
    ("/status", "session status"),
    ("/model", "model label"),
    ("/version", "product version"),
    ("/copy", "save last reply to a file"),
    ("/expand", "show full last tool output"),
    ("/clear", "clear screen and history"),
    ("/exit", "quit 0zcode"),
];

const HISTORY_CAP: usize = 500;

#[derive(Debug, Default)]
pub struct Editor {
    buf: Vec<char>,
    cur: usize,
    hist: Vec<String>,
    hpos: Option<usize>,
    draft: Vec<char>,
    sel: usize,
    dismissed: bool,
}

impl Editor {
    pub fn new() -> Self {
        Self::default()
    }

    // ── content ──────────────────────────────────────────────

    pub fn chars(&self) -> &[char] {
        &self.buf
    }

    pub fn cursor(&self) -> usize {
        self.cur
    }

    pub fn text(&self) -> String {
        self.buf.iter().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    pub fn clear(&mut self) {
        self.buf.clear();
        self.cur = 0;
        self.touch();
    }

    fn touch(&mut self) {
        self.sel = 0;
        self.dismissed = false;
        self.hpos = None;
    }

    pub fn insert(&mut self, c: char) {
        if c.is_control() {
            return;
        }
        self.buf.insert(self.cur, c);
        self.cur += 1;
        self.touch();
    }

    /// Insert pasted text; newlines and tabs become spaces (single-line input).
    pub fn insert_str(&mut self, s: &str) {
        for c in s.chars() {
            let c = if c == '\n' || c == '\r' || c == '\t' { ' ' } else { c };
            self.insert(c);
        }
    }

    pub fn backspace(&mut self) {
        if self.cur > 0 {
            self.cur -= 1;
            self.buf.remove(self.cur);
            self.touch();
        }
    }

    pub fn delete(&mut self) {
        if self.cur < self.buf.len() {
            self.buf.remove(self.cur);
            self.touch();
        }
    }

    // ── movement ─────────────────────────────────────────────

    pub fn left(&mut self) {
        self.cur = self.cur.saturating_sub(1);
    }

    pub fn right(&mut self) {
        if self.cur < self.buf.len() {
            self.cur += 1;
        }
    }

    pub fn home(&mut self) {
        self.cur = 0;
    }

    pub fn end(&mut self) {
        self.cur = self.buf.len();
    }

    fn prev_word_start(&self) -> usize {
        let mut i = self.cur;
        while i > 0 && self.buf[i - 1].is_whitespace() {
            i -= 1;
        }
        while i > 0 && !self.buf[i - 1].is_whitespace() {
            i -= 1;
        }
        i
    }

    fn next_word_end(&self) -> usize {
        let mut i = self.cur;
        let n = self.buf.len();
        while i < n && self.buf[i].is_whitespace() {
            i += 1;
        }
        while i < n && !self.buf[i].is_whitespace() {
            i += 1;
        }
        i
    }

    pub fn word_left(&mut self) {
        self.cur = self.prev_word_start();
    }

    pub fn word_right(&mut self) {
        self.cur = self.next_word_end();
    }

    // ── kills ────────────────────────────────────────────────

    pub fn kill_word_back(&mut self) {
        let start = self.prev_word_start();
        self.buf.drain(start..self.cur);
        self.cur = start;
        self.touch();
    }

    pub fn kill_to_start(&mut self) {
        self.buf.drain(..self.cur);
        self.cur = 0;
        self.touch();
    }

    pub fn kill_to_end(&mut self) {
        self.buf.truncate(self.cur);
        self.touch();
    }

    // ── history ──────────────────────────────────────────────

    pub fn load_history(&mut self, lines: Vec<String>) {
        self.hist = lines;
        let n = self.hist.len();
        if n > HISTORY_CAP {
            self.hist.drain(..n - HISTORY_CAP);
        }
    }

    pub fn push_history(&mut self, line: &str) {
        let t = line.trim();
        if t.is_empty() {
            return;
        }
        if self.hist.last().map(String::as_str) != Some(t) {
            self.hist.push(t.to_string());
            if self.hist.len() > HISTORY_CAP {
                self.hist.remove(0);
            }
        }
        self.hpos = None;
    }

    fn load(&mut self, s: &str) {
        self.buf = s.chars().collect();
        self.cur = self.buf.len();
        self.sel = 0;
        self.dismissed = true; // browsing history must not pop the slash menu
    }

    pub fn history_prev(&mut self) {
        if self.hist.is_empty() {
            return;
        }
        let idx = match self.hpos {
            None => {
                self.draft = self.buf.clone();
                self.hist.len() - 1
            }
            Some(0) => return,
            Some(i) => i - 1,
        };
        self.hpos = Some(idx);
        let s = self.hist[idx].clone();
        self.load(&s);
    }

    pub fn history_next(&mut self) {
        let Some(i) = self.hpos else { return };
        if i + 1 < self.hist.len() {
            self.hpos = Some(i + 1);
            let s = self.hist[i + 1].clone();
            self.load(&s);
        } else {
            self.hpos = None;
            self.buf = std::mem::take(&mut self.draft);
            self.cur = self.buf.len();
            self.sel = 0;
            self.dismissed = false;
        }
    }

    // ── slash menu ───────────────────────────────────────────

    pub fn suggestions(&self) -> Vec<(&'static str, &'static str)> {
        if self.dismissed {
            return Vec::new();
        }
        let t = self.text().to_lowercase();
        if !t.starts_with('/') || t.contains(char::is_whitespace) {
            return Vec::new();
        }
        let m: Vec<_> = COMMANDS
            .iter()
            .copied()
            .filter(|(c, _)| c.starts_with(&t))
            .collect();
        if m.len() == 1 && m[0].0 == t {
            return Vec::new();
        }
        m
    }

    pub fn menu_visible(&self) -> bool {
        !self.suggestions().is_empty()
    }

    pub fn selected_index(&self) -> usize {
        let n = self.suggestions().len();
        if n == 0 {
            0
        } else {
            self.sel.min(n - 1)
        }
    }

    pub fn selected_command(&self) -> Option<&'static str> {
        self.suggestions().get(self.selected_index()).map(|(c, _)| *c)
    }

    pub fn menu_move(&mut self, delta: isize) {
        let n = self.suggestions().len() as isize;
        if n == 0 {
            return;
        }
        let cur = self.selected_index() as isize;
        self.sel = (cur + delta).rem_euclid(n) as usize;
    }

    pub fn dismiss_menu(&mut self) {
        self.dismissed = true;
    }

    /// Replace the buffer with the highlighted command (keeps the menu state).
    pub fn apply_selected(&mut self) -> bool {
        let Some(cmd) = self.selected_command() else {
            return false;
        };
        let sel = self.sel;
        self.buf = cmd.chars().collect();
        self.cur = self.buf.len();
        self.sel = sel;
        true
    }
}

// ── persisted history ────────────────────────────────────────

pub fn history_path() -> Option<PathBuf> {
    if let Ok(p) = env::var("ZEROZ_HISTORY_FILE") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    if let Ok(x) = env::var("XDG_STATE_HOME") {
        if !x.is_empty() {
            return Some(PathBuf::from(x).join("0zcode/history"));
        }
    }
    let home = env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".local/state/0zcode/history"))
}

pub fn load_history() -> Vec<String> {
    let Some(p) = history_path() else {
        return Vec::new();
    };
    let Ok(text) = fs::read_to_string(&p) else {
        return Vec::new();
    };
    let mut lines: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect();
    if lines.len() > HISTORY_CAP * 2 {
        let keep = lines.split_off(lines.len() - HISTORY_CAP);
        let _ = fs::write(&p, keep.join("\n") + "\n");
        return keep;
    }
    lines
}

pub fn append_history(line: &str) {
    let t = line.trim();
    if t.is_empty() || t.contains('\n') {
        return;
    }
    let Some(p) = history_path() else { return };
    if let Some(dir) = p.parent() {
        let _ = fs::create_dir_all(dir);
    }
    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&p) {
        let _ = writeln!(f, "{t}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn typed(s: &str) -> Editor {
        let mut e = Editor::new();
        e.insert_str(s);
        e
    }

    #[test]
    fn insert_and_cursor_edit() {
        let mut e = typed("helo");
        e.left();
        e.insert('l');
        assert_eq!(e.text(), "hello");
        assert_eq!(e.cursor(), 4);
        e.home();
        e.delete();
        assert_eq!(e.text(), "ello");
        e.end();
        e.backspace();
        assert_eq!(e.text(), "ell");
    }

    #[test]
    fn word_motion_and_kills() {
        let mut e = typed("weather in  Mumbai");
        e.kill_word_back();
        assert_eq!(e.text(), "weather in  ");
        e.kill_word_back();
        assert_eq!(e.text(), "weather ");
        e.home();
        e.word_right();
        assert_eq!(e.cursor(), 7);
        e.kill_to_end();
        assert_eq!(e.text(), "weather");
        e.kill_to_start();
        assert!(e.is_empty());
    }

    #[test]
    fn paste_is_single_line_and_controls_ignored() {
        let mut e = Editor::new();
        e.insert_str("a\nb\tc");
        e.insert('\u{7}');
        assert_eq!(e.text(), "a b c");
    }

    #[test]
    fn history_keeps_draft() {
        let mut e = Editor::new();
        e.load_history(vec!["one".into(), "two".into()]);
        e.insert_str("dra");
        e.history_prev();
        assert_eq!(e.text(), "two");
        e.history_prev();
        assert_eq!(e.text(), "one");
        e.history_prev();
        assert_eq!(e.text(), "one");
        e.history_next();
        assert_eq!(e.text(), "two");
        e.history_next();
        assert_eq!(e.text(), "dra");
    }

    #[test]
    fn history_dedupes_consecutive() {
        let mut e = Editor::new();
        e.push_history("x");
        e.push_history("x");
        e.push_history("  ");
        e.push_history("y");
        e.history_prev();
        assert_eq!(e.text(), "y");
        e.history_prev();
        assert_eq!(e.text(), "x");
    }

    #[test]
    fn slash_menu_filters_and_applies() {
        let mut e = typed("/h");
        let names: Vec<_> = e.suggestions().iter().map(|s| s.0).collect();
        assert_eq!(names, vec!["/help", "/history"]);
        e.menu_move(1);
        assert_eq!(e.selected_command(), Some("/history"));
        e.menu_move(1);
        assert_eq!(e.selected_command(), Some("/help")); // wraps
        assert!(e.apply_selected());
        assert_eq!(e.text(), "/help");
        // exact + unique → menu hides so Enter submits
        assert!(!e.menu_visible());
    }

    #[test]
    fn menu_hidden_for_arguments_and_plain_text() {
        assert!(!typed("/mcp add x").menu_visible());
        assert!(!typed("weather").menu_visible());
        let mut e = typed("/s");
        assert!(e.menu_visible());
        e.dismiss_menu();
        assert!(!e.menu_visible());
        e.insert('t');
        assert!(e.menu_visible()); // editing re-opens it
    }

    #[test]
    fn browsing_history_does_not_open_menu() {
        let mut e = Editor::new();
        e.load_history(vec!["/he".into()]);
        e.history_prev();
        assert_eq!(e.text(), "/he");
        assert!(!e.menu_visible());
    }
}
