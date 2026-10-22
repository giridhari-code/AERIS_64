//! 0zcode TUI renderer — Claude Code / OpenCode style, ANSI + std only.
//!
//! Everything here returns `String`s (pure, unit-tested); `Screen` is the one
//! piece that touches the terminal, redrawing the prompt frame in place.
//!
//! Layout language:
//!   welcome card · `●` assistant blocks · `● tool(arg)` + `⎿` result trees ·
//!   rounded input box · slash menu · dim status footer · braille spinner.

use crate::config::Config;
use crate::editor::COMMANDS;
use crate::policy::PRODUCT_VERSION;
use crate::term::{self, char_width, str_width, visible_width};
use std::env;
use std::path::PathBuf;

const DIM: &str = "38;5;245";
const BORDER: &str = "38;5;240";
const ACCENT: &str = "38;5;215";
const OK: &str = "38;5;78";
const ERR: &str = "38;5;203";
const WARN: &str = "38;5;221";
const INFO: &str = "38;5;110";

pub const SPINNER: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

/// Collapse tool output beyond this many lines (`ctrl+o` / `/expand` shows all).
pub const COLLAPSE_AFTER: usize = 14;

/// Narrowest layout we draw (phones, tmux splits).
pub const MIN_W: usize = 30;

pub struct Theme {
    pub color: bool,
    pub cwd: String,
}

/// A rendered prompt frame plus where the terminal cursor belongs inside it.
pub struct Frame {
    pub lines: Vec<String>,
    pub cursor_row: usize,
    pub cursor_col: usize,
}

/// Text under the input box.
pub struct Footer {
    pub left: String,
    pub right: String,
    pub warn: bool,
}

impl Theme {
    pub fn from_config(cfg: &Config) -> Self {
        let cwd = env::current_dir()
            .ok()
            .and_then(shorten_path)
            .unwrap_or_else(|| "~".into());
        Self {
            color: cfg.color,
            cwd,
        }
    }

    pub fn paint(&self, code: &str, s: &str) -> String {
        if self.color && !s.is_empty() {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn dim(&self, s: &str) -> String {
        self.paint(DIM, s)
    }

    fn border(&self, s: &str) -> String {
        self.paint(BORDER, s)
    }

    pub fn clear_seq() -> &'static str {
        "\x1b[2J\x1b[3J\x1b[H"
    }

    // ───────────── boxes ─────────────

    /// Rounded box of total width `bw`; `rows` may contain ANSI colour.
    fn boxed(&self, title: Option<&str>, rows: &[String], bw: usize) -> String {
        let bw = bw.max(20);
        let inner = bw - 4;
        let mut out = String::new();
        let top = match title {
            Some(t) => {
                let t = fit(t, bw.saturating_sub(8));
                let fill = bw.saturating_sub(5 + str_width(&t));
                format!(
                    "{}{}{}{}",
                    self.border("╭─ "),
                    self.paint(&format!("{ACCENT};1"), &t),
                    self.border(" "),
                    self.border(&format!("{}╮", "─".repeat(fill)))
                )
            }
            None => self.border(&format!("╭{}╮", "─".repeat(bw - 2))),
        };
        out.push_str(&top);
        out.push('\n');
        for r in rows {
            let r = clip_ansi(r, inner);
            let pad = inner.saturating_sub(visible_width(&r));
            out.push_str(&format!(
                "{} {}{} {}\n",
                self.border("│"),
                r,
                " ".repeat(pad),
                self.border("│")
            ));
        }
        out.push_str(&self.border(&format!("╰{}╯", "─".repeat(bw - 2))));
        out
    }

    fn kv(&self, k: &str, v: &str) -> String {
        format!("{}  {}", self.dim(&format!("{k:<8}")), v)
    }

    // ───────────── static panels ─────────────

    pub fn render_welcome(&self, width: usize, model: &str, offline: bool) -> String {
        let bw = width.clamp(MIN_W, 100);
        let inner = bw - 4;
        let mode = if offline {
            self.paint(WARN, "offline · calculator + commands only")
        } else {
            "agent · no code generation".to_string()
        };
        let rows = vec![
            String::new(),
            format!(
                "{}  {}  {}",
                self.paint(&format!("{ACCENT};1"), "▣"),
                self.paint("1", "Welcome to 0zcode"),
                self.dim(&format!("v{PRODUCT_VERSION}"))
            ),
            format!(
                "   {}",
                self.dim(&fit(
                    "tools-first research agent · zero source code",
                    inner.saturating_sub(3)
                ))
            ),
            String::new(),
            self.kv("cwd", &fit(&self.cwd, inner.saturating_sub(10))),
            self.kv("model", &fit(model, inner.saturating_sub(10))),
            self.kv("mode", &mode),
            String::new(),
        ];
        let mut out = String::from("\n");
        out.push_str(&self.boxed(None, &rows, bw));
        out.push('\n');
        let tips = "Ask in English, Hindi or Hinglish · / for commands · /help for shortcuts";
        for l in wrap(tips, bw.saturating_sub(2)) {
            out.push_str(&format!("  {}\n", self.dim(&l)));
        }
        out
    }

    pub fn render_help(&self, width: usize) -> String {
        let bw = width.clamp(MIN_W, 100);
        let mut rows = vec![self.paint(&format!("{ACCENT};1"), "Commands")];
        for (c, d) in COMMANDS {
            rows.push(format!("{}  {}", self.paint(INFO, &format!("{c:<9}")), self.dim(d)));
        }
        rows.push(String::new());
        rows.push(self.paint(&format!("{ACCENT};1"), "Shortcuts"));
        for (k, d) in [
            ("↑ ↓", "input history · menu"),
            ("tab", "complete slash command"),
            ("ctrl+a / e", "start / end of line"),
            ("alt+b / f", "word left / right"),
            ("ctrl+w", "delete word"),
            ("ctrl+u / k", "delete to start / end"),
            ("ctrl+o", "expand last tool output"),
            ("ctrl+l", "clear screen"),
            ("esc", "interrupt a running tool"),
            ("ctrl+c", "clear input · twice to exit"),
            ("ctrl+d", "exit on empty line"),
        ] {
            rows.push(format!("{}  {}", self.paint(INFO, &format!("{k:<11}")), self.dim(d)));
        }
        rows.push(String::new());
        rows.push(self.dim("Try: weather in Mumbai · 938+29 · 10 USD to INR · headlines"));
        self.boxed(Some("Help"), &rows, bw)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn render_status(
        &self,
        width: usize,
        model: &str,
        mode: &str,
        offline: bool,
        turns: usize,
        last_tool: &str,
        uptime_secs: u64,
        config_path: &str,
    ) -> String {
        let bw = width.clamp(MIN_W, 100);
        let room = bw.saturating_sub(14);
        let rows = vec![
            self.kv("model", &fit(model, room)),
            self.kv("mode", mode),
            self.kv(
                "network",
                &if offline {
                    self.paint(WARN, "offline")
                } else {
                    self.paint(OK, "online")
                },
            ),
            self.kv("turns", &turns.to_string()),
            self.kv("last", if last_tool.is_empty() { "—" } else { last_tool }),
            self.kv("uptime", &fmt_uptime(uptime_secs)),
            self.kv("cwd", &fit(&self.cwd, room)),
            self.kv("config", &fit(config_path, room)),
        ];
        self.boxed(Some("Status"), &rows, bw)
    }

    // ───────────── conversation blocks ─────────────

    /// `● text` with hanging indent, wrapped to `width`.
    pub fn render_assistant(&self, text: &str, width: usize) -> String {
        let body_w = width.clamp(MIN_W, 110) - 2;
        let mut out = String::new();
        let mut first = true;
        for line in wrap(text, body_w) {
            if first {
                out.push_str(&format!("{} {}\n", self.paint("1", "●"), line));
                first = false;
            } else {
                out.push_str(&format!("  {line}\n"));
            }
        }
        out.trim_end_matches('\n').to_string()
    }

    /// `● tool(arg)  1.2s` followed by a `⎿` result tree.
    pub fn render_tool(
        &self,
        name: &str,
        query: &str,
        lines: &[String],
        secs: Option<f32>,
        width: usize,
        collapse: bool,
    ) -> String {
        let w = width.clamp(MIN_W, 110);
        let mut out = format!(
            "{} {}{}",
            self.paint(OK, "●"),
            self.paint("1", name),
            self.dim(&format!("({})", fit(query.trim(), 48))),
        );
        if let Some(s) = secs {
            out.push_str(&format!("  {}", self.dim(&format!("{s:.1}s"))));
        }
        out.push('\n');
        let mut wrapped: Vec<String> = Vec::new();
        for l in lines {
            wrapped.extend(wrap(l, w - 5));
        }
        while wrapped.last().is_some_and(|l| l.trim().is_empty()) {
            wrapped.pop();
        }
        if wrapped.is_empty() {
            wrapped.push(String::from("(no output)"));
        }
        let shown = if collapse && wrapped.len() > COLLAPSE_AFTER {
            COLLAPSE_AFTER
        } else {
            wrapped.len()
        };
        for (i, l) in wrapped.iter().take(shown).enumerate() {
            if i == 0 {
                out.push_str(&format!("  {}  {}\n", self.dim("⎿"), l));
            } else {
                out.push_str(&format!("     {l}\n"));
            }
        }
        if shown < wrapped.len() {
            out.push_str(&format!(
                "     {}\n",
                self.dim(&format!(
                    "… +{} lines (ctrl+o to expand)",
                    wrapped.len() - shown
                ))
            ));
        }
        out.trim_end_matches('\n').to_string()
    }

    /// Dim, indented single line (summaries, hints).
    pub fn render_dim(&self, msg: &str) -> String {
        format!("  {}", self.dim(msg))
    }

    pub fn render_error(&self, msg: &str, width: usize) -> String {
        let body_w = width.clamp(MIN_W, 110) - 2;
        let mut out = String::new();
        for (i, line) in wrap(msg, body_w).into_iter().enumerate() {
            if i == 0 {
                out.push_str(&format!(
                    "{} {}\n",
                    self.paint(&format!("{ERR};1"), "●"),
                    self.paint(&format!("{ERR};1"), &line)
                ));
            } else {
                out.push_str(&format!("  {}\n", self.paint(ERR, &line)));
            }
        }
        out.trim_end_matches('\n').to_string()
    }

    pub fn render_ok(&self, msg: &str) -> String {
        format!("{} {}", self.paint(OK, "✔"), self.paint(OK, msg))
    }

    pub fn render_notice(&self, msg: &str) -> String {
        format!("{} {}", self.paint(WARN, "⎿"), self.dim(msg))
    }

    // ───────────── spinner ─────────────

    pub fn render_spinner(&self, tick: usize, label: &str, secs: u64) -> String {
        format!(
            "\r\x1b[2K{} {} {}",
            self.paint(ACCENT, SPINNER[tick % SPINNER.len()]),
            label,
            self.dim(&format!("({secs}s · esc to interrupt)"))
        )
    }

    // ───────────── prompt frame ─────────────

    #[allow(clippy::too_many_arguments)]
    pub fn prompt_frame(
        &self,
        width: usize,
        buf: &[char],
        cur: usize,
        menu: &[(&str, &str)],
        sel: usize,
        footer: &Footer,
        submitted: bool,
    ) -> Frame {
        let bw = width.clamp(MIN_W, 100);
        let inner = bw - 4;
        let cap = inner - 2; // "❯ " prefix

        // wrap the buffer into rows by display width, tracking the cursor
        let mut rows: Vec<String> = vec![String::new()];
        let mut col = 0usize;
        let mut cur_pos = (0usize, 0usize);
        for (i, &c) in buf.iter().enumerate() {
            let w = char_width(c);
            if col + w > cap {
                rows.push(String::new());
                col = 0;
            }
            if i == cur {
                cur_pos = (rows.len() - 1, col);
            }
            if let Some(r) = rows.last_mut() {
                r.push(c);
            }
            col += w;
        }
        if cur >= buf.len() {
            if col + 1 > cap {
                rows.push(String::new());
                col = 0;
            }
            cur_pos = (rows.len() - 1, col);
        }

        let mut lines = vec![self.border(&format!("╭{}╮", "─".repeat(bw - 2)))];
        for (ri, r) in rows.iter().enumerate() {
            let prefix = if ri == 0 {
                self.paint(&format!("{ACCENT};1"), "❯")
            } else {
                " ".to_string()
            };
            let (text, tw) = if buf.is_empty() && ri == 0 {
                let ph = "Ask anything — or type / for commands";
                (self.dim(ph), str_width(ph))
            } else {
                (r.clone(), str_width(r))
            };
            let pad = cap.saturating_sub(tw);
            lines.push(format!(
                "{} {} {}{} {}",
                self.border("│"),
                prefix,
                text,
                " ".repeat(pad),
                self.border("│")
            ));
        }
        lines.push(self.border(&format!("╰{}╯", "─".repeat(bw - 2))));

        if !submitted {
            if !menu.is_empty() {
                for (i, (cmd, desc)) in menu.iter().take(8).enumerate() {
                    let line = if i == sel {
                        format!(
                            "  {} {}  {}",
                            self.paint(ACCENT, "▸"),
                            self.paint(&format!("{ACCENT};1"), &format!("{cmd:<9}")),
                            self.paint("37", desc)
                        )
                    } else {
                        format!(
                            "    {}  {}",
                            self.dim(&format!("{cmd:<9}")),
                            self.dim(desc)
                        )
                    };
                    lines.push(line);
                }
            } else {
                let room = bw.saturating_sub(2);
                let (left, right) = fit_footer(&footer.left, &footer.right, room);
                let gap = room
                    .saturating_sub(str_width(&left) + str_width(&right))
                    .max(1);
                let l = if footer.warn {
                    self.paint(WARN, &left)
                } else {
                    self.dim(&left)
                };
                lines.push(format!("  {}{}{}", l, " ".repeat(gap), self.dim(&right)));
            }
        }

        Frame {
            lines,
            cursor_row: 1 + cur_pos.0,
            cursor_col: 4 + cur_pos.1,
        }
    }
}

// ───────────── in-place redraw ─────────────

/// Rows between the top of the last frame and the cursor *after the terminal
/// reflowed it*. Narrowing the window wraps the old (wider) lines, so the
/// naive "cursor_row" would rewind too little and leave a stale frame behind.
/// Widening never un-wraps our lines, so the plain row count still holds.
pub fn rows_above_cursor(
    widths_above: &[usize],
    cursor_col: usize,
    old_cols: usize,
    new_cols: usize,
) -> usize {
    if new_cols == 0 || new_cols >= old_cols {
        return widths_above.len();
    }
    // ceil-div by hand: `usize::div_ceil` needs Rust 1.73, the crate promises 1.70
    let wrapped: usize = widths_above
        .iter()
        .map(|w| ((*w).max(1) + new_cols - 1) / new_cols)
        .sum();
    wrapped + cursor_col / new_cols
}

/// Owns the "where is the cursor inside the last drawn frame" bookkeeping.
#[derive(Default)]
pub struct Screen {
    drawn: bool,
    cursor_col: usize,
    /// Visible width of every frame line above the cursor row.
    widths_above: Vec<usize>,
    /// Terminal width when the frame was drawn.
    cols: usize,
}

impl Screen {
    pub fn new() -> Self {
        Self::default()
    }

    fn rewind(&self, out: &mut String, cols: usize) {
        if self.drawn {
            let up = rows_above_cursor(&self.widths_above, self.cursor_col, self.cols, cols);
            if up > 0 {
                out.push_str(&format!("\x1b[{up}A"));
            }
        }
        out.push_str("\r\x1b[J");
    }

    fn paint_frame(&mut self, f: &Frame, cols: usize, out: &mut String) {
        out.push_str(&f.lines.join("\r\n"));
        let up = f.lines.len().saturating_sub(1 + f.cursor_row);
        if up > 0 {
            out.push_str(&format!("\x1b[{up}A"));
        }
        out.push('\r');
        if f.cursor_col > 0 {
            out.push_str(&format!("\x1b[{}C", f.cursor_col));
        }
        self.widths_above = f.lines[..f.cursor_row.min(f.lines.len())]
            .iter()
            .map(|l| visible_width(l))
            .collect();
        self.cursor_col = f.cursor_col;
        self.cols = cols;
        self.drawn = true;
    }

    /// Redraw the frame in place. `cols` is the current terminal width.
    pub fn draw(&mut self, f: &Frame, cols: usize) {
        let mut out = String::from("\x1b[?25l");
        self.rewind(&mut out, cols);
        self.paint_frame(f, cols, &mut out);
        out.push_str("\x1b[?25h");
        term::emit(&out);
    }

    /// Erase the frame; the cursor ends where the frame started.
    pub fn clear(&mut self, cols: usize) {
        let mut out = String::new();
        self.rewind(&mut out, cols);
        term::emit(&out);
        self.drawn = false;
        self.widths_above.clear();
    }

    /// Leave `f` in the scrollback and move the cursor below it.
    pub fn finish(&mut self, f: &Frame, cols: usize) {
        let mut out = String::from("\x1b[?25l");
        self.rewind(&mut out, cols);
        self.paint_frame(f, cols, &mut out);
        let down = f.lines.len().saturating_sub(1 + f.cursor_row);
        if down > 0 {
            out.push_str(&format!("\x1b[{down}B"));
        }
        out.push_str("\r\n\x1b[?25h");
        term::emit(&out);
        self.drawn = false;
        self.widths_above.clear();
    }
}

// ───────────── text helpers ─────────────

/// Choose what fits in the footer row. The hint segments (separated by ` · `)
/// are dropped from the end first; the right-hand label goes only when even the
/// first hint cannot sit next to it.
pub fn fit_footer(left: &str, right: &str, room: usize) -> (String, String) {
    let right = fit(right, room / 2);
    let rw = str_width(&right);
    let mut segs: Vec<&str> = left.split(" · ").collect();
    loop {
        let cand = segs.join(" · ");
        if str_width(&cand) + rw + 2 <= room {
            return (cand, right);
        }
        if segs.len() > 1 {
            segs.pop();
        } else {
            return (fit(&cand, room), String::new());
        }
    }
}

/// Truncate a string that may contain SGR escapes to a display width,
/// keeping the escapes intact and closing with a reset.
pub fn clip_ansi(s: &str, w: usize) -> String {
    if visible_width(s) <= w {
        return s.to_string();
    }
    let limit = w.saturating_sub(1);
    let mut out = String::new();
    let mut used = 0;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            out.push(c);
            if chars.peek() == Some(&'[') {
                if let Some(b) = chars.next() {
                    out.push(b);
                }
                for n in chars.by_ref() {
                    out.push(n);
                    if ('\u{40}'..='\u{7e}').contains(&n) {
                        break;
                    }
                }
            }
            continue;
        }
        let cw = char_width(c);
        if used + cw > limit {
            break;
        }
        out.push(c);
        used += cw;
    }
    out.push('…');
    out.push_str("\x1b[0m");
    out
}

/// Truncate to a display width, adding `…`.
pub fn fit(s: &str, w: usize) -> String {
    if w == 0 {
        return String::new();
    }
    if str_width(s) <= w {
        return s.to_string();
    }
    let mut out = String::new();
    let mut used = 0;
    for c in s.chars() {
        let cw = char_width(c);
        if used + cw + 1 > w {
            break;
        }
        out.push(c);
        used += cw;
    }
    out.push('…');
    out
}

fn hanging_indent(body: &str) -> usize {
    for p in ["- ", "· ", "• ", "* "] {
        if body.starts_with(p) {
            return 2;
        }
    }
    let digits = body.chars().take_while(|c| c.is_ascii_digit()).count();
    if digits > 0 && digits <= 3 {
        let rest = &body[digits..];
        if rest.starts_with(". ") || rest.starts_with(") ") {
            return digits + 2;
        }
    }
    0
}

/// Word-wrap to a display width. Lines that already fit are kept verbatim
/// (preserves aligned columns); long lines wrap with a hanging indent for
/// list items. Over-long tokens are hard-broken.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out = Vec::new();
    for raw in text.lines() {
        let raw = raw.trim_end();
        if str_width(raw) <= width {
            out.push(raw.to_string());
            continue;
        }
        let indent = raw.chars().take_while(|c| *c == ' ').count();
        let body = &raw[indent..];
        let hang = hanging_indent(body);
        let lead = " ".repeat(indent);
        let cont = " ".repeat(indent + hang);
        let mut line = lead.clone();
        let mut lw = indent;
        let mut has_word = false;
        for word in body.split(' ').filter(|w| !w.is_empty()) {
            let ww = str_width(word);
            if has_word && lw + 1 + ww > width {
                out.push(std::mem::take(&mut line));
                line = cont.clone();
                lw = indent + hang;
                has_word = false;
            }
            if ww > width.saturating_sub(lw) && !has_word {
                // hard-break a token wider than the line
                for c in word.chars() {
                    let cw = char_width(c);
                    if lw + cw > width {
                        out.push(std::mem::take(&mut line));
                        line = cont.clone();
                        lw = indent + hang;
                    }
                    line.push(c);
                    lw += cw;
                }
                has_word = true;
                continue;
            }
            if has_word {
                line.push(' ');
                lw += 1;
            }
            line.push_str(word);
            lw += ww;
            has_word = true;
        }
        out.push(line);
    }
    out
}

/// One `┌ title ──` … `└──` frame from tool output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Section {
    pub title: String,
    pub lines: Vec<String>,
}

/// Tools hand back pre-framed text (`┌ name ──`, `│ line`, `└──`). Split it into
/// sections so each tool call gets its own block. Blank lines between frames
/// and leading/trailing blanks inside a frame are dropped.
pub fn split_frames(text: &str) -> Vec<Section> {
    let mut out: Vec<Section> = Vec::new();
    let mut cur: Option<Section> = None;
    for line in text.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix('┌') {
            if let Some(c) = cur.take() {
                out.push(c);
            }
            let title = rest.trim().trim_matches('─').trim().to_string();
            cur = Some(Section {
                title,
                lines: Vec::new(),
            });
        } else if t.starts_with('└') {
            if let Some(c) = cur.take() {
                out.push(c);
            }
        } else if let Some(sec) = cur.as_mut() {
            let l = match t.strip_prefix('│') {
                Some(r) => r.strip_prefix(' ').unwrap_or(r),
                None => line,
            };
            sec.lines.push(l.trim_end().to_string());
        }
    }
    if let Some(c) = cur.take() {
        out.push(c);
    }
    for sec in &mut out {
        while sec.lines.last().is_some_and(|l| l.trim().is_empty()) {
            sec.lines.pop();
        }
        while sec.lines.first().is_some_and(|l| l.trim().is_empty()) {
            sec.lines.remove(0);
        }
    }
    out
}

fn fmt_uptime(secs: u64) -> String {
    if secs >= 3600 {
        format!("{}h {:02}m", secs / 3600, (secs % 3600) / 60)
    } else if secs >= 60 {
        format!("{}m {:02}s", secs / 60, secs % 60)
    } else {
        format!("{secs}s")
    }
}

fn shorten_path(p: PathBuf) -> Option<String> {
    let s = p.to_string_lossy().to_string();
    if let Ok(home) = env::var("HOME") {
        if !home.is_empty() && s.starts_with(&home) {
            return Some(s.replacen(&home, "~", 1));
        }
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain() -> Theme {
        Theme {
            color: false,
            cwd: "~/proj".into(),
        }
    }

    fn footer() -> Footer {
        Footer {
            left: "/ commands".into(),
            right: "model".into(),
            warn: false,
        }
    }

    #[test]
    fn wrap_keeps_short_lines_and_breaks_long_ones() {
        assert_eq!(wrap("a   b", 20), vec!["a   b"]);
        let w = wrap("the quick brown fox jumps over the lazy dog", 16);
        assert!(w.iter().all(|l| str_width(l) <= 16), "{w:?}");
        assert_eq!(w.join(" "), "the quick brown fox jumps over the lazy dog");
    }

    #[test]
    fn wrap_hangs_list_items() {
        let w = wrap("- alpha beta gamma delta epsilon zeta", 14);
        assert!(w[0].starts_with("- "));
        assert!(w[1].starts_with("  "), "{w:?}");
    }

    #[test]
    fn wrap_hard_breaks_long_tokens() {
        let w = wrap(&"x".repeat(40), 16);
        assert!(w.len() >= 3);
        assert!(w.iter().all(|l| str_width(l) <= 16));
    }

    #[test]
    fn split_frames_parses_sections() {
        let v = split_frames("┌ calc ────\n│ 2+2 = 4\n│ extra\n└────\n\n┌ time ──\n│ noon\n└──");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].title, "calc");
        assert_eq!(v[0].lines, vec!["2+2 = 4", "extra"]);
        assert_eq!(v[1].title, "time");
        assert!(split_frames("plain text").is_empty());
    }

    #[test]
    fn split_frames_keeps_inner_indent_and_trims_blank_edges() {
        let v = split_frames("┌ x ──\n│\n│   indented\n│ \n└──");
        assert_eq!(v[0].lines, vec!["  indented"]);
    }

    fn lines(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("line {i}")).collect()
    }

    #[test]
    fn tool_block_collapses_long_output() {
        let t = plain();
        let c = t.render_tool("news", "headlines", &lines(30), Some(1.2), 80, true);
        assert!(c.contains("● news(headlines)  1.2s"));
        assert!(c.contains("⎿  line 0"));
        assert!(c.contains("… +16 lines (ctrl+o to expand)"), "{c}");
        let full = t.render_tool("news", "headlines", &lines(30), Some(1.2), 80, false);
        assert!(full.contains("line 29"));
        assert!(!full.contains("ctrl+o"));
    }

    #[test]
    fn tool_block_without_time_and_empty_body() {
        let t = plain();
        let c = t.render_tool("x", "q", &[], None, 80, true);
        assert_eq!(c.lines().next(), Some("● x(q)"));
        assert!(c.contains("(no output)"));
    }

    #[test]
    fn prompt_frame_places_cursor() {
        let t = plain();
        let buf: Vec<char> = "hello".chars().collect();
        let f = t.prompt_frame(60, &buf, 5, &[], 0, &footer(), false);
        assert_eq!(f.cursor_row, 1);
        assert_eq!(f.cursor_col, 4 + 5);
        assert_eq!(f.lines.len(), 4); // top, input, bottom, footer
        assert!(f.lines[1].contains("❯ hello"));
        // every box line is exactly the box width
        for l in &f.lines[..3] {
            assert_eq!(visible_width(l), 60, "{l}");
        }
    }

    #[test]
    fn prompt_frame_wraps_and_tracks_cursor_row() {
        let t = plain();
        let buf: Vec<char> = "a".repeat(100).chars().collect();
        let f = t.prompt_frame(40, &buf, 100, &[], 0, &footer(), false);
        assert!(f.cursor_row >= 3, "row {}", f.cursor_row);
        for l in &f.lines[..f.lines.len() - 1] {
            assert!(visible_width(l) <= 40);
        }
    }

    #[test]
    fn prompt_frame_empty_shows_placeholder_and_menu_replaces_footer() {
        let t = plain();
        let f = t.prompt_frame(60, &[], 0, &[], 0, &footer(), false);
        assert!(f.lines[1].contains("Ask anything"));
        assert_eq!((f.cursor_row, f.cursor_col), (1, 4));
        let menu = [("/help", "commands"), ("/history", "turns")];
        let f = t.prompt_frame(60, &['/', 'h'], 2, &menu, 1, &footer(), false);
        assert_eq!(f.lines.len(), 5);
        assert!(f.lines[4].contains("▸") && f.lines[4].contains("/history"));
        assert!(!f.lines.iter().any(|l| l.contains("model")));
    }

    #[test]
    fn submitted_frame_has_no_footer() {
        let t = plain();
        let f = t.prompt_frame(60, &['x'], 1, &[], 0, &footer(), true);
        assert_eq!(f.lines.len(), 3);
    }

    #[test]
    fn devanagari_cursor_uses_display_width() {
        let t = plain();
        let buf: Vec<char> = "नमस्ते".chars().collect();
        let f = t.prompt_frame(60, &buf, buf.len(), &[], 0, &footer(), false);
        assert_eq!(f.cursor_col, 4 + 4);
    }

    #[test]
    fn welcome_and_help_render() {
        let t = plain();
        let w = t.render_welcome(80, "local / tools-first", false);
        assert!(w.contains("Welcome to 0zcode"));
        assert!(w.contains("local / tools-first"));
        let h = t.render_help(80);
        assert!(h.contains("/expand") && h.contains("ctrl+o"));
        for l in h.lines() {
            assert!(visible_width(l) <= 80, "{l}");
        }
    }

    #[test]
    fn fit_truncates_by_width() {
        assert_eq!(fit("abcdef", 4), "abc…");
        assert_eq!(fit("abc", 4), "abc");
    }

    #[test]
    fn panels_never_exceed_terminal_width() {
        let colored = Theme {
            color: true,
            cwd: "~/a/very/long/working/directory/path/that/keeps/going/and/going".into(),
        };
        for t in [plain(), colored] {
            for width in [24usize, 30, 40, 50, 80, 120] {
                let limit = width.clamp(MIN_W, 100);
                let panels = [
                    t.render_welcome(width, "a-rather-long-model-label / tools-first", true),
                    t.render_help(width),
                    t.render_status(width, "m", "agent", false, 3, "calc", 3700, "/etc/x/config"),
                ];
                for p in panels {
                    for l in p.lines() {
                        assert!(
                            visible_width(l) <= limit,
                            "width {width}: {} > {limit}: {l:?}",
                            visible_width(l)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn boxed_rows_are_equal_width() {
        let t = plain();
        let out = t.render_help(60);
        let widths: Vec<_> = out.lines().map(visible_width).collect();
        assert!(widths.iter().all(|w| *w == 60), "{widths:?}");
    }

    #[test]
    fn clip_ansi_keeps_escapes_and_width() {
        let s = "\x1b[31mhello world\x1b[0m";
        let c = clip_ansi(s, 6);
        assert_eq!(visible_width(&c), 6);
        assert!(c.starts_with("\x1b[31m") && c.ends_with("\x1b[0m"));
        assert_eq!(clip_ansi("short", 10), "short");
    }

    #[test]
    fn footer_drops_hints_before_the_label() {
        let left = "/ commands · ctrl+d exit · ctrl+o expand · ctrl+c clear";
        let (l, r) = fit_footer(left, "local / tools-first · offline", 78);
        assert_eq!(l, "/ commands · ctrl+d exit · ctrl+o expand");
        assert_eq!(r, "local / tools-first · offline");
        let (l, r) = fit_footer(left, "local", 120);
        assert_eq!((l.as_str(), r.as_str()), (left, "local"));
        // too narrow for anything but the first hint
        let (l, r) = fit_footer(left, "local / tools-first · offline", 20);
        assert!(str_width(&l) + str_width(&r) < 20, "{l:?} {r:?}");
        assert!(l.starts_with("/ commands"), "{l:?}");
    }

    #[test]
    fn reflow_estimate_on_narrowing() {
        // widening or same width: plain row count
        assert_eq!(rows_above_cursor(&[80, 80, 80], 4, 80, 120), 3);
        assert_eq!(rows_above_cursor(&[80, 80, 80], 4, 80, 80), 3);
        // 80 → 40: each full line wraps to 2 rows, cursor col 4 stays on first
        assert_eq!(rows_above_cursor(&[80, 80, 80], 4, 80, 40), 6);
        // 80 → 30: ceil(80/30)=3 rows each, cursor col 65 sits 2 rows down
        assert_eq!(rows_above_cursor(&[80, 10], 65, 80, 30), 3 + 1 + 2);
        // empty lines still occupy a row
        assert_eq!(rows_above_cursor(&[0, 0], 0, 80, 40), 2);
    }
}
