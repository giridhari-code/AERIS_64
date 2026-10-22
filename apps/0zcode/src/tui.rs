//! 0zcode TUI — Claude Code / OpenCode visual language (ANSI, std only).
//!
//! Layout: header · activity · $ tool panel · ● answer · ❯ input
//! Colors: soft dark UI — muted gray text, one accent, no loud boxes.

use crate::config::Config;
use crate::policy::PRODUCT_VERSION;
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;

pub struct Theme {
    pub color: bool,
    pub cwd: String,
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

    fn paint(&self, code: &str, s: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    pub fn clear_screen(&self) {
        print!("\x1b[2J\x1b[H");
        let _ = io::stdout().flush();
    }

    /// Claude-style header: mark · name · meta · path
    pub fn print_header(&self) {
        let mark = self.paint("33;1", "▣");
        let name = self.paint("37;1", "0zcode");
        let ver = self.paint("90", &format!("v{PRODUCT_VERSION}"));
        println!();
        println!("  {mark}  {name}  {ver}");
        println!(
            "  {}",
            self.paint("90", "tools-first · zero source · research agent")
        );
        println!("  {}", self.paint("36", &self.cwd));
        println!();
        println!(
            "  {}",
            self.paint("90", "Session ready.  /help  /tools  /mcp  /status  /model")
        );
        println!();
    }

    /// OpenCode/Claude prompt: ❯
    pub fn prompt(&self) -> String {
        format!("{} ", self.paint("37;1", "❯"))
    }

    /// Dim activity: → Explored · 1 tool(s)
    pub fn print_activity(&self, kind: &str, detail: &str) {
        println!(
            "  {} {}",
            self.paint("90", "→"),
            self.paint("90", &format!("{kind}: {detail}"))
        );
    }

    pub fn print_thought(&self, ms_label: &str) {
        println!(
            "  {} {}",
            self.paint("33", "+"),
            self.paint("33", &format!("Thought · {ms_label}"))
        );
    }

    /// ● assistant body (Claude answer style)
    pub fn print_assistant(&self, text: &str) {
        let mut first = true;
        for line in text.lines() {
            let t = line.trim_end();
            // skip raw box-drawing from tool strings
            if t.starts_with('┌')
                || t.starts_with('└')
                || t.starts_with('─')
                || t.chars().all(|c| c == '─' || c == ' ' || c == '│')
            {
                continue;
            }
            let body = t
                .strip_prefix('│')
                .map(str::trim_start)
                .unwrap_or(t);
            if body.is_empty() {
                println!();
                continue;
            }
            if first {
                println!(
                    "  {} {}",
                    self.paint("37;1", "●"),
                    self.paint("37", body)
                );
                first = false;
            } else if body.starts_with("- ") || body.starts_with("· ") {
                println!("    {}", self.paint("37", body));
            } else {
                println!("  {}", self.paint("37", body));
            }
        }
        println!();
    }

    /// Soft tool panel: $ name + gray body (OpenCode shell block)
    pub fn print_tool_box(&self, tool: &str, body: &str) {
        println!();
        println!(
            "  {} {}",
            self.paint("32;1", "$"),
            self.paint("32;1", tool)
        );
        for line in body.lines() {
            let mut l = line.trim();
            if let Some(s) = l.strip_prefix('│') {
                l = s.trim();
            }
            if l.starts_with('┌')
                || l.starts_with('└')
                || l.starts_with('─')
                || l.starts_with('┃')
                || l.is_empty()
            {
                continue;
            }
            let clean: String = l
                .chars()
                .filter(|c| *c != '│' && *c != '┃')
                .collect::<String>()
                .trim()
                .to_string();
            if clean.is_empty() {
                continue;
            }
            println!("  {}", self.paint("90", &clean));
        }
        println!();
    }

    pub fn print_ok(&self, msg: &str) {
        println!(
            "  {} {}",
            self.paint("32", "✔"),
            self.paint("37", msg)
        );
    }

    pub fn print_err(&self, msg: &str) {
        let mut lines = msg.lines();
        if let Some(first) = lines.next() {
            println!(
                "  {} {}",
                self.paint("31;1", "●"),
                self.paint("31", first)
            );
        }
        for line in lines {
            println!("  {}", self.paint("31", line));
        }
    }

    pub fn print_progress(&self, label: &str) {
        print!(
            "  {} {}",
            self.paint("33", "+"),
            self.paint("33", &format!("Thought · {label}"))
        );
        let _ = io::stdout().flush();
    }

    pub fn clear_progress_line(&self) {
        print!("\r\x1b[2K");
        let _ = io::stdout().flush();
    }

    pub fn print_status_rich(
        &self,
        model: &str,
        mode: &str,
        offline: bool,
        color: bool,
        turns: usize,
        last_tool: &str,
        uptime_secs: u64,
        config_path: &str,
    ) {
        println!();
        println!("  {}", self.paint("37;1", "Status"));
        println!(
            "  {}",
            self.paint(
                "90",
                &format!(
                    "model {model} · mode {mode} · offline {} · color {}",
                    if offline { "yes" } else { "no" },
                    if color { "on" } else { "off" }
                )
            )
        );
        println!(
            "  {}",
            self.paint(
                "90",
                &format!(
                    "turns {turns} · last {} · up {uptime_secs}s",
                    if last_tool.is_empty() {
                        "—"
                    } else {
                        last_tool
                    }
                )
            )
        );
        println!("  {}", self.paint("90", &format!("cwd  {}", self.cwd)));
        println!("  {}", self.paint("90", &format!("cfg  {config_path}")));
        println!();
    }

    pub fn print_status(&self, model: &str, files: u32, mode: &str) {
        self.print_status_rich(
            model,
            mode,
            false,
            self.color,
            files as usize,
            "",
            0,
            "",
        );
    }

    pub fn print_help_panel(&self) {
        println!();
        println!("  {}", self.paint("37;1", "Commands"));
        for (c, d) in [
            ("/help", "this panel"),
            ("/tools", "built-in tools"),
            ("/mcp", "configure user tools"),
            ("/history", "session turns"),
            ("/status", "session status"),
            ("/model", "display label"),
            ("/copy", "save last reply"),
            ("/clear", "clear screen"),
            ("/quit", "exit"),
        ] {
            println!(
                "  {}  {}",
                self.paint("36", &format!("{c:<10}")),
                self.paint("90", d)
            );
        }
        println!();
        println!(
            "  {}",
            self.paint(
                "90",
                "Examples: time · weather in Mumbai · 938+29 · /help"
            )
        );
        println!();
    }

    pub fn print_footer_hint(&self, model: &str) {
        println!(
            "  {}",
            self.paint(
                "90",
                &format!("tools-first · ? help · {model} · coding disabled")
            )
        );
    }
}

fn shorten_path(p: PathBuf) -> Option<String> {
    let s = p.to_string_lossy().to_string();
    if let Ok(home) = env::var("HOME") {
        if s.starts_with(&home) {
            return Some(s.replacen(&home, "~", 1));
        }
    }
    Some(s)
}
