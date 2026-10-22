//! 0zcode TUI — Claude Code–style terminal chrome (ANSI only, no extra crates).
use std::io::{self, Write};

use crate::config::Config;
use crate::policy::{PRODUCT_NAME, PRODUCT_TAGLINE, PRODUCT_VERSION};
use std::env;
use std::path::PathBuf;

pub struct Theme {
    pub color: bool,
    pub cwd: String,
}

impl Theme {
    pub fn from_config(cfg: &Config) -> Self {
        let cwd = env::current_dir()
            .ok()
            .and_then(|p| shorten_path(p))
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
    }

    /// Robot mark used in header and docs (text logo).
    pub fn logo_mark(&self) -> String {
        // Compact robot glyph block
        let robot = r#" ╭───╮
 │ ◉ ◉ │
 │  ▽  │
 ╰─┬─╯
   │"#;
        self.paint("36;1", robot)
    }

    pub fn print_logo_inline(&self) -> String {
        // Single-line brand for narrow terminals
        self.paint("36;1", "⬡ 0z")
    }

    pub fn print_header(&self) {
        println!("{}", self.logo_mark());
        let brand = self.paint("37;1", "0zcode");
        let ver = self.paint("90", &format!("v{PRODUCT_VERSION}"));
        println!("{brand}  {ver}");
        let sub = self.paint(
            "90",
            &format!("AI terminal agent · {PRODUCT_TAGLINE}"),
        );
        println!("{sub}");
        println!("{}", self.paint("36", &self.cwd));
        println!();
        println!(
            "{}",
            self.paint(
                "90",
                "────────────────────────────────────────────────"
            )
        );
        println!(
            "{}",
            self.paint(
                "90",
                "/help /tools /mcp /status /history /copy /quit  ·  ? shortcuts"
            )
        );
        println!(
            "{}",
            self.paint(
                "90",
                "────────────────────────────────────────────────"
            )
        );
        println!();
    }

    pub fn prompt(&self) -> String {
        format!("{} ", self.paint("37;1", "❯"))
    }

    pub fn print_user(&self, line: &str) {
        let chev = self.paint("90", "❯");
        let msg = self.paint("37", line);
        println!("\n{chev} {msg}");
    }

    pub fn print_assistant(&self, text: &str) {
        let dot = self.paint("32;1", "●");
        for (i, line) in text.lines().enumerate() {
            if i == 0 {
                println!("{dot} {}", self.paint("37", line));
            } else {
                println!("  {}", self.paint("37", line));
            }
        }
    }

    pub fn print_tool_box(&self, tool: &str, body: &str) {
        let title = self.paint("33;1", &format!("┌ {tool} "));
        let pad = self.paint("33", &"─".repeat(40usize.saturating_sub(tool.len())));
        println!("{title}{pad}");
        for line in body.lines() {
            println!("{} {}", self.paint("33", "│"), line);
        }
        println!("{}", self.paint("33", &format!("└{}", "─".repeat(44))));
    }

    pub fn print_ok(&self, msg: &str) {
        println!("{} {}", self.paint("32;1", "✔"), self.paint("32", msg));
    }

    pub fn print_err(&self, msg: &str) {
        println!("{} {}", self.paint("31;1", "✖"), self.paint("31", msg));
    }

    pub fn print_progress(&self, label: &str) {
        // Same-line progress; caller should print newline after work finishes.
        print!("{} {}", self.paint("90", "…"), self.paint("90", label));
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }

    pub fn clear_progress_line(&self) {
        print!("\r\x1b[2K");
        let _ = std::io::Write::flush(&mut std::io::stdout());
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
        println!("{}", self.paint("90", "─".repeat(48).as_str()));
        println!("{}", self.paint("37;1", " status"));
        println!("  {} {}", self.paint("90", "model"), self.paint("37", model));
        println!("  {} {}", self.paint("90", "workspace"), self.paint("36", &self.cwd));
        println!("  {} {}", self.paint("90", "mode"), self.paint("37", mode));
        println!(
            "  {} {}",
            self.paint("90", "offline"),
            self.paint(if offline { "33" } else { "32" }, if offline { "yes" } else { "no" })
        );
        println!(
            "  {} {}",
            self.paint("90", "color"),
            self.paint("37", if color { "on" } else { "off" })
        );
        println!("  {} {}", self.paint("90", "history turns"), self.paint("37", &turns.to_string()));
        println!(
            "  {} {}",
            self.paint("90", "last tool"),
            self.paint("37", if last_tool.is_empty() { "—" } else { last_tool })
        );
        println!("  {} {}s", self.paint("90", "uptime"), self.paint("37", &uptime_secs.to_string()));
        println!("  {} {}", self.paint("90", "config"), self.paint("90", config_path));
        println!("{}", self.paint("90", "─".repeat(48).as_str()));
    }

    pub fn print_status(&self, model: &str, files: u32, mode: &str) {
        println!();
        println!(
            "{}",
            self.paint(
                "90",
                "────────────────────────────────────────────────"
            )
        );
        let line = format!(
            " model: {model}  ·  workspace: {}  ·  mode: {mode}  ·  {PRODUCT_NAME}",
            self.cwd
        );
        println!("{}", self.paint("90", &line));
        let _ = files;
    }

    pub fn print_help_panel(&self) {
        let h = self.paint("37;1", "0zcode — shortcuts");
        println!("\n{h}\n");
        let rows = [
            ("?", "/help", "This panel"),
            ("", "/tools", "List tools"),
            ("", "/history", "Session Q&A"),
            ("", "/status", "Rich session status"),
            ("", "/copy", "Save last reply to file"),
            ("", "/model", "Display model label"),
            ("", "/version", "Version banner"),
            ("", "/clear", "Clear screen + history mark"),
            ("q", "/quit", "Exit"),
        ];
        for (key, cmd, desc) in rows {
            let left = if key.is_empty() {
                format!("    {}", self.paint("36;1", cmd))
            } else {
                format!("  {}  {}", self.paint("33;1", key), self.paint("36;1", cmd))
            };
            println!("{}  {}", left, self.paint("90", desc));
        }
        println!();
        println!(
            "{}",
            self.paint("90", "Examples: weather in Mumbai · time · 10 USD to INR · 2+2")
        );
        println!();
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
