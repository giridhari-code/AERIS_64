//! 0zcode TUI — lively terminal chrome (ANSI only, no extra crates).

use crate::config::Config;
use crate::policy::{PRODUCT_TAGLINE, PRODUCT_VERSION};
use std::env;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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

    pub fn print_header(&self) {
        // Brand row
        let bot = self.paint("36;1", "◉");
        let brand = self.paint("37;1", "0zcode");
        let ver = self.paint("36", &format!("v{PRODUCT_VERSION}"));
        println!();
        println!("  {bot}  {brand}  {ver}");
        println!(
            "     {}",
            self.paint("90", &format!("AI agent · {PRODUCT_TAGLINE}"))
        );
        println!("     {}", self.paint("36", &self.cwd));
        println!();

        // Rotating tip — makes the shell feel alive
        let tip = tip_of_session();
        println!(
            "  {} {}",
            self.paint("33;1", "✦"),
            self.paint("33", tip)
        );
        println!();
        println!(
            "  {}",
            self.paint("90", "Type anything · tools answer · no code gen")
        );
        println!();
    }

    pub fn prompt(&self) -> String {
        format!(
            "{}{}{} ",
            self.paint("36;1", "❯"),
            self.paint("90", ""),
            ""
        )
    }

    pub fn print_assistant(&self, text: &str) {
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if i == 0 {
                println!(
                    "  {} {}",
                    self.paint("32;1", "●"),
                    self.paint("37", line)
                );
            } else {
                println!("    {}", self.paint("37", line));
            }
        }
    }

    pub fn print_tool_box(&self, tool: &str, body: &str) {
        // Color by tool family
        let accent = match tool {
            t if t.contains("calc") || t.contains("units") => "33", // amber
            t if t.contains("weather") || t.contains("time") => "36", // cyan
            t if t.contains("error") || t.contains("mcp") => "35", // magenta
            t if t.contains("wiki") || t.contains("search") => "34", // blue
            _ => "33",
        };
        let title = format!(" {tool} ");
        let bar_len = 40usize.saturating_sub(title.len());
        let top = format!("┌{title}{}", "─".repeat(bar_len));
        println!();
        println!("  {}", self.paint(&format!("{accent};1"), &top));

        // Prefer plain content lines; strip old box artifacts if present
        for line in body.lines() {
            let mut l = line.trim_start();
            if let Some(s) = l.strip_prefix('│') {
                l = s.trim_start();
            }
            if l.starts_with('┌') || l.starts_with('└') || l.starts_with('─') {
                continue;
            }
            if l.is_empty() {
                continue;
            }
            println!(
                "  {} {}",
                self.paint(accent, "│"),
                self.paint("37", l)
            );
        }
        println!(
            "  {}",
            self.paint(accent, &format!("└{}", "─".repeat(40)))
        );
    }

    pub fn print_ok(&self, msg: &str) {
        println!(
            "  {} {}",
            self.paint("32;1", "✔"),
            self.paint("32", msg)
        );
    }

    pub fn print_err(&self, msg: &str) {
        // Soften long policy walls — first line bold
        let mut lines = msg.lines();
        if let Some(first) = lines.next() {
            println!(
                "  {} {}",
                self.paint("31;1", "✖"),
                self.paint("31;1", first)
            );
        }
        for line in lines {
            println!("    {}", self.paint("31", line));
        }
    }

    pub fn print_progress(&self, label: &str) {
        print!(
            "  {} {}",
            self.paint("36", "⋯"),
            self.paint("90", label)
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
        println!("  {}", self.paint("36;1", "status"));
        println!("  {}", self.paint("90", &"─".repeat(40)));
        row(self, "model", model);
        row(self, "workspace", &self.cwd);
        row(self, "mode", mode);
        row(
            self,
            "offline",
            if offline { "yes" } else { "no" },
        );
        row(self, "color", if color { "on" } else { "off" });
        row(self, "turns", &turns.to_string());
        row(
            self,
            "last tool",
            if last_tool.is_empty() { "—" } else { last_tool },
        );
        row(self, "uptime", &format!("{uptime_secs}s"));
        row(self, "config", config_path);
        println!("  {}", self.paint("90", &"─".repeat(40)));
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
        println!("  {}", self.paint("37;1", "shortcuts"));
        println!("  {}", self.paint("90", &"─".repeat(40)));
        let rows = [
            ("?", "/help", "this panel"),
            ("", "/tools", "built-in tools"),
            ("", "/mcp", "your MCP tools"),
            ("", "/history", "this session"),
            ("", "/status", "session stats"),
            ("", "/copy", "save last reply"),
            ("", "/clear", "fresh screen"),
            ("q", "/quit", "exit"),
        ];
        for (key, cmd, desc) in rows {
            if key.is_empty() {
                println!(
                    "      {}  {}",
                    self.paint("36;1", cmd),
                    self.paint("90", desc)
                );
            } else {
                println!(
                    "  {}   {}  {}",
                    self.paint("33;1", key),
                    self.paint("36;1", cmd),
                    self.paint("90", desc)
                );
            }
        }
        println!();
        println!(
            "  {}",
            self.paint("33", "try:  hi  ·  time  ·  938+29  ·  weather in Mumbai")
        );
        println!();
    }
}

fn row(theme: &Theme, k: &str, v: &str) {
    println!(
        "  {}  {}",
        theme.paint("90", &format!("{k:<10}")),
        theme.paint("37", v)
    );
}

fn tip_of_session() -> &'static str {
    const TIPS: &[&str] = &[
        "tip: 938+29 works inside a sentence",
        "tip: /mcp add email inbox,mail https://…",
        "tip: note save ideas ship faster",
        "tip: time works offline if the net drops",
        "tip: /copy saves the last answer",
        "tip: weather in Delhi · btc price · headlines",
        "tip: say hi — I won't lecture you",
    ];
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as usize)
        .unwrap_or(0);
    TIPS[n % TIPS.len()]
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
