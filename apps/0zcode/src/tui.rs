//! 0zcode TUI — original “signal / zero” identity (ANSI only).
//!
//! Visual language:
//!   · Signal mark  ⌁  (not a generic robot clone)
//!   · Vertical rail instead of heavy boxes
//!   · Dim gutter + bright payload
//!   · No Claude-style chrome bars

use crate::config::Config;
use crate::policy::PRODUCT_VERSION;
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
        // Unique mark: zero + signal wave (not robot head)
        println!();
        println!(
            "  {}{}{}",
            self.paint("36;1", "0"),
            self.paint("37;1", "z"),
            self.paint("36", "⌁")
        );
        println!(
            "  {}  {}",
            self.paint("37;1", "code"),
            self.paint("90", &format!("v{PRODUCT_VERSION}"))
        );
        println!(
            "  {}",
            self.paint("90", "tools · zero source · signal clean")
        );
        println!("  {}", self.paint("36", &self.cwd));
        println!();
        let tip = tip_of_session();
        println!(
            "  {} {}",
            self.paint("35;1", "·"),
            self.paint("35", tip)
        );
        println!();
    }

    pub fn prompt(&self) -> String {
        // Unique prompt: signal caret, not ❯ clone
        format!("{} ", self.paint("36;1", "⌁"))
    }

    pub fn print_assistant(&self, text: &str) {
        let mut first = true;
        for line in text.lines() {
            if first {
                println!(
                    "  {} {}",
                    self.paint("32;1", "▸"),
                    self.paint("37", line)
                );
                first = false;
            } else {
                println!("    {}", self.paint("37", line));
            }
        }
    }

    pub fn print_tool_box(&self, tool: &str, body: &str) {
        // Vertical rail card — distinctive vs rounded Claude boxes
        let accent = tool_accent(tool);
        println!();
        println!(
            "  {}{}{}",
            self.paint(&format!("{accent};1"), "┃ "),
            self.paint(&format!("{accent};1"), tool),
            self.paint("90", " · live")
        );
        for line in body.lines() {
            let mut l = line.trim_start();
            if let Some(s) = l.strip_prefix('│') {
                l = s.trim_start();
            }
            if l.starts_with('┌') || l.starts_with('└') || l.starts_with('─') || l.starts_with('┃') {
                continue;
            }
            if l.is_empty() {
                continue;
            }
            println!(
                "  {}{}",
                self.paint(accent, "┃ "),
                self.paint("37", l)
            );
        }
        println!("  {}", self.paint(accent, "┖─"));
    }

    pub fn print_ok(&self, msg: &str) {
        println!(
            "  {} {}",
            self.paint("32;1", "▸"),
            self.paint("32", msg)
        );
    }

    pub fn print_err(&self, msg: &str) {
        let mut lines = msg.lines();
        if let Some(first) = lines.next() {
            println!(
                "  {} {}",
                self.paint("31;1", "▸"),
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
            self.paint("36", "⌁"),
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
        println!("  {}", self.paint("36;1", "⌁ status"));
        kv(self, "model", model);
        kv(self, "path", &self.cwd);
        kv(self, "mode", mode);
        kv(self, "offline", if offline { "yes" } else { "no" });
        kv(self, "color", if color { "on" } else { "off" });
        kv(self, "turns", &turns.to_string());
        kv(
            self,
            "last",
            if last_tool.is_empty() { "—" } else { last_tool },
        );
        kv(self, "up", &format!("{uptime_secs}s"));
        kv(self, "cfg", config_path);
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
        println!("  {}", self.paint("36;1", "⌁ commands"));
        let rows = [
            ("?", "help"),
            ("/tools", "built-ins"),
            ("/mcp", "your tools"),
            ("/history", "session"),
            ("/status", "stats"),
            ("/copy", "save last"),
            ("/clear", "reset"),
            ("/quit", "exit"),
        ];
        for (c, d) in rows {
            println!(
                "  {}  {}",
                self.paint("36", &format!("{c:<10}")),
                self.paint("90", d)
            );
        }
        println!();
        println!(
            "  {}",
            self.paint("35", "signal:  time  ·  938+29  ·  weather in Mumbai")
        );
        println!();
    }
}

fn tool_accent(tool: &str) -> &'static str {
    if tool.contains("calc") || tool.contains("units") {
        "33" // amber
    } else if tool.contains("weather") || tool.contains("time") {
        "36" // cyan
    } else if tool.contains("error") {
        "31"
    } else if tool.contains("mcp") || tool.contains("email") || tool.contains("notes") {
        "35" // magenta
    } else if tool.contains("wiki") || tool.contains("search") || tool.contains("github") {
        "34"
    } else {
        "36"
    }
}

fn kv(theme: &Theme, k: &str, v: &str) {
    println!(
        "  {}  {}",
        theme.paint("90", &format!("{k:<8}")),
        theme.paint("37", v)
    );
}

fn tip_of_session() -> &'static str {
    const TIPS: &[&str] = &[
        "938+29 inside a sentence still works",
        "/mcp add  →  wire your own endpoints",
        "time falls back to device clock",
        "note save  →  local sandbox only",
        "zero code is the product, not a bug",
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
