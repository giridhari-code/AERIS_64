//! 0zcode TUI — Claude Code / OpenCode–grade chrome (ANSI, std only).
//! Layout language: header · activity · code panels · status footer.

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

    /// Claude-style header: mark · model line · path
    pub fn print_header(&self) {
        // Nested-square mark (product glyph) + name
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
            self.paint(
                "90",
                "Session ready.  /help  /tools  /mcp  /status  /model"
            )
        );
        println!();
    }

    /// OpenCode/Claude prompt: ❯
    pub fn prompt(&self) -> String {
        format!("{} ", self.paint("37;1", "❯"))
    }

    /// Dim activity line: → Explored / Ran tool
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

    pub fn print_assistant(&self, text: &str) {
        let mut first = true;
        for line in text.lines() {
            if first {
                println!(
                    "  {} {}",
                    self.paint("37;1", "●"),
                    self.paint("37", line)
                );
                first = false;
            } else if line.starts_with("- ") || line.starts_with("· ") {
                println!("    {}", self.paint("37", line));
            } else {
                println!("  {}", self.paint("37", line));
            }
        }
    }

    /// Shell/code panel like OpenCode `$ ls` blocks
    pub fn print_tool_box(&self, tool: &str, body: &str) {
        println!();
        // Meta line (Claude: "Read 2 files, ran 1 shell command")
        println!(
            "  {}",
            self.paint("90", &format!("Ran tool · {tool}"))
        );
        // Panel
        let border = self.paint("90", "────────────────────────────────────────");
        println!("  {border}");
        println!(
            "  {} {}",
            self.paint("32;1", "$"),
            self.paint("32;1", tool)
        );
        for line in body.lines() {
            let mut l = line.trim_start();
            if let Some(s) = l.strip_prefix('│') {
                l = s.trim_start();
            }
            if l.starts_with('┌')
                || l.starts_with('└')
                || l.starts_with('─')
                || l.starts_with('┃')
                || l.starts_with('┖')
            {
                continue;
            }
            if l.is_empty() {
                continue;
            }
            // Tree-style continuation like Claude └
            println!("  {}", self.paint("37", l));
        }
        println!("  {border}");
    }

    pub fn print_ok(&self, msg: &str) {
        println!(
            "  {} {}",
            self.paint("32", "✔"),
            self.paint("32", msg)
        );
    }

    pub fn print_err(&self, msg: &str) {
        let mut lines = msg.lines();
        if let Some(first) = lines.next() {
            println!(
                "  {} {}",
                self.paint("31;1", "●"),
                self.paint("31;1", first)
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
                    if last_tool.is_empty() { "—" } else { last_tool }
                )
            )
        );
        println!("  {}", self.paint("90", &format!("cwd  {}", self.cwd)));
        println!("  {}", self.paint("90", &format!("cfg  {config_path}")));
        println!();
    }

    pub fn print_status(&self, model: &str, files: u32, mode: &str) {
        self.print_status_rich(model, mode, false, self.color, files as usize, "", 0, "");
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
            self.paint("90", "Examples: time · weather in Mumbai · 938+29 · headlines")
        );
        println!();
    }

    /// Footer strip (OpenCode-style session line)
    pub fn print_footer_hint(&self, model: &str) {
        println!(
            "  {}",
            self.paint(
                "90",
                &format!("──  {model}  ·  ? help  ·  esc not required  ·  0zcode  ──")
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
