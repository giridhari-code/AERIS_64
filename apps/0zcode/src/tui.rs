//! Professional terminal presentation (ANSI, no TUI crate).

use crate::config::Config;
use crate::policy::{banner_line, PRODUCT_NAME};

pub struct Theme {
    pub color: bool,
}

impl Theme {
    pub fn from_config(cfg: &Config) -> Self {
        Self { color: cfg.color }
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

    pub fn print_header(&self) {
        let title = self.paint("33;1", PRODUCT_NAME);
        let rest = self.paint("90", &format!("— {}", banner_line().split("—").nth(1).unwrap_or("").trim()));
        let ready = self.paint("32", "ready");
        println!("{title} {rest}  |  {ready}\n");
        println!(
            "{}",
            self.paint("90", "/tools /help /version /clear /quit · coding disabled")
        );
        println!();
    }

    pub fn prompt(&self) -> String {
        format!("{} ", self.paint("32;1", "0z>"))
    }
}
