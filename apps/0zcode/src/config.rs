//! Runtime configuration (env-overridable, company standard).

use std::env;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Config {
    pub product_name: String,
    pub http_timeout: Duration,
    pub user_agent: String,
    pub max_output_chars: usize,
    pub color: bool,
    /// Display-only model label in TUI (not a backend selector).
    pub model_label: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            product_name: "0z code".into(),
            http_timeout: Duration::from_secs(12),
            user_agent: format!("0zcode/{}", env!("CARGO_PKG_VERSION")),
            max_output_chars: 8_000,
            color: true,
            model_label: "local / tools-first".into(),
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        // Color: NO_COLOR or ZEROZ_NO_COLOR disables; ZEROZ_COLOR=1 forces on
        if env::var("NO_COLOR").is_ok() || env::var("ZEROZ_NO_COLOR").is_ok() {
            c.color = false;
        }
        if env::var("ZEROZ_COLOR").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false)
        {
            c.color = true;
        }
        if let Ok(s) = env::var("ZEROZ_HTTP_TIMEOUT_SECS") {
            if let Ok(n) = s.parse::<u64>() {
                c.http_timeout = Duration::from_secs(n.clamp(1, 60));
            }
        }
        if let Ok(s) = env::var("ZEROZ_MAX_OUTPUT_CHARS") {
            if let Ok(n) = s.parse::<usize>() {
                c.max_output_chars = n.clamp(500, 50_000);
            }
        }
        if let Ok(s) = env::var("ZEROZ_MODEL_LABEL") {
            let t = s.trim();
            if !t.is_empty() {
                c.model_label = t.to_string();
            }
        }
        if let Ok(s) = env::var("ZEROZ_USER_AGENT") {
            let t = s.trim();
            if !t.is_empty() {
                c.user_agent = t.to_string();
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_timeout_12s() {
        assert_eq!(Config::default().http_timeout, Duration::from_secs(12));
    }
}
