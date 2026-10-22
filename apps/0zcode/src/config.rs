//! Runtime configuration: env + optional `~/.config/0zcode/config.toml`.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct Config {
    pub product_name: String,
    pub http_timeout: Duration,
    pub user_agent: String,
    pub max_output_chars: usize,
    pub color: bool,
    pub model_label: String,
    /// Skip network tools; calculator + meta still work.
    pub offline: bool,
    pub history_size: usize,
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
            offline: false,
            history_size: 40,
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        // File first, then env overrides (12-factor-ish)
        if let Some(path) = config_path() {
            if path.is_file() {
                if let Ok(text) = fs::read_to_string(&path) {
                    apply_toml_lite(&mut c, &text);
                }
            }
        }
        apply_env(&mut c);
        c
    }

    pub fn config_path_display() -> String {
        config_path()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "~/.config/0zcode/config.toml".into())
    }
}

fn config_path() -> Option<PathBuf> {
    if let Ok(p) = env::var("ZEROZ_CONFIG") {
        return Some(PathBuf::from(p));
    }
    let home = env::var("HOME").ok()?;
    Some(PathBuf::from(home).join(".config/0zcode/config.toml"))
}

/// Minimal key = value / key = "str" parser (no serde).
fn apply_toml_lite(c: &mut Config, text: &str) {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let k = k.trim();
        let v = v.trim().trim_matches('"').trim_matches('\'');
        match k {
            "http_timeout_secs" => {
                if let Ok(n) = v.parse::<u64>() {
                    c.http_timeout = Duration::from_secs(n.clamp(1, 60));
                }
            }
            "max_output_chars" => {
                if let Ok(n) = v.parse::<usize>() {
                    c.max_output_chars = n.clamp(500, 50_000);
                }
            }
            "color" => c.color = v == "true" || v == "1",
            "offline" => c.offline = v == "true" || v == "1",
            "model_label" => {
                if !v.is_empty() {
                    c.model_label = v.to_string();
                }
            }
            "user_agent" => {
                if !v.is_empty() {
                    c.user_agent = v.to_string();
                }
            }
            "history_size" => {
                if let Ok(n) = v.parse::<usize>() {
                    c.history_size = n.clamp(4, 200);
                }
            }
            _ => {}
        }
    }
}

fn apply_env(c: &mut Config) {
    if env::var("NO_COLOR").is_ok() || env::var("ZEROZ_NO_COLOR").is_ok() {
        c.color = false;
    }
    if env::var("ZEROZ_COLOR")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
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
    if env::var("ZEROZ_OFFLINE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        c.offline = true;
    }
    if let Ok(s) = env::var("ZEROZ_HISTORY_SIZE") {
        if let Ok(n) = s.parse::<usize>() {
            c.history_size = n.clamp(4, 200);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_timeout_12s() {
        assert_eq!(Config::default().http_timeout, Duration::from_secs(12));
    }

    #[test]
    fn parses_lite_toml() {
        let mut c = Config::default();
        apply_toml_lite(
            &mut c,
            "offline = true\nhistory_size = 12\nmodel_label = \"demo\"\n",
        );
        assert!(c.offline);
        assert_eq!(c.history_size, 12);
        assert_eq!(c.model_label, "demo");
    }
}
