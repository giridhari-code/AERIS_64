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
    /// Optional AERIS / SENTINEL chat endpoint (e.g. http://127.0.0.1:8000).
    /// When set and offline=false, free-form questions can be routed to the LLM.
    pub aeris_url: Option<String>,
    /// `X-API-Key` for a server started with `NEUROFIELD_API_KEYS`.
    pub aeris_key: Option<String>,
    /// Give up when the server sends nothing for this long (model can be slow).
    pub aeris_timeout: Duration,
    /// `max_new_tokens` sent with every chat request (server caps at 8192).
    pub aeris_max_tokens: u32,
    /// Stream tokens live (`/v1/chat/stream`) instead of one reply (`/v1/chat`).
    pub aeris_stream: bool,
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
            model_label: "AERIS-64 · tools-first".into(),
            aeris_url: None,
            aeris_key: None,
            aeris_timeout: Duration::from_secs(60),
            aeris_max_tokens: 256,
            aeris_stream: true,
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

fn non_empty(s: &str) -> Option<String> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
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
            "aeris_url" => {
                let t = v.trim();
                c.aeris_url = if t.is_empty() { None } else { Some(t.to_string()) };
            }
            "aeris_key" => c.aeris_key = non_empty(v),
            "aeris_timeout_secs" => {
                if let Ok(n) = v.parse::<u64>() {
                    c.aeris_timeout = Duration::from_secs(n.clamp(1, 600));
                }
            }
            "aeris_max_tokens" => {
                if let Ok(n) = v.parse::<u32>() {
                    c.aeris_max_tokens = n.clamp(1, 8192);
                }
            }
            "aeris_stream" => c.aeris_stream = v == "true" || v == "1",
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
    if let Ok(s) = env::var("ZEROZ_AERIS_URL") {
        let t = s.trim();
        c.aeris_url = if t.is_empty() { None } else { Some(t.to_string()) };
    }
    if let Ok(s) = env::var("ZEROZ_AERIS_KEY") {
        c.aeris_key = non_empty(&s);
    }
    if let Ok(s) = env::var("ZEROZ_AERIS_TIMEOUT_SECS") {
        if let Ok(n) = s.parse::<u64>() {
            c.aeris_timeout = Duration::from_secs(n.clamp(1, 600));
        }
    }
    if let Ok(s) = env::var("ZEROZ_AERIS_MAX_TOKENS") {
        if let Ok(n) = s.parse::<u32>() {
            c.aeris_max_tokens = n.clamp(1, 8192);
        }
    }
    if let Ok(s) = env::var("ZEROZ_AERIS_STREAM") {
        c.aeris_stream = !(s == "0" || s.eq_ignore_ascii_case("false"));
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

    #[test]
    fn parses_aeris_settings() {
        let mut c = Config::default();
        assert!(c.aeris_stream && c.aeris_key.is_none());
        apply_toml_lite(
            &mut c,
            "aeris_url = \"http://127.0.0.1:8000\"\naeris_key = \"k1\"\naeris_timeout_secs = 5\naeris_max_tokens = 99999\naeris_stream = false\n",
        );
        assert_eq!(c.aeris_url.as_deref(), Some("http://127.0.0.1:8000"));
        assert_eq!(c.aeris_key.as_deref(), Some("k1"));
        assert_eq!(c.aeris_timeout, Duration::from_secs(5));
        assert_eq!(c.aeris_max_tokens, 8192);
        assert!(!c.aeris_stream);
    }
}
