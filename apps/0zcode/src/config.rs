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
}

impl Default for Config {
    fn default() -> Self {
        Self {
            product_name: "0z code".into(),
            http_timeout: Duration::from_secs(12),
            user_agent: format!("0zcode/{}", env!("CARGO_PKG_VERSION")),
            max_output_chars: 8_000,
            color: env::var("NO_COLOR").is_err(),
        }
    }
}

impl Config {
    pub fn from_env() -> Self {
        let mut c = Self::default();
        if let Ok(s) = env::var("ZEROZ_HTTP_TIMEOUT_SECS") {
            if let Ok(n) = s.parse::<u64>() {
                c.http_timeout = Duration::from_secs(n.clamp(1, 60));
            }
        }
        if env::var("ZEROZ_NO_COLOR").is_ok() {
            c.color = false;
        }
        c
    }
}
