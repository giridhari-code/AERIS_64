//! Typed errors for 0z runtime — clear, user-facing messages.

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Http(String),
    Parse(String),
    Tool(String),
    Policy(String),
    Io(String),
    Offline(String),
    /// AERIS-64 / SENTINEL LLM link (own prefix: not an internet problem).
    Llm(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Http(m) => write!(
                f,
                "network error — {m} (check internet or set ZEROZ_OFFLINE=1 for calc-only)"
            ),
            Error::Parse(m) => write!(f, "could not parse response — {m}"),
            Error::Tool(m) => write!(f, "tool error — {m}"),
            Error::Policy(m) => write!(f, "policy — {m}"),
            Error::Io(m) => write!(f, "io — {m}"),
            Error::Offline(m) => write!(f, "offline mode — {m}"),
            Error::Llm(m) => write!(f, "AERIS-64 — {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}
