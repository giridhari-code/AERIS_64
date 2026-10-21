//! Typed errors for 0z runtime.

use std::fmt;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Http(String),
    Parse(String),
    Tool(String),
    Policy(String),
    Io(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Http(m) => write!(f, "http: {m}"),
            Error::Parse(m) => write!(f, "parse: {m}"),
            Error::Tool(m) => write!(f, "tool: {m}"),
            Error::Policy(m) => write!(f, "policy: {m}"),
            Error::Io(m) => write!(f, "io: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}
