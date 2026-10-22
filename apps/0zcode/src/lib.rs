//! # 0z code — company-standard non-coding agent.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]

pub mod agent;
pub mod config;
pub mod error;
pub mod history;
pub mod http;
pub mod json_util;
pub mod locale;
pub mod mcp_store;
pub mod notes;
pub mod policy;
pub mod registry;
pub mod theme;
pub mod tools;
pub mod session;
pub mod ui;

pub use agent::{handle, AgentResponse};
pub use config::Config;
pub use error::{Error, Result};
pub use policy::{PRODUCT_NAME, PRODUCT_TAGLINE, PRODUCT_VERSION};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
