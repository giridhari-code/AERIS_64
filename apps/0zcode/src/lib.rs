//! # 0z code
//!
//! Company-standard **non-coding** agent runtime.
//!
//! - Policy-gated requests (no source-code generation)
//! - Deterministic tool routing
//! - Network tools over plain HTTP (no third-party crates)
//!
//! Binary entry: `0zcode`.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]

pub mod agent;
pub mod config;
pub mod error;
pub mod history;
pub mod locale;
pub mod registry;
pub mod notes;
pub mod mcp_store;
pub mod http;
pub mod json_util;
pub mod policy;
pub mod tools;
pub mod tui;

pub use agent::{handle, AgentResponse};
pub use config::Config;
pub use error::{Error, Result};
pub use policy::{PRODUCT_NAME, PRODUCT_TAGLINE, PRODUCT_VERSION};

/// Library version (crate).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
