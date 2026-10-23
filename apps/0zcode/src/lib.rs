//! # 0z code — company-standard non-coding agent.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]

pub mod aeris_boot;
pub mod agent;
pub mod config;
pub mod editor;
pub mod error;
pub mod history;
pub mod http;
pub mod json_util;
pub mod llm;
pub mod locale;
pub mod mcp_store;
pub mod notes;
pub mod policy;
pub mod registry;
pub mod repl;
pub mod term;
pub mod tools;
pub mod workspace;
pub mod harness;
pub mod tui;

pub use agent::{handle, handle_with, AgentResponse, ChatCtx};
pub use config::Config;
pub use error::{Error, Result};
pub use policy::{PRODUCT_NAME, PRODUCT_TAGLINE, PRODUCT_VERSION};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
