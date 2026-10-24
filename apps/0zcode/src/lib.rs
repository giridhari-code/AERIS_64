//! # 0z code — company-standard non-coding agent.

#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used)]

pub mod aeris_boot;
pub mod backend;
pub mod agent;
pub mod approval;
pub mod brief;
pub mod buddy;
pub mod background;
pub mod commands;
pub mod computer;
pub mod config;
pub mod editor;
pub mod error;
pub mod grounding;
pub mod history;
pub mod http;
pub mod json_util;
pub mod llm;
pub mod locale;
pub mod lsp;
pub mod mcp_catalog;
pub mod mcp_store;
pub mod memory;
pub mod notes;
pub mod permissions;
pub mod plugins;
pub mod policy;
pub mod receipts;
pub mod registry;
pub mod repl;
pub mod rules;
pub mod term;
pub mod tools;
pub mod workspace;
pub mod harness;
pub mod subagents;
pub mod teams;
pub mod tasks;
pub mod cron;
pub mod plan_mode;
pub mod skills;
pub mod tui;

pub use agent::{handle, handle_with, AgentResponse, ChatCtx};
pub use config::Config;
pub use error::{Error, Result};
pub use policy::{PRODUCT_NAME, PRODUCT_TAGLINE, PRODUCT_VERSION};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
