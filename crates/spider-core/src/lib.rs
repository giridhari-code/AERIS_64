//! Spider-Web Looped MoE — personal-scale LLM core.
//!
//! Architecture: Prelude (2) → Recurrent Spider-MoE (×r shared) → Coda (2) → LM head.
//! Orthogonal axes: MoE capacity · looped depth · dynamic expert graph diversity.

pub mod config;
pub mod graph;
pub mod model;
pub mod moe;
pub mod stability;

pub use config::SpiderConfig;
pub use model::SpiderWebModel;

/// Crate version aligned with architecture doc v1.0 scaffold.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
