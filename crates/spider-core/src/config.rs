//! Model hyper-parameters for Spider-Web Looped MoE.

use serde::{Deserialize, Serialize};

/// Default personal-scale / proto settings (H=512, E=8, r=6).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpiderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub ffn_mult: usize,
    /// Number of experts in the MoE bank.
    pub num_experts: usize,
    /// Top-k experts activated per token.
    pub top_k: usize,
    /// Shared recurrent loop count.
    pub num_loops: usize,
    /// Sparse neighbors per expert in the spider graph.
    pub edges_per_expert: usize,
    /// Residual scale α (variance control).
    pub residual_alpha: f32,
    /// Input reinjection β.
    pub reinject_beta: f32,
    /// Early-exit confidence threshold (inference).
    pub exit_threshold: f32,
    pub prelude_layers: usize,
    pub coda_layers: usize,
}

impl Default for SpiderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_000,
            hidden_size: 512,
            num_heads: 8,
            ffn_mult: 4,
            num_experts: 8,
            top_k: 2,
            num_loops: 6,
            edges_per_expert: 3,
            residual_alpha: 0.5,
            reinject_beta: 0.1,
            exit_threshold: 0.9,
            prelude_layers: 2,
            coda_layers: 2,
        }
    }
}

impl SpiderConfig {
    /// ~400M-class target (still needs real weight init sizing).
    pub fn personal_scale() -> Self {
        Self {
            hidden_size: 1024,
            num_heads: 16,
            num_experts: 16,
            top_k: 2,
            num_loops: 6,
            edges_per_expert: 3,
            ..Self::default()
        }
    }

    pub fn ffn_hidden(&self) -> usize {
        self.hidden_size * self.ffn_mult
    }
}
