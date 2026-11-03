//! Unshared entry Transformer layers (not looped).

use crate::config::SpiderConfig;
use ndarray::Array2;

#[derive(Debug)]
pub struct Prelude {
    pub num_layers: usize,
    pub hidden: usize,
}

impl Prelude {
    pub fn new(cfg: &SpiderConfig) -> Self {
        Self {
            num_layers: cfg.prelude_layers,
            hidden: cfg.hidden_size,
        }
    }

    /// Placeholder residual stack: identity for scaffold.
    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        // Real: Attn + FFN per layer with unique params.
        let mut h = x.clone();
        for _ in 0..self.num_layers {
            // h = h + attn(ln(h)); h = h + ffn(ln(h))
            let _ = self.hidden;
            h = h; // identity until weights land
        }
        h
    }
}
