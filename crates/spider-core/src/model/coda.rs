//! Unshared exit Transformer layers (not looped).

use crate::config::SpiderConfig;
use ndarray::Array2;

#[derive(Debug)]
pub struct Coda {
    pub num_layers: usize,
    pub hidden: usize,
}

impl Coda {
    pub fn new(cfg: &SpiderConfig) -> Self {
        Self {
            num_layers: cfg.coda_layers,
            hidden: cfg.hidden_size,
        }
    }

    pub fn forward(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut h = x.clone();
        for _ in 0..self.num_layers {
            let _ = self.hidden;
            h = h;
        }
        h
    }
}
