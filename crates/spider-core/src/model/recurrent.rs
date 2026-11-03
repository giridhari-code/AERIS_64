//! Shared recurrent block: run Spider-MoE layer `r` times.

use crate::config::SpiderConfig;
use crate::model::early_exit::EarlyExit;
use crate::model::spider_moe::SpiderMoeLayer;
use ndarray::Array2;

#[derive(Debug)]
pub struct RecurrentBlock {
    pub layer: SpiderMoeLayer,
    pub num_loops: usize,
}

impl RecurrentBlock {
    pub fn new(cfg: &SpiderConfig) -> Self {
        Self {
            layer: SpiderMoeLayer::new(cfg),
            num_loops: cfg.num_loops,
        }
    }

    /// Returns (hidden, loops_executed).
    pub fn forward(
        &self,
        h0: &Array2<f32>,
        x0: &Array2<f32>,
        early_exit: &EarlyExit,
        training: bool,
    ) -> (Array2<f32>, usize) {
        let mut h = h0.clone();
        let mut used = 0usize;
        for t in 1..=self.num_loops {
            h = self.layer.forward(&h, x0, t);
            used = t;
            if !training && early_exit.should_exit(&h) {
                break;
            }
        }
        (h, used)
    }
}
