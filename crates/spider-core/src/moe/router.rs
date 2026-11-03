//! Token → expert TopK router with per-loop bias.

use crate::config::SpiderConfig;
use ndarray::Array2;

#[derive(Debug)]
pub struct Router {
    pub num_experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    /// Per-loop bias rows: [max_loops+1][E] stored flat for scaffold.
    pub loop_bias: Vec<f32>,
    pub max_loops: usize,
}

impl Router {
    pub fn new(cfg: &SpiderConfig) -> Self {
        let max_loops = cfg.num_loops;
        Self {
            num_experts: cfg.num_experts,
            top_k: cfg.top_k,
            hidden: cfg.hidden_size,
            loop_bias: vec![0.0; (max_loops + 1) * cfg.num_experts],
            max_loops,
        }
    }

    /// Returns (indices [N, K], weights [N, K]) as flat vectors for scaffold.
    pub fn topk(&self, h: &Array2<f32>, loop_idx: usize) -> (Vec<usize>, Vec<f32>) {
        let n = h.nrows();
        let k = self.top_k.min(self.num_experts).max(1);
        let mut idx = Vec::with_capacity(n * k);
        let mut w = Vec::with_capacity(n * k);
        let t = loop_idx.min(self.max_loops);
        let _bias_off = t * self.num_experts;

        // Scaffold: round-robin experts (replace with h @ W_r + b[t])
        for i in 0..n {
            for j in 0..k {
                idx.push((i + j + t) % self.num_experts);
                w.push(1.0 / k as f32);
            }
        }
        let _ = self.hidden;
        (idx, w)
    }
}
