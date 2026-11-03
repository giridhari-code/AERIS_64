//! Loop-conditioned sparse edge generation.

use crate::config::SpiderConfig;
use crate::graph::adjacency::CsrAdjacency;
use ndarray::Array2;

#[derive(Debug)]
pub struct EdgeGenerator {
    pub hidden: usize,
    pub num_experts: usize,
}

impl EdgeGenerator {
    pub fn new(cfg: &SpiderConfig) -> Self {
        Self {
            hidden: cfg.hidden_size,
            num_experts: cfg.num_experts,
        }
    }

    /// Build A_t from h and loop index. Scaffold returns empty CSR
    /// (identity message path until learned edge weights exist).
    pub fn generate(
        &self,
        h: &Array2<f32>,
        loop_idx: usize,
        edges_per_expert: usize,
    ) -> CsrAdjacency {
        let n = self.num_experts;
        let mut adj = CsrAdjacency::empty(n);
        // Deterministic ring edges as a non-collapse starting point
        let m = edges_per_expert.max(1);
        let mut cols = Vec::new();
        let mut wts = Vec::new();
        let mut ptr = vec![0usize; n + 1];
        for i in 0..n {
            for k in 0..m {
                let j = (i + k + 1 + loop_idx) % n;
                cols.push(j);
                wts.push(1.0 / m as f32);
            }
            ptr[i + 1] = cols.len();
        }
        adj.row_ptr = ptr;
        adj.col_idx = cols;
        adj.weights = wts;
        let _ = h;
        adj
    }
}
