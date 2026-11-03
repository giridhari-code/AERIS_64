//! Expert FFN bank (shared across loops).

use crate::config::SpiderConfig;
use ndarray::Array2;

#[derive(Debug)]
pub struct ExpertBank {
    pub num_experts: usize,
    pub hidden: usize,
    pub ffn_hidden: usize,
}

impl ExpertBank {
    pub fn new(cfg: &SpiderConfig) -> Self {
        Self {
            num_experts: cfg.num_experts,
            hidden: cfg.hidden_size,
            ffn_hidden: cfg.ffn_hidden(),
        }
    }

    /// Weighted sum of selected experts. Scaffold: scaled identity.
    pub fn forward(&self, m: &Array2<f32>, idx: &[usize], w: &[f32]) -> Array2<f32> {
        let mut out = Array2::<f32>::zeros(m.raw_dim());
        let n = m.nrows();
        let k = if n == 0 {
            0
        } else {
            idx.len() / n
        };
        if k == 0 {
            return m.clone();
        }
        for i in 0..n {
            for j in 0..k {
                let _e = idx[i * k + j] % self.num_experts;
                let wt = w[i * k + j];
                for d in 0..m.ncols() {
                    out[(i, d)] += wt * m[(i, d)];
                }
            }
        }
        let _ = self.ffn_hidden;
        out
    }
}
