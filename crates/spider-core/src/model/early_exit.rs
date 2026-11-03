//! Inference-only confidence gate after each loop epilogue.

use ndarray::Array2;

#[derive(Debug)]
pub struct EarlyExit {
    /// Linear head w_exit: [H]
    pub w: Vec<f32>,
    pub threshold: f32,
}

impl EarlyExit {
    pub fn new(hidden: usize) -> Self {
        Self {
            w: vec![0.0; hidden],
            threshold: 0.9,
        }
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.threshold = t;
        self
    }

    /// Mean sigmoid confidence over batch rows; exit if above threshold.
    pub fn should_exit(&self, h: &Array2<f32>) -> bool {
        if h.nrows() == 0 || self.w.is_empty() {
            return false;
        }
        let mut sum = 0f32;
        let n = h.nrows();
        for r in 0..n {
            let mut logit = 0f32;
            for (d, &w) in self.w.iter().enumerate() {
                if d < h.ncols() {
                    logit += h[(r, d)] * w;
                }
            }
            sum += 1.0 / (1.0 + (-logit).exp());
        }
        (sum / n as f32) > self.threshold
    }
}
