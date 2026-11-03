//! Auxiliary load-balancing loss helpers (training).

/// Soft load-balance penalty target: encourage uniform expert usage.
#[derive(Debug, Default)]
pub struct LoadBalanceStats {
    pub counts: Vec<usize>,
}

impl LoadBalanceStats {
    pub fn new(num_experts: usize) -> Self {
        Self {
            counts: vec![0; num_experts],
        }
    }

    pub fn record(&mut self, expert_idx: usize) {
        if let Some(c) = self.counts.get_mut(expert_idx) {
            *c += 1;
        }
    }

    /// Coefficient of variation of loads (0 = perfect balance).
    pub fn imbalance(&self) -> f32 {
        let n = self.counts.len() as f32;
        if n == 0.0 {
            return 0.0;
        }
        let sum: f32 = self.counts.iter().map(|&c| c as f32).sum();
        let mean = sum / n;
        if mean < 1e-6 {
            return 0.0;
        }
        let var: f32 = self
            .counts
            .iter()
            .map(|&c| {
                let d = c as f32 - mean;
                d * d
            })
            .sum::<f32>()
            / n;
        var.sqrt() / mean
    }
}
