//! Top-level Spider-Web model: Prelude → Recurrent → Coda → head.

mod coda;
mod early_exit;
mod prelude;
mod recurrent;
mod spider_moe;

pub use coda::Coda;
pub use early_exit::EarlyExit;
pub use prelude::Prelude;
pub use recurrent::RecurrentBlock;
pub use spider_moe::SpiderMoeLayer;

use crate::config::SpiderConfig;
use ndarray::Array2;

/// Full model shell. Tensor backend is ndarray for the scaffold;
/// replace with candle/burn kernels without changing the control flow.
#[derive(Debug)]
pub struct SpiderWebModel {
    pub cfg: SpiderConfig,
    pub prelude: Prelude,
    pub recurrent: RecurrentBlock,
    pub coda: Coda,
    pub early_exit: EarlyExit,
    /// Tied embedding / LM head placeholder: [V, H]
    pub embed: Array2<f32>,
}

impl SpiderWebModel {
    pub fn new(cfg: SpiderConfig) -> Self {
        let h = cfg.hidden_size;
        let v = cfg.vocab_size;
        Self {
            prelude: Prelude::new(&cfg),
            recurrent: RecurrentBlock::new(&cfg),
            coda: Coda::new(&cfg),
            early_exit: EarlyExit::new(h),
            embed: Array2::zeros((v, h)),
            cfg,
        }
    }

    /// Forward one batch of token ids → logits [B, T, V] (scaffold shape notes).
    ///
    /// `training`: if true, always run full `num_loops` (no early exit).
    pub fn forward(&self, token_ids: &Array2<usize>, training: bool) -> ForwardOutput {
        let (b, t) = token_ids.dim();
        let h = self.cfg.hidden_size;

        // Embed: gather rows (scaffold — no real learned table yet beyond zeros)
        let mut x = Array2::<f32>::zeros((b * t, h));
        for bi in 0..b {
            for ti in 0..t {
                let id = token_ids[(bi, ti)].min(self.cfg.vocab_size.saturating_sub(1));
                for d in 0..h {
                    x[(bi * t + ti, d)] = self.embed[(id, d)];
                }
            }
        }
        let x0 = x.clone();

        let h0 = self.prelude.forward(&x);
        let (hr, loops_used) = self.recurrent.forward(&h0, &x0, &self.early_exit, training);
        let z = self.coda.forward(&hr);

        // logits ≈ z @ embed^T  (tied)
        let mut logits = Array2::<f32>::zeros((b * t, self.cfg.vocab_size));
        // scaffold: leave zeros; real matmul when weights exist
        let _ = z;

        ForwardOutput {
            logits,
            loops_used,
            batch: b,
            seq: t,
        }
    }
}

#[derive(Debug)]
pub struct ForwardOutput {
    pub logits: Array2<f32>,
    pub loops_used: usize,
    pub batch: usize,
    pub seq: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn forward_smoke() {
        let m = SpiderWebModel::new(SpiderConfig::default());
        let ids = arr2(&[[1usize, 2, 3], [4, 5, 6]]);
        let out = m.forward(&ids, true);
        assert_eq!(out.batch, 2);
        assert_eq!(out.seq, 3);
        assert_eq!(out.loops_used, m.cfg.num_loops);
    }
}
