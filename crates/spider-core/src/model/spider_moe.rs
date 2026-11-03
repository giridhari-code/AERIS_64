//! One Spider-MoE iteration: router → edges → dispatch → experts → epilogue.

use crate::config::SpiderConfig;
use crate::graph::adjacency::CsrAdjacency;
use crate::graph::edge_generator::EdgeGenerator;
use crate::moe::experts::ExpertBank;
use crate::moe::router::Router;
use ndarray::Array2;

#[derive(Debug)]
pub struct SpiderMoeLayer {
    pub router: Router,
    pub edge_gen: EdgeGenerator,
    pub experts: ExpertBank,
    pub alpha: f32,
    pub beta: f32,
    pub edges_per_expert: usize,
}

impl SpiderMoeLayer {
    pub fn new(cfg: &SpiderConfig) -> Self {
        Self {
            router: Router::new(cfg),
            edge_gen: EdgeGenerator::new(cfg),
            experts: ExpertBank::new(cfg),
            alpha: cfg.residual_alpha,
            beta: cfg.reinject_beta,
            edges_per_expert: cfg.edges_per_expert,
        }
    }

    pub fn forward(&self, h_prev: &Array2<f32>, x0: &Array2<f32>, loop_idx: usize) -> Array2<f32> {
        // ① Router
        let (_idx, _w) = self.router.topk(h_prev, loop_idx);

        // ② Edge generator → sparse A_t
        let adj: CsrAdjacency = self.edge_gen.generate(h_prev, loop_idx, self.edges_per_expert);

        // ③ Sparse dispatch (scaffold: identity mix)
        let m = adj.message_pass(h_prev);

        // ④ Expert FFN
        let delta = self.experts.forward(&m, &_idx, &_w);

        // ⑤ Epilogue: u = h + α Δ + β x₀
        let mut u = h_prev + &(&delta * self.alpha);
        u = u + &(x0 * self.beta);
        // LayerNorm placeholder: row-wise normalize lightly
        row_l2_normalize(&mut u);
        u
    }
}

fn row_l2_normalize(x: &mut Array2<f32>) {
    let cols = x.ncols();
    for mut row in x.rows_mut() {
        let mut s = 0f32;
        for c in 0..cols {
            s += row[c] * row[c];
        }
        let n = s.sqrt().max(1e-6);
        for c in 0..cols {
            row[c] /= n;
        }
    }
}
