//! CSR sparse adjacency over experts / tokens (scaffold).

use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct CsrAdjacency {
    pub row_ptr: Vec<usize>,
    pub col_idx: Vec<usize>,
    pub weights: Vec<f32>,
    pub num_nodes: usize,
}

impl CsrAdjacency {
    pub fn empty(num_nodes: usize) -> Self {
        Self {
            row_ptr: vec![0; num_nodes + 1],
            col_idx: Vec::new(),
            weights: Vec::new(),
            num_nodes,
        }
    }

    /// Message pass scaffold: weighted sum of neighbor rows of `h`.
    /// Here nodes ≈ batch*seq rows for the prototype.
    pub fn message_pass(&self, h: &Array2<f32>) -> Array2<f32> {
        let mut out = h.clone();
        if self.col_idx.is_empty() {
            return out;
        }
        // Minimal prototype: leave h (full graph kernels later)
        let _ = self.num_nodes;
        out
    }
}
