//! Training-time stability monitors (variance, norms, expert load).

use ndarray::Array2;

#[derive(Debug, Default)]
pub struct StabilityMonitor {
    pub alerts: Vec<String>,
}

impl StabilityMonitor {
    pub fn check_hidden(&mut self, h0: &Array2<f32>, ht: &Array2<f32>, loop_idx: usize) {
        let v0 = row_var_mean(h0);
        let vt = row_var_mean(ht);
        if v0 > 1e-8 && vt > 10.0 * v0 {
            self.alerts.push(format!(
                "loop {loop_idx}: variance blow-up ({vt:.4} vs {v0:.4})"
            ));
        }
        let n0 = mean_l2(h0);
        let nt = mean_l2(ht);
        if n0 > 1e-8 && nt > 5.0 * n0 {
            self.alerts.push(format!(
                "loop {loop_idx}: norm drift ({nt:.4} vs {n0:.4})"
            ));
        }
    }

    pub fn clear(&mut self) {
        self.alerts.clear();
    }
}

fn row_var_mean(x: &Array2<f32>) -> f32 {
    if x.nrows() == 0 {
        return 0.0;
    }
    let mut acc = 0f32;
    for row in x.rows() {
        let mean = row.mean().unwrap_or(0.0);
        let mut v = 0f32;
        for &v_i in row.iter() {
            let d = v_i - mean;
            v += d * d;
        }
        acc += v / row.len().max(1) as f32;
    }
    acc / x.nrows() as f32
}

fn mean_l2(x: &Array2<f32>) -> f32 {
    if x.nrows() == 0 {
        return 0.0;
    }
    let mut acc = 0f32;
    for row in x.rows() {
        let s: f32 = row.iter().map(|v| v * v).sum();
        acc += s.sqrt();
    }
    acc / x.nrows() as f32
}
