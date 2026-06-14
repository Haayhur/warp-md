#[derive(Debug, Clone)]
pub(super) struct GaussianProcess {
    x: Vec<Vec<f64>>,
    y_mean: f64,
    y_std: f64,
    alpha: Vec<f64>,
    chol: Vec<Vec<f64>>,
    lengthscales: Vec<f64>,
}

impl GaussianProcess {
    pub(super) fn fit(x: &[Vec<f64>], y_raw: &[f64], noise_variance: f64) -> Option<Self> {
        if x.len() < 2 || x.len() != y_raw.len() {
            return None;
        }
        let y_mean = y_raw.iter().sum::<f64>() / y_raw.len() as f64;
        let y_var = y_raw
            .iter()
            .map(|value| (value - y_mean).powi(2))
            .sum::<f64>()
            / (y_raw.len().saturating_sub(1).max(1)) as f64;
        let y_std = y_var.sqrt().max(1.0e-12);
        let y = y_raw
            .iter()
            .map(|value| (value - y_mean) / y_std)
            .collect::<Vec<_>>();
        let lengthscales = ard_lengthscales(x);
        let mut kernel = vec![vec![0.0; x.len()]; x.len()];
        for row in 0..x.len() {
            for col in 0..=row {
                let value = matern52(&x[row], &x[col], &lengthscales);
                kernel[row][col] = value;
                kernel[col][row] = value;
            }
            kernel[row][row] += noise_variance.max(1.0e-12);
        }
        let chol = cholesky(kernel)?;
        let alpha = solve_cholesky(&chol, &y);
        Some(Self {
            x: x.to_vec(),
            y_mean,
            y_std,
            alpha,
            chol,
            lengthscales,
        })
    }

    pub(super) fn predict(&self, x_star: &[f64]) -> (f64, f64) {
        let k_star = self
            .x
            .iter()
            .map(|x| matern52(x, x_star, &self.lengthscales))
            .collect::<Vec<_>>();
        let mean_norm = dot(&k_star, &self.alpha);
        let v = solve_lower(&self.chol, &k_star);
        let variance_norm = (1.0 - dot(&v, &v)).max(1.0e-12);
        (
            self.y_mean + mean_norm * self.y_std,
            variance_norm.sqrt() * self.y_std,
        )
    }
}

fn ard_lengthscales(x: &[Vec<f64>]) -> Vec<f64> {
    let dims = x.first().map_or(0, Vec::len);
    (0..dims)
        .map(|dim| {
            let mean = x.iter().map(|row| row[dim]).sum::<f64>() / x.len() as f64;
            let var = x.iter().map(|row| (row[dim] - mean).powi(2)).sum::<f64>() / x.len() as f64;
            var.sqrt().clamp(0.05, 1.0)
        })
        .collect()
}

fn matern52(a: &[f64], b: &[f64], lengthscales: &[f64]) -> f64 {
    let mut r_sq = 0.0;
    for ((a, b), lengthscale) in a.iter().zip(b.iter()).zip(lengthscales.iter()) {
        let diff = (a - b) / lengthscale.max(1.0e-12);
        r_sq += diff * diff;
    }
    let r = r_sq.sqrt();
    let sqrt5_r = 5.0_f64.sqrt() * r;
    (1.0 + sqrt5_r + 5.0 * r_sq / 3.0) * (-sqrt5_r).exp()
}

fn cholesky(mut matrix: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    for row in 0..n {
        for col in 0..=row {
            let mut sum = matrix[row][col];
            for k in 0..col {
                sum -= matrix[row][k] * matrix[col][k];
            }
            if row == col {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                matrix[row][col] = sum.sqrt();
            } else {
                matrix[row][col] = sum / matrix[col][col];
            }
        }
        for col in row + 1..n {
            matrix[row][col] = 0.0;
        }
    }
    Some(matrix)
}

fn solve_cholesky(chol: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let y = solve_lower(chol, b);
    solve_upper(chol, &y)
}

fn solve_lower(chol: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = chol.len();
    let mut y = vec![0.0; n];
    for row in 0..n {
        let mut sum = b[row];
        for (col, value) in y.iter().enumerate().take(row) {
            sum -= chol[row][col] * value;
        }
        y[row] = sum / chol[row][row];
    }
    y
}

fn solve_upper(chol: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let n = chol.len();
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = y[row];
        for col in row + 1..n {
            sum -= chol[col][row] * x[col];
        }
        x[row] = sum / chol[row][row];
    }
    x
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
}
