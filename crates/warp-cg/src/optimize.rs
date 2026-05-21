use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::parameters::{AngleStats, BondStats, BondedStats, DihedralStats};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ParameterBound {
    pub name: String,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EvaluationRecord {
    pub iteration: usize,
    pub objective: f64,
    pub parameters: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OptimizationReport {
    pub status: String,
    pub method: String,
    pub objective: String,
    pub objective_value: f64,
    pub converged: bool,
    pub bounds: Vec<ParameterBound>,
    pub best_parameters: Vec<(String, f64)>,
    pub evaluations: Vec<EvaluationRecord>,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub method: String,
    pub objective: String,
    pub max_evaluations: usize,
    pub seed: u64,
    pub swarm_size: Option<usize>,
}

pub fn optimize_bonded_parameters(
    stats: &[BondStats],
    config: &OptimizationConfig,
) -> OptimizationReport {
    optimize_bonded_terms(
        &BondedStats {
            bonds: stats.to_vec(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
        config,
    )
}

pub fn optimize_bonded_terms(
    stats: &BondedStats,
    config: &OptimizationConfig,
) -> OptimizationReport {
    let bounds = default_bounds(stats);
    if bounds.is_empty() {
        return OptimizationReport {
            status: "skipped".to_string(),
            method: config.method.clone(),
            objective: config.objective.clone(),
            objective_value: 0.0,
            converged: true,
            bounds,
            best_parameters: Vec::new(),
            evaluations: Vec::new(),
            message: "No bonded reference statistics were available for parameter tuning."
                .to_string(),
        };
    }

    let target = target_vector(stats);
    let max_evaluations = config.max_evaluations.max(1);
    let (best, best_value, evaluations) = match config.method.as_str() {
        "pso" => run_pso(
            &bounds,
            &target,
            max_evaluations,
            config.seed,
            config.swarm_size,
        ),
        _ => run_bayesian_optimization(&bounds, &target, max_evaluations, config.seed),
    };
    let best_parameters = bounds
        .iter()
        .zip(best.iter())
        .map(|(bound, value)| (bound.name.clone(), *value))
        .collect();

    OptimizationReport {
        status: "ok".to_string(),
        method: config.method.clone(),
        objective: config.objective.clone(),
        objective_value: best_value,
        converged: best_value <= 1.0e-8,
        bounds,
        best_parameters,
        evaluations,
        message: "Optimized bonded parameters against mapped reference bond, angle, and dihedral statistics."
            .to_string(),
    }
}

fn default_bounds(stats: &BondedStats) -> Vec<ParameterBound> {
    let mut bounds =
        Vec::with_capacity((stats.bonds.len() + stats.angles.len() + stats.dihedrals.len()) * 2);
    for stat in &stats.bonds {
        let spread = stat.std.max(0.05);
        bounds.push(ParameterBound {
            name: format!("bond_{}_{}_length_angstrom", stat.bead_i, stat.bead_j),
            min: (stat.mean - 4.0 * spread).max(0.01),
            max: stat.mean + 4.0 * spread,
        });
        bounds.push(ParameterBound {
            name: format!("bond_{}_{}_force", stat.bead_i, stat.bead_j),
            min: 1.0,
            max: 5000.0,
        });
    }
    for stat in &stats.angles {
        let spread = stat.std_deg.max(5.0);
        let min = (stat.mean_deg - 4.0 * spread).clamp(0.0, 180.0);
        let max = (stat.mean_deg + 4.0 * spread).clamp(0.0, 180.0);
        bounds.push(ParameterBound {
            name: angle_value_name(stat),
            min,
            max: max.max(min + 1.0e-6),
        });
        bounds.push(ParameterBound {
            name: angle_force_name(stat),
            min: 1.0,
            max: 500.0,
        });
    }
    for stat in &stats.dihedrals {
        let spread = stat.std_deg.max(10.0);
        let min = (stat.mean_deg - 4.0 * spread).clamp(-180.0, 180.0);
        let max = (stat.mean_deg + 4.0 * spread).clamp(-180.0, 180.0);
        bounds.push(ParameterBound {
            name: dihedral_value_name(stat),
            min,
            max: max.max(min + 1.0e-6),
        });
        bounds.push(ParameterBound {
            name: dihedral_force_name(stat),
            min: 0.1,
            max: 100.0,
        });
    }
    bounds
}

fn target_vector(stats: &BondedStats) -> Vec<f64> {
    let mut target =
        Vec::with_capacity((stats.bonds.len() + stats.angles.len() + stats.dihedrals.len()) * 2);
    for stat in &stats.bonds {
        target.push(stat.mean);
        let variance = stat.std.max(0.02).powi(2);
        target.push((1.0 / variance).clamp(1.0, 5000.0));
    }
    for stat in &stats.angles {
        target.push(stat.mean_deg.clamp(0.0, 180.0));
        let variance = stat.std_deg.max(1.0).powi(2);
        target.push((10_000.0 / variance).clamp(1.0, 500.0));
    }
    for stat in &stats.dihedrals {
        target.push(stat.mean_deg.clamp(-180.0, 180.0));
        let variance = stat.std_deg.max(1.0).powi(2);
        target.push((1_000.0 / variance).clamp(0.1, 100.0));
    }
    target
}

fn angle_value_name(stat: &AngleStats) -> String {
    format!(
        "angle_{}_{}_{}_angle_deg",
        stat.bead_i, stat.bead_j, stat.bead_k
    )
}

fn angle_force_name(stat: &AngleStats) -> String {
    format!(
        "angle_{}_{}_{}_force",
        stat.bead_i, stat.bead_j, stat.bead_k
    )
}

fn dihedral_value_name(stat: &DihedralStats) -> String {
    format!(
        "dihedral_{}_{}_{}_{}_phase_deg",
        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
    )
}

fn dihedral_force_name(stat: &DihedralStats) -> String {
    format!(
        "dihedral_{}_{}_{}_{}_force",
        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
    )
}

fn objective(params: &[f64], target: &[f64], bounds: &[ParameterBound]) -> f64 {
    params
        .iter()
        .zip(target.iter())
        .zip(bounds.iter())
        .map(|((value, target), bound)| {
            let scale = (bound.max - bound.min).max(1.0e-9);
            ((value - target) / scale).powi(2)
        })
        .sum::<f64>()
        / params.len().max(1) as f64
}

fn run_bayesian_optimization(
    bounds: &[ParameterBound],
    target: &[f64],
    max_evaluations: usize,
    seed: u64,
) -> (Vec<f64>, f64, Vec<EvaluationRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let dims = bounds.len();
    let mut evaluations = Vec::with_capacity(max_evaluations);
    let mut best = midpoint(bounds);
    let mut best_value = f64::INFINITY;
    let warmup = max_evaluations.min((dims + 1).clamp(4, 12));

    for iteration in 0..max_evaluations {
        let params = if iteration < warmup {
            latin_hypercube_point(bounds, iteration, warmup, &mut rng)
        } else {
            expected_improvement_candidate(bounds, &evaluations, best_value, &best, &mut rng)
                .unwrap_or_else(|| random_position(bounds, &mut rng))
        };
        let value = objective(&params, target, bounds);
        if value < best_value {
            best_value = value;
            best = params.clone();
        }
        evaluations.push(EvaluationRecord {
            iteration,
            objective: value,
            parameters: params,
        });
    }

    (best, best_value, evaluations)
}

fn latin_hypercube_point(
    bounds: &[ParameterBound],
    iteration: usize,
    total: usize,
    rng: &mut StdRng,
) -> Vec<f64> {
    bounds
        .iter()
        .enumerate()
        .map(|(dim, bound)| {
            let offset: f64 = rng.gen();
            let rank = (iteration + dim * 3) % total;
            let unit = (rank as f64 + offset) / total as f64;
            bound.min + unit * (bound.max - bound.min)
        })
        .collect()
}

fn expected_improvement_candidate(
    bounds: &[ParameterBound],
    evaluations: &[EvaluationRecord],
    best_value: f64,
    best: &[f64],
    rng: &mut StdRng,
) -> Option<Vec<f64>> {
    let model = GaussianProcess::fit(bounds, evaluations)?;
    let dims = bounds.len();
    let candidate_count = (dims * 96).clamp(256, 4096);
    let mut best_candidate = None;
    let mut best_ei = f64::NEG_INFINITY;

    for candidate_idx in 0..candidate_count {
        let candidate = if candidate_idx == 0 {
            midpoint(bounds)
        } else if candidate_idx % 4 == 0 {
            local_candidate(bounds, best, rng)
        } else {
            random_position(bounds, rng)
        };
        let (mean, sigma) = model.predict(&candidate);
        let ei = expected_improvement(best_value, mean, sigma);
        if ei > best_ei {
            best_ei = ei;
            best_candidate = Some(candidate);
        }
    }

    best_candidate
}

fn local_candidate(bounds: &[ParameterBound], center: &[f64], rng: &mut StdRng) -> Vec<f64> {
    bounds
        .iter()
        .zip(center.iter())
        .map(|(bound, center)| {
            let width = 0.15 * (bound.max - bound.min);
            (center + rng.gen_range(-width..=width)).clamp(bound.min, bound.max)
        })
        .collect()
}

fn expected_improvement(best: f64, mean: f64, sigma: f64) -> f64 {
    if sigma <= 1.0e-12 || !sigma.is_finite() {
        return 0.0;
    }
    let improvement = best - mean;
    let z = improvement / sigma;
    improvement * normal_cdf(z) + sigma * normal_pdf(z)
}

struct GaussianProcess {
    bounds: Vec<ParameterBound>,
    x: Vec<Vec<f64>>,
    y_mean: f64,
    y_std: f64,
    alpha: Vec<f64>,
    chol: Vec<Vec<f64>>,
}

impl GaussianProcess {
    fn fit(bounds: &[ParameterBound], evaluations: &[EvaluationRecord]) -> Option<Self> {
        if evaluations.len() < 2 {
            return None;
        }
        let x: Vec<Vec<f64>> = evaluations
            .iter()
            .map(|record| normalize_params(&record.parameters, bounds))
            .collect();
        let y_raw: Vec<f64> = evaluations.iter().map(|record| record.objective).collect();
        let y_mean = y_raw.iter().sum::<f64>() / y_raw.len() as f64;
        let y_var = y_raw
            .iter()
            .map(|value| (value - y_mean).powi(2))
            .sum::<f64>()
            / y_raw.len() as f64;
        let y_std = y_var.sqrt().max(1.0e-12);
        let y: Vec<f64> = y_raw.iter().map(|value| (value - y_mean) / y_std).collect();

        let mut kernel = vec![vec![0.0; x.len()]; x.len()];
        for row in 0..x.len() {
            for col in 0..=row {
                let value = rbf_kernel(&x[row], &x[col]);
                kernel[row][col] = value;
                kernel[col][row] = value;
            }
            kernel[row][row] += 1.0e-8;
        }
        let chol = cholesky(kernel)?;
        let alpha = solve_cholesky(&chol, &y);
        Some(Self {
            bounds: bounds.to_vec(),
            x,
            y_mean,
            y_std,
            alpha,
            chol,
        })
    }

    fn predict(&self, params: &[f64]) -> (f64, f64) {
        let x_star = normalize_params(params, &self.bounds);
        let k_star: Vec<f64> = self.x.iter().map(|x| rbf_kernel(x, &x_star)).collect();
        let mean_norm = dot(&k_star, &self.alpha);
        let v = solve_lower(&self.chol, &k_star);
        let variance_norm = (1.0 - dot(&v, &v)).max(1.0e-12);
        (
            self.y_mean + mean_norm * self.y_std,
            variance_norm.sqrt() * self.y_std,
        )
    }
}

fn normalize_params(params: &[f64], bounds: &[ParameterBound]) -> Vec<f64> {
    params
        .iter()
        .zip(bounds.iter())
        .map(|(value, bound)| {
            let scale = (bound.max - bound.min).max(1.0e-12);
            ((value - bound.min) / scale).clamp(0.0, 1.0)
        })
        .collect()
}

fn rbf_kernel(a: &[f64], b: &[f64]) -> f64 {
    let length_scale = 0.35_f64;
    let dist2 = a
        .iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>();
    (-0.5 * dist2 / length_scale.powi(2)).exp()
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

fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / 2.0_f64.sqrt()))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

fn run_pso(
    bounds: &[ParameterBound],
    target: &[f64],
    max_evaluations: usize,
    seed: u64,
    swarm_size: Option<usize>,
) -> (Vec<f64>, f64, Vec<EvaluationRecord>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let dims = bounds.len();
    let swarm_size = swarm_size.unwrap_or_else(|| (dims * 4).clamp(8, 32)).max(1);
    let iterations = (max_evaluations / swarm_size).max(1);
    let mut particles = Vec::with_capacity(swarm_size);
    let mut global_best = midpoint(bounds);
    let mut global_value = f64::INFINITY;
    let mut evaluations = Vec::new();

    for _ in 0..swarm_size {
        let position = random_position(bounds, &mut rng);
        let value = objective(&position, target, bounds);
        if value < global_value {
            global_value = value;
            global_best = position.clone();
        }
        particles.push((position.clone(), vec![0.0; dims], position, value));
    }

    for iteration in 0..iterations {
        for (position, velocity, personal_best, personal_value) in particles.iter_mut() {
            for dim in 0..dims {
                let rp: f64 = rng.gen();
                let rg: f64 = rng.gen();
                velocity[dim] = 0.65 * velocity[dim]
                    + 1.4 * rp * (personal_best[dim] - position[dim])
                    + 1.4 * rg * (global_best[dim] - position[dim]);
                position[dim] =
                    (position[dim] + velocity[dim]).clamp(bounds[dim].min, bounds[dim].max);
            }
            let value = objective(position, target, bounds);
            if value < *personal_value {
                *personal_value = value;
                *personal_best = position.clone();
            }
            if value < global_value {
                global_value = value;
                global_best = position.clone();
            }
            evaluations.push(EvaluationRecord {
                iteration: evaluations.len().min(max_evaluations.saturating_sub(1)),
                objective: value,
                parameters: position.clone(),
            });
            if evaluations.len() >= max_evaluations {
                return (global_best, global_value, evaluations);
            }
        }
        if iteration + 1 == iterations {
            break;
        }
    }

    (global_best, global_value, evaluations)
}

fn midpoint(bounds: &[ParameterBound]) -> Vec<f64> {
    bounds
        .iter()
        .map(|bound| 0.5 * (bound.min + bound.max))
        .collect()
}

fn random_position(bounds: &[ParameterBound], rng: &mut StdRng) -> Vec<f64> {
    bounds
        .iter()
        .map(|bound| rng.gen_range(bound.min..=bound.max))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stats() -> Vec<BondStats> {
        vec![BondStats {
            bead_i: 0,
            bead_j: 1,
            mean: 3.0,
            std: 0.2,
            samples: 16,
        }]
    }

    fn sample_bonded_stats() -> BondedStats {
        BondedStats {
            bonds: sample_stats(),
            angles: vec![AngleStats {
                bead_i: 0,
                bead_j: 1,
                bead_k: 2,
                mean_deg: 120.0,
                std_deg: 8.0,
                samples: 16,
            }],
            dihedrals: vec![DihedralStats {
                bead_i: 0,
                bead_j: 1,
                bead_k: 2,
                bead_l: 3,
                mean_deg: 180.0,
                std_deg: 20.0,
                samples: 16,
            }],
        }
    }

    #[test]
    fn bayesian_optimization_uses_expected_improvement_trace() {
        let report = optimize_bonded_parameters(
            &sample_stats(),
            &OptimizationConfig {
                method: "bayesian_optimization".to_string(),
                objective: "bonded_parameter_parity".to_string(),
                max_evaluations: 20,
                seed: 7,
                swarm_size: None,
            },
        );

        assert_eq!(report.status, "ok");
        assert_eq!(report.evaluations.len(), 20);
        assert!(report.objective_value.is_finite());
        assert!(
            report.objective_value < report.evaluations[0].objective,
            "BO should improve beyond the first warmup evaluation"
        );
    }

    #[test]
    fn pso_uses_same_bonded_objective() {
        let report = optimize_bonded_parameters(
            &sample_stats(),
            &OptimizationConfig {
                method: "pso".to_string(),
                objective: "bonded_parameter_parity".to_string(),
                max_evaluations: 20,
                seed: 11,
                swarm_size: Some(6),
            },
        );

        assert_eq!(report.status, "ok");
        assert!(!report.evaluations.is_empty());
        assert_eq!(report.best_parameters.len(), report.bounds.len());
    }

    #[test]
    fn bonded_term_optimization_includes_angles_and_dihedrals() {
        let report = optimize_bonded_terms(
            &sample_bonded_stats(),
            &OptimizationConfig {
                method: "bayesian_optimization".to_string(),
                objective: "bonded_parameter_parity".to_string(),
                max_evaluations: 24,
                seed: 17,
                swarm_size: None,
            },
        );
        let names: Vec<&str> = report
            .best_parameters
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();

        assert!(names.contains(&"bond_0_1_length_angstrom"));
        assert!(names.contains(&"angle_0_1_2_angle_deg"));
        assert!(names.contains(&"dihedral_0_1_2_3_phase_deg"));
        assert_eq!(report.best_parameters.len(), 6);
    }
}
