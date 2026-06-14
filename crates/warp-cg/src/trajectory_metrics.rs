use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

pub(super) const DEFAULT_SASA_BEAD_RADIUS_NM: f64 = 0.235;
pub(super) const DEFAULT_SASA_PROBE_RADIUS_NM: f64 = 0.26;
pub(super) const DEFAULT_SASA_SPHERE_POINTS: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStats {
    pub mean: f64,
    pub std: f64,
    pub samples: usize,
}

#[derive(Debug, Clone, Default)]
pub(super) struct RunningMetricStats {
    values: Vec<f64>,
}

impl RunningMetricStats {
    pub(super) fn push(&mut self, value: f64) {
        if value.is_finite() {
            self.values.push(value);
        }
    }

    pub(super) fn finish(self) -> Option<MetricStats> {
        if self.values.is_empty() {
            return None;
        }
        let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
        let variance = self
            .values
            .iter()
            .map(|value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f64>()
            / self.values.len() as f64;
        Some(MetricStats {
            mean,
            std: variance.sqrt(),
            samples: self.values.len(),
        })
    }
}

pub(super) fn radius_of_gyration(coords: &[[f32; 3]], masses: Option<&[f64]>) -> Option<f64> {
    if coords.is_empty() {
        return None;
    }
    let mut total_mass = 0.0f64;
    let mut center = [0.0f64; 3];
    for (idx, coord) in coords.iter().enumerate() {
        let mass = masses
            .and_then(|values| values.get(idx))
            .copied()
            .filter(|mass| mass.is_finite() && *mass > 0.0)
            .unwrap_or(1.0);
        total_mass += mass;
        center[0] += coord[0] as f64 * mass;
        center[1] += coord[1] as f64 * mass;
        center[2] += coord[2] as f64 * mass;
    }
    if total_mass <= 0.0 {
        return None;
    }
    center[0] /= total_mass;
    center[1] /= total_mass;
    center[2] /= total_mass;

    let sum_sq = coords
        .iter()
        .enumerate()
        .map(|(idx, coord)| {
            let mass = masses
                .and_then(|values| values.get(idx))
                .copied()
                .filter(|mass| mass.is_finite() && *mass > 0.0)
                .unwrap_or(1.0);
            let dx = coord[0] as f64 - center[0];
            let dy = coord[1] as f64 - center[1];
            let dz = coord[2] as f64 - center[2];
            mass * (dx * dx + dy * dy + dz * dz)
        })
        .sum::<f64>();
    Some((sum_sq / total_mass).sqrt())
}

pub(super) fn sasa_approx(
    coords: &[[f32; 3]],
    bead_radius_nm: f64,
    probe_radius_nm: f64,
    n_sphere_points: usize,
) -> Option<f64> {
    if coords.is_empty()
        || !bead_radius_nm.is_finite()
        || !probe_radius_nm.is_finite()
        || bead_radius_nm <= 0.0
        || probe_radius_nm < 0.0
        || n_sphere_points == 0
    {
        return None;
    }

    let expanded_radius = bead_radius_nm + probe_radius_nm;
    let expanded_radius_sq = expanded_radius * expanded_radius;
    let sphere_points = fibonacci_sphere(n_sphere_points);
    let mut total_area = 0.0;

    for (atom_idx, center) in coords.iter().enumerate() {
        let mut exposed = 0usize;
        for point in &sphere_points {
            let sample = [
                center[0] as f64 + expanded_radius * point[0],
                center[1] as f64 + expanded_radius * point[1],
                center[2] as f64 + expanded_radius * point[2],
            ];
            let occluded = coords.iter().enumerate().any(|(other_idx, other)| {
                if other_idx == atom_idx {
                    return false;
                }
                let dx = sample[0] - other[0] as f64;
                let dy = sample[1] - other[1] as f64;
                let dz = sample[2] - other[2] as f64;
                dx * dx + dy * dy + dz * dz < expanded_radius_sq
            });
            if !occluded {
                exposed += 1;
            }
        }
        total_area += 4.0 * PI * expanded_radius_sq * exposed as f64 / n_sphere_points as f64;
    }

    Some(total_area)
}

fn fibonacci_sphere(n_points: usize) -> Vec<[f64; 3]> {
    let golden_angle = PI * (3.0 - 5.0_f64.sqrt());
    (0..n_points)
        .map(|idx| {
            let y = 1.0 - (idx as f64 + 0.5) * 2.0 / n_points as f64;
            let radius = (1.0 - y * y).sqrt();
            let theta = golden_angle * idx as f64;
            [theta.cos() * radius, y, theta.sin() * radius]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sasa_approx_single_bead_matches_expanded_sphere_area() {
        let sasa = sasa_approx(
            &[[0.0, 0.0, 0.0]],
            DEFAULT_SASA_BEAD_RADIUS_NM,
            DEFAULT_SASA_PROBE_RADIUS_NM,
            DEFAULT_SASA_SPHERE_POINTS,
        )
        .unwrap();
        let radius = DEFAULT_SASA_BEAD_RADIUS_NM + DEFAULT_SASA_PROBE_RADIUS_NM;
        let expected = 4.0 * PI * radius * radius;

        assert!((sasa - expected).abs() < 1.0e-12);
    }

    #[test]
    fn sasa_approx_decreases_when_two_beads_overlap() {
        let single = sasa_approx(
            &[[0.0, 0.0, 0.0]],
            DEFAULT_SASA_BEAD_RADIUS_NM,
            DEFAULT_SASA_PROBE_RADIUS_NM,
            DEFAULT_SASA_SPHERE_POINTS,
        )
        .unwrap();
        let pair = sasa_approx(
            &[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
            DEFAULT_SASA_BEAD_RADIUS_NM,
            DEFAULT_SASA_PROBE_RADIUS_NM,
            DEFAULT_SASA_SPHERE_POINTS,
        )
        .unwrap();

        assert!(pair > single);
        assert!(pair < 2.0 * single);
    }
}
