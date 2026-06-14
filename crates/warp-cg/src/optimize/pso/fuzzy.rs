use super::settings::ParticleControls;

pub(super) fn fuzzy_controls(
    worst_value: f64,
    global_value: f64,
    previous_value: f64,
    new_value: f64,
    movement: f64,
    distance_from_best: f64,
    max_distance: f64,
    fallback: ParticleControls,
) -> ParticleControls {
    if movement <= 0.0 || max_distance <= 0.0 {
        return fallback;
    }
    let denominator = (worst_value - global_value).abs().max(1.0e-12);
    let phi = (movement / max_distance)
        * ((worst_value.min(new_value) - worst_value.min(previous_value)) / denominator);
    let delta = (distance_from_best / max_distance).clamp(0.0, 1.0);
    let memberships = fuzzy_memberships(phi.clamp(-1.0, 1.0), delta);

    ParticleControls {
        inertia: weighted_average(&[
            (memberships.phi_worse, 0.3),
            (memberships.phi_same, 0.5),
            (memberships.phi_better, 1.0),
            (memberships.delta_same, 0.3),
            (memberships.delta_near, 0.5),
            (memberships.delta_far, 0.3),
        ])
        .unwrap_or(fallback.inertia),
        social: weighted_average(&[
            (memberships.phi_worse, 3.0),
            (memberships.phi_same, 2.0),
            (memberships.phi_better, 1.0),
            (memberships.delta_same, 2.0),
            (memberships.delta_near, 1.0),
            (memberships.delta_far, 2.0),
        ])
        .unwrap_or(fallback.social),
        cognitive: weighted_average(&[
            (memberships.phi_worse, 1.5),
            (memberships.phi_same, 1.5),
            (memberships.phi_better, 3.0),
            (memberships.delta_same, 1.5),
            (memberships.delta_near, 1.5),
            (memberships.delta_far, 1.5),
        ])
        .unwrap_or(fallback.cognitive),
        min_velocity_fraction: weighted_average(&[
            (memberships.phi_worse, 0.01),
            (memberships.phi_same, 0.0),
            (memberships.phi_better, 0.0),
            (memberships.delta_same, 0.001),
            (memberships.delta_near, 0.001),
            (memberships.delta_far, 0.001),
        ])
        .unwrap_or(fallback.min_velocity_fraction),
        max_velocity_fraction: weighted_average(&[
            (memberships.phi_worse, 0.2),
            (memberships.phi_same, 0.15),
            (memberships.phi_better, 0.15),
            (memberships.delta_same, 0.1),
            (memberships.delta_near, 0.15),
            (memberships.delta_far, 0.1),
        ])
        .unwrap_or(fallback.max_velocity_fraction),
    }
}

#[derive(Debug, Clone, Copy)]
struct FuzzyMemberships {
    phi_worse: f64,
    phi_same: f64,
    phi_better: f64,
    delta_same: f64,
    delta_near: f64,
    delta_far: f64,
}

fn fuzzy_memberships(phi: f64, delta: f64) -> FuzzyMemberships {
    FuzzyMemberships {
        phi_worse: rising(phi, 0.0, 1.0),
        phi_same: triangular(phi, -1.0, 0.0, 1.0),
        phi_better: falling(phi, -1.0, 0.0),
        delta_same: falling(delta, 0.2, 0.4),
        delta_near: triangular(delta, 0.2, 0.4, 0.6),
        delta_far: rising(delta, 0.4, 0.6),
    }
}

fn weighted_average(values: &[(f64, f64)]) -> Option<f64> {
    let weight_sum = values.iter().map(|(weight, _)| weight).sum::<f64>();
    if weight_sum <= 1.0e-12 {
        return None;
    }
    Some(
        values
            .iter()
            .map(|(weight, value)| weight * value)
            .sum::<f64>()
            / weight_sum,
    )
}

fn triangular(value: f64, left: f64, center: f64, right: f64) -> f64 {
    if value <= left || value >= right {
        0.0
    } else if value <= center {
        rising(value, left, center)
    } else {
        falling(value, center, right)
    }
}

fn rising(value: f64, left: f64, right: f64) -> f64 {
    if value <= left {
        0.0
    } else if value >= right {
        1.0
    } else {
        (value - left) / (right - left)
    }
}

fn falling(value: f64, left: f64, right: f64) -> f64 {
    if value <= left {
        1.0
    } else if value >= right {
        0.0
    } else {
        (right - value) / (right - left)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::pso::movement::default_swarm_size;

    #[test]
    fn fuzzy_memberships_classify_better_same_and_worse_phi() {
        let better = fuzzy_memberships(-0.75, 0.1);
        let same = fuzzy_memberships(0.0, 0.4);
        let worse = fuzzy_memberships(0.75, 0.7);

        assert!(better.phi_better > better.phi_worse);
        assert!(same.phi_same > same.phi_better);
        assert!(same.phi_same > same.phi_worse);
        assert!(worse.phi_worse > worse.phi_better);
    }

    #[test]
    fn fuzzy_controls_increase_exploration_after_improvement() {
        let fallback = ParticleControls {
            inertia: 0.65,
            cognitive: 1.4,
            social: 1.4,
            max_velocity_fraction: 1.0,
            min_velocity_fraction: 0.0,
        };
        let controls = fuzzy_controls(10.0, 1.0, 9.0, 0.0, 1.0, 0.5, 1.0, fallback);

        assert!(controls.inertia > 0.5);
        assert!(controls.cognitive > controls.social);
        assert!(controls.max_velocity_fraction <= 0.2);
    }

    #[test]
    fn default_swarm_size_matches_fst_pso_floor_heuristic() {
        assert_eq!(default_swarm_size(1), 12);
        assert_eq!(default_swarm_size(2), 12);
        assert_eq!(default_swarm_size(9), 16);
    }
}
