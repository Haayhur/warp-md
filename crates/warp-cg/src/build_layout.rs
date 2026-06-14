use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[path = "build_layout_ops.rs"]
mod build_layout_ops;

use build_layout_ops::{
    apply_edge_push, apply_exclusion_push, deterministic_unit_vector, layout_delta,
    max_displacement_from_reference, max_edge_violation, max_exclusion_violation,
    neighbor_pairs_with_periodicity_basis, optimizer_neighbor_cutoff, optimizer_trace_sample,
    placement_metrics, wrap_point_into_periodic_bounds,
};

#[cfg(test)]
use build_layout_ops::min_pair_clearance;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayoutPoint {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LayoutBounds {
    pub xmin: f32,
    pub xmax: f32,
    pub ymin: f32,
    pub ymax: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CircularExclusion {
    pub x: f32,
    pub y: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RelaxationConfig {
    pub enabled: bool,
    pub max_steps: usize,
    pub push_tolerance: f32,
    pub lipid_push_multiplier: f32,
    pub edge_push_multiplier: f32,
    pub occupation_modifier: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct LayoutPeriodicity {
    pub x: bool,
    pub y: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct LayoutBasis2D {
    pub origin: [f32; 2],
    pub a: [f32; 2],
    pub b: [f32; 2],
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct PlacementMetrics {
    pub relaxation_enabled: bool,
    pub relaxation_steps: usize,
    pub max_push_angstrom: f32,
    pub trajectory_frame_count: usize,
    pub optimizer_neighbor_cutoff_angstrom: Option<f32>,
    pub neighbor_search_rebuild_count: usize,
    pub max_total_displacement_angstrom: f32,
    pub mean_total_displacement_angstrom: f32,
    pub optimizer_trace: Vec<OptimizerTraceSample>,
    pub min_pair_clearance_angstrom: Option<f32>,
    pub max_edge_violation_angstrom: f32,
    pub max_exclusion_violation_angstrom: f32,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct OptimizerTraceSample {
    pub step: usize,
    pub step_multiplier: f32,
    pub max_push_angstrom: f32,
    pub max_total_displacement_angstrom: f32,
    pub mean_total_displacement_angstrom: f32,
    pub min_pair_clearance_angstrom: Option<f32>,
    pub max_edge_violation_angstrom: f32,
    pub max_exclusion_violation_angstrom: f32,
}

pub(crate) fn rectangular_leaflet_grid(
    radii: &[f32],
    apl_angstrom2: f32,
    bounds: LayoutBounds,
) -> Vec<LayoutPoint> {
    let count = radii.len();
    if count == 0 {
        return Vec::new();
    }
    let spacing = apl_angstrom2.sqrt();
    let x_len = bounds.xmax - bounds.xmin;
    let y_len = bounds.ymax - bounds.ymin;
    let aspect = if y_len > 0.0 { x_len / y_len } else { 1.0 };
    let cols = ((count as f32 * aspect.max(0.1)).sqrt().ceil() as usize).max(1);
    let rows = count.div_ceil(cols).max(1);
    let grid_width = (cols.saturating_sub(1)) as f32 * spacing;
    let grid_height = (rows.saturating_sub(1)) as f32 * spacing;
    let center_x = (bounds.xmin + bounds.xmax) * 0.5;
    let center_y = (bounds.ymin + bounds.ymax) * 0.5;

    let mut points = Vec::with_capacity(count);
    for row in 0..rows {
        for col in 0..cols {
            if points.len() == count {
                return points;
            }
            points.push(LayoutPoint {
                x: center_x + grid_width * 0.5 - col as f32 * spacing,
                y: center_y + row as f32 * spacing - grid_height * 0.5,
                radius: radii[points.len()],
            });
        }
    }
    points
}

pub(crate) fn relax_leaflet_points_periodic(
    points: &mut [LayoutPoint],
    bounds: LayoutBounds,
    exclusions: &[CircularExclusion],
    config: RelaxationConfig,
    periodicity: LayoutPeriodicity,
) -> PlacementMetrics {
    relax_leaflet_points_with_projector_periodic(
        points,
        bounds,
        exclusions,
        config,
        periodicity,
        |_| None,
    )
}

pub(crate) fn relax_leaflet_points_with_projector_periodic<F>(
    points: &mut [LayoutPoint],
    bounds: LayoutBounds,
    exclusions: &[CircularExclusion],
    config: RelaxationConfig,
    periodicity: LayoutPeriodicity,
    project_point: F,
) -> PlacementMetrics
where
    F: FnMut(LayoutPoint) -> Option<LayoutPoint>,
{
    relax_leaflet_points_with_projector_basis(
        points,
        bounds,
        exclusions,
        config,
        periodicity,
        None,
        project_point,
    )
}

pub(crate) fn relax_leaflet_points_with_projector_basis<F>(
    points: &mut [LayoutPoint],
    bounds: LayoutBounds,
    exclusions: &[CircularExclusion],
    config: RelaxationConfig,
    periodicity: LayoutPeriodicity,
    basis: Option<LayoutBasis2D>,
    mut project_point: F,
) -> PlacementMetrics
where
    F: FnMut(LayoutPoint) -> Option<LayoutPoint>,
{
    let initial_points = points
        .iter()
        .map(|point| (point.x, point.y))
        .collect::<Vec<_>>();
    if points.is_empty() || !config.enabled || config.max_steps == 0 {
        let trace = vec![optimizer_trace_sample(
            points,
            bounds,
            exclusions,
            periodicity,
            basis,
            0,
            0.0,
            0.0,
            &initial_points,
        )];
        return placement_metrics(
            points,
            bounds,
            exclusions,
            periodicity,
            basis,
            false,
            0,
            0.0,
            None,
            0,
            &initial_points,
            trace,
        );
    }

    let neighbor_cutoff = optimizer_neighbor_cutoff(points);
    let neighbor_rebuild_threshold = neighbor_cutoff.unwrap_or(1.0) * 0.5;
    let mut neighbor_pairs_cache =
        neighbor_pairs_with_periodicity_basis(points, bounds, periodicity, basis);
    let mut neighbor_rebuild_reference = initial_points.clone();
    let mut neighbor_search_rebuild_count = 0usize;
    let mut bounce_counters = vec![0.0f32; points.len()];
    let mut optimizer_trace = vec![optimizer_trace_sample(
        points,
        bounds,
        exclusions,
        periodicity,
        basis,
        0,
        0.0,
        0.0,
        &initial_points,
    )];
    let mut max_push = 0.0f32;
    let mut steps = 0usize;
    for step in 1..=config.max_steps {
        steps = step;
        let step_multiplier = if step >= 15 {
            1.0 / 15.0
        } else {
            1.0 - ((step - 1) as f32 / 15.0)
        };
        let mut pushes = vec![[0.0f32, 0.0f32, 0.0f32]; points.len()];
        for &(i, j) in &neighbor_pairs_cache {
            let [dx, dy] = layout_delta(points[j], points[i], bounds, periodicity, basis);
            let dist = (dx * dx + dy * dy).sqrt();
            let combined = points[i].radius + points[j].radius;
            let ideal_upper = combined * (1.0 + 2.0 * config.occupation_modifier);
            if dist > ideal_upper {
                continue;
            }
            let (ux, uy) = if dist > f32::EPSILON {
                (dx / dist, dy / dist)
            } else {
                deterministic_unit_vector(i, j)
            };
            let mut push = (ideal_upper - dist) * step_multiplier;
            if dist < combined {
                push += (combined - dist) * 0.5;
            }
            let push = push * config.lipid_push_multiplier;
            let max_radius = points[i].radius.max(points[j].radius);
            let i_mult = points[j].radius / max_radius;
            let j_mult = points[i].radius / max_radius;
            pushes[i][0] -= ux * push * i_mult;
            pushes[i][1] -= uy * push * i_mult;
            pushes[i][2] += 1.0;
            pushes[j][0] += ux * push * j_mult;
            pushes[j][1] += uy * push * j_mult;
            pushes[j][2] += 1.0;
        }

        max_push = 0.0;
        for ((point, push), bounce_counter) in points
            .iter_mut()
            .zip(pushes.iter())
            .zip(bounce_counters.iter_mut())
        {
            if push[2] > 0.0 {
                let n = push[2] as f32;
                let px = push[0] / n;
                let py = push[1] / n;
                point.x += px;
                point.y += py;
                max_push = max_push.max((px * px + py * py).sqrt());
            }
            if *bounce_counter > 0.0 {
                *bounce_counter = (*bounce_counter - 0.1).max(0.0);
            }
            max_push = max_push.max(apply_edge_push(
                point,
                bounds,
                periodicity,
                config.edge_push_multiplier,
                bounce_counter,
            ));
            max_push = max_push.max(apply_exclusion_push(
                point,
                exclusions,
                bounds,
                periodicity,
                basis,
                config.edge_push_multiplier,
                bounce_counter,
            ));
            if let Some(projected) = project_point(*point) {
                let dx = projected.x - point.x;
                let dy = projected.y - point.y;
                point.x = projected.x;
                point.y = projected.y;
                max_push = max_push.max((dx * dx + dy * dy).sqrt());
                *bounce_counter += 1.0;
            }
            wrap_point_into_periodic_bounds(point, bounds, periodicity, basis);
        }
        let max_displacement_since_rebuild = max_displacement_from_reference(
            points,
            &neighbor_rebuild_reference,
            bounds,
            periodicity,
            basis,
        );
        if max_displacement_since_rebuild >= neighbor_rebuild_threshold {
            neighbor_pairs_cache =
                neighbor_pairs_with_periodicity_basis(points, bounds, periodicity, basis);
            neighbor_rebuild_reference = points.iter().map(|point| (point.x, point.y)).collect();
            neighbor_search_rebuild_count += 1;
        }
        optimizer_trace.push(optimizer_trace_sample(
            points,
            bounds,
            exclusions,
            periodicity,
            basis,
            step,
            step_multiplier,
            max_push,
            &initial_points,
        ));

        if max_push < config.push_tolerance
            && max_edge_violation(points, bounds, periodicity, basis) <= 0.0
            && max_exclusion_violation(points, exclusions, bounds, periodicity, basis) <= 0.0
        {
            break;
        }
    }
    placement_metrics(
        points,
        bounds,
        exclusions,
        periodicity,
        basis,
        true,
        steps,
        max_push,
        neighbor_cutoff,
        neighbor_search_rebuild_count,
        &initial_points,
        optimizer_trace,
    )
}

#[cfg(test)]
#[path = "build_layout_tests.rs"]
mod tests;
