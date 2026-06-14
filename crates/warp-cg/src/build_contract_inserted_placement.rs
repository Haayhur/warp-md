use super::build_contract_math3d::squared_distance3;
use super::*;

#[derive(Clone, Debug)]
pub(super) struct InsertedCenterPlan {
    pub(super) centers: Vec<[f32; 3]>,
    pub(super) target_count: usize,
    pub(super) initial_candidate_count: usize,
    pub(super) candidate_count: usize,
    pub(super) grid_squeeze_pass_count: usize,
    pub(super) squeezed_candidate_count: usize,
    pub(super) min_spacing_angstrom: Option<f32>,
    pub(super) kick_attempt_count: usize,
    pub(super) kicked_inserted_count: usize,
}

pub(super) fn accumulate_inserted_flood_summary(
    summary: &mut InsertedFloodPlacementSummary,
    plan: &InsertedCenterPlan,
) {
    summary.density.target_count += plan.target_count;
    summary.density.placed_count += plan.centers.len();
    summary.density.initial_candidate_count += plan.initial_candidate_count;
    summary.density.final_candidate_count += plan.candidate_count;
    summary.candidate_count += plan.candidate_count;
    summary.grid_squeeze_pass_count += plan.grid_squeeze_pass_count;
    summary.squeezed_candidate_count += plan.squeezed_candidate_count;
    summary.min_spacing_angstrom = match (summary.min_spacing_angstrom, plan.min_spacing_angstrom) {
        (Some(current), Some(next)) => Some(current.min(next)),
        (None, Some(next)) => Some(next),
        (current, None) => current,
    };
    summary.kick_attempt_count += plan.kick_attempt_count;
    summary.kicked_inserted_count += plan.kicked_inserted_count;
    finalize_phase_density(&mut summary.density, summary.grid_squeeze_pass_count);
}

pub(super) fn inserted_copy_centers(
    component: &InsertedComponent,
    system: &BuildSystem,
    occupied: &[EmittedBead],
    offsets: &[[f32; 3]],
    excluded_volume_factor: f32,
) -> Result<InsertedCenterPlan> {
    if component.count <= 1 {
        return Ok(InsertedCenterPlan {
            centers: vec![component.placement.center_angstrom],
            target_count: component.count,
            initial_candidate_count: 0,
            candidate_count: 0,
            grid_squeeze_pass_count: 0,
            squeezed_candidate_count: 0,
            min_spacing_angstrom: None,
            kick_attempt_count: 0,
            kicked_inserted_count: 0,
        });
    }
    if offsets.is_empty() {
        return Ok(InsertedCenterPlan {
            centers: vec![component.placement.center_angstrom; component.count],
            target_count: component.count,
            initial_candidate_count: 0,
            candidate_count: 0,
            grid_squeeze_pass_count: 0,
            squeezed_candidate_count: 0,
            min_spacing_angstrom: None,
            kick_attempt_count: 0,
            kicked_inserted_count: 0,
        });
    }

    let molecule_radius = offsets
        .iter()
        .map(|offset| (offset[0].powi(2) + offset[1].powi(2) + offset[2].powi(2)).sqrt())
        .fold(0.0f32, f32::max)
        .max(1.0);
    let occupied_radius = default_solvation_bead_radius();
    let base_spacing = (molecule_radius * 2.0 + occupied_radius * 2.0).max(occupied_radius * 2.0);
    let mut spacing = base_spacing;
    let mut candidate_count = 0usize;
    let mut initial_candidate_count = 0usize;
    let mut grid_squeeze_pass_count = 0usize;
    let mut squeezed_candidate_count = 0usize;
    let mut min_spacing_angstrom = None;
    let mut kick_attempt_count = 0usize;
    let mut kicked_inserted_count = 0usize;
    if placement_uses_random_candidates(system) {
        let seed = placement_seed(system).expect("validated seeded random candidate source");
        let random_plan = random_inserted_center_plan(
            component,
            system,
            occupied,
            molecule_radius,
            occupied_radius,
            seed,
        );
        candidate_count += random_plan.candidate_count;
        initial_candidate_count = random_plan.initial_candidate_count;
        min_spacing_angstrom = random_plan.min_spacing_angstrom;
        if random_plan.centers.len() == component.count {
            return Ok(random_plan);
        }
    }

    for retry_idx in 0..10 {
        let mut candidates = inserted_center_candidates(
            system,
            molecule_radius,
            spacing,
            inserted_candidate_phase_offsets(retry_idx),
        );
        if retry_idx == 0 {
            initial_candidate_count = candidates.len();
        }
        candidate_count += candidates.len();
        if retry_idx > 0 {
            grid_squeeze_pass_count += 1;
            squeezed_candidate_count += candidates.len();
        }
        min_spacing_angstrom = Some(match min_spacing_angstrom {
            Some(current) => f32::min(current, spacing),
            None => spacing,
        });
        if let Some(seed) = placement_seed(system) {
            shuffle_points3(
                &mut candidates,
                mix_seed(
                    seed,
                    "inserted_component",
                    component.name.as_bytes(),
                    component.count,
                ),
            );
        } else {
            candidates.sort_by(|left, right| {
                let dl = squared_distance3(*left, component.placement.center_angstrom);
                let dr = squared_distance3(*right, component.placement.center_angstrom);
                dl.partial_cmp(&dr)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        left[2]
                            .partial_cmp(&right[2])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| {
                        left[1]
                            .partial_cmp(&right[1])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .then_with(|| {
                        left[0]
                            .partial_cmp(&right[0])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
            });
        }

        let mut centers = Vec::with_capacity(component.count);
        for candidate in &candidates {
            if !inserted_center_is_clear(
                *candidate,
                molecule_radius,
                occupied,
                occupied_radius,
                &centers,
                system,
            ) {
                continue;
            }
            centers.push(*candidate);
            if centers.len() == component.count {
                return Ok(InsertedCenterPlan {
                    centers,
                    target_count: component.count,
                    initial_candidate_count,
                    candidate_count,
                    grid_squeeze_pass_count,
                    squeezed_candidate_count,
                    min_spacing_angstrom,
                    kick_attempt_count,
                    kicked_inserted_count,
                });
            }
        }
        if let Some(seed) = placement_seed(system) {
            let (attempts, inserted) = extend_seeded_kicked_inserted_centers(
                system,
                component,
                molecule_radius,
                occupied,
                occupied_radius,
                &candidates,
                &mut centers,
                seed,
                retry_idx,
            );
            kick_attempt_count += attempts;
            kicked_inserted_count += inserted;
            if centers.len() == component.count {
                return Ok(InsertedCenterPlan {
                    centers,
                    target_count: component.count,
                    initial_candidate_count,
                    candidate_count,
                    grid_squeeze_pass_count,
                    squeezed_candidate_count,
                    min_spacing_angstrom,
                    kick_attempt_count,
                    kicked_inserted_count,
                });
            }
        }
        spacing *= 0.9;
    }

    if excluded_volume_factor == 0.0 {
        Err(anyhow!(
            "inserted component {} could not flood-place {} non-overlapping copies in the system box",
            component.name,
            component.count
        ))
    } else {
        Err(anyhow!(
            "inserted protein component {} could not place {} non-overlapping copies in the system box",
            component.name,
            component.count
        ))
    }
}

fn random_inserted_center_plan(
    component: &InsertedComponent,
    system: &BuildSystem,
    occupied: &[EmittedBead],
    molecule_radius: f32,
    occupied_radius: f32,
    seed: u64,
) -> InsertedCenterPlan {
    let mut state = mix_seed(
        seed,
        "inserted_component_random_candidates",
        component.name.as_bytes(),
        component.count,
    );
    let target = component.count;
    let max_attempts = (target * 512).max(4096).min(250_000);
    let mut centers = Vec::with_capacity(target);
    let mut attempts = 0usize;
    while centers.len() < target && attempts < max_attempts {
        attempts += 1;
        let candidate = random_inserted_center(system, molecule_radius, &mut state);
        if !inserted_center_inside_box(candidate, system, molecule_radius) {
            continue;
        }
        if !inserted_center_is_clear(
            candidate,
            molecule_radius,
            occupied,
            occupied_radius,
            &centers,
            system,
        ) {
            continue;
        }
        centers.push(candidate);
    }
    InsertedCenterPlan {
        centers,
        target_count: target,
        initial_candidate_count: attempts,
        candidate_count: attempts,
        grid_squeeze_pass_count: 0,
        squeezed_candidate_count: 0,
        min_spacing_angstrom: Some((molecule_radius + occupied_radius) * 2.0),
        kick_attempt_count: 0,
        kicked_inserted_count: 0,
    }
}

fn random_inserted_center(system: &BuildSystem, molecule_radius: f32, state: &mut u64) -> [f32; 3] {
    if let Some(vectors) = distance_box_vectors(system) {
        return random_cell_center([0.0, 0.0, 0.0], vectors, molecule_radius, state);
    }
    let [box_x, box_y, box_z] = placement_box_size_angstrom(system);
    let min_x = -box_x * 0.5 + molecule_radius;
    let max_x = box_x * 0.5 - molecule_radius;
    let min_y = -box_y * 0.5 + molecule_radius;
    let max_y = box_y * 0.5 - molecule_radius;
    let min_z = -box_z * 0.5 + molecule_radius;
    let max_z = box_z * 0.5 - molecule_radius;
    [
        min_x + (max_x - min_x) * seeded_unit_f32(state),
        min_y + (max_y - min_y) * seeded_unit_f32(state),
        min_z + (max_z - min_z) * seeded_unit_f32(state),
    ]
}

pub(super) fn inserted_center_candidates(
    system: &BuildSystem,
    molecule_radius: f32,
    spacing: f32,
    phase_offsets: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    if let Some(vectors) = distance_box_vectors(system) {
        return cell_center_candidates(
            [0.0, 0.0, 0.0],
            vectors,
            molecule_radius,
            spacing,
            phase_offsets,
        );
    }
    let [box_x, box_y, box_z] = placement_box_size_angstrom(system);
    let min_x = -box_x * 0.5 + molecule_radius;
    let max_x = box_x * 0.5 - molecule_radius;
    let min_y = -box_y * 0.5 + molecule_radius;
    let max_y = box_y * 0.5 - molecule_radius;
    let min_z = -box_z * 0.5 + molecule_radius;
    let max_z = box_z * 0.5 - molecule_radius;
    if min_x > max_x || min_y > max_y || min_z > max_z {
        return Vec::new();
    }

    let nx = (((max_x - min_x) / spacing).floor() as usize + 1).max(1);
    let ny = (((max_y - min_y) / spacing).floor() as usize + 1).max(1);
    let nz = (((max_z - min_z) / spacing).floor() as usize + 1).max(1);
    let mut candidates = Vec::with_capacity(nx * ny * nz * phase_offsets.len().max(1));
    for phase in phase_offsets {
        for iz in 0..nz {
            let z = min_z + iz as f32 * spacing + phase[2] * spacing;
            if z > max_z {
                continue;
            }
            for iy in 0..ny {
                let y = min_y + iy as f32 * spacing + phase[1] * spacing;
                if y > max_y {
                    continue;
                }
                for ix in 0..nx {
                    let x = min_x + ix as f32 * spacing + phase[0] * spacing;
                    if x > max_x {
                        continue;
                    }
                    candidates.push([x, y, z]);
                }
            }
        }
    }
    candidates
}

pub(super) fn inserted_candidate_phase_offsets(retry_idx: usize) -> &'static [[f32; 3]] {
    const BASE: [[f32; 3]; 1] = [[0.0, 0.0, 0.0]];
    const STAGGERED: [[f32; 3]; 9] = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
    ];
    if retry_idx == 0 {
        &BASE
    } else {
        &STAGGERED
    }
}

pub(super) fn extend_seeded_kicked_inserted_centers(
    system: &BuildSystem,
    component: &InsertedComponent,
    molecule_radius: f32,
    occupied: &[EmittedBead],
    occupied_radius: f32,
    candidates: &[[f32; 3]],
    centers: &mut Vec<[f32; 3]>,
    seed: u64,
    retry_idx: usize,
) -> (usize, usize) {
    if centers.len() >= component.count || candidates.is_empty() {
        return (0, 0);
    }
    let mut anchors = centers.clone();
    if anchors.is_empty() {
        anchors.extend_from_slice(candidates);
    }
    let mut state = mix_seed(
        seed,
        "inserted_component_kick",
        component.name.as_bytes(),
        retry_idx,
    );
    let max_attempts = (component.count.saturating_sub(centers.len()) * 64)
        .max(anchors.len() * 8)
        .min(100_000);
    let mut attempts = 0usize;
    let mut inserted = 0usize;
    while centers.len() < component.count && attempts < max_attempts {
        attempts += 1;
        let anchor = anchors[(splitmix64_next(&mut state) as usize) % anchors.len()];
        let kicked = kicked_inserted_center(anchor, molecule_radius, &mut state);
        if !inserted_center_inside_box(kicked, system, molecule_radius) {
            continue;
        }
        if !inserted_center_is_clear(
            kicked,
            molecule_radius,
            occupied,
            occupied_radius,
            centers,
            system,
        ) {
            continue;
        }
        centers.push(kicked);
        anchors.push(kicked);
        inserted += 1;
    }
    (attempts, inserted)
}

fn kicked_inserted_center(anchor: [f32; 3], molecule_radius: f32, state: &mut u64) -> [f32; 3] {
    let radius = molecule_radius * (2.05 + 1.5 * seeded_unit_f32(state));
    let theta = std::f32::consts::TAU * seeded_unit_f32(state);
    let z = 2.0 * seeded_unit_f32(state) - 1.0;
    let r_xy = (1.0 - z * z).max(0.0).sqrt();
    [
        anchor[0] + radius * r_xy * theta.cos(),
        anchor[1] + radius * r_xy * theta.sin(),
        anchor[2] + radius * z,
    ]
}

pub(super) fn inserted_center_inside_box(
    center: [f32; 3],
    system: &BuildSystem,
    molecule_radius: f32,
) -> bool {
    if let Some(vectors) = distance_box_vectors(system) {
        return point_inside_vector_cell(center, [0.0, 0.0, 0.0], vectors, molecule_radius);
    }
    let [box_x, box_y, box_z] = placement_box_size_angstrom(system);
    center[0] >= -box_x * 0.5 + molecule_radius
        && center[0] <= box_x * 0.5 - molecule_radius
        && center[1] >= -box_y * 0.5 + molecule_radius
        && center[1] <= box_y * 0.5 - molecule_radius
        && center[2] >= -box_z * 0.5 + molecule_radius
        && center[2] <= box_z * 0.5 - molecule_radius
}

pub(super) fn inserted_center_is_clear(
    candidate: [f32; 3],
    molecule_radius: f32,
    occupied: &[EmittedBead],
    occupied_radius: f32,
    centers: &[[f32; 3]],
    system: &BuildSystem,
) -> bool {
    let occupied_cutoff_sq = (molecule_radius + occupied_radius).powi(2);
    if occupied.iter().any(|bead| {
        squared_distance3_for_system(candidate, bead.position_angstrom, system) < occupied_cutoff_sq
    }) {
        return false;
    }

    let copy_cutoff_sq = (molecule_radius * 2.0).powi(2);
    !centers
        .iter()
        .any(|center| squared_distance3_for_system(candidate, *center, system) < copy_cutoff_sq)
}
