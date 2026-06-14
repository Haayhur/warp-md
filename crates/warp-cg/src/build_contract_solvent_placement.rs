use super::*;

pub(super) fn solvent_candidates(
    request: &BuildRequest,
    occupied: &[EmittedBead],
    solvent: &SolventPolicy,
    spacing: f32,
    phase_offsets: &[[f32; 3]],
) -> Vec<[f32; 3]> {
    if solvent.box_size_angstrom.is_none() {
        if let Some(vectors) = distance_box_vectors(&request.system) {
            let mut out = Vec::new();
            let candidates = cell_center_candidates(
                solvent.center_angstrom.unwrap_or([0.0, 0.0, 0.0]),
                vectors,
                spacing * 0.5,
                spacing,
                phase_offsets,
            );
            let exclusion =
                solvent.excluded_bead_radius_angstrom + solvent.exclusion_buffer_angstrom;
            let exclusion_sq = exclusion * exclusion;
            let occupied_bins = occupied_position_bins(occupied, exclusion);
            let occupied_points = pbc_occupied_points(occupied, &request.system);
            for candidate in candidates {
                if candidate_overlaps_occupied_for_system(
                    candidate,
                    &occupied_bins,
                    &occupied_points,
                    &[],
                    exclusion,
                    exclusion_sq,
                    &request.system,
                ) {
                    continue;
                }
                if candidate_in_unsolvated_membrane_void(request, candidate) {
                    continue;
                }
                out.push(candidate);
            }
            return out;
        }
    }
    let [box_x, box_y, box_z] = solvent_box_size_angstrom(&request.system, solvent);
    let [center_x, center_y, center_z] = solvent.center_angstrom.unwrap_or([0.0, 0.0, 0.0]);
    let min_x = center_x - box_x * 0.5 + spacing * 0.5;
    let min_y = center_y - box_y * 0.5 + spacing * 0.5;
    let min_z = center_z - box_z * 0.5 + spacing * 0.5;
    let max_x = center_x + box_x * 0.5 - spacing * 0.5;
    let max_y = center_y + box_y * 0.5 - spacing * 0.5;
    let max_z = center_z + box_z * 0.5 - spacing * 0.5;
    let nx = (box_x / spacing).floor().max(0.0) as usize;
    let ny = (box_y / spacing).floor().max(0.0) as usize;
    let nz = (box_z / spacing).floor().max(0.0) as usize;
    let exclusion = solvent.excluded_bead_radius_angstrom + solvent.exclusion_buffer_angstrom;
    let exclusion_sq = exclusion * exclusion;
    let occupied_bins = occupied_position_bins(occupied, exclusion);
    let occupied_points = pbc_occupied_points(occupied, &request.system);
    let mut out = Vec::new();
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
                    if candidate_overlaps_occupied_for_system(
                        [x, y, z],
                        &occupied_bins,
                        &occupied_points,
                        &[],
                        exclusion,
                        exclusion_sq,
                        &request.system,
                    ) {
                        continue;
                    }
                    if candidate_in_unsolvated_membrane_void(request, [x, y, z]) {
                        continue;
                    }
                    out.push([x, y, z]);
                }
            }
        }
    }
    out
}

pub(super) fn solvent_placement_plan(
    request: &BuildRequest,
    occupied: &[EmittedBead],
    solvent: &SolventPolicy,
    target_count: usize,
) -> SolventPlacementPlan {
    let mut spacing = solvent.grid_spacing_angstrom;
    if placement_uses_random_candidates(&request.system) {
        let seed =
            placement_seed(&request.system).expect("validated seeded random candidate source");
        let candidates = random_solvent_candidates(request, occupied, solvent, target_count, seed);
        if candidates.len() >= target_count {
            return SolventPlacementPlan {
                final_candidate_count: candidates.len(),
                grid_point_count: candidates.len(),
                candidates,
                grid_squeeze_pass_count: 0,
                squeezed_candidate_count: 0,
                min_grid_spacing_angstrom: Some(spacing),
                kick_attempt_count: 0,
                kicked_inserted_count: 0,
            };
        }
    }
    let mut candidates = solvent_candidates(
        request,
        occupied,
        solvent,
        spacing,
        solvent_candidate_phase_offsets(0),
    );
    let grid_point_count = candidates.len();
    let mut grid_squeeze_pass_count = 0usize;
    let mut squeezed_candidate_count = 0usize;
    let mut min_grid_spacing_angstrom = Some(spacing);
    let seed = placement_seed(&request.system);
    if let Some(seed) = seed {
        shuffle_points3(
            &mut candidates,
            mix_seed(seed, "solvent", b"candidate_queue", occupied.len()),
        );
    }

    if candidates.len() < target_count {
        for squeeze_idx in 1..=8 {
            spacing *= 0.85;
            min_grid_spacing_angstrom = Some(
                min_grid_spacing_angstrom
                    .unwrap_or(solvent.grid_spacing_angstrom)
                    .min(spacing),
            );
            let added = extend_squeezed_solvent_candidates(
                request,
                occupied,
                solvent,
                &mut candidates,
                spacing,
                solvent_candidate_phase_offsets(squeeze_idx),
                seed,
                squeeze_idx,
            );
            grid_squeeze_pass_count += 1;
            squeezed_candidate_count += added;
            if candidates.len() >= target_count {
                break;
            }
        }
    }

    if candidates.len() >= target_count || candidates.is_empty() {
        return SolventPlacementPlan {
            final_candidate_count: candidates.len(),
            candidates,
            grid_point_count,
            grid_squeeze_pass_count,
            squeezed_candidate_count,
            min_grid_spacing_angstrom,
            kick_attempt_count: 0,
            kicked_inserted_count: 0,
        };
    }

    let Some(seed) = seed else {
        return SolventPlacementPlan {
            final_candidate_count: candidates.len(),
            candidates,
            grid_point_count,
            grid_squeeze_pass_count,
            squeezed_candidate_count,
            min_grid_spacing_angstrom,
            kick_attempt_count: 0,
            kicked_inserted_count: 0,
        };
    };
    let (kick_attempt_count, kicked_inserted_count) = extend_seeded_kicked_solvent_candidates(
        request,
        occupied,
        solvent,
        &mut candidates,
        target_count,
        seed,
    );
    SolventPlacementPlan {
        final_candidate_count: candidates.len(),
        candidates,
        grid_point_count,
        grid_squeeze_pass_count,
        squeezed_candidate_count,
        min_grid_spacing_angstrom,
        kick_attempt_count,
        kicked_inserted_count,
    }
}

fn random_solvent_candidates(
    request: &BuildRequest,
    occupied: &[EmittedBead],
    solvent: &SolventPolicy,
    target_count: usize,
    seed: u64,
) -> Vec<[f32; 3]> {
    if target_count == 0 {
        return Vec::new();
    }
    let exclusion = solvent.excluded_bead_radius_angstrom + solvent.exclusion_buffer_angstrom;
    let exclusion_sq = exclusion * exclusion;
    let mut bins = occupied_position_bins(occupied, exclusion);
    let occupied_points = pbc_occupied_points(occupied, &request.system);
    let mut candidates = Vec::with_capacity(target_count);
    let mut state = mix_seed(
        seed,
        "solvent_random_candidates",
        b"candidate_queue",
        occupied.len(),
    );
    let max_attempts = (target_count * 512).max(4096).min(500_000);
    let mut attempts = 0usize;
    while candidates.len() < target_count && attempts < max_attempts {
        attempts += 1;
        let candidate = random_solvent_candidate(request, solvent, &mut state);
        if !solvent_candidate_allowed(request, solvent, candidate) {
            continue;
        }
        if candidate_overlaps_occupied_for_system(
            candidate,
            &bins,
            &occupied_points,
            &candidates,
            exclusion,
            exclusion_sq,
            &request.system,
        ) {
            continue;
        }
        bins.entry(bin_index3(candidate, exclusion.max(1.0)))
            .or_default()
            .push(candidate);
        candidates.push(candidate);
    }
    candidates
}

fn random_solvent_candidate(
    request: &BuildRequest,
    solvent: &SolventPolicy,
    state: &mut u64,
) -> [f32; 3] {
    if solvent.box_size_angstrom.is_none() {
        if let Some(vectors) = distance_box_vectors(&request.system) {
            return random_cell_center(
                solvent.center_angstrom.unwrap_or([0.0, 0.0, 0.0]),
                vectors,
                0.0,
                state,
            );
        }
    }
    let [box_x, box_y, box_z] = solvent_box_size_angstrom(&request.system, solvent);
    let [center_x, center_y, center_z] = solvent.center_angstrom.unwrap_or([0.0, 0.0, 0.0]);
    [
        center_x + (seeded_unit_f32(state) - 0.5) * box_x,
        center_y + (seeded_unit_f32(state) - 0.5) * box_y,
        center_z + (seeded_unit_f32(state) - 0.5) * box_z,
    ]
}

fn extend_squeezed_solvent_candidates(
    request: &BuildRequest,
    occupied: &[EmittedBead],
    solvent: &SolventPolicy,
    candidates: &mut Vec<[f32; 3]>,
    spacing: f32,
    phase_offsets: &[[f32; 3]],
    seed: Option<u64>,
    squeeze_idx: usize,
) -> usize {
    let exclusion = solvent.excluded_bead_radius_angstrom + solvent.exclusion_buffer_angstrom;
    let exclusion_sq = exclusion * exclusion;
    let mut bins = occupied_position_bins(occupied, exclusion);
    let occupied_points = pbc_occupied_points(occupied, &request.system);
    let mut candidate_points = candidates.clone();
    for candidate in candidates.iter().copied() {
        bins.entry(bin_index3(candidate, exclusion.max(1.0)))
            .or_default()
            .push(candidate);
    }
    let mut extra = solvent_candidates(request, occupied, solvent, spacing, phase_offsets);
    if let Some(seed) = seed {
        shuffle_points3(
            &mut extra,
            mix_seed(seed, "solvent_squeeze", b"candidate_queue", squeeze_idx),
        );
    }
    let mut inserted = 0usize;
    for candidate in extra {
        if candidate_overlaps_occupied_for_system(
            candidate,
            &bins,
            &occupied_points,
            &candidate_points,
            exclusion,
            exclusion_sq,
            &request.system,
        ) {
            continue;
        }
        bins.entry(bin_index3(candidate, exclusion.max(1.0)))
            .or_default()
            .push(candidate);
        candidate_points.push(candidate);
        candidates.push(candidate);
        inserted += 1;
    }
    inserted
}

fn solvent_candidate_phase_offsets(squeeze_idx: usize) -> &'static [[f32; 3]] {
    inserted_candidate_phase_offsets(squeeze_idx)
}

fn extend_seeded_kicked_solvent_candidates(
    request: &BuildRequest,
    occupied: &[EmittedBead],
    solvent: &SolventPolicy,
    candidates: &mut Vec<[f32; 3]>,
    target_count: usize,
    seed: u64,
) -> (usize, usize) {
    let exclusion = solvent.excluded_bead_radius_angstrom + solvent.exclusion_buffer_angstrom;
    let exclusion_sq = exclusion * exclusion;
    let mut bins = occupied_position_bins(occupied, exclusion);
    let occupied_points = pbc_occupied_points(occupied, &request.system);
    let mut candidate_points = candidates.clone();
    for candidate in candidates.iter().copied() {
        bins.entry(bin_index3(candidate, exclusion.max(1.0)))
            .or_default()
            .push(candidate);
    }
    let base = candidates.clone();
    let mut state = mix_seed(seed, "solvent_kick", b"candidate_retry", occupied.len());
    let max_attempts = (target_count.saturating_sub(candidates.len()) * 64)
        .max(base.len() * 8)
        .min(200_000);
    let mut attempts = 0usize;
    let mut inserted = 0usize;
    while candidates.len() < target_count && attempts < max_attempts {
        attempts += 1;
        let anchor = base[(splitmix64_next(&mut state) as usize) % base.len()];
        let kicked = kicked_solvent_candidate(anchor, solvent, &mut state);
        if !solvent_candidate_allowed(request, solvent, kicked) {
            continue;
        }
        if candidate_overlaps_occupied_for_system(
            kicked,
            &bins,
            &occupied_points,
            &candidate_points,
            exclusion,
            exclusion_sq,
            &request.system,
        ) {
            continue;
        }
        bins.entry(bin_index3(kicked, exclusion.max(1.0)))
            .or_default()
            .push(kicked);
        candidate_points.push(kicked);
        candidates.push(kicked);
        inserted += 1;
    }
    (attempts, inserted)
}

fn kicked_solvent_candidate(
    anchor: [f32; 3],
    solvent: &SolventPolicy,
    state: &mut u64,
) -> [f32; 3] {
    let radius = solvent.grid_spacing_angstrom * (0.2 + 0.6 * seeded_unit_f32(state));
    let theta = std::f32::consts::TAU * seeded_unit_f32(state);
    let z = 2.0 * seeded_unit_f32(state) - 1.0;
    let r_xy = (1.0 - z * z).max(0.0).sqrt();
    [
        anchor[0] + radius * r_xy * theta.cos(),
        anchor[1] + radius * r_xy * theta.sin(),
        anchor[2] + radius * z,
    ]
}

pub(super) fn solvent_candidate_allowed(
    request: &BuildRequest,
    solvent: &SolventPolicy,
    candidate: [f32; 3],
) -> bool {
    if solvent.box_size_angstrom.is_none() {
        if let Some(vectors) = distance_box_vectors(&request.system) {
            if !point_inside_vector_cell(
                candidate,
                solvent.center_angstrom.unwrap_or([0.0, 0.0, 0.0]),
                vectors,
                0.0,
            ) {
                return false;
            }
            return !candidate_in_unsolvated_membrane_void(request, candidate);
        }
    }
    let [box_x, box_y, box_z] = solvent_box_size_angstrom(&request.system, solvent);
    let [center_x, center_y, center_z] = solvent.center_angstrom.unwrap_or([0.0, 0.0, 0.0]);
    let half = [box_x * 0.5, box_y * 0.5, box_z * 0.5];
    if candidate[0] < center_x - half[0]
        || candidate[0] > center_x + half[0]
        || candidate[1] < center_y - half[1]
        || candidate[1] > center_y + half[1]
        || candidate[2] < center_z - half[2]
        || candidate[2] > center_z + half[2]
    {
        return false;
    }
    !candidate_in_unsolvated_membrane_void(request, candidate)
}

fn candidate_in_unsolvated_membrane_void(request: &BuildRequest, candidate: [f32; 3]) -> bool {
    request.membranes.iter().any(|membrane| {
        !membrane.solvate_voids
            && (candidate[2] - membrane.center_z_angstrom).abs()
                <= membrane.solvent_exclusion_half_thickness_angstrom
            && membrane_leaflet_void_contains_xy(membrane, [candidate[0], candidate[1]])
    })
}

fn membrane_leaflet_void_contains_xy(membrane: &MembraneRequest, point: [f32; 2]) -> bool {
    membrane.leaflets.iter().any(|leaflet| {
        let hole_contains = leaflet
            .regions
            .iter()
            .any(|region| region.role == "hole" && region_contains_point(region, point));
        if hole_contains {
            return true;
        }
        let patch_regions = leaflet
            .regions
            .iter()
            .filter(|region| region.role == "patch")
            .collect::<Vec<_>>();
        !patch_regions.is_empty()
            && !patch_regions
                .iter()
                .any(|region| region_contains_point(region, point))
    })
}

pub(super) fn occupied_position_bins(
    occupied: &[EmittedBead],
    bin_size_angstrom: f32,
) -> HashMap<(i32, i32, i32), Vec<[f32; 3]>> {
    let bin_size = bin_size_angstrom.max(1.0);
    let mut bins: HashMap<(i32, i32, i32), Vec<[f32; 3]>> = HashMap::new();
    for bead in occupied {
        bins.entry(bin_index3(bead.position_angstrom, bin_size))
            .or_default()
            .push(bead.position_angstrom);
    }
    bins
}

fn candidate_overlaps_occupied(
    candidate: [f32; 3],
    occupied_bins: &HashMap<(i32, i32, i32), Vec<[f32; 3]>>,
    bin_size_angstrom: f32,
    exclusion_sq: f32,
) -> bool {
    let bin_size = bin_size_angstrom.max(1.0);
    let (bx, by, bz) = bin_index3(candidate, bin_size);
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                let Some(points) = occupied_bins.get(&(bx + dx, by + dy, bz + dz)) else {
                    continue;
                };
                if points.iter().any(|point| {
                    let px = point[0] - candidate[0];
                    let py = point[1] - candidate[1];
                    let pz = point[2] - candidate[2];
                    px * px + py * py + pz * pz < exclusion_sq
                }) {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn candidate_overlaps_occupied_for_system(
    candidate: [f32; 3],
    occupied_bins: &HashMap<(i32, i32, i32), Vec<[f32; 3]>>,
    occupied_points: &[[f32; 3]],
    candidate_points: &[[f32; 3]],
    bin_size_angstrom: f32,
    exclusion_sq: f32,
    system: &BuildSystem,
) -> bool {
    if has_periodic_axis(&system.pbc) {
        return occupied_points
            .iter()
            .chain(candidate_points.iter())
            .any(|point| squared_distance3_for_system(candidate, *point, system) < exclusion_sq);
    }
    candidate_overlaps_occupied(candidate, occupied_bins, bin_size_angstrom, exclusion_sq)
}

pub(super) fn pbc_occupied_points(occupied: &[EmittedBead], system: &BuildSystem) -> Vec<[f32; 3]> {
    if has_periodic_axis(&system.pbc) {
        occupied.iter().map(|bead| bead.position_angstrom).collect()
    } else {
        Vec::new()
    }
}

fn bin_index3(position: [f32; 3], bin_size: f32) -> (i32, i32, i32) {
    (
        (position[0] / bin_size).floor() as i32,
        (position[1] / bin_size).floor() as i32,
        (position[2] / bin_size).floor() as i32,
    )
}
