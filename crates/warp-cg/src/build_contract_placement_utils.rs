use super::*;

pub(super) fn placement_diagnostics(
    system: &BuildSystem,
    beads: &[EmittedBead],
) -> PlacementDiagnostics {
    const TOLERANCE_ANGSTROM: f32 = 1.0e-3;
    const MAX_EXCLUSION_EXAMPLES: usize = 4;
    let mut residue_ids = BTreeSet::new();
    let mut excluded_bead_count = 0usize;
    let mut bounds_min = [f32::INFINITY; 3];
    let mut bounds_max = [f32::NEG_INFINITY; 3];
    for bead in beads {
        residue_ids.insert(bead.residue_id);
        if bead.excluded_volume_factor > 0.0 {
            excluded_bead_count += 1;
        }
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(bead.position_angstrom[axis]);
            bounds_max[axis] = bounds_max[axis].max(bead.position_angstrom[axis]);
        }
    }

    let mut min_inter_residue_distance_angstrom = None;
    let mut min_exclusion_margin_angstrom = None;
    let mut exclusion_violation_count = 0usize;
    let mut exclusion_violation_examples = Vec::new();
    for left_idx in 0..beads.len() {
        for right_idx in (left_idx + 1)..beads.len() {
            let left = &beads[left_idx];
            let right = &beads[right_idx];
            if left.residue_id == right.residue_id {
                continue;
            }
            let distance = squared_distance3_for_system(
                left.position_angstrom,
                right.position_angstrom,
                system,
            )
            .sqrt();
            min_inter_residue_distance_angstrom = Some(match min_inter_residue_distance_angstrom {
                Some(current) => f32::min(current, distance),
                None => distance,
            });
            let exclusion_distance =
                diagnostic_exclusion_radius(left) + diagnostic_exclusion_radius(right);
            if exclusion_distance <= 0.0 {
                continue;
            }
            let margin = distance - exclusion_distance;
            min_exclusion_margin_angstrom = Some(match min_exclusion_margin_angstrom {
                Some(current) => f32::min(current, margin),
                None => margin,
            });
            if margin < -TOLERANCE_ANGSTROM {
                exclusion_violation_count += 1;
                if exclusion_violation_examples.len() < MAX_EXCLUSION_EXAMPLES {
                    exclusion_violation_examples.push(PlacementExclusionViolation {
                        left_residue_id: left.residue_id,
                        left_residue_name: left.residue_name.clone(),
                        left_atom_name: left.atom_name.clone(),
                        left_position_angstrom: left.position_angstrom,
                        right_residue_id: right.residue_id,
                        right_residue_name: right.residue_name.clone(),
                        right_atom_name: right.atom_name.clone(),
                        right_position_angstrom: right.position_angstrom,
                        distance_angstrom: distance,
                        exclusion_distance_angstrom: exclusion_distance,
                        margin_angstrom: margin,
                    });
                }
            }
        }
    }

    let pbc = pbc_axes(&system.pbc);
    PlacementDiagnostics {
        bead_count: beads.len(),
        distinct_residue_count: residue_ids.len(),
        excluded_bead_count,
        pbc_axes: pbc,
        uses_minimum_image: pbc.iter().any(|enabled| *enabled),
        tolerance_angstrom: TOLERANCE_ANGSTROM,
        bounds_min_angstrom: (!beads.is_empty()).then_some(bounds_min),
        bounds_max_angstrom: (!beads.is_empty()).then_some(bounds_max),
        min_inter_residue_distance_angstrom,
        min_exclusion_margin_angstrom,
        exclusion_violation_count,
        exclusion_violation_examples,
    }
}

pub(super) fn diagnostic_exclusion_radius(bead: &EmittedBead) -> f32 {
    default_solvation_bead_radius() * bead.excluded_volume_factor.max(0.0)
}

pub(super) fn phase_density_summary(
    target_count: usize,
    placed_count: usize,
    initial_candidate_count: usize,
    final_candidate_count: usize,
    grid_squeeze_pass_count: usize,
) -> PlacementPhaseDensitySummary {
    let mut summary = PlacementPhaseDensitySummary {
        target_count,
        placed_count,
        initial_candidate_count,
        final_candidate_count,
        candidate_to_target_ratio: None,
        placement_fill_fraction: None,
        grid_squeeze_required: false,
    };
    finalize_phase_density(&mut summary, grid_squeeze_pass_count);
    summary
}

pub(super) fn finalize_phase_density(
    summary: &mut PlacementPhaseDensitySummary,
    grid_squeeze_pass_count: usize,
) {
    summary.candidate_to_target_ratio = (summary.target_count > 0)
        .then_some(summary.final_candidate_count as f32 / summary.target_count as f32);
    summary.placement_fill_fraction = (summary.final_candidate_count > 0)
        .then_some(summary.placed_count as f32 / summary.final_candidate_count as f32);
    summary.grid_squeeze_required = grid_squeeze_pass_count > 0;
}

pub(super) fn placement_seed(system: &BuildSystem) -> Option<u64> {
    (system.placement.mode == "seeded")
        .then_some(system.placement.random_seed)
        .flatten()
}

pub(super) fn placement_uses_random_candidates(system: &BuildSystem) -> bool {
    system.placement.mode == "seeded" && system.placement.candidate_source == "random"
}

pub(super) fn placement_algorithm_name(system: &BuildSystem) -> String {
    if placement_uses_random_candidates(system) {
        "seeded_random_leaflet_candidates_pair_edge_exclusion_relaxation".to_string()
    } else {
        "rectangular_grid_pair_edge_exclusion_relaxation".to_string()
    }
}

pub(super) fn solvent_algorithm_name(system: &BuildSystem, multi_zone: bool) -> String {
    match (
        multi_zone,
        placement_seed(system).is_some(),
        placement_uses_random_candidates(system),
    ) {
        (true, true, true) => "multi_zone_free_volume_count_seeded_random".to_string(),
        (true, true, false) => "multi_zone_free_volume_count_seeded_grid".to_string(),
        (true, false, _) => "multi_zone_free_volume_count_deterministic_grid".to_string(),
        (false, true, true) => "free_volume_count_seeded_random".to_string(),
        (false, true, false) => "free_volume_count_seeded_grid".to_string(),
        (false, false, _) => "free_volume_count_deterministic_grid".to_string(),
    }
}

pub(super) fn seeded_inserted_orientation_degrees(
    system: &BuildSystem,
    component: &InsertedComponent,
    copy_idx: usize,
) -> [f32; 3] {
    if component.placement.orientation != "seeded_random" {
        return [0.0, 0.0, 0.0];
    }
    let Some(seed) = placement_seed(system) else {
        return [0.0, 0.0, 0.0];
    };
    let mut state = mix_seed(
        seed,
        "inserted_orientation",
        component.name.as_bytes(),
        copy_idx,
    );
    [
        seeded_angle_degrees(&mut state),
        seeded_angle_degrees(&mut state),
        seeded_angle_degrees(&mut state),
    ]
}

pub(super) fn seeded_angle_degrees(state: &mut u64) -> f32 {
    let value = splitmix64_next(state);
    ((value >> 11) as f64 * (360.0 / ((1u64 << 53) as f64))) as f32
}

pub(super) fn seeded_unit_f32(state: &mut u64) -> f32 {
    let value = splitmix64_next(state);
    ((value >> 11) as f64 / ((1u64 << 53) as f64)) as f32
}

pub(super) fn shuffle_points3(points: &mut [[f32; 3]], seed: u64) {
    if points.len() < 2 {
        return;
    }
    let mut state = seed.max(1);
    for idx in (1..points.len()).rev() {
        state = splitmix64_next(&mut state);
        let swap_idx = (state as usize) % (idx + 1);
        points.swap(idx, swap_idx);
    }
}

pub(super) fn shuffle_usize(values: &mut [usize], seed: u64) {
    if values.len() < 2 {
        return;
    }
    let mut state = seed.max(1);
    for idx in (1..values.len()).rev() {
        state = splitmix64_next(&mut state);
        let swap_idx = (state as usize) % (idx + 1);
        values.swap(idx, swap_idx);
    }
}

pub(super) fn mix_seed(seed: u64, phase: &str, name: &[u8], ordinal: usize) -> u64 {
    let mut mixed = seed ^ 0x9E37_79B9_7F4A_7C15u64;
    for byte in phase.as_bytes().iter().chain(name.iter()) {
        mixed ^= *byte as u64;
        mixed = mixed.wrapping_mul(0x1000_0000_01B3);
    }
    mixed ^ (ordinal as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
}

pub(super) fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub(super) fn transform_inserted_position(
    position: [f32; 3],
    source_center: [f32; 3],
    rotate_degrees_xyz: [f32; 3],
    target: [f32; 3],
) -> [f32; 3] {
    let centered = [
        position[0] - source_center[0],
        position[1] - source_center[1],
        position[2] - source_center[2],
    ];
    let rotated = rotate_xyz_reference_order(centered, rotate_degrees_xyz);
    [
        rotated[0] + target[0],
        rotated[1] + target[1],
        rotated[2] + target[2],
    ]
}

pub(super) fn rotate_xyz_reference_order(
    position: [f32; 3],
    rotate_degrees_xyz: [f32; 3],
) -> [f32; 3] {
    let [rx, ry, rz] = rotate_degrees_xyz.map(f32::to_radians);
    let (sin_x, cos_x) = rx.sin_cos();
    let (sin_y, cos_y) = ry.sin_cos();
    let (sin_z, cos_z) = rz.sin_cos();
    let rot_x = [[1.0, 0.0, 0.0], [0.0, cos_x, -sin_x], [0.0, sin_x, cos_x]];
    let rot_y = [[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]];
    let rot_z = [[cos_z, -sin_z, 0.0], [sin_z, cos_z, 0.0], [0.0, 0.0, 1.0]];
    let rot_xy = mat3_mul(rot_x, rot_y);
    let rot_xyz = mat3_mul(rot_xy, rot_z);
    [
        rot_xyz[0][0] * position[0] + rot_xyz[0][1] * position[1] + rot_xyz[0][2] * position[2],
        rot_xyz[1][0] * position[0] + rot_xyz[1][1] * position[1] + rot_xyz[1][2] * position[2],
        rot_xyz[2][0] * position[0] + rot_xyz[2][1] * position[1] + rot_xyz[2][2] * position[2],
    ]
}

pub(super) fn mat3_mul(left: [[f32; 3]; 3], right: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = left[row][0] * right[0][col]
                + left[row][1] * right[1][col]
                + left[row][2] * right[2][col];
        }
    }
    out
}

pub(super) fn molecule_center_of_geometry(atoms: &[AtomRecord]) -> [f32; 3] {
    let inv_count = 1.0 / atoms.len() as f32;
    let mut sum = [0.0f32; 3];
    for atom in atoms {
        sum[0] += atom.position.x;
        sum[1] += atom.position.y;
        sum[2] += atom.position.z;
    }
    [sum[0] * inv_count, sum[1] * inv_count, sum[2] * inv_count]
}

pub(super) fn molecule_residue_templates(atoms: &[AtomRecord]) -> Vec<(i32, String)> {
    let mut residues = Vec::new();
    for atom in atoms {
        let key = (atom.resid, atom.resname.clone());
        if !residues.contains(&key) {
            residues.push(key);
        }
    }
    residues
}
