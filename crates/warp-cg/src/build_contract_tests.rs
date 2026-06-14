use super::*;
use serde_json::json;

fn read_gro_positions(path: &std::path::Path) -> Vec<[f32; 3]> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(2)
        .filter_map(|line| {
            if line.len() < 44 || line.trim().split_whitespace().count() == 3 {
                return None;
            }
            Some([
                line[20..28].trim().parse::<f32>().unwrap() * 10.0,
                line[28..36].trim().parse::<f32>().unwrap() * 10.0,
                line[36..44].trim().parse::<f32>().unwrap() * 10.0,
            ])
        })
        .collect()
}

fn read_gro_residue_positions(path: &std::path::Path, resname_filter: &str) -> Vec<[f32; 3]> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(2)
        .filter_map(|line| {
            if line.len() < 44 || line.trim().split_whitespace().count() == 3 {
                return None;
            }
            let resname = line[5..10].trim();
            (resname == resname_filter).then(|| {
                [
                    line[20..28].trim().parse::<f32>().unwrap() * 10.0,
                    line[28..36].trim().parse::<f32>().unwrap() * 10.0,
                    line[36..44].trim().parse::<f32>().unwrap() * 10.0,
                ]
            })
        })
        .collect()
}

fn read_gro_residue_atoms(path: &std::path::Path, resname_filter: &str) -> Vec<(String, [f32; 3])> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(2)
        .filter_map(|line| {
            if line.len() < 44 || line.trim().split_whitespace().count() == 3 {
                return None;
            }
            let resname = line[5..10].trim();
            (resname == resname_filter).then(|| {
                (
                    line[10..15].trim().to_string(),
                    [
                        line[20..28].trim().parse::<f32>().unwrap() * 10.0,
                        line[28..36].trim().parse::<f32>().unwrap() * 10.0,
                        line[36..44].trim().parse::<f32>().unwrap() * 10.0,
                    ],
                )
            })
        })
        .collect()
}

fn topology_molecule_charge_sum(text: &str, molecule_name: &str) -> f32 {
    let mut in_target_atoms = false;
    let mut saw_target = false;
    let mut sum = 0.0f32;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("[ moleculetype ]") {
            in_target_atoms = false;
            saw_target = false;
            continue;
        }
        if trimmed.starts_with('[') && trimmed != "[ atoms ]" {
            in_target_atoms = false;
            continue;
        }
        if !saw_target {
            let fields = trimmed.split_whitespace().collect::<Vec<_>>();
            if fields.first() == Some(&molecule_name) {
                saw_target = true;
            }
            continue;
        }
        if trimmed == "[ atoms ]" {
            in_target_atoms = true;
            continue;
        }
        if !in_target_atoms || trimmed.is_empty() || trimmed.starts_with(';') {
            continue;
        }
        let fields = trimmed.split_whitespace().collect::<Vec<_>>();
        if fields.len() >= 7 {
            sum += fields[6].parse::<f32>().unwrap();
        }
    }
    sum
}

fn test_system_with_pbc(pbc: &str) -> BuildSystem {
    BuildSystem {
        force_field: default_force_field(),
        box_type: default_box_type(),
        pbc: pbc.to_string(),
        box_size_angstrom: [10.0, 10.0, 10.0],
        unit_cell_angstrom: None,
        box_vectors_angstrom: None,
        placement: PlacementOptions::default(),
    }
}

fn test_triclinic_system_with_pbc(pbc: &str) -> BuildSystem {
    BuildSystem {
        force_field: default_force_field(),
        box_type: "triclinic".to_string(),
        pbc: pbc.to_string(),
        box_size_angstrom: [20.0, 20.0, 20.0],
        unit_cell_angstrom: Some([10.0, 10.0, 10.0, 90.0, 90.0, 60.0]),
        box_vectors_angstrom: None,
        placement: PlacementOptions::default(),
    }
}

fn test_bead(position_angstrom: [f32; 3]) -> EmittedBead {
    EmittedBead {
        residue_id: 1,
        residue_name: "OCC".to_string(),
        atom_name: "B".to_string(),
        charge_e: 0.0,
        position_angstrom,
        excluded_volume_factor: 1.0,
    }
}

fn test_bead_with_residue(residue_id: i32, position_angstrom: [f32; 3]) -> EmittedBead {
    EmittedBead {
        residue_id,
        residue_name: "OCC".to_string(),
        atom_name: "B".to_string(),
        charge_e: 0.0,
        position_angstrom,
        excluded_volume_factor: 1.0,
    }
}

fn numeric_circle_ellipse_overlap_area(
    circle: CircleRegion,
    ellipse_center: [f32; 2],
    ellipse_radius: [f32; 2],
    steps: usize,
) -> f32 {
    let bounds = LayoutBounds {
        xmin: (circle.center[0] - circle.radius).max(ellipse_center[0] - ellipse_radius[0]),
        xmax: (circle.center[0] + circle.radius).min(ellipse_center[0] + ellipse_radius[0]),
        ymin: (circle.center[1] - circle.radius).min(ellipse_center[1] - ellipse_radius[1]),
        ymax: (circle.center[1] + circle.radius).max(ellipse_center[1] + ellipse_radius[1]),
    };
    numeric_circle_ellipse_overlap_area_clipped(
        circle,
        ellipse_center,
        ellipse_radius,
        bounds,
        steps,
    )
}
fn numeric_circle_ellipse_overlap_area_clipped(
    circle: CircleRegion,
    ellipse_center: [f32; 2],
    ellipse_radius: [f32; 2],
    bounds: LayoutBounds,
    steps: usize,
) -> f32 {
    let xmin = (circle.center[0] - circle.radius)
        .max(ellipse_center[0] - ellipse_radius[0])
        .max(bounds.xmin);
    let xmax = (circle.center[0] + circle.radius)
        .min(ellipse_center[0] + ellipse_radius[0])
        .min(bounds.xmax);
    if xmin >= xmax || steps == 0 {
        return 0.0;
    }
    let dx = (xmax - xmin) / steps as f32;
    let mut area = 0.0;
    for idx in 0..steps {
        let x = xmin + (idx as f32 + 0.5) * dx;
        let circle_chord = (circle.radius.powi(2) - (x - circle.center[0]).powi(2))
            .max(0.0)
            .sqrt();
        let ellipse_chord = ellipse_radius[1]
            * (1.0 - ((x - ellipse_center[0]) / ellipse_radius[0]).powi(2))
                .max(0.0)
                .sqrt();
        let top = (circle.center[1] + circle_chord)
            .min(ellipse_center[1] + ellipse_chord)
            .min(bounds.ymax);
        let bottom = (circle.center[1] - circle_chord)
            .max(ellipse_center[1] - ellipse_chord)
            .max(bounds.ymin);
        area += (top - bottom).max(0.0) * dx;
    }
    area
}
fn numeric_axis_aligned_ellipse_pair_union_area(
    left_center: [f32; 2],
    left_radius: [f32; 2],
    right_center: [f32; 2],
    right_radius: [f32; 2],
    bounds: LayoutBounds,
    steps: usize,
) -> f32 {
    if steps == 0 {
        return 0.0;
    }
    let domain_xmin = bounds
        .xmin
        .max((left_center[0] - left_radius[0]).min(right_center[0] - right_radius[0]));
    let domain_xmax = bounds
        .xmax
        .min((left_center[0] + left_radius[0]).max(right_center[0] + right_radius[0]));
    if domain_xmin >= domain_xmax {
        return 0.0;
    }
    let dx = (domain_xmax - domain_xmin) / steps as f32;
    let mut area = 0.0;
    for idx in 0..steps {
        let x = domain_xmin + (idx as f32 + 0.5) * dx;
        let mut intervals = Vec::new();
        for (center, radius) in [(left_center, left_radius), (right_center, right_radius)] {
            let normalized_x = (x - center[0]) / radius[0];
            if normalized_x.abs() <= 1.0 {
                let chord = radius[1] * (1.0 - normalized_x.powi(2)).sqrt();
                let ymin = (center[1] - chord).max(bounds.ymin);
                let ymax = (center[1] + chord).min(bounds.ymax);
                if ymin < ymax {
                    intervals.push((ymin, ymax));
                }
            }
        }
        intervals.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut length = 0.0;
        let mut current: Option<(f32, f32)> = None;
        for interval in intervals {
            match current {
                Some((start, end)) if interval.0 <= end => {
                    current = Some((start, end.max(interval.1)));
                }
                Some((start, end)) => {
                    length += end - start;
                    current = Some(interval);
                }
                None => current = Some(interval),
            }
        }
        if let Some((start, end)) = current {
            length += end - start;
        }
        area += length * dx;
    }
    area
}
fn numeric_rotated_ellipse_rectangle_area(
    center: [f32; 2],
    radius: [f32; 2],
    rotate_degrees: f32,
    bounds: LayoutBounds,
    steps: usize,
) -> f32 {
    if steps == 0 {
        return 0.0;
    }
    let dx = (bounds.xmax - bounds.xmin) / steps as f32;
    let dy = (bounds.ymax - bounds.ymin) / steps as f32;
    let cell_area = dx * dy;
    let mut inside = 0usize;
    for ix in 0..steps {
        let x = bounds.xmin + (ix as f32 + 0.5) * dx;
        for iy in 0..steps {
            let y = bounds.ymin + (iy as f32 + 0.5) * dy;
            let local = inverse_rotated_xy([x, y], center, rotate_degrees);
            if (local[0] / radius[0]).powi(2) + (local[1] / radius[1]).powi(2) <= 1.0 {
                inside += 1;
            }
        }
    }
    inside as f32 * cell_area
}
fn numeric_oriented_ellipse_rotated_rectangle_union_area(
    ellipse_center: [f32; 2],
    ellipse_radius: [f32; 2],
    ellipse_rotate_degrees: f32,
    rectangle_center: [f32; 2],
    rectangle_size: [f32; 2],
    rectangle_rotate_degrees: f32,
    bounds: LayoutBounds,
    steps: usize,
) -> f32 {
    if steps == 0 {
        return 0.0;
    }
    let dx = (bounds.xmax - bounds.xmin) / steps as f32;
    let dy = (bounds.ymax - bounds.ymin) / steps as f32;
    let cell_area = dx * dy;
    let mut inside = 0usize;
    for ix in 0..steps {
        let x = bounds.xmin + (ix as f32 + 0.5) * dx;
        for iy in 0..steps {
            let y = bounds.ymin + (iy as f32 + 0.5) * dy;
            let ellipse_local = inverse_rotated_xy([x, y], ellipse_center, ellipse_rotate_degrees);
            let inside_ellipse = (ellipse_local[0] / ellipse_radius[0]).powi(2)
                + (ellipse_local[1] / ellipse_radius[1]).powi(2)
                <= 1.0;
            let rectangle_local =
                inverse_rotated_xy([x, y], rectangle_center, rectangle_rotate_degrees);
            let inside_rectangle = rectangle_local[0].abs() <= rectangle_size[0] * 0.5
                && rectangle_local[1].abs() <= rectangle_size[1] * 0.5;
            if inside_ellipse || inside_rectangle {
                inside += 1;
            }
        }
    }
    inside as f32 * cell_area
}
fn numeric_rotated_ellipse_pair_union_area(
    left_center: [f32; 2],
    left_radius: [f32; 2],
    left_rotate_degrees: f32,
    right_center: [f32; 2],
    right_radius: [f32; 2],
    right_rotate_degrees: f32,
    bounds: LayoutBounds,
    steps: usize,
) -> f32 {
    if steps == 0 {
        return 0.0;
    }
    let dx = (bounds.xmax - bounds.xmin) / steps as f32;
    let dy = (bounds.ymax - bounds.ymin) / steps as f32;
    let cell_area = dx * dy;
    let mut inside = 0usize;
    for ix in 0..steps {
        let x = bounds.xmin + (ix as f32 + 0.5) * dx;
        for iy in 0..steps {
            let y = bounds.ymin + (iy as f32 + 0.5) * dy;
            let left_local = inverse_rotated_xy([x, y], left_center, left_rotate_degrees);
            let right_local = inverse_rotated_xy([x, y], right_center, right_rotate_degrees);
            let in_left = (left_local[0] / left_radius[0]).powi(2)
                + (left_local[1] / left_radius[1]).powi(2)
                <= 1.0;
            let in_right = (right_local[0] / right_radius[0]).powi(2)
                + (right_local[1] / right_radius[1]).powi(2)
                <= 1.0;
            if in_left || in_right {
                inside += 1;
            }
        }
    }
    inside as f32 * cell_area
}

#[path = "build_contract_test_parts/cell_geometry_and_capabilities.rs"]
mod cell_geometry_and_capabilities;
#[path = "build_contract_test_parts/circle_rectangle_region_unions.rs"]
mod circle_rectangle_region_unions;
#[path = "build_contract_test_parts/clipped_and_overlapping_ellipse_unions.rs"]
mod clipped_and_overlapping_ellipse_unions;
#[path = "build_contract_test_parts/coordinate_less_solvent_registry_inserted.rs"]
mod coordinate_less_solvent_registry_inserted;
#[path = "build_contract_test_parts/disjoint_mixed_shape_unions.rs"]
mod disjoint_mixed_shape_unions;
#[path = "build_contract_test_parts/ellipse_multi_shape_region_unions.rs"]
mod ellipse_multi_shape_region_unions;
#[path = "build_contract_test_parts/geo_backend_wrapping_and_holes.rs"]
mod geo_backend_wrapping_and_holes;
#[path = "build_contract_test_parts/geo_polygon_boolean_backend.rs"]
mod geo_polygon_boolean_backend;
#[path = "build_contract_test_parts/inserted_builtin_solute_charges.rs"]
mod inserted_builtin_solute_charges;
#[path = "build_contract_test_parts/inserted_component_schema_validation.rs"]
mod inserted_component_schema_validation;
#[path = "build_contract_test_parts/ltf_complex_lipid_solvent_aliases.rs"]
mod ltf_complex_lipid_solvent_aliases;
#[path = "build_contract_test_parts/ltf_sphingolipid_solvent_aliases.rs"]
mod ltf_sphingolipid_solvent_aliases;
#[path = "build_contract_test_parts/ltf_tailcode_solvent_aliases.rs"]
mod ltf_tailcode_solvent_aliases;
#[path = "build_contract_test_parts/mixed_curve_polygon_union_exact.rs"]
mod mixed_curve_polygon_union_exact;
#[path = "build_contract_test_parts/nonconvex_polygon_ellipse_unions.rs"]
mod nonconvex_polygon_ellipse_unions;
#[path = "build_contract_test_parts/output_and_charge_summary.rs"]
mod output_and_charge_summary;
#[path = "build_contract_test_parts/patch_region_area_counts.rs"]
mod patch_region_area_counts;
#[path = "build_contract_test_parts/protein_boundary_polygon_hulls.rs"]
mod protein_boundary_polygon_hulls;
#[path = "build_contract_test_parts/reference_counts_and_charge_overrides.rs"]
mod reference_counts_and_charge_overrides;
#[path = "build_contract_test_parts/seeded_candidate_sources.rs"]
mod seeded_candidate_sources;
#[path = "build_contract_test_parts/solvent_library_martini_small_molecules.rs"]
mod solvent_library_martini_small_molecules;
#[path = "build_contract_test_parts/solvent_library_nucleic_sirah_topology.rs"]
mod solvent_library_nucleic_sirah_topology;
#[path = "build_contract_test_parts/solvent_tailcodes_and_ions.rs"]
mod solvent_tailcodes_and_ions;
#[path = "build_contract_test_parts/stacked_membranes_and_protein_boundaries.rs"]
mod stacked_membranes_and_protein_boundaries;
#[path = "build_contract_test_parts/triclinic_eighth_neighbor_wrapping.rs"]
mod triclinic_eighth_neighbor_wrapping;
#[path = "build_contract_test_parts/triclinic_third_neighbor_wrapping.rs"]
mod triclinic_third_neighbor_wrapping;
#[path = "build_contract_test_parts/triclinic_wrapped_region_adaptive.rs"]
mod triclinic_wrapped_region_adaptive;
#[path = "build_contract_test_parts/triclinic_wrapped_region_adversarial.rs"]
mod triclinic_wrapped_region_adversarial;
