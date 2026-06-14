use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use warp_common::charge::{
    charges_match, spread_total_charge, sum_bead_charges, ComponentCharge, GromacsMoleculeTopology,
};
use warp_structure::{io::read_molecule, AtomRecord};

use crate::build_box::{has_periodic_axis, pbc_axes, resolved_box_metadata, validate_box_contract};
use crate::build_layout::{
    rectangular_leaflet_grid, relax_leaflet_points_periodic,
    relax_leaflet_points_with_projector_basis, CircularExclusion, LayoutBasis2D, LayoutBounds,
    LayoutPeriodicity, LayoutPoint, PlacementMetrics, RelaxationConfig,
};
use crate::build_lipids::lookup_lipid_template;
use crate::build_solutes::{lookup_solute_template, lookup_solute_template_bonds};

#[path = "build_contract_contract.rs"]
mod build_contract_contract;
#[path = "build_contract_defaults.rs"]
mod build_contract_defaults;
#[path = "build_contract_geometry_backend/mod.rs"]
mod build_contract_geometry_backend;
#[path = "build_contract_inserted.rs"]
mod build_contract_inserted;
#[path = "build_contract_inserted_library.rs"]
mod build_contract_inserted_library;
#[path = "build_contract_inserted_placement.rs"]
mod build_contract_inserted_placement;
#[path = "build_contract_ion_library.rs"]
mod build_contract_ion_library;
#[path = "build_contract_leaflet_area.rs"]
mod build_contract_leaflet_area;
#[path = "build_contract_leaflet_constraints.rs"]
mod build_contract_leaflet_constraints;
#[path = "build_contract_leaflet_emit.rs"]
mod build_contract_leaflet_emit;
#[path = "build_contract_leaflet_grid.rs"]
mod build_contract_leaflet_grid;
#[path = "build_contract_math3d.rs"]
mod build_contract_math3d;
#[path = "build_contract_molecule.rs"]
mod build_contract_molecule;
#[path = "build_contract_neutralization.rs"]
mod build_contract_neutralization;
#[path = "build_contract_output.rs"]
mod build_contract_output;
#[path = "build_contract_output_files.rs"]
mod build_contract_output_files;
#[path = "build_contract_placement_utils.rs"]
mod build_contract_placement_utils;
#[path = "build_contract_polygon_utils.rs"]
mod build_contract_polygon_utils;
#[path = "build_contract_protein_boundary.rs"]
mod build_contract_protein_boundary;
#[path = "build_contract_protein_boundary_build.rs"]
mod build_contract_protein_boundary_build;
#[path = "build_contract_protein_footprint.rs"]
mod build_contract_protein_footprint;
#[path = "build_contract_protein_hull.rs"]
mod build_contract_protein_hull;
#[path = "build_contract_protein_inset_area.rs"]
mod build_contract_protein_inset_area;
#[path = "build_contract_region_area.rs"]
mod build_contract_region_area;
#[path = "build_contract_region_circle.rs"]
mod build_contract_region_circle;
#[path = "build_contract_region_circle_polygon.rs"]
mod build_contract_region_circle_polygon;
#[path = "build_contract_region_core.rs"]
mod build_contract_region_core;
#[path = "build_contract_region_curve_overlap.rs"]
mod build_contract_region_curve_overlap;
#[path = "build_contract_region_curve_polygon.rs"]
mod build_contract_region_curve_polygon;
#[path = "build_contract_region_curve_polygon_mixed.rs"]
mod build_contract_region_curve_polygon_mixed;
#[path = "build_contract_region_disjoint_mixed.rs"]
mod build_contract_region_disjoint_mixed;
#[path = "build_contract_region_ellipse.rs"]
mod build_contract_region_ellipse;
#[path = "build_contract_region_ellipse_rectangle.rs"]
mod build_contract_region_ellipse_rectangle;
#[path = "build_contract_region_geometry.rs"]
mod build_contract_region_geometry;
#[path = "build_contract_region_mixed.rs"]
mod build_contract_region_mixed;
#[path = "build_contract_region_polygon.rs"]
mod build_contract_region_polygon;
#[path = "build_contract_region_spatial.rs"]
mod build_contract_region_spatial;
#[path = "build_contract_request.rs"]
mod build_contract_request;
#[path = "build_contract_result.rs"]
mod build_contract_result;
#[path = "build_contract_run.rs"]
mod build_contract_run;
#[path = "build_contract_runtime.rs"]
mod build_contract_runtime;
#[path = "build_contract_solvent_amino.rs"]
mod build_contract_solvent_amino;
#[path = "build_contract_solvent_emission.rs"]
mod build_contract_solvent_emission;
#[path = "build_contract_solvent_library.rs"]
mod build_contract_solvent_library;
#[path = "build_contract_solvent_lookup.rs"]
mod build_contract_solvent_lookup;
#[path = "build_contract_solvent_placement.rs"]
mod build_contract_solvent_placement;
#[path = "build_contract_solvent_registry.rs"]
mod build_contract_solvent_registry;
#[path = "build_contract_solvent_tailcode.rs"]
mod build_contract_solvent_tailcode;
#[path = "build_contract_solvent_topology.rs"]
mod build_contract_solvent_topology;
#[path = "build_contract_stacked.rs"]
mod build_contract_stacked;
#[path = "build_contract_validation.rs"]
mod build_contract_validation;
#[path = "build_contract_validation_fields.rs"]
mod build_contract_validation_fields;

pub use build_contract_contract::{capabilities, example_request, schema_json};
use build_contract_defaults::*;
use build_contract_geometry_backend::{
    geo_nested_polygon_forest_inset_area, geo_nested_polygon_inset_area,
    geo_polygonized_region_union_area, geo_simple_polygon_union_area,
};
use build_contract_inserted::{
    append_inserted_charge, append_inserted_counts, emit_inserted_component,
    inserted_charge_topology_paths, inserted_molecule_types, load_molecule_definition,
    molecule_definition_beads,
};
use build_contract_inserted_placement::inserted_candidate_phase_offsets;
#[cfg(test)]
use build_contract_inserted_placement::{
    extend_seeded_kicked_inserted_centers, inserted_center_candidates, inserted_center_inside_box,
    inserted_center_is_clear,
};
use build_contract_ion_library::{
    known_anion_library_names, known_cation_library_names, lookup_ion_library,
};
use build_contract_leaflet_area::resolve_leaflet_lipid_counts;
use build_contract_leaflet_constraints::{
    analytic_allowed_leaflet_projection_with_boundary, confine_points_to_allowed_leaflet_regions,
    confine_points_to_protein_boundary, leaflet_allows_point, leaflet_has_spatial_regions,
    membrane_allows_layout_point, membrane_allows_layout_point_with_boundary,
    membrane_has_spatial_constraints, point_inside_layout_domain,
};
#[cfg(test)]
use build_contract_leaflet_constraints::{
    leaflet_allows_point_periodic, leaflet_allows_point_periodic_basis,
    nearest_allowed_leaflet_point, point_inside_bounds_with_radius_periodic,
};
use build_contract_leaflet_emit::{
    combined_leaflet_exclusions, default_lipid_radius, layout_leaflet_beads,
    normalized_lipid_beads, resolve_lipid_charge,
};
#[cfg(test)]
use build_contract_leaflet_grid::random_leaflet_grid;
use build_contract_leaflet_grid::{
    basis_fractional_margins, initial_leaflet_grid, leaflet_lipid_sequence,
};
use build_contract_math3d::{
    cell_center_candidates, distance_box_vectors, placement_box_size_angstrom,
    point_inside_vector_cell, random_cell_center, squared_distance3_for_system,
};
#[cfg(test)]
use build_contract_math3d::{squared_distance3, squared_distance3_pbc};
pub use build_contract_molecule::{
    ComponentFootprint, InsertedPlacement, MoleculeDefinition, MoleculeDefinitionAngle,
    MoleculeDefinitionBond, MoleculeDefinitionDihedral, MoleculeDefinitionResidue,
};
use build_contract_neutralization::{
    apply_delta_to_ions, ion_charge_sum, representative_charge, resolve_neutralization,
    solvent_neutralization_deltas,
};
#[cfg(test)]
use build_contract_output::solvent_library_topology_block;
use build_contract_output::write_coordinates_and_topology;
use build_contract_output_files::write_manifest;
use build_contract_placement_utils::{
    finalize_phase_density, mix_seed, molecule_center_of_geometry, molecule_residue_templates,
    phase_density_summary, placement_algorithm_name, placement_diagnostics, placement_seed,
    placement_uses_random_candidates, rotate_xyz_reference_order,
    seeded_inserted_orientation_degrees, seeded_unit_f32, shuffle_points3, shuffle_usize,
    solvent_algorithm_name, splitmix64_next, transform_inserted_position,
};
pub(in crate::build_contract) use build_contract_polygon_utils::{
    axis_aligned_rectangle_polygon_bounds, convex_polygon_inset_polygon, distance2,
    exact_simple_polygon_inset_area_before_topology_event, nested_polygon_forest_contains_point,
    nested_polygon_ring_depths, point_in_polygon_or_boundary, points_close2, polygon_area,
    polygon_boundary_contains_point, polygon_boundary_gap, polygon_distance,
    polygon_hole_rejects_point, polygon_non_adjacent_edge_clearance, polygon_perimeter,
    project_point_outside_polygon_hole, project_point_to_nested_polygon_forest, rectangle_area,
    rectangle_bounds_distance, rectangle_bounds_strictly_contains, rectangle_containment_gap,
    rounded_rectangle_dilation_area, segments_intersect, segments_share_endpoint,
    signed_polygon_area, squared_distance2,
};
use build_contract_protein_boundary::{ProteinBoundaryCircle, ProteinBoundaryGeometry};
pub(in crate::build_contract) use build_contract_protein_boundary_build::{
    leaflet_geometry_diagnostics, membrane_boundary_matches_protein, protein_boundary_geometry,
    IfEmptyThen,
};
#[cfg(test)]
use build_contract_protein_boundary_build::{
    nested_polygon_forest_from_components, nested_polygon_from_components,
};
use build_contract_protein_footprint::{
    boundary_protein_exclusion_area_for_leaflet, boundary_protein_exclusions_for_leaflet,
    protein_component_exclusions, protein_footprint_area_for_leaflet, xy_center_of_points,
    xy_radius_of_points,
};
use build_contract_protein_hull::{
    alpha_shape, alpha_shape_components, buffered_convex_hull_area, concave_hull, convex_hull,
    convex_hull_area, default_alpha_radius, ordered_concave_boundary,
    polygon_has_self_intersections,
};
#[cfg(test)]
use build_contract_protein_inset_area::{
    exact_axis_aligned_rectangle_nested_forest_inset_area, exact_convex_nested_forest_inset_area,
    exact_simple_nested_forest_inset_area_before_topology_event,
};
use build_contract_protein_inset_area::{
    exact_axis_aligned_rectangle_nested_inset_area, exact_convex_multipolygon_inset_union_area,
    exact_convex_nested_inset_area, exact_simple_nested_inset_area_before_topology_event,
    multipolygon_area_with_inset_union, multipolygon_components_overlap,
    nested_polygon_forest_area_estimate, polygon_boundary_area_estimate,
    polygon_collection_layout_bounds,
};
#[cfg(test)]
use build_contract_region_area::{region_area_estimate, region_union_area_angstrom2};
use build_contract_region_circle::{
    circle_polygon_intersection_area, circle_rectangle_intersection_area,
    exact_circle_region_union_area, exact_clipped_circle_union_area, CircleRegion,
};
#[cfg(test)]
use build_contract_region_circle::{circle_upper_arc_integral, exact_unclipped_circle_union_area};
use build_contract_region_circle_polygon::{
    axis_aligned_bounds_intersection, circle_convex_polygon_union_intersection_area,
    exact_circle_axis_aligned_rectangle_region_union_area,
    exact_circle_axis_aligned_rectangles_region_union_area,
    exact_circle_convex_polygon_region_union_area, exact_circle_convex_polygons_region_union_area,
    exact_circle_rotated_rectangle_region_union_area,
    exact_circles_axis_aligned_rectangle_region_union_area,
    exact_circles_axis_aligned_rectangles_region_union_area,
    exact_convex_polygon_union_area_from_polygons,
    exact_disjoint_circles_convex_polygons_region_union_area,
};
#[cfg(test)]
use build_contract_region_circle_polygon::{
    circle_axis_aligned_rectangle_bounds_union_intersection_area,
    clipped_circle_union_axis_aligned_rectangle_bounds_union_intersection_area,
};
use build_contract_region_core::{
    axis_aligned_bounds_overlap, axis_aligned_rectangle_bounds,
    clipped_axis_aligned_rectangle_bounds, clipped_bounds, convex_polygon_intersection,
    ensure_ccw_polygon, exact_single_region_area, layout_bounds_polygon, line_segment_intersection,
    point_left_of_edge, polygon_is_convex, region_bounds, region_union_bounds,
};
use build_contract_region_curve_overlap::{
    axis_aligned_ellipse_pair_intersection_area_clipped,
    circle_axis_aligned_ellipse_intersection_area,
    circle_axis_aligned_ellipse_intersection_area_clipped,
    rotated_ellipse_pair_intersection_area_clipped,
};
#[cfg(test)]
use build_contract_region_curve_polygon::simple_polygon_for_region_clipped_to_bounds;
use build_contract_region_curve_polygon::{
    exact_circle_simple_polygon_region_union_area, exact_circle_simple_polygons_region_union_area,
    exact_ellipse_convex_polygon_region_union_area,
    exact_ellipse_convex_polygons_region_union_area,
    exact_ellipse_simple_polygon_region_union_area,
    exact_ellipse_simple_polygons_region_union_area,
};
use build_contract_region_curve_polygon_mixed::{
    exact_circle_convex_polygons_disjoint_ellipses_region_union_area,
    exact_circle_convex_polygons_disjoint_mixed_shapes_region_union_area,
    exact_ellipse_convex_polygons_disjoint_circles_region_union_area,
    exact_ellipse_convex_polygons_disjoint_mixed_shapes_region_union_area,
};
use build_contract_region_disjoint_mixed::{
    exact_circle_disjoint_ellipses_region_union_area,
    exact_circle_disjoint_mixed_shapes_region_union_area,
    exact_convex_polygon_disjoint_mixed_shapes_region_union_area,
    exact_ellipse_disjoint_circles_region_union_area,
    exact_ellipse_disjoint_mixed_shapes_region_union_area,
    exact_rectangle_disjoint_mixed_shapes_region_union_area, region_geometry_kind,
};
#[cfg(test)]
use build_contract_region_ellipse::ellipse_upper_arc_integral;
use build_contract_region_ellipse::{
    axis_aligned_ellipse_rectangle_intersection_area, ellipse_rectangle_intersection_area,
    exact_disjoint_ellipse_region_union_area, exact_similar_oriented_ellipse_region_union_area,
};
#[cfg(test)]
use build_contract_region_ellipse_rectangle::ellipse_axis_aligned_rectangle_bounds_union_intersection_area;
use build_contract_region_ellipse_rectangle::{
    exact_axis_aligned_ellipse_axis_aligned_rectangle_region_union_area,
    exact_axis_aligned_ellipse_pair_region_union_area,
    exact_circle_oriented_ellipse_region_union_area,
    exact_clipped_circle_rotated_ellipse_region_union_area,
    exact_disjoint_ellipses_axis_aligned_rectangles_region_union_area,
    exact_ellipse_axis_aligned_rectangles_region_union_area,
    exact_oriented_ellipse_rotated_rectangle_region_union_area,
    exact_rotated_ellipse_axis_aligned_rectangle_region_union_area,
    exact_rotated_ellipse_pair_region_union_area,
};
use build_contract_region_geometry::{
    conservative_grid_region_union_error_bound, convex_polygon_clip_half_plane,
    exact_axis_aligned_rectangle_polygon_inset_area, exact_convex_polygon_inset_area,
    forward_rotated_xy, inverse_rotated_point, inverse_rotated_xy,
    nearest_point_on_polygon_boundary, nearest_point_on_segment, point_in_polygon,
    polygon_area_with_inset, polygon_boundary_distance, polygon_bounds, polygon_bounds_center,
    polygon_centroid, project_point_to_polygon_with_margin, rectangle_boundary_distance,
    transformed_polygon_points,
};
use build_contract_region_mixed::{
    exact_component_mixed_region_union_area, exact_disjoint_mixed_region_union_area,
    exact_pair_region_union_area_without_disjoint, regions_are_exactly_disjoint,
};
use build_contract_region_polygon::{
    convex_polygon_for_region, exact_axis_aligned_rectangle_bounds_union_area,
    exact_axis_aligned_rectangle_union_area, exact_convex_polygon_region_union_area,
    exact_disjoint_simple_polygon_region_union_area,
    exact_rectangle_simple_polygon_region_union_area, exact_simple_polygon_region_union_area,
    exact_simple_polygon_union_area_from_polygons, grid_region_union_area, polygon_within_bounds,
};
#[cfg(test)]
use build_contract_region_spatial::periodic_point_images_basis;
use build_contract_region_spatial::{
    closest_periodic_region_image_basis, protein_boundary_margin_violation_distance,
    region_boundary_distance_periodic_basis, region_contains_point,
    region_contains_point_periodic_basis, wrap_point_for_layout_basis,
};
pub use build_contract_request::{
    BuildBeadTemplate, BuildRequest, BuildSystem, ExclusionZone, InsertedComponent, LeafletRegion,
    LeafletRequest, LipidComponent, MembraneRequest, PlacementOptions, ProteinBoundaryRequest,
    RegionGeometry, StackedMembraneLayer, StackedMembranesRequest,
};
pub use build_contract_result::{
    BuildArtifacts, BuildBoxSummary, BuildEnvironment, BuildEvent, BuildIssue, BuildOutputPolicy,
    BuildOutputs, BuildResult, BuildSummary, ChargeBuildSummary, ComponentChargeSummary,
    InsertedFloodPlacementSummary, IonComponent, IonPolicy, LeafletAreaSummary,
    LeafletGeometryConstraintDiagnostic, LeafletGeometryDiagnostics, LeafletPlacementSummary,
    NeutralizationSummary, PlacementBuildSummary, PlacementDiagnostics,
    PlacementExclusionViolation, PlacementPhaseDensitySummary, SolventComponent,
    SolventPlacementSummary, SolventPolicy, SolventZone,
};
pub use build_contract_run::{run_request, run_request_json, validate_request_json};
use build_contract_runtime::{
    EmittedBead, InsertedKind, ResolvedLipid, SolventEmission, SolventPlacementPlan,
};
use build_contract_solvent_amino::{
    known_amino_acid_solvent_names, lookup_amino_acid_solvent_library,
};
use build_contract_solvent_emission::{emit_solvent_and_ions, solvent_box_size_angstrom};
#[cfg(test)]
use build_contract_solvent_emission::{point_inside_solvent_zone, resolved_ion_species};
use build_contract_solvent_library::{
    atomistic_tip3_water_entry, atomistic_tip4_water_entry, atomistic_tip5_water_entry,
    ion_atom_name, ion_residue_name, lookup_small_molecule_solvent_library, normalize_library_name,
    resolved_solvent_species, solvent_library_bead, standard_solvent_entry, IonLibraryEntry,
    ResolvedIonSpecies, ResolvedSolventSpecies, SolventLibraryBead, SolventLibraryEntry,
};
pub(in crate::build_contract) use build_contract_solvent_lookup::{
    known_solvent_library_names, lookup_solvent_library,
};
use build_contract_solvent_placement::solvent_placement_plan;
#[cfg(test)]
use build_contract_solvent_placement::{
    candidate_overlaps_occupied_for_system, occupied_position_bins, pbc_occupied_points,
    solvent_candidate_allowed, solvent_candidates,
};
use build_contract_solvent_registry::lookup_standard_solvent_library;
use build_contract_solvent_tailcode::{
    known_ltf_named_tailcode_solvent_names, lookup_tailcode_solvent_library,
};
use build_contract_solvent_topology::{standard_solvent_angles, standard_solvent_bonds};
use build_contract_stacked::expand_stacked_membranes;
use build_contract_validation::validate_request;
use build_contract_validation_fields::split_tailcode_solvent_name;

pub const BUILD_SCHEMA_VERSION: &str = "warp-cg.build.v1";
const MOLECULE_DEFINITION_SCHEMA_VERSION: &str = "warp-cg.molecule_definition.v1";
const AVOGADRO: f64 = 6.022_140_76e23;
const DEFAULT_PROTEIN_FOOTPRINT_HEIGHT_ANGSTROM: f32 = 18.0;
const DEFAULT_PROTEIN_FOOTPRINT_Z_BUFFER_ANGSTROM: f32 = 1.32;
const DEFAULT_PROTEIN_FOOTPRINT_AREA_BUFFER_ANGSTROM: f32 = 0.2;
const REGION_UNION_COARSE_GRID_SPACING_ANGSTROM: f32 = 1.0;
const REGION_UNION_FINE_GRID_SPACING_ANGSTROM: f32 = 0.5;
const REGION_BUFFER_SEGMENTS: f32 = 64.0;

fn resolve_component_counts(
    components: &[LipidComponent],
    target_total: usize,
    fraction_sum: f32,
) -> Vec<usize> {
    if components.is_empty() {
        return Vec::new();
    }

    let explicit_sum: usize = components.iter().filter_map(|lipid| lipid.count).sum();
    let remaining = target_total.saturating_sub(explicit_sum);
    let mut counts = vec![0usize; components.len()];
    for (idx, lipid) in components.iter().enumerate() {
        if let Some(count) = lipid.count {
            counts[idx] = count;
        }
    }

    if remaining == 0 {
        return counts;
    }

    let missing_indices = components
        .iter()
        .enumerate()
        .filter_map(|(idx, lipid)| lipid.count.is_none().then_some(idx))
        .collect::<Vec<_>>();
    if missing_indices.is_empty() {
        return counts;
    }

    if fraction_sum > 0.0 {
        let ratio_sum = missing_indices
            .iter()
            .map(|idx| components[*idx].fraction.unwrap_or(0.0))
            .sum::<f32>();
        if ratio_sum <= 0.0 {
            return counts;
        }
        let mut assigned = 0usize;
        for idx in &missing_indices {
            let exact = remaining as f32 * components[*idx].fraction.unwrap_or(0.0) / ratio_sum;
            let floor = exact.floor() as usize;
            counts[*idx] = floor;
            assigned += floor;
        }
        while assigned < remaining {
            let current_total = assigned.max(1) as f32;
            let idx = missing_indices
                .iter()
                .copied()
                .max_by(|left, right| {
                    let left_diff = components[*left].fraction.unwrap_or(0.0)
                        - counts[*left] as f32 / (current_total / ratio_sum);
                    let right_diff = components[*right].fraction.unwrap_or(0.0)
                        - counts[*right] as f32 / (current_total / ratio_sum);
                    left_diff
                        .partial_cmp(&right_diff)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| right.cmp(left))
                })
                .unwrap_or(missing_indices[0]);
            counts[idx] += 1;
            assigned += 1;
        }
        return counts;
    }

    let base = remaining / missing_indices.len();
    let extra = remaining % missing_indices.len();
    for (rank, idx) in missing_indices.into_iter().enumerate() {
        counts[idx] = base + usize::from(rank < extra);
    }
    counts
}

fn membrane_layout_bounds(system: &BuildSystem, membrane: &MembraneRequest) -> LayoutBounds {
    let center = membrane.center_xy_angstrom.unwrap_or([0.0, 0.0]);
    let placement_box = placement_box_size_angstrom(system);
    let size = membrane
        .size_xy_angstrom
        .unwrap_or([placement_box[0], placement_box[1]]);
    LayoutBounds {
        xmin: center[0] - size[0] * 0.5,
        xmax: center[0] + size[0] * 0.5,
        ymin: center[1] - size[1] * 0.5,
        ymax: center[1] + size[1] * 0.5,
    }
}

fn membrane_layout_basis(
    system: &BuildSystem,
    membrane: &MembraneRequest,
    _bounds: LayoutBounds,
    periodicity: LayoutPeriodicity,
) -> Option<LayoutBasis2D> {
    if !periodicity.x || !periodicity.y {
        return None;
    }
    let vectors = distance_box_vectors(system)?;
    let placement_box = placement_box_size_angstrom(system);
    let size = membrane
        .size_xy_angstrom
        .unwrap_or([placement_box[0], placement_box[1]]);
    let scale_x = if placement_box[0] > 0.0 {
        size[0] / placement_box[0]
    } else {
        1.0
    };
    let scale_y = if placement_box[1] > 0.0 {
        size[1] / placement_box[1]
    } else {
        1.0
    };
    if !scale_x.is_finite() || !scale_y.is_finite() || scale_x <= 0.0 || scale_y <= 0.0 {
        return None;
    }
    let a = [vectors[0][0] * scale_x, vectors[0][1] * scale_x];
    let b = [vectors[1][0] * scale_y, vectors[1][1] * scale_y];
    let center = membrane.center_xy_angstrom.unwrap_or([0.0, 0.0]);
    let origin = [
        center[0] - 0.5 * (a[0] + b[0]),
        center[1] - 0.5 * (a[1] + b[1]),
    ];
    LayoutBasis2D::new(origin, a, b)
}

fn layout_exclusions(exclusions: &[ExclusionZone]) -> Vec<CircularExclusion> {
    exclusions
        .iter()
        .map(|zone| CircularExclusion {
            x: zone.center_angstrom[0],
            y: zone.center_angstrom[1],
            radius: zone.radius_angstrom,
        })
        .collect()
}

fn leaflet_occupation_modifier(bounds: LayoutBounds, radii: &[f32]) -> f32 {
    let leaflet_area = (bounds.xmax - bounds.xmin) * (bounds.ymax - bounds.ymin);
    if leaflet_area <= 0.0 {
        return 0.0;
    }
    let square_area = radii
        .iter()
        .map(|radius| (radius * 2.0) * (radius * 2.0))
        .sum::<f32>();
    ((leaflet_area - square_area) / (leaflet_area * 2.0)).max(0.0)
}

#[cfg(test)]
#[path = "build_contract_tests.rs"]
mod tests;
