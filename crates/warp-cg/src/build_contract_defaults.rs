use super::{BuildOutputs, BUILD_SCHEMA_VERSION, MOLECULE_DEFINITION_SCHEMA_VERSION};

pub(super) fn default_schema_version() -> String {
    BUILD_SCHEMA_VERSION.to_string()
}

pub(super) fn default_molecule_definition_schema() -> String {
    MOLECULE_DEFINITION_SCHEMA_VERSION.to_string()
}

pub(super) fn default_mode() -> String {
    "membrane".to_string()
}

pub(super) fn default_force_field() -> String {
    "martini3".to_string()
}

pub(super) fn default_box_type() -> String {
    "orthorhombic".to_string()
}

pub(super) fn default_pbc() -> String {
    "xyz".to_string()
}

pub(super) fn default_placement_mode() -> String {
    "deterministic".to_string()
}

pub(super) fn default_candidate_source() -> String {
    "grid".to_string()
}

pub(super) fn default_solvate_voids() -> bool {
    true
}

pub(super) fn default_membrane_solvent_exclusion_half_thickness() -> f32 {
    20.0
}

pub(super) fn default_relaxation_enabled() -> bool {
    true
}

pub(super) fn default_relaxation_max_steps() -> usize {
    100
}

pub(super) fn default_relaxation_push_tolerance() -> f32 {
    0.01
}

pub(super) fn default_lipid_push_multiplier() -> f32 {
    0.25
}

pub(super) fn default_edge_push_multiplier() -> f32 {
    0.5
}

pub(super) fn default_component_count() -> usize {
    1
}

pub(super) fn default_component_ratio() -> f32 {
    1.0
}

pub(super) fn default_inserted_center() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

pub(super) fn default_inserted_center_method() -> String {
    "cog".to_string()
}

pub(super) fn default_inserted_orientation() -> String {
    "fixed".to_string()
}

pub(super) fn default_protein_footprint_buffer() -> f32 {
    5.0
}

pub(super) fn default_protein_boundary_mode() -> String {
    "inside".to_string()
}

pub(super) fn default_protein_boundary_geometry() -> String {
    "circle".to_string()
}

pub(super) fn default_protein_boundary_radius_strategy() -> String {
    "outer".to_string()
}

pub(super) fn default_protein_boundary_bead_exclusion_radius() -> f32 {
    1.32
}

pub(super) fn default_cation() -> String {
    "Na+".to_string()
}

pub(super) fn default_anion() -> String {
    "Cl-".to_string()
}

pub(super) fn default_cation_charge() -> i32 {
    1
}

pub(super) fn default_salt_method() -> String {
    "add".to_string()
}

pub(super) fn default_anion_charge() -> i32 {
    -1
}

pub(super) fn default_salt_molarity() -> f32 {
    0.15
}

pub(super) fn default_solvent_name() -> String {
    "W".to_string()
}

pub(super) fn default_solvent_molarity() -> f32 {
    55.56
}

pub(super) fn default_solvent_mapping_ratio() -> f32 {
    4.0
}

pub(super) fn default_solvent_molar_mass() -> f32 {
    18.01528
}

pub(super) fn default_solvent_density() -> f32 {
    996.69
}

pub(super) fn default_solvation_bead_radius() -> f32 {
    2.64
}

pub(super) fn default_solvation_grid_spacing() -> f32 {
    2.64
}

pub(super) fn default_solvation_exclusion_buffer() -> f32 {
    2.0
}

pub(super) fn default_solvent_per_lipid_cutoff() -> f32 {
    0.5
}

pub(super) fn default_outputs() -> BuildOutputs {
    BuildOutputs {
        coordinates: Some("outputs/membrane.gro".to_string()),
        gro: None,
        pdb: None,
        cif: None,
        topology: Some("outputs/topol.top".to_string()),
        log: None,
        snapshot: None,
        overwrite: default_outputs_overwrite(),
        backup_existing: false,
        manifest: "outputs/membrane_manifest.json".to_string(),
    }
}

pub(super) fn default_outputs_overwrite() -> bool {
    true
}

pub(super) fn default_stacked_membranes_pbc() -> String {
    "split".to_string()
}

pub(super) fn default_stacked_membranes_distance() -> Vec<f32> {
    vec![50.0]
}

pub(super) fn default_stacked_membranes_distance_type() -> Vec<String> {
    vec!["surface".to_string()]
}
