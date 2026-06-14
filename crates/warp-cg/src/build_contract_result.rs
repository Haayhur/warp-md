use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::build_layout::PlacementMetrics;

use super::build_contract_defaults::*;

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BuildEnvironment {
    #[serde(default)]
    pub ions: IonPolicy,
    #[serde(default)]
    pub solvent: SolventPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IonPolicy {
    #[serde(default)]
    pub neutralize: bool,
    #[serde(default = "default_salt_method")]
    #[schemars(default = "default_salt_method")]
    pub salt_method: String,
    #[serde(default = "default_salt_molarity")]
    #[schemars(default = "default_salt_molarity")]
    pub salt_molarity_mol_l: f32,
    #[serde(default = "default_cation")]
    #[schemars(default = "default_cation")]
    pub cation: String,
    #[serde(default = "default_anion")]
    #[schemars(default = "default_anion")]
    pub anion: String,
    #[serde(default = "default_cation_charge")]
    #[schemars(default = "default_cation_charge")]
    pub cation_charge_e: i32,
    #[serde(default = "default_anion_charge")]
    #[schemars(default = "default_anion_charge")]
    pub anion_charge_e: i32,
    #[serde(default)]
    pub cations: Vec<IonComponent>,
    #[serde(default)]
    pub anions: Vec<IonComponent>,
}

impl Default for IonPolicy {
    fn default() -> Self {
        Self {
            neutralize: false,
            salt_method: default_salt_method(),
            salt_molarity_mol_l: default_salt_molarity(),
            cation: default_cation(),
            anion: default_anion(),
            cation_charge_e: default_cation_charge(),
            anion_charge_e: default_anion_charge(),
            cations: Vec::new(),
            anions: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct IonComponent {
    pub name: String,
    #[serde(default = "default_component_ratio")]
    #[schemars(default = "default_component_ratio")]
    pub ratio: f32,
    #[serde(default)]
    pub charge_e: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SolventPolicy {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_solvent_name")]
    #[schemars(default = "default_solvent_name")]
    pub name: String,
    #[serde(default = "default_solvent_molarity")]
    #[schemars(default = "default_solvent_molarity")]
    pub molarity_mol_l: f32,
    #[serde(default = "default_solvent_mapping_ratio")]
    #[schemars(default = "default_solvent_mapping_ratio")]
    pub mapping_ratio: f32,
    #[serde(default = "default_solvent_molar_mass")]
    #[schemars(default = "default_solvent_molar_mass")]
    pub molar_mass_g_mol: f32,
    #[serde(default = "default_solvent_density")]
    #[schemars(default = "default_solvent_density")]
    pub density_kg_m3: f32,
    #[serde(default)]
    pub box_size_angstrom: Option<[f32; 3]>,
    #[serde(default)]
    pub center_angstrom: Option<[f32; 3]>,
    #[serde(default = "default_solvation_bead_radius")]
    #[schemars(default = "default_solvation_bead_radius")]
    pub excluded_bead_radius_angstrom: f32,
    #[serde(default = "default_solvation_grid_spacing")]
    #[schemars(default = "default_solvation_grid_spacing")]
    pub grid_spacing_angstrom: f32,
    #[serde(default = "default_solvation_exclusion_buffer")]
    #[schemars(default = "default_solvation_exclusion_buffer")]
    pub exclusion_buffer_angstrom: f32,
    #[serde(default)]
    pub solvent_per_lipid: Option<f32>,
    #[serde(default = "default_solvent_per_lipid_cutoff")]
    #[schemars(default = "default_solvent_per_lipid_cutoff")]
    pub solvent_per_lipid_cutoff: f32,
    #[serde(default)]
    pub species: Vec<SolventComponent>,
    #[serde(default)]
    pub zones: Vec<SolventZone>,
}

impl Default for SolventPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            name: default_solvent_name(),
            molarity_mol_l: default_solvent_molarity(),
            mapping_ratio: default_solvent_mapping_ratio(),
            molar_mass_g_mol: default_solvent_molar_mass(),
            density_kg_m3: default_solvent_density(),
            box_size_angstrom: None,
            center_angstrom: None,
            excluded_bead_radius_angstrom: default_solvation_bead_radius(),
            grid_spacing_angstrom: default_solvation_grid_spacing(),
            exclusion_buffer_angstrom: default_solvation_exclusion_buffer(),
            solvent_per_lipid: None,
            solvent_per_lipid_cutoff: default_solvent_per_lipid_cutoff(),
            species: Vec::new(),
            zones: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SolventZone {
    pub name: Option<String>,
    #[serde(default)]
    pub box_size_angstrom: Option<[f32; 3]>,
    #[serde(default)]
    pub center_angstrom: Option<[f32; 3]>,
    #[serde(default)]
    pub molarity_mol_l: Option<f32>,
    #[serde(default)]
    pub salt_molarity_mol_l: Option<f32>,
    #[serde(default)]
    pub solvent_per_lipid: Option<f32>,
    #[serde(default)]
    pub solvent_per_lipid_cutoff: Option<f32>,
    #[serde(default)]
    pub species: Vec<SolventComponent>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SolventComponent {
    pub name: String,
    #[serde(default = "default_component_ratio")]
    #[schemars(default = "default_component_ratio")]
    pub ratio: f32,
    #[serde(default = "default_solvent_mapping_ratio")]
    #[schemars(default = "default_solvent_mapping_ratio")]
    pub mapping_ratio: f32,
    #[serde(default = "default_solvent_molar_mass")]
    #[schemars(default = "default_solvent_molar_mass")]
    pub molar_mass_g_mol: f32,
    #[serde(default = "default_solvent_density")]
    #[schemars(default = "default_solvent_density")]
    pub density_kg_m3: f32,
    #[serde(default)]
    pub charge_e: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BuildOutputs {
    pub coordinates: Option<String>,
    #[serde(default)]
    pub gro: Option<String>,
    #[serde(default)]
    pub pdb: Option<String>,
    #[serde(default)]
    pub cif: Option<String>,
    pub topology: Option<String>,
    #[serde(default)]
    pub log: Option<String>,
    #[serde(default)]
    pub snapshot: Option<String>,
    #[serde(default = "default_outputs_overwrite")]
    #[schemars(default = "default_outputs_overwrite")]
    pub overwrite: bool,
    #[serde(default)]
    pub backup_existing: bool,
    pub manifest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildResult {
    pub schema_version: String,
    pub status: String,
    pub run_id: Option<String>,
    pub mode: String,
    pub box_meta: BuildBoxSummary,
    pub summary: BuildSummary,
    pub charge: ChargeBuildSummary,
    pub placement: PlacementBuildSummary,
    pub artifacts: BuildArtifacts,
    pub warnings: Vec<BuildIssue>,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildBoxSummary {
    pub box_type: String,
    pub pbc: String,
    pub box_size_angstrom: [f32; 3],
    pub unit_cell_angstrom: [f32; 6],
    pub box_vectors_angstrom: [[f32; 3]; 3],
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildSummary {
    pub membrane_count: usize,
    pub leaflet_count: usize,
    pub lipid_counts: BTreeMap<String, usize>,
    pub inserted_counts: BTreeMap<String, usize>,
    pub bead_count: usize,
    pub solvent_counts: BTreeMap<String, usize>,
    pub protein_count: usize,
    pub solute_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PlacementBuildSummary {
    pub algorithm: String,
    pub mode: String,
    pub candidate_source: String,
    pub random_seed: Option<u64>,
    pub inserted_flood: InsertedFloodPlacementSummary,
    pub leaflet_metrics: Vec<LeafletPlacementSummary>,
    pub solvent: Option<SolventPlacementSummary>,
    pub diagnostics: PlacementDiagnostics,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct InsertedFloodPlacementSummary {
    pub candidate_count: usize,
    pub grid_squeeze_pass_count: usize,
    pub squeezed_candidate_count: usize,
    pub min_spacing_angstrom: Option<f32>,
    pub kick_attempt_count: usize,
    pub kicked_inserted_count: usize,
    pub density: PlacementPhaseDensitySummary,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct PlacementPhaseDensitySummary {
    pub target_count: usize,
    pub placed_count: usize,
    pub initial_candidate_count: usize,
    pub final_candidate_count: usize,
    pub candidate_to_target_ratio: Option<f32>,
    pub placement_fill_fraction: Option<f32>,
    pub grid_squeeze_required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct LeafletPlacementSummary {
    pub membrane: String,
    pub leaflet: String,
    pub lipid_count: usize,
    pub exclusion_count: usize,
    pub area: LeafletAreaSummary,
    pub metrics: PlacementMetrics,
    pub geometry: LeafletGeometryDiagnostics,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct LeafletAreaSummary {
    pub available_area_angstrom2: Option<f32>,
    pub method: String,
    pub is_exact: bool,
    pub reported_error_bound_angstrom2: Option<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct LeafletGeometryDiagnostics {
    pub tolerance_angstrom: f32,
    pub constraint_count: usize,
    pub violation_count: usize,
    pub max_violation_angstrom: f32,
    pub constraints: Vec<LeafletGeometryConstraintDiagnostic>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct LeafletGeometryConstraintDiagnostic {
    pub name: String,
    pub kind: String,
    pub role: String,
    pub violation_count: usize,
    pub max_violation_angstrom: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PlacementDiagnostics {
    pub bead_count: usize,
    pub distinct_residue_count: usize,
    pub excluded_bead_count: usize,
    pub pbc_axes: [bool; 3],
    pub uses_minimum_image: bool,
    pub tolerance_angstrom: f32,
    pub bounds_min_angstrom: Option<[f32; 3]>,
    pub bounds_max_angstrom: Option<[f32; 3]>,
    pub min_inter_residue_distance_angstrom: Option<f32>,
    pub min_exclusion_margin_angstrom: Option<f32>,
    pub exclusion_violation_count: usize,
    pub exclusion_violation_examples: Vec<PlacementExclusionViolation>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PlacementExclusionViolation {
    pub left_residue_id: i32,
    pub left_residue_name: String,
    pub left_atom_name: String,
    pub left_position_angstrom: [f32; 3],
    pub right_residue_id: i32,
    pub right_residue_name: String,
    pub right_atom_name: String,
    pub right_position_angstrom: [f32; 3],
    pub distance_angstrom: f32,
    pub exclusion_distance_angstrom: f32,
    pub margin_angstrom: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeBuildSummary {
    pub component_charges: Vec<ComponentChargeSummary>,
    pub net_charge_before_neutralization_e: Option<f32>,
    pub solvent_charge_e: Option<f32>,
    pub baseline_ion_charge_e: Option<f32>,
    pub neutralization_input_charge_e: Option<f32>,
    pub neutralization: NeutralizationSummary,
    pub charge_sources: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SolventPlacementSummary {
    pub algorithm: String,
    pub mode: String,
    pub random_seed: Option<u64>,
    pub box_volume_nm3: f32,
    pub excluded_volume_nm3: f32,
    pub free_volume_nm3: f32,
    pub solvent_material_volume_nm3: f32,
    pub grid_point_count: usize,
    pub inserted_count: usize,
    pub grid_squeeze_pass_count: usize,
    pub squeezed_candidate_count: usize,
    pub min_grid_spacing_angstrom: Option<f32>,
    pub kick_attempt_count: usize,
    pub kicked_inserted_count: usize,
    pub density: PlacementPhaseDensitySummary,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ComponentChargeSummary {
    pub name: String,
    pub count: usize,
    pub per_instance_net_charge_e: Option<f32>,
    pub per_instance_bead_charge_sum_e: Option<f32>,
    pub charge_balance_delta_e: Option<f32>,
    pub total_charge_e: Option<f32>,
    pub source: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct NeutralizationSummary {
    pub enabled: bool,
    pub salt_method: String,
    pub counterion: Option<String>,
    pub counterion_count: usize,
    pub counterion_charge_e: Option<i32>,
    pub cation_delta: isize,
    pub anion_delta: isize,
    pub residual_charge_e: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildArtifacts {
    pub coordinates: Option<String>,
    pub gro: Option<String>,
    pub pdb: Option<String>,
    pub cif: Option<String>,
    pub topology: Option<String>,
    pub log: Option<String>,
    pub snapshot: Option<String>,
    pub manifest: String,
    pub output_policy: BuildOutputPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildOutputPolicy {
    pub overwrite: bool,
    pub backup_existing: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildIssue {
    pub code: String,
    pub path: String,
    pub message: String,
    pub severity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "event")]
pub enum BuildEvent {
    BuildStarted {
        schema_version: String,
        run_id: Option<String>,
    },
    ChargeResolved {
        schema_version: String,
        net_charge_e: Option<f32>,
    },
    BuildComplete {
        schema_version: String,
        status: String,
    },
}
