use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::build_contract_defaults::*;
use super::{BuildEnvironment, BuildOutputs, ComponentFootprint, InsertedPlacement, SolventZone};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BuildRequest {
    #[serde(default = "default_schema_version")]
    #[schemars(default = "default_schema_version")]
    pub schema_version: String,
    pub run_id: Option<String>,
    #[serde(default = "default_mode")]
    #[schemars(default = "default_mode")]
    pub mode: String,
    pub system: BuildSystem,
    pub membranes: Vec<MembraneRequest>,
    #[serde(default)]
    pub stacked_membranes: Vec<StackedMembranesRequest>,
    #[serde(default)]
    pub proteins: Vec<InsertedComponent>,
    #[serde(default)]
    pub solutes: Vec<InsertedComponent>,
    #[serde(default)]
    pub environment: BuildEnvironment,
    #[serde(default = "default_outputs")]
    #[schemars(default = "default_outputs")]
    pub outputs: BuildOutputs,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StackedMembranesRequest {
    pub name: Option<String>,
    #[serde(default = "default_stacked_membranes_pbc")]
    #[schemars(default = "default_stacked_membranes_pbc")]
    pub pbc: String,
    #[serde(default = "default_stacked_membranes_distance")]
    #[schemars(default = "default_stacked_membranes_distance")]
    pub distance_angstrom: Vec<f32>,
    #[serde(default = "default_stacked_membranes_distance_type")]
    #[schemars(default = "default_stacked_membranes_distance_type")]
    pub distance_type: Vec<String>,
    pub layers: Vec<StackedMembraneLayer>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StackedMembraneLayer {
    pub membrane: MembraneRequest,
    #[serde(default)]
    pub solvent: Option<SolventZone>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BuildSystem {
    #[serde(default = "default_force_field")]
    #[schemars(default = "default_force_field")]
    pub force_field: String,
    #[serde(default = "default_box_type")]
    #[schemars(default = "default_box_type")]
    pub box_type: String,
    #[serde(default = "default_pbc")]
    #[schemars(default = "default_pbc")]
    pub pbc: String,
    pub box_size_angstrom: [f32; 3],
    #[serde(default)]
    pub unit_cell_angstrom: Option<[f32; 6]>,
    #[serde(default)]
    pub box_vectors_angstrom: Option<[[f32; 3]; 3]>,
    #[serde(default)]
    pub placement: PlacementOptions,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlacementOptions {
    #[serde(default = "default_placement_mode")]
    #[schemars(default = "default_placement_mode")]
    pub mode: String,
    #[serde(default = "default_candidate_source")]
    #[schemars(default = "default_candidate_source")]
    pub candidate_source: String,
    #[serde(default)]
    pub random_seed: Option<u64>,
    #[serde(default = "default_relaxation_enabled")]
    #[schemars(default = "default_relaxation_enabled")]
    pub relaxation: bool,
    #[serde(default = "default_relaxation_max_steps")]
    #[schemars(default = "default_relaxation_max_steps")]
    pub max_steps: usize,
    #[serde(default = "default_relaxation_push_tolerance")]
    #[schemars(default = "default_relaxation_push_tolerance")]
    pub push_tolerance_angstrom: f32,
    #[serde(default = "default_lipid_push_multiplier")]
    #[schemars(default = "default_lipid_push_multiplier")]
    pub lipid_push_multiplier: f32,
    #[serde(default = "default_edge_push_multiplier")]
    #[schemars(default = "default_edge_push_multiplier")]
    pub edge_push_multiplier: f32,
}

impl Default for PlacementOptions {
    fn default() -> Self {
        Self {
            mode: default_placement_mode(),
            candidate_source: default_candidate_source(),
            random_seed: None,
            relaxation: default_relaxation_enabled(),
            max_steps: default_relaxation_max_steps(),
            push_tolerance_angstrom: default_relaxation_push_tolerance(),
            lipid_push_multiplier: default_lipid_push_multiplier(),
            edge_push_multiplier: default_edge_push_multiplier(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MembraneRequest {
    pub name: String,
    #[serde(default)]
    pub center_xy_angstrom: Option<[f32; 2]>,
    #[serde(default)]
    pub size_xy_angstrom: Option<[f32; 2]>,
    #[serde(default)]
    pub center_z_angstrom: f32,
    #[serde(default = "default_solvate_voids")]
    #[schemars(default = "default_solvate_voids")]
    pub solvate_voids: bool,
    #[serde(default = "default_membrane_solvent_exclusion_half_thickness")]
    #[schemars(default = "default_membrane_solvent_exclusion_half_thickness")]
    pub solvent_exclusion_half_thickness_angstrom: f32,
    #[serde(default)]
    pub protein_boundary: Option<ProteinBoundaryRequest>,
    pub leaflets: Vec<LeafletRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ProteinBoundaryRequest {
    #[serde(default = "default_protein_boundary_mode")]
    #[schemars(default = "default_protein_boundary_mode")]
    pub mode: String,
    #[serde(default = "default_protein_boundary_geometry")]
    #[schemars(default = "default_protein_boundary_geometry")]
    pub geometry: String,
    #[serde(default = "default_protein_boundary_radius_strategy")]
    #[schemars(default = "default_protein_boundary_radius_strategy")]
    pub radius_strategy: String,
    #[serde(default)]
    pub radius_quantile: Option<f32>,
    #[serde(default)]
    pub protein: Option<String>,
    #[serde(default)]
    pub center_angstrom: Option<[f32; 2]>,
    #[serde(default)]
    pub radius_angstrom: Option<f32>,
    #[serde(default)]
    pub alpha_radius_angstrom: Option<f32>,
    #[serde(default)]
    pub buffer_angstrom: f32,
    #[serde(default = "default_protein_boundary_bead_exclusion_radius")]
    #[schemars(default = "default_protein_boundary_bead_exclusion_radius")]
    pub bead_exclusion_radius_angstrom: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LeafletRequest {
    pub name: String,
    pub side: String,
    #[serde(default)]
    pub apl_angstrom2: Option<f32>,
    #[serde(default)]
    pub exclusions: Vec<ExclusionZone>,
    #[serde(default)]
    pub regions: Vec<LeafletRegion>,
    pub composition: Vec<LipidComponent>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ExclusionZone {
    pub name: Option<String>,
    pub center_angstrom: [f32; 2],
    pub radius_angstrom: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LeafletRegion {
    pub name: Option<String>,
    pub role: String,
    pub geometry: RegionGeometry,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "shape", deny_unknown_fields)]
pub enum RegionGeometry {
    #[serde(rename = "circle")]
    Circle {
        center_angstrom: [f32; 2],
        radius_angstrom: f32,
    },
    #[serde(rename = "ellipse")]
    Ellipse {
        center_angstrom: [f32; 2],
        radius_angstrom: [f32; 2],
        #[serde(default)]
        rotate_degrees: f32,
    },
    #[serde(rename = "rectangle")]
    Rectangle {
        center_angstrom: [f32; 2],
        size_angstrom: [f32; 2],
        #[serde(default)]
        rotate_degrees: f32,
    },
    #[serde(rename = "polygon")]
    Polygon {
        points_angstrom: Vec<[f32; 2]>,
        #[serde(default)]
        scale_xy: Option<[f32; 2]>,
        #[serde(default)]
        rotate_degrees: f32,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LipidComponent {
    pub lipid: String,
    #[serde(default)]
    pub count: Option<usize>,
    #[serde(default)]
    pub fraction: Option<f32>,
    #[serde(default)]
    pub charge_e: Option<f32>,
    #[serde(default)]
    pub radius_angstrom: Option<f32>,
    #[serde(default)]
    pub beads: Vec<BuildBeadTemplate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BuildBeadTemplate {
    pub name: String,
    #[serde(default)]
    pub offset_angstrom: [f32; 3],
    #[serde(default)]
    pub charge_e: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InsertedComponent {
    pub name: String,
    #[serde(default = "default_component_count")]
    #[schemars(default = "default_component_count")]
    pub count: usize,
    #[serde(default)]
    pub net_charge_e: Option<f32>,
    #[serde(default)]
    pub charge_source: Option<String>,
    #[serde(default)]
    pub charge_topology: Option<String>,
    #[serde(default)]
    pub charge_topologies: Vec<String>,
    #[serde(default)]
    pub charge_molecule_type: Option<String>,
    #[serde(default)]
    pub molecule_types: Vec<String>,
    #[serde(default)]
    pub coordinates: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub definition: Option<String>,
    #[serde(default)]
    pub beads: Vec<BuildBeadTemplate>,
    #[serde(default)]
    pub placement: InsertedPlacement,
    #[serde(default)]
    pub footprint: Option<ComponentFootprint>,
}
