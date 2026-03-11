use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const TOPOLOGY_GRAPH_VERSION: &str = "polymer-build.topology-graph.v5";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct TerminiRequest {
    pub head: String,
    pub tail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Pair {
    pub a: usize,
    pub b: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Angle {
    pub a: usize,
    pub b: usize,
    pub c: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Torsion {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub d: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Exclusion {
    pub atom_index: usize,
    pub excluded_atoms: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Atom {
    pub index: usize,
    pub name: String,
    pub element: String,
    pub resid: i32,
    pub resname: String,
    pub charge_e: f32,
    pub mass: f32,
    pub atom_type_index: i32,
    pub amber_atom_type: String,
    pub lj_class: String,
    pub position: [f32; 3],
    pub neighbors: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ResiduePort {
    pub name: String,
    pub attach_atom: Option<String>,
    pub leaving_atoms: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Residue {
    pub resid: usize,
    pub resname: String,
    #[serde(default)]
    pub node_id: Option<String>,
    #[serde(default)]
    pub request_node_id: Option<String>,
    #[serde(default)]
    pub sequence_token: Option<String>,
    #[serde(default)]
    pub token_kind: Option<String>,
    #[serde(default)]
    pub source_token: Option<String>,
    #[serde(default)]
    pub motif_instance_id: Option<String>,
    #[serde(default)]
    pub motif_token: Option<String>,
    #[serde(default)]
    pub branch_depth: Option<usize>,
    #[serde(default)]
    pub branch_path: Option<String>,
    pub atom_indices: Vec<usize>,
    #[serde(default)]
    pub ports: Vec<ResiduePort>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BuildPlan {
    pub target_mode: String,
    pub realization_mode: String,
    pub resolved_sequence: Vec<String>,
    #[serde(default)]
    pub request_root_node_id: Option<String>,
    #[serde(default)]
    pub expanded_root_node_id: Option<String>,
    #[serde(default)]
    pub root_token: Option<String>,
    #[serde(default)]
    pub arm_count: Option<usize>,
    #[serde(default)]
    pub max_branch_depth: Option<usize>,
    #[serde(default)]
    pub graph_has_cycle: Option<bool>,
    pub requested_termini: TerminiRequest,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct NonbondedTyping {
    pub atom_type_indices: Vec<i32>,
    pub amber_atom_types: Vec<String>,
    pub lj_classes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct BranchPoint {
    pub atom_index: usize,
    pub degree: usize,
    #[serde(default)]
    pub resid: Option<i32>,
    #[serde(default)]
    pub atom_name: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct InterResidueBond {
    pub a: usize,
    pub b: usize,
    pub resid_a: i32,
    pub resid_b: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ConnectionDefinition {
    #[serde(default)]
    pub edge_id: Option<String>,
    pub parent_resid: usize,
    pub child_resid: usize,
    pub parent_port: String,
    pub child_port: String,
    pub parent_junction: String,
    pub child_junction: String,
    pub parent_attach_atom: String,
    pub child_attach_atom: String,
    pub parent_leaving_atoms: Vec<String>,
    pub child_leaving_atoms: Vec<String>,
    pub bond_order: u8,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MotifPortBinding {
    pub port_name: String,
    pub node_id: String,
    pub junction: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CapBinding {
    pub token: String,
    #[serde(default)]
    pub port: Option<String>,
    #[serde(default)]
    pub junction: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MotifInstance {
    pub motif_instance_id: String,
    pub motif_token: String,
    pub request_node_id: String,
    pub expanded_node_ids: Vec<String>,
    pub expanded_resids: Vec<usize>,
    pub exposed_ports: Vec<MotifPortBinding>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CycleRecord {
    pub cycle_id: String,
    pub node_ids: Vec<String>,
    pub residue_ids: Vec<usize>,
    pub edge_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct OpenPort {
    pub node_id: String,
    #[serde(default)]
    pub request_node_id: Option<String>,
    pub resid: usize,
    pub port_name: String,
    pub junction: String,
    #[serde(default)]
    pub port_class: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct PortPolicy {
    pub node_id: String,
    #[serde(default)]
    pub request_node_id: Option<String>,
    pub resid: usize,
    pub port_name: String,
    pub junction: String,
    #[serde(default)]
    pub port_class: Option<String>,
    #[serde(default)]
    pub default_cap: Option<CapBinding>,
    #[serde(default)]
    pub allowed_caps: Vec<CapBinding>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AppliedCap {
    pub node_id: String,
    #[serde(default)]
    pub request_node_id: Option<String>,
    pub resid: usize,
    pub port_name: String,
    pub junction: String,
    pub cap: CapBinding,
    pub application_source: String,
    pub cap_node_id: String,
    pub cap_resid: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ConformerEdge {
    pub edge_id: String,
    pub layout_mode: String,
    pub torsion_mode: String,
    #[serde(default)]
    pub torsion_deg: Option<f32>,
    #[serde(default)]
    pub torsion_window_deg: Option<[f32; 2]>,
    #[serde(default)]
    pub ring_mode: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct AlignmentPath {
    pub kind: String,
    pub residue_ids: Vec<usize>,
    pub node_ids: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct RelaxMetadata {
    pub mode: String,
    pub steps_requested: usize,
    pub steps_executed: usize,
    pub initial_max_clash: f32,
    pub final_max_clash: f32,
    pub rms_displacement: f32,
    #[serde(default)]
    pub raw_coordinates: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct TopologyGraph {
    pub version: String,
    pub request_id: String,
    pub bundle_id: String,
    pub build_plan: BuildPlan,
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Pair>,
    pub angles: Vec<Angle>,
    pub dihedrals: Vec<Torsion>,
    pub impropers: Vec<Torsion>,
    pub exclusions: Vec<Exclusion>,
    pub branch_points: Vec<BranchPoint>,
    pub residue_connections: Vec<Pair>,
    pub inter_residue_bonds: Vec<InterResidueBond>,
    pub connection_definitions: Vec<ConnectionDefinition>,
    pub nonbonded_typing: NonbondedTyping,
    pub residues: Vec<Residue>,
    pub sequence: Vec<String>,
    pub template_sequence_resnames: Vec<String>,
    pub applied_residue_resnames: Vec<String>,
    #[serde(default)]
    pub motif_instances: Vec<MotifInstance>,
    #[serde(default)]
    pub cycle_basis: Vec<CycleRecord>,
    #[serde(default)]
    pub open_ports: Vec<OpenPort>,
    #[serde(default)]
    pub port_policies: Vec<PortPolicy>,
    #[serde(default)]
    pub applied_caps: Vec<AppliedCap>,
    #[serde(default)]
    pub conformer_edges: Vec<ConformerEdge>,
    #[serde(default)]
    pub alignment_paths: Vec<AlignmentPath>,
    #[serde(default)]
    pub relax_metadata: Option<RelaxMetadata>,
    #[serde(default)]
    pub metadata: Value,
}
