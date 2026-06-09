use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Result};
use schemars::{schema_for, JsonSchema};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::mapping::{map_molecule, map_molecule_with_options, MappingOptions, MappingResult};
use crate::molecule::Molecule;
use crate::optimize::{optimize_bonded_terms, OptimizationConfig, OptimizationReport};
use crate::parameters::{AngleStats, BondStats, BondedStats, DihedralStats};
use crate::trajectory::{map_native_trajectory, BeadMapping, NativeTrajectoryOptions};
use crate::xtb::{run_xtb_pipeline_with_config, XtbRunConfig};
use warp_structure::io::{read_molecule, read_prmtop_topology, read_system_auto};
use warp_structure::AtomRecord;

pub const AGENT_SCHEMA_VERSION: &str = "warp-cg.agent.v2";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CgRequest {
    #[serde(default = "default_schema_version")]
    #[schemars(default = "default_schema_version")]
    pub schema_version: String,
    pub name: String,
    #[serde(default)]
    pub smiles: Option<String>,
    #[serde(default)]
    pub repeat_smiles: Option<String>,
    #[serde(default)]
    pub source: Option<CgSource>,
    #[serde(default)]
    pub mapping: Option<CgMappingRequest>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub trajectory_source: Option<TrajectorySource>,
    #[serde(default)]
    pub reference_source: Option<ReferenceSource>,
    #[serde(default)]
    pub optimization: Option<ParameterTuningRequest>,
    #[serde(default = "default_output")]
    #[schemars(default = "default_output")]
    pub output: CgOutputRequest,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CgSource {
    pub kind: String,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub coordinates: Option<String>,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub charge_manifest: Option<String>,
    #[serde(default)]
    pub trajectory: Option<String>,
    #[serde(default)]
    pub target_selection: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub topology_format: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CgMappingRequest {
    #[serde(default = "default_mapping_mode")]
    #[schemars(default = "default_mapping_mode")]
    pub mode: String,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default)]
    pub target_bead_size: Option<usize>,
    #[serde(default)]
    pub preserve_functional_groups: Option<bool>,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub repeat_unit_hint: Option<String>,
    #[serde(default)]
    pub terminal_aware: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TrajectorySource {
    /// External trajectory path accepted by warp-md loaders, e.g. xtc, dcd, trr, pdb, xyz.
    pub path: String,
    #[serde(default)]
    pub topology: Option<String>,
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub topology_format: Option<String>,
    #[serde(default = "default_external_trajectory_kind")]
    #[schemars(default = "default_external_trajectory_kind")]
    pub kind: String,
    #[serde(default)]
    pub stride: Option<usize>,
    #[serde(default)]
    pub start: Option<usize>,
    #[serde(default)]
    pub stop: Option<usize>,
    #[serde(default)]
    pub length_scale: Option<f32>,
    #[serde(default)]
    pub target_selection: Option<String>,
    #[serde(default)]
    pub environment_selection: Option<String>,
    #[serde(default)]
    pub atom_indices: Option<Vec<usize>>,
    #[serde(default)]
    pub mass_weighted: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSource {
    #[serde(default = "default_reference_kind")]
    #[schemars(default = "default_reference_kind")]
    pub kind: String,
    #[serde(default)]
    pub xtb: Option<XtbRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct XtbRequest {
    #[serde(default)]
    pub mode: Option<String>,
    #[serde(default)]
    pub temperature_k: Option<f64>,
    #[serde(default)]
    pub time_ps: Option<f64>,
    #[serde(default)]
    pub timestep_fs: Option<f64>,
    #[serde(default)]
    pub dump_fs: Option<f64>,
    #[serde(default)]
    pub gfn: Option<String>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub work_dir: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ParameterTuningRequest {
    #[serde(default = "default_tuning_enabled")]
    #[schemars(default = "default_tuning_enabled")]
    pub enabled: bool,
    #[serde(default = "default_tuning_source")]
    #[schemars(default = "default_tuning_source")]
    pub source: String,
    #[serde(default = "default_tuning_method")]
    #[schemars(default = "default_tuning_method")]
    pub method: String,
    #[serde(default)]
    pub max_evaluations: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub swarm_size: Option<usize>,
    #[serde(default = "default_tuning_objective")]
    #[schemars(default = "default_tuning_objective")]
    pub objective: String,
    #[serde(default)]
    pub target_terms: Option<Vec<String>>,
    #[serde(default)]
    pub xtb: Option<XtbRequest>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CgOutputRequest {
    #[serde(default = "default_out_dir")]
    #[schemars(default = "default_out_dir")]
    pub out_dir: String,
    #[serde(default)]
    pub mapped_trajectory: Option<String>,
    #[serde(default = "default_write_mapping")]
    #[schemars(default = "default_write_mapping")]
    pub write_mapping_json: bool,
    #[serde(default = "default_write_topology_itp")]
    #[schemars(default = "default_write_topology_itp")]
    pub write_topology_itp: bool,
    #[serde(default = "default_write_topology_top")]
    #[schemars(default = "default_write_topology_top")]
    pub write_topology_top: bool,
    #[serde(default = "default_write_cg_pdb")]
    #[schemars(default = "default_write_cg_pdb")]
    pub write_cg_pdb: bool,
    #[serde(default)]
    pub cg_pdb: Option<String>,
    #[serde(default = "default_write_bonded_parameter_map")]
    #[schemars(default = "default_write_bonded_parameter_map")]
    pub write_bonded_parameter_map: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgBead {
    pub index: usize,
    pub name: String,
    pub atom_indices: Vec<usize>,
    pub features: Vec<String>,
    pub formal_charge: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgArtifact {
    pub path: String,
    pub kind: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgResult {
    pub schema_version: String,
    pub status: String,
    pub exit_code: i32,
    pub name: String,
    pub summary: CgSummary,
    pub bead_count: usize,
    pub beads: Vec<CgBead>,
    pub connections: Vec<[usize; 2]>,
    pub artifacts: Vec<CgArtifact>,
    pub artifact_paths: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization: Option<ParameterTuningResult>,
    pub elapsed_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgSummary {
    pub input_mode: String,
    pub mapping_mode: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aa_atom_count: Option<usize>,
    pub cg_bead_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mapped_residue_count: Option<usize>,
    pub optimized_terms: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization_source: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ParameterTuningResult {
    pub status: String,
    pub method: String,
    pub source: String,
    pub objective: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<OptimizationReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct CgEvent {
    pub schema_version: String,
    pub event: String,
    pub message: String,
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Clone, Debug)]
struct SourceHandoff {
    coordinates: String,
    topology: Option<String>,
    trajectory: Option<String>,
    coordinate_format: Option<String>,
    topology_format: Option<String>,
}

#[derive(Clone, Debug)]
struct SourceResidue {
    resid: i32,
    resname: String,
    chain: char,
    atom_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
struct SourceBeadRecord {
    index: usize,
    name: String,
    bead_type: String,
    features: Vec<String>,
    formal_charge: i32,
    resid: i32,
    resname: String,
    chain: char,
    atom_indices: Vec<usize>,
    atom_names: Vec<String>,
    coord: [f32; 3],
}

struct SourceMappingResult {
    mapping: MappingResult,
    beads: Vec<SourceBeadRecord>,
    residue_count: usize,
    aa_atom_count: usize,
    templates: Value,
    provenance: Value,
}

fn default_schema_version() -> String {
    AGENT_SCHEMA_VERSION.to_string()
}

fn default_out_dir() -> String {
    ".".to_string()
}

fn default_write_mapping() -> bool {
    true
}

fn default_write_topology_itp() -> bool {
    true
}

fn default_write_topology_top() -> bool {
    false
}

fn default_write_cg_pdb() -> bool {
    true
}

fn default_write_bonded_parameter_map() -> bool {
    true
}

fn default_external_trajectory_kind() -> String {
    "external".to_string()
}

fn default_reference_kind() -> String {
    "external".to_string()
}

fn default_mapping_mode() -> String {
    "auto".to_string()
}

fn default_tuning_enabled() -> bool {
    false
}

fn default_tuning_source() -> String {
    "external_trajectory".to_string()
}

fn default_tuning_method() -> String {
    "bayesian_optimization".to_string()
}

fn default_tuning_objective() -> String {
    "bonded_parameter_parity".to_string()
}

fn default_output() -> CgOutputRequest {
    CgOutputRequest {
        out_dir: default_out_dir(),
        mapped_trajectory: None,
        write_mapping_json: true,
        write_topology_itp: true,
        write_topology_top: true,
        write_cg_pdb: true,
        cg_pdb: None,
        write_bonded_parameter_map: true,
    }
}

pub fn schema_json(kind: &str) -> Result<String> {
    let value = match kind {
        "request" => serde_json::to_value(schema_for!(CgRequest))?,
        "result" => serde_json::to_value(schema_for!(CgResult))?,
        "event" => serde_json::to_value(schema_for!(CgEvent))?,
        other => return Err(anyhow!("unknown warp-cg schema kind: {other}")),
    };
    Ok(serde_json::to_string_pretty(&value)?)
}

pub fn example_request() -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "trajectory_source": {
            "path": "traj.xtc",
            "topology": "topology.pdb",
            "kind": "external",
            "target_selection": "resname BENZ",
            "stride": 1
        },
        "output": {
            "out_dir": "results/cg/benzene",
            "mapped_trajectory": "benzene_cg.xtc",
            "write_mapping_json": true,
            "write_topology_itp": true,
            "write_topology_top": true,
            "write_cg_pdb": true,
            "write_bonded_parameter_map": true
        },
        "optimization": {
            "enabled": false,
            "source": "external_trajectory",
            "method": "bayesian_optimization",
            "objective": "bonded_parameter_parity"
        }
    })
}

pub fn example_requests() -> Value {
    json!({
        "smiles_small_molecule": example_request(),
        "paa_repeat_model": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_repeat",
            "repeat_smiles": "C(C(=O)O)",
            "mapping": {"mode": "auto", "repeat_unit_hint": "PAA"},
            "output": {"out_dir": "cg/paa_repeat"}
        },
        "pes_repeat_model": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "pes_repeat",
            "repeat_smiles": "O=S(=O)(c1ccc(Oc2ccc(*)cc2)cc1)",
            "mapping": {"mode": "auto", "repeat_unit_hint": "PES"},
            "output": {"out_dir": "cg/pes_repeat"}
        },
        "aa_trajectory_tuning": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene_traj_tuned",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "kind": "external",
                "path": "benzene.xtc",
                "topology": "benzene.pdb",
                "target_selection": "resname BENZ"
            },
            "optimization": {
                "enabled": true,
                "source": "aa_trajectory",
                "method": "bayesian_optimization",
                "target_terms": ["bonds", "angles", "dihedrals"],
                "max_evaluations": 64
            },
            "output": {"out_dir": "cg/benzene_traj_tuned"}
        },
        "xtb_tuning": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "ethanol_xtb",
            "smiles": "CCO",
            "reference_source": {
                "kind": "xtb",
                "xtb": {
                    "mode": "optimize_and_md",
                    "gfn": "gfn2",
                    "temperature_k": 300.0,
                    "time_ps": 10.0
                }
            },
            "optimization": {
                "enabled": true,
                "source": "xtb",
                "method": "bayesian_optimization",
                "target_terms": ["bonds", "angles"],
                "max_evaluations": 64
            },
            "output": {"out_dir": "cg/ethanol_xtb"}
        },
        "polymer_build_manifest_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_50mer",
            "source": {
                "kind": "polymer_build_manifest",
                "path": "paa_50mer.build.json"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_50mer"}
        },
        "polymer_pack_manifest_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_50mer_box",
            "source": {
                "kind": "polymer_pack_manifest",
                "path": "polymer_pack_manifest.json"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_50mer_box"}
        },
        "source_manifest_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_source_manifest",
            "source": {
                "kind": "source_manifest",
                "path": "source_manifest.json"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_source_manifest"}
        },
        "coordinates_topology_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_coordinates_topology",
            "source": {
                "kind": "coordinates_topology",
                "coordinates": "paa_50mer.pdb",
                "topology": "paa_50mer.pdb",
                "format": "pdb",
                "topology_format": "pdb"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_coordinates_topology"}
        },
        "coordinates_topology_charge_manifest_to_cg": {
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_coordinates_topology_charge",
            "source": {
                "kind": "coordinates_topology_charge_manifest",
                "coordinates": "paa_50mer.pdb",
                "topology": "paa_50mer.pdb",
                "charge_manifest": "paa_50mer_charge_manifest.json",
                "format": "pdb",
                "topology_format": "pdb"
            },
            "mapping": {
                "mode": "auto",
                "strategy": "polymer_residue_graph",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": "cg/paa_coordinates_topology_charge"}
        }
    })
}

pub fn capabilities() -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "tool": "warp-cg",
        "contracts": {
            "request": AGENT_SCHEMA_VERSION,
            "result": AGENT_SCHEMA_VERSION,
            "event": AGENT_SCHEMA_VERSION
        },
        "inputs": {
            "identity_fields": {
                "requires_one_of": ["smiles", "repeat_smiles", "source"],
                "smiles": "small molecule template input",
                "repeat_smiles": "polymer repeat-unit template input",
                "source": "built-system or APS handoff input",
                "mapping.template": "known or generated warp-cg.mapping_template.v1 path for mapping.mode=template"
            },
            "accepted_source_kinds": {
                "polymer_build_manifest": {"required": ["source.path"], "optional": ["source.target_selection"], "example": "examples/warp_cg/polymer_build_manifest_to_cg_request.json"},
                "polymer_pack_manifest": {"required": ["source.path"], "optional": ["source.target_selection"], "example": "examples/warp_cg/polymer_pack_manifest_to_cg_request.json"},
                "coordinates_topology": {"required": ["source.coordinates", "source.topology"], "optional": ["source.target_selection"], "example": "examples/warp_cg/coordinates_topology_to_cg_request.json"},
                "coordinates_topology_charge_manifest": {"required": ["source.coordinates", "source.topology", "source.charge_manifest"], "optional": ["source.target_selection"], "example": "examples/warp_cg/coordinates_topology_charge_manifest_to_cg_request.json"},
                "source_manifest": {"required": ["source.path"], "optional": ["source.target_selection"], "example": "examples/warp_cg/source_manifest_to_cg_request.json"}
            },
            "source_selection_semantics": {
                "default": "when source.target_selection is omitted, source-driven polymer mapping uses every atom and residue in the resolved source coordinates",
                "target_selection": "must be a valid warp-md topology selection expression, for example 'resname PAA' or 'chain A'; the literal string 'polymer' is not a selector",
                "execution_scope": "source-driven auto/template mapping currently builds residue-aware CG topology from the selected/default coordinate scope; manifest examples omit target_selection to select the full polymer handoff",
                "provenance": "aa_to_cg_mapping_provenance records selection policy, selected atom/residue counts, terminal roles, residue names/counts, repeat hint, and atom-to-bead links"
            },
            "trajectory_mapping": {
                "preferred_field": "trajectory_source",
                "formats_tested": ["dcd", "xtc", "gro", "g96", "cpt", "h5md", "tng", "trr", "pdb", "pdbqt"],
                "target_description": "topology plus warp-md selection expressions or explicit atom indices",
                "environment_selection": "accepted for future solvent/environment metadata; mapping currently uses target_selection or atom_indices"
            },
            "topology": "used to validate solvated external trajectories and resolve target_selection",
            "xtb": {
                "reference_source": "reference_source.kind=xtb can initiate xTB optimization/MD",
                "executable_detection": "PATH lookup for xtb",
                "env": ["PATH", "OMP_NUM_THREADS"],
                "gfn": ["gfnff", "gfn0", "gfn1", "gfn2"]
            }
        },
        "optimization": {
            "status": "implemented",
            "methods": ["bayesian_optimization", "pso"],
            "sources": {
                "external_trajectory": {
                    "required": ["smiles or repeat_smiles", "trajectory_source.path", "trajectory_source.topology"],
                    "selection": "trajectory_source.target_selection or trajectory_source.atom_indices selects the mapped solute; omit only for single-molecule trajectories",
                    "units": {"length_scale": "multiply input coordinates by this value before writing CG output; use 10.0 for nm-to-Angstrom input when needed"},
                    "artifacts": ["coarse_grained_trajectory", "bond_stats_json", "bonded_stats_json", "bonded_optimization_report"],
                    "example": "examples/warp_cg/solvated_external_bo_request.json"
                },
                "aa_trajectory": {
                    "required": ["smiles or repeat_smiles", "trajectory_source.path", "trajectory_source.topology", "optimization.enabled=true", "optimization.source=aa_trajectory"],
                    "selection": "same as external_trajectory",
                    "artifacts": ["coarse_grained_trajectory", "bonded_parameter_map_json", "bonded_optimization_report"],
                    "example": "examples/warp_cg/aa_trajectory_tuning_request.json"
                },
                "xtb": {
                    "required": ["smiles or repeat_smiles", "reference_source.kind=xtb", "optimization.enabled=true", "optimization.source=xtb"],
                    "units": {"temperature_k": "K", "time_ps": "ps", "timestep_fs": "fs", "dump_fs": "fs"},
                    "artifacts": ["xtb_optimized_xyz", "xtb_reference_trajectory", "bonded_optimization_report"],
                    "example": "examples/warp_cg/xtb_tuning_request.json"
                }
            },
            "objective": "bonded_parameter_parity",
            "defaults": {"method": "bayesian_optimization", "max_evaluations": 32, "seed": 42},
            "terms": ["bonds", "angles", "dihedrals"],
            "empty_stats_behavior": "returns skipped report when no bonded reference statistics are available"
        },
        "polymer_mapping": {
            "status": "implemented_auto_and_template_modes",
            "execution_status": "built-system source handoffs execute in auto mode and emit reusable mapping templates; template mode replays generated/curated residue-role atom-name mappings",
            "modes": {
                "auto": "shared graph mapper fed by either SMILES or source coordinates/topology; source mode adds residue roles, templates, and provenance",
                "template": "replay a warp-cg.mapping_template.v1 file against source residue atom names"
            },
            "supported_handoff_schemas": ["warp-build.manifest.v1", "warp-pack manifest", "coordinates_topology"],
            "terminal_aware_polymers": true,
            "multi_residue_templates": true,
            "chemistry_generalization_limits": {
                "tested_auto_classes": ["PAA-like vinyl polymers with carboxylic acid/carboxylate groups", "PES-like aromatic sulfone/ether repeat units", "small organic SMILES examples"],
                "auto_mapping_strengths": ["connected residue graph partitioning", "functional-group preservation for carboxylate/carboxylic acid, amide-like, sulfone/sulfonate-like, and phosphate-like groups", "terminal-aware head/middle/tail residue roles"],
                "requires_template_or_manual_review": ["ambiguous residue atom names", "crosslinked or network polymers", "branched polymers without clear residue order", "metals/coordination chemistry", "charged groups outside implemented functional-group detectors", "multi-component residues where one residue is not one repeat unit"],
                "template_required_when": "auto mapping does not preserve intended chemistry, source residue atom names must match a curated mapping, or reproducibility across related polymers is required"
            },
            "implemented_outputs": [
                "mapping_template_json",
                "repeat_unit_bead_template",
                "head_middle_tail_bead_templates",
                "residue_to_bead_map",
                "bead_connectivity_across_residues",
                "cg_topology_for_chain",
                "aa_to_cg_mapping_provenance"
            ]
        },
        "validation_checks": [
            "source files exist",
            "coordinate atom count can be read",
            "topology atom count can be read when supported",
            "coordinate/topology atom counts match",
            "target_selection is evaluated when a topology is available",
            "template-mode replay preconditions are reported",
            "bonded stats path preconditions are reported",
            "xtb executable availability is reported when requested",
            "optimization runtime/cost estimate is reported"
        ],
        "examples": example_requests(),
        "outputs": [
            "martini_bead_mapping_json",
            "mapping_template_json",
            "coarse_grained_pdb",
            "optional_coarse_grained_trajectory_xtc_dcd_gro_g96_cpt_trr_h5md",
            "bond_stats_json",
            "bonded_stats_json",
            "bonded_parameter_map_json",
            "bonded_optimization_report",
            "aa_to_cg_mapping_provenance",
            "martini_topology_itp",
            "martini_topology_top"
        ],
        "force_field": "Martini 3 coarse-grained bead assignment"
    })
}

pub fn validate_request_json(text: &str) -> (i32, Value) {
    match parse_request(text).and_then(validate_request) {
        Ok(request) => {
            let validation = validation_report(&request);
            let valid = validation["errors"]
                .as_array()
                .map_or(true, |errors| errors.is_empty());
            let code = if valid { 0 } else { 2 };
            (
                code,
                json!({
                    "schema_version": AGENT_SCHEMA_VERSION,
                    "valid": valid,
                    "status": if valid { "ok" } else { "error" },
                    "name": request.name,
                    "summary": validation["summary"].clone(),
                    "checks": validation["checks"].clone(),
                    "warnings": validation["warnings"].clone(),
                    "errors": validation["errors"].clone(),
                }),
            )
        }
        Err(err) => (
            2,
            json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "valid": false,
                "status": "error",
                "error": {
                    "code": "warp_cg.invalid_request",
                    "message": err.to_string()
                }
            }),
        ),
    }
}

pub fn run_request_json(text: &str, stream_ndjson: bool) -> (i32, Value) {
    let started = Instant::now();
    let request = match parse_request(text).and_then(validate_request) {
        Ok(request) => request,
        Err(err) => {
            return (
                2,
                json!({
                    "schema_version": AGENT_SCHEMA_VERSION,
                    "status": "error",
                    "exit_code": 2,
                    "error": {
                        "code": "warp_cg.invalid_request",
                        "message": err.to_string()
                    }
                }),
            );
        }
    };

    emit_event(
        stream_ndjson,
        "operation_started",
        "mapping molecule",
        Some(&request.name),
    );
    match run_request(&request, started) {
        Ok(result) => {
            emit_event(
                stream_ndjson,
                "operation_complete",
                "mapping complete",
                Some(&request.name),
            );
            (
                0,
                serde_json::to_value(result).expect("serialize warp-cg result"),
            )
        }
        Err(err) => {
            emit_event(
                stream_ndjson,
                "error",
                &err.to_string(),
                Some(&request.name),
            );
            (
                1,
                json!({
                    "schema_version": AGENT_SCHEMA_VERSION,
                    "status": "error",
                    "exit_code": 1,
                    "name": request.name,
                    "error": {
                        "code": "warp_cg.run_failed",
                        "message": err.to_string()
                    },
                    "elapsed_ms": started.elapsed().as_millis()
                }),
            )
        }
    }
}

fn parse_request(text: &str) -> Result<CgRequest> {
    serde_json::from_str(text).map_err(|err| anyhow!("invalid JSON request: {err}"))
}

fn validate_request(request: CgRequest) -> Result<CgRequest> {
    if request.schema_version != AGENT_SCHEMA_VERSION {
        return Err(anyhow!(
            "schema_version must be {AGENT_SCHEMA_VERSION}, got {}",
            request.schema_version
        ));
    }
    if request.name.trim().is_empty() {
        return Err(anyhow!("name is required"));
    }
    if request
        .smiles
        .as_ref()
        .is_some_and(|smiles| smiles.trim().is_empty())
    {
        return Err(anyhow!("smiles must not be empty"));
    }
    if request
        .repeat_smiles
        .as_ref()
        .is_some_and(|smiles| smiles.trim().is_empty())
    {
        return Err(anyhow!("repeat_smiles must not be empty"));
    }
    let identity_count = [
        request.smiles.as_ref().map(|_| "smiles"),
        request.repeat_smiles.as_ref().map(|_| "repeat_smiles"),
        request.source.as_ref().map(|_| "source"),
    ]
    .into_iter()
    .flatten()
    .count();
    if identity_count == 0 {
        return Err(anyhow!(
            "request requires one of smiles, repeat_smiles, or source"
        ));
    }
    if let Some(source) = &request.source {
        validate_source_shape(source)?;
    }
    if let Some(mapping) = &request.mapping {
        if mapping.mode.trim().is_empty() {
            return Err(anyhow!("mapping.mode must not be empty"));
        }
        if !matches!(mapping.mode.as_str(), "auto" | "template") {
            return Err(anyhow!("mapping.mode must be auto or template"));
        }
        if mapping.mode == "template" && mapping.template.is_none() && request.source.is_some() {
            return Err(anyhow!("mapping.mode=template requires mapping.template"));
        }
        if mapping
            .template
            .as_ref()
            .is_some_and(|template| template.trim().is_empty())
        {
            return Err(anyhow!("mapping.template must not be empty"));
        }
        if mapping.target_bead_size.is_some_and(|size| size == 0) {
            return Err(anyhow!(
                "mapping.target_bead_size must be greater than zero"
            ));
        }
        if mapping
            .repeat_unit_hint
            .as_ref()
            .is_some_and(|hint| hint.trim().is_empty())
        {
            return Err(anyhow!("mapping.repeat_unit_hint must not be empty"));
        }
    }
    if request
        .topology
        .as_ref()
        .is_some_and(|topology| topology.trim().is_empty())
    {
        return Err(anyhow!("topology must not be empty"));
    }
    if request.output.out_dir.trim().is_empty() {
        return Err(anyhow!("output.out_dir must not be empty"));
    }
    if request
        .output
        .mapped_trajectory
        .as_ref()
        .is_some_and(|path| path.trim().is_empty())
    {
        return Err(anyhow!("output.mapped_trajectory must not be empty"));
    }
    if request
        .output
        .cg_pdb
        .as_ref()
        .is_some_and(|path| path.trim().is_empty())
    {
        return Err(anyhow!("output.cg_pdb must not be empty"));
    }
    if request.output.write_topology_top && !request.output.write_topology_itp {
        return Err(anyhow!(
            "output.write_topology_top requires output.write_topology_itp because the .top includes the generated .itp"
        ));
    }
    if request.output.mapped_trajectory.is_some() {
        let has_xtb_reference = request
            .reference_source
            .as_ref()
            .is_some_and(|source| source.kind == "xtb");
        let has_source_trajectory = request
            .source
            .as_ref()
            .and_then(|source| source.trajectory.as_ref())
            .is_some();
        if request.trajectory_source.is_none() && !has_source_trajectory && !has_xtb_reference {
            return Err(anyhow!(
                "output.mapped_trajectory requires trajectory_source, source.trajectory, or reference_source.kind=xtb"
            ));
        }
    }
    if let Some(source) = &request.trajectory_source {
        if source.path.trim().is_empty() {
            return Err(anyhow!("trajectory_source.path is required"));
        }
        if source
            .topology
            .as_ref()
            .is_some_and(|topology| topology.trim().is_empty())
        {
            return Err(anyhow!("trajectory_source.topology must not be empty"));
        }
        if source.kind != "external" {
            return Err(anyhow!("trajectory_source.kind must be external"));
        }
        if source.stride == Some(0) {
            return Err(anyhow!(
                "trajectory_source.stride must be greater than zero"
            ));
        }
        if source.stop.is_some() && source.start.is_some() && source.stop <= source.start {
            return Err(anyhow!("trajectory_source.stop must be greater than start"));
        }
        if source
            .atom_indices
            .as_ref()
            .is_some_and(|indices| indices.is_empty())
        {
            return Err(anyhow!("trajectory_source.atom_indices must not be empty"));
        }
        if source.target_selection.is_some() && source.atom_indices.is_some() {
            return Err(anyhow!(
                "use either trajectory_source.target_selection or atom_indices, not both"
            ));
        }
        if source
            .target_selection
            .as_ref()
            .is_some_and(|selection| selection.trim().is_empty())
        {
            return Err(anyhow!(
                "trajectory_source.target_selection must not be empty"
            ));
        }
        if source
            .environment_selection
            .as_ref()
            .is_some_and(|selection| selection.trim().is_empty())
        {
            return Err(anyhow!(
                "trajectory_source.environment_selection must not be empty"
            ));
        }
        if source.mass_weighted == Some(true)
            && source.topology.is_none()
            && request.topology.is_none()
        {
            return Err(anyhow!(
                "trajectory_source.mass_weighted requires trajectory_source.topology or top-level topology"
            ));
        }
    }
    if let Some(reference) = &request.reference_source {
        if reference.kind != "external" && reference.kind != "xtb" {
            return Err(anyhow!("reference_source.kind must be external or xtb"));
        }
        if let Some(xtb) = &reference.xtb {
            validate_xtb_request(xtb, "reference_source.xtb")?;
        }
    }
    if let Some(tuning) = &request.optimization {
        validate_tuning_request(tuning, "optimization", &request)?;
    }
    Ok(request)
}

fn validate_source_shape(source: &CgSource) -> Result<()> {
    if source.kind.trim().is_empty() {
        return Err(anyhow!("source.kind is required"));
    }
    let valid_kind = matches!(
        source.kind.as_str(),
        "polymer_build_manifest"
            | "polymer_pack_manifest"
            | "coordinates_topology"
            | "coordinates_topology_charge_manifest"
            | "source_manifest"
    );
    if !valid_kind {
        return Err(anyhow!(
            "source.kind must be polymer_build_manifest, polymer_pack_manifest, coordinates_topology, coordinates_topology_charge_manifest, or source_manifest"
        ));
    }
    for (field, value) in [
        ("source.path", source.path.as_ref()),
        ("source.coordinates", source.coordinates.as_ref()),
        ("source.topology", source.topology.as_ref()),
        ("source.charge_manifest", source.charge_manifest.as_ref()),
        ("source.trajectory", source.trajectory.as_ref()),
        ("source.target_selection", source.target_selection.as_ref()),
    ] {
        if value.is_some_and(|item| item.trim().is_empty()) {
            return Err(anyhow!("{field} must not be empty"));
        }
    }
    match source.kind.as_str() {
        "polymer_build_manifest" | "polymer_pack_manifest" | "source_manifest" => {
            if source.path.is_none() {
                return Err(anyhow!("source.path is required for {}", source.kind));
            }
        }
        "coordinates_topology" => {
            if source.coordinates.is_none() || source.topology.is_none() {
                return Err(anyhow!(
                    "source.coordinates and source.topology are required for coordinates_topology"
                ));
            }
        }
        "coordinates_topology_charge_manifest" => {
            if source.coordinates.is_none()
                || source.topology.is_none()
                || source.charge_manifest.is_none()
            {
                return Err(anyhow!(
                    "source.coordinates, source.topology, and source.charge_manifest are required for coordinates_topology_charge_manifest"
                ));
            }
        }
        _ => {}
    }
    Ok(())
}

fn validate_xtb_request(xtb: &XtbRequest, field: &str) -> Result<()> {
    if xtb
        .mode
        .as_ref()
        .is_some_and(|mode| mode != "optimize" && mode != "optimize_and_md")
    {
        return Err(anyhow!("{field}.mode must be optimize or optimize_and_md"));
    }
    validate_positive(xtb.temperature_k, &format!("{field}.temperature_k"))?;
    validate_positive(xtb.time_ps, &format!("{field}.time_ps"))?;
    validate_positive(xtb.timestep_fs, &format!("{field}.timestep_fs"))?;
    validate_positive(xtb.dump_fs, &format!("{field}.dump_fs"))?;
    if xtb.gfn.as_ref().is_some_and(|gfn| gfn.trim().is_empty()) {
        return Err(anyhow!("{field}.gfn must not be empty"));
    }
    Ok(())
}

fn validate_tuning_request(
    tuning: &ParameterTuningRequest,
    field: &str,
    request: &CgRequest,
) -> Result<()> {
    if tuning.method != "bayesian_optimization" && tuning.method != "pso" {
        return Err(anyhow!(
            "{field}.method must be bayesian_optimization or pso"
        ));
    }
    if tuning.max_evaluations == Some(0) {
        return Err(anyhow!("{field}.max_evaluations must be greater than zero"));
    }
    if tuning.swarm_size == Some(0) {
        return Err(anyhow!("{field}.swarm_size must be greater than zero"));
    }
    if tuning.source != "external_trajectory"
        && tuning.source != "aa_trajectory"
        && tuning.source != "xtb"
    {
        return Err(anyhow!(
            "{field}.source must be external_trajectory, aa_trajectory, or xtb"
        ));
    }
    if let Some(terms) = &tuning.target_terms {
        if terms.is_empty() {
            return Err(anyhow!("{field}.target_terms must not be empty"));
        }
        for term in terms {
            if term != "bonds" && term != "angles" && term != "dihedrals" {
                return Err(anyhow!(
                    "{field}.target_terms entries must be bonds, angles, or dihedrals"
                ));
            }
        }
    }
    if let Some(xtb) = &tuning.xtb {
        validate_xtb_request(xtb, &format!("{field}.xtb"))?;
    }
    if tuning.enabled
        && (tuning.source == "external_trajectory" || tuning.source == "aa_trajectory")
        && request.trajectory_source.is_none()
        && request
            .source
            .as_ref()
            .and_then(|source| source.trajectory.as_ref())
            .is_none()
    {
        return Err(anyhow!(
            "{field} trajectory tuning requires trajectory_source or source.trajectory"
        ));
    }
    if tuning.enabled
        && tuning.source == "xtb"
        && !request
            .reference_source
            .as_ref()
            .is_some_and(|source| source.kind == "xtb")
    {
        return Err(anyhow!(
            "{field} xtb parameter tuning requires reference_source.kind=xtb"
        ));
    }
    Ok(())
}

fn input_mode(request: &CgRequest) -> &str {
    if let Some(source) = &request.source {
        source.kind.as_str()
    } else if request.repeat_smiles.is_some() {
        "repeat_smiles"
    } else if request.smiles.is_some() {
        "smiles"
    } else {
        "unknown"
    }
}

fn mapping_mode(request: &CgRequest) -> &str {
    request
        .mapping
        .as_ref()
        .map(|mapping| mapping.mode.as_str())
        .unwrap_or("small_molecule_auto")
}

fn artifact_paths(artifacts: &[CgArtifact]) -> BTreeMap<String, String> {
    artifacts
        .iter()
        .map(|artifact| (artifact.kind.clone(), artifact.path.clone()))
        .collect()
}

fn active_tuning_request(request: &CgRequest) -> Option<&ParameterTuningRequest> {
    request.optimization.as_ref()
}

fn validation_report(request: &CgRequest) -> Value {
    let mut checks = Vec::new();
    let mut warnings = Vec::new();
    let mut errors = Vec::new();
    let mut aa_atom_count = None;

    if let Some(source) = &request.source {
        validate_source_files(
            source,
            &mut checks,
            &mut warnings,
            &mut errors,
            &mut aa_atom_count,
        );
    }
    validate_xtb_available(request, &mut checks, &mut warnings);
    validate_optimization_cost(request, &mut checks);

    let has_bonded_stats_source = request.trajectory_source.is_some()
        || request
            .source
            .as_ref()
            .and_then(|source| source.trajectory.as_ref())
            .is_some()
        || request
            .reference_source
            .as_ref()
            .is_some_and(|source| source.kind == "xtb");
    checks.push(json!({
        "name": "bonded_stats_preconditions",
        "status": if has_bonded_stats_source { "ok" } else { "not_available" },
        "message": "bonded stats require trajectory_source, source.trajectory, or xTB reference trajectory"
    }));

    json!({
        "summary": {
            "input_mode": input_mode(request),
            "mapping_mode": mapping_mode(request),
            "aa_atom_count": aa_atom_count,
            "optimization_source": active_tuning_request(request).filter(|tuning| tuning.enabled).map(|tuning| tuning.source.clone()),
            "optimized_terms": active_tuning_request(request).and_then(|tuning| tuning.target_terms.clone()).unwrap_or_else(default_target_terms)
        },
        "checks": checks,
        "warnings": warnings,
        "errors": errors
    })
}

fn validate_source_files(
    source: &CgSource,
    checks: &mut Vec<Value>,
    warnings: &mut Vec<Value>,
    errors: &mut Vec<Value>,
    aa_atom_count: &mut Option<usize>,
) {
    let manifest = source
        .path
        .as_ref()
        .and_then(|path| read_manifest_json(path, checks, errors));
    let coordinates = source.coordinates.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "coordinates"))
    });
    let topology = source.topology.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "topology"))
    });
    let charge_manifest = source.charge_manifest.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "charge_manifest"))
    });

    if let Some(path) = &source.path {
        validate_path_exists("source.path", path, checks, errors);
    }
    if let Some(path) = &coordinates {
        validate_path_exists("source.coordinates", path, checks, errors);
    }
    if let Some(path) = &topology {
        validate_path_exists("source.topology", path, checks, errors);
    }
    if let Some(path) = &charge_manifest {
        validate_path_exists("source.charge_manifest", path, checks, errors);
    }
    if let Some(path) = &source.trajectory {
        validate_path_exists("source.trajectory", path, checks, errors);
    }

    let coordinate_atoms = coordinates
        .as_ref()
        .and_then(|path| coordinate_atom_count(path, source.format.as_deref(), checks, warnings));
    let topology_atoms = topology.as_ref().and_then(|path| {
        topology_atom_count(path, source.topology_format.as_deref(), checks, warnings)
    });
    if aa_atom_count.is_none() {
        *aa_atom_count = coordinate_atoms.or(topology_atoms);
    }
    if let (Some(coord_atoms), Some(top_atoms)) = (coordinate_atoms, topology_atoms) {
        let status = if coord_atoms == top_atoms {
            "ok"
        } else {
            "error"
        };
        checks.push(json!({
            "name": "coordinate_topology_atom_count_match",
            "status": status,
            "coordinate_atoms": coord_atoms,
            "topology_atoms": top_atoms
        }));
        if coord_atoms != top_atoms {
            errors.push(json!({
                "code": "warp_cg.atom_count_mismatch",
                "message": "source coordinate and topology atom counts differ",
                "coordinate_atoms": coord_atoms,
                "topology_atoms": top_atoms
            }));
        }
    }

    validate_target_selection(
        source,
        topology.as_deref(),
        coordinate_atoms,
        checks,
        warnings,
        errors,
    );
    checks.push(json!({
        "name": "template_match_preconditions",
        "status": if source_mapping_mode_for_source(source).is_some() {
            "available"
        } else {
            "not_available"
        },
        "message": "template mode can replay warp-cg.mapping_template.v1 files against source residue atom names; auto mode emits a reusable generated template"
    }));
}

fn source_mapping_mode_for_source(source: &CgSource) -> Option<&'static str> {
    matches!(
        source.kind.as_str(),
        "polymer_build_manifest"
            | "polymer_pack_manifest"
            | "coordinates_topology"
            | "coordinates_topology_charge_manifest"
            | "source_manifest"
    )
    .then_some("source")
}

fn validate_target_selection(
    source: &CgSource,
    topology: Option<&str>,
    coordinate_atoms: Option<usize>,
    checks: &mut Vec<Value>,
    warnings: &mut Vec<Value>,
    errors: &mut Vec<Value>,
) {
    let Some(selection_expr) = source.target_selection.as_deref() else {
        checks.push(json!({
            "name": "target_selection_declared",
            "status": "not_declared"
        }));
        return;
    };
    checks.push(json!({
        "name": "target_selection_declared",
        "status": "declared",
        "target_selection": selection_expr
    }));
    let Some(topology) = topology else {
        warnings.push(json!({
            "code": "warp_cg.selection_validation_unavailable",
            "message": "source.target_selection was declared but no readable topology path was available for selection evaluation"
        }));
        checks.push(json!({
            "name": "target_selection_selects_atoms",
            "status": "not_available",
            "target_selection": selection_expr
        }));
        return;
    };
    match read_system_auto(Path::new(topology), source.topology_format.as_deref()) {
        Ok(mut system) => match system.select(selection_expr) {
            Ok(selection) => {
                let selected_atoms = selection.indices.len();
                let status = if selected_atoms > 0 { "ok" } else { "error" };
                checks.push(json!({
                    "name": "target_selection_selects_atoms",
                    "status": status,
                    "target_selection": selection_expr,
                    "selected_atoms": selected_atoms
                }));
                if selected_atoms == 0 {
                    errors.push(json!({
                        "code": "warp_cg.selection_empty",
                        "message": "source.target_selection selected no atoms",
                        "target_selection": selection_expr
                    }));
                }
                if let Some(coord_atoms) = coordinate_atoms {
                    checks.push(json!({
                        "name": "selection_topology_atom_count_context",
                        "status": if system.n_atoms() == coord_atoms { "ok" } else { "warning" },
                        "topology_atoms": system.n_atoms(),
                        "coordinate_atoms": coord_atoms
                    }));
                }
            }
            Err(err) => {
                errors.push(json!({
                    "code": "warp_cg.selection_invalid",
                    "message": err.to_string(),
                    "target_selection": selection_expr
                }));
                checks.push(json!({
                    "name": "target_selection_selects_atoms",
                    "status": "error",
                    "target_selection": selection_expr
                }));
            }
        },
        Err(err) => {
            warnings.push(json!({
                "code": "warp_cg.selection_validation_unavailable",
                "message": format!("failed to read topology for source.target_selection: {err}")
            }));
            checks.push(json!({
                "name": "target_selection_selects_atoms",
                "status": "not_available",
                "target_selection": selection_expr
            }));
        }
    }
}

fn read_manifest_json(
    path: &str,
    checks: &mut Vec<Value>,
    errors: &mut Vec<Value>,
) -> Option<Value> {
    let manifest_path = Path::new(path);
    if !manifest_path.exists() {
        return None;
    }
    match std::fs::read_to_string(manifest_path)
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
    {
        Some(value) => {
            checks.push(json!({
                "name": "source_manifest_parse",
                "status": "ok",
                "path": path
            }));
            Some(value)
        }
        None => {
            errors.push(json!({
                "code": "warp_cg.source_manifest_parse_failed",
                "path": path,
                "message": "source.path exists but is not readable JSON"
            }));
            None
        }
    }
}

fn manifest_artifact_path(manifest: &Value, key: &str) -> Option<String> {
    manifest
        .pointer(&format!("/artifacts/{key}"))
        .and_then(manifest_path_value)
        .or_else(|| {
            manifest
                .pointer(&format!("/artifact_paths/{key}"))
                .and_then(manifest_path_value)
        })
        .or_else(|| {
            manifest
                .pointer(&format!("/md_ready_handoff/{key}"))
                .and_then(manifest_path_value)
        })
}

fn manifest_path_value(value: &Value) -> Option<String> {
    value.as_str().map(str::to_string).or_else(|| {
        value
            .get("path")
            .and_then(Value::as_str)
            .map(str::to_string)
    })
}

fn resolve_relative_manifest_path(manifest_path: Option<&str>, path: &str) -> String {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        return candidate.to_string_lossy().to_string();
    }
    manifest_path
        .and_then(|path| Path::new(path).parent())
        .map(|parent| parent.join(candidate).to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

fn resolve_source_handoff(source: &CgSource) -> Result<SourceHandoff> {
    let manifest = source.path.as_ref().and_then(|path| {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
    });
    let manifest_path = source.path.as_deref();
    let coordinates = source
        .coordinates
        .clone()
        .or_else(|| {
            manifest
                .as_ref()
                .and_then(|value| manifest_artifact_path(value, "coordinates"))
        })
        .ok_or_else(|| anyhow!("source coordinates are required for source-driven CG execution"))?;
    let topology = source.topology.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "topology"))
    });
    let trajectory = source.trajectory.clone().or_else(|| {
        manifest
            .as_ref()
            .and_then(|value| manifest_artifact_path(value, "trajectory"))
    });
    Ok(SourceHandoff {
        coordinates: resolve_relative_manifest_path(manifest_path, &coordinates),
        topology: topology.map(|path| resolve_relative_manifest_path(manifest_path, &path)),
        trajectory: trajectory.map(|path| resolve_relative_manifest_path(manifest_path, &path)),
        coordinate_format: source.format.clone(),
        topology_format: source.topology_format.clone(),
    })
}

fn validate_path_exists(field: &str, path: &str, checks: &mut Vec<Value>, errors: &mut Vec<Value>) {
    let exists = Path::new(path).exists();
    checks.push(json!({
        "name": "file_exists",
        "field": field,
        "path": path,
        "status": if exists { "ok" } else { "error" }
    }));
    if !exists {
        errors.push(json!({
            "code": "warp_cg.source_missing",
            "field": field,
            "path": path,
            "message": format!("{field} does not exist")
        }));
    }
}

fn coordinate_atom_count(
    path: &str,
    format: Option<&str>,
    checks: &mut Vec<Value>,
    warnings: &mut Vec<Value>,
) -> Option<usize> {
    match read_molecule(Path::new(path), format, false, true, None) {
        Ok(molecule) => {
            checks.push(json!({
                "name": "coordinate_atom_count",
                "status": "ok",
                "path": path,
                "atom_count": molecule.atoms.len()
            }));
            Some(molecule.atoms.len())
        }
        Err(err) => {
            warnings.push(json!({
                "code": "warp_cg.coordinate_atom_count_unavailable",
                "path": path,
                "message": err.to_string()
            }));
            None
        }
    }
}

fn topology_atom_count(
    path: &str,
    format: Option<&str>,
    checks: &mut Vec<Value>,
    warnings: &mut Vec<Value>,
) -> Option<usize> {
    let is_prmtop = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("prmtop"));
    let result = if is_prmtop {
        read_prmtop_topology(Path::new(path)).map(|topology| topology.atom_names.len())
    } else {
        read_molecule(Path::new(path), format, false, true, None)
            .map(|molecule| molecule.atoms.len())
    };
    match result {
        Ok(atom_count) => {
            checks.push(json!({
                "name": "topology_atom_count",
                "status": "ok",
                "path": path,
                "atom_count": atom_count
            }));
            Some(atom_count)
        }
        Err(err) => {
            warnings.push(json!({
                "code": "warp_cg.topology_atom_count_unavailable",
                "path": path,
                "message": err.to_string()
            }));
            None
        }
    }
}

fn validate_xtb_available(request: &CgRequest, checks: &mut Vec<Value>, warnings: &mut Vec<Value>) {
    let needs_xtb = request
        .reference_source
        .as_ref()
        .is_some_and(|source| source.kind == "xtb")
        || active_tuning_request(request)
            .is_some_and(|tuning| tuning.enabled && tuning.source == "xtb");
    if !needs_xtb {
        return;
    }
    let found = which::which("xtb").ok();
    checks.push(json!({
        "name": "xtb_executable",
        "status": if found.is_some() { "ok" } else { "missing" },
        "path": found.as_ref().map(|path| path.to_string_lossy().to_string())
    }));
    if found.is_none() {
        warnings.push(json!({
            "code": "warp_cg.xtb_missing",
            "message": "xTB was requested but no xtb executable was found on PATH"
        }));
    }
}

fn validate_optimization_cost(request: &CgRequest, checks: &mut Vec<Value>) {
    if let Some(tuning) = active_tuning_request(request).filter(|tuning| tuning.enabled) {
        let max_evaluations = tuning.max_evaluations.unwrap_or(32);
        checks.push(json!({
            "name": "optimization_runtime_estimate",
            "status": "estimated",
            "source": tuning.source,
            "method": tuning.method,
            "max_evaluations": max_evaluations,
            "cost_class": if tuning.source == "xtb" { "high" } else { "medium" },
            "message": if tuning.source == "xtb" {
                "xTB optimization/MD plus parameter tuning can be minutes to hours depending on system size"
            } else {
                "AA trajectory bonded-stat tuning cost scales with trajectory frames and bead count"
            }
        }));
    }
}

fn default_target_terms() -> Vec<String> {
    vec!["bonds".into(), "angles".into(), "dihedrals".into()]
}

fn source_residues(atoms: &[AtomRecord]) -> Vec<SourceResidue> {
    let mut residues = Vec::<SourceResidue>::new();
    let mut lookup = BTreeMap::<(char, i32, String), usize>::new();
    for (atom_idx, atom) in atoms.iter().enumerate() {
        let chain = if atom.chain == ' ' { 'A' } else { atom.chain };
        let resname = if atom.resname.trim().is_empty() {
            "MOL".to_string()
        } else {
            atom.resname.trim().to_string()
        };
        let key = (chain, atom.resid, resname.clone());
        let residue_idx = if let Some(idx) = lookup.get(&key).copied() {
            idx
        } else {
            let idx = residues.len();
            lookup.insert(key, idx);
            residues.push(SourceResidue {
                resid: atom.resid,
                resname,
                chain,
                atom_indices: Vec::new(),
            });
            idx
        };
        residues[residue_idx].atom_indices.push(atom_idx);
    }
    residues
}

fn source_bead_name(
    residue_idx: usize,
    bead_idx: usize,
    bead_count: usize,
    terminal_aware: bool,
) -> String {
    let prefix = if terminal_aware {
        if residue_idx == 0 {
            "H"
        } else {
            "M"
        }
    } else {
        "B"
    };
    if bead_count == 1 {
        prefix.to_string()
    } else {
        format!("{prefix}{}", bead_idx + 1)
    }
}

fn source_atom_groups_for_residue(
    residue: &SourceResidue,
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    target_bead_size: usize,
) -> Vec<(String, Vec<String>, i32, Vec<usize>)> {
    let residue_atoms = residue.atom_indices.iter().copied().collect::<Vec<_>>();
    let local_by_global = residue_atoms
        .iter()
        .enumerate()
        .map(|(local, global)| (*global, local))
        .collect::<BTreeMap<_, _>>();
    let elements = residue_atoms
        .iter()
        .map(|idx| source_atom_element(&atoms[*idx]))
        .collect::<Vec<_>>();
    let local_bonds = bonds
        .iter()
        .filter_map(|&(a, b)| Some((*local_by_global.get(&a)?, *local_by_global.get(&b)?)))
        .collect::<Vec<_>>();
    let residue_molecule = Molecule::from_elements_and_bonds(&elements, &local_bonds);
    let mapping = map_molecule_with_options(
        &residue_molecule,
        &MappingOptions {
            target_bead_size: target_bead_size.max(1),
        },
    );
    mapping
        .bead_names
        .into_iter()
        .zip(mapping.bead_features)
        .zip(mapping.bead_formal_charges)
        .zip(mapping.atom_groups)
        .map(|(((bead_type, features), formal_charge), local_group)| {
            (
                bead_type,
                features,
                formal_charge,
                local_group
                    .into_iter()
                    .filter_map(|local| residue_atoms.get(local).copied())
                    .collect::<Vec<_>>(),
            )
        })
        .filter(|(_, _, _, group)| !group.is_empty())
        .collect()
}

fn bead_center(atom_indices: &[usize], atoms: &[AtomRecord]) -> [f32; 3] {
    let mut center = [0.0f32; 3];
    let count = atom_indices.len().max(1) as f32;
    for idx in atom_indices {
        let pos = atoms[*idx].position;
        center[0] += pos.x;
        center[1] += pos.y;
        center[2] += pos.z;
    }
    [center[0] / count, center[1] / count, center[2] / count]
}

fn source_mapping_mode(request: &CgRequest) -> &str {
    request
        .mapping
        .as_ref()
        .map(|mapping| mapping.mode.as_str())
        .unwrap_or("auto")
}

fn source_mapping_template_ref(request: &CgRequest) -> Option<&str> {
    request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.template.as_deref())
}

fn is_template_source_mapping(request: &CgRequest) -> bool {
    source_mapping_mode(request) == "template"
}

fn load_mapping_template(path: &str) -> Result<Value> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| anyhow!("failed to read mapping template {path}: {err}"))?;
    let value = serde_json::from_str::<Value>(&text)
        .map_err(|err| anyhow!("failed to parse mapping template {path}: {err}"))?;
    let schema = value
        .get("schema_version")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if schema != "warp-cg.mapping_template.v1" {
        return Err(anyhow!(
            "mapping template {path} must use schema_version warp-cg.mapping_template.v1"
        ));
    }
    Ok(value)
}

fn residue_role(residue_idx: usize, residue_count: usize) -> &'static str {
    if residue_idx == 0 {
        "head"
    } else if residue_idx + 1 == residue_count {
        "tail"
    } else {
        "middle"
    }
}

fn source_atom_name(atom: &AtomRecord) -> String {
    atom.name.trim().to_string()
}

fn source_atom_element(atom: &AtomRecord) -> String {
    let element = atom.element.trim();
    if !element.is_empty() {
        return element.to_string();
    }
    atom.name
        .chars()
        .find(|ch| ch.is_ascii_alphabetic())
        .map(|ch| ch.to_ascii_uppercase().to_string())
        .unwrap_or_else(|| "X".to_string())
}

fn residue_template_from_beads(
    role: &str,
    residue: &SourceResidue,
    residue_beads: &[usize],
    beads: &[SourceBeadRecord],
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
) -> Value {
    json!({
        "role": role,
        "resname": residue.resname,
        "beads": residue_beads.iter().filter_map(|bead_idx| beads.get(*bead_idx)).map(|bead| {
            json!({
                "name": bead.name,
                "bead_type": bead.bead_type,
                "features": bead.features,
                "formal_charge": bead.formal_charge,
                "atom_names": bead.atom_names,
                "elements": bead.atom_indices.iter().map(|idx| source_atom_element(&atoms[*idx])).collect::<Vec<_>>(),
                "local_bonds": atom_name_bonds_for_group(&bead.atom_indices, atoms, bonds),
                "connected": atom_group_is_connected(&bead.atom_indices, bonds)
            })
        }).collect::<Vec<_>>()
    })
}

fn atom_name_bonds_for_group(
    atom_indices: &[usize],
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
) -> Vec<[String; 2]> {
    let group = atom_indices.iter().copied().collect::<BTreeSet<_>>();
    let mut pairs = bonds
        .iter()
        .filter_map(|&(a, b)| {
            if group.contains(&a) && group.contains(&b) {
                let mut names = [source_atom_name(&atoms[a]), source_atom_name(&atoms[b])];
                names.sort();
                Some(names)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    pairs.sort();
    pairs.dedup();
    pairs
}

fn atom_group_is_connected(atom_indices: &[usize], bonds: &[(usize, usize)]) -> bool {
    if atom_indices.len() <= 1 {
        return true;
    }
    let group = atom_indices.iter().copied().collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut queue = VecDeque::from([atom_indices[0]]);
    while let Some(current) = queue.pop_front() {
        if !seen.insert(current) {
            continue;
        }
        for &(a, b) in bonds {
            let neighbor = if a == current {
                b
            } else if b == current {
                a
            } else {
                continue;
            };
            if group.contains(&neighbor) && !seen.contains(&neighbor) {
                queue.push_back(neighbor);
            }
        }
    }
    seen.len() == atom_indices.len()
}

fn build_generated_mapping_template(
    request: &CgRequest,
    residues: &[SourceResidue],
    residue_to_bead_indices: &[Vec<usize>],
    beads: &[SourceBeadRecord],
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    terminal_aware: bool,
) -> Value {
    let head = residues.first().map(|residue| {
        residue_template_from_beads(
            "head",
            residue,
            residue_to_bead_indices
                .first()
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            beads,
            atoms,
            bonds,
        )
    });
    let middle_idx = if residues.len() > 2 { 1 } else { 0 };
    let middle = residues.get(middle_idx).map(|residue| {
        residue_template_from_beads(
            "middle",
            residue,
            residue_to_bead_indices
                .get(middle_idx)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            beads,
            atoms,
            bonds,
        )
    });
    let tail = residues.last().map(|residue| {
        residue_template_from_beads(
            "tail",
            residue,
            residue_to_bead_indices
                .last()
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            beads,
            atoms,
            bonds,
        )
    });
    json!({
        "schema_version": "warp-cg.mapping_template.v1",
        "name": request.mapping.as_ref().and_then(|mapping| mapping.repeat_unit_hint.clone()).unwrap_or_else(|| request.name.clone()),
        "generated_by": "warp-cg.auto",
        "mapping_granularity": "residue_graph_partition",
        "strategy": request.mapping.as_ref().and_then(|mapping| mapping.strategy.clone()).unwrap_or_else(|| "polymer_residue_graph".to_string()),
        "target_bead_size": request.mapping.as_ref().and_then(|mapping| mapping.target_bead_size).unwrap_or(4),
        "preserve_functional_groups": request.mapping.as_ref().and_then(|mapping| mapping.preserve_functional_groups).unwrap_or(true),
        "terminal_aware": terminal_aware,
        "repeat_unit_hint": request.mapping.as_ref().and_then(|mapping| mapping.repeat_unit_hint.clone()),
        "residue_role_templates": {
            "head": head,
            "middle": middle,
            "tail": tail
        },
        "validation": {
            "require_connected_beads": true,
            "match_by": ["atom_name", "element", "local_graph"]
        }
    })
}

fn source_mapping_provenance(
    request: &CgRequest,
    handoff: &SourceHandoff,
    residues: &[SourceResidue],
    atoms: &[AtomRecord],
    residue_to_beads: Vec<Value>,
    atom_to_bead: &BTreeMap<usize, usize>,
    mode: &str,
) -> Value {
    let source = request.source.as_ref();
    let terminal_aware = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.terminal_aware)
        .unwrap_or(true);
    let mut residue_name_counts = BTreeMap::<String, usize>::new();
    for residue in residues {
        *residue_name_counts
            .entry(residue.resname.clone())
            .or_default() += 1;
    }
    let selected_atom_indices = atom_to_bead.keys().copied().collect::<Vec<_>>();
    json!({
        "mapping_mode": mode,
        "source_coordinates": handoff.coordinates.clone(),
        "source_topology": handoff.topology.clone(),
        "source_trajectory": handoff.trajectory.clone(),
        "selection": {
            "target_selection": source.and_then(|source| source.target_selection.clone()),
            "policy": if source.and_then(|source| source.target_selection.as_ref()).is_some() {
                "source.target_selection declared; provenance records atoms mapped by the resolved source coordinates"
            } else {
                "default_all_source_coordinate_atoms_and_residues"
            },
            "default_scope": "all atoms and residues in resolved source coordinates",
            "selected_atom_count": atom_to_bead.len(),
            "selected_residue_count": residues.len(),
            "selected_atom_indices": selected_atom_indices
        },
        "residue_interpretation": {
            "terminal_aware": terminal_aware,
            "repeat_unit_hint": request.mapping.as_ref().and_then(|mapping| mapping.repeat_unit_hint.clone()),
            "repeat_unit_interpretation": "one source residue is treated as one polymer repeat/terminal unit for source-driven polymer mapping",
            "residue_count": residues.len(),
            "residue_name_counts": residue_name_counts,
            "residues": residues.iter().enumerate().map(|(idx, residue)| {
                json!({
                    "residue_index": idx,
                    "role": residue_role(idx, residues.len()),
                    "resid": residue.resid,
                    "resname": residue.resname,
                    "chain": residue.chain.to_string(),
                    "atom_count": residue.atom_indices.len(),
                    "atom_indices": residue.atom_indices,
                    "atom_names": residue.atom_indices.iter().map(|atom_idx| {
                        source_atom_name(&atoms[*atom_idx])
                    }).collect::<Vec<_>>()
                })
            }).collect::<Vec<_>>()
        },
        "residue_to_bead_map": residue_to_beads,
        "aa_atom_to_cg_bead": atom_to_bead.iter().map(|(atom_idx, bead_idx)| {
            json!({"aa_atom_index": atom_idx, "cg_bead_index": bead_idx})
        }).collect::<Vec<_>>()
    })
}

fn source_connections_from_mapping(
    molecule_bonds: &[(usize, usize)],
    atom_to_bead: &BTreeMap<usize, usize>,
    residue_to_bead_indices: &[Vec<usize>],
) -> Vec<(usize, usize)> {
    let mut connections = molecule_bonds
        .iter()
        .filter_map(|(a, b)| {
            let bead_a = atom_to_bead.get(a).copied()?;
            let bead_b = atom_to_bead.get(b).copied()?;
            (bead_a != bead_b).then_some((bead_a.min(bead_b), bead_a.max(bead_b)))
        })
        .collect::<Vec<_>>();
    for residue_beads in residue_to_bead_indices {
        for pair in residue_beads.windows(2) {
            connections.push((pair[0].min(pair[1]), pair[0].max(pair[1])));
        }
    }
    let residue_first_last = residue_to_bead_indices
        .iter()
        .filter_map(|beads| Some((*beads.first()?, *beads.last()?)))
        .collect::<Vec<_>>();
    for pair in residue_first_last.windows(2) {
        connections.push((pair[0].1.min(pair[1].0), pair[0].1.max(pair[1].0)));
    }
    connections.sort_unstable();
    connections.dedup();
    connections
}

fn template_beads_for_role<'a>(template: &'a Value, role: &str) -> Result<&'a Vec<Value>> {
    template
        .pointer(&format!("/residue_role_templates/{role}/beads"))
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("mapping template is missing residue_role_templates.{role}.beads"))
}

fn template_atom_indices_for_residue(
    residue: &SourceResidue,
    atoms: &[AtomRecord],
    atom_names: &[Value],
) -> Result<Vec<usize>> {
    let names = atom_names
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_string)
                .ok_or_else(|| anyhow!("mapping template atom_names entries must be strings"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut indices = Vec::new();
    for wanted in &names {
        let matches = residue
            .atom_indices
            .iter()
            .copied()
            .filter(|idx| source_atom_name(&atoms[*idx]) == *wanted)
            .collect::<Vec<_>>();
        match matches.as_slice() {
            [idx] => indices.push(*idx),
            [] => {
                return Err(anyhow!(
                    "mapping template missing atom {wanted} in residue {} {}",
                    residue.resname,
                    residue.resid
                ))
            }
            _ => {
                return Err(anyhow!(
                    "mapping template atom {wanted} matched multiple atoms in residue {} {}",
                    residue.resname,
                    residue.resid
                ))
            }
        }
    }
    Ok(indices)
}

fn value_string_list(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn value_bond_list(value: Option<&Value>) -> Vec<[String; 2]> {
    let mut bonds = value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| {
                    let pair = item.as_array()?;
                    let mut names = [
                        pair.first()?.as_str()?.to_string(),
                        pair.get(1)?.as_str()?.to_string(),
                    ];
                    names.sort();
                    Some(names)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    bonds.sort();
    bonds.dedup();
    bonds
}

fn validate_template_bead_match(
    residue: &SourceResidue,
    bead_name: &str,
    group: &[usize],
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    bead_template: &Value,
) -> Result<()> {
    let expected_elements = value_string_list(bead_template.get("elements"));
    if !expected_elements.is_empty() {
        let actual_elements = group
            .iter()
            .map(|idx| source_atom_element(&atoms[*idx]))
            .collect::<Vec<_>>();
        if actual_elements != expected_elements {
            return Err(anyhow!(
                "mapping template bead {bead_name} element mismatch in residue {} {}: expected {:?}, got {:?}",
                residue.resname,
                residue.resid,
                expected_elements,
                actual_elements
            ));
        }
    }
    let expected_local_bonds = value_bond_list(bead_template.get("local_bonds"));
    if !expected_local_bonds.is_empty() {
        let actual_local_bonds = atom_name_bonds_for_group(group, atoms, bonds);
        if actual_local_bonds != expected_local_bonds {
            return Err(anyhow!(
                "mapping template bead {bead_name} local bond mismatch in residue {} {}",
                residue.resname,
                residue.resid
            ));
        }
    }
    if bead_template
        .get("connected")
        .and_then(Value::as_bool)
        .unwrap_or(false)
        && !atom_group_is_connected(group, bonds)
    {
        return Err(anyhow!(
            "mapping template bead {bead_name} is disconnected in residue {} {}",
            residue.resname,
            residue.resid
        ));
    }
    Ok(())
}

fn build_template_source_mapping(
    request: &CgRequest,
    handoff: &SourceHandoff,
) -> Result<SourceMappingResult> {
    let template_path = source_mapping_template_ref(request)
        .ok_or_else(|| anyhow!("mapping.mode=template requires mapping.template"))?;
    let template = load_mapping_template(template_path)?;
    let molecule = read_molecule(
        Path::new(&handoff.coordinates),
        handoff.coordinate_format.as_deref(),
        false,
        true,
        handoff.topology.as_deref().map(Path::new),
    )
    .map_err(|err| anyhow!("failed to read source coordinates: {err}"))?;
    let residues = source_residues(&molecule.atoms);
    if residues.is_empty() {
        return Err(anyhow!("source coordinates contain no residues"));
    }

    let mut bead_names = Vec::new();
    let mut atom_groups = Vec::new();
    let mut beads = Vec::new();
    let mut residue_to_beads = Vec::new();
    let mut residue_to_bead_indices = Vec::new();
    for (residue_idx, residue) in residues.iter().enumerate() {
        let role = residue_role(residue_idx, residues.len());
        let bead_templates = template_beads_for_role(&template, role)?;
        let mut residue_bead_indices = Vec::new();
        for bead_template in bead_templates {
            let bead_name = bead_template
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("mapping template bead requires name"))?
                .to_string();
            let bead_type = bead_template
                .get("bead_type")
                .and_then(Value::as_str)
                .unwrap_or(&bead_name)
                .to_string();
            let atom_names = bead_template
                .get("atom_names")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("mapping template bead {bead_name} requires atom_names"))?;
            let group = template_atom_indices_for_residue(residue, &molecule.atoms, atom_names)?;
            validate_template_bead_match(
                residue,
                &bead_name,
                &group,
                &molecule.atoms,
                &molecule.bonds,
                bead_template,
            )?;
            let global_bead_idx = bead_names.len();
            let coord = bead_center(&group, &molecule.atoms);
            bead_names.push(bead_name.clone());
            atom_groups.push(group.clone());
            residue_bead_indices.push(global_bead_idx);
            beads.push(SourceBeadRecord {
                index: global_bead_idx,
                name: bead_name.clone(),
                bead_type,
                features: bead_template
                    .get("features")
                    .and_then(Value::as_array)
                    .map(|features| {
                        features
                            .iter()
                            .filter_map(Value::as_str)
                            .map(str::to_string)
                            .collect()
                    })
                    .unwrap_or_default(),
                formal_charge: bead_template
                    .get("formal_charge")
                    .and_then(Value::as_i64)
                    .unwrap_or(0) as i32,
                resid: residue.resid,
                resname: residue.resname.clone(),
                chain: residue.chain,
                atom_names: group
                    .iter()
                    .map(|idx| source_atom_name(&molecule.atoms[*idx]))
                    .collect(),
                atom_indices: group,
                coord,
            });
        }
        residue_to_bead_indices.push(residue_bead_indices.clone());
        residue_to_beads.push(json!({
            "residue_index": residue_idx,
            "role": role,
            "resid": residue.resid,
            "resname": residue.resname,
            "chain": residue.chain.to_string(),
            "beads": residue_bead_indices
        }));
    }

    let mut atom_to_bead = BTreeMap::<usize, usize>::new();
    for (bead_idx, group) in atom_groups.iter().enumerate() {
        for atom_idx in group {
            atom_to_bead.insert(*atom_idx, bead_idx);
        }
    }
    let connections =
        source_connections_from_mapping(&molecule.bonds, &atom_to_bead, &residue_to_bead_indices);
    let mut templates = template.clone();
    if let Some(object) = templates.as_object_mut() {
        object.insert(
            "template_match_report".to_string(),
            json!({
                "status": "ok",
                "template": template_path,
                "residue_count": residues.len(),
                "matched_residues": residues.len(),
                "unmapped_atoms": molecule.atoms.len().saturating_sub(atom_to_bead.len()),
                "missing_atoms": [],
                "extra_atoms": []
            }),
        );
    }
    let provenance = source_mapping_provenance(
        request,
        handoff,
        &residues,
        &molecule.atoms,
        residue_to_beads,
        &atom_to_bead,
        "template",
    );

    Ok(SourceMappingResult {
        mapping: MappingResult {
            bead_names,
            atom_groups,
            connections,
            bead_features: beads.iter().map(|bead| bead.features.clone()).collect(),
            bead_formal_charges: beads.iter().map(|bead| bead.formal_charge).collect(),
        },
        beads,
        residue_count: residues.len(),
        aa_atom_count: molecule.atoms.len(),
        templates,
        provenance,
    })
}

fn build_source_mapping(
    request: &CgRequest,
    handoff: &SourceHandoff,
) -> Result<SourceMappingResult> {
    if is_template_source_mapping(request) {
        return build_template_source_mapping(request, handoff);
    }
    let molecule = read_molecule(
        Path::new(&handoff.coordinates),
        handoff.coordinate_format.as_deref(),
        false,
        true,
        handoff.topology.as_deref().map(Path::new),
    )
    .map_err(|err| anyhow!("failed to read source coordinates: {err}"))?;
    let residues = source_residues(&molecule.atoms);
    if residues.is_empty() {
        return Err(anyhow!("source coordinates contain no residues"));
    }
    let terminal_aware = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.terminal_aware)
        .unwrap_or(true);
    let target_bead_size = request
        .mapping
        .as_ref()
        .and_then(|mapping| mapping.target_bead_size)
        .unwrap_or(4);
    let mut bead_names = Vec::new();
    let mut atom_groups = Vec::new();
    let mut beads = Vec::new();
    let mut residue_to_beads = Vec::new();
    let mut residue_to_bead_indices = Vec::new();
    for (residue_idx, residue) in residues.iter().enumerate() {
        let groups = source_atom_groups_for_residue(
            residue,
            &molecule.atoms,
            &molecule.bonds,
            target_bead_size,
        );
        let mut residue_beads = Vec::new();
        for (local_bead_idx, (mapped_bead_type, features, formal_charge, group)) in
            groups.iter().enumerate()
        {
            let global_bead_idx = bead_names.len();
            let is_tail = terminal_aware && residue_idx + 1 == residues.len();
            let mut bead_name =
                source_bead_name(residue_idx, local_bead_idx, groups.len(), terminal_aware);
            if !terminal_aware {
                bead_name = mapped_bead_type.clone();
            }
            if is_tail {
                bead_name = if groups.len() == 1 {
                    "T".to_string()
                } else {
                    format!("T{}", local_bead_idx + 1)
                };
            }
            let coord = bead_center(group, &molecule.atoms);
            bead_names.push(bead_name.clone());
            atom_groups.push(group.clone());
            residue_beads.push(global_bead_idx);
            beads.push(SourceBeadRecord {
                index: global_bead_idx,
                name: bead_name.clone(),
                bead_type: mapped_bead_type.clone(),
                features: features.clone(),
                formal_charge: *formal_charge,
                resid: residue.resid,
                resname: residue.resname.clone(),
                chain: residue.chain,
                atom_indices: group.clone(),
                atom_names: group
                    .iter()
                    .map(|idx| source_atom_name(&molecule.atoms[*idx]))
                    .collect(),
                coord,
            });
        }
        residue_to_bead_indices.push(residue_beads.clone());
        residue_to_beads.push(json!({
            "residue_index": residue_idx,
            "role": residue_role(residue_idx, residues.len()),
            "resid": residue.resid,
            "resname": residue.resname,
            "chain": residue.chain.to_string(),
            "beads": residue_beads
        }));
    }

    let mut atom_to_bead = BTreeMap::<usize, usize>::new();
    for (bead_idx, group) in atom_groups.iter().enumerate() {
        for atom_idx in group {
            atom_to_bead.insert(*atom_idx, bead_idx);
        }
    }
    let connections =
        source_connections_from_mapping(&molecule.bonds, &atom_to_bead, &residue_to_bead_indices);

    let templates = build_generated_mapping_template(
        request,
        &residues,
        &residue_to_bead_indices,
        &beads,
        &molecule.atoms,
        &molecule.bonds,
        terminal_aware,
    );
    let provenance = source_mapping_provenance(
        request,
        handoff,
        &residues,
        &molecule.atoms,
        residue_to_beads,
        &atom_to_bead,
        "auto",
    );

    Ok(SourceMappingResult {
        mapping: MappingResult {
            bead_names,
            atom_groups,
            connections,
            bead_features: beads.iter().map(|bead| bead.features.clone()).collect(),
            bead_formal_charges: beads.iter().map(|bead| bead.formal_charge).collect(),
        },
        beads,
        residue_count: residues.len(),
        aa_atom_count: molecule.atoms.len(),
        templates,
        provenance,
    })
}

fn run_request(request: &CgRequest, started: Instant) -> Result<CgResult> {
    if request.source.is_some() {
        return run_source_request(request, started);
    }
    let molecule_identity = request
        .smiles
        .as_ref()
        .or(request.repeat_smiles.as_ref())
        .ok_or_else(|| {
            anyhow!(
                "warp-cg execution requires smiles, repeat_smiles, or an executable source handoff"
            )
        })?;
    let mol = Molecule::from_smiles(molecule_identity)?;
    let mapping = map_molecule(&mol);
    let out_dir = PathBuf::from(&request.output.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let mut artifacts = Vec::new();
    if request.output.write_mapping_json {
        let mapping_path = out_dir.join(format!("{}_martini_mapping.json", request.name));
        let mapping_value = mapping_json(request, &mapping);
        std::fs::write(&mapping_path, serde_json::to_vec_pretty(&mapping_value)?)?;
        artifacts.push(CgArtifact {
            path: mapping_path.to_string_lossy().to_string(),
            kind: "martini_mapping_json".to_string(),
        });
    }

    let reference_kind = request
        .reference_source
        .as_ref()
        .map(|source| source.kind.as_str())
        .unwrap_or("external");
    let mut xtb_reference_path: Option<PathBuf> = None;
    if reference_kind == "xtb" {
        let xtb_out_dir = request
            .reference_source
            .as_ref()
            .and_then(|source| source.xtb.as_ref())
            .and_then(|xtb| xtb.work_dir.as_ref())
            .map(PathBuf::from)
            .unwrap_or_else(|| out_dir.clone());
        let xtb_config = request
            .reference_source
            .as_ref()
            .and_then(|source| source.xtb.as_ref())
            .map(xtb_run_config)
            .unwrap_or_default();
        let xtb_res = run_xtb_pipeline_with_config(
            &request.name,
            molecule_identity,
            &xtb_out_dir,
            &xtb_config,
        )?;
        artifacts.push(CgArtifact {
            path: xtb_res.opt_xyz.to_string_lossy().to_string(),
            kind: "xtb_optimized_xyz".to_string(),
        });
        if let Some(trj) = xtb_res.trajectory_trj.as_ref() {
            artifacts.push(CgArtifact {
                path: trj.to_string_lossy().to_string(),
                kind: "xtb_reference_trajectory".to_string(),
            });
            xtb_reference_path = Some(trj.clone());
        } else {
            xtb_reference_path = Some(xtb_res.opt_xyz);
        }
    }

    let trajectory_path = xtb_reference_path
        .as_ref()
        .map(|path| path.to_string_lossy().to_string())
        .or_else(|| {
            request
                .trajectory_source
                .as_ref()
                .map(|source| source.path.clone())
        });

    let mut bond_stats: Vec<BondStats> = Vec::new();
    let mut angle_stats: Vec<AngleStats> = Vec::new();
    let mut dihedral_stats: Vec<DihedralStats> = Vec::new();
    let mut first_cg_coords: Option<Vec<[f32; 3]>> = None;

    if let Some(input_traj) = trajectory_path {
        let mapped_name = request
            .output
            .mapped_trajectory
            .clone()
            .unwrap_or_else(|| format!("{}_cg.xtc", request.name));
        let output_path = resolve_output_path(&out_dir, &mapped_name);
        let bead_mapping = BeadMapping {
            bead_names: mapping.bead_names.clone(),
            atom_indices: mapping.atom_groups.clone(),
        };
        let source = normalized_trajectory_source(request, &input_traj);
        let report = map_native_trajectory(
            Path::new(&input_traj),
            Some(&output_path),
            &bead_mapping,
            &mapping.connections,
            &native_options(request, source.as_ref()),
        )?;
        bond_stats = report.bond_stats;
        angle_stats = report.angle_stats;
        dihedral_stats = report.dihedral_stats;
        first_cg_coords = report.first_cg_coords;
        artifacts.push(CgArtifact {
            path: output_path.to_string_lossy().to_string(),
            kind: "coarse_grained_trajectory".to_string(),
        });
    }

    if request.output.write_cg_pdb {
        let pdb_name = request
            .output
            .cg_pdb
            .clone()
            .unwrap_or_else(|| format!("{}_cg.pdb", request.name));
        let pdb_path = resolve_output_path(&out_dir, &pdb_name);
        let pdb = render_cg_pdb(&request.name, &mapping, first_cg_coords.as_deref());
        std::fs::write(&pdb_path, pdb)?;
        artifacts.push(CgArtifact {
            path: pdb_path.to_string_lossy().to_string(),
            kind: "coarse_grained_pdb".to_string(),
        });
    }

    if !bond_stats.is_empty() {
        let stats_path = out_dir.join(format!("{}_bond_stats.json", request.name));
        std::fs::write(&stats_path, serde_json::to_vec_pretty(&bond_stats)?)?;
        artifacts.push(CgArtifact {
            path: stats_path.to_string_lossy().to_string(),
            kind: "bond_stats_json".to_string(),
        });
    }
    if !angle_stats.is_empty() || !dihedral_stats.is_empty() {
        let stats_path = out_dir.join(format!("{}_bonded_stats.json", request.name));
        std::fs::write(
            &stats_path,
            serde_json::to_vec_pretty(&json!({
                "bonds": bond_stats,
                "angles": angle_stats,
                "dihedrals": dihedral_stats,
            }))?,
        )?;
        artifacts.push(CgArtifact {
            path: stats_path.to_string_lossy().to_string(),
            kind: "bonded_stats_json".to_string(),
        });
    }

    let active_tuning = active_tuning_request(request);
    let optimization_result = active_tuning
        .filter(|tuning| tuning.enabled)
        .map(|tuning| {
            let bonded_stats = BondedStats {
                bonds: bond_stats.clone(),
                angles: angle_stats.clone(),
                dihedrals: dihedral_stats.clone(),
            };
            run_optimization(
                tuning,
                &bonded_stats,
                &out_dir,
                &request.name,
                &mut artifacts,
            )
        })
        .transpose()?;

    if request.output.write_topology_itp {
        let itp_path = out_dir.join(format!("{}_martini.itp", request.name));
        let itp = render_martini_itp(
            &request.name,
            &mapping,
            &bond_stats,
            &angle_stats,
            &dihedral_stats,
            optimization_result
                .as_ref()
                .and_then(|tuning| tuning.report.as_ref()),
        );
        std::fs::write(&itp_path, itp)?;
        artifacts.push(CgArtifact {
            path: itp_path.to_string_lossy().to_string(),
            kind: "martini_topology_itp".to_string(),
        });
        if request.output.write_bonded_parameter_map {
            let map_path = out_dir.join(format!("{}_bonded_parameter_map.json", request.name));
            let parameter_map = bonded_parameter_map_json(
                request,
                &mapping,
                &bond_stats,
                &angle_stats,
                &dihedral_stats,
                optimization_result
                    .as_ref()
                    .and_then(|tuning| tuning.report.as_ref()),
            );
            std::fs::write(&map_path, serde_json::to_vec_pretty(&parameter_map)?)?;
            artifacts.push(CgArtifact {
                path: map_path.to_string_lossy().to_string(),
                kind: "bonded_parameter_map_json".to_string(),
            });
        }
    }
    if request.output.write_topology_top {
        let top_path = out_dir.join(format!("{}_martini.top", request.name));
        let itp_file = format!("{}_martini.itp", request.name);
        let top = render_martini_top(&request.name, &itp_file);
        std::fs::write(&top_path, top)?;
        artifacts.push(CgArtifact {
            path: top_path.to_string_lossy().to_string(),
            kind: "martini_topology_top".to_string(),
        });
    }

    Ok(CgResult {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        status: "ok".to_string(),
        exit_code: 0,
        name: request.name.clone(),
        summary: CgSummary {
            input_mode: input_mode(request).to_string(),
            mapping_mode: mapping_mode(request).to_string(),
            aa_atom_count: Some(mol.graph.node_count()),
            cg_bead_count: mapping.bead_names.len(),
            mapped_residue_count: None,
            optimized_terms: optimization_result
                .as_ref()
                .map(|_| {
                    active_tuning
                        .and_then(|tuning| tuning.target_terms.clone())
                        .unwrap_or_else(default_target_terms)
                })
                .unwrap_or_default(),
            optimization_source: active_tuning
                .filter(|tuning| tuning.enabled)
                .map(|tuning| tuning.source.clone()),
        },
        bead_count: mapping.bead_names.len(),
        beads: beads(&mapping),
        connections: mapping.connections.iter().map(|&(i, j)| [i, j]).collect(),
        artifact_paths: artifact_paths(&artifacts),
        artifacts,
        optimization: optimization_result,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn run_source_request(request: &CgRequest, started: Instant) -> Result<CgResult> {
    let source = request
        .source
        .as_ref()
        .ok_or_else(|| anyhow!("source is required"))?;
    let handoff = resolve_source_handoff(source)?;
    let source_mapping = build_source_mapping(request, &handoff)?;
    let out_dir = PathBuf::from(&request.output.out_dir);
    std::fs::create_dir_all(&out_dir)?;

    let mut artifacts = Vec::new();
    if request.output.write_mapping_json {
        let mapping_path = out_dir.join(format!("{}_martini_mapping.json", request.name));
        std::fs::write(
            &mapping_path,
            serde_json::to_vec_pretty(&source_mapping_json(request, &source_mapping))?,
        )?;
        artifacts.push(CgArtifact {
            path: mapping_path.to_string_lossy().to_string(),
            kind: "martini_mapping_json".to_string(),
        });
        let provenance_path = out_dir.join(format!("{}_aa_to_cg_provenance.json", request.name));
        std::fs::write(
            &provenance_path,
            serde_json::to_vec_pretty(&source_mapping.provenance)?,
        )?;
        artifacts.push(CgArtifact {
            path: provenance_path.to_string_lossy().to_string(),
            kind: "aa_to_cg_mapping_provenance".to_string(),
        });
        let template_path = out_dir.join(format!("{}_mapping_template.json", request.name));
        std::fs::write(
            &template_path,
            serde_json::to_vec_pretty(&source_mapping.templates)?,
        )?;
        artifacts.push(CgArtifact {
            path: template_path.to_string_lossy().to_string(),
            kind: "mapping_template_json".to_string(),
        });
    }

    let mut bond_stats: Vec<BondStats> = Vec::new();
    let mut angle_stats: Vec<AngleStats> = Vec::new();
    let mut dihedral_stats: Vec<DihedralStats> = Vec::new();
    let mut first_cg_coords = Some(
        source_mapping
            .beads
            .iter()
            .map(|bead| bead.coord)
            .collect::<Vec<_>>(),
    );

    if let Some(input_traj) = handoff.trajectory.clone() {
        let mapped_name = request
            .output
            .mapped_trajectory
            .clone()
            .unwrap_or_else(|| format!("{}_cg.xtc", request.name));
        let output_path = resolve_output_path(&out_dir, &mapped_name);
        let bead_mapping = BeadMapping {
            bead_names: source_mapping.mapping.bead_names.clone(),
            atom_indices: source_mapping.mapping.atom_groups.clone(),
        };
        let report = map_native_trajectory(
            Path::new(&input_traj),
            Some(&output_path),
            &bead_mapping,
            &source_mapping.mapping.connections,
            &source_native_options(source, &handoff),
        )?;
        bond_stats = report.bond_stats;
        angle_stats = report.angle_stats;
        dihedral_stats = report.dihedral_stats;
        first_cg_coords = report.first_cg_coords.or(first_cg_coords);
        artifacts.push(CgArtifact {
            path: output_path.to_string_lossy().to_string(),
            kind: "coarse_grained_trajectory".to_string(),
        });
    }

    if request.output.write_cg_pdb {
        let pdb_name = request
            .output
            .cg_pdb
            .clone()
            .unwrap_or_else(|| format!("{}_cg.pdb", request.name));
        let pdb_path = resolve_output_path(&out_dir, &pdb_name);
        let pdb = render_source_cg_pdb(&source_mapping, first_cg_coords.as_deref());
        std::fs::write(&pdb_path, pdb)?;
        artifacts.push(CgArtifact {
            path: pdb_path.to_string_lossy().to_string(),
            kind: "coarse_grained_pdb".to_string(),
        });
    }

    if !bond_stats.is_empty() {
        let stats_path = out_dir.join(format!("{}_bond_stats.json", request.name));
        std::fs::write(&stats_path, serde_json::to_vec_pretty(&bond_stats)?)?;
        artifacts.push(CgArtifact {
            path: stats_path.to_string_lossy().to_string(),
            kind: "bond_stats_json".to_string(),
        });
    }
    if !angle_stats.is_empty() || !dihedral_stats.is_empty() {
        let stats_path = out_dir.join(format!("{}_bonded_stats.json", request.name));
        std::fs::write(
            &stats_path,
            serde_json::to_vec_pretty(&json!({
                "bonds": bond_stats,
                "angles": angle_stats,
                "dihedrals": dihedral_stats,
            }))?,
        )?;
        artifacts.push(CgArtifact {
            path: stats_path.to_string_lossy().to_string(),
            kind: "bonded_stats_json".to_string(),
        });
    }

    let active_tuning = active_tuning_request(request);
    let optimization_result = active_tuning
        .filter(|tuning| tuning.enabled)
        .map(|tuning| {
            let bonded_stats = BondedStats {
                bonds: bond_stats.clone(),
                angles: angle_stats.clone(),
                dihedrals: dihedral_stats.clone(),
            };
            run_optimization(
                tuning,
                &bonded_stats,
                &out_dir,
                &request.name,
                &mut artifacts,
            )
        })
        .transpose()?;

    if request.output.write_topology_itp {
        let itp_path = out_dir.join(format!("{}_martini.itp", request.name));
        let itp = render_source_martini_itp(
            &request.name,
            &source_mapping,
            &bond_stats,
            &angle_stats,
            &dihedral_stats,
            optimization_result
                .as_ref()
                .and_then(|tuning| tuning.report.as_ref()),
        );
        std::fs::write(&itp_path, itp)?;
        artifacts.push(CgArtifact {
            path: itp_path.to_string_lossy().to_string(),
            kind: "martini_topology_itp".to_string(),
        });
        if request.output.write_bonded_parameter_map {
            let map_path = out_dir.join(format!("{}_bonded_parameter_map.json", request.name));
            let parameter_map = source_bonded_parameter_map_json(
                request,
                &source_mapping,
                &bond_stats,
                &angle_stats,
                &dihedral_stats,
                optimization_result
                    .as_ref()
                    .and_then(|tuning| tuning.report.as_ref()),
            );
            std::fs::write(&map_path, serde_json::to_vec_pretty(&parameter_map)?)?;
            artifacts.push(CgArtifact {
                path: map_path.to_string_lossy().to_string(),
                kind: "bonded_parameter_map_json".to_string(),
            });
        }
    }
    if request.output.write_topology_top {
        let top_path = out_dir.join(format!("{}_martini.top", request.name));
        let itp_file = format!("{}_martini.itp", request.name);
        let top = render_martini_top(&request.name, &itp_file);
        std::fs::write(&top_path, top)?;
        artifacts.push(CgArtifact {
            path: top_path.to_string_lossy().to_string(),
            kind: "martini_topology_top".to_string(),
        });
    }

    Ok(CgResult {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        status: "ok".to_string(),
        exit_code: 0,
        name: request.name.clone(),
        summary: CgSummary {
            input_mode: input_mode(request).to_string(),
            mapping_mode: mapping_mode(request).to_string(),
            aa_atom_count: Some(source_mapping.aa_atom_count),
            cg_bead_count: source_mapping.mapping.bead_names.len(),
            mapped_residue_count: Some(source_mapping.residue_count),
            optimized_terms: optimization_result
                .as_ref()
                .map(|_| {
                    active_tuning
                        .and_then(|tuning| tuning.target_terms.clone())
                        .unwrap_or_else(default_target_terms)
                })
                .unwrap_or_default(),
            optimization_source: active_tuning
                .filter(|tuning| tuning.enabled)
                .map(|tuning| tuning.source.clone()),
        },
        bead_count: source_mapping.mapping.bead_names.len(),
        beads: beads(&source_mapping.mapping),
        connections: source_mapping
            .mapping
            .connections
            .iter()
            .map(|&(i, j)| [i, j])
            .collect(),
        artifact_paths: artifact_paths(&artifacts),
        artifacts,
        optimization: optimization_result,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn run_optimization(
    tuning: &ParameterTuningRequest,
    bonded_stats: &BondedStats,
    out_dir: &Path,
    name: &str,
    artifacts: &mut Vec<CgArtifact>,
) -> Result<ParameterTuningResult> {
    let report = optimize_bonded_terms(
        bonded_stats,
        &OptimizationConfig {
            method: tuning.method.clone(),
            objective: tuning.objective.clone(),
            max_evaluations: tuning.max_evaluations.unwrap_or(32),
            seed: tuning.seed.unwrap_or(42),
            swarm_size: tuning.swarm_size,
        },
    );
    let report_path = out_dir.join(format!("{}_tuning_report.json", name));
    std::fs::write(&report_path, serde_json::to_vec_pretty(&report)?)?;
    artifacts.push(CgArtifact {
        path: report_path.to_string_lossy().to_string(),
        kind: "bonded_optimization_report".to_string(),
    });

    Ok(ParameterTuningResult {
        status: report.status.clone(),
        method: tuning.method.clone(),
        source: tuning.source.clone(),
        objective: tuning.objective.clone(),
        message: report.message.clone(),
        report: Some(report),
    })
}

fn mapping_json(request: &CgRequest, mapping: &MappingResult) -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "kind": "martini_mapping",
        "name": request.name,
        "smiles": request.smiles,
        "repeat_smiles": request.repeat_smiles,
        "source": request.source,
        "mapping_mode": mapping_mode(request),
        "bead_count": mapping.bead_names.len(),
        "beads": beads(mapping),
        "connections": mapping.connections.iter().map(|&(i, j)| [i, j]).collect::<Vec<_>>()
    })
}

fn source_mapping_json(request: &CgRequest, source_mapping: &SourceMappingResult) -> Value {
    json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "kind": "martini_source_residue_mapping",
        "name": request.name,
        "source": request.source,
        "mapping_mode": mapping_mode(request),
        "repeat_smiles": request.repeat_smiles,
        "aa_atom_count": source_mapping.aa_atom_count,
        "mapped_residue_count": source_mapping.residue_count,
        "bead_count": source_mapping.mapping.bead_names.len(),
        "beads": source_mapping.beads.iter().map(|bead| {
            json!({
                "index": bead.index,
                "name": bead.name,
                "bead_type": bead.bead_type,
                "features": bead.features,
                "formal_charge": bead.formal_charge,
                "resid": bead.resid,
                "resname": bead.resname,
                "chain": bead.chain.to_string(),
                "atom_indices": bead.atom_indices,
                "atom_names": bead.atom_names,
                "coord": bead.coord
            })
        }).collect::<Vec<_>>(),
        "connections": source_mapping
            .mapping
            .connections
            .iter()
            .map(|&(i, j)| [i, j])
            .collect::<Vec<_>>(),
        "repeat_unit_bead_template": source_mapping.templates.pointer("/residue_role_templates/middle"),
        "head_middle_tail_bead_templates": source_mapping.templates.get("residue_role_templates"),
        "generated_mapping_template": source_mapping.templates,
        "residue_to_bead_map": source_mapping
            .provenance
            .get("residue_to_bead_map")
            .cloned()
            .unwrap_or_else(|| json!([])),
        "provenance": source_mapping.provenance
    })
}

fn source_bonded_parameter_map_json(
    request: &CgRequest,
    source_mapping: &SourceMappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    tuning: Option<&OptimizationReport>,
) -> Value {
    let mut value = bonded_parameter_map_json(
        request,
        &source_mapping.mapping,
        bond_stats,
        angle_stats,
        dihedral_stats,
        tuning,
    );
    if let Some(object) = value.as_object_mut() {
        object.insert(
            "source_mapping_templates".to_string(),
            source_mapping.templates.clone(),
        );
        object.insert(
            "mapping_provenance".to_string(),
            source_mapping.provenance.clone(),
        );
    }
    value
}

fn bonded_parameter_map_json(
    request: &CgRequest,
    mapping: &MappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    tuning: Option<&OptimizationReport>,
) -> Value {
    json!({
        "schema_version": "warp-cg.bonded-parameter-map.v1",
        "name": request.name,
        "smiles": request.smiles,
        "repeat_smiles": request.repeat_smiles,
        "source": request.source,
        "mapping_mode": mapping_mode(request),
        "itp": format!("{}_martini.itp", request.name),
        "units": {
            "bond_length": "nm",
            "bond_force": "kJ mol^-1 nm^-2",
            "angle": "degree",
            "angle_force": "kJ mol^-1 rad^-2",
            "dihedral_phase": "degree",
            "dihedral_force": "kJ mol^-1"
        },
        "bonds": mapping.connections.iter().map(|&(i, j)| {
            let a = i.min(j);
            let b = i.max(j);
            let (length_nm, force) = bonded_pair_parameters(i, j, bond_stats, tuning);
            json!({
                "itp_section": "bonds",
                "beads_zero_based": [i, j],
                "itp_atoms_one_based": [i + 1, j + 1],
                "parameter_names": {
                    "length_angstrom": format!("bond_{a}_{b}_length_angstrom"),
                    "force": format!("bond_{a}_{b}_force")
                },
                "source_stat": bond_stats.iter().find(|stat| stat.bead_i == a && stat.bead_j == b),
                "itp_values": {
                    "funct": 1,
                    "length_nm": length_nm,
                    "force": force
                }
            })
        }).collect::<Vec<_>>(),
        "angles": angle_stats.iter().map(|stat| {
            let (angle_deg, force) = bonded_angle_parameters(stat, tuning);
            json!({
                "itp_section": "angles",
                "beads_zero_based": [stat.bead_i, stat.bead_j, stat.bead_k],
                "itp_atoms_one_based": [stat.bead_i + 1, stat.bead_j + 1, stat.bead_k + 1],
                "parameter_names": {
                    "angle_deg": format!("angle_{}_{}_{}_angle_deg", stat.bead_i, stat.bead_j, stat.bead_k),
                    "force": format!("angle_{}_{}_{}_force", stat.bead_i, stat.bead_j, stat.bead_k)
                },
                "source_stat": stat,
                "itp_values": {
                    "funct": 2,
                    "angle_deg": angle_deg,
                    "force": force
                }
            })
        }).collect::<Vec<_>>(),
        "dihedrals": dihedral_stats.iter().map(|stat| {
            let (phase_deg, force) = bonded_dihedral_parameters(stat, tuning);
            json!({
                "itp_section": "dihedrals",
                "beads_zero_based": [stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l],
                "itp_atoms_one_based": [stat.bead_i + 1, stat.bead_j + 1, stat.bead_k + 1, stat.bead_l + 1],
                "parameter_names": {
                    "phase_deg": format!(
                        "dihedral_{}_{}_{}_{}_phase_deg",
                        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
                    ),
                    "force": format!(
                        "dihedral_{}_{}_{}_{}_force",
                        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
                    )
                },
                "source_stat": stat,
                "itp_values": {
                    "funct": 1,
                    "phase_deg": phase_deg,
                    "force": force,
                    "multiplicity": 1
                }
            })
        }).collect::<Vec<_>>()
    })
}

fn beads(mapping: &MappingResult) -> Vec<CgBead> {
    mapping
        .bead_names
        .iter()
        .zip(mapping.atom_groups.iter())
        .enumerate()
        .map(|(index, (name, atom_indices))| CgBead {
            index,
            name: name.clone(),
            atom_indices: atom_indices.clone(),
            features: mapping
                .bead_features
                .get(index)
                .cloned()
                .unwrap_or_default(),
            formal_charge: mapping.bead_formal_charges.get(index).copied().unwrap_or(0),
        })
        .collect()
}

fn render_martini_itp(
    name: &str,
    mapping: &MappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    tuning: Option<&OptimizationReport>,
) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    out.push_str("[ moleculetype ]\n");
    out.push_str("; name  nrexcl\n");
    out.push_str(&format!("{molecule:<16} 1\n\n"));
    out.push_str("[ atoms ]\n");
    out.push_str("; nr  type  resnr  residue  atom  cgnr  charge\n");
    for (idx, bead) in mapping.bead_names.iter().enumerate() {
        out.push_str(&format!(
            "{:>5} {:<6} {:>5} {:<8} {:<6} {:>5} {:>8.3}\n",
            idx + 1,
            bead,
            1,
            molecule,
            format!("B{}", idx + 1),
            idx + 1,
            0.0
        ));
    }
    if !mapping.connections.is_empty() {
        out.push_str("\n[ bonds ]\n");
        out.push_str("; i  j  funct  length(nm)  force\n");
        for &(i, j) in &mapping.connections {
            let (length_nm, force) = bonded_pair_parameters(i, j, bond_stats, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>10.5} {:>10.3}\n",
                i + 1,
                j + 1,
                1,
                length_nm,
                force
            ));
        }
    }
    if !angle_stats.is_empty() {
        out.push_str("\n[ angles ]\n");
        out.push_str("; i  j  k  funct  angle(deg)  force\n");
        for stat in angle_stats {
            let (angle, force) = bonded_angle_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                2,
                angle,
                force
            ));
        }
    }
    if !dihedral_stats.is_empty() {
        out.push_str("\n[ dihedrals ]\n");
        out.push_str("; i  j  k  l  funct  angle(deg)  force  multiplicity\n");
        for stat in dihedral_stats {
            let (phase, force) = bonded_dihedral_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3} {:>5}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                stat.bead_l + 1,
                1,
                phase,
                force,
                1
            ));
        }
    }
    out
}

fn render_source_martini_itp(
    name: &str,
    source_mapping: &SourceMappingResult,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
    tuning: Option<&OptimizationReport>,
) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    out.push_str("[ moleculetype ]\n");
    out.push_str("; name  nrexcl\n");
    out.push_str(&format!("{molecule:<16} 1\n\n"));
    out.push_str("[ atoms ]\n");
    out.push_str("; nr  type  resnr  residue  atom  cgnr  charge\n");
    for bead in &source_mapping.beads {
        out.push_str(&format!(
            "{:>5} {:<6} {:>5} {:<8} {:<6} {:>5} {:>8.3}\n",
            bead.index + 1,
            bead.name,
            bead.resid.max(1),
            bead.resname.chars().take(8).collect::<String>(),
            pdb_atom_name(&bead.name, bead.index),
            bead.index + 1,
            0.0
        ));
    }
    if !source_mapping.mapping.connections.is_empty() {
        out.push_str("\n[ bonds ]\n");
        out.push_str("; i  j  funct  length(nm)  force\n");
        for &(i, j) in &source_mapping.mapping.connections {
            let (length_nm, force) = bonded_pair_parameters(i, j, bond_stats, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>10.5} {:>10.3}\n",
                i + 1,
                j + 1,
                1,
                length_nm,
                force
            ));
        }
    }
    if !angle_stats.is_empty() {
        out.push_str("\n[ angles ]\n");
        out.push_str("; i  j  k  funct  angle(deg)  force\n");
        for stat in angle_stats {
            let (angle, force) = bonded_angle_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                2,
                angle,
                force
            ));
        }
    }
    if !dihedral_stats.is_empty() {
        out.push_str("\n[ dihedrals ]\n");
        out.push_str("; i  j  k  l  funct  angle(deg)  force  multiplicity\n");
        for stat in dihedral_stats {
            let (phase, force) = bonded_dihedral_parameters(stat, tuning);
            out.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>5} {:>10.4} {:>10.3} {:>5}\n",
                stat.bead_i + 1,
                stat.bead_j + 1,
                stat.bead_k + 1,
                stat.bead_l + 1,
                1,
                phase,
                force,
                1
            ));
        }
    }
    out
}

fn render_cg_pdb(name: &str, mapping: &MappingResult, coords: Option<&[[f32; 3]]>) -> String {
    let residue = topology_name(name);
    let residue = residue.chars().take(3).collect::<String>();
    let mut out = String::new();
    out.push_str("REMARK Generated by warp-cg\n");
    if coords.is_none() {
        out.push_str("REMARK Coordinates are deterministic scaffold positions; prefer mapped trajectory/GRO when available.\n");
    }
    for (idx, bead) in mapping.bead_names.iter().enumerate() {
        let coord = coords
            .and_then(|values| values.get(idx))
            .copied()
            .unwrap_or([idx as f32 * 4.7, 0.0, 0.0]);
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:<3} A{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}\n",
            idx + 1,
            pdb_atom_name(bead, idx),
            residue,
            1,
            coord[0],
            coord[1],
            coord[2],
            pdb_element(bead)
        ));
    }
    for &(i, j) in &mapping.connections {
        out.push_str(&format!("CONECT{:>5}{:>5}\n", i + 1, j + 1));
    }
    out.push_str("END\n");
    out
}

fn render_source_cg_pdb(
    source_mapping: &SourceMappingResult,
    coords: Option<&[[f32; 3]]>,
) -> String {
    let mut out = String::new();
    out.push_str("REMARK Generated by warp-cg\n");
    if coords.is_none() {
        out.push_str("REMARK Coordinates are residue bead centers from source coordinates.\n");
    }
    for bead in &source_mapping.beads {
        let coord = coords
            .and_then(|values| values.get(bead.index))
            .copied()
            .unwrap_or(bead.coord);
        out.push_str(&format!(
            "ATOM  {:>5} {:<4} {:<3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}  1.00  0.00          {:>2}\n",
            bead.index + 1,
            pdb_atom_name(&bead.name, bead.index),
            bead.resname.chars().take(3).collect::<String>(),
            bead.chain,
            bead.resid,
            coord[0],
            coord[1],
            coord[2],
            pdb_element(&bead.name)
        ));
    }
    for &(i, j) in &source_mapping.mapping.connections {
        out.push_str(&format!("CONECT{:>5}{:>5}\n", i + 1, j + 1));
    }
    out.push_str("END\n");
    out
}

fn pdb_atom_name(bead: &str, idx: usize) -> String {
    let mut name: String = bead
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(3)
        .collect();
    if name.is_empty() {
        name.push('B');
    }
    format!("{name}{}", (idx + 1) % 10)
}

fn pdb_element(bead: &str) -> &'static str {
    if bead.starts_with('P') {
        "P"
    } else if bead.starts_with('N') {
        "N"
    } else {
        "C"
    }
}

fn angle_force(stat: &AngleStats) -> f64 {
    (1.0 / stat.std_deg.max(1.0).powi(2) * 10_000.0).clamp(1.0, 500.0)
}

fn dihedral_force(stat: &DihedralStats) -> f64 {
    (1.0 / stat.std_deg.max(1.0).powi(2) * 1_000.0).clamp(0.1, 100.0)
}

fn bonded_angle_parameters(stat: &AngleStats, tuning: Option<&OptimizationReport>) -> (f64, f64) {
    let mut angle = stat.mean_deg;
    let mut force = angle_force(stat);
    if let Some(tuning) = tuning {
        let angle_name = format!(
            "angle_{}_{}_{}_angle_deg",
            stat.bead_i, stat.bead_j, stat.bead_k
        );
        let force_name = format!(
            "angle_{}_{}_{}_force",
            stat.bead_i, stat.bead_j, stat.bead_k
        );
        for (name, value) in &tuning.best_parameters {
            if name == &angle_name {
                angle = *value;
            } else if name == &force_name {
                force = *value;
            }
        }
    }
    (angle, force)
}

fn bonded_dihedral_parameters(
    stat: &DihedralStats,
    tuning: Option<&OptimizationReport>,
) -> (f64, f64) {
    let mut phase = stat.mean_deg;
    let mut force = dihedral_force(stat);
    if let Some(tuning) = tuning {
        let phase_name = format!(
            "dihedral_{}_{}_{}_{}_phase_deg",
            stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
        );
        let force_name = format!(
            "dihedral_{}_{}_{}_{}_force",
            stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
        );
        for (name, value) in &tuning.best_parameters {
            if name == &phase_name {
                phase = *value;
            } else if name == &force_name {
                force = *value;
            }
        }
    }
    (phase, force)
}

fn bonded_pair_parameters(
    i: usize,
    j: usize,
    bond_stats: &[BondStats],
    tuning: Option<&OptimizationReport>,
) -> (f64, f64) {
    let (a, b) = if i < j { (i, j) } else { (j, i) };
    let stat = bond_stats
        .iter()
        .find(|stat| stat.bead_i == a && stat.bead_j == b);
    let mut length_angstrom = stat.map(|stat| stat.mean).unwrap_or(4.7);
    let mut force = stat
        .map(|stat| (1.0 / stat.std.max(0.02).powi(2)).clamp(1.0, 5000.0))
        .unwrap_or(1250.0);

    if let Some(tuning) = tuning {
        for (name, value) in &tuning.best_parameters {
            if name == &format!("bond_{a}_{b}_length_angstrom") {
                length_angstrom = *value;
            } else if name == &format!("bond_{a}_{b}_force") {
                force = *value;
            }
        }
    }
    (length_angstrom / 10.0, force)
}

fn topology_name(name: &str) -> String {
    let mut out: String = name
        .chars()
        .filter_map(|ch| {
            if ch.is_ascii_alphanumeric() {
                Some(ch.to_ascii_uppercase())
            } else if ch == '_' || ch == '-' {
                Some('_')
            } else {
                None
            }
        })
        .take(12)
        .collect();
    if out.is_empty() {
        out.push_str("MOL");
    }
    out
}

fn render_martini_top(name: &str, itp_file: &str) -> String {
    let molecule = topology_name(name);
    let mut out = String::new();
    out.push_str("; Generated by warp-cg\n");
    out.push_str("; Include the Martini force-field file used by your simulation engine before this molecule include.\n");
    out.push_str(&format!("#include \"{itp_file}\"\n\n"));
    out.push_str("[ system ]\n");
    out.push_str(&format!("{molecule} coarse-grained system\n\n"));
    out.push_str("[ molecules ]\n");
    out.push_str("; molecule  count\n");
    out.push_str(&format!("{molecule:<16} 1\n"));
    out
}

fn resolve_output_path(out_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        out_dir.join(candidate)
    }
}

fn normalized_trajectory_source(request: &CgRequest, path: &str) -> Option<TrajectorySource> {
    request.trajectory_source.clone().or_else(|| {
        Some(TrajectorySource {
            path: path.to_string(),
            topology: request.topology.clone(),
            format: None,
            topology_format: None,
            kind: default_external_trajectory_kind(),
            stride: None,
            start: None,
            stop: None,
            length_scale: None,
            target_selection: None,
            environment_selection: None,
            atom_indices: None,
            mass_weighted: None,
        })
    })
}

fn native_options(
    request: &CgRequest,
    source: Option<&TrajectorySource>,
) -> NativeTrajectoryOptions {
    NativeTrajectoryOptions {
        topology: source
            .and_then(|source| source.topology.clone())
            .or_else(|| request.topology.clone()),
        topology_format: source.and_then(|source| source.topology_format.clone()),
        format: source.and_then(|source| source.format.clone()),
        start: source.and_then(|source| source.start),
        stop: source.and_then(|source| source.stop),
        stride: source.and_then(|source| source.stride),
        length_scale: source.and_then(|source| source.length_scale),
        target_selection: source.and_then(|source| source.target_selection.clone()),
        atom_indices: source.and_then(|source| source.atom_indices.clone()),
        mass_weighted: source
            .and_then(|source| source.mass_weighted)
            .unwrap_or(false),
        chunk_frames: None,
    }
}

fn source_native_options(source: &CgSource, handoff: &SourceHandoff) -> NativeTrajectoryOptions {
    NativeTrajectoryOptions {
        topology: handoff
            .topology
            .clone()
            .or_else(|| Some(handoff.coordinates.clone())),
        topology_format: handoff.topology_format.clone(),
        format: None,
        start: None,
        stop: None,
        stride: None,
        length_scale: None,
        target_selection: source
            .target_selection
            .as_ref()
            .filter(|selection| selection.as_str() != "polymer")
            .cloned(),
        atom_indices: None,
        mass_weighted: false,
        chunk_frames: None,
    }
}

fn xtb_run_config(request: &XtbRequest) -> XtbRunConfig {
    let default = XtbRunConfig::default();
    XtbRunConfig {
        temperature_k: request.temperature_k.unwrap_or(default.temperature_k),
        time_ps: request.time_ps.unwrap_or(default.time_ps),
        timestep_fs: request.timestep_fs.unwrap_or(default.timestep_fs),
        dump_fs: request.dump_fs.unwrap_or(default.dump_fs),
        gfn: request.gfn.clone().unwrap_or(default.gfn),
        seed: request.seed.unwrap_or(default.seed),
    }
}

fn validate_positive(value: Option<f64>, field: &str) -> Result<()> {
    if value.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err(anyhow!("{field} must be finite and greater than zero"));
    }
    Ok(())
}

fn emit_event(enabled: bool, event: &str, message: &str, name: Option<&str>) {
    if !enabled {
        return;
    }
    let payload = CgEvent {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        event: event.to_string(),
        message: message.to_string(),
        name: name.map(str::to_string),
    };
    eprintln!(
        "{}",
        serde_json::to_string(&payload).expect("serialize warp-cg event")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_polymer_pdb() -> String {
        [
            "ATOM      1 C1   STA A   1       0.000   0.000   0.000  1.00  0.00           C",
            "ATOM      2 C2   STA A   1       1.400   0.000   0.000  1.00  0.00           C",
            "ATOM      3 C1   MID A   2       2.800   0.000   0.000  1.00  0.00           C",
            "ATOM      4 C2   MID A   2       4.200   0.000   0.000  1.00  0.00           C",
            "ATOM      5 C1   END A   3       5.600   0.000   0.000  1.00  0.00           C",
            "ATOM      6 C2   END A   3       7.000   0.000   0.000  1.00  0.00           C",
            "CONECT    1    2",
            "CONECT    2    3",
            "CONECT    3    4",
            "CONECT    4    5",
            "CONECT    5    6",
            "END",
            "",
        ]
        .join("\n")
    }

    fn paa_like_polymer_pdb() -> String {
        [
            "ATOM      1 C1   STA A   1       0.000   0.000   0.000  1.00  0.00           C",
            "ATOM      2 C2   STA A   1       1.400   0.000   0.000  1.00  0.00           C",
            "ATOM      3 C3   STA A   1       2.800   0.000   0.000  1.00  0.00           C",
            "ATOM      4 O1   STA A   1       3.500   0.900   0.000  1.00  0.00           O",
            "ATOM      5 O2   STA A   1       3.500  -0.900   0.000  1.00  0.00           O",
            "ATOM      6 C1   MID A   2       4.200   0.000   0.000  1.00  0.00           C",
            "ATOM      7 C2   MID A   2       5.600   0.000   0.000  1.00  0.00           C",
            "ATOM      8 C3   MID A   2       7.000   0.000   0.000  1.00  0.00           C",
            "ATOM      9 O1   MID A   2       7.700   0.900   0.000  1.00  0.00           O",
            "ATOM     10 O2   MID A   2       7.700  -0.900   0.000  1.00  0.00           O",
            "ATOM     11 C1   END A   3       8.400   0.000   0.000  1.00  0.00           C",
            "ATOM     12 C2   END A   3       9.800   0.000   0.000  1.00  0.00           C",
            "ATOM     13 C3   END A   3      11.200   0.000   0.000  1.00  0.00           C",
            "ATOM     14 O1   END A   3      11.900   0.900   0.000  1.00  0.00           O",
            "ATOM     15 O2   END A   3      11.900  -0.900   0.000  1.00  0.00           O",
            "CONECT    1    2",
            "CONECT    2    3    6",
            "CONECT    3    4    5",
            "CONECT    6    7",
            "CONECT    7    8   11",
            "CONECT    8    9   10",
            "CONECT   11   12",
            "CONECT   12   13",
            "CONECT   13   14   15",
            "END",
            "",
        ]
        .join("\n")
    }

    fn source_request(
        name: &str,
        source_path: &Path,
        out_dir: &Path,
        mode: &str,
        template: Option<String>,
    ) -> CgRequest {
        CgRequest {
            schema_version: AGENT_SCHEMA_VERSION.to_string(),
            name: name.to_string(),
            smiles: None,
            repeat_smiles: None,
            source: Some(CgSource {
                kind: "coordinates_topology".to_string(),
                path: None,
                coordinates: Some(source_path.to_string_lossy().to_string()),
                topology: Some(source_path.to_string_lossy().to_string()),
                charge_manifest: None,
                trajectory: None,
                target_selection: None,
                format: Some("pdb".to_string()),
                topology_format: Some("pdb".to_string()),
            }),
            mapping: Some(CgMappingRequest {
                mode: mode.to_string(),
                strategy: Some("polymer_residue_graph".to_string()),
                target_bead_size: Some(4),
                preserve_functional_groups: Some(true),
                template,
                repeat_unit_hint: Some("PAA".to_string()),
                terminal_aware: Some(true),
            }),
            topology: None,
            trajectory_source: None,
            reference_source: None,
            optimization: None,
            output: CgOutputRequest {
                out_dir: out_dir.to_string_lossy().to_string(),
                mapped_trajectory: None,
                write_mapping_json: true,
                write_topology_itp: true,
                write_topology_top: true,
                write_cg_pdb: true,
                cg_pdb: None,
                write_bonded_parameter_map: true,
            },
        }
    }

    #[test]
    fn example_request_validates() {
        let text = serde_json::to_string(&example_request()).unwrap();
        let (exit_code, value) = validate_request_json(&text);
        assert_eq!(exit_code, 0);
        assert_eq!(value["valid"], true);
    }

    #[test]
    fn benzene_mapping_has_three_beads() {
        let request = CgRequest {
            schema_version: AGENT_SCHEMA_VERSION.to_string(),
            name: "benzene".to_string(),
            smiles: Some("c1ccccc1".to_string()),
            repeat_smiles: None,
            source: None,
            mapping: None,
            topology: None,
            trajectory_source: None,
            reference_source: None,
            optimization: None,
            output: CgOutputRequest {
                out_dir: tempfile::tempdir()
                    .unwrap()
                    .path()
                    .to_string_lossy()
                    .to_string(),
                mapped_trajectory: None,
                write_mapping_json: false,
                write_topology_itp: false,
                write_topology_top: false,
                write_cg_pdb: false,
                cg_pdb: None,
                write_bonded_parameter_map: false,
            },
        };
        let result = run_request(&request, Instant::now()).unwrap();
        assert_eq!(result.bead_count, 3);
        assert!(result.beads.iter().all(|bead| bead.name == "TC5"));
    }

    #[test]
    fn downstream_setup_artifacts_include_pdb_itp_top_and_parameter_map() {
        let tmp = tempfile::tempdir().unwrap();
        let request = CgRequest {
            schema_version: AGENT_SCHEMA_VERSION.to_string(),
            name: "benzene".to_string(),
            smiles: Some("c1ccccc1".to_string()),
            repeat_smiles: None,
            source: None,
            mapping: None,
            topology: None,
            trajectory_source: None,
            reference_source: None,
            optimization: None,
            output: CgOutputRequest {
                out_dir: tmp.path().to_string_lossy().to_string(),
                mapped_trajectory: None,
                write_mapping_json: true,
                write_topology_itp: true,
                write_topology_top: true,
                write_cg_pdb: true,
                cg_pdb: None,
                write_bonded_parameter_map: true,
            },
        };

        let result = run_request(&request, Instant::now()).unwrap();
        let artifact_kinds: Vec<&str> = result
            .artifacts
            .iter()
            .map(|artifact| artifact.kind.as_str())
            .collect();

        assert!(artifact_kinds.contains(&"coarse_grained_pdb"));
        assert!(artifact_kinds.contains(&"martini_topology_itp"));
        assert!(artifact_kinds.contains(&"martini_topology_top"));
        assert!(artifact_kinds.contains(&"bonded_parameter_map_json"));
        assert!(tmp.path().join("benzene_cg.pdb").exists());
        assert!(tmp.path().join("benzene_martini.itp").exists());
        assert!(tmp.path().join("benzene_martini.top").exists());
        assert!(tmp
            .path()
            .join("benzene_bonded_parameter_map.json")
            .exists());
    }

    #[test]
    fn source_manifest_request_validates_without_smiles() {
        let tmp = tempfile::tempdir().unwrap();
        let manifest_path = tmp.path().join("polymer_pack_manifest.json");
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": "warp-pack.manifest.v1",
                "artifacts": {}
            }))
            .unwrap(),
        )
        .unwrap();
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "paa_50mer",
            "source": {
                "kind": "polymer_pack_manifest",
                "path": manifest_path.to_string_lossy()
            },
            "mapping": {
                "mode": "auto",
                "repeat_unit_hint": "PAA",
                "terminal_aware": true
            },
            "output": {"out_dir": tmp.path().to_string_lossy()}
        });
        let (exit_code, value) = validate_request_json(&request.to_string());
        assert_eq!(exit_code, 0, "{value}");
        assert_eq!(value["valid"], true);
        assert_eq!(value["summary"]["input_mode"], "polymer_pack_manifest");
    }

    #[test]
    fn coordinates_topology_source_runs_residue_mapping_without_smiles() {
        let tmp = tempfile::tempdir().unwrap();
        let source_path = tmp.path().join("source.pdb");
        std::fs::write(&source_path, source_polymer_pdb()).unwrap();
        let request = CgRequest {
            schema_version: AGENT_SCHEMA_VERSION.to_string(),
            name: "paa_source".to_string(),
            smiles: None,
            repeat_smiles: None,
            source: Some(CgSource {
                kind: "coordinates_topology".to_string(),
                path: None,
                coordinates: Some(source_path.to_string_lossy().to_string()),
                topology: Some(source_path.to_string_lossy().to_string()),
                charge_manifest: None,
                trajectory: None,
                target_selection: None,
                format: Some("pdb".to_string()),
                topology_format: Some("pdb".to_string()),
            }),
            mapping: Some(CgMappingRequest {
                mode: "auto".to_string(),
                strategy: Some("polymer_residue_graph".to_string()),
                target_bead_size: Some(4),
                preserve_functional_groups: Some(true),
                template: None,
                repeat_unit_hint: Some("PAA".to_string()),
                terminal_aware: Some(true),
            }),
            topology: None,
            trajectory_source: None,
            reference_source: None,
            optimization: None,
            output: CgOutputRequest {
                out_dir: tmp.path().to_string_lossy().to_string(),
                mapped_trajectory: None,
                write_mapping_json: true,
                write_topology_itp: true,
                write_topology_top: true,
                write_cg_pdb: true,
                cg_pdb: None,
                write_bonded_parameter_map: true,
            },
        };

        let result = run_request(&request, Instant::now()).unwrap();
        let artifact_kinds: Vec<&str> = result
            .artifacts
            .iter()
            .map(|artifact| artifact.kind.as_str())
            .collect();

        assert_eq!(result.summary.input_mode, "coordinates_topology");
        assert_eq!(result.summary.aa_atom_count, Some(6));
        assert_eq!(result.summary.mapped_residue_count, Some(3));
        assert_eq!(result.bead_count, 3);
        assert_eq!(result.beads[0].name, "H");
        assert_eq!(result.beads[1].name, "M");
        assert_eq!(result.beads[2].name, "T");
        assert!(artifact_kinds.contains(&"martini_mapping_json"));
        assert!(artifact_kinds.contains(&"aa_to_cg_mapping_provenance"));
        assert!(artifact_kinds.contains(&"coarse_grained_pdb"));
        assert!(artifact_kinds.contains(&"martini_topology_itp"));
        assert!(artifact_kinds.contains(&"martini_topology_top"));
        assert!(artifact_kinds.contains(&"bonded_parameter_map_json"));
        assert!(artifact_kinds.contains(&"mapping_template_json"));

        let pdb = std::fs::read_to_string(tmp.path().join("paa_source_cg.pdb")).unwrap();
        assert_eq!(pdb.matches("\nTER").count(), 0);
        assert!(pdb.contains(" STA A   1"));
        assert!(pdb.contains(" MID A   2"));
        assert!(pdb.contains(" END A   3"));

        let mapping: Value = serde_json::from_slice(
            &std::fs::read(tmp.path().join("paa_source_martini_mapping.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(mapping["kind"], "martini_source_residue_mapping");
        assert_eq!(mapping["beads"][0]["bead_type"], "SC2");
        assert_eq!(
            mapping["generated_mapping_template"]["residue_role_templates"]["middle"]["beads"][0]
                ["bead_type"],
            "SC2"
        );
        assert_eq!(
            mapping["generated_mapping_template"]["residue_role_templates"]["middle"]["beads"][0]
                ["features"][0],
            "hydrocarbon"
        );
        assert_eq!(
            mapping["generated_mapping_template"]["residue_role_templates"]["middle"]["beads"][0]
                ["local_bonds"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            mapping["generated_mapping_template"]["validation"]["match_by"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        assert_eq!(mapping["residue_to_bead_map"].as_array().unwrap().len(), 3);
        assert_eq!(
            mapping["provenance"]["aa_atom_to_cg_bead"]
                .as_array()
                .unwrap()
                .len(),
            6
        );
        assert_eq!(
            mapping["provenance"]["selection"]["policy"],
            "default_all_source_coordinate_atoms_and_residues"
        );
        assert_eq!(mapping["provenance"]["selection"]["selected_atom_count"], 6);
        assert_eq!(
            mapping["provenance"]["selection"]["selected_residue_count"],
            3
        );
        assert_eq!(
            mapping["provenance"]["residue_interpretation"]["repeat_unit_hint"],
            "PAA"
        );
        assert_eq!(
            mapping["provenance"]["residue_interpretation"]["terminal_aware"],
            true
        );
        assert_eq!(
            mapping["provenance"]["residue_interpretation"]["residue_name_counts"]["MID"],
            1
        );
        assert_eq!(
            mapping["provenance"]["residue_interpretation"]["residues"][0]["role"],
            "head"
        );
        assert_eq!(
            mapping["provenance"]["residue_interpretation"]["residues"][2]["role"],
            "tail"
        );

        let replay_tmp = tempfile::tempdir().unwrap();
        let replay_request = CgRequest {
            schema_version: AGENT_SCHEMA_VERSION.to_string(),
            name: "paa_source_replay".to_string(),
            smiles: None,
            repeat_smiles: None,
            source: request.source.clone(),
            mapping: Some(CgMappingRequest {
                mode: "template".to_string(),
                strategy: None,
                target_bead_size: None,
                preserve_functional_groups: None,
                template: Some(
                    tmp.path()
                        .join("paa_source_mapping_template.json")
                        .to_string_lossy()
                        .to_string(),
                ),
                repeat_unit_hint: Some("PAA".to_string()),
                terminal_aware: Some(true),
            }),
            topology: None,
            trajectory_source: None,
            reference_source: None,
            optimization: None,
            output: CgOutputRequest {
                out_dir: replay_tmp.path().to_string_lossy().to_string(),
                mapped_trajectory: None,
                write_mapping_json: true,
                write_topology_itp: true,
                write_topology_top: false,
                write_cg_pdb: true,
                cg_pdb: None,
                write_bonded_parameter_map: true,
            },
        };
        let replay = run_request(&replay_request, Instant::now()).unwrap();
        assert_eq!(replay.summary.mapping_mode, "template");
        assert_eq!(replay.bead_count, result.bead_count);
        assert_eq!(replay.connections, result.connections);
    }

    #[test]
    fn paa_like_source_auto_template_replay_preserves_carboxylate_features() {
        let tmp = tempfile::tempdir().unwrap();
        let source_path = tmp.path().join("paa_like.pdb");
        std::fs::write(&source_path, paa_like_polymer_pdb()).unwrap();
        let request = source_request("paa_like", &source_path, tmp.path(), "auto", None);

        let result = run_request(&request, Instant::now()).unwrap();
        assert_eq!(result.summary.mapped_residue_count, Some(3));
        assert!(result.beads.iter().any(|bead| bead
            .features
            .iter()
            .any(|feature| feature == "carboxylate_or_carboxylic_acid")));

        let mapping: Value = serde_json::from_slice(
            &std::fs::read(tmp.path().join("paa_like_martini_mapping.json")).unwrap(),
        )
        .unwrap();
        let middle_beads = mapping["generated_mapping_template"]["residue_role_templates"]
            ["middle"]["beads"]
            .as_array()
            .unwrap();
        assert!(middle_beads.iter().any(|bead| bead["features"]
            .as_array()
            .unwrap()
            .iter()
            .any(|feature| feature == "carboxylate_or_carboxylic_acid")));

        let replay_tmp = tempfile::tempdir().unwrap();
        let replay = run_request(
            &source_request(
                "paa_like_replay",
                &source_path,
                replay_tmp.path(),
                "template",
                Some(
                    tmp.path()
                        .join("paa_like_mapping_template.json")
                        .to_string_lossy()
                        .to_string(),
                ),
            ),
            Instant::now(),
        )
        .unwrap();
        assert_eq!(replay.bead_count, result.bead_count);
        assert_eq!(replay.connections, result.connections);
    }

    #[test]
    fn template_replay_rejects_wrong_local_bond_signature() {
        let tmp = tempfile::tempdir().unwrap();
        let source_path = tmp.path().join("paa_like.pdb");
        std::fs::write(&source_path, paa_like_polymer_pdb()).unwrap();
        let request = source_request(
            "paa_like_bad_template",
            &source_path,
            tmp.path(),
            "auto",
            None,
        );
        run_request(&request, Instant::now()).unwrap();

        let template_path = tmp
            .path()
            .join("paa_like_bad_template_mapping_template.json");
        let mut template: Value =
            serde_json::from_slice(&std::fs::read(&template_path).unwrap()).unwrap();
        template["residue_role_templates"]["middle"]["beads"][0]["local_bonds"] =
            json!([["C1", "O2"]]);
        let bad_template_path = tmp.path().join("bad_mapping_template.json");
        std::fs::write(
            &bad_template_path,
            serde_json::to_vec_pretty(&template).unwrap(),
        )
        .unwrap();

        let err = run_request(
            &source_request(
                "paa_like_bad_replay",
                &source_path,
                tmp.path(),
                "template",
                Some(bad_template_path.to_string_lossy().to_string()),
            ),
            Instant::now(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("local bond mismatch"));
    }

    #[test]
    fn polymer_manifest_source_run_resolves_relative_artifacts() {
        let tmp = tempfile::tempdir().unwrap();
        let source_path = tmp.path().join("source.pdb");
        let manifest_path = tmp.path().join("polymer_build_manifest.json");
        std::fs::write(&source_path, source_polymer_pdb()).unwrap();
        std::fs::write(
            &manifest_path,
            serde_json::to_vec_pretty(&json!({
                "schema_version": "warp-build.manifest.v1",
                "artifacts": {
                    "coordinates": "source.pdb",
                    "topology": {"path": "source.pdb"}
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let request = CgRequest {
            schema_version: AGENT_SCHEMA_VERSION.to_string(),
            name: "manifest_source".to_string(),
            smiles: None,
            repeat_smiles: None,
            source: Some(CgSource {
                kind: "polymer_build_manifest".to_string(),
                path: Some(manifest_path.to_string_lossy().to_string()),
                coordinates: None,
                topology: None,
                charge_manifest: None,
                trajectory: None,
                target_selection: None,
                format: Some("pdb".to_string()),
                topology_format: Some("pdb".to_string()),
            }),
            mapping: Some(CgMappingRequest {
                mode: "auto".to_string(),
                strategy: Some("polymer_residue_graph".to_string()),
                target_bead_size: Some(4),
                preserve_functional_groups: Some(true),
                template: None,
                repeat_unit_hint: Some("PAA".to_string()),
                terminal_aware: Some(true),
            }),
            topology: None,
            trajectory_source: None,
            reference_source: None,
            optimization: None,
            output: CgOutputRequest {
                out_dir: tmp.path().to_string_lossy().to_string(),
                mapped_trajectory: None,
                write_mapping_json: true,
                write_topology_itp: false,
                write_topology_top: false,
                write_cg_pdb: false,
                cg_pdb: None,
                write_bonded_parameter_map: false,
            },
        };

        let result = run_request(&request, Instant::now()).unwrap();
        assert_eq!(result.summary.input_mode, "polymer_build_manifest");
        assert_eq!(result.summary.mapped_residue_count, Some(3));
        assert!(tmp
            .path()
            .join("manifest_source_martini_mapping.json")
            .exists());
        assert!(tmp
            .path()
            .join("manifest_source_aa_to_cg_provenance.json")
            .exists());
    }

    #[test]
    fn request_requires_one_identity_mode() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "missing_identity",
            "output": {"out_dir": "."}
        });
        let (exit_code, value) = validate_request_json(&request.to_string());
        assert_eq!(exit_code, 2);
        assert_eq!(value["valid"], false);
    }

    #[test]
    fn legacy_v1_fields_are_rejected() {
        for request in [
            json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "name": "legacy_tuning",
                "smiles": "CCO",
                "parameter_tuning": {"enabled": false}
            }),
            json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "name": "legacy_trajectory",
                "smiles": "CCO",
                "trajectory": "traj.xtc"
            }),
            json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "name": "legacy_template",
                "source": {"kind": "coordinates_topology", "coordinates": "a.pdb", "topology": "a.pdb"},
                "mapping_template": "template.json"
            }),
        ] {
            let (exit_code, value) = validate_request_json(&request.to_string());
            assert_eq!(exit_code, 2, "{value}");
            assert_eq!(value["valid"], false);
        }
    }

    #[test]
    fn old_agent_schema_version_is_rejected() {
        let request = json!({
            "schema_version": "warp-cg.agent.v1",
            "name": "old_schema",
            "smiles": "CCO"
        });
        let (exit_code, value) = validate_request_json(&request.to_string());
        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("warp-cg.agent.v2"));
    }

    #[test]
    fn external_trajectory_optimization_requires_trajectory_source() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "optimization": {
                "enabled": true,
                "source": "external_trajectory",
                "method": "bayesian_optimization"
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());
        assert_eq!(exit_code, 2);
        assert_eq!(value["valid"], false);
    }

    #[test]
    fn topology_top_requires_itp_output() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "output": {
                "write_topology_itp": false,
                "write_topology_top": true
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("write_topology_top requires"));
    }

    #[test]
    fn disabling_itp_without_top_field_remains_valid() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "output": {
                "write_topology_itp": false
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 0);
        assert_eq!(value["valid"], true);
    }

    #[test]
    fn xtb_gfn_must_not_be_empty() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "ethanol",
            "smiles": "CCO",
            "reference_source": {
                "kind": "xtb",
                "xtb": {
                    "gfn": "  "
                }
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("reference_source.xtb.gfn"));
    }

    #[test]
    fn xtb_optimization_requires_xtb_reference_source() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "ethanol",
            "smiles": "CCO",
            "optimization": {
                "enabled": true,
                "source": "xtb",
                "method": "pso"
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("xtb parameter tuning requires"));
    }

    #[test]
    fn target_selection_and_atom_indices_are_mutually_exclusive() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "path": "traj.xtc",
                "topology": "topology.pdb",
                "target_selection": "resname BENZ",
                "atom_indices": [0, 1, 2, 3, 4, 5]
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("target_selection or atom_indices"));
    }

    #[test]
    fn tuning_counts_must_be_positive() {
        let base = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "kind": "external",
                "path": "traj.xtc"
            }
        });
        for tuning in [
            json!({
                "enabled": true,
                "source": "external_trajectory",
                "method": "bayesian_optimization",
                "max_evaluations": 0
            }),
            json!({
                "enabled": true,
                "source": "external_trajectory",
                "method": "pso",
                "swarm_size": 0
            }),
        ] {
            let mut request = base.clone();
            request["optimization"] = tuning;
            let (exit_code, value) = validate_request_json(&request.to_string());

            assert_eq!(exit_code, 2);
            assert_eq!(value["valid"], false);
        }
    }

    #[test]
    fn trajectory_source_kind_must_be_external() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "kind": "xtb",
                "path": "traj.xtc"
            }
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("trajectory_source.kind"));
    }

    #[test]
    fn trajectory_source_string_fields_must_not_be_empty() {
        for field in ["topology", "target_selection", "environment_selection"] {
            let mut request = json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "name": "benzene",
                "smiles": "c1ccccc1",
                "trajectory_source": {
                    "kind": "external",
                    "path": "traj.xtc"
                }
            });
            request["trajectory_source"][field] = json!("  ");
            let (exit_code, value) = validate_request_json(&request.to_string());

            assert_eq!(exit_code, 2);
            assert!(value["error"]["message"].as_str().unwrap().contains(field));
        }
    }

    #[test]
    fn top_level_topology_must_not_be_empty() {
        let request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "topology": "  "
        });
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("topology must not be empty"));
    }

    #[test]
    fn output_paths_must_not_be_empty() {
        for (field, value) in [("out_dir", json!("  ")), ("mapped_trajectory", json!("  "))] {
            let mut request = json!({
                "schema_version": AGENT_SCHEMA_VERSION,
                "name": "benzene",
                "smiles": "c1ccccc1",
                "output": {
                    "out_dir": "."
                }
            });
            request["output"][field] = value;
            let (exit_code, result) = validate_request_json(&request.to_string());

            assert_eq!(exit_code, 2);
            assert!(result["error"]["message"].as_str().unwrap().contains(field));
        }
    }

    #[test]
    fn martini_itp_contains_atoms_and_bonds() {
        let mol = Molecule::from_smiles("c1ccccc1").unwrap();
        let mapping = map_molecule(&mol);
        let itp = render_martini_itp("benzene", &mapping, &[], &[], &[], None);

        assert!(itp.contains("[ moleculetype ]"));
        assert!(itp.contains("[ atoms ]"));
        assert!(itp.contains("[ bonds ]"));
        assert!(itp.contains("BENZENE"));
    }

    #[test]
    fn martini_itp_contains_angle_and_dihedral_sections() {
        let mapping = MappingResult {
            bead_names: vec![
                "C1".to_string(),
                "C1".to_string(),
                "C1".to_string(),
                "C1".to_string(),
            ],
            atom_groups: vec![vec![0], vec![1], vec![2], vec![3]],
            connections: vec![(0, 1), (1, 2), (2, 3)],
            bead_features: vec![
                vec!["hydrocarbon".to_string()],
                vec!["hydrocarbon".to_string()],
                vec!["hydrocarbon".to_string()],
                vec!["hydrocarbon".to_string()],
            ],
            bead_formal_charges: vec![0, 0, 0, 0],
        };
        let itp = render_martini_itp(
            "chain",
            &mapping,
            &[],
            &[AngleStats {
                bead_i: 0,
                bead_j: 1,
                bead_k: 2,
                mean_deg: 120.0,
                std_deg: 5.0,
                samples: 4,
            }],
            &[DihedralStats {
                bead_i: 0,
                bead_j: 1,
                bead_k: 2,
                bead_l: 3,
                mean_deg: 180.0,
                std_deg: 10.0,
                samples: 4,
            }],
            None,
        );

        assert!(itp.contains("[ angles ]"));
        assert!(itp.contains("[ dihedrals ]"));
    }

    #[test]
    fn martini_top_includes_generated_itp_and_molecule_count() {
        let top = render_martini_top("benzene", "benzene_martini.itp");

        assert!(top.contains("#include \"benzene_martini.itp\""));
        assert!(top.contains("[ system ]"));
        assert!(top.contains("[ molecules ]"));
        assert!(top.contains("BENZENE"));
    }
}
