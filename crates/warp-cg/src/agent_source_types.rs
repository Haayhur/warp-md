use crate::mapping::MappingResult;
use serde_json::Value;

#[derive(Clone, Debug)]
pub(super) struct SourceHandoff {
    pub(super) coordinates: String,
    pub(super) topology: Option<String>,
    pub(super) trajectory: Option<String>,
    pub(super) coordinate_format: Option<String>,
    pub(super) topology_format: Option<String>,
}

#[derive(Clone, Debug)]
pub(super) struct SourceResidue {
    pub(super) resid: i32,
    pub(super) resname: String,
    pub(super) chain: char,
    pub(super) atom_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub(super) struct SourceBeadRecord {
    pub(super) index: usize,
    pub(super) name: String,
    pub(super) bead_type: String,
    pub(super) features: Vec<String>,
    pub(super) formal_charge: i32,
    pub(super) resid: i32,
    pub(super) resname: String,
    pub(super) chain: char,
    pub(super) atom_indices: Vec<usize>,
    pub(super) atom_names: Vec<String>,
    pub(super) coord: [f32; 3],
}

pub(super) struct SourceMappingResult {
    pub(super) mapping: MappingResult,
    pub(super) beads: Vec<SourceBeadRecord>,
    pub(super) residue_count: usize,
    pub(super) aa_atom_count: usize,
    pub(super) templates: Value,
    pub(super) provenance: Value,
    pub(super) warnings: Vec<Value>,
    pub(super) mapping_summary: Value,
}
