use serde::{Deserialize, Serialize};
use traj_core::geom::Vec3;

pub type BoxVectors = [[f32; 3]; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomRecordKind {
    Atom,
    HetAtom,
}

#[derive(Clone, Debug, Default)]
pub struct PdbAtomMetadata {
    pub occupancy: Option<f32>,
    pub temp_factor: Option<f32>,
    pub altloc: Option<char>,
    pub insertion_code: Option<char>,
    pub formal_charge: Option<String>,
    pub pqr_radius: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct AtomRecord {
    pub record_kind: AtomRecordKind,
    pub name: String,
    pub element: String,
    pub resname: String,
    pub resid: i32,
    pub chain: char,
    pub segid: String,
    pub charge: f32,
    pub position: Vec3,
    pub mol_id: i32,
    pub pdb_metadata: Option<PdbAtomMetadata>,
}

#[derive(Clone, Debug)]
pub struct PackOutput {
    pub atoms: Vec<AtomRecord>,
    pub bonds: Vec<(usize, usize)>,
    pub box_size: [f32; 3],
    pub box_vectors: Option<BoxVectors>,
    pub ter_after: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputSpec {
    pub path: String,
    #[serde(default)]
    pub format: String,
    #[serde(default)]
    pub scale: Option<f32>,
}
