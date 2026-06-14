use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::build_contract_defaults::*;
use super::BuildBeadTemplate;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoleculeDefinition {
    #[serde(default = "default_molecule_definition_schema")]
    #[schemars(default = "default_molecule_definition_schema")]
    pub schema_version: String,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub beads: Vec<BuildBeadTemplate>,
    #[serde(default)]
    pub residues: Vec<MoleculeDefinitionResidue>,
    #[serde(default)]
    pub bonds: Vec<MoleculeDefinitionBond>,
    #[serde(default)]
    pub angles: Vec<MoleculeDefinitionAngle>,
    #[serde(default)]
    pub dihedrals: Vec<MoleculeDefinitionDihedral>,
    #[serde(default)]
    pub net_charge_e: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoleculeDefinitionBond {
    pub bead_indices: [usize; 2],
    #[serde(default)]
    pub length_nm: Option<f32>,
    #[serde(default)]
    pub force_kj_mol_nm2: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoleculeDefinitionAngle {
    pub bead_indices: [usize; 3],
    #[serde(default)]
    pub angle_degrees: Option<f32>,
    #[serde(default)]
    pub force_kj_mol_rad2: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoleculeDefinitionDihedral {
    pub bead_indices: [usize; 4],
    #[serde(default)]
    pub phase_degrees: Option<f32>,
    #[serde(default)]
    pub force_kj_mol: Option<f32>,
    #[serde(default)]
    pub multiplicity: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MoleculeDefinitionResidue {
    #[serde(default)]
    pub name: Option<String>,
    pub beads: Vec<BuildBeadTemplate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InsertedPlacement {
    #[serde(default = "default_inserted_center")]
    #[schemars(default = "default_inserted_center")]
    pub center_angstrom: [f32; 3],
    #[serde(default = "default_inserted_center_method")]
    #[schemars(default = "default_inserted_center_method")]
    pub center_method: String,
    #[serde(default = "default_inserted_orientation")]
    #[schemars(default = "default_inserted_orientation")]
    pub orientation: String,
    #[serde(default)]
    pub rotate_degrees_xyz: [f32; 3],
}

impl Default for InsertedPlacement {
    fn default() -> Self {
        Self {
            center_angstrom: default_inserted_center(),
            center_method: default_inserted_center_method(),
            orientation: default_inserted_orientation(),
            rotate_degrees_xyz: [0.0, 0.0, 0.0],
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ComponentFootprint {
    #[serde(default)]
    pub center_angstrom: Option<[f32; 2]>,
    #[serde(default)]
    pub radius_angstrom: Option<f32>,
    #[serde(default = "default_protein_footprint_buffer")]
    #[schemars(default = "default_protein_footprint_buffer")]
    pub buffer_angstrom: f32,
}
