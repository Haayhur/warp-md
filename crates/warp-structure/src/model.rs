use serde::{Deserialize, Serialize};
use traj_core::geometry::Vec3;

use crate::error::{StructureError, StructureResult};

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

impl OutputSpec {
    pub const SUPPORTED_FORMATS: &'static [&'static str] = &[
        "pdb",
        "pdb-strict",
        "brk",
        "ent",
        "pqr",
        "xyz",
        "pdbx",
        "cif",
        "mmcif",
        "gro",
        "g96",
        "gromos96",
        "lammps",
        "lammps-data",
        "lmp",
        "mol2",
        "crd",
        "inpcrd",
        "rst",
        "rst7",
    ];

    pub fn infer_format_from_path(path: &str) -> String {
        std::path::Path::new(path)
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("pdb")
            .to_ascii_lowercase()
    }

    pub fn resolved_format(&self) -> String {
        if self.format.trim().is_empty() {
            Self::infer_format_from_path(&self.path)
        } else {
            self.format.to_ascii_lowercase()
        }
    }

    pub fn default_scale_for_format(format: &str) -> f32 {
        match format {
            "gro" | "g96" | "gromos96" => 0.1,
            _ => 1.0,
        }
    }

    pub fn validate(&self) -> StructureResult<()> {
        if self.path.trim().is_empty() {
            return Err(StructureError::Invalid(
                "output path cannot be empty".into(),
            ));
        }
        let format = self.resolved_format();
        if !Self::SUPPORTED_FORMATS.contains(&format.as_str()) {
            return Err(StructureError::Invalid(format!(
                "unsupported output format: {format}"
            )));
        }
        if let Some(scale) = self.scale {
            if scale <= 0.0 {
                return Err(StructureError::Invalid(
                    "output scale must be positive".into(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::OutputSpec;

    #[test]
    fn output_spec_infers_format_from_path() {
        let spec = OutputSpec {
            path: "system.g96".into(),
            format: String::new(),
            scale: None,
        };
        assert_eq!(spec.resolved_format(), "g96");
    }

    #[test]
    fn output_spec_accepts_structure_owned_formats() {
        let spec = OutputSpec {
            path: "system.pqr".into(),
            format: "pqr".into(),
            scale: Some(1.0),
        };
        assert!(spec.validate().is_ok());
    }

    #[test]
    fn output_spec_rejects_unknown_format() {
        let spec = OutputSpec {
            path: "system.out".into(),
            format: "custom".into(),
            scale: None,
        };
        let error = spec.validate().unwrap_err();
        assert!(error.to_string().contains("unsupported output format"));
    }
}
