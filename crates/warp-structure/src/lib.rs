pub mod error;
pub mod io;
pub mod model;
pub mod ndjson;

pub mod config {
    pub use crate::model::OutputSpec;
}

pub mod geometry {
    pub use traj_core::geometry::*;
}

pub mod pack {
    pub use crate::model::{AtomRecord, AtomRecordKind, PackOutput, PdbAtomMetadata};
}

pub use error::{StructureError, StructureResult};
pub use io::{
    read_gro_system, read_pdb_system, read_pdbqt_system, read_system_auto, system_from_molecule,
    MoleculeData, OutputWriteResult,
};
pub use model::{AtomRecord, AtomRecordKind, BoxVectors, OutputSpec, PackOutput, PdbAtomMetadata};
pub use traj_core::{
    center_of_geometry, normalize_vec3, rotate_about_axis_vec3, rotate_from_to_vec3, Quaternion,
    Vec3,
};
