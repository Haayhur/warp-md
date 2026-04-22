#![forbid(unsafe_code)]

pub mod centers;
pub mod constraint_penalty;
pub mod constraints;
pub mod elements;
pub mod error;
pub mod frame;
pub mod geometry;
pub mod geometry_math;
pub mod inertia;
pub mod interner;
pub mod pbc;
pub mod pbc_math;
pub mod pdb_gro;
pub mod rng;
pub mod selection;
pub mod selection_expression;
pub mod spatial_hash;
pub mod spatial_hash_grid;
pub mod system;
pub mod vector_math;

pub use centers::{center_of_coords, center_of_selection};
pub use constraints::{ConstraintMode, ConstraintSpec, ShapeSpec};
pub use error::{TrajError, TrajResult};
pub use frame::{Box3, FrameChunk, FrameChunkBuilder};
pub use geometry::{center_of_geometry, Quaternion, Vec3};
pub use geometry_math::{angle_diff, angle_from_vectors, dihedral_from_vectors, rotate_about_axis};
pub use inertia::principal_axes_from_inertia;
pub use interner::StringInterner;
pub use pbc::PbcBox;
pub use pbc_math::{
    apply_pbc, apply_pbc_triclinic, box_diagonal_extents, box_lengths, cell_and_inv_from_box,
    minimum_image_delta, minimum_image_displacement, minimum_image_vector, orthorhombic_lengths,
};
pub use rng::{gaussian_pair, next_f64, next_u64};
pub use selection::Selection;
pub use selection_expression::{
    is_backbone_atom, is_protein_resname, is_sidechain_heavy_atom, parse_selection_expression,
    ResidRange, SelectionExpr, SelectionPredicate,
};
pub use spatial_hash::SpatialHash;
pub use spatial_hash_grid::{SpatialHashGrid, SpatialHashStats};
pub use system::{AtomTable, System};
pub use vector_math::{normalize_vec3, rotate_about_axis_vec3, rotate_from_to_vec3};
