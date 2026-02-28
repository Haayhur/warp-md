use std::collections::HashMap;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::pbc_utils;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

use super::common::{
    dims_from_bounds, keep_frame_internal, mean_center_all_atoms, mean_center_indices,
    orientation_bin, pair_key, sorted_unique_indices, validate_water_vectors, voxel_flat,
};
use super::scaling::finalize_counts_orientation;

#[cfg(feature = "cuda")]
use super::gpu::ensure_gist_gpu_hist_buffers;
#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuBufferU32, GpuContext, GpuCoords};

const COULOMB_CONST: f64 = 138.935456;

include!("direct_core.rs");
include!("direct_plan.rs");
