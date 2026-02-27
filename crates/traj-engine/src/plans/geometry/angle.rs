use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::{PbcMode, ReferenceMode};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferF32, GpuGroups, GpuReference, GpuSelection};

include!("angle_part1.rs");
include!("angle_part2.rs");
