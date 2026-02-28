use nalgebra::Matrix3;

#[cfg(feature = "cuda")]
use nalgebra::Vector3;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::ReferenceMode;

include!("align_part1.rs");
include!("align_part2.rs");
