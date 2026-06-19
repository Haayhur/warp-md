use nalgebra::Matrix3;

#[cfg(feature = "cuda")]
use nalgebra::Vector3;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use super::geometry_math::*;
use crate::executor::{Device, Plan, PlanOutput, PlanRequirements, TrajectoryOutput};
use crate::plans::ReferenceMode;

#[path = "align/plan_shapes.rs"]
mod shapes;

pub use shapes::{AlignPlan, PrincipalAxesPlan, SuperposePlan, SuperposeTrajectoryPlan};
