use nalgebra::{Matrix3, Vector3};

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};
use crate::plans::{PbcMode, ReferenceMode};

include!("rmsd_part1.rs");
include!("rmsd_part2.rs");
include!("rmsd_part3.rs");
