use std::collections::{HashMap, HashSet};

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use super::geometry_math::*;
use crate::executor::{Device, Plan, PlanOutput};
#[cfg(feature = "cuda")]
use crate::plans::PbcMode;
use crate::ImagePlan;

include!("pbc/image_correction.rs");
include!("pbc/cell_geometry.rs");
