use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::system::System;

use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::LagMode;
use crate::executor::{Device, PlanOutput};

include!("runtime_part1.rs");
include!("runtime_part2.rs");
include!("runtime_part3.rs");
