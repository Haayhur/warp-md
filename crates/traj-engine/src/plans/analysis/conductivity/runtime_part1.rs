#[cfg(feature = "cuda")]
use crate::plans::analysis::common::groups_to_csr;
use crate::plans::analysis::grouping::GroupSpec;

#[cfg(feature = "cuda")]
use traj_gpu::coords_as_float4;

use super::accumulate::{
    accumulate_conductivity_streams_total, accumulate_conductivity_total,
    accumulate_conductivity_transference,
};
use super::fft::conductivity_fft;
use super::utils::{average_volume, box_lengths, build_conductivity_output, lag_allowed};
use super::ConductivityPlan;
