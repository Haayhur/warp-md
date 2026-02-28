mod accumulate;
mod builder;
mod execute;
mod fft;
mod utils;

use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::{LagMode, LagSettings};
use crate::plans::analysis::grouping::{GroupBy, GroupMap};
use traj_core::selection::Selection;

#[cfg(feature = "cuda")]
use traj_gpu::{GpuBufferF32, GpuContext, GpuGroups};

#[derive(Clone, Copy)]
pub struct FrameDecimation {
    pub start: usize,
    pub stride: usize,
}

#[derive(Clone, Copy)]
pub struct DtDecimation {
    pub cut1: usize,
    pub stride1: usize,
    pub cut2: usize,
    pub stride2: usize,
}

#[derive(Clone, Copy)]
pub struct TimeBinning {
    pub eps_num: f64,
    pub eps_add: f64,
}

impl Default for TimeBinning {
    fn default() -> Self {
        Self {
            eps_num: 1.0e-5,
            eps_add: 1.0e-4,
        }
    }
}

pub struct MsdPlan {
    selection: Selection,
    group_by: GroupBy,
    group_types: Option<Vec<usize>>,
    axis: Option<[f64; 3]>,
    length_scale: f64,
    frame_decimation: Option<FrameDecimation>,
    dt_decimation: Option<DtDecimation>,
    time_binning: TimeBinning,
    lag: LagSettings,
    frames_hint: Option<usize>,
    resolved_mode: LagMode,
    io_selection: Option<Vec<u32>>,
    atom_io_fastpath: bool,
    groups: Option<GroupMap>,
    type_ids: Vec<usize>,
    type_counts: Vec<usize>,
    inv_type_counts: Vec<f64>,
    inv_n_groups: f64,
    n_groups: usize,
    n_atoms: usize,
    masses: Vec<f64>,
    group_inv_mass: Vec<f64>,
    dt0: Option<f64>,
    uniform_time: bool,
    last_time: Option<f64>,
    frame_index: usize,
    samples_seen: usize,
    last_wrapped: Vec<[f64; 3]>,
    wrapped_curr: Vec<[f64; 3]>,
    unwrap_prev: Vec<[f64; 3]>,
    unwrap_curr: Vec<[f64; 3]>,
    sample_f32: Vec<f32>,
    lags: Vec<usize>,
    acc: Vec<f64>,
    multi_tau: Option<MultiTauBuffer>,
    ring: Option<RingBuffer>,
    series: Vec<f32>,
    // Optional perf counters enabled by WARP_MD_TRANSPORT_PROFILE=1.
    profile_transport: bool,
    perf_chunks: usize,
    perf_frames: usize,
    perf_read_ns: u64,
    perf_unwrap_ns: u64,
    perf_accum_ns: u64,
    perf_finalize_ns: u64,
    #[cfg(feature = "cuda")]
    gpu: Option<MsdGpuState>,
}

#[cfg(feature = "cuda")]
struct MsdGpuState {
    ctx: GpuContext,
    groups: GpuGroups,
    masses: GpuBufferF32,
}
