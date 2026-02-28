mod accumulate;
mod builder;
mod execute;
mod fft;
mod runtime;
mod utils;

use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::{LagMode, LagSettings};
use crate::plans::analysis::grouping::{GroupBy, GroupMap};
use crate::plans::analysis::msd::{DtDecimation, FrameDecimation, TimeBinning};
use traj_core::selection::Selection;

#[cfg(feature = "cuda")]
use traj_gpu::{GpuBufferF32, GpuContext, GpuGroups};

pub struct ConductivityPlan {
    selection: Selection,
    group_by: GroupBy,
    group_types: Option<Vec<usize>>,
    charges: Vec<f64>,
    temperature: f64,
    transference: bool,
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
    n_groups: usize,
    n_atoms: usize,
    masses: Vec<f64>,
    group_inv_mass: Vec<f64>,
    group_charge: Vec<f64>,
    charged_groups: Vec<usize>,
    dt0: Option<f64>,
    uniform_time: bool,
    last_time: Option<f64>,
    frame_index: usize,
    samples_seen: usize,
    vol_sum: f64,
    vol_count: usize,
    last_wrapped: Vec<[f64; 3]>,
    wrapped_curr: Vec<[f64; 3]>,
    unwrap_prev: Vec<[f64; 3]>,
    unwrap_curr: Vec<[f64; 3]>,
    sample_f32: Vec<f32>,
    sample_total_f32: [f32; 3],
    total_stream_fastpath: bool,
    lags: Vec<usize>,
    acc: Vec<f64>,
    multi_tau: Option<MultiTauBuffer>,
    ring: Option<RingBuffer>,
    series: Vec<f32>,
    fft_store_groups: bool,
    // Reusable scratch to avoid per-lag and per-frame allocations in hot paths.
    type_disp_sums: Vec<[f64; 3]>,
    fft_stream_sums: Vec<[f64; 3]>,
    // Optional perf counters enabled by WARP_MD_TRANSPORT_PROFILE=1.
    profile_transport: bool,
    perf_chunks: usize,
    perf_frames: usize,
    perf_read_ns: u64,
    perf_unwrap_ns: u64,
    perf_accum_ns: u64,
    perf_finalize_ns: u64,
    #[cfg(feature = "cuda")]
    gpu: Option<ConductivityGpuState>,
}

#[cfg(feature = "cuda")]
struct ConductivityGpuState {
    ctx: GpuContext,
    groups: GpuGroups,
    masses: GpuBufferF32,
}
