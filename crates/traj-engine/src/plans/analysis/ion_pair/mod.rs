mod accumulate;
mod builder;
mod execute;
mod utils;

use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::{LagMode, LagSettings};
use crate::plans::analysis::grouping::{GroupBy, GroupMap};
use traj_core::selection::Selection;

#[cfg(feature = "cuda")]
use traj_gpu::{GpuBufferF32, GpuContext, GpuGroups};

pub struct IonPairCorrelationPlan {
    selection: Selection,
    group_by: GroupBy,
    group_types: Option<Vec<usize>>,
    cation_type: usize,
    anion_type: usize,
    rclust_cat: f64,
    rclust_ani: f64,
    max_cluster: usize,
    length_scale: f64,
    lag: LagSettings,
    frames_hint: Option<usize>,
    resolved_mode: LagMode,
    groups: Option<GroupMap>,
    type_ids: Vec<usize>,
    n_groups: usize,
    n_atoms: usize,
    masses: Vec<f64>,
    cat_indices: Vec<usize>,
    ani_indices: Vec<usize>,
    dt0: Option<f64>,
    uniform_time: bool,
    last_time: Option<f64>,
    frame_index: usize,
    samples_seen: usize,
    wrapped_curr: Vec<[f64; 3]>,
    sample_f32: Vec<f32>,
    pair_idx: Vec<u32>,
    cluster_hash: Vec<u64>,
    fft_com: Vec<f32>,
    fft_box: Vec<f32>,
    lags: Vec<usize>,
    acc: Vec<f64>,
    multi_tau: Option<MultiTauBuffer>,
    ring: Option<RingBuffer>,
    #[cfg(feature = "cuda")]
    gpu: Option<IonPairGpuState>,
}

#[cfg(feature = "cuda")]
struct IonPairGpuState {
    ctx: GpuContext,
    groups: GpuGroups,
    masses: GpuBufferF32,
}
