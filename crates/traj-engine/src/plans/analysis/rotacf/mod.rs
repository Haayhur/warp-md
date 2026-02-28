mod accumulate;
mod builder;
mod execute;
mod fft;
mod utils;

use crate::correlators::multi_tau::MultiTauBuffer;
use crate::correlators::ring::RingBuffer;
use crate::correlators::{LagMode, LagSettings};
use crate::plans::analysis::grouping::{GroupBy, GroupMap};
use crate::plans::analysis::msd::{DtDecimation, FrameDecimation, TimeBinning};
use traj_core::selection::Selection;

#[cfg(feature = "cuda")]
use traj_gpu::{GpuAnchors, GpuContext};

#[derive(Clone)]
pub enum OrientationSpec {
    PlaneIndices([usize; 3]),
    VectorIndices([usize; 2]),
    PlaneSelections([Selection; 3]),
    VectorSelections([Selection; 2]),
}

#[derive(Clone, Copy)]
enum OrientationKind {
    Plane,
    Vector,
}

pub struct RotAcfPlan {
    selection: Selection,
    group_by: GroupBy,
    group_types: Option<Vec<usize>>,
    orientation: OrientationSpec,
    orientation_kind: OrientationKind,
    length_scale: f64,
    frame_decimation: Option<FrameDecimation>,
    dt_decimation: Option<DtDecimation>,
    time_binning: TimeBinning,
    p2_legendre: bool,
    lag: LagSettings,
    frames_hint: Option<usize>,
    resolved_mode: LagMode,
    groups: Option<GroupMap>,
    anchors: Option<Vec<[usize; 3]>>,
    type_ids: Vec<usize>,
    type_counts: Vec<usize>,
    n_groups: usize,
    dt0: Option<f64>,
    uniform_time: bool,
    last_time: Option<f64>,
    frame_index: usize,
    samples_seen: usize,
    sample_f32: Vec<f32>,
    lags: Vec<usize>,
    acc: Vec<f64>,
    multi_tau: Option<MultiTauBuffer>,
    ring: Option<RingBuffer>,
    series: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<RotAcfGpuState>,
}

#[cfg(feature = "cuda")]
struct RotAcfGpuState {
    ctx: GpuContext,
    anchors: GpuAnchors,
    kind: OrientationKind,
}
