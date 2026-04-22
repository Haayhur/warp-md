use crate::plans::{PbcMode, ReferenceMode};
use traj_core::selection::Selection;

#[cfg(feature = "cuda")]
use traj_gpu::{GpuReference, GpuSelection};

#[cfg(feature = "cuda")]
pub(crate) struct RmsdGpuState {
    pub selection: GpuSelection,
    pub reference: GpuReference,
}

pub struct RmsdPlan {
    pub(crate) selection: Selection,
    pub(crate) selection_usize: Vec<usize>,
    pub(crate) dense_selection_usize: Vec<usize>,
    pub(crate) use_selected_input: bool,
    pub(crate) align: bool,
    pub(crate) reference_mode: ReferenceMode,
    pub(crate) reference: Option<Vec<[f32; 4]>>,
    pub(crate) results: Vec<f32>,

    #[cfg(feature = "cuda")]
    pub(crate) gpu: Option<RmsdGpuState>,
}

pub struct SymmRmsdPlan {
    pub(crate) inner: RmsdPlan,
}

#[derive(Debug, Clone, Copy)]

pub enum PairwiseMetric {
    Rms,
    Nofit,
    Dme,
}

pub struct PairwiseRmsdPlan {
    pub(crate) selection: Selection,
    pub(crate) metric: PairwiseMetric,
    pub(crate) pbc: PbcMode,
    pub(crate) use_selected_input: bool,
    pub(crate) frames: Vec<Vec<[f32; 4]>>,
    pub(crate) boxes: Vec<Option<(f64, f64, f64)>>,
}

pub struct DistanceRmsdPlan {
    pub(crate) selection: Selection,
    pub(crate) reference_mode: ReferenceMode,
    pub(crate) pbc: PbcMode,
    pub(crate) reference_dists: Option<Vec<f64>>,
    pub(crate) results: Vec<f32>,
}
