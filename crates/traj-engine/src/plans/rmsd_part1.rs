#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuReference, GpuSelection};

pub struct RmsdPlan {
    selection: Selection,
    selection_usize: Vec<usize>,
    dense_selection_usize: Vec<usize>,
    use_selected_input: bool,
    align: bool,
    reference_mode: ReferenceMode,
    reference: Option<Vec<[f32; 4]>>,
    results: Vec<f32>,

    #[cfg(feature = "cuda")]
    gpu: Option<RmsdGpuState>,
}

pub struct SymmRmsdPlan {
    inner: RmsdPlan,
}

#[derive(Debug, Clone, Copy)]

pub enum PairwiseMetric {
    Rms,
    Nofit,
    Dme,
}

pub struct PairwiseRmsdPlan {
    selection: Selection,
    metric: PairwiseMetric,
    pbc: PbcMode,
    use_selected_input: bool,
    frames: Vec<Vec<[f32; 4]>>,
    boxes: Vec<Option<(f64, f64, f64)>>,
}

pub struct DistanceRmsdPlan {
    selection: Selection,
    reference_mode: ReferenceMode,
    pbc: PbcMode,
    reference_dists: Option<Vec<f64>>,
    results: Vec<f32>,
}
