use crate::correlators::{LagMode, LagSettings};
use crate::plans::analysis::grouping::GroupBy;
use crate::plans::analysis::msd::{DtDecimation, FrameDecimation, TimeBinning};
use traj_core::selection::Selection;

use super::{OrientationKind, OrientationSpec, RotAcfPlan};

impl RotAcfPlan {
    pub fn new(selection: Selection, group_by: GroupBy, orientation: OrientationSpec) -> Self {
        let kind = match orientation {
            OrientationSpec::PlaneIndices(_) | OrientationSpec::PlaneSelections(_) => {
                OrientationKind::Plane
            }
            OrientationSpec::VectorIndices(_) | OrientationSpec::VectorSelections(_) => {
                OrientationKind::Vector
            }
        };
        Self {
            selection,
            group_by,
            group_types: None,
            orientation,
            orientation_kind: kind,
            length_scale: 1.0,
            frame_decimation: None,
            dt_decimation: None,
            time_binning: TimeBinning::default(),
            p2_legendre: true,
            lag: LagSettings::default(),
            frames_hint: None,
            resolved_mode: LagMode::Auto,
            groups: None,
            anchors: None,
            type_ids: Vec::new(),
            type_counts: Vec::new(),
            n_groups: 0,
            dt0: None,
            uniform_time: true,
            last_time: None,
            frame_index: 0,
            samples_seen: 0,
            sample_f32: Vec::new(),
            lags: Vec::new(),
            acc: Vec::new(),
            multi_tau: None,
            ring: None,
            series: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }

    pub fn with_frame_decimation(mut self, dec: FrameDecimation) -> Self {
        self.frame_decimation = Some(dec);
        self
    }

    pub fn with_dt_decimation(mut self, dec: DtDecimation) -> Self {
        self.dt_decimation = Some(dec);
        self
    }

    pub fn with_time_binning(mut self, bin: TimeBinning) -> Self {
        self.time_binning = bin;
        self
    }

    pub fn with_p2_legendre(mut self, enabled: bool) -> Self {
        self.p2_legendre = enabled;
        self
    }

    pub fn with_group_types(mut self, types: Vec<usize>) -> Self {
        self.group_types = Some(types);
        self
    }

    pub fn with_lag_mode(mut self, mode: LagMode) -> Self {
        self.lag = self.lag.with_mode(mode);
        self
    }

    pub fn with_max_lag(mut self, max_lag: usize) -> Self {
        self.lag = self.lag.with_max_lag(max_lag);
        self
    }

    pub fn with_memory_budget_bytes(mut self, budget: usize) -> Self {
        self.lag = self.lag.with_memory_budget_bytes(budget);
        self
    }

    pub fn with_multi_tau_m(mut self, m: usize) -> Self {
        self.lag = self.lag.with_multi_tau_m(m);
        self
    }

    pub fn with_multi_tau_levels(mut self, levels: usize) -> Self {
        self.lag = self.lag.with_multi_tau_levels(levels);
        self
    }
}
