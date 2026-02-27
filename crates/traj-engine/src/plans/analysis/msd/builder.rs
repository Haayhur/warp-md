use traj_core::selection::Selection;

use crate::correlators::{LagMode, LagSettings};
use crate::plans::analysis::grouping::GroupBy;

use super::{DtDecimation, FrameDecimation, MsdPlan, TimeBinning};

impl MsdPlan {
    pub fn new(selection: Selection, group_by: GroupBy) -> Self {
        Self {
            selection,
            group_by,
            group_types: None,
            axis: None,
            length_scale: 1.0,
            frame_decimation: None,
            dt_decimation: None,
            time_binning: TimeBinning::default(),
            lag: LagSettings::default(),
            frames_hint: None,
            resolved_mode: LagMode::Auto,
            io_selection: None,
            atom_io_fastpath: false,
            groups: None,
            type_ids: Vec::new(),
            type_counts: Vec::new(),
            inv_type_counts: Vec::new(),
            inv_n_groups: 0.0,
            n_groups: 0,
            n_atoms: 0,
            masses: Vec::new(),
            group_inv_mass: Vec::new(),
            dt0: None,
            uniform_time: true,
            last_time: None,
            frame_index: 0,
            samples_seen: 0,
            last_wrapped: Vec::new(),
            wrapped_curr: Vec::new(),
            unwrap_prev: Vec::new(),
            unwrap_curr: Vec::new(),
            sample_f32: Vec::new(),
            lags: Vec::new(),
            acc: Vec::new(),
            multi_tau: None,
            ring: None,
            series: Vec::new(),
            profile_transport: false,
            perf_chunks: 0,
            perf_frames: 0,
            perf_read_ns: 0,
            perf_unwrap_ns: 0,
            perf_accum_ns: 0,
            perf_finalize_ns: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_axis(mut self, axis: [f64; 3]) -> Self {
        let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
        if norm > 0.0 {
            self.axis = Some([axis[0] / norm, axis[1] / norm, axis[2] / norm]);
        } else {
            self.axis = Some(axis);
        }
        self
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
