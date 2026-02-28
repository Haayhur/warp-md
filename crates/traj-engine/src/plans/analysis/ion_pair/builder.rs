use crate::correlators::{LagMode, LagSettings};
use crate::plans::analysis::grouping::GroupBy;
use traj_core::selection::Selection;

use super::IonPairCorrelationPlan;

impl IonPairCorrelationPlan {
    pub fn new(selection: Selection, group_by: GroupBy, rclust_cat: f64, rclust_ani: f64) -> Self {
        Self {
            selection,
            group_by,
            group_types: None,
            cation_type: 0,
            anion_type: 1,
            rclust_cat,
            rclust_ani,
            max_cluster: 10,
            length_scale: 1.0,
            lag: LagSettings::default(),
            frames_hint: None,
            resolved_mode: LagMode::Auto,
            groups: None,
            type_ids: Vec::new(),
            n_groups: 0,
            n_atoms: 0,
            masses: Vec::new(),
            cat_indices: Vec::new(),
            ani_indices: Vec::new(),
            dt0: None,
            uniform_time: true,
            last_time: None,
            frame_index: 0,
            samples_seen: 0,
            wrapped_curr: Vec::new(),
            sample_f32: Vec::new(),
            pair_idx: Vec::new(),
            cluster_hash: Vec::new(),
            fft_com: Vec::new(),
            fft_box: Vec::new(),
            lags: Vec::new(),
            acc: Vec::new(),
            multi_tau: None,
            ring: None,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_group_types(mut self, types: Vec<usize>) -> Self {
        self.group_types = Some(types);
        self
    }

    pub fn with_types(mut self, cation_type: usize, anion_type: usize) -> Self {
        self.cation_type = cation_type;
        self.anion_type = anion_type;
        self
    }

    pub fn with_max_cluster(mut self, max_cluster: usize) -> Self {
        self.max_cluster = max_cluster.max(1);
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
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
