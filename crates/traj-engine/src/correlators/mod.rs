pub mod multi_tau;
pub mod ring;

pub use multi_tau::DecimationMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LagMode {
    Auto,
    MultiTau,
    Ring,
    Fft,
}

#[derive(Clone, Copy, Debug)]
pub struct LagSettings {
    pub mode: LagMode,
    pub max_lag: usize,
    pub memory_budget_bytes: usize,
    pub multi_tau_m: usize,
    pub multi_tau_max_levels: usize,
}

impl Default for LagSettings {
    fn default() -> Self {
        Self {
            mode: LagMode::Auto,
            max_lag: 100_000,
            memory_budget_bytes: 512 * 1024 * 1024,
            multi_tau_m: 16,
            multi_tau_max_levels: 20,
        }
    }
}

impl LagSettings {
    pub fn with_mode(mut self, mode: LagMode) -> Self {
        self.mode = mode;
        self
    }

    pub fn with_max_lag(mut self, max_lag: usize) -> Self {
        self.max_lag = max_lag;
        self
    }

    pub fn with_memory_budget_bytes(mut self, budget: usize) -> Self {
        self.memory_budget_bytes = budget;
        self
    }

    pub fn with_multi_tau_m(mut self, m: usize) -> Self {
        self.multi_tau_m = m.max(2);
        self
    }

    pub fn with_multi_tau_levels(mut self, levels: usize) -> Self {
        self.multi_tau_max_levels = levels.max(1);
        self
    }

    pub fn ring_max_lag(&self, n_streams: usize, dim: usize, bytes_per_value: usize) -> usize {
        if n_streams == 0 || dim == 0 || bytes_per_value == 0 {
            return 0;
        }
        let denom = n_streams
            .saturating_mul(dim)
            .saturating_mul(bytes_per_value);
        if denom == 0 {
            return 0;
        }
        self.memory_budget_bytes / denom
    }

    pub fn ring_max_lag_capped(
        &self,
        n_streams: usize,
        dim: usize,
        bytes_per_value: usize,
    ) -> usize {
        let budget_lag = self.ring_max_lag(n_streams, dim, bytes_per_value);
        self.max_lag.min(budget_lag).max(1)
    }

    pub fn fft_fits(
        &self,
        n_frames: usize,
        n_streams: usize,
        dim: usize,
        bytes_per_value: usize,
    ) -> bool {
        if n_frames == 0 || n_streams == 0 || dim == 0 || bytes_per_value == 0 {
            return false;
        }
        let needed = n_frames
            .saturating_mul(n_streams)
            .saturating_mul(dim)
            .saturating_mul(bytes_per_value);
        needed <= self.memory_budget_bytes
    }
}
