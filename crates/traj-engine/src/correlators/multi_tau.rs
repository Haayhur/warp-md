use std::cmp::min;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecimationMode {
    Average,
    Latest,
}

#[derive(Debug, Clone)]
pub struct MultiTauBuffer {
    m: usize,
    decimation: usize,
    decimation_mode: DecimationMode,
    n_streams: usize,
    dim: usize,
    max_levels: usize,
    levels: usize,
    head: Vec<usize>,
    count: Vec<usize>,
    buffers: Vec<Vec<f32>>,
    scratch: Vec<Vec<f32>>,
    out_index: Vec<Vec<Option<usize>>>,
    out_lags: Vec<usize>,
    n_pairs: Vec<u64>,
}

impl MultiTauBuffer {
    pub fn new(n_streams: usize, dim: usize, m: usize, max_levels: usize) -> Self {
        Self::new_with_mode(n_streams, dim, m, max_levels, DecimationMode::Average)
    }

    pub fn new_with_mode(
        n_streams: usize,
        dim: usize,
        m: usize,
        max_levels: usize,
        decimation_mode: DecimationMode,
    ) -> Self {
        let mut buf = Self {
            m: m.max(2),
            decimation: 2,
            decimation_mode,
            n_streams,
            dim,
            max_levels: max_levels.max(1),
            levels: 0,
            head: Vec::new(),
            count: Vec::new(),
            buffers: Vec::new(),
            scratch: Vec::new(),
            out_index: Vec::new(),
            out_lags: Vec::new(),
            n_pairs: Vec::new(),
        };
        buf.add_level();
        buf
    }

    pub fn n_streams(&self) -> usize {
        self.n_streams
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn out_lags(&self) -> &[usize] {
        &self.out_lags
    }

    pub fn n_pairs(&self) -> &[u64] {
        &self.n_pairs
    }

    pub fn out_len(&self) -> usize {
        self.out_lags.len()
    }

    pub fn update<F>(&mut self, samples: &[f32], mut on_lag: F)
    where
        F: FnMut(usize, &[f32], &[f32]),
    {
        if samples.len() != self.n_streams * self.dim {
            return;
        }
        self.update_level(0, samples, &mut on_lag);
    }

    fn add_level(&mut self) {
        if self.levels >= self.max_levels {
            return;
        }
        let level = self.levels;
        let stride = self.n_streams * self.dim;
        self.buffers.push(vec![0.0f32; self.m * stride]);
        self.scratch.push(vec![0.0f32; stride]);
        self.head.push(0);
        self.count.push(0);
        self.out_index.push(vec![None; self.m]);

        let start = if level == 0 { 1 } else { self.m / 2 };
        let factor = 1usize << level;
        for lag in start..self.m {
            let out_idx = self.out_lags.len();
            self.out_lags.push(lag * factor);
            self.n_pairs.push(0);
            self.out_index[level][lag] = Some(out_idx);
        }
        self.levels += 1;
    }

    fn update_level<F>(&mut self, level: usize, samples: &[f32], on_lag: &mut F)
    where
        F: FnMut(usize, &[f32], &[f32]),
    {
        if level >= self.levels {
            return;
        }
        let m = self.m;
        let stride = self.n_streams * self.dim;
        let new_head = (self.head[level] + 1) % m;
        self.head[level] = new_head;
        self.count[level] += 1;
        let count = self.count[level];
        {
            let buf = &mut self.buffers[level];
            let dst = &mut buf[new_head * stride..(new_head + 1) * stride];
            dst.copy_from_slice(samples);
            let max_lag = min(m - 1, count - 1);

            for lag in 0..=max_lag {
                if let Some(out_idx) = self.out_index[level][lag] {
                    let idx_old = (new_head + m - lag) % m;
                    let old = &buf[idx_old * stride..(idx_old + 1) * stride];
                    on_lag(out_idx, samples, old);
                    self.n_pairs[out_idx] += 1;
                }
            }
        }

        if level + 1 >= self.max_levels {
            return;
        }
        if count % self.decimation != 0 {
            return;
        }

        let mut scratch_vec = std::mem::take(&mut self.scratch[level]);
        if scratch_vec.len() != stride {
            scratch_vec.resize(stride, 0.0);
        }
        {
            let prev_idx = (new_head + m - 1) % m;
            let buf = &self.buffers[level];
            let prev = &buf[prev_idx * stride..(prev_idx + 1) * stride];
            match self.decimation_mode {
                DecimationMode::Average => {
                    for i in 0..stride {
                        scratch_vec[i] = 0.5 * (samples[i] + prev[i]);
                    }
                }
                DecimationMode::Latest => {
                    scratch_vec.copy_from_slice(samples);
                }
            }
        }
        if level + 1 == self.levels {
            self.add_level();
        }
        self.update_level(level + 1, &scratch_vec, on_lag);
        self.scratch[level] = scratch_vec;
    }
}
