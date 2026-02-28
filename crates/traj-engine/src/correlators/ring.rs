use std::cmp::min;

#[derive(Debug, Clone)]
pub struct RingBuffer {
    max_lag: usize,
    n_streams: usize,
    dim: usize,
    head: usize,
    count: usize,
    buffer: Vec<f32>,
    n_pairs: Vec<u64>,
}

impl RingBuffer {
    pub fn new(n_streams: usize, dim: usize, max_lag: usize) -> Self {
        let max_lag = max_lag.max(1);
        let stride = n_streams * dim;
        Self {
            max_lag,
            n_streams,
            dim,
            head: 0,
            count: 0,
            buffer: vec![0.0f32; (max_lag + 1) * stride],
            n_pairs: vec![0u64; max_lag + 1],
        }
    }

    pub fn max_lag(&self) -> usize {
        self.max_lag
    }

    pub fn n_pairs(&self) -> &[u64] {
        &self.n_pairs
    }

    pub fn update<F>(&mut self, samples: &[f32], mut on_lag: F)
    where
        F: FnMut(usize, &[f32], &[f32]),
    {
        let stride = self.n_streams * self.dim;
        if samples.len() != stride {
            return;
        }
        let size = self.max_lag + 1;
        self.head = (self.head + 1) % size;
        let dst = &mut self.buffer[self.head * stride..(self.head + 1) * stride];
        dst.copy_from_slice(samples);
        self.count += 1;

        let available = min(self.max_lag, self.count - 1);
        for lag in 1..=available {
            let idx_old = (self.head + size - lag) % size;
            let old = &self.buffer[idx_old * stride..(idx_old + 1) * stride];
            on_lag(lag, samples, old);
            self.n_pairs[lag] += 1;
        }
    }
}
