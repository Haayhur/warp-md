use traj_engine::correlators::multi_tau::{DecimationMode, MultiTauBuffer};
use traj_engine::correlators::ring::RingBuffer;

#[test]
fn multi_tau_counts_match_internal_pairs() {
    let mut buf = MultiTauBuffer::new(1, 1, 4, 3);
    let mut counts = Vec::new();
    for t in 0..12 {
        let sample = [t as f32];
        buf.update(&sample, |idx, _, _| {
            if idx >= counts.len() {
                counts.resize(idx + 1, 0);
            }
            counts[idx] += 1;
        });
    }
    if counts.len() < buf.n_pairs().len() {
        counts.resize(buf.n_pairs().len(), 0);
    }
    assert_eq!(counts, buf.n_pairs());
    assert_eq!(buf.out_lags(), &[1, 2, 3, 4, 6, 8, 12]);
}

#[test]
fn multi_tau_latest_decimation_preserves_samples() {
    let mut buf = MultiTauBuffer::new_with_mode(1, 1, 4, 2, DecimationMode::Latest);
    let mut seen = Vec::new();
    for t in 0..6 {
        let sample = [t as f32];
        buf.update(&sample, |idx, cur, old| {
            if idx == 1 {
                seen.push((cur[0] as i32, old[0] as i32));
            }
        });
    }
    assert!(!seen.is_empty());
}

#[test]
fn ring_counts_match_internal_pairs() {
    let mut buf = RingBuffer::new(1, 1, 4);
    let mut counts = vec![0u64; buf.max_lag() + 1];
    for t in 0..10 {
        let sample = [t as f32];
        buf.update(&sample, |lag, _, _| {
            counts[lag] += 1;
        });
    }
    assert_eq!(counts, buf.n_pairs());
}
