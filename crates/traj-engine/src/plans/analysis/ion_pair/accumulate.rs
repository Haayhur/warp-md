pub(super) fn accumulate_ion_pair(
    acc: &mut [f64],
    lag_idx: usize,
    cat_indices: &[usize],
    ani_indices: &[usize],
    cur: &[f32],
    old: &[f32],
) {
    let base = lag_idx * 6;
    for &cat in cat_indices {
        let idx = cat * 3;
        let pair_cur = cur[idx].to_bits();
        let pair_old = old[idx].to_bits();
        if pair_cur == pair_old {
            acc[base + 1] += 1.0;
        }
        let hash_cur = ((cur[idx + 2].to_bits() as u64) << 32) | cur[idx + 1].to_bits() as u64;
        let hash_old = ((old[idx + 2].to_bits() as u64) << 32) | old[idx + 1].to_bits() as u64;
        if hash_cur == hash_old {
            acc[base + 4] += 1.0;
        }
    }
    for &ani in ani_indices {
        let idx = ani * 3;
        let pair_cur = cur[idx].to_bits();
        let pair_old = old[idx].to_bits();
        if pair_cur == pair_old {
            acc[base + 2] += 1.0;
        }
        let hash_cur = ((cur[idx + 2].to_bits() as u64) << 32) | cur[idx + 1].to_bits() as u64;
        let hash_old = ((old[idx + 2].to_bits() as u64) << 32) | old[idx + 1].to_bits() as u64;
        if hash_cur == hash_old {
            acc[base + 5] += 1.0;
        }
    }
}
