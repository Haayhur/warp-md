pub(super) fn accumulate_rotacf(
    acc: &mut [f64],
    lag_idx: usize,
    cols: usize,
    n_groups: usize,
    type_ids: &[usize],
    type_counts: &[usize],
    cur: &[f32],
    old: &[f32],
) {
    let n_types = type_counts.len();
    let n_groups_f = n_groups as f64;
    let base = lag_idx * cols;
    for g in 0..n_groups {
        let idx = g * 3;
        let ux = cur[idx] as f64;
        let uy = cur[idx + 1] as f64;
        let uz = cur[idx + 2] as f64;
        let vx = old[idx] as f64;
        let vy = old[idx + 1] as f64;
        let vz = old[idx + 2] as f64;
        let dot = ux * vx + uy * vy + uz * vz;
        let dot2 = dot * dot;
        let type_id = type_ids[g];
        let type_count = type_counts[type_id] as f64;
        acc[base + type_id] += dot / type_count;
        acc[base + n_types] += dot / n_groups_f;
        acc[base + (n_types + 1) + type_id] += dot2 / type_count;
        acc[base + (n_types + 1) + n_types] += dot2 / n_groups_f;
    }
}
