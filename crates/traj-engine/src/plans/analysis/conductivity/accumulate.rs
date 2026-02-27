#[inline(always)]
pub(super) fn accumulate_conductivity_transference(
    acc: &mut [f64],
    lag_idx: usize,
    n_types: usize,
    type_ids: &[usize],
    group_charge: &[f64],
    charged_groups: &[usize],
    type_sums: &mut [[f64; 3]],
    cur: &[f32],
    old: &[f32],
) {
    let cols = n_types * n_types + 1;
    let base = lag_idx * cols;
    let mut sum_all = [0.0f64; 3];
    for sum in type_sums.iter_mut().take(n_types) {
        *sum = [0.0; 3];
    }
    for &g in charged_groups {
        let idx = g * 3;
        let q = group_charge[g];
        let dx = (cur[idx] - old[idx]) as f64 * q;
        let dy = (cur[idx + 1] - old[idx + 1]) as f64 * q;
        let dz = (cur[idx + 2] - old[idx + 2]) as f64 * q;
        sum_all[0] += dx;
        sum_all[1] += dy;
        sum_all[2] += dz;
        let t = type_ids[g];
        type_sums[t][0] += dx;
        type_sums[t][1] += dy;
        type_sums[t][2] += dz;
    }

    for i in 0..n_types {
        let v1 = type_sums[i];
        let dot = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
        acc[base + i + i * n_types] += dot;
        for j in (i + 1)..n_types {
            let v2 = type_sums[j];
            let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
            acc[base + j + i * n_types] += dot;
            acc[base + i + j * n_types] += dot;
        }
    }

    let dot = sum_all[0] * sum_all[0] + sum_all[1] * sum_all[1] + sum_all[2] * sum_all[2];
    acc[base + (cols - 1)] += dot;
}

#[inline(always)]
pub(super) fn accumulate_conductivity_total(
    acc: &mut [f64],
    lag_idx: usize,
    group_charge: &[f64],
    charged_groups: &[usize],
    cur: &[f32],
    old: &[f32],
) {
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sz = 0.0f64;
    let mut i = 0usize;
    while i + 3 < charged_groups.len() {
        let g0 = charged_groups[i];
        let g1 = charged_groups[i + 1];
        let g2 = charged_groups[i + 2];
        let g3 = charged_groups[i + 3];

        let b0 = g0 * 3;
        let b1 = g1 * 3;
        let b2 = g2 * 3;
        let b3 = g3 * 3;

        let q0 = group_charge[g0];
        let q1 = group_charge[g1];
        let q2 = group_charge[g2];
        let q3 = group_charge[g3];

        sx += (cur[b0] - old[b0]) as f64 * q0;
        sy += (cur[b0 + 1] - old[b0 + 1]) as f64 * q0;
        sz += (cur[b0 + 2] - old[b0 + 2]) as f64 * q0;

        sx += (cur[b1] - old[b1]) as f64 * q1;
        sy += (cur[b1 + 1] - old[b1 + 1]) as f64 * q1;
        sz += (cur[b1 + 2] - old[b1 + 2]) as f64 * q1;

        sx += (cur[b2] - old[b2]) as f64 * q2;
        sy += (cur[b2 + 1] - old[b2 + 1]) as f64 * q2;
        sz += (cur[b2 + 2] - old[b2 + 2]) as f64 * q2;

        sx += (cur[b3] - old[b3]) as f64 * q3;
        sy += (cur[b3 + 1] - old[b3 + 1]) as f64 * q3;
        sz += (cur[b3 + 2] - old[b3 + 2]) as f64 * q3;

        i += 4;
    }
    while i < charged_groups.len() {
        let g = charged_groups[i];
        let base = g * 3;
        let q = group_charge[g];
        sx += (cur[base] - old[base]) as f64 * q;
        sy += (cur[base + 1] - old[base + 1]) as f64 * q;
        sz += (cur[base + 2] - old[base + 2]) as f64 * q;
        i += 1;
    }
    acc[lag_idx] += sx * sx + sy * sy + sz * sz;
}

#[inline(always)]
pub(super) fn accumulate_conductivity_streams_total(
    acc: &mut [f64],
    lag_idx: usize,
    cur: &[f32],
    old: &[f32],
) {
    let vx = (cur[0] - old[0]) as f64;
    let vy = (cur[1] - old[1]) as f64;
    let vz = (cur[2] - old[2]) as f64;
    acc[lag_idx] += vx * vx + vy * vy + vz * vz;
}
