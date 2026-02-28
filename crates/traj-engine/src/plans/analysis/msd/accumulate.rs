pub(super) fn accumulate_msd(
    acc: &mut [f64],
    lag_idx: usize,
    cols: usize,
    axis: Option<[f64; 3]>,
    n_groups: usize,
    type_ids: &[usize],
    inv_type_counts: &[f64],
    inv_n_groups: f64,
    cur: &[f32],
    old: &[f32],
) {
    let n_types = inv_type_counts.len();
    if n_types == 0 || n_groups == 0 {
        return;
    }
    let base = lag_idx * cols;
    let block = n_types + 1;
    let total_offset = base + n_types;

    // Dominant benchmark path: atom grouping with a single type.
    if n_types == 1 {
        if let Some(a) = axis {
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sz = 0.0f64;
            let mut sa = 0.0f64;
            let mut i = 0usize;
            while i < n_groups * 3 {
                let dx = (cur[i] - old[i]) as f64;
                let dy = (cur[i + 1] - old[i + 1]) as f64;
                let dz = (cur[i + 2] - old[i + 2]) as f64;
                let mx = dx * dx;
                let my = dy * dy;
                let mz = dz * dz;
                sx += mx;
                sy += my;
                sz += mz;
                let proj = dx * a[0] + dy * a[1] + dz * a[2];
                sa += proj * proj;
                i += 3;
            }
            let x = sx * inv_n_groups;
            let y = sy * inv_n_groups;
            let z = sz * inv_n_groups;
            let axis_v = sa * inv_n_groups;
            let tot = x + y + z;
            acc[base] += x;
            acc[base + block] += y;
            acc[base + 2 * block] += z;
            acc[base + 3 * block] += axis_v;
            acc[base + 4 * block] += tot;
            acc[total_offset] += x;
            acc[total_offset + block] += y;
            acc[total_offset + 2 * block] += z;
            acc[total_offset + 3 * block] += axis_v;
            acc[total_offset + 4 * block] += tot;
        } else {
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            let mut sz = 0.0f64;
            let mut i = 0usize;
            while i < n_groups * 3 {
                let dx = (cur[i] - old[i]) as f64;
                let dy = (cur[i + 1] - old[i + 1]) as f64;
                let dz = (cur[i + 2] - old[i + 2]) as f64;
                sx += dx * dx;
                sy += dy * dy;
                sz += dz * dz;
                i += 3;
            }
            let x = sx * inv_n_groups;
            let y = sy * inv_n_groups;
            let z = sz * inv_n_groups;
            let tot = x + y + z;
            acc[base] += x;
            acc[base + block] += y;
            acc[base + 2 * block] += z;
            acc[base + 3 * block] += tot;
            acc[total_offset] += x;
            acc[total_offset + block] += y;
            acc[total_offset + 2 * block] += z;
            acc[total_offset + 3 * block] += tot;
        }
        return;
    }

    if let Some(a) = axis {
        for g in 0..n_groups {
            let idx = g * 3;
            let dx = (cur[idx] - old[idx]) as f64;
            let dy = (cur[idx + 1] - old[idx + 1]) as f64;
            let dz = (cur[idx + 2] - old[idx + 2]) as f64;
            let msd_x = dx * dx;
            let msd_y = dy * dy;
            let msd_z = dz * dz;
            let msd_tot = msd_x + msd_y + msd_z;
            let proj = dx * a[0] + dy * a[1] + dz * a[2];
            let msd_axis = proj * proj;
            let type_id = type_ids[g];
            let inv_type = inv_type_counts[type_id];
            let offset = base + type_id;

            acc[offset] += msd_x * inv_type;
            acc[offset + block] += msd_y * inv_type;
            acc[offset + 2 * block] += msd_z * inv_type;
            acc[offset + 3 * block] += msd_axis * inv_type;
            acc[offset + 4 * block] += msd_tot * inv_type;

            acc[total_offset] += msd_x * inv_n_groups;
            acc[total_offset + block] += msd_y * inv_n_groups;
            acc[total_offset + 2 * block] += msd_z * inv_n_groups;
            acc[total_offset + 3 * block] += msd_axis * inv_n_groups;
            acc[total_offset + 4 * block] += msd_tot * inv_n_groups;
        }
    } else {
        for g in 0..n_groups {
            let idx = g * 3;
            let dx = (cur[idx] - old[idx]) as f64;
            let dy = (cur[idx + 1] - old[idx + 1]) as f64;
            let dz = (cur[idx + 2] - old[idx + 2]) as f64;
            let msd_x = dx * dx;
            let msd_y = dy * dy;
            let msd_z = dz * dz;
            let msd_tot = msd_x + msd_y + msd_z;
            let type_id = type_ids[g];
            let inv_type = inv_type_counts[type_id];
            let offset = base + type_id;

            acc[offset] += msd_x * inv_type;
            acc[offset + block] += msd_y * inv_type;
            acc[offset + 2 * block] += msd_z * inv_type;
            acc[offset + 3 * block] += msd_tot * inv_type;

            acc[total_offset] += msd_x * inv_n_groups;
            acc[total_offset + block] += msd_y * inv_n_groups;
            acc[total_offset + 2 * block] += msd_z * inv_n_groups;
            acc[total_offset + 3 * block] += msd_tot * inv_n_groups;
        }
    }
}
