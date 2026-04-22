use traj_core::frame::Box3;
use traj_core::pbc_math::{apply_pbc, orthorhombic_lengths};

use crate::plans::analysis::grouping::GroupMap;

pub fn alloc_group_positions(n_groups: usize) -> Vec<[f64; 3]> {
    vec![[0.0f64; 3]; n_groups]
}

pub fn alloc_group_unwrap_buffers(
    n_groups: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>, Vec<[f64; 3]>) {
    (
        alloc_group_positions(n_groups),
        alloc_group_positions(n_groups),
        alloc_group_positions(n_groups),
        alloc_group_positions(n_groups),
    )
}

pub fn compute_group_inv_mass(groups: &GroupMap, masses: &[f64]) -> Vec<f64> {
    groups
        .groups
        .iter()
        .map(|atoms| {
            let mass_sum = atoms.iter().map(|&atom_idx| masses[atom_idx]).sum::<f64>();
            if mass_sum > 0.0 {
                1.0 / mass_sum
            } else {
                0.0
            }
        })
        .collect()
}

pub fn fill_frame_group_centers(
    coords: &[[f32; 4]],
    frame_offset: usize,
    groups: &[Vec<usize>],
    masses: &[f64],
    group_inv_mass: &[f64],
    length_scale: f64,
    out: &mut [[f64; 3]],
) {
    debug_assert_eq!(groups.len(), out.len());
    for (g_idx, (atoms, dst)) in groups.iter().zip(out.iter_mut()).enumerate() {
        let mut sum = [0.0f64; 3];
        for &atom_idx in atoms {
            let p = coords[frame_offset + atom_idx];
            let m = masses[atom_idx];
            sum[0] += (p[0] as f64) * m;
            sum[1] += (p[1] as f64) * m;
            sum[2] += (p[2] as f64) * m;
        }
        let scale = group_inv_mass.get(g_idx).copied().unwrap_or(0.0) * length_scale;
        dst[0] = sum[0] * scale;
        dst[1] = sum[1] * scale;
        dst[2] = sum[2] * scale;
    }
}

pub fn fill_frame_atom_positions(
    coords: &[[f32; 4]],
    frame_offset: usize,
    length_scale: f64,
    out: &mut [[f64; 3]],
) {
    for (g_idx, dst) in out.iter_mut().enumerate() {
        let p = coords[frame_offset + g_idx];
        dst[0] = (p[0] as f64) * length_scale;
        dst[1] = (p[1] as f64) * length_scale;
        dst[2] = (p[2] as f64) * length_scale;
    }
}

pub fn fill_frame_group_positions(
    coords: &[[f32; 4]],
    frame_offset: usize,
    groups: Option<&[Vec<usize>]>,
    masses: &[f64],
    group_inv_mass: &[f64],
    length_scale: f64,
    out: &mut [[f64; 3]],
) {
    if let Some(groups) = groups {
        fill_frame_group_centers(
            coords,
            frame_offset,
            groups,
            masses,
            group_inv_mass,
            length_scale,
            out,
        );
    } else {
        fill_frame_atom_positions(coords, frame_offset, length_scale, out);
    }
}

pub fn seed_unwrapped_groups(
    current: &[[f64; 3]],
    last_wrapped: &mut [[f64; 3]],
    unwrap_prev: &mut [[f64; 3]],
    unwrap_curr: &mut [[f64; 3]],
) {
    last_wrapped.copy_from_slice(current);
    unwrap_prev.copy_from_slice(current);
    unwrap_curr.copy_from_slice(current);
}

pub fn unwrap_frame_groups(
    current: &[[f64; 3]],
    last_wrapped: &mut [[f64; 3]],
    unwrap_prev: &mut [[f64; 3]],
    unwrap_curr: &mut [[f64; 3]],
    box_: &Box3,
    length_scale: f64,
) {
    let box_l = orthorhombic_lengths(box_)
        .map(|[lx, ly, lz]| [lx * length_scale, ly * length_scale, lz * length_scale]);
    for g in 0..current.len() {
        let curr = current[g];
        let prev = last_wrapped[g];
        let mut dx = curr[0] - prev[0];
        let mut dy = curr[1] - prev[1];
        let mut dz = curr[2] - prev[2];
        if let Some([lx, ly, lz]) = box_l {
            apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
        }
        unwrap_curr[g][0] = unwrap_prev[g][0] + dx;
        unwrap_curr[g][1] = unwrap_prev[g][1] + dy;
        unwrap_curr[g][2] = unwrap_prev[g][2] + dz;
    }
    unwrap_prev.copy_from_slice(unwrap_curr);
    last_wrapped.copy_from_slice(current);
}

pub fn compute_group_com(
    coords: &[[f32; 4]],
    n_atoms: usize,
    groups: &GroupMap,
    masses: &[f64],
    length_scale: f64,
) -> Vec<[f64; 3]> {
    let n_frames = coords.len() / n_atoms;
    let n_groups = groups.n_groups();
    let group_inv_mass = compute_group_inv_mass(groups, masses);
    let mut com = vec![[0.0f64; 3]; n_frames * n_groups];
    for frame in 0..n_frames {
        let frame_offset = frame * n_atoms;
        let frame_com = &mut com[frame * n_groups..(frame + 1) * n_groups];
        fill_frame_group_centers(
            coords,
            frame_offset,
            &groups.groups,
            masses,
            &group_inv_mass,
            length_scale,
            frame_com,
        );
    }
    com
}

#[cfg(test)]
pub fn unwrap_groups(
    com: &[[f64; 3]],
    boxes: &[Box3],
    n_groups: usize,
    length_scale: f64,
) -> Vec<[f64; 3]> {
    let n_frames = com.len() / n_groups;
    let mut out = vec![[0.0f64; 3]; n_frames * n_groups];
    for g in 0..n_groups {
        out[g] = com[g];
        for frame in 1..n_frames {
            let idx = frame * n_groups + g;
            let prev = (frame - 1) * n_groups + g;
            let mut dx = com[idx][0] - com[prev][0];
            let mut dy = com[idx][1] - com[prev][1];
            let mut dz = com[idx][2] - com[prev][2];
            if let Some([lx, ly, lz]) = orthorhombic_lengths(&boxes[frame]) {
                apply_pbc(
                    &mut dx,
                    &mut dy,
                    &mut dz,
                    lx * length_scale,
                    ly * length_scale,
                    lz * length_scale,
                );
            }
            out[idx][0] = out[prev][0] + dx;
            out[idx][1] = out[prev][1] + dy;
            out[idx][2] = out[prev][2] + dz;
        }
    }
    out
}

#[cfg(feature = "cuda")]
pub fn groups_to_csr(groups: &[Vec<usize>]) -> (Vec<u32>, Vec<u32>, usize) {
    let mut offsets = Vec::with_capacity(groups.len() + 1);
    let mut indices = Vec::new();
    let mut max_len = 0usize;
    offsets.push(0u32);
    for group in groups {
        max_len = max_len.max(group.len());
        for &idx in group {
            indices.push(idx as u32);
        }
        offsets.push(indices.len() as u32);
    }
    (offsets, indices, max_len)
}

#[cfg(feature = "cuda")]
pub fn anchors_to_u32(anchors: &[[usize; 3]]) -> Vec<u32> {
    let mut out = Vec::with_capacity(anchors.len() * 3);
    for anchor in anchors {
        out.push(anchor[0] as u32);
        out.push(anchor[1] as u32);
        out.push(anchor[2] as u32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plans::analysis::grouping::GroupMap;

    fn sample_groups() -> GroupMap {
        GroupMap {
            groups: vec![vec![0, 1], vec![2]],
            group_types: None,
        }
    }

    #[test]
    fn fill_frame_group_positions_uses_atom_fastpath_when_groups_absent() {
        let coords = vec![
            [1.0, 2.0, 3.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
            [7.0, 8.0, 9.0, 0.0],
        ];
        let mut out = alloc_group_positions(3);
        fill_frame_group_positions(&coords, 0, None, &[], &[], 0.5, &mut out);
        assert_eq!(out, vec![[0.5, 1.0, 1.5], [2.0, 2.5, 3.0], [3.5, 4.0, 4.5]]);
    }

    #[test]
    fn compute_group_com_matches_frame_center_helper() {
        let groups = sample_groups();
        let masses = vec![12.0, 1.0, 16.0];
        let coords = vec![
            [0.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0, 0.0],
            [11.0, 0.0, 0.0, 0.0],
        ];
        let group_inv_mass = compute_group_inv_mass(&groups, &masses);
        let all = compute_group_com(&coords, 3, &groups, &masses, 1.0);

        let mut frame0 = vec![[0.0; 3]; groups.n_groups()];
        fill_frame_group_centers(
            &coords,
            0,
            &groups.groups,
            &masses,
            &group_inv_mass,
            1.0,
            &mut frame0,
        );
        assert_eq!(frame0, all[..groups.n_groups()]);
        assert!((frame0[0][0] - (2.0 / 13.0)).abs() < 1.0e-12);
        assert_eq!(frame0[1], [10.0, 0.0, 0.0]);

        let mut frame1 = vec![[0.0; 3]; groups.n_groups()];
        fill_frame_group_centers(
            &coords,
            3,
            &groups.groups,
            &masses,
            &group_inv_mass,
            1.0,
            &mut frame1,
        );
        assert_eq!(frame1, all[groups.n_groups()..]);
        assert!((frame1[0][0] - (15.0 / 13.0)).abs() < 1.0e-12);
        assert_eq!(frame1[1], [11.0, 0.0, 0.0]);
    }

    #[test]
    fn incremental_unwrap_matches_batch_unwrap() {
        let n_groups = 2;
        let wrapped = vec![
            [9.5, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ];
        let boxes = vec![
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
        ];

        let batch = unwrap_groups(&wrapped, &boxes, n_groups, 1.0);

        let mut last_wrapped = vec![[0.0; 3]; n_groups];
        let mut unwrap_prev = vec![[0.0; 3]; n_groups];
        let mut unwrap_curr = vec![[0.0; 3]; n_groups];
        let mut incremental = wrapped[..n_groups].to_vec();
        seed_unwrapped_groups(
            &wrapped[..n_groups],
            &mut last_wrapped,
            &mut unwrap_prev,
            &mut unwrap_curr,
        );
        for frame in 1..boxes.len() {
            let start = frame * n_groups;
            let end = start + n_groups;
            unwrap_frame_groups(
                &wrapped[start..end],
                &mut last_wrapped,
                &mut unwrap_prev,
                &mut unwrap_curr,
                &boxes[frame],
                1.0,
            );
            incremental.extend_from_slice(&unwrap_curr);
        }

        assert_eq!(incremental, batch);
        assert_eq!(batch[2], [10.5, 0.0, 0.0]);
        assert_eq!(batch[4], [11.5, 0.0, 0.0]);
    }
}
