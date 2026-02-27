use traj_core::frame::Box3;

use crate::plans::analysis::grouping::GroupMap;

pub fn compute_group_com(
    coords: &[[f32; 4]],
    n_atoms: usize,
    groups: &GroupMap,
    masses: &[f64],
    length_scale: f64,
) -> Vec<[f64; 3]> {
    let n_frames = coords.len() / n_atoms;
    let n_groups = groups.n_groups();
    let mut com = vec![[0.0f64; 3]; n_frames * n_groups];
    for frame in 0..n_frames {
        let frame_offset = frame * n_atoms;
        for (g_idx, atoms) in groups.groups.iter().enumerate() {
            let mut sum = [0.0f64; 3];
            let mut mass_sum = 0.0f64;
            for &atom_idx in atoms {
                let p = coords[frame_offset + atom_idx];
                let m = masses[atom_idx];
                sum[0] += (p[0] as f64) * m;
                sum[1] += (p[1] as f64) * m;
                sum[2] += (p[2] as f64) * m;
                mass_sum += m;
            }
            let inv = if mass_sum > 0.0 { 1.0 / mass_sum } else { 0.0 };
            let idx = frame * n_groups + g_idx;
            com[idx][0] = sum[0] * inv * length_scale;
            com[idx][1] = sum[1] * inv * length_scale;
            com[idx][2] = sum[2] * inv * length_scale;
        }
    }
    com
}

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
            let mut diff = [
                com[idx][0] - com[prev][0],
                com[idx][1] - com[prev][1],
                com[idx][2] - com[prev][2],
            ];
            if let Some(box_l) = box_lengths(&boxes[frame]) {
                for k in 0..3 {
                    let l = box_l[k] * length_scale;
                    if l > 0.0 {
                        diff[k] -= (diff[k] / l).round() * l;
                    }
                }
            }
            out[idx][0] = out[prev][0] + diff[0];
            out[idx][1] = out[prev][1] + diff[1];
            out[idx][2] = out[prev][2] + diff[2];
        }
    }
    out
}

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

pub fn anchors_to_u32(anchors: &[[usize; 3]]) -> Vec<u32> {
    let mut out = Vec::with_capacity(anchors.len() * 3);
    for anchor in anchors {
        out.push(anchor[0] as u32);
        out.push(anchor[1] as u32);
        out.push(anchor[2] as u32);
    }
    out
}

fn box_lengths(box_: &Box3) -> Option<[f64; 3]> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => Some([*lx as f64, *ly as f64, *lz as f64]),
        Box3::Triclinic { .. } => None,
        Box3::None => None,
    }
}
