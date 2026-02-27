use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;

pub(super) fn validate_water_vectors(
    oxygen_indices: &[u32],
    hydrogen1_indices: &[u32],
    hydrogen2_indices: &[u32],
    orientation_valid: &[u8],
) -> TrajResult<()> {
    let n_waters = oxygen_indices.len();
    if hydrogen1_indices.len() != n_waters
        || hydrogen2_indices.len() != n_waters
        || orientation_valid.len() != n_waters
    {
        return Err(TrajError::Mismatch(
            "gist water index vectors must have identical length".into(),
        ));
    }
    Ok(())
}

pub(super) fn sorted_unique_indices(mut idx: Vec<usize>) -> Vec<usize> {
    idx.sort_unstable();
    idx.dedup();
    idx
}

pub(super) fn keep_frame_internal(
    max_frames: Option<usize>,
    accepted_frames: usize,
    filter: Option<&Vec<usize>>,
    filter_pos: &mut usize,
    abs_frame: usize,
) -> bool {
    if let Some(max_frames) = max_frames {
        if accepted_frames >= max_frames {
            return false;
        }
    }
    let Some(filter) = filter else {
        return true;
    };
    while *filter_pos < filter.len() && filter[*filter_pos] < abs_frame {
        *filter_pos += 1;
    }
    if *filter_pos >= filter.len() {
        return false;
    }
    if filter[*filter_pos] == abs_frame {
        *filter_pos += 1;
        return true;
    }
    false
}

pub(super) fn dims_from_bounds(
    min_xyz: [f64; 3],
    max_xyz: [f64; 3],
    padding: f64,
    spacing: f64,
) -> [usize; 3] {
    let span_x = (max_xyz[0] - min_xyz[0]) + 2.0 * padding;
    let span_y = (max_xyz[1] - min_xyz[1]) + 2.0 * padding;
    let span_z = (max_xyz[2] - min_xyz[2]) + 2.0 * padding;
    [
        ((span_x / spacing).ceil().max(1.0)) as usize,
        ((span_y / spacing).ceil().max(1.0)) as usize,
        ((span_z / spacing).ceil().max(1.0)) as usize,
    ]
}

pub(super) fn mean_center_all_atoms(
    chunk: &FrameChunk,
    frame: usize,
    length_scale: f64,
) -> TrajResult<[f64; 3]> {
    let n_atoms = chunk.n_atoms;
    if n_atoms == 0 {
        return Err(TrajError::Mismatch(
            "gist requires at least one atom".into(),
        ));
    }
    let base = frame * n_atoms;
    let mut center = [0.0f64; 3];
    for atom_idx in 0..n_atoms {
        let p = chunk.coords[base + atom_idx];
        center[0] += p[0] as f64 * length_scale;
        center[1] += p[1] as f64 * length_scale;
        center[2] += p[2] as f64 * length_scale;
    }
    center[0] /= n_atoms as f64;
    center[1] /= n_atoms as f64;
    center[2] /= n_atoms as f64;
    Ok(center)
}

pub(super) fn mean_center_indices(
    chunk: &FrameChunk,
    frame: usize,
    length_scale: f64,
    indices: &[u32],
) -> TrajResult<[f64; 3]> {
    if indices.is_empty() {
        return mean_center_all_atoms(chunk, frame, length_scale);
    }
    let n_atoms = chunk.n_atoms;
    let base = frame * n_atoms;
    let mut center = [0.0f64; 3];
    for &idx in indices.iter() {
        let atom_idx = idx as usize;
        if atom_idx >= n_atoms {
            return Err(TrajError::Mismatch(
                "gist solute selection index out of bounds".into(),
            ));
        }
        let p = chunk.coords[base + atom_idx];
        center[0] += p[0] as f64 * length_scale;
        center[1] += p[1] as f64 * length_scale;
        center[2] += p[2] as f64 * length_scale;
    }
    let denom = indices.len() as f64;
    center[0] /= denom;
    center[1] /= denom;
    center[2] /= denom;
    Ok(center)
}

pub(super) fn voxel_flat(
    pos: [f64; 3],
    origin: [f64; 3],
    spacing: f64,
    dims: [usize; 3],
) -> Option<usize> {
    let fx = (pos[0] - origin[0]) / spacing;
    let fy = (pos[1] - origin[1]) / spacing;
    let fz = (pos[2] - origin[2]) / spacing;
    if fx < 0.0 || fy < 0.0 || fz < 0.0 {
        return None;
    }
    let ix = fx.floor() as usize;
    let iy = fy.floor() as usize;
    let iz = fz.floor() as usize;
    if ix >= dims[0] || iy >= dims[1] || iz >= dims[2] {
        return None;
    }
    Some(ix + dims[0] * (iy + dims[1] * iz))
}

pub(super) fn orientation_bin(hvec: [f64; 3], rvec: [f64; 3], n_bins: usize) -> Option<usize> {
    let hnorm = (hvec[0] * hvec[0] + hvec[1] * hvec[1] + hvec[2] * hvec[2]).sqrt();
    let rnorm = (rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2]).sqrt();
    if hnorm <= 0.0 || rnorm <= 0.0 {
        return None;
    }
    let mut cos_t = (hvec[0] * rvec[0] + hvec[1] * rvec[1] + hvec[2] * rvec[2]) / (hnorm * rnorm);
    cos_t = cos_t.clamp(-1.0, 1.0);
    let mut bin = (((cos_t + 1.0) * 0.5) * n_bins as f64).floor() as usize;
    if bin >= n_bins {
        bin = n_bins - 1;
    }
    Some(bin)
}

pub(super) fn pair_key(a: usize, b: usize) -> u64 {
    let lo = a.min(b) as u64;
    let hi = a.max(b) as u64;
    (lo << 32) | hi
}
