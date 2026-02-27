use nalgebra::{Matrix3, Vector3};

pub(crate) use traj_core::centers::{center_of_coords, center_of_selection};
#[cfg(feature = "cuda")]
use traj_core::error::TrajError;
use traj_core::error::TrajResult;
use traj_core::frame::{Box3, FrameChunk};
pub(crate) use traj_core::geometry_utils::{
    angle_diff, angle_from_vectors, dihedral_from_vectors, rotate_about_axis,
};
pub(crate) use traj_core::inertia::principal_axes_from_inertia;
use traj_core::pbc_utils as core_pbc;
use traj_core::rng_utils as core_rng;

use crate::plans::PbcMode;

#[cfg(feature = "cuda")]
use traj_gpu::Float4;

pub(crate) fn box_lengths(chunk: &FrameChunk, frame: usize) -> TrajResult<(f64, f64, f64)> {
    core_pbc::box_lengths(chunk.box_[frame])
}

pub(crate) fn apply_pbc(dx: &mut f64, dy: &mut f64, dz: &mut f64, lx: f64, ly: f64, lz: f64) {
    core_pbc::apply_pbc(dx, dy, dz, lx, ly, lz);
}

pub(crate) fn apply_pbc_triclinic(
    dx: &mut f64,
    dy: &mut f64,
    dz: &mut f64,
    cell: &[[f64; 3]; 3],
    inv: &[[f64; 3]; 3],
) {
    core_pbc::apply_pbc_triclinic(dx, dy, dz, cell, inv);
}

pub(crate) fn next_f64(state: &mut u64) -> f64 {
    core_rng::next_f64(state)
}

pub(crate) fn gaussian_pair(state: &mut u64) -> (f64, f64) {
    core_rng::gaussian_pair(state)
}

pub(crate) fn cell_and_inv_from_box(box_: Box3) -> TrajResult<([[f64; 3]; 3], [[f64; 3]; 3])> {
    core_pbc::cell_and_inv_from_box(box_)
}

pub(crate) fn dihedral_value(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    d: [f64; 3],
    pbc: PbcMode,
    frame: Option<(&FrameChunk, usize)>,
    degrees: bool,
    range360: bool,
) -> TrajResult<f32> {
    let mut b0x = a[0] - b[0];
    let mut b0y = a[1] - b[1];
    let mut b0z = a[2] - b[2];
    let mut b1x = c[0] - b[0];
    let mut b1y = c[1] - b[1];
    let mut b1z = c[2] - b[2];
    let mut b2x = d[0] - c[0];
    let mut b2y = d[1] - c[1];
    let mut b2z = d[2] - c[2];
    if matches!(pbc, PbcMode::Orthorhombic) {
        if let Some((chunk, frame_idx)) = frame {
            let (lx, ly, lz) = box_lengths(chunk, frame_idx)?;
            apply_pbc(&mut b0x, &mut b0y, &mut b0z, lx, ly, lz);
            apply_pbc(&mut b1x, &mut b1y, &mut b1z, lx, ly, lz);
            apply_pbc(&mut b2x, &mut b2y, &mut b2z, lx, ly, lz);
        }
    }
    Ok(dihedral_from_vectors(
        [b0x, b0y, b0z],
        [b1x, b1y, b1z],
        [b2x, b2y, b2z],
        degrees,
        range360,
    ))
}

#[cfg(feature = "cuda")]
pub(crate) fn rotation_from_cov(cov: &[f32; 9]) -> Matrix3<f64> {
    let cov_f64 = [
        cov[0] as f64,
        cov[1] as f64,
        cov[2] as f64,
        cov[3] as f64,
        cov[4] as f64,
        cov[5] as f64,
        cov[6] as f64,
        cov[7] as f64,
        cov[8] as f64,
    ];
    let m = Matrix3::from_row_slice(&cov_f64);
    let svd = m.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return Matrix3::identity(),
    };
    let mut r = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_t_adj = v_t;
        v_t_adj.row_mut(2).neg_mut();
        r = v_t_adj.transpose() * u.transpose();
    }
    r
}

#[cfg(feature = "cuda")]
pub(crate) fn chunk_boxes(chunk: &FrameChunk, pbc: PbcMode) -> TrajResult<Vec<Float4>> {
    let mut out = Vec::with_capacity(chunk.n_frames);
    for frame in 0..chunk.n_frames {
        match pbc {
            PbcMode::Orthorhombic => match chunk.box_[frame] {
                Box3::Orthorhombic { lx, ly, lz } => out.push(Float4 {
                    x: lx,
                    y: ly,
                    z: lz,
                    w: 0.0,
                }),
                _ => {
                    return Err(TrajError::Mismatch(
                        "orthorhombic box required for PBC".into(),
                    ))
                }
            },
            PbcMode::None => out.push(Float4 {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                w: 0.0,
            }),
        }
    }
    Ok(out)
}

#[cfg(feature = "cuda")]
pub(crate) fn chunk_cell_mats(chunk: &FrameChunk) -> TrajResult<(Vec<Float4>, Vec<Float4>)> {
    let mut cell = Vec::with_capacity(chunk.n_frames * 3);
    let mut inv = Vec::with_capacity(chunk.n_frames * 3);
    for frame in 0..chunk.n_frames {
        let (cell_rows, inv_rows) = cell_and_inv_from_box(chunk.box_[frame])?;
        for row in cell_rows.iter() {
            cell.push(Float4 {
                x: row[0] as f32,
                y: row[1] as f32,
                z: row[2] as f32,
                w: 0.0,
            });
        }
        for row in inv_rows.iter() {
            inv.push(Float4 {
                x: row[0] as f32,
                y: row[1] as f32,
                z: row[2] as f32,
                w: 0.0,
            });
        }
    }
    Ok((cell, inv))
}

pub(crate) fn kabsch_rotation(
    frame: &[[f32; 4]],
    reference: &[[f32; 4]],
) -> (Matrix3<f64>, Vector3<f64>, Vector3<f64>) {
    kabsch_rotation_weighted(frame, reference, None)
}

pub(crate) fn centroid_weighted(points: &[[f32; 4]], weights: Option<&[f32]>) -> Vector3<f64> {
    let n = points.len();
    if n == 0 {
        return Vector3::zeros();
    }
    let mut sum = Vector3::zeros();
    let mut wsum = 0.0f64;
    match weights {
        Some(w) => {
            for i in 0..n {
                let wi = w[i] as f64;
                sum[0] += points[i][0] as f64 * wi;
                sum[1] += points[i][1] as f64 * wi;
                sum[2] += points[i][2] as f64 * wi;
                wsum += wi;
            }
        }
        None => {
            for i in 0..n {
                sum[0] += points[i][0] as f64;
                sum[1] += points[i][1] as f64;
                sum[2] += points[i][2] as f64;
            }
            wsum = n as f64;
        }
    }
    if wsum == 0.0 {
        Vector3::zeros()
    } else {
        sum / wsum
    }
}

pub(crate) fn kabsch_rotation_weighted(
    frame: &[[f32; 4]],
    reference: &[[f32; 4]],
    weights: Option<&[f32]>,
) -> (Matrix3<f64>, Vector3<f64>, Vector3<f64>) {
    let n = frame.len().min(reference.len());
    if n == 0 {
        return (Matrix3::identity(), Vector3::zeros(), Vector3::zeros());
    }
    let frame = &frame[..n];
    let reference = &reference[..n];
    let weights = weights.map(|w| &w[..n]);

    let cx = centroid_weighted(frame, weights);
    let cy = centroid_weighted(reference, weights);
    let mut h = Matrix3::zeros();
    match weights {
        Some(w) => {
            for i in 0..n {
                let wi = w[i] as f64;
                let xr = Vector3::new(
                    frame[i][0] as f64 - cx[0],
                    frame[i][1] as f64 - cx[1],
                    frame[i][2] as f64 - cx[2],
                );
                let yr = Vector3::new(
                    reference[i][0] as f64 - cy[0],
                    reference[i][1] as f64 - cy[1],
                    reference[i][2] as f64 - cy[2],
                );
                h += (xr * yr.transpose()) * wi;
            }
        }
        None => {
            for i in 0..n {
                let xr = Vector3::new(
                    frame[i][0] as f64 - cx[0],
                    frame[i][1] as f64 - cx[1],
                    frame[i][2] as f64 - cx[2],
                );
                let yr = Vector3::new(
                    reference[i][0] as f64 - cy[0],
                    reference[i][1] as f64 - cy[1],
                    reference[i][2] as f64 - cy[2],
                );
                h += xr * yr.transpose();
            }
        }
    }

    let svd = h.svd(true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return (Matrix3::identity(), cx, cy),
    };
    let mut r = v_t.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let mut v_t_adj = v_t;
        v_t_adj.row_mut(2).neg_mut();
        r = v_t_adj.transpose() * u.transpose();
    }
    (r, cx, cy)
}
