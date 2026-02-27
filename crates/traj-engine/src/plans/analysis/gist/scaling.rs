use traj_core::error::{TrajError, TrajResult};

use crate::executor::PlanOutput;

pub(super) fn finalize_counts_orientation(
    counts: &[u32],
    orient_counts: &[u32],
    dims: [usize; 3],
    orientation_bins: usize,
) -> TrajResult<PlanOutput> {
    if counts.is_empty() {
        return Ok(PlanOutput::Matrix {
            data: Vec::new(),
            rows: 0,
            cols: orientation_bins + 1,
        });
    }
    let n_cells = dims[0] * dims[1] * dims[2];
    if counts.len() != n_cells || orient_counts.len() != n_cells * orientation_bins {
        return Err(TrajError::Mismatch(
            "gist output buffer shape mismatch".into(),
        ));
    }
    let cols = orientation_bins + 1;
    let mut data = vec![0.0f32; n_cells * cols];
    for cell in 0..n_cells {
        let row_base = cell * cols;
        data[row_base] = counts[cell] as f32;
        let orient_base = cell * orientation_bins;
        for b in 0..orientation_bins {
            data[row_base + 1 + b] = orient_counts[orient_base + b] as f32;
        }
    }
    Ok(PlanOutput::Matrix {
        data,
        rows: n_cells,
        cols,
    })
}
