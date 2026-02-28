use traj_core::error::{TrajError, TrajResult};

#[cfg(feature = "cuda")]
use traj_gpu::{GpuBufferU32, GpuContext};

pub(super) fn ensure_gist_gpu_hist_buffers(
    ctx: &GpuContext,
    counts: &mut Option<GpuBufferU32>,
    orient_counts: &mut Option<GpuBufferU32>,
    n_cells_state: &mut usize,
    n_cells: usize,
    orientation_bins: usize,
) -> TrajResult<()> {
    if n_cells == 0 {
        *counts = None;
        *orient_counts = None;
        *n_cells_state = 0;
        return Ok(());
    }
    if *n_cells_state != 0 && *n_cells_state != n_cells {
        return Err(TrajError::Mismatch(
            "gist cuda histogram grid shape changed across frames".into(),
        ));
    }
    let orient_len = n_cells
        .checked_mul(orientation_bins)
        .ok_or_else(|| TrajError::Mismatch("gist orientation histogram size overflow".into()))?;
    if counts.is_none() || orient_counts.is_none() {
        *counts = Some(ctx.upload_u32(&vec![0u32; n_cells])?);
        *orient_counts = Some(ctx.upload_u32(&vec![0u32; orient_len])?);
    }
    *n_cells_state = n_cells;
    Ok(())
}
