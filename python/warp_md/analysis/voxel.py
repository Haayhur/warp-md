# Usage:
# from warp_md.analysis.voxel import count_in_voxel
# out = count_in_voxel(traj, system, "name O", "protein", (0.25, 0.25, 0.25), (4.0, 4.0, 4.0))

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from ._runtime import run_native_grid_plan


def count_in_voxel(
    traj,
    system,
    selection,
    center_selection,
    box_unit: Tuple[float, float, float],
    region_size: Tuple[float, float, float],
    shift: Optional[Tuple[float, float, float]] = None,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Return raw native GridOutput for voxel counts."""
    return run_native_grid_plan(
        "CountInVoxelPlan",
        traj,
        system,
        selection,
        center_selection,
        box_unit,
        region_size,
        shift=shift,
        length_scale=length_scale,
        frame_indices=frame_indices,
        chunk_frames=chunk_frames,
        device=device,
    )


__all__ = ["count_in_voxel"]
