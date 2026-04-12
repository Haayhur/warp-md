# Usage:
# from warp_md.analysis.closest import closest, closest_atom
# idx = closest_atom(system, coords, point=(0,0,0))

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._runtime import (
    load_native_symbol,
    native_inputs,
    native_selection,
    read_all_frames,
    selection_indices,
    subset_frames,
)
from .trajectory import ArrayTrajectory


def closest_atom(system, frame: np.ndarray, point=(0.0, 0.0, 0.0), mask: str = "") -> int:
    idx = selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")
    coords = frame[idx]
    diff = coords - np.asarray(point, dtype=np.float64)[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    return int(idx[int(np.argmin(dist2))])


def closest(
    traj,
    system,
    mask: str = "*",
    solvent_mask: Optional[str] = None,
    n_solvents: int = 10,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
):
    """Keep only closest n_solvents atoms."""
    coords, box, time = read_all_frames(traj, chunk_frames, include_box=True, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords, box, time = subset_frames(coords, frame_indices, box=box, time=time)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return ArrayTrajectory(coords, box=None, time_ps=time)

    target_mask = mask
    probe_mask = solvent_mask if solvent_mask else mask
    target_idx = selection_indices(system, target_mask)
    probe_idx = selection_indices(system, probe_mask)
    if target_idx.size == 0 or probe_idx.size == 0:
        raise ValueError("selection resolved to empty set")

    n_keep = max(0, int(n_solvents))
    if n_keep == 0:
        return ArrayTrajectory(np.empty((coords.shape[0], 0, 3), dtype=np.float32), box=None, time_ps=time)

    plan_cls = load_native_symbol("ClosestPlan")
    if plan_cls is None:
        raise RuntimeError("ClosestPlan binding unavailable")
    source = ArrayTrajectory(coords, box=box, time_ps=time)
    native_traj, native_system = native_inputs(
        source,
        system,
        chunk_frames,
        include_box=True,
        include_time=True,
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native closest inputs")
    try:
        plan = plan_cls(
            native_selection(system, native_system, target_mask),
            native_selection(system, native_system, probe_mask),
            n_keep,
            "none",
        )
        keep_idx = np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.int64,
        )
    except Exception as exc:
        raise RuntimeError("native ClosestPlan execution failed") from exc
    if keep_idx.shape != (coords.shape[0], n_keep):
        raise RuntimeError("native ClosestPlan returned unexpected shape")

    out = np.empty((coords.shape[0], n_keep, 3), dtype=np.float32)
    for frame in range(coords.shape[0]):
        out[frame] = coords[frame][keep_idx[frame]]
    return ArrayTrajectory(out, box=None, time_ps=time)


__all__ = ["closest", "closest_atom"]
