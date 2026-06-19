# Usage:
# from warp_md.analysis.autoimage import autoimage
# new_traj = autoimage(traj, system, mask="all")

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._runtime import (
    is_native_traj,
    load_native_symbol,
    native_inputs,
    native_selection,
    read_all_frames,
    subset_frames,
)
from .trajectory import ArrayTrajectory


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _payload_to_array(payload) -> ArrayTrajectory:
    coords = np.asarray(payload["coords"], dtype=np.float32)
    box = payload.get("box")
    time = payload.get("time_ps")
    return ArrayTrajectory(
        coords,
        box=None if box is None else np.asarray(box, dtype=np.float32),
        time_ps=None if time is None else np.asarray(time, dtype=np.float64),
    )


def _native_autoimage_live(
    traj,
    system,
    mask: str,
    frame_indices: Optional[Sequence[int]],
    chunk_frames: Optional[int],
):
    if not is_native_traj(traj):
        return None
    plan_cls = load_native_symbol("AutoImageTrajectoryPlan")
    if plan_cls is None:
        return None
    native_traj, native_system = native_inputs(
        traj,
        system,
        chunk_frames,
        include_box=True,
        include_time=True,
    )
    if native_traj is None or native_system is None:
        return None
    try:
        plan = plan_cls(native_selection(system, native_system, mask))
        payload = plan.run(
            native_traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        )
    except Exception as exc:
        raise RuntimeError("native AutoImageTrajectoryPlan execution failed") from exc
    return _payload_to_array(payload)


def autoimage(
    traj,
    system,
    mask: str = "",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> ArrayTrajectory:
    """Move selected atoms into primary unit cell (orthorhombic only)."""
    native = _native_autoimage_live(traj, system, mask, frame_indices, chunk_frames)
    if native is not None:
        return native

    coords, box, time = read_all_frames(traj, chunk_frames, include_box=True, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords, box, time = subset_frames(coords, frame_indices, box=box, time=time)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)
    if box is None:
        raise ValueError("autoimage requires box lengths")

    plan_cls = load_native_symbol("AutoImagePlan")
    if plan_cls is None:
        raise RuntimeError("AutoImagePlan binding unavailable")
    source = ArrayTrajectory(coords, box=np.asarray(box, dtype=np.float32), time_ps=time)
    native_traj, native_system = native_inputs(
        source,
        system,
        chunk_frames,
        include_box=True,
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native autoimage inputs")
    try:
        selection = native_selection(system, native_system, mask)
        plan = plan_cls(selection)
        values = np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        ).reshape(coords.shape)
    except Exception as exc:
        raise RuntimeError("native AutoImagePlan execution failed") from exc
    return ArrayTrajectory(values, box=np.asarray(box, dtype=np.float32), time_ps=time)


__all__ = ["autoimage"]
