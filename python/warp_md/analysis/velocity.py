# Usage:
# from warp_md.analysis.velocity import get_velocity
# v = get_velocity(traj, system, mask="name CA", frame_indices=[0, 5])

from __future__ import annotations

from typing import Optional, Sequence, Union

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


def get_velocity(
    traj,
    system,
    mask: Optional[Union[str, Sequence[int]]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    length_scale: float = 1.0,
    time_scale: float = 1.0,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Finite-difference velocities from coordinates."""
    coords, _box, times = read_all_frames(traj, chunk_frames, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords, _box, times = subset_frames(coords, frame_indices, time=times)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return np.empty((0, 0, 3), dtype=np.float32)

    idx = selection_indices(system, mask)
    if idx.size == 0:
        return np.empty((coords.shape[0], 0, 3), dtype=np.float32)

    plan_cls = load_native_symbol("GetVelocityPlan")
    if plan_cls is None:
        raise RuntimeError("GetVelocityPlan binding unavailable")
    source = ArrayTrajectory(coords, time_ps=times)
    native_traj, native_system = native_inputs(
        source,
        system,
        chunk_frames,
        include_time=True,
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native velocity inputs")
    try:
        plan = plan_cls(native_selection(system, native_system, mask))
        values = np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native GetVelocityPlan execution failed") from exc
    if values.shape != (coords.shape[0], idx.size * 3):
        raise RuntimeError("native GetVelocityPlan returned unexpected shape")

    scale = np.float32(float(length_scale) / float(time_scale))
    return values.reshape(coords.shape[0], idx.size, 3) * scale


__all__ = ["get_velocity"]
