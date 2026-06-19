# Usage:
# from warp_md.analysis.velocity import get_velocity
# v = get_velocity(traj, system, mask="name CA", frame_indices=[0, 5])

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_selection,
    selection_indices,
)


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


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
    if not is_native_traj(traj):
        raise RuntimeError(
            "get_velocity requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for get_velocity")
    idx = selection_indices(system, mask)
    plan_cls = load_native_symbol("GetVelocityPlan")
    if plan_cls is None:
        raise RuntimeError(
            "GetVelocityPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    try:
        plan = plan_cls(native_selection(system, native_system, mask))
        values = np.asarray(
            plan.run(
                traj,
                native_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=_frame_indices_arg(frame_indices),
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native GetVelocityPlan execution failed") from exc
    if values.ndim != 2 or values.shape[1] != idx.size * 3:
        raise RuntimeError("native GetVelocityPlan returned unexpected shape")

    scale = np.float32(float(length_scale) / float(time_scale))
    return values.reshape(values.shape[0], idx.size, 3) * scale


__all__ = ["get_velocity"]
