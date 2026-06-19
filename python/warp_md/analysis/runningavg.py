from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_selection,
)
from .trajectory import ArrayTrajectory


MaskLike = Union[str, Sequence[int], np.ndarray, None]


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _payload_to_array_trajectory(payload) -> ArrayTrajectory:
    coords = np.asarray(payload["coords"], dtype=np.float32)
    box = payload.get("box")
    time = payload.get("time_ps")
    return ArrayTrajectory(
        coords,
        box=None if box is None else np.asarray(box, dtype=np.float32),
        time_ps=None if time is None else np.asarray(time, dtype=np.float64),
    )


def runningavg(
    traj,
    system,
    selection: MaskLike = "",
    *,
    window: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "ndarray",
    device: str = "auto",
):
    """Compute cumulative or fixed-window coordinate running averages in Rust."""
    if not is_native_traj(traj):
        raise RuntimeError(
            "runningavg requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for runningavg")
    plan_cls = load_native_symbol("RunningAveragePlan")
    if plan_cls is None:
        raise RuntimeError(
            "RunningAveragePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    window_value = 0 if window is None else int(window)
    if window_value < 0:
        raise ValueError("window must be >= 0")
    key = str(dtype).lower()
    if key not in ("ndarray", "array", "traj", "trajectory", "dict"):
        raise ValueError("dtype must be 'ndarray', 'trajectory', or 'dict'")

    sel = native_selection(system, native_system, selection, allow_at_indices=True)

    if key in ("traj", "trajectory"):
        traj_plan_cls = load_native_symbol("RunningAverageTrajectoryPlan")
        if traj_plan_cls is not None:
            plan = traj_plan_cls(sel, window=window_value)
            payload = plan.run(
                traj,
                native_system,
                chunk_frames=chunk_frames,
                device=device,
                frame_indices=_frame_indices_arg(frame_indices),
            )
            return _payload_to_array_trajectory(payload)

    plan = plan_cls(sel, window=window_value)
    values = np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )
    n_atoms = len(list(sel.indices))
    coords = values.reshape(values.shape[0], n_atoms, 3)
    if key in ("ndarray", "array"):
        return coords
    if key in ("traj", "trajectory"):
        return ArrayTrajectory(coords)
    if key == "dict":
        return {
            "coords": coords,
            "atom_indices": np.asarray(list(sel.indices), dtype=np.int64),
            "window": window_value,
            "mode": "cumulative" if window_value == 0 else "window",
        }


__all__ = ["runningavg"]
