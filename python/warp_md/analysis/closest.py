# Usage:
# from warp_md.analysis.closest import closest, closest_atom
# idx = closest_atom(system, coords, point=(0,0,0))

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import warp_md

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_inputs,
    native_selection,
    read_all_frames,
    selection_indices,
    subset_frames,
)
from .trajectory import ArrayTrajectory


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def closest_atom(system, frame: np.ndarray, point=(0.0, 0.0, 0.0), mask: str = "") -> int:
    idx = selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")
    coords = frame[idx]
    diff = coords - np.asarray(point, dtype=np.float64)[None, :]
    dist2 = np.sum(diff * diff, axis=1)
    return int(idx[int(np.argmin(dist2))])


def _closest_gather(coords: np.ndarray, keep_idx: np.ndarray) -> np.ndarray:
    fn = getattr(warp_md, "closest_gather_array", None)
    if fn is None or getattr(fn, "__name__", "") != "closest_gather_array":
        raise RuntimeError("closest_gather_array native binding unavailable")
    return np.asarray(
        fn(np.asarray(coords, dtype=np.float32), np.asarray(keep_idx, dtype=np.int64)),
        dtype=np.float32,
    )


def _closest_native_live(
    traj,
    system,
    target_mask,
    probe_mask,
    n_keep,
    frame_indices,
    chunk_frames,
):
    plan_cls = load_native_symbol("ClosestCoordsPlan")
    if plan_cls is None:
        return None
    native_system = coerce_native_system(system)
    if native_system is None:
        return None
    try:
        plan = plan_cls(
            native_selection(system, native_system, target_mask),
            native_selection(system, native_system, probe_mask),
            n_keep,
            "none",
        )
        payload = plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        )
    except Exception as exc:
        raise RuntimeError("native ClosestCoordsPlan execution failed") from exc

    coords = np.asarray(payload["coords"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != n_keep * 3:
        raise RuntimeError("native ClosestCoordsPlan returned unexpected shape")
    coords = coords.reshape(coords.shape[0], n_keep, 3)
    box = payload.get("box")
    if box is not None:
        box = np.asarray(box, dtype=np.float32)
    time = payload.get("time_ps")
    if time is not None:
        time = np.asarray(time, dtype=np.float64)
    return ArrayTrajectory(coords, box=box, time_ps=time)


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
    target_mask = mask
    probe_mask = solvent_mask if solvent_mask else mask
    n_keep = max(0, int(n_solvents))
    target_idx = selection_indices(system, target_mask)
    probe_idx = selection_indices(system, probe_mask)
    if target_idx.size == 0 or probe_idx.size == 0:
        raise ValueError("selection resolved to empty set")

    if is_native_traj(traj):
        native_out = _closest_native_live(
            traj,
            system,
            target_mask,
            probe_mask,
            n_keep,
            frame_indices,
            chunk_frames,
        )
        if native_out is not None:
            return native_out

    coords, box, time = read_all_frames(traj, chunk_frames, include_box=True, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords, box, time = subset_frames(coords, frame_indices, box=box, time=time)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)

    if n_keep == 0:
        return ArrayTrajectory(
            np.empty((coords.shape[0], 0, 3), dtype=np.float32),
            box=box,
            time_ps=time,
        )

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

    out = _closest_gather(coords, keep_idx)
    return ArrayTrajectory(out, box=box, time_ps=time)


__all__ = ["closest", "closest_atom"]
