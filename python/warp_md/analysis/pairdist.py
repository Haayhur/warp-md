# Usage:
# from warp_md.analysis.pairdist import pairdist
# out = pairdist(traj, system, mask='@CA', delta=0.1)

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


def _rmax_upper_bound(coords: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    sel_a = coords[:, idx_a, :]
    sel_b = coords[:, idx_b, :]
    lo = np.minimum(sel_a.min(axis=(0, 1)), sel_b.min(axis=(0, 1)))
    hi = np.maximum(sel_a.max(axis=(0, 1)), sel_b.max(axis=(0, 1)))
    return float(np.linalg.norm(hi - lo))


def pairdist(
    traj,
    system,
    mask: str = "*",
    mask2: str = "",
    delta: float = 0.1,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "dict",
    chunk_frames: Optional[int] = None,
):
    """Compute pair distance histogram."""
    if delta <= 0.0:
        raise ValueError("delta must be positive")

    coords, _box, _time = read_all_frames(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords, _box, _time = subset_frames(coords, frame_indices)
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {"bin_centers": empty, "hist": empty, "n_frames": 0}

    idx_a = selection_indices(system, mask)
    idx_b = selection_indices(system, mask2) if mask2 else idx_a
    same = not mask2 or np.array_equal(idx_a, idx_b)
    if idx_a.size == 0 or idx_b.size == 0 or (same and idx_a.size < 2):
        empty = np.empty((0,), dtype=np.float32)
        return {"bin_centers": empty, "hist": empty, "n_frames": int(coords.shape[0])}

    plan_cls = load_native_symbol("PairDistPlan")
    if plan_cls is None:
        raise RuntimeError("PairDistPlan binding unavailable")

    upper = _rmax_upper_bound(coords, idx_a, idx_b)
    n_bins = max(1, int(np.ceil(max(upper, float(delta)) / float(delta))))
    r_max = np.float32(n_bins * float(delta))

    source = ArrayTrajectory(coords)
    native_traj, native_system = native_inputs(source, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native pairdist inputs")
    try:
        sel_a = native_selection(system, native_system, mask)
        sel_b = sel_a if same else native_selection(system, native_system, mask2)
        out = plan_cls(
            sel_a,
            sel_b,
            n_bins,
            r_max,
            "none",
        ).run(native_traj, native_system, chunk_frames=chunk_frames, device="auto")
        centers, counts = out
        centers = np.asarray(centers, dtype=np.float32)
        counts = np.asarray(counts, dtype=np.float64)
    except Exception as exc:
        raise RuntimeError("native PairDistPlan execution failed") from exc

    if same:
        counts *= 0.5
    nonzero = np.nonzero(counts > 0)[0]
    if nonzero.size == 0:
        centers = centers[:0]
        counts = counts[:0]
    else:
        stop = int(nonzero[-1]) + 1
        centers = centers[:stop]
        counts = counts[:stop]

    out = {
        "bin_centers": centers.astype(np.float32),
        "hist": counts.astype(np.float32),
        "n_frames": int(coords.shape[0]),
    }
    if dtype == "dict":
        return out
    return out


__all__ = ["pairdist"]
