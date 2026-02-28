# Usage:
# from warp_md.analysis.check_structure import check_structure
# counts, report = check_structure(traj, system)

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._chunk_io import read_chunk_fields


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    chunk = read_chunk_fields(traj, max_chunk)
    if chunk is None:
        return None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk)
    return np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))


def check_structure(
    traj,
    system,
    mask: str = "",
    options: str = "",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "ndarray",
) -> Tuple[np.ndarray, str]:
    """Basic structure checks (NaN/inf)."""
    del system, mask, options
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.int64), ""

    n_frames = coords.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        coords = coords[select]
        n_frames = coords.shape[0]

    counts = np.zeros(n_frames, dtype=np.int64)
    report_lines = []
    for i in range(n_frames):
        frame = coords[i]
        bad = ~np.isfinite(frame)
        n_bad = int(np.any(bad, axis=1).sum())
        counts[i] = n_bad
        if n_bad > 0:
            report_lines.append(f"frame {i}: {n_bad} atoms with invalid coords")
    report = "\n".join(report_lines)

    if dtype == "ndarray":
        return counts.astype(np.int64), report
    return counts.astype(np.int64), report


__all__ = ["check_structure"]
