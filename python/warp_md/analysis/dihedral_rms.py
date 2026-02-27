# Usage:
# from warp_md.analysis.dihedral_rms import dihedral_rms
# rms = dihedral_rms(traj, system, "@1 @2 @3 @4", ref=0)

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .geometry import dihedral


MaskLike = Union[str, Sequence[str], np.ndarray]


def _wrap_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def dihedral_rms(
    traj,
    system,
    mask: MaskLike = "",
    ref: Optional[Union[int, Sequence[float], np.ndarray]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Compute RMS of dihedral angles relative to a reference.

    Parameters
    ----------
    mask : str | list[str] | (n,4) int array
        Dihedral definitions.
    ref : int | array-like | None
        Reference frame index or reference dihedral values.
    """
    _ = dtype
    values = dihedral(traj, system, mask, frame_indices=frame_indices, chunk_frames=chunk_frames)
    if values.size == 0:
        return np.empty((0,), dtype=np.float32)

    if values.ndim == 1:
        values = values[None, :]

    n_dihedrals, n_frames = values.shape
    if n_frames == 0:
        return np.empty((0,), dtype=np.float32)

    if ref is None:
        ref_vals = values[:, 0]
    elif isinstance(ref, (int, np.integer)):
        idx = int(ref)
        if idx < 0:
            idx = n_frames + idx
        if idx < 0 or idx >= n_frames:
            raise ValueError("ref frame index out of range")
        ref_vals = values[:, idx]
    else:
        ref_arr = np.asarray(ref, dtype=np.float64)
        if ref_arr.ndim == 1:
            if ref_arr.size != n_dihedrals:
                raise ValueError("ref array must match number of dihedrals")
            ref_vals = ref_arr
        elif ref_arr.ndim == 2:
            if ref_arr.shape[0] != n_dihedrals:
                raise ValueError("ref array first dimension must match number of dihedrals")
            ref_vals = ref_arr[:, 0]
        else:
            raise ValueError("ref must be int or 1D/2D array")

    ref_vals = ref_vals.reshape(n_dihedrals, 1)
    delta = _wrap_delta(values - ref_vals)
    rms = np.sqrt(np.mean(delta**2, axis=0))
    return rms.astype(np.float32)


__all__ = ["dihedral_rms"]
