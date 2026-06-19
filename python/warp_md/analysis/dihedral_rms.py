# Usage:
# from warp_md.analysis.dihedral_rms import dihedral_rms
# rms = dihedral_rms(traj, system, "@1 @2 @3 @4", ref=0)

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import load_native_symbol, native_inputs
from .geometry import (
    _frame_indices_arg,
    _multi_dihedral_command_groups,
    _native_selected,
    _selection_indices,
    dihedral,
)


MaskLike = Union[str, Sequence[str], np.ndarray]


def _wrap_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + 180.0) % 360.0 - 180.0


def _dihedral_groups(system, native_system, mask: MaskLike):
    if isinstance(mask, str):
        parts = str(mask).split()
        if len(parts) != 4:
            raise ValueError("dihedral mask must have 4 parts")
        groups = [
            (
                _selection_indices(system, parts[0]),
                _selection_indices(system, parts[1]),
                _selection_indices(system, parts[2]),
                _selection_indices(system, parts[3]),
            )
        ]
    else:
        arr = np.asarray(mask)
        if arr.dtype.kind == "i":
            if arr.ndim != 2 or arr.shape[1] != 4:
                raise ValueError("dihedral index array must be shape (n, 4)")
            groups = [
                (
                    np.asarray([int(a)], dtype=np.int64),
                    np.asarray([int(b)], dtype=np.int64),
                    np.asarray([int(c)], dtype=np.int64),
                    np.asarray([int(d)], dtype=np.int64),
                )
                for a, b, c, d in arr
            ]
        else:
            groups = _multi_dihedral_command_groups(system, list(mask))
    return [
        (
            _native_selected(native_system, np.asarray(a, dtype=np.int64)),
            _native_selected(native_system, np.asarray(b, dtype=np.int64)),
            _native_selected(native_system, np.asarray(c, dtype=np.int64)),
            _native_selected(native_system, np.asarray(d, dtype=np.int64)),
        )
        for a, b, c, d in groups
    ]


def _native_dihedral_rms(
    traj,
    system,
    mask: MaskLike,
    ref,
    frame_indices: Optional[Sequence[int]],
    chunk_frames: Optional[int],
) -> Optional[np.ndarray]:
    if ref is not None and not (isinstance(ref, (int, np.integer)) and int(ref) == 0):
        return None
    plan_cls = load_native_symbol("DihedralRmsPlan")
    if plan_cls is None:
        return None
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        return None
    groups = _dihedral_groups(system, native_system, mask)
    if not groups:
        return np.empty((0,), dtype=np.float32)
    try:
        plan = plan_cls(
            groups,
            reference="frame0",
            mass_weighted=False,
            pbc="none",
            degrees=True,
            range360=False,
        )
        return np.asarray(
            plan.run(
                native_traj,
                native_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=_frame_indices_arg(frame_indices),
            ),
            dtype=np.float32,
        ).reshape(-1)
    except Exception as exc:
        raise RuntimeError("native DihedralRmsPlan execution failed") from exc


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
    native_out = _native_dihedral_rms(traj, system, mask, ref, frame_indices, chunk_frames)
    if native_out is not None:
        return native_out

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
