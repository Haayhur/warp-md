from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_selection,
)


MaskLike = Union[str, Sequence[int], np.ndarray, None]


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _charges_from_system(system, charges) -> list[float]:
    if charges is not None:
        return [float(value) for value in charges]
    if not hasattr(system, "atom_table"):
        raise ValueError("charges are required when system has no atom_table()")
    atoms = system.atom_table()
    values = atoms.get("charge", atoms.get("charges", None))
    if values is None:
        raise ValueError("charges are required or atom_table must include charge/charges")
    return [float(value) for value in values]


def dipole_moments(
    traj,
    system,
    selection: MaskLike = "",
    *,
    charges=None,
    group_by: str = "resid",
    length_scale: Optional[float] = None,
    group_types: Optional[Sequence[int]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "dict",
    device: str = "auto",
):
    """Compute per-frame dipole vectors in Rust."""
    if not is_native_traj(traj):
        raise RuntimeError(
            "dipole_moments requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for dipole_moments")
    plan_cls = load_native_symbol("DipoleMomentPlan")
    if plan_cls is None:
        raise RuntimeError(
            "DipoleMomentPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )

    charge_values = _charges_from_system(system, charges)
    sel = native_selection(system, native_system, selection, allow_at_indices=True)
    plan = plan_cls(
        sel,
        charge_values,
        group_by=group_by,
        length_scale=length_scale,
        group_types=None if group_types is None else [int(value) for value in group_types],
    )
    time, values = plan.run(
        traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=_frame_indices_arg(frame_indices),
    )
    time = np.asarray(time, dtype=np.float32)
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] % 3 != 0:
        raise RuntimeError("native DipoleMomentPlan returned unexpected output")
    dipoles = matrix.reshape(matrix.shape[0], matrix.shape[1] // 3, 3)

    key = str(dtype).lower()
    if key in ("dict", "mapping"):
        return {
            "time": time,
            "dipole": dipoles,
            "group_by": str(group_by),
            "length_scale": None if length_scale is None else float(length_scale),
        }
    if key in ("ndarray", "array"):
        return dipoles
    if key in ("tuple", "time"):
        return time, dipoles
    raise ValueError("dtype must be 'dict', 'ndarray', or 'tuple'")


__all__ = ["dipole_moments"]
