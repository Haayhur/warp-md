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


def drid(
    traj,
    system,
    selection: MaskLike = "",
    *,
    atom_indices: Optional[Sequence[int]] = None,
    exclude_bonds: bool = True,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "ndarray",
    device: str = "auto",
):
    """Compute distribution-of-reciprocal-distance descriptors in Rust."""
    if not is_native_traj(traj):
        raise RuntimeError("drid requires a Rust-backed trajectory so frame/atom loops stay in Rust.")
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for drid")
    plan_cls = load_native_symbol("DridPlan")
    if plan_cls is None:
        raise RuntimeError("DridPlan binding unavailable. Rebuild bindings with `maturin develop`.")

    mask = atom_indices if atom_indices is not None else selection
    sel = native_selection(system, native_system, mask, allow_at_indices=True)
    plan = plan_cls(sel, exclude_bonds=bool(exclude_bonds))
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
    key = str(dtype).lower()
    if key in ("ndarray", "array", "matrix"):
        return values
    if key == "dict":
        n_atoms = len(list(sel.indices))
        return {
            "values": values,
            "moments": values.reshape(values.shape[0], n_atoms, 3),
            "atom_indices": np.asarray(list(sel.indices), dtype=np.int64),
            "exclude_bonds": bool(exclude_bonds),
        }
    raise ValueError("dtype must be 'ndarray' or 'dict'")


__all__ = ["drid"]
