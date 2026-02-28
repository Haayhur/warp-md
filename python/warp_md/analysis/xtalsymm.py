# Usage:
# from warp_md.analysis.xtalsymm import xtalsymm
# new_traj = xtalsymm(traj, system, mask="all", repeats=(2, 2, 2))

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .trajectory import ArrayTrajectory


MaskLike = Union[str, Sequence[int], np.ndarray]


def _all_resid_mask(system) -> str:
    if not hasattr(system, "atom_table"):
        return "all"
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_from_mask(system, mask: MaskLike):
    if isinstance(mask, str):
        query = mask if mask not in ("", "*", "all", None) else _all_resid_mask(system)
        return system.select(query)
    return system.select_indices(np.asarray(mask, dtype=np.int64).reshape(-1).tolist())


def _normalize_symmetry_ops(
    symmetry_ops: Optional[Sequence[Sequence[float]]],
) -> Optional[list[list[float]]]:
    if symmetry_ops is None:
        return None
    affine_last_row = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    out: list[list[float]] = []
    for op in symmetry_ops:
        arr = np.asarray(op, dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise ValueError("symmetry op values must be finite")
        if arr.shape == (3, 3):
            flat = arr.reshape(-1)
        elif arr.shape == (3, 4):
            flat = arr.reshape(-1)
        elif arr.shape == (4, 4):
            if not np.allclose(arr[3, :], affine_last_row, rtol=0.0, atol=1.0e-9):
                raise ValueError("4x4 symmetry op must be affine with last row [0,0,0,1]")
            flat = arr[:3, :].reshape(-1)
        elif arr.ndim == 1 and arr.size in (9, 12, 16):
            if arr.size == 16:
                mat = arr.reshape(4, 4)
                if not np.allclose(mat[3, :], affine_last_row, rtol=0.0, atol=1.0e-9):
                    raise ValueError("4x4 symmetry op must be affine with last row [0,0,0,1]")
                flat = mat[:3, :].reshape(-1)
            else:
                flat = arr
        else:
            raise ValueError(
                "each symmetry op must be shape (3,3), (3,4), (4,4), or flat length 9/12/16"
            )
        out.append(flat.astype(np.float64, copy=False).tolist())
    if not out:
        raise ValueError("symmetry_ops must contain at least one transform")
    return out


def _normalize_repeats(repeats: Tuple[int, int, int]) -> tuple[int, int, int]:
    if not isinstance(repeats, (list, tuple)) or len(repeats) != 3:
        raise ValueError("repeats must be a 3-item tuple/list")
    reps = tuple(int(r) for r in repeats)
    if any(r <= 0 for r in reps):
        raise ValueError("repeats must contain positive integers")
    return reps


def xtalsymm(
    traj,
    system,
    mask: MaskLike = "",
    repeats: Tuple[int, int, int] = (1, 1, 1),
    symmetry_ops: Optional[Sequence[Sequence[float]]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> ArrayTrajectory:
    """Replicate unit cell using Rust plan path only."""
    reps = _normalize_repeats(repeats)
    ops_payload = _normalize_symmetry_ops(symmetry_ops)

    try:
        from warp_md import XtalSymmPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "XtalSymmPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(XtalSymmPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "XtalSymmPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel = _selection_from_mask(system, mask)
    n_sel = len(sel.indices)
    if n_sel == 0:
        raise ValueError("selection resolved to empty set")

    try:
        plan = XtalSymmPlan(sel, list(reps), symmetry_ops=ops_payload)
    except TypeError as exc:
        raise RuntimeError(
            "xtalsymm requires updated Rust bindings with symmetry_ops support. Rebuild with `maturin develop`."
        ) from exc
    try:
        flat = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices,
        )
    except TypeError as exc:
        raise RuntimeError(
            "xtalsymm requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    arr = np.asarray(flat, dtype=np.float32)
    n_atoms = n_sel * (len(ops_payload) if ops_payload is not None else reps[0] * reps[1] * reps[2])
    if n_atoms <= 0:
        return ArrayTrajectory(np.empty((0, 0, 3), dtype=np.float32))
    if arr.size == 0:
        return ArrayTrajectory(np.empty((0, n_atoms, 3), dtype=np.float32))
    if arr.size % (n_atoms * 3) != 0:
        raise ValueError("xtalsymm plan output shape is inconsistent with selection/repeats/symmetry_ops")
    out = arr.reshape((-1, n_atoms, 3))
    return ArrayTrajectory(out.astype(np.float32, copy=False))


__all__ = ["xtalsymm"]
