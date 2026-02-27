# Usage:
# from warp_md.analysis.fiximagedbonds import fiximagedbonds
# new_traj = fiximagedbonds(traj, system, mask="all")

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._stream import infer_n_atoms
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


def fiximagedbonds(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> ArrayTrajectory:
    """Image selected atoms using Rust plan path only."""
    try:
        from warp_md import FixImageBondsPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "FixImageBondsPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(FixImageBondsPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "FixImageBondsPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel = _selection_from_mask(system, mask)
    if len(sel.indices) == 0:
        raise ValueError("selection resolved to empty set")

    plan = FixImageBondsPlan(sel)
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
            "fiximagedbonds requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    arr = np.asarray(flat, dtype=np.float32)
    n_atoms = infer_n_atoms(system, traj=traj)
    if n_atoms <= 0:
        return ArrayTrajectory(np.empty((0, 0, 3), dtype=np.float32))
    if arr.size == 0:
        return ArrayTrajectory(np.empty((0, n_atoms, 3), dtype=np.float32))
    if arr.size % (n_atoms * 3) != 0:
        raise ValueError("fiximagedbonds plan output shape is inconsistent with atom count")
    out = arr.reshape((-1, n_atoms, 3))
    return ArrayTrajectory(out.astype(np.float32, copy=False))


__all__ = ["fiximagedbonds"]
