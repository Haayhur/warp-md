from __future__ import annotations

from typing import Optional

import numpy as np

import warp_md


_DsspPlan = (
    getattr(warp_md.traj_py, "PyDsspPlan", None) if getattr(warp_md, "traj_py", None) else None
)


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _avg_counts(ss: np.ndarray) -> dict:
    if ss.size == 0:
        return {}
    values, counts = np.unique(ss, return_counts=True)
    return {str(v): float(c) for v, c in zip(values, counts)}


def dssp(
    traj,
    system,
    mask: str = "protein",
    simplified: bool = False,
    chunk_frames: Optional[int] = None,
):
    """Return DSSP-like labels via Rust plan path."""
    if _DsspPlan is None:
        raise RuntimeError(
            "PyDsspPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    selection = system.select(mask) if mask else system.select(_all_resid_mask(system))
    if len(selection.indices) == 0:
        selection = system.select(_all_resid_mask(system))
    plan = _DsspPlan(selection)
    try:
        labels, codes = plan.run(traj, system, chunk_frames=chunk_frames, device="auto")
    except TypeError as exc:
        raise RuntimeError(
            "dssp requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    labels = np.asarray(labels, dtype="U16")
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.size == 0:
        return labels, np.empty((0, labels.shape[0]), dtype="U1"), {}
    full_codes = np.clip(codes.astype(np.int64, copy=False), 0, 7)
    if simplified:
        lut = np.asarray(["C", "H", "E"], dtype="U1")
        collapse = np.asarray([0, 1, 2, 2, 1, 1, 0, 0], dtype=np.int64)
        ss = lut[collapse[full_codes]]
    else:
        lut = np.asarray(["C", "H", "B", "E", "G", "I", "T", "S"], dtype="U1")
        ss = lut[full_codes]
    avg = _avg_counts(ss)
    return labels, ss, avg


def dssp_allatoms(traj, system, mask: str = "protein", simplified: bool = False):
    """Return DSSP-like labels for all atoms."""
    return dssp(traj, system, mask=mask, simplified=simplified)[1]


def dssp_allresidues(traj, system, mask: str = "protein", simplified: bool = False):
    """Return DSSP-like labels for all residues."""
    return dssp(traj, system, mask=mask, simplified=simplified)[1]
