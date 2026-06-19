from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import warp_md
from ._runtime import coerce_native_system, is_native_traj


_DsspPlan = (
    getattr(warp_md.traj_py, "PyDsspPlan", None) if getattr(warp_md, "traj_py", None) else None
)


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def dssp(
    traj,
    system,
    mask: str = "protein",
    simplified: bool = True,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "tuple",
):
    """Return DSSP-like labels via Rust plan path."""
    if not is_native_traj(traj):
        raise RuntimeError(
            "dssp requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native dssp system")
    if _DsspPlan is None:
        raise RuntimeError(
            "PyDsspPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    selection = (
        native_system.select(mask)
        if mask
        else native_system.select(_all_resid_mask(native_system))
    )
    if len(selection.indices) == 0:
        selection = native_system.select(_all_resid_mask(native_system))
    plan = _DsspPlan(selection)
    frame_indices_arg = None if frame_indices is None else [int(v) for v in frame_indices]
    try:
        result = plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=frame_indices_arg,
            simplified=bool(simplified),
        )
    except TypeError as exc:
        raise RuntimeError(
            "dssp requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    labels, codes, symbols, rows, cols, avg = result
    labels = np.asarray(labels, dtype="U16")
    codes = np.asarray(codes, dtype=np.uint8)
    rows = int(rows)
    cols = int(cols)
    if codes.size == 0:
        empty = np.empty((0, labels.shape[0]), dtype="U1")
        key = str(dtype).lower()
        if key == "full":
            return {
                "residues": labels,
                "ss": empty,
                "codes": np.empty((0, labels.shape[0]), dtype=np.uint8),
                "avg": {},
            }
        if key == "dict":
            return {"residues": labels, "ss": empty, "avg": {}}
        return labels, empty, {}
    ss = np.asarray(symbols, dtype="U1").reshape((rows, cols))
    key = str(dtype).lower()
    if key in ("dataset", "integer", "codes"):
        return labels, codes, avg
    if key == "full":
        return {"residues": labels, "ss": ss, "codes": codes, "avg": avg}
    if key == "dict":
        return {"residues": labels, "ss": ss, "avg": avg}
    return labels, ss, avg


def dssp_allatoms(traj, system, mask: str = "protein", simplified: bool = True):
    """Return DSSP-like labels for all atoms."""
    return dssp(traj, system, mask=mask, simplified=simplified)[1]


def dssp_allresidues(traj, system, mask: str = "protein", simplified: bool = True):
    """Return DSSP-like labels for all residues."""
    return dssp(traj, system, mask=mask, simplified=simplified)[1]
