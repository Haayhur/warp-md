from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md


MaskLike = Union[str, Sequence[int], np.ndarray]

_RamaPlan = (
    getattr(warp_md.traj_py, "PyRamaPlan", None) if getattr(warp_md, "traj_py", None) else None
)


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _select(system, selection: Optional[MaskLike]):
    if isinstance(selection, str) or selection is None:
        expr = selection
        if selection in ("", "*", "all", None):
            expr = _all_resid_mask(system) if hasattr(system, "atom_table") else "all"
        sel = system.select(expr)
        if len(sel.indices) == 0 and hasattr(system, "atom_table"):
            return system.select(_all_resid_mask(system))
        return sel
    indices = np.asarray(selection, dtype=np.int64).reshape(-1).tolist()
    if hasattr(system, "select_indices"):
        return system.select_indices(indices)
    raise RuntimeError("system.select_indices is required for non-string rama selections")


def rama(
    traj,
    system,
    selection: Optional[MaskLike] = "protein",
    range360: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Backbone phi/psi Ramachandran analysis via Rust plan path."""
    if _RamaPlan is None:
        raise RuntimeError(
            "PyRamaPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    sel = _select(system, selection)
    plan = _RamaPlan(sel, range360=range360)
    try:
        out = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
        )
    except TypeError as exc:
        raise RuntimeError(
            "rama requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "labels": np.asarray(out["labels"], dtype="U16"),
        "phi": np.asarray(out["phi"], dtype=np.float32),
        "psi": np.asarray(out["psi"], dtype=np.float32),
    }


__all__ = ["rama"]
