from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md


MaskLike = Union[str, Sequence[int], np.ndarray]

_SaltBridgePlan = (
    getattr(warp_md.traj_py, "PySaltBridgePlan", None)
    if getattr(warp_md, "traj_py", None)
    else None
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
        return system.select(expr)
    indices = np.asarray(selection, dtype=np.int64).reshape(-1).tolist()
    if hasattr(system, "select_indices"):
        return system.select_indices(indices)
    raise RuntimeError("system.select_indices is required for non-string saltbr selections")


def _resolve_charges(system, charges: Optional[Sequence[float]]) -> list[float]:
    if charges is not None:
        values = np.asarray(charges, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("charges must contain only finite values")
        return values.tolist()
    atoms = system.atom_table() if hasattr(system, "atom_table") else {}
    values = atoms.get("charge") if isinstance(atoms, dict) else None
    if values is None or len(values) == 0:
        raise RuntimeError(
            "saltbr requires per-atom charges. Pass `charges=` or provide `atom_table()['charge']`."
        )
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("system atom_table()['charge'] must contain only finite values")
    return arr.tolist()


def saltbr(
    traj,
    system,
    selection: Optional[MaskLike] = "",
    charges: Optional[Sequence[float]] = None,
    group_by: str = "atom",
    truncate: Optional[float] = None,
    contact_cutoff: Optional[float] = None,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Salt-bridge distance analysis via Rust plan path."""
    if _SaltBridgePlan is None:
        raise RuntimeError(
            "PySaltBridgePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    sel = _select(system, selection)
    charge_list = _resolve_charges(system, charges)
    plan = _SaltBridgePlan(
        sel,
        charge_list,
        group_by=group_by,
        truncate=truncate,
        contact_cutoff=contact_cutoff,
        length_scale=length_scale,
    )
    try:
        return plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
        )
    except TypeError as exc:
        raise RuntimeError(
            "saltbr requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc


__all__ = ["saltbr"]
