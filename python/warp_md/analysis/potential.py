from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md


MaskLike = Union[str, Sequence[int], np.ndarray]

_PotentialPlan = (
    getattr(warp_md.traj_py, "PyPotentialPlan", None)
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
    raise RuntimeError("system.select_indices is required for non-string potential selections")


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
            "potential requires per-atom charges. Pass `charges=` or provide `atom_table()['charge']`."
        )
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("system atom_table()['charge'] must contain only finite values")
    return arr.tolist()


def potential(
    traj,
    system,
    selection: Optional[MaskLike] = "",
    charges: Optional[Sequence[float]] = None,
    axis: str = "z",
    bin: float = 0.25,
    n_slices: Optional[int] = None,
    center: Optional[MaskLike] = None,
    symmetrize: bool = False,
    correct: bool = False,
    discard_start: int = 0,
    discard_end: int = 0,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Electrostatic potential profile via Rust plan path.

    Returns 1D charge-density, electric-field, and potential profiles along
    one axis. This v1 contract is planar only; spherical micelle mode from
    `gmx potential` is not exposed yet.
    """
    if _PotentialPlan is None:
        raise RuntimeError(
            "PyPotentialPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    sel = _select(system, selection)
    center_sel = None if center is None else _select(system, center)
    charge_list = _resolve_charges(system, charges)
    plan = _PotentialPlan(
        sel,
        charge_list,
        axis=axis,
        bin=bin,
        n_slices=n_slices,
        center_selection=center_sel,
        symmetrize=symmetrize,
        correct=correct,
        discard_start=int(discard_start),
        discard_end=int(discard_end),
        length_scale=length_scale,
    )
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
            "potential requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "coordinate": np.asarray(out["coordinate"], dtype=np.float32),
        "charge_density": np.asarray(out["charge_density"], dtype=np.float32),
        "field": np.asarray(out["field"], dtype=np.float32),
        "potential": np.asarray(out["potential"], dtype=np.float32),
        "axis": str(out["axis"]),
        "bounds": np.asarray(out["bounds"], dtype=np.float32),
        "slice_width": float(out["slice_width"]),
        "n_frames": int(out["n_frames"]),
        "used_box": bool(out["used_box"]),
        "centered": bool(out["centered"]),
        "symmetrized": bool(out["symmetrized"]),
        "corrected": bool(out["corrected"]),
        "length_scale": float(out["length_scale"]),
        "discard_start": int(out["discard_start"]),
        "discard_end": int(out["discard_end"]),
    }


__all__ = ["potential"]
