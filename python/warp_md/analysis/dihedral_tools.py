# Usage:
# from warp_md.analysis.dihedral_tools import rotate_dihedral, set_dihedral
# out = rotate_dihedral(traj, system, atoms=[0, 1, 2, 3], rotate_mask=[3], angle=30.0)

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._stream import infer_n_atoms
from .trajectory import ArrayTrajectory


AtomLike = Union[int, str]
MaskLike = Union[str, Sequence[int], np.ndarray]


def _selection_from_item(system, item: AtomLike):
    if isinstance(item, str):
        return system.select(item)
    try:
        idx = int(item)
    except Exception as exc:
        raise ValueError("atom entry must be int or selection string") from exc
    return system.select_indices([idx])


def _selection_from_mask(system, mask: MaskLike):
    if isinstance(mask, str):
        return system.select(mask)
    return system.select_indices(list(np.asarray(mask, dtype=np.int64).reshape(-1)))


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be bool")


def _normalize_pbc(value: str) -> str:
    mode = str(value).lower()
    if mode not in ("none", "orthorhombic"):
        raise ValueError("pbc must be 'none' or 'orthorhombic'")
    return mode


def _reshape_plan_coords(arr: np.ndarray, system, traj, message: str) -> np.ndarray:
    n_atoms = infer_n_atoms(system, traj=traj)
    if n_atoms <= 0:
        return np.empty((0, 0, 3), dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, n_atoms, 3), dtype=np.float32)
    if arr.size % (n_atoms * 3) != 0:
        raise ValueError(message)
    return arr.reshape((-1, n_atoms, 3))


def rotate_dihedral(
    traj,
    system,
    atoms: Sequence[AtomLike],
    rotate_mask: Optional[MaskLike] = None,
    angle: float = 0.0,
    mass: bool = False,
    degrees: bool = True,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Rotate a selection around the B-C dihedral axis using Rust plan only."""
    if len(atoms) != 4:
        raise ValueError("atoms must contain 4 items")
    mass = _as_bool(mass, "mass")
    degrees = _as_bool(degrees, "degrees")

    try:
        from warp_md import RotateDihedralPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "RotateDihedralPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(RotateDihedralPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "RotateDihedralPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel_a = _selection_from_item(system, atoms[0])
    sel_b = _selection_from_item(system, atoms[1])
    sel_c = _selection_from_item(system, atoms[2])
    sel_d = _selection_from_item(system, atoms[3])
    rotate_sel = sel_d if rotate_mask is None else _selection_from_mask(system, rotate_mask)
    plan = RotateDihedralPlan(sel_a, sel_b, sel_c, sel_d, rotate_sel, float(angle), mass, degrees)
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
            "rotate_dihedral requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    arr = np.asarray(flat, dtype=np.float32)
    out = _reshape_plan_coords(arr, system, traj, "rotate_dihedral plan output shape is inconsistent with atom count")
    return ArrayTrajectory(out.astype(np.float32, copy=False))


def set_dihedral(
    traj,
    system,
    atoms: Sequence[AtomLike],
    rotate_mask: Optional[MaskLike] = None,
    target: float = 0.0,
    mass: bool = False,
    pbc: str = "none",
    degrees: bool = True,
    range360: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Rotate selection to set dihedral to a target angle using Rust plan only."""
    if len(atoms) != 4:
        raise ValueError("atoms must contain 4 items")
    mass = _as_bool(mass, "mass")
    degrees = _as_bool(degrees, "degrees")
    range360 = _as_bool(range360, "range360")
    pbc = _normalize_pbc(pbc)

    try:
        from warp_md import SetDihedralPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "SetDihedralPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(SetDihedralPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "SetDihedralPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel_a = _selection_from_item(system, atoms[0])
    sel_b = _selection_from_item(system, atoms[1])
    sel_c = _selection_from_item(system, atoms[2])
    sel_d = _selection_from_item(system, atoms[3])
    rotate_sel = sel_d if rotate_mask is None else _selection_from_mask(system, rotate_mask)
    plan = SetDihedralPlan(
        sel_a,
        sel_b,
        sel_c,
        sel_d,
        rotate_sel,
        float(target),
        mass,
        pbc,
        degrees,
        range360,
    )
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
            "set_dihedral requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    arr = np.asarray(flat, dtype=np.float32)
    out = _reshape_plan_coords(arr, system, traj, "set_dihedral plan output shape is inconsistent with atom count")
    return ArrayTrajectory(out.astype(np.float32, copy=False))


__all__ = ["rotate_dihedral", "set_dihedral"]
