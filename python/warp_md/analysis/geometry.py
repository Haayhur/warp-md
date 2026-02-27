# Usage:
# from warp_md.analysis.geometry import angle, dihedral
# vals = angle(traj, system, "@1 @2 @3")

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields


MaskLike = Union[str, Sequence[str], np.ndarray]


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: str) -> np.ndarray:
    if mask in ("", "*", "all", None):
        sel = system.select(_all_resid_mask(system))
        return np.asarray(list(sel.indices), dtype=np.int64)
    if isinstance(mask, str) and mask.strip().startswith("@"):
        toks = mask.replace(",", " ").split()
        idx = []
        for tok in toks:
            if tok.startswith("@") and tok[1:].lstrip("-").isdigit():
                idx.append(int(tok[1:]) - 1)
        if idx:
            return np.asarray(idx, dtype=np.int64)
    sel = system.select(mask)
    return np.asarray(list(sel.indices), dtype=np.int64)


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    chunk = read_chunk_fields(traj, max_chunk)
    if chunk is None:
        return None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk)
    return np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))


def _center(coords: np.ndarray, idx: np.ndarray, masses: Optional[np.ndarray], mass: bool) -> np.ndarray:
    sel = coords[:, idx, :]
    if sel.size == 0:
        raise ValueError("selection resolved to empty set")
    if mass and masses is not None and masses.size > 0:
        w = masses[idx]
        wsum = np.sum(w)
        if wsum <= 0.0:
            w = np.ones_like(w)
            wsum = np.sum(w)
        return (sel * w[None, :, None]).sum(axis=1) / wsum
    return sel.mean(axis=1)


def _angle_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot = np.einsum("ij,ij->i", v1, v2)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.where(denom > 0.0, dot / denom, 1.0)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _dihedral_from_points(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=1)
    b1n = b1 / b1_norm[:, None]
    v = b0 - (np.einsum("ij,ij->i", b0, b1n))[:, None] * b1n
    w = b2 - (np.einsum("ij,ij->i", b2, b1n))[:, None] * b1n
    x = np.einsum("ij,ij->i", v, w)
    y = np.einsum("ij,ij->i", np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


def angle(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    mass: bool = False,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Compute angle between three masks or index triplets."""
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32)

    n_frames = coords.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        coords = coords[select]
        n_frames = coords.shape[0]

    masses = None
    if mass:
        atoms = system.atom_table()
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)

    if isinstance(mask, str):
        parts = mask.split()
        if len(parts) != 3:
            raise ValueError("angle mask must have 3 parts")
        idx_a = _selection_indices(system, parts[0])
        idx_b = _selection_indices(system, parts[1])
        idx_c = _selection_indices(system, parts[2])
        a = _center(coords, idx_a, masses, mass)
        b = _center(coords, idx_b, masses, mass)
        c = _center(coords, idx_c, masses, mass)
        v1 = a - b
        v2 = c - b
        out = _angle_from_vectors(v1, v2).astype(np.float32)
        return out

    arr = np.asarray(mask)
    if arr.dtype.kind == "i":
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("angle index array must be shape (n, 3)")
        out = np.zeros((arr.shape[0], n_frames), dtype=np.float64)
        for i, (a, b, c) in enumerate(arr):
            pa = coords[:, a, :]
            pb = coords[:, b, :]
            pc = coords[:, c, :]
            v1 = pa - pb
            v2 = pc - pb
            out[i] = _angle_from_vectors(v1, v2)
        return out.astype(np.float32)

    # list of strings
    commands = list(mask)
    out = np.zeros((len(commands), n_frames), dtype=np.float64)
    for i, cmd in enumerate(commands):
        out[i] = angle(traj, system, cmd, frame_indices=frame_indices, mass=mass, chunk_frames=chunk_frames)
    return out.astype(np.float32)


def dihedral(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    mass: bool = False,
    range360: bool = False,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Compute dihedral angle between four masks or index quartets."""
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32)

    n_frames = coords.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        coords = coords[select]
        n_frames = coords.shape[0]

    masses = None
    if mass:
        atoms = system.atom_table()
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)

    if isinstance(mask, str):
        parts = mask.split()
        if len(parts) != 4:
            raise ValueError("dihedral mask must have 4 parts")
        idx_a = _selection_indices(system, parts[0])
        idx_b = _selection_indices(system, parts[1])
        idx_c = _selection_indices(system, parts[2])
        idx_d = _selection_indices(system, parts[3])
        a = _center(coords, idx_a, masses, mass)
        b = _center(coords, idx_b, masses, mass)
        c = _center(coords, idx_c, masses, mass)
        d = _center(coords, idx_d, masses, mass)
        out = _dihedral_from_points(a, b, c, d)
        if range360:
            out = np.where(out < 0.0, out + 360.0, out)
        return out.astype(np.float32)

    arr = np.asarray(mask)
    if arr.dtype.kind == "i":
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("dihedral index array must be shape (n, 4)")
        out = np.zeros((arr.shape[0], n_frames), dtype=np.float64)
        for i, (a, b, c, d) in enumerate(arr):
            pa = coords[:, a, :]
            pb = coords[:, b, :]
            pc = coords[:, c, :]
            pd = coords[:, d, :]
            vals = _dihedral_from_points(pa, pb, pc, pd)
            if range360:
                vals = np.where(vals < 0.0, vals + 360.0, vals)
            out[i] = vals
        return out.astype(np.float32)

    commands = list(mask)
    out = np.zeros((len(commands), n_frames), dtype=np.float64)
    for i, cmd in enumerate(commands):
        out[i] = dihedral(traj, system, cmd, frame_indices=frame_indices, mass=mass, range360=range360, chunk_frames=chunk_frames)
    return out.astype(np.float32)


__all__ = ["angle", "dihedral"]
