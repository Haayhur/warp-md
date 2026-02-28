# Usage:
# from warp_md.analysis.set_velocity import set_velocity
# vel = set_velocity(traj, system, temperature=298.0, ig=10, mask="all")

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._chunk_io import read_chunk_fields


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: Optional[str]) -> np.ndarray:
    if mask in ("", "*", "all", None):
        mask = None
    if mask:
        sel = system.select(mask)
    else:
        sel = system.select(_all_resid_mask(system))
    return np.asarray(list(sel.indices), dtype=np.int64)


def set_velocity(
    traj,
    system,
    temperature: float = 298.0,
    ig: int = 10,
    mask: str = "",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Generate Maxwellian velocities for selected atoms.

    Returns velocity array shape (n_frames, n_sel, 3).
    """
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 0, 3), dtype=np.float32)

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

    idx = _selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    atoms = system.atom_table()
    masses = np.asarray(atoms.get("mass", []), dtype=np.float64)
    if masses.size == 0:
        masses = np.ones(coords.shape[1], dtype=np.float64)

    temp = float(max(0.0, temperature))
    rng = np.random.default_rng(int(ig))

    vel = np.zeros((n_frames, idx.size, 3), dtype=np.float64)
    for i, atom_idx in enumerate(idx):
        mass = max(float(masses[atom_idx]), 0.0)
        sigma = np.sqrt(temp / mass) if mass > 0.0 else 0.0
        vel[:, i, :] = rng.normal(0.0, sigma, size=(n_frames, 3))
    return vel.astype(np.float32)


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


__all__ = ["set_velocity"]
