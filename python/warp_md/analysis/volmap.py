# Usage:
# from warp_md.analysis.volmap import volmap
# grid = volmap(traj, system, mask=':WAT@O', grid_spacing=(0.5, 0.5, 0.5), buffer=2.0)

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields


MaskLike = Union[str, Sequence[str]]

_VDW_BY_SYMBOL = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
}

_VDW_BY_NUMBER = {
    1: 1.20,
    6: 1.70,
    7: 1.55,
    8: 1.52,
    9: 1.47,
    15: 1.80,
    16: 1.80,
    17: 1.75,
    35: 1.85,
    53: 1.98,
}


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: str) -> np.ndarray:
    if mask in ("", "*", "all", None):
        sel = system.select(_all_resid_mask(system))
    else:
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


def _select_frames(coords: np.ndarray, frame_indices: Optional[Sequence[int]]) -> np.ndarray:
    if frame_indices is None:
        return coords
    n_frames = coords.shape[0]
    select = []
    for i in frame_indices:
        j = int(i)
        if j < 0:
            j = n_frames + j
        if 0 <= j < n_frames:
            select.append(j)
    return coords[select]


def _atom_radii(system, indices: np.ndarray, default_radius: float = 1.5) -> np.ndarray:
    atoms = system.atom_table()
    if "vdw_radius" in atoms:
        radii = np.asarray(atoms.get("vdw_radius", []), dtype=np.float64)
    elif "radius" in atoms:
        radii = np.asarray(atoms.get("radius", []), dtype=np.float64)
    elif "atomic_number" in atoms:
        nums = np.asarray(atoms.get("atomic_number", []), dtype=np.int64)
        radii = np.array([_VDW_BY_NUMBER.get(int(n), default_radius) for n in nums], dtype=np.float64)
    elif "element" in atoms:
        elems = atoms.get("element", [])
        radii = np.array([
            _VDW_BY_SYMBOL.get(str(e).upper(), default_radius) for e in elems
        ], dtype=np.float64)
    else:
        radii = np.full(len(atoms.get("name", [])), default_radius, dtype=np.float64)
    if radii.size == 0:
        radii = np.full(len(atoms.get("name", [])), default_radius, dtype=np.float64)
    return radii[indices]


def _grid_from_bounds(origin: np.ndarray, size: np.ndarray, spacing: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges_x = np.arange(origin[0], origin[0] + size[0] + spacing[0] * 0.5, spacing[0])
    edges_y = np.arange(origin[1], origin[1] + size[1] + spacing[1] * 0.5, spacing[1])
    edges_z = np.arange(origin[2], origin[2] + size[2] + spacing[2] * 0.5, spacing[2])
    cx = (edges_x[:-1] + edges_x[1:]) * 0.5
    cy = (edges_y[:-1] + edges_y[1:]) * 0.5
    cz = (edges_z[:-1] + edges_z[1:]) * 0.5
    return cx, cy, cz


def volmap(
    traj,
    system,
    mask: str = "*",
    grid_spacing: Union[Tuple[float, float, float], float] = (0.5, 0.5, 0.5),
    size: Optional[Tuple[float, float, float]] = None,
    center: Optional[Tuple[float, float, float]] = None,
    buffer: float = 3.0,
    centermask: str = "*",
    radscale: float = 1.36,
    peakcut: float = 0.05,
    dtype: str = "ndarray",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    options: str = "",
):
    """Grid data as a volumetric map using a Gaussian atom density model."""
    _ = options
    if isinstance(grid_spacing, (int, float)):
        spacing = np.array([float(grid_spacing)] * 3, dtype=np.float64)
    else:
        if len(grid_spacing) != 3:
            raise ValueError("grid_spacing must be a tuple of length 3")
        spacing = np.array([float(x) for x in grid_spacing], dtype=np.float64)
    if np.any(spacing <= 0.0):
        raise ValueError("grid_spacing values must be positive")

    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        empty = np.empty((0, 0, 0), dtype=np.float32)
        return empty if dtype != "dict" else {"grid": empty, "origin": None, "spacing": spacing}

    coords = _select_frames(coords, frame_indices)
    if coords.size == 0:
        empty = np.empty((0, 0, 0), dtype=np.float32)
        return empty if dtype != "dict" else {"grid": empty, "origin": None, "spacing": spacing}

    sel_idx = _selection_indices(system, mask)
    if sel_idx.size == 0:
        empty = np.empty((0, 0, 0), dtype=np.float32)
        return empty if dtype != "dict" else {"grid": empty, "origin": None, "spacing": spacing}

    if size is not None:
        if len(size) != 3:
            raise ValueError("size must be a tuple of length 3")
        size_vec = np.array(size, dtype=np.float64)
        if center is None:
            center_vec = coords[:, sel_idx, :].mean(axis=(0, 1))
        else:
            center_vec = np.array(center, dtype=np.float64)
        origin = center_vec - 0.5 * size_vec
    else:
        center_idx = _selection_indices(system, centermask)
        if center_idx.size == 0:
            center_idx = sel_idx
        sel_coords = coords[:, center_idx, :]
        mins = sel_coords.min(axis=(0, 1))
        maxs = sel_coords.max(axis=(0, 1))
        origin = mins - float(buffer)
        size_vec = (maxs + float(buffer)) - origin

    cx, cy, cz = _grid_from_bounds(origin, size_vec, spacing)
    if cx.size == 0 or cy.size == 0 or cz.size == 0:
        empty = np.empty((0, 0, 0), dtype=np.float32)
        return empty if dtype != "dict" else {"grid": empty, "origin": origin, "spacing": spacing}

    grid = np.zeros((cx.size, cy.size, cz.size), dtype=np.float64)
    radii = _atom_radii(system, sel_idx)
    sigmas = radii / float(radscale) if radscale != 0 else radii
    sigmas = np.where(sigmas > 0.0, sigmas, 1.0)
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])

    n_frames = coords.shape[0]
    for frame in coords:
        frame_grid = np.zeros_like(grid)
        sel = frame[sel_idx]
        for atom_idx, pos in enumerate(sel):
            sigma = sigmas[atom_idx]
            cutoff = 3.0 * sigma
            min_bound = pos - cutoff
            max_bound = pos + cutoff
            ix0 = max(0, int(np.floor((min_bound[0] - origin[0]) / spacing[0])))
            iy0 = max(0, int(np.floor((min_bound[1] - origin[1]) / spacing[1])))
            iz0 = max(0, int(np.floor((min_bound[2] - origin[2]) / spacing[2])))
            ix1 = min(cx.size - 1, int(np.floor((max_bound[0] - origin[0]) / spacing[0])))
            iy1 = min(cy.size - 1, int(np.floor((max_bound[1] - origin[1]) / spacing[1])))
            iz1 = min(cz.size - 1, int(np.floor((max_bound[2] - origin[2]) / spacing[2])))
            if ix1 < ix0 or iy1 < iy0 or iz1 < iz0:
                continue

            xs = cx[ix0 : ix1 + 1]
            ys = cy[iy0 : iy1 + 1]
            zs = cz[iz0 : iz1 + 1]
            dx = xs - pos[0]
            dy = ys - pos[1]
            dz = zs - pos[2]
            gx = np.exp(-0.5 * (dx / sigma) ** 2)
            gy = np.exp(-0.5 * (dy / sigma) ** 2)
            gz = np.exp(-0.5 * (dz / sigma) ** 2)
            norm = 1.0 / ((2.0 * np.pi) ** 1.5 * sigma**3)
            contrib = norm * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
            frame_grid[ix0 : ix1 + 1, iy0 : iy1 + 1, iz0 : iz1 + 1] += contrib

        grid += frame_grid

    if n_frames > 0:
        grid /= float(n_frames)

    if 0.0 < peakcut < 1.0:
        threshold = peakcut * float(grid.max())
        grid = np.where(grid >= threshold, grid, 0.0)

    out_grid = grid.astype(np.float32)
    if dtype != "dict":
        return out_grid

    return {
        "grid": out_grid,
        "origin": origin.astype(np.float64),
        "spacing": spacing.astype(np.float64),
        "size": size_vec.astype(np.float64),
        "center": (origin + 0.5 * size_vec).astype(np.float64),
        "n_frames": int(n_frames),
        "voxel_volume": voxel_volume,
    }


__all__ = ["volmap"]
