# Usage:
# from warp_md.analysis.density import density
# out = density(traj, system, mask=':WAT@O', density_type='number', delta=0.5, direction='z')

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields


MaskLike = Union[str, Sequence[str]]


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


def _weights_for_density(system, density_type: str, mass: bool) -> np.ndarray:
    atoms = system.atom_table()
    dtype = density_type.lower()
    if dtype == "number":
        return np.ones(len(atoms.get("name", [])), dtype=np.float64)
    if dtype == "mass":
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)
        if masses.size == 0:
            raise ValueError("mass density requested but atom masses are unavailable")
        if not mass:
            return np.ones_like(masses, dtype=np.float64)
        return masses
    if dtype == "charge":
        charges = np.asarray(atoms.get("charge", []), dtype=np.float64)
        if charges.size == 0:
            raise ValueError("charge density requested but atom charges are unavailable")
        return charges
    if dtype == "electron":
        atomic_numbers = atoms.get("atomic_number")
        if atomic_numbers is None or len(atomic_numbers) == 0:
            raise ValueError("electron density requested but atomic_number is unavailable")
        return np.asarray(atomic_numbers, dtype=np.float64)
    raise ValueError("density_type must be one of number, mass, charge, electron")


def _restrict_indices(system, restrict: Optional[str]) -> Optional[np.ndarray]:
    if restrict is None:
        return None
    idx = _selection_indices(system, restrict)
    return idx if idx.size else None


def _apply_restrict(mask_idx: np.ndarray, restrict_idx: Optional[np.ndarray]) -> np.ndarray:
    if restrict_idx is None or mask_idx.size == 0:
        return mask_idx
    restrict_set = set(int(i) for i in restrict_idx)
    keep = [i for i in mask_idx if int(i) in restrict_set]
    return np.asarray(keep, dtype=np.int64)


def _center_coords(coords: np.ndarray, indices: np.ndarray, weights: np.ndarray, use_mass: bool) -> np.ndarray:
    if indices.size == 0:
        return coords
    sel = coords[:, indices, :]
    if use_mass:
        w = weights[indices]
        wsum = float(w.sum())
        if wsum <= 0.0:
            w = np.ones_like(w)
            wsum = float(w.sum())
        center = (sel * w[None, :, None]).sum(axis=1) / wsum
    else:
        center = sel.mean(axis=1)
    return coords - center[:, None, :]


def density(
    traj,
    system,
    mask: MaskLike = "*",
    density_type: str = "number",
    delta: float = 0.25,
    direction: str = "z",
    cutoff: Optional[float] = None,
    center: bool = False,
    mass: bool = True,
    restrict: Optional[str] = None,
    dtype: str = "dict",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
):
    """Compute density (number/mass/charge/electron) along a coordinate axis.

    Returns a dict with density and std across frames.
    """
    density_kind = density_type.lower()
    if density_kind not in {"number", "mass", "charge", "electron"}:
        raise ValueError("density_type must be one of number, mass, charge, electron")

    axis = direction.lower()
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError("direction must be x, y, or z")
    axis_idx = axis_map[axis]

    if delta <= 0.0:
        raise ValueError("delta must be positive")

    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {"density": empty, "density_std": empty, axis: empty, "bins": empty, "n_frames": 0}

    coords = _select_frames(coords, frame_indices)
    if coords.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return {"density": empty, "density_std": empty, axis: empty, "bins": empty, "n_frames": 0}

    weights_all = _weights_for_density(system, density_kind, mass)
    restrict_idx = _restrict_indices(system, restrict)

    if isinstance(mask, str):
        masks = [mask]
    else:
        masks = list(mask)

    results = []
    for mask_item in masks:
        idx = _selection_indices(system, mask_item)
        idx = _apply_restrict(idx, restrict_idx)
        if idx.size == 0:
            empty = np.empty((0,), dtype=np.float32)
            results.append({"density": empty, "density_std": empty, axis: empty, "bins": empty, "n_frames": 0})
            continue

        work_coords = coords
        if center:
            work_coords = _center_coords(coords, idx, weights_all, mass)
        sel = work_coords[:, idx, :]

        if cutoff is not None:
            cutoff = float(cutoff)
            if cutoff <= 0.0:
                raise ValueError("cutoff must be positive")
            radii = np.linalg.norm(sel, axis=2)
            mask_ok = radii <= cutoff
        else:
            mask_ok = None

        axis_vals = sel[:, :, axis_idx]
        if mask_ok is not None:
            axis_vals = np.where(mask_ok, axis_vals, np.nan)

        min_val = np.nanmin(axis_vals)
        max_val = np.nanmax(axis_vals)
        if not np.isfinite(min_val) or not np.isfinite(max_val):
            empty = np.empty((0,), dtype=np.float32)
            results.append({"density": empty, "density_std": empty, axis: empty, "bins": empty, "n_frames": 0})
            continue

        edges = np.arange(min_val, max_val + delta, delta, dtype=np.float64)
        if edges.size < 2:
            edges = np.array([min_val, min_val + delta], dtype=np.float64)
        centers = (edges[:-1] + edges[1:]) * 0.5

        weights = weights_all[idx]
        counts = np.zeros((axis_vals.shape[0], centers.size), dtype=np.float64)
        for f in range(axis_vals.shape[0]):
            vals = axis_vals[f]
            if mask_ok is not None:
                valid = np.isfinite(vals)
                vals = vals[valid]
                frame_weights = weights[valid]
            else:
                frame_weights = weights
            if vals.size == 0:
                continue
            counts[f], _ = np.histogram(vals, bins=edges, weights=frame_weights)

        density_frame = counts / float(delta)
        mean = density_frame.mean(axis=0) if density_frame.size else np.empty((0,), dtype=np.float64)
        if density_frame.shape[0] > 1:
            std = density_frame.std(axis=0, ddof=1)
        else:
            std = np.zeros_like(mean)

        out = {
            "density": mean.astype(np.float32),
            "density_std": std.astype(np.float32),
            axis: centers.astype(np.float32),
            "bins": centers.astype(np.float32),
            "n_frames": int(density_frame.shape[0]),
            "density_type": density_kind,
        }
        results.append(out)

    if isinstance(mask, str):
        return results[0]
    if dtype == "dict":
        return results
    return results


__all__ = ["density"]
