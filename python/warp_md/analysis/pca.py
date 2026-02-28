# Usage:
# from warp_md.analysis.pca import pca, projection
# proj, (evals, evecs) = pca(traj, system, mask="name CA", n_vecs=2)

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._chunk_io import read_chunk_fields
from .trajectory import ArrayTrajectory


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


def _kabsch_transform(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if x.shape != y.shape:
        raise ValueError("selection and reference selection must have same size")
    if x.size == 0:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    w = weights.astype(np.float64)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        w = np.ones_like(w)
        wsum = float(np.sum(w))
    w = w / wsum
    cx = np.sum(x * w[:, None], axis=0)
    cy = np.sum(y * w[:, None], axis=0)
    x0 = x - cx
    y0 = y - cy
    h = (x0 * w[:, None]).T @ y0
    u, _s, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = cy - r @ cx
    return r, t


def _apply_transform(coords: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return coords @ r.T + t


def _align_coords(
    coords: np.ndarray,
    ref_coords: np.ndarray,
    sel_idx: np.ndarray,
) -> np.ndarray:
    weights = np.ones(sel_idx.size, dtype=np.float64)
    aligned = np.empty_like(coords)
    for i in range(coords.shape[0]):
        r, t = _kabsch_transform(coords[i][sel_idx], ref_coords[sel_idx], weights)
        aligned[i] = _apply_transform(coords[i], r, t)
    return aligned


def _flatten_coords(coords: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return coords[:, idx, :].reshape(coords.shape[0], -1)


def _maybe_apply_mass(coords_flat: np.ndarray, masses: Optional[np.ndarray]) -> np.ndarray:
    if masses is None:
        return coords_flat
    w = np.sqrt(np.clip(masses, 0.0, None))
    w_rep = np.repeat(w, 3)
    return coords_flat * w_rep[None, :]


def projection(
    traj,
    system,
    mask: str = "",
    eigenvectors: Optional[np.ndarray] = None,
    eigenvalues: Optional[np.ndarray] = None,
    scalar_type: str = "covar",
    average_coords: Optional[np.ndarray] = None,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Project trajectory onto provided eigenvectors.

    Returns array shape (n_vecs, n_frames).
    """
    if eigenvectors is None or eigenvalues is None:
        raise ValueError("eigenvectors and eigenvalues are required")
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 0), dtype=np.float32)

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

    coords_flat = _flatten_coords(coords, idx)

    masses = None
    if scalar_type.lower() == "mwcovar":
        atoms = system.atom_table()
        masses_all = np.asarray(atoms.get("mass", []), dtype=np.float64)
        if masses_all.size > 0:
            masses = masses_all[idx]
    coords_flat = _maybe_apply_mass(coords_flat, masses)

    if average_coords is None:
        mean = coords_flat.mean(axis=0)
    else:
        avg = np.asarray(average_coords, dtype=np.float64)
        if avg.ndim != 2 or avg.shape[1] != 3:
            raise ValueError("average_coords must be (n_atoms, 3)")
        avg_flat = avg[idx, :].reshape(-1)
        avg_flat = _maybe_apply_mass(avg_flat[None, :], masses)[0]
        mean = avg_flat

    vecs = np.asarray(eigenvectors, dtype=np.float64)
    if vecs.ndim != 2:
        raise ValueError("eigenvectors must be 2D")
    if vecs.shape[0] != mean.shape[0] and vecs.shape[1] != mean.shape[0]:
        raise ValueError("eigenvectors shape does not match features")
    if vecs.shape[0] == mean.shape[0]:
        vecs = vecs.T

    centered = coords_flat - mean[None, :]
    proj = centered @ vecs.T
    proj = proj.T.astype(np.float32)
    if dtype == "ndarray":
        return proj
    return proj


def pca(
    traj,
    system,
    mask: str,
    n_vecs: int = 2,
    fit: bool = True,
    ref: Optional[object] = None,
    ref_mask: Optional[str] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
):
    """Perform PCA and return projection plus (eigenvalues, eigenvectors)."""
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 0), dtype=np.float32), (np.empty(0), np.empty((0, 0)))

    if fit:
        align_mask = ref_mask if ref_mask is not None else mask
        sel_align = _selection_indices(system, align_mask)
        if sel_align.size == 0:
            raise ValueError("alignment selection resolved to empty set")

        if ref is None:
            ref_coords = coords[0]
            coords = _align_coords(coords, ref_coords, sel_align)
            avg_coords = coords.mean(axis=0)
            coords = _align_coords(coords, avg_coords, sel_align)
        else:
            if isinstance(ref, int):
                ref_idx = ref
                if ref_idx < 0:
                    ref_idx = coords.shape[0] + ref_idx
                if ref_idx < 0 or ref_idx >= coords.shape[0]:
                    raise ValueError("ref index out of range")
                ref_coords = coords[ref_idx]
            else:
                ref_coords = np.asarray(ref, dtype=np.float64)
                if ref_coords.shape != coords[0].shape:
                    raise ValueError("ref coords must match frame shape")
            coords = _align_coords(coords, ref_coords, sel_align)

    idx = _selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    sel = _flatten_coords(coords, idx)
    mean = sel.mean(axis=0)
    x = sel - mean
    cov = (x.T @ x) / float(max(sel.shape[0], 1))

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    n_features = evals.size
    n_vecs = n_features if n_vecs < 0 else int(n_vecs)
    n_vecs = max(1, min(n_vecs, n_features))

    evals = evals[:n_vecs]
    evecs = evecs[:, :n_vecs].T

    proj = (x @ evecs.T).T.astype(np.float32)
    if dtype != "ndarray":
        proj = proj
    return proj, (evals.astype(np.float32), evecs.astype(np.float32))


__all__ = ["pca", "projection"]
