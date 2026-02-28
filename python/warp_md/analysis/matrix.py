# Usage:
# from warp_md.analysis.matrix import covar, correl, dist, mwcovar
# mat = covar(traj, system, mask="name CA")

from __future__ import annotations

from typing import Optional

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


def covar(traj, system, mask: str = "", chunk_frames: Optional[int] = None) -> np.ndarray:
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    idx = _selection_indices(system, mask)
    sel = coords[:, idx, :].reshape(coords.shape[0], -1)
    mean = sel.mean(axis=0, keepdims=True)
    x = sel - mean
    cov = (x.T @ x) / float(max(sel.shape[0], 1))
    return cov.astype(np.float32)


def mwcovar(traj, system, mask: str = "", chunk_frames: Optional[int] = None) -> np.ndarray:
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    idx = _selection_indices(system, mask)
    atoms = system.atom_table()
    masses = np.asarray(atoms.get("mass", []), dtype=np.float64)
    if masses.size == 0:
        weights = np.ones(idx.size, dtype=np.float64)
    else:
        weights = np.clip(masses[idx], 0.0, None)
    w = np.sqrt(weights)
    sel = coords[:, idx, :] * w[None, :, None]
    sel = sel.reshape(coords.shape[0], -1)
    mean = sel.mean(axis=0, keepdims=True)
    x = sel - mean
    cov = (x.T @ x) / float(max(sel.shape[0], 1))
    return cov.astype(np.float32)


def dist(traj, system, mask: str = "", chunk_frames: Optional[int] = None) -> np.ndarray:
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    idx = _selection_indices(system, mask)
    sel = coords[:, idx, :]
    n_frames, n_atoms, _ = sel.shape
    out = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for f in range(n_frames):
        frame = sel[f]
        diff = frame[:, None, :] - frame[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        out += dist
    out /= float(max(n_frames, 1))
    return out.astype(np.float32)


def correl(traj, system, mask: str = "", chunk_frames: Optional[int] = None) -> np.ndarray:
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    idx = _selection_indices(system, mask)
    sel = coords[:, idx, :]
    n_frames, n_atoms, _ = sel.shape
    mean = sel.mean(axis=0)
    disp = sel - mean[None, :, :]
    var = np.mean(np.sum(disp * disp, axis=2), axis=0)
    denom = np.sqrt(np.outer(var, var))
    corr = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for f in range(n_frames):
        d = disp[f]
        corr += d @ d.T
    corr /= float(max(n_frames, 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0.0, corr / denom, 0.0)
    diag = np.diag(corr)
    for i, val in enumerate(diag):
        if denom[i, i] > 0.0:
            continue
        corr[i, i] = 1.0 if np.isfinite(val) else 0.0
    return corr.astype(np.float32)


__all__ = ["covar", "mwcovar", "dist", "correl"]
