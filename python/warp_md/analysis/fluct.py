# Usage:
# from warp_md.analysis.fluct import rmsf, atomicfluct, bfactors
# data = rmsf(traj, system, mask="name CA", byres=True)

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._chunk_io import read_chunk_fields


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask) -> np.ndarray:
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
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    return coords


def _apply_frame_indices(coords: np.ndarray, frame_indices: Optional[Sequence[int]]) -> np.ndarray:
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


def _group_by_resid(system, indices: np.ndarray):
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        raise ValueError("system atom table has no resid data")
    resids = np.asarray(resids, dtype=np.int64)
    order = []
    groups = {}
    for pos, atom_idx in enumerate(indices.tolist()):
        if atom_idx >= resids.size:
            continue
        resid = int(resids[atom_idx])
        if resid not in groups:
            groups[resid] = []
            order.append(resid)
        groups[resid].append(pos)
    return order, groups


def _pack_indexed(indices: Sequence[int], values: Sequence[float]) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    if idx.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack((idx.astype(np.float32), vals.astype(np.float32)))


def _aggregate(values: np.ndarray, indices: np.ndarray, system, byres: bool, bymask: bool) -> np.ndarray:
    if bymask:
        mean_val = float(np.mean(values)) if values.size else 0.0
        return _pack_indexed([0], [mean_val])
    if byres:
        order, groups = _group_by_resid(system, indices)
        out_vals = []
        for resid in order:
            idx = groups.get(resid, [])
            if not idx:
                out_vals.append(0.0)
            else:
                out_vals.append(float(np.mean(values[idx])))
        return _pack_indexed(order, out_vals)
    return _pack_indexed(indices, values)


def rmsf(
    traj,
    system,
    mask: str = "",
    byres: bool = False,
    bymask: bool = False,
    calcadp: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Compute RMSF for selected atoms.

    Returns a 2-column array: [index, rmsf] or [resid, rmsf] if byres.
    If bymask, returns a single row with index 0.
    If calcadp, returns a 7-column array: [index, uxx, uyy, uzz, uxy, uxz, uyz].
    """
    if byres and bymask:
        raise ValueError("byres and bymask are mutually exclusive")
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        if calcadp:
            return np.empty((0, 7), dtype=np.float32)
        return np.empty((0, 2), dtype=np.float32)
    coords = _apply_frame_indices(coords, frame_indices)
    if coords.size == 0:
        if calcadp:
            return np.empty((0, 7), dtype=np.float32)
        return np.empty((0, 2), dtype=np.float32)
    coords = coords * float(length_scale)
    indices = _selection_indices(system, mask)
    if indices.size == 0:
        raise ValueError("selection resolved to empty set")
    sel_coords = coords[:, indices, :]
    mean = sel_coords.mean(axis=0)
    disp = sel_coords - mean
    if calcadp:
        uxx = np.mean(disp[:, :, 0] * disp[:, :, 0], axis=0)
        uyy = np.mean(disp[:, :, 1] * disp[:, :, 1], axis=0)
        uzz = np.mean(disp[:, :, 2] * disp[:, :, 2], axis=0)
        uxy = np.mean(disp[:, :, 0] * disp[:, :, 1], axis=0)
        uxz = np.mean(disp[:, :, 0] * disp[:, :, 2], axis=0)
        uyz = np.mean(disp[:, :, 1] * disp[:, :, 2], axis=0)
        idx = indices.astype(np.float32)
        out = np.column_stack((
            idx,
            uxx.astype(np.float32),
            uyy.astype(np.float32),
            uzz.astype(np.float32),
            uxy.astype(np.float32),
            uxz.astype(np.float32),
            uyz.astype(np.float32),
        ))
        return out
    rmsf_vals = np.sqrt(np.mean(np.sum(disp * disp, axis=2), axis=0))
    return _aggregate(rmsf_vals, indices, system, byres, bymask)


def atomicfluct(
    traj,
    system,
    mask: str = "",
    byres: bool = False,
    bymask: bool = False,
    calcadp: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Alias of RMSF with optional ADP output."""
    if calcadp and (byres or bymask):
        raise ValueError("calcadp output is only supported for byatom mode")
    return rmsf(
        traj,
        system,
        mask=mask,
        byres=byres,
        bymask=bymask,
        calcadp=calcadp,
        frame_indices=frame_indices,
        chunk_frames=chunk_frames,
        length_scale=length_scale,
    )


def bfactors(
    traj,
    system,
    mask: str = "",
    byres: bool = True,
    bymask: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Compute pseudo B-factors from RMSF.

    Returns a 2-column array: [index, bfactor] or [resid, bfactor] if byres.
    If bymask, returns a single row with index 0.
    """
    if byres and bymask:
        raise ValueError("byres and bymask are mutually exclusive")
    rmsf_data = rmsf(
        traj,
        system,
        mask=mask,
        byres=False,
        bymask=False,
        calcadp=False,
        frame_indices=frame_indices,
        chunk_frames=chunk_frames,
        length_scale=length_scale,
    )
    if rmsf_data.size == 0:
        return rmsf_data
    indices = rmsf_data[:, 0].astype(np.int64)
    rmsf_vals = rmsf_data[:, 1]
    factor = 8.0 * np.pi * np.pi / 3.0
    bvals = factor * rmsf_vals * rmsf_vals
    return _aggregate(bvals, indices, system, byres, bymask)


__all__ = ["rmsf", "atomicfluct", "bfactors"]
