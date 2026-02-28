# Usage:
# from warp_md.analysis.align import align, align_principal_axis
# aligned = align(traj, system, mask="protein", ref=0, mass=True)
# aligned_coords = aligned.read_chunk()["coords"]

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields
from .trajectory import ArrayTrajectory

RefLike = Union[int, str]


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: Optional[str]):
    if mask in ("", "*", "all", None):
        mask = None
    if mask:
        sel = system.select(mask)
    else:
        sel = system.select(_all_resid_mask(system))
    return np.asarray(list(sel.indices), dtype=np.int64)


def _mass_weights(system, indices: np.ndarray, mass: bool) -> np.ndarray:
    if not mass:
        return np.ones(indices.size, dtype=np.float64)
    atoms = system.atom_table()
    masses = atoms.get("mass", [])
    if not masses:
        return np.ones(indices.size, dtype=np.float64)
    w = np.asarray(masses, dtype=np.float64)
    return w[indices]


def _kabsch_transform(
    coords: np.ndarray,
    ref_coords: np.ndarray,
    sel_idx: np.ndarray,
    ref_idx: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    x = coords[sel_idx]
    y = ref_coords[ref_idx]
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


def _principal_axes_transform(
    coords: np.ndarray,
    sel_idx: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if sel_idx.size == 0:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)
    w = weights.astype(np.float64)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        w = np.ones_like(w)
        wsum = float(np.sum(w))
    w = w / wsum
    center = np.sum(coords[sel_idx] * w[:, None], axis=0)
    rel = coords[sel_idx] - center
    x = rel[:, 0]
    y = rel[:, 1]
    z = rel[:, 2]
    i_xx = np.sum(w * (y * y + z * z))
    i_yy = np.sum(w * (x * x + z * z))
    i_zz = np.sum(w * (x * x + y * y))
    i_xy = -np.sum(w * x * y)
    i_xz = -np.sum(w * x * z)
    i_yz = -np.sum(w * y * z)
    inertia = np.array(
        [[i_xx, i_xy, i_xz], [i_xy, i_yy, i_yz], [i_xz, i_yz, i_zz]],
        dtype=np.float64,
    )
    vals, vecs = np.linalg.eigh(inertia)
    order = np.argsort(vals)
    axes = vecs[:, order]
    r = axes.T
    t = -r @ center
    return r, t


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list: List[np.ndarray] = []
    box_list: List[np.ndarray] = []
    time_list: List[np.ndarray] = []
    chunk = read_chunk_fields(traj, max_chunk, include_box=True, include_time=True)
    if chunk is None:
        return None, None, None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        box = chunk.get("box")
        if box is not None:
            box_list.append(np.asarray(box, dtype=np.float64))
        time = chunk.get("time")
        if time is None:
            time = chunk.get("time_ps")
        if time is not None:
            time_list.append(np.asarray(time, dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk, include_box=True, include_time=True)
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    box = np.concatenate(box_list, axis=0) if box_list else None
    time = np.concatenate(time_list, axis=0) if time_list else None
    return coords, box, time


def align(
    traj,
    system,
    mask: str = "",
    ref: RefLike = 0,
    ref_mask: Optional[str] = None,
    mass: bool = False,
    chunk_frames: Optional[int] = None,
    return_transforms: bool = False,
):
    """Align trajectory frames to a reference via Kabsch."""
    sel_idx = _selection_indices(system, mask)
    ref_mask = mask if ref_mask is None else ref_mask
    ref_idx = _selection_indices(system, ref_mask)
    if sel_idx.size != ref_idx.size:
        raise ValueError("mask and ref_mask selections must have same size")
    weights = _mass_weights(system, sel_idx, mass)

    ref_coords = None
    ref_kind = ref
    if isinstance(ref, str):
        key = ref.strip().lower()
        if key in ("topology", "top", "topo"):
            if hasattr(system, "positions0"):
                ref_coords = system.positions0()
        if ref_coords is None and key in ("frame0", "first", "0", "topology", "top", "topo"):
            ref_kind = 0

    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    n_frames = coords.shape[0]
    if n_frames == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)

    if ref_coords is None:
        if isinstance(ref_kind, int):
            if ref_kind < 0:
                ref_index = n_frames + ref_kind
            else:
                ref_index = ref_kind
            if ref_index < 0 or ref_index >= n_frames:
                raise ValueError("ref index out of range")
            ref_coords = coords[ref_index]
        else:
            ref_coords = coords[0]
    else:
        ref_coords = np.asarray(ref_coords, dtype=np.float64)

    aligned = np.empty_like(coords)
    transforms = [] if return_transforms else None
    for i in range(n_frames):
        r, t = _kabsch_transform(coords[i], ref_coords, sel_idx, ref_idx, weights)
        aligned[i] = _apply_transform(coords[i], r, t)
        if return_transforms:
            transforms.append(
                np.array(
                    [
                        r[0, 0], r[0, 1], r[0, 2],
                        r[1, 0], r[1, 1], r[1, 2],
                        r[2, 0], r[2, 1], r[2, 2],
                        t[0], t[1], t[2],
                    ],
                    dtype=np.float32,
                )
            )
    out = ArrayTrajectory(aligned.astype(np.float32), box=box, time_ps=time)
    if return_transforms:
        return out, np.vstack(transforms)
    return out


def superpose(
    traj,
    system,
    mask: str = "",
    ref: RefLike = 0,
    ref_mask: Optional[str] = None,
    mass: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
):
    """Superpose trajectory frames to a reference (returns aligned coords)."""
    sel_idx = _selection_indices(system, mask)
    ref_mask = mask if (ref_mask is None or ref_mask == "") else ref_mask
    ref_idx_sel = _selection_indices(system, ref_mask)
    if sel_idx.size != ref_idx_sel.size:
        raise ValueError("mask and ref_mask selections must have same size")
    weights = _mass_weights(system, sel_idx, mass)

    ref_coords = None
    ref_kind = ref
    if isinstance(ref, str):
        key = ref.strip().lower()
        if key in ("topology", "top", "topo"):
            if hasattr(system, "positions0"):
                ref_coords = system.positions0()
        if ref_coords is None and key in ("frame0", "first", "0", "topology", "top", "topo"):
            ref_kind = 0

    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    n_frames = coords.shape[0]
    if n_frames == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)

    if ref_coords is None:
        if isinstance(ref_kind, int):
            if ref_kind < 0:
                ref_index = n_frames + ref_kind
            else:
                ref_index = ref_kind
            if ref_index < 0 or ref_index >= n_frames:
                raise ValueError("ref index out of range")
            ref_coords = coords[ref_index]
        else:
            ref_coords = coords[0]
    else:
        ref_coords = np.asarray(ref_coords, dtype=np.float64)

    frame_set = None
    if frame_indices is not None:
        frame_set = set()
        for idx in frame_indices:
            i = int(idx)
            if i < 0:
                i = n_frames + i
            if 0 <= i < n_frames:
                frame_set.add(i)

    aligned = coords.copy()
    for i in range(n_frames):
        if frame_set is not None and i not in frame_set:
            continue
        r, t = _kabsch_transform(coords[i], ref_coords, sel_idx, ref_idx_sel, weights)
        aligned[i] = _apply_transform(coords[i], r, t)
    return ArrayTrajectory(aligned.astype(np.float32), box=box, time_ps=time)


def align_principal_axis(
    traj,
    system,
    mask: str = "protein",
    mass: bool = False,
    chunk_frames: Optional[int] = None,
    return_transforms: bool = False,
):
    """Align trajectory frames to principal axes."""
    sel_idx = _selection_indices(system, mask)
    weights = _mass_weights(system, sel_idx, mass)

    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    n_frames = coords.shape[0]
    if n_frames == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)

    aligned = np.empty_like(coords)
    transforms = [] if return_transforms else None
    for i in range(n_frames):
        r, t = _principal_axes_transform(coords[i], sel_idx, weights)
        aligned[i] = _apply_transform(coords[i], r, t)
        if return_transforms:
            transforms.append(
                np.array(
                    [
                        r[0, 0], r[0, 1], r[0, 2],
                        r[1, 0], r[1, 1], r[1, 2],
                        r[2, 0], r[2, 1], r[2, 2],
                        t[0], t[1], t[2],
                    ],
                    dtype=np.float32,
                )
            )
    out = ArrayTrajectory(aligned.astype(np.float32), box=box, time_ps=time)
    if return_transforms:
        return out, np.vstack(transforms)
    return out
