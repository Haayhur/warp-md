# Usage:
# from warp_md.analysis.rmsd import distance_rmsd
# vals = distance_rmsd(traj, system, mask="name CA", ref=0, pbc="none")

from __future__ import annotations

from typing import Optional, Tuple

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
    box_list = []
    chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    if chunk is None:
        return None, None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        box = chunk.get("box")
        if box is not None:
            box_list.append(np.asarray(box, dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    box = np.concatenate(box_list, axis=0) if box_list else None
    return coords, box


def _kabsch_rmsd(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.shape != y.shape:
        raise ValueError("frame shapes must match for RMSD")
    cx = x.mean(axis=0)
    cy = y.mean(axis=0)
    x0 = x - cx
    y0 = y - cy
    h = x0.T @ y0
    u, _s, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    x_rot = x0 @ r.T
    diff = x_rot - y0
    return float(np.sqrt((diff * diff).sum() / x.shape[0]))


def _rmsd_raw(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    diff = x - y
    return float(np.sqrt((diff * diff).sum() / x.shape[0]))


def _pair_distances_compact(coords: np.ndarray, pbc: str, box: Optional[np.ndarray]) -> np.ndarray:
    n_atoms = coords.shape[0]
    n_pairs = n_atoms * (n_atoms - 1) // 2
    out = np.empty(n_pairs, dtype=np.float64)
    if pbc == "orthorhombic":
        if box is None or np.any(box == 0.0):
            raise ValueError("pbc='orthorhombic' requires box lengths")
    k = 0
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            dx, dy, dz = coords[j] - coords[i]
            if pbc == "orthorhombic":
                dx -= np.round(dx / box[0]) * box[0]
                dy -= np.round(dy / box[1]) * box[1]
                dz -= np.round(dz / box[2]) * box[2]
            out[k] = np.sqrt(dx * dx + dy * dy + dz * dz)
            k += 1
    return out


def _pair_distances(
    coords: np.ndarray,
    indices: np.ndarray,
    pbc: str,
    box: Optional[np.ndarray],
) -> np.ndarray:
    n_sel = indices.size
    n_pairs = n_sel * (n_sel - 1) // 2
    if n_pairs == 0:
        return np.empty(0, dtype=np.float64)
    out = np.empty(n_pairs, dtype=np.float64)
    k = 0
    if pbc == "orthorhombic":
        if box is None or np.any(box == 0.0):
            raise ValueError("pbc='orthorhombic' requires box lengths")
    for i in range(n_sel):
        a = coords[indices[i]]
        for j in range(i + 1, n_sel):
            b = coords[indices[j]]
            dx, dy, dz = b - a
            if pbc == "orthorhombic":
                dx -= np.round(dx / box[0]) * box[0]
                dy -= np.round(dy / box[1]) * box[1]
                dz -= np.round(dz / box[2]) * box[2]
            out[k] = np.sqrt(dx * dx + dy * dy + dz * dz)
            k += 1
    return out


def distance_rmsd(
    traj,
    system,
    mask: str = "",
    ref: int = 0,
    pbc: str = "none",
    length_scale: float = 1.0,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Distance RMSD vs a reference frame."""
    pbc = pbc.lower()
    if pbc not in ("none", "orthorhombic"):
        raise ValueError("pbc must be 'none' or 'orthorhombic'")
    indices = _selection_indices(system, mask)
    coords, box = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty(0, dtype=np.float32)
    coords = coords * float(length_scale)
    if box is not None:
        box = box * float(length_scale)

    n_frames = coords.shape[0]
    ref_idx = ref
    if ref_idx < 0:
        ref_idx = n_frames + ref_idx
    if ref_idx < 0 or ref_idx >= n_frames:
        raise ValueError("ref index out of range")

    ref_box = None if box is None else box[ref_idx]
    ref_dists = _pair_distances(coords[ref_idx], indices, pbc, ref_box)
    if ref_dists.size == 0:
        return np.zeros(n_frames, dtype=np.float32)

    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame_box = None if box is None else box[i]
        dists = _pair_distances(coords[i], indices, pbc, frame_box)
        diff = dists - ref_dists
        out[i] = np.sqrt(np.mean(diff * diff))
    return out.astype(np.float32)


def pairwise_rmsd(
    traj,
    system,
    mask: str = "",
    metric: str = "rms",
    mat_type: str = "full",
    pbc: str = "none",
    length_scale: float = 1.0,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Compute pairwise RMSD matrix (rms, nofit, dme, srmsd)."""
    metric = metric.lower()
    if metric == "srmsd":
        metric = "rms"
    if metric not in ("rms", "nofit", "dme"):
        raise ValueError("metric must be 'rms', 'nofit', 'dme', or 'srmsd'")
    mat_type = mat_type.lower()
    if mat_type not in ("full", "half"):
        raise ValueError("mat_type must be 'full' or 'half'")
    pbc = pbc.lower()
    if pbc not in ("none", "orthorhombic"):
        raise ValueError("pbc must be 'none' or 'orthorhombic'")

    coords, box = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords = coords * float(length_scale)
    if box is not None:
        box = box * float(length_scale)

    idx = _selection_indices(system, mask)
    frames = coords[:, idx, :]

    n_frames = frames.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        frames = frames[select]
        if box is not None:
            box = box[select]
        n_frames = frames.shape[0]

    if n_frames == 0:
        return np.empty((0, 0), dtype=np.float32) if mat_type == "full" else np.empty((0,), dtype=np.float32)

    if metric == "dme":
        dist_vecs = []
        for i in range(n_frames):
            dist_vecs.append(_pair_distances_compact(frames[i], pbc, None if box is None else box[i]))
        dist_vecs = np.asarray(dist_vecs, dtype=np.float64)

    if mat_type == "full":
        out = np.zeros((n_frames, n_frames), dtype=np.float64)
        for i in range(n_frames):
            for j in range(i + 1, n_frames):
                if metric == "rms":
                    val = _kabsch_rmsd(frames[i], frames[j])
                elif metric == "nofit":
                    val = _rmsd_raw(frames[i], frames[j])
                else:
                    diff = dist_vecs[i] - dist_vecs[j]
                    val = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
                out[i, j] = val
                out[j, i] = val
        return out.astype(np.float32)

    # half
    n_pairs = n_frames * (n_frames - 1) // 2
    out = np.zeros(n_pairs, dtype=np.float64)
    k = 0
    for i in range(n_frames):
        for j in range(i + 1, n_frames):
            if metric == "rms":
                val = _kabsch_rmsd(frames[i], frames[j])
            elif metric == "nofit":
                val = _rmsd_raw(frames[i], frames[j])
            else:
                diff = dist_vecs[i] - dist_vecs[j]
                val = float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0
            out[k] = val
            k += 1
    return out.astype(np.float32)
