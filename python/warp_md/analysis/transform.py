# Usage:
# from warp_md.analysis.transform import center, translate, rotate, scale, transform
# out = center(traj, system, mask="protein", mode="origin")

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

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
    if mask:
        sel = system.select(mask)
    else:
        sel = system.select(_all_resid_mask(system))
    return np.asarray(list(sel.indices), dtype=np.int64)


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    box_list = []
    time_list = []
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


def _as_vec3(value: Union[Sequence[float], np.ndarray], name: str) -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64)
    if vec.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector")
    return vec


def _as_mat3(value: Union[Sequence[float], np.ndarray], name: str) -> np.ndarray:
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape == (9,):
        mat = mat.reshape(3, 3)
    if mat.shape != (3, 3):
        raise ValueError(f"{name} must be a 3x3 matrix")
    return mat


def translate(traj, delta: Sequence[float], chunk_frames: Optional[int] = None):
    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    delta = _as_vec3(delta, "delta")
    out = coords + delta[None, None, :]
    return ArrayTrajectory(out.astype(np.float32), box=box, time_ps=time)


def scale(traj, factor: Union[float, Sequence[float]], chunk_frames: Optional[int] = None):
    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if isinstance(factor, (float, int)):
        vec = np.array([float(factor)] * 3, dtype=np.float64)
    else:
        vec = _as_vec3(factor, "factor")
    out = coords * vec[None, None, :]
    return ArrayTrajectory(out.astype(np.float32), box=box, time_ps=time)


def rotate(traj, rotation: Sequence[float], chunk_frames: Optional[int] = None):
    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    r = _as_mat3(rotation, "rotation")
    out = coords @ r.T
    return ArrayTrajectory(out.astype(np.float32), box=box, time_ps=time)


def transform(
    traj,
    rotation: Optional[Sequence[float]] = None,
    translation: Optional[Sequence[float]] = None,
    chunk_frames: Optional[int] = None,
):
    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    out = coords
    if rotation is not None:
        r = _as_mat3(rotation, "rotation")
        out = out @ r.T
    if translation is not None:
        t = _as_vec3(translation, "translation")
        out = out + t[None, None, :]
    return ArrayTrajectory(out.astype(np.float32), box=box, time_ps=time)


def center(
    traj,
    system,
    mask: str = "",
    mode: str = "origin",
    point: Optional[Sequence[float]] = None,
    mass: bool = False,
    chunk_frames: Optional[int] = None,
):
    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return ArrayTrajectory(coords.astype(np.float32), box=box, time_ps=time)
    indices = _selection_indices(system, mask)
    if indices.size == 0:
        raise ValueError("selection resolved to empty set")
    sel = coords[:, indices, :]
    if mass:
        atoms = system.atom_table()
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)
        if masses.size == 0:
            weights = np.ones(indices.size, dtype=np.float64)
        else:
            weights = masses[indices]
        wsum = weights.sum()
        if wsum == 0.0:
            weights = np.ones(indices.size, dtype=np.float64)
            wsum = weights.sum()
        com = (sel * weights[None, :, None]).sum(axis=1) / wsum
    else:
        com = sel.mean(axis=1)

    mode = mode.lower()
    if mode == "origin":
        target = np.zeros_like(com)
    elif mode == "point":
        if point is None:
            raise ValueError("point is required when mode='point'")
        target = np.tile(_as_vec3(point, "point"), (coords.shape[0], 1))
    elif mode == "box":
        if box is None:
            raise ValueError("box lengths required when mode='box'")
        target = box * 0.5
    else:
        raise ValueError("mode must be 'origin', 'point', or 'box'")

    shift = target - com
    out = coords + shift[:, None, :]
    return ArrayTrajectory(out.astype(np.float32), box=box, time_ps=time)


__all__ = ["center", "translate", "rotate", "scale", "transform"]
