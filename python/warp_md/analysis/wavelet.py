# Usage:
# from warp_md.analysis.wavelet import wavelet
# details = wavelet(traj, system, selection_a=":1", selection_b=":10")

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _resolve_indices(system, selection: Union[str, Sequence[int], None]) -> np.ndarray:
    if selection is None or selection == "" or selection == "*":
        sel = system.select(_all_resid_mask(system))
        return np.asarray(list(sel.indices), dtype=np.int64)
    if isinstance(selection, str):
        sel = system.select(selection)
        return np.asarray(list(sel.indices), dtype=np.int64)
    return np.asarray([int(i) for i in selection], dtype=np.int64)


def _center(coords: np.ndarray, indices: np.ndarray, masses: Optional[np.ndarray], mass: bool) -> np.ndarray:
    sel = coords[indices]
    if sel.size == 0:
        return np.zeros(3, dtype=np.float64)
    if mass and masses is not None and masses.size > 0:
        w = masses[indices].astype(np.float64)
        wsum = w.sum()
        if wsum == 0.0:
            return sel.mean(axis=0)
        return (sel * w[:, None]).sum(axis=0) / wsum
    return sel.mean(axis=0)


def _haar_details(series: np.ndarray) -> np.ndarray:
    if series.size < 2:
        return np.empty((0, 0), dtype=np.float32)
    current = series.astype(np.float64)
    max_cols = current.size // 2
    details = []
    while current.size >= 2:
        n = current.size // 2
        a = current[0 : 2 * n : 2]
        b = current[1 : 2 * n : 2]
        next_level = 0.5 * (a + b)
        detail = 0.5 * (a - b)
        details.append(detail)
        current = next_level
    rows = len(details)
    out = np.zeros((rows, max_cols), dtype=np.float64)
    for r, detail in enumerate(details):
        out[r, : detail.size] = detail
    return out.astype(np.float32)


def wavelet(
    traj,
    system,
    selection_a: Union[str, Sequence[int], None],
    selection_b: Union[str, Sequence[int], None],
    mass: bool = False,
    pbc: str = "none",
    length_scale: float = 0.1,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    return_series: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Wavelet details from distance between two selections (Haar)."""
    pbc = pbc.lower()
    if pbc not in ("none", "orthorhombic"):
        raise ValueError("pbc must be 'none' or 'orthorhombic'")
    idx_a = _resolve_indices(system, selection_a)
    idx_b = _resolve_indices(system, selection_b)
    if idx_a.size == 0 or idx_b.size == 0:
        raise ValueError("selections must be non-empty")
    atoms = system.atom_table()
    masses = np.asarray(atoms.get("mass", []), dtype=np.float64)

    max_chunk = chunk_frames or 128
    chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    if chunk is None:
        raise ValueError("trajectory has no frames")

    if frame_indices is not None:
        frame_set = {int(i) for i in frame_indices if int(i) >= 0}
    else:
        frame_set = None
    global_frame = 0

    series = []
    while chunk is not None:
        coords = np.asarray(chunk["coords"], dtype=np.float64) * float(length_scale)
        box = chunk.get("box")
        if box is not None:
            box = np.asarray(box, dtype=np.float64) * float(length_scale)
        for f in range(coords.shape[0]):
            if frame_set is not None and global_frame not in frame_set:
                global_frame += 1
                continue
            pos = coords[f]
            a = _center(pos, idx_a, masses, mass)
            b = _center(pos, idx_b, masses, mass)
            dx, dy, dz = b - a
            if pbc == "orthorhombic":
                if box is None:
                    raise ValueError("pbc='orthorhombic' requires box in trajectory")
                bdim = box[f]
                if np.any(bdim == 0.0):
                    raise ValueError("box lengths must be non-zero for pbc")
                dx -= np.round(dx / bdim[0]) * bdim[0]
                dy -= np.round(dy / bdim[1]) * bdim[1]
                dz -= np.round(dz / bdim[2]) * bdim[2]
            series.append(np.sqrt(dx * dx + dy * dy + dz * dz))
            global_frame += 1
        chunk = read_chunk_fields(traj, max_chunk, include_box=True)

    series_arr = np.asarray(series, dtype=np.float32)
    details = _haar_details(series_arr)
    if return_series:
        return details, series_arr
    return details


__all__ = ["wavelet"]
