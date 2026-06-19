# Usage:
# from warp_md.analysis.lipidscd import lipidscd
# out = lipidscd(traj, system, selection="resname POPC and name C*", axis="z")

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import warp_md

from ._chunk_io import read_chunk_fields

PairLike = Union[Tuple[int, int], Sequence[int]]


def _axis_vector(axis: Union[str, Sequence[float]]) -> np.ndarray:
    if isinstance(axis, str):
        axis = axis.lower()
        if axis == "x":
            vec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        elif axis == "y":
            vec = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        elif axis == "z":
            vec = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        else:
            raise ValueError("axis must be 'x', 'y', 'z', or a 3-vector")
    else:
        vec = np.asarray(axis, dtype=np.float64)
        if vec.shape != (3,):
            raise ValueError("axis must be 'x', 'y', 'z', or a 3-vector")
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        raise ValueError("axis vector must be non-zero")
    return vec / norm


def _lipid_scd_chunk(
    coords: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    axis_vec: np.ndarray,
    pbc_mode: str,
    box: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    fn = getattr(warp_md, "lipid_scd_chunk_array", None)
    if fn is None or getattr(fn, "__name__", "") != "lipid_scd_chunk_array":
        raise RuntimeError("lipid_scd_chunk_array native binding unavailable")
    scd, valid = fn(
        np.asarray(coords, dtype=np.float64),
        np.asarray(idx_a, dtype=np.int64),
        np.asarray(idx_b, dtype=np.int64),
        np.asarray(axis_vec, dtype=np.float64),
        pbc_mode,
        None if box is None else np.asarray(box, dtype=np.float64),
    )
    return np.asarray(scd, dtype=np.float32), np.asarray(valid, dtype=np.int64)


def _resolve_pairs(
    system,
    selection: Optional[str],
    pairs: Optional[Sequence[PairLike]],
    pair_mode: str,
) -> List[Tuple[int, int]]:
    pair_mode = pair_mode.lower()
    if pair_mode not in ("residue", "global"):
        raise ValueError("pair_mode must be 'residue' or 'global'")

    def pairs_from_indices(indices: List[int]) -> List[Tuple[int, int]]:
        if len(indices) < 2:
            return []
        if pair_mode == "global":
            return [(int(indices[i]), int(indices[i + 1])) for i in range(len(indices) - 1)]
        atoms = system.atom_table()
        resid = atoms.get("resid", [])
        resname = atoms.get("resname", [])
        chain_id = atoms.get("chain_id", [])
        groups: dict = {}
        for idx in indices:
            key = (
                chain_id[idx] if idx < len(chain_id) else 0,
                resid[idx] if idx < len(resid) else 0,
                resname[idx] if idx < len(resname) else "RES",
            )
            groups.setdefault(key, []).append(idx)
        out: List[Tuple[int, int]] = []
        for _, grp in groups.items():
            grp_sorted = sorted(grp)
            if len(grp_sorted) < 2:
                continue
            out.extend(
                [(int(grp_sorted[i]), int(grp_sorted[i + 1])) for i in range(len(grp_sorted) - 1)]
            )
        return out

    if pairs is None:
        if not selection:
            raise ValueError("selection is required when pairs is None")
        sel = system.select(selection)
        indices = list(sel.indices)
        if len(indices) < 2:
            raise ValueError("selection must contain at least two atoms")
        resolved = pairs_from_indices(indices)
        if not resolved:
            raise ValueError("selection resolved to empty pair list")
        return resolved

    if isinstance(pairs, str):
        if pairs.lower() != "sequential":
            raise ValueError("pairs string must be 'sequential'")
        if not selection:
            raise ValueError("selection is required for sequential pairs")
        sel = system.select(selection)
        indices = list(sel.indices)
        if len(indices) < 2:
            raise ValueError("selection must contain at least two atoms")
        resolved = pairs_from_indices(indices)
        if not resolved:
            raise ValueError("selection resolved to empty pair list")
        return resolved

    resolved: List[Tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each pair must have length 2")
        a, b = pair
        if isinstance(a, str) and isinstance(b, str):
            sel_a = system.select(a)
            sel_b = system.select(b)
            if len(sel_a.indices) != len(sel_b.indices):
                raise ValueError("selection pairs must match in length")
            resolved.extend(
                list(zip((int(i) for i in sel_a.indices), (int(i) for i in sel_b.indices)))
            )
        else:
            resolved.append((int(a), int(b)))
    if not resolved:
        raise ValueError("pairs resolved to empty list")
    return resolved


def lipidscd(
    traj,
    system,
    selection: Optional[str] = None,
    pairs: Optional[Sequence[PairLike]] = None,
    axis: Union[str, Sequence[float]] = "z",
    length_scale: float = 0.1,
    pbc: str = "none",
    pair_mode: str = "residue",
    frame_indices: Optional[Sequence[int]] = None,
    group_by: Optional[str] = None,
    per_frame: bool = False,
    chunk_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
):
    """Compute lipid order parameters (SCD) for bond vectors.

    Parameters
    ----------
    selection : str, optional
        Selection used when pairs is None or "sequential".
    pairs : list of (i, j) or list of selection-string pairs
        Explicit bond pairs.
    axis : 'x'|'y'|'z' or 3-vector
        Bilayer normal.
    length_scale : float
        Coordinate scale to nm (if needed).
    pair_mode : 'residue'|'global'
        Sequential pairing within each residue or across selection.
    frame_indices : list of int, optional
        Only evaluate selected frame indices.
    per_frame : bool
        Return per-frame SCD instead of average.
    """
    axis_vec = _axis_vector(axis)
    pbc_mode = pbc.lower()
    if pbc_mode not in ("none", "orthorhombic"):
        raise ValueError("pbc must be 'none' or 'orthorhombic'")
    pair_list = _resolve_pairs(system, selection, pairs, pair_mode)
    idx_a = np.array([p[0] for p in pair_list], dtype=np.int64)
    idx_b = np.array([p[1] for p in pair_list], dtype=np.int64)

    n_pairs = len(pair_list)
    max_chunk = chunk_frames or 128
    chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    if chunk is None:
        raise ValueError("trajectory has no frames")

    sums = np.zeros(n_pairs, dtype=np.float64)
    counts = np.zeros(n_pairs, dtype=np.int64)
    frames_out = []
    n_frames = 0

    if frame_indices is not None:
        frame_set = {int(i) for i in frame_indices if int(i) >= 0}
    else:
        frame_set = None
    global_frame = 0

    while chunk is not None:
        coords = np.asarray(chunk["coords"], dtype=np.float64) * length_scale
        box = chunk.get("box")
        if box is not None:
            box = np.asarray(box, dtype=np.float64) * length_scale
        frames = coords.shape[0]
        if frame_set is None:
            local_indices = np.arange(frames, dtype=np.int64)
        else:
            local_indices = np.array(
                [f for f in range(frames) if global_frame + f in frame_set],
                dtype=np.int64,
            )
        global_frame += frames
        if max_frames is not None:
            remaining = max_frames - n_frames
            if remaining <= 0:
                break
            local_indices = local_indices[:remaining]
        if local_indices.size > 0:
            selected_coords = coords[local_indices]
            selected_box = None if box is None else box[local_indices]
            scd, valid = _lipid_scd_chunk(selected_coords, idx_a, idx_b, axis_vec, pbc_mode, selected_box)
            sums += (scd.astype(np.float64) * valid).sum(axis=0)
            counts += valid.sum(axis=0)
            if per_frame:
                frames_out.append(scd.astype(np.float32, copy=False))
            n_frames += scd.shape[0]
        if max_frames is not None and n_frames >= max_frames:
            break
        chunk = read_chunk_fields(traj, max_chunk, include_box=True)

    if per_frame:
        scd_out = np.vstack(frames_out) if frames_out else np.empty((0, n_pairs), dtype=np.float32)
    else:
        avg = np.where(counts > 0, sums / counts, 0.0)
        scd_out = avg.astype(np.float32)
    out = {
        "bond_indices": np.stack([idx_a, idx_b], axis=1),
        "scd": scd_out,
    }
    if group_by:
        atoms = system.atom_table()
        resid = atoms["resid"]
        resname = atoms["resname"]
        chain_id = atoms["chain_id"]
        group_map = {}
        for i, a in enumerate(idx_a):
            if group_by == "resid":
                key = resid[a]
            elif group_by == "resname":
                key = resname[a]
            elif group_by == "chain":
                key = chain_id[a]
            elif group_by in ("resid_chain", "chain_resid"):
                key = (chain_id[a], resid[a])
            else:
                raise ValueError("group_by must be resid, resname, chain, or resid_chain")
            group_map.setdefault(key, []).append(i)
        keys = list(group_map.keys())
        if per_frame:
            grouped = np.zeros((scd_out.shape[0], len(keys)), dtype=np.float32)
            for gi, key in enumerate(keys):
                grouped[:, gi] = scd_out[:, group_map[key]].mean(axis=1)
        else:
            grouped = np.zeros(len(keys), dtype=np.float32)
            for gi, key in enumerate(keys):
                grouped[gi] = scd_out[group_map[key]].mean()
        out["group_keys"] = keys
        out["scd_grouped"] = grouped
    return out


__all__ = ["lipidscd"]
