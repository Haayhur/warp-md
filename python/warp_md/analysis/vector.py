# Usage:
# from warp_md.analysis.vector import vector, vector_mask
# v = vector(traj, system, "@CA @CB")

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields
from .trajectory import ArrayTrajectory

CommandLike = Union[str, Sequence[str]]


class _DummySelection:
    def __init__(self, indices: Sequence[int]):
        self.indices = indices


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: str) -> np.ndarray:
    if mask in ("", "*", "all", None):
        sel = system.select(_all_resid_mask(system))
        return np.asarray(list(sel.indices), dtype=np.int64)
    if isinstance(mask, str) and mask.strip().startswith("@"):
        tokens = mask.replace(",", " ").split()
        idx = []
        for tok in tokens:
            if tok.startswith("@") and tok[1:].lstrip("-").isdigit():
                val = int(tok[1:]) - 1
                idx.append(val)
        if idx:
            return np.asarray(idx, dtype=np.int64)
    sel = system.select(mask)
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


def _center(coords: np.ndarray, idx: np.ndarray, masses: Optional[np.ndarray], mass: bool) -> np.ndarray:
    sel = coords[:, idx, :]
    if sel.size == 0:
        return np.zeros((coords.shape[0], 3), dtype=np.float64)
    if mass and masses is not None and masses.size > 0:
        w = masses[idx]
        wsum = np.sum(w)
        if wsum <= 0.0:
            w = np.ones_like(w)
            wsum = np.sum(w)
        return (sel * w[None, :, None]).sum(axis=1) / wsum
    return sel.mean(axis=1)


def _apply_pbc(vec: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
    if box is None:
        return vec
    out = vec.copy()
    for f in range(out.shape[0]):
        lx, ly, lz = box[f]
        if lx > 0:
            out[f, 0] -= np.round(out[f, 0] / lx) * lx
        if ly > 0:
            out[f, 1] -= np.round(out[f, 1] / ly) * ly
        if lz > 0:
            out[f, 2] -= np.round(out[f, 2] / lz) * lz
    return out


def _parse_vector_command(cmd: str) -> Tuple[str, dict]:
    tokens = cmd.split()
    if not tokens:
        raise ValueError("command is empty")
    head = tokens[0].lower()
    opts = {"mass": False, "pbc": "none"}
    if head == "center":
        opts["mode"] = "center"
        opts["mask"] = " ".join(tokens[1:]) if len(tokens) > 1 else ""
        if "mass" in tokens:
            opts["mass"] = True
        return head, opts
    if head in ("box", "boxcenter", "ucellx", "ucelly", "ucellz"):
        opts["mode"] = head
        return head, opts
    # default: assume two masks
    opts["mode"] = "mask"
    if len(tokens) < 2:
        raise ValueError("vector command requires two masks")
    opts["mask_a"] = tokens[0]
    opts["mask_b"] = tokens[1]
    if "mass" in tokens:
        opts["mass"] = True
    if "image" in tokens:
        opts["pbc"] = "orthorhombic"
    return head, opts


def vector(
    traj,
    system,
    command: CommandLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
):
    """Compute vectors for cpptraj-like commands.

    Supports:
    - "@maskA @maskB" (vector from A to B)
    - "center <mask> [mass]"
    - "box", "boxcenter", "ucellx", "ucelly", "ucellz"
    """
    coords, box = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.float32)

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
        if box is not None:
            box = box[select]
        n_frames = coords.shape[0]

    commands = [command] if isinstance(command, str) else list(command)
    results: List[np.ndarray] = []
    atoms = system.atom_table()
    masses = np.asarray(atoms.get("mass", []), dtype=np.float64) if atoms else None

    for cmd in commands:
        _head, opts = _parse_vector_command(cmd)
        mode = opts["mode"]
        if mode == "center":
            idx = _selection_indices(system, opts.get("mask", ""))
            vec = _center(coords, idx, masses, opts.get("mass", False))
        elif mode == "box":
            if box is None:
                raise ValueError("box command requires box lengths")
            vec = box.astype(np.float64)
        elif mode == "boxcenter":
            if box is None:
                raise ValueError("boxcenter command requires box lengths")
            vec = box.astype(np.float64) * 0.5
        elif mode in ("ucellx", "ucelly", "ucellz"):
            if box is None:
                raise ValueError("ucell* command requires box lengths")
            axis = {"ucellx": 0, "ucelly": 1, "ucellz": 2}[mode]
            vec = np.zeros((n_frames, 3), dtype=np.float64)
            vec[:, axis] = box[:, axis]
        else:
            idx_a = _selection_indices(system, opts["mask_a"])
            idx_b = _selection_indices(system, opts["mask_b"])
            com_a = _center(coords, idx_a, masses, opts.get("mass", False))
            com_b = _center(coords, idx_b, masses, opts.get("mass", False))
            vec = com_b - com_a
            if opts.get("pbc") == "orthorhombic":
                vec = _apply_pbc(vec, box)
        results.append(vec.astype(np.float32))

    if len(results) == 1:
        if dtype == "ndarray" and isinstance(command, (list, tuple)):
            return np.stack(results, axis=0)
        return results[0]
    out = np.stack(results, axis=0)
    if dtype == "ndarray":
        return out
    return out


def vector_mask(
    traj,
    system,
    mask: Union[str, Sequence[str], np.ndarray],
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
):
    """Compute vectors from mask pairs (string list or index pairs)."""
    if isinstance(mask, str):
        commands = [mask]
    else:
        arr = np.asarray(mask)
        if arr.dtype.kind == "i":
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("mask array must be shape (n, 2)")
            commands = [f"@{a+1} @{b+1}" for a, b in arr]
        else:
            commands = list(mask)
    return vector(traj, system, commands, frame_indices=frame_indices, dtype=dtype, chunk_frames=chunk_frames)


__all__ = ["vector", "vector_mask"]
