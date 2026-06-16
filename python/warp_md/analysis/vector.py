# Usage:
# from warp_md.analysis.vector import vector, vector_mask
# v = vector(traj, system, "@CA @CB")

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._runtime import (
    load_native_symbol,
    native_inputs,
    native_selection,
    normalize_frame_indices,
    read_all_frames,
    reset_traj,
    selection_indices,
    subset_frames,
)

CommandLike = Union[str, Sequence[str]]


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
    for frame in range(out.shape[0]):
        lx, ly, lz = box[frame]
        if lx > 0:
            out[frame, 0] -= np.round(out[frame, 0] / lx) * lx
        if ly > 0:
            out[frame, 1] -= np.round(out[frame, 1] / ly) * ly
        if lz > 0:
            out[frame, 2] -= np.round(out[frame, 2] / lz) * lz
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


def _native_mask_vector(
    traj,
    system,
    opts: dict,
    frame_indices: Optional[Sequence[int]],
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("VectorPlan")
    if plan_cls is None:
        return None
    native_traj, native_system = native_inputs(
        traj,
        system,
        chunk_frames,
        include_box=opts.get("pbc", "none") == "orthorhombic",
    )
    if native_traj is None or native_system is None or not reset_traj(native_traj):
        return None
    try:
        sel_a = native_selection(system, native_system, opts["mask_a"], allow_at_indices=True)
        sel_b = native_selection(system, native_system, opts["mask_b"], allow_at_indices=True)
        plan = plan_cls(
            sel_a,
            sel_b,
            mass_weighted=opts.get("mass", False),
            pbc=opts.get("pbc", "none"),
        )
        vec = np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
        if frame_indices is not None:
            selected = normalize_frame_indices(frame_indices, vec.shape[0]) or []
            vec = vec[selected]
        return vec
    except Exception:
        return None


def vector(
    traj,
    system,
    command: CommandLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
):
    """Compute vectors for analysis commands."""
    commands = [command] if isinstance(command, str) else list(command)
    coords = None
    box = None
    masses = None
    results: List[np.ndarray] = []

    for cmd in commands:
        _head, opts = _parse_vector_command(cmd)
        mode = opts["mode"]

        if mode == "mask":
            native_vec = _native_mask_vector(traj, system, opts, frame_indices, chunk_frames)
            if native_vec is not None:
                results.append(native_vec.astype(np.float32, copy=False))
                continue

        if coords is None:
            reset_traj(traj)
            coords, box, _time = read_all_frames(traj, chunk_frames, include_box=True)
            if coords is None:
                raise ValueError("trajectory has no frames")
            if coords.size == 0:
                return np.empty((0, 3), dtype=np.float32)
            coords, box, _time = subset_frames(coords, frame_indices, box=box)
            atoms = system.atom_table()
            masses = np.asarray(atoms.get("mass", []), dtype=np.float64) if atoms else None

        n_frames = coords.shape[0]
        if mode == "center":
            idx = selection_indices(system, opts.get("mask", ""), allow_at_indices=True)
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
            idx_a = selection_indices(system, opts["mask_a"], allow_at_indices=True)
            idx_b = selection_indices(system, opts["mask_b"], allow_at_indices=True)
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
    return vector(
        traj,
        system,
        commands,
        frame_indices=frame_indices,
        dtype=dtype,
        chunk_frames=chunk_frames,
    )


__all__ = ["vector", "vector_mask"]
