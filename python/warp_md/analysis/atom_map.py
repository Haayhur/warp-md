# Usage:
# from warp_md.analysis.atom_map import atom_map
# mask, rmsd = atom_map(traj, system, ref=0, rmsfit=True)

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields
from ._runtime import (
    clone_native_system_with_positions0,
    is_native_traj,
    kabsch_rmsd,
    load_native_symbol,
    native_inputs,
    native_selection,
    native_system_from_atom_count,
    read_all_frames,
    read_frame_subset,
    reset_traj,
    selection_indices,
)

RefLike = Union[int, np.ndarray, object]


def atom_map(
    traj,
    system,
    ref: RefLike,
    rmsfit: bool = False,
    mask: str = "",
    chunk_frames: Optional[int] = None,
):
    """Nearest-neighbor atom mapping between traj and reference.

    Returns (mask_string, rmsd_array if rmsfit else empty array).
    """
    native_out = _native_atom_map(
        traj,
        system,
        ref,
        rmsfit,
        mask,
        chunk_frames,
    )
    if native_out is not None:
        return native_out

    coords, _box, _time = read_all_frames(traj, chunk_frames, include_box=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return "", np.empty((0,), dtype=np.float32)

    ref_coords = _resolve_ref(ref, coords)
    idx = selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    target = coords[0][idx]
    ref_sel = ref_coords[idx]
    if rmsfit:
        # align target to reference before mapping
        target = _align_coords(target, ref_sel)

    mapping = _native_nearest_mapping(target, ref_sel, chunk_frames)
    if mapping is None:
        mapping = _nearest_mapping(target, ref_sel)
    mask_str = "@" + ",".join(str(int(i) + 1) for i in mapping)

    if not rmsfit:
        return mask_str, np.empty((0,), dtype=np.float32)

    # compute RMSD for each frame with mapping applied
    out = np.zeros(coords.shape[0], dtype=np.float64)
    ref_mapped = ref_sel[mapping]
    for f in range(coords.shape[0]):
        cur = coords[f][idx][mapping]
        out[f] = kabsch_rmsd(cur, ref_mapped)
    return mask_str, out.astype(np.float32)


def _native_atom_map(
    traj,
    system,
    ref: RefLike,
    rmsfit: bool,
    mask: str,
    chunk_frames: Optional[int],
):
    if not is_native_traj(traj):
        return None
    idx = selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    first_coords, _box, _time, first_indices = read_frame_subset(
        traj,
        [0],
        chunk_frames,
        include_box=False,
    )
    if first_coords is None or first_coords.shape[0] != 1 or first_indices.size != 1:
        return "", np.empty((0,), dtype=np.float32)
    ref_coords = _resolve_ref_native(ref, traj, system, chunk_frames)

    target = np.asarray(first_coords[0], dtype=np.float64)[idx]
    ref_sel = np.asarray(ref_coords, dtype=np.float64)[idx]
    if rmsfit:
        target = _align_coords(target, ref_sel)

    mapping = _native_nearest_mapping(target, ref_sel, chunk_frames)
    if mapping is None:
        mapping = _nearest_mapping(target, ref_sel)
    mask_str = "@" + ",".join(str(int(i) + 1) for i in mapping)
    if not rmsfit:
        return mask_str, np.empty((0,), dtype=np.float32)

    rmsd = _native_atom_map_rmsd(
        traj,
        system,
        idx,
        mapping,
        np.asarray(ref_coords, dtype=np.float32),
        chunk_frames,
    )
    if rmsd is None:
        return None
    return mask_str, rmsd


def _resolve_ref_native(ref: RefLike, traj, system, chunk_frames: Optional[int]) -> np.ndarray:
    if isinstance(ref, int):
        coords, _box, _time, source_indices = read_frame_subset(
            traj,
            [int(ref)],
            chunk_frames,
            include_box=False,
        )
        if coords is None or coords.shape[0] != 1 or source_indices.size != 1:
            raise ValueError("ref index out of range")
        return np.asarray(coords[0], dtype=np.float32)
    if isinstance(ref, np.ndarray):
        if ref.ndim != 2 or ref.shape[1] != 3:
            raise ValueError("ref coords must be (n_atoms, 3)")
        return ref.astype(np.float32)
    if isinstance(ref, str):
        key = ref.strip().lower()
        if key in ("topology", "top", "topo"):
            if hasattr(system, "positions0"):
                positions0 = system.positions0()
                if positions0 is not None:
                    return np.asarray(positions0, dtype=np.float32)
            return _resolve_ref_native(0, traj, system, chunk_frames)
        if key in ("frame0", "first", "0"):
            return _resolve_ref_native(0, traj, system, chunk_frames)
    if hasattr(ref, "read_chunk"):
        return _resolve_ref(ref, np.empty((0, 0, 3), dtype=np.float32))
    raise ValueError("unsupported ref type")


def _native_atom_map_rmsd(
    traj,
    system,
    idx: np.ndarray,
    mapping: np.ndarray,
    ref_coords: np.ndarray,
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("RmsdPlan")
    if plan_cls is None:
        return None
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        return None

    mapped_idx = np.asarray(idx[mapping], dtype=np.int64)
    n_atoms = int(native_system.n_atoms())
    positions0 = np.zeros((n_atoms, 3), dtype=np.float32)
    if hasattr(native_system, "positions0"):
        existing = native_system.positions0()
        if existing is not None:
            positions0[:] = np.asarray(existing, dtype=np.float32)
    positions0[mapped_idx] = np.asarray(ref_coords, dtype=np.float32)[mapped_idx]
    reference_system = clone_native_system_with_positions0(native_system, positions0)
    if reference_system is None:
        return None

    if not reset_traj(native_traj):
        raise RuntimeError("failed to reset native trajectory")
    selection = native_selection(system, reference_system, mapped_idx)
    plan = plan_cls(selection, reference="topology", align=True)
    try:
        return np.asarray(
            plan.run(
                native_traj,
                reference_system,
                chunk_frames=chunk_frames,
                device="auto",
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native RmsdPlan execution failed for atom_map") from exc


def _native_nearest_mapping(
    target: np.ndarray,
    ref: np.ndarray,
    chunk_frames: Optional[int],
) -> Optional[np.ndarray]:
    plan_cls = load_native_symbol("AtomMapPlan")
    traj_cls = load_native_symbol("Trajectory")
    if plan_cls is None or traj_cls is None:
        return None
    target_arr = np.asarray(target, dtype=np.float32)
    ref_arr = np.asarray(ref, dtype=np.float32)
    if (
        target_arr.ndim != 2
        or ref_arr.ndim != 2
        or target_arr.shape[1] != 3
        or ref_arr.shape[1] != 3
    ):
        return None
    if target_arr.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)

    n_target = int(target_arr.shape[0])
    n_ref = int(ref_arr.shape[0])
    combined = np.concatenate([target_arr, ref_arr], axis=0).reshape(
        1, n_target + n_ref, 3
    )
    native_system = native_system_from_atom_count(n_target + n_ref, positions0=combined[0])
    if native_system is None or not hasattr(native_system, "select_indices"):
        return None
    try:
        native_traj = traj_cls.from_numpy(combined.astype(np.float32, copy=False))
        sel_target = native_system.select_indices(list(range(n_target)))
        sel_ref = native_system.select_indices(list(range(n_target, n_target + n_ref)))
        plan = plan_cls(sel_target, sel_ref, pbc="none")
        values = np.asarray(
            plan.run(
                native_traj,
                native_system,
                chunk_frames=max(1, int(chunk_frames or 1)),
                device="cpu",
            ),
            dtype=np.float32,
        )
    except Exception:
        return None
    if values.shape != (1, n_target):
        return None
    mapping = values[0].astype(np.int64, copy=False) - n_target
    if np.any(mapping < 0) or np.any(mapping >= n_ref):
        return None
    return mapping


def _resolve_ref(ref: RefLike, coords: np.ndarray) -> np.ndarray:
    if isinstance(ref, int):
        idx = ref
        if idx < 0:
            idx = coords.shape[0] + idx
        if idx < 0 or idx >= coords.shape[0]:
            raise ValueError("ref index out of range")
        return coords[idx]
    if isinstance(ref, np.ndarray):
        if ref.ndim != 2 or ref.shape[1] != 3:
            raise ValueError("ref coords must be (n_atoms, 3)")
        return ref.astype(np.float64)
    if hasattr(ref, "read_chunk"):
        chunk = read_chunk_fields(ref, 1)
        if chunk is None:
            raise ValueError("reference has no frames")
        coords_ref = np.asarray(chunk["coords"], dtype=np.float64)
        if coords_ref.shape[0] == 0:
            raise ValueError("reference has no frames")
        return coords_ref[0]
    raise ValueError("unsupported ref type")


def _align_coords(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # Kabsch align target to ref (no weights)
    if target.shape != ref.shape:
        raise ValueError("target/ref shape mismatch")
    cx = target.mean(axis=0)
    cy = ref.mean(axis=0)
    x0 = target - cx
    y0 = ref - cy
    h = x0.T @ y0
    u, _s, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    return x0 @ r.T + cy


def _nearest_mapping(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # map each target atom to nearest ref atom
    mapping = np.zeros(target.shape[0], dtype=np.int64)
    for i in range(target.shape[0]):
        diff = ref - target[i]
        dist2 = np.sum(diff * diff, axis=1)
        mapping[i] = int(np.argmin(dist2))
    return mapping


__all__ = ["atom_map"]
