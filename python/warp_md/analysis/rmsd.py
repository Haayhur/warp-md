# Usage:
# from warp_md.analysis.rmsd import distance_rmsd
# vals = distance_rmsd(traj, system, mask="name CA", ref=0, pbc="none")

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

import warp_md
from ._runtime import (
    clone_native_system_with_positions0,
    load_native_symbol,
    native_inputs,
    native_selection,
    prepend_reference_frame,
    read_all_frames,
    read_frame_subset,
    reset_traj,
    selection_indices,
)
from .trajectory import ArrayTrajectory


def _selection_indices(system, mask: str = "") -> np.ndarray:
    return selection_indices(system, mask)


def _read_all(traj, chunk_frames: Optional[int] = None):
    coords, box, _time = read_all_frames(traj, chunk_frames, include_box=True)
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


def _pair_distances(coords: np.ndarray, pbc: str, box: Optional[np.ndarray]) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    pbc = str(pbc).lower()
    fn = getattr(warp_md, "pair_distances_array", None)
    if fn is not None and not (
        getattr(fn, "__name__", "") == "pair_distances_array"
        and getattr(warp_md, "traj_py", None) is None
    ):
        try:
            box_arg = None if box is None else np.asarray(box, dtype=np.float64)
            return np.asarray(fn(coords, pbc, box_arg), dtype=np.float64)
        except RuntimeError:
            pass
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


def _native_distance_rmsd(
    traj,
    system,
    mask: str,
    ref,
    pbc: str,
    length_scale: float,
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("DistanceRmsdPlan")
    if plan_cls is None:
        raise RuntimeError("DistanceRmsdPlan binding unavailable")
    native_traj, native_system = native_inputs(
        traj,
        system,
        chunk_frames,
        include_box=(pbc == "orthorhombic"),
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native distance RMSD inputs")
    try:
        selection = native_selection(system, native_system, mask)
        ref_index = int(ref)
        run_traj = native_traj
        run_system = native_system
        trim_first = False
        if ref_index == 0:
            if not reset_traj(run_traj):
                raise RuntimeError("failed to reset native trajectory")
        elif pbc == "none":
            ref_coords, _ref_box, _ref_time, source_indices = read_frame_subset(
                native_traj,
                [ref_index],
                chunk_frames,
            )
            if ref_coords is None or ref_coords.shape[0] != 1 or source_indices.size != 1:
                raise ValueError("ref index out of range")
            run_system = clone_native_system_with_positions0(native_system, ref_coords[0])
            if run_system is None:
                raise RuntimeError("failed to build native topology reference system")
            if not reset_traj(run_traj):
                raise RuntimeError("failed to reset native trajectory")
        else:
            run_traj = prepend_reference_frame(
                native_traj,
                ref_index,
                chunk_frames,
                include_box=(pbc == "orthorhombic"),
            )
            if run_traj is None:
                raise RuntimeError("failed to build native reference trajectory")
            trim_first = True
        reference = "topology" if ref_index != 0 and pbc == "none" else "frame0"
        plan = plan_cls(selection, reference=reference, pbc=pbc)
        values = np.asarray(
            plan.run(run_traj, run_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
        if trim_first:
            values = values[1:]
        return values * np.float32(length_scale)
    except Exception as exc:
        raise RuntimeError("native DistanceRmsdPlan execution failed") from exc


def _native_pairwise_rmsd(
    traj,
    system,
    mask: str,
    metric: str,
    pbc: str,
    length_scale: float,
    frame_indices: Optional[Sequence[int]],
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("PairwiseRmsdPlan")
    if plan_cls is None:
        raise RuntimeError("PairwiseRmsdPlan binding unavailable")
    native_traj, native_system = native_inputs(
        traj,
        system,
        chunk_frames,
        include_box=(pbc == "orthorhombic"),
    )
    if native_traj is None or native_system is None or not reset_traj(native_traj):
        raise RuntimeError("failed to prepare native pairwise RMSD inputs")
    try:
        selection = native_selection(system, native_system, mask)
        plan = plan_cls(selection, metric=metric, pbc=pbc)
        kwargs = {
            "chunk_frames": chunk_frames,
            "device": "auto",
        }
        if frame_indices is not None:
            kwargs["frame_indices"] = [int(value) for value in frame_indices]
        values = np.asarray(
            plan.run(
                native_traj,
                native_system,
                **kwargs,
            ),
            dtype=np.float32,
        )
        return values * np.float32(length_scale)
    except Exception as exc:
        raise RuntimeError("native PairwiseRmsdPlan execution failed") from exc


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

    if int(ref) == 0 or pbc == "none":
        return _native_distance_rmsd(
            traj,
            system,
            mask,
            ref,
            pbc,
            length_scale,
            chunk_frames,
        )

    coords, box, _time = read_all_frames(traj, chunk_frames, include_box=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty(0, dtype=np.float32)
    source = ArrayTrajectory(coords, box=box)
    return _native_distance_rmsd(
        source,
        system,
        mask,
        ref,
        pbc,
        length_scale,
        chunk_frames,
    )


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

    native_values = _native_pairwise_rmsd(
        traj,
        system,
        mask,
        metric,
        pbc,
        length_scale,
        frame_indices,
        chunk_frames,
    )
    if mat_type == "full":
        return native_values
    if native_values.size == 0:
        return np.empty((0,), dtype=np.float32)
    tri = np.triu_indices(native_values.shape[0], k=1)
    return np.asarray(native_values[tri], dtype=np.float32)


__all__ = ["distance_rmsd", "pairwise_rmsd"]
