# Usage:
# from warp_md.analysis.rotation import rotation_matrix
# mats = rotation_matrix(traj, system, mask="name CA", ref=0)

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    load_native_symbol,
    native_selection,
    native_reference_inputs,
    normalize_frame_indices,
    read_all_frames,
)
from .trajectory import ArrayTrajectory

RefLike = Union[int, str]


def _run_native_rotation_matrix(
    traj,
    system,
    mask: str,
    ref: RefLike,
    mass: bool,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    plan_cls = load_native_symbol("RotationMatrixPlan")
    if plan_cls is None:
        raise RuntimeError("RotationMatrixPlan binding unavailable")
    native_traj, reference_system = native_reference_inputs(
        traj,
        system,
        mask,
        ref,
        mask,
        chunk_frames,
    )
    if native_traj is None or reference_system is None:
        raise RuntimeError("failed to prepare native rotation inputs")
    try:
        selection = native_selection(system, reference_system, mask)
        plan = plan_cls(selection, reference="topology", mass=mass)
        return np.asarray(
            plan.run(
                native_traj,
                reference_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=None if frame_indices is None else list(frame_indices),
            ),
            dtype=np.float32,
        ).reshape(-1, 3, 3)
    except Exception as exc:
        raise RuntimeError("native RotationMatrixPlan execution failed") from exc


def _run_native_rmsd(
    traj,
    system,
    mask: str,
    ref: RefLike,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    plan_cls = load_native_symbol("RmsdPlan")
    if plan_cls is None:
        raise RuntimeError("RmsdPlan binding unavailable")
    native_traj, reference_system = native_reference_inputs(
        traj,
        system,
        mask,
        ref,
        mask,
        chunk_frames,
    )
    if native_traj is None or reference_system is None:
        raise RuntimeError("failed to prepare native rmsd inputs")
    try:
        selection = native_selection(system, reference_system, mask)
        plan = plan_cls(selection, reference="topology", align=True)
        return np.asarray(
            plan.run(
                native_traj,
                reference_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=None if frame_indices is None else list(frame_indices),
            ),
            dtype=np.float32,
        ).reshape(-1)
    except Exception as exc:
        raise RuntimeError("native RmsdPlan execution failed") from exc


def rotation_matrix(
    traj,
    system,
    mask: str = "",
    ref: RefLike = 0,
    mass: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    with_rmsd: bool = False,
):
    """Compute rotation matrices to align frames to reference."""
    coords, _box, _time = read_all_frames(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    n_frames = coords.shape[0]
    if n_frames == 0:
        mats = np.empty((0, 3, 3), dtype=np.float32)
        if with_rmsd:
            return mats, np.empty((0,), dtype=np.float32)
        return mats

    selected = normalize_frame_indices(frame_indices, n_frames)
    source_coords = np.asarray(coords, dtype=np.float32)

    def make_source():
        return ArrayTrajectory(source_coords)

    mats = np.broadcast_to(np.eye(3, dtype=np.float32), (n_frames, 3, 3)).copy()
    rmsd_vals = np.zeros(n_frames, dtype=np.float32) if with_rmsd else None

    if selected is None:
        native_mats = _run_native_rotation_matrix(
            make_source(),
            system,
            mask,
            ref,
            mass,
            chunk_frames,
        )
        if native_mats.shape[0] != n_frames:
            raise RuntimeError("native RotationMatrixPlan returned unexpected frame count")
        mats[:] = native_mats
        if with_rmsd:
            native_rmsd = _run_native_rmsd(make_source(), system, mask, ref, chunk_frames)
            if native_rmsd.shape[0] != n_frames:
                raise RuntimeError("native RmsdPlan returned unexpected frame count")
            rmsd_vals[:] = native_rmsd
    elif selected:
        native_mats = _run_native_rotation_matrix(
            make_source(),
            system,
            mask,
            ref,
            mass,
            chunk_frames,
            frame_indices=selected,
        )
        if native_mats.shape[0] != len(selected):
            raise RuntimeError("native RotationMatrixPlan returned unexpected frame subset")
        mats[np.asarray(selected, dtype=np.int64)] = native_mats
        if with_rmsd:
            native_rmsd = _run_native_rmsd(
                make_source(),
                system,
                mask,
                ref,
                chunk_frames,
                frame_indices=selected,
            )
            if native_rmsd.shape[0] != len(selected):
                raise RuntimeError("native RmsdPlan returned unexpected frame subset")
            rmsd_vals[np.asarray(selected, dtype=np.int64)] = native_rmsd

    if with_rmsd:
        return mats, rmsd_vals
    return mats


__all__ = ["rotation_matrix"]
