# Usage:
# from warp_md.analysis.align import align, align_principal_axis
# aligned = align(traj, system, mask="protein", ref=0, mass=True)
# aligned_coords = aligned.read_chunk()["coords"]

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    is_native_traj,
    load_native_symbol,
    native_inputs,
    native_selection,
    native_reference_inputs,
    normalize_frame_indices,
    read_all_frames,
    reset_traj,
)
from .trajectory import ArrayTrajectory

RefLike = Union[int, str]

def _apply_transform(coords: np.ndarray, r: np.ndarray, t: np.ndarray) -> np.ndarray:
    return coords @ r.T + t


def _apply_transform_rows(
    coords: np.ndarray,
    transforms: np.ndarray,
    *,
    frame_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    aligned = coords.copy()
    selected = None
    if frame_indices is not None:
        selected = set(normalize_frame_indices(frame_indices, coords.shape[0]) or [])
    for frame in range(coords.shape[0]):
        if selected is not None and frame not in selected:
            continue
        row = transforms[frame]
        aligned[frame] = _apply_transform(
            coords[frame],
            row[:9].reshape(3, 3),
            row[9:12],
        )
    return aligned


def _trajectory_payload_to_array(payload):
    coords = np.asarray(payload["coords"], dtype=np.float32)
    box = payload.get("box")
    time = payload.get("time_ps")
    return ArrayTrajectory(
        coords,
        box=None if box is None else np.asarray(box, dtype=np.float32),
        time_ps=None if time is None else np.asarray(time, dtype=np.float64),
    )


def _run_native_superpose_trajectory(
    traj,
    system,
    mask: str,
    ref: RefLike,
    ref_mask: str,
    mass: bool,
    chunk_frames: Optional[int],
):
    if not is_native_traj(traj):
        return None
    plan_cls = load_native_symbol("SuperposeTrajectoryPlan")
    if plan_cls is None:
        return None
    native_traj, reference_system = native_reference_inputs(
        traj,
        system,
        mask,
        ref,
        ref_mask,
        chunk_frames,
        include_box=True,
        include_time=True,
    )
    if native_traj is None or reference_system is None or not reset_traj(native_traj):
        return None
    try:
        selection = native_selection(system, reference_system, mask)
        plan = plan_cls(selection, reference="topology", mass=mass, norotate=False)
        payload = plan.run(
            native_traj,
            reference_system,
            chunk_frames=chunk_frames,
            device="auto",
        )
    except Exception as exc:
        raise RuntimeError("native SuperposeTrajectoryPlan execution failed") from exc
    reset_traj(native_traj)
    return _trajectory_payload_to_array(payload)


def _run_native_align_transforms(
    traj,
    system,
    mask: str,
    ref: RefLike,
    ref_mask: str,
    mass: bool,
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("AlignPlan")
    if plan_cls is None:
        raise RuntimeError("AlignPlan binding unavailable")
    native_traj, reference_system = native_reference_inputs(
        traj,
        system,
        mask,
        ref,
        ref_mask,
        chunk_frames,
    )
    if native_traj is None or reference_system is None or not reset_traj(native_traj):
        raise RuntimeError("failed to prepare native align inputs")
    try:
        selection = native_selection(system, reference_system, mask)
        plan = plan_cls(selection, reference="topology", mass=mass, norotate=False)
        values = np.asarray(
            plan.run(
                native_traj,
                reference_system,
                chunk_frames=chunk_frames,
                device="auto",
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native AlignPlan execution failed") from exc
    reset_traj(native_traj)
    return native_traj, values


def _run_native_principal_axis_transforms(
    traj,
    system,
    mask: str,
    mass: bool,
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("AlignPrincipalAxisPlan")
    if plan_cls is None:
        raise RuntimeError("AlignPrincipalAxisPlan binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native principal-axis inputs")
    try:
        selection = native_selection(system, native_system, mask)
        plan = plan_cls(selection, mass=mass)
        values = np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native AlignPrincipalAxisPlan execution failed") from exc
    if values.ndim != 2 or values.shape[1] != 12:
        raise RuntimeError("native AlignPrincipalAxisPlan returned unexpected transforms")
    return values


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
    ref_mask = mask if ref_mask is None else ref_mask
    if not return_transforms:
        native = _run_native_superpose_trajectory(
            traj,
            system,
            mask,
            ref,
            ref_mask,
            mass,
            chunk_frames,
        )
        if native is not None:
            return native
    coords, box, time = read_all_frames(traj, chunk_frames, include_box=True, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        out = ArrayTrajectory(coords, box=box, time_ps=time)
        if return_transforms:
            return out, np.empty((0, 12), dtype=np.float32)
        return out
    source = ArrayTrajectory(coords, box=box, time_ps=time)
    _native_traj, transforms = _run_native_align_transforms(
        source,
        system,
        mask,
        ref,
        ref_mask,
        mass,
        chunk_frames,
    )
    if transforms.shape[0] != coords.shape[0]:
        raise RuntimeError("native AlignPlan returned unexpected frame count")
    out = ArrayTrajectory(
        _apply_transform_rows(coords, transforms).astype(np.float32),
        box=box,
        time_ps=time,
    )
    if return_transforms:
        return out, transforms
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
    ref_mask = mask if (ref_mask is None or ref_mask == "") else ref_mask
    if frame_indices is None:
        native = _run_native_superpose_trajectory(
            traj,
            system,
            mask,
            ref,
            ref_mask,
            mass,
            chunk_frames,
        )
        if native is not None:
            return native
    coords, box, time = read_all_frames(traj, chunk_frames, include_box=True, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)
    source = ArrayTrajectory(coords, box=box, time_ps=time)
    _native_traj, transforms = _run_native_align_transforms(
        source,
        system,
        mask,
        ref,
        ref_mask,
        mass,
        chunk_frames,
    )
    if transforms.shape[0] != coords.shape[0]:
        raise RuntimeError("native AlignPlan returned unexpected frame count")
    return ArrayTrajectory(
        _apply_transform_rows(coords, transforms, frame_indices=frame_indices).astype(np.float32),
        box=box,
        time_ps=time,
    )


def align_principal_axis(
    traj,
    system,
    mask: str = "protein",
    mass: bool = False,
    chunk_frames: Optional[int] = None,
    return_transforms: bool = False,
):
    """Align trajectory frames to principal axes."""
    coords, box, time = read_all_frames(traj, chunk_frames, include_box=True, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        out = ArrayTrajectory(coords, box=box, time_ps=time)
        if return_transforms:
            return out, np.empty((0, 12), dtype=np.float32)
        return out

    source = ArrayTrajectory(np.asarray(coords, dtype=np.float32), box=box, time_ps=time)
    transforms = _run_native_principal_axis_transforms(
        source,
        system,
        mask,
        mass,
        chunk_frames,
    )
    if transforms.shape[0] != coords.shape[0]:
        raise RuntimeError("native AlignPrincipalAxisPlan returned unexpected frame count")
    aligned = _apply_transform_rows(coords, transforms)
    out = ArrayTrajectory(aligned.astype(np.float32), box=box, time_ps=time)
    if return_transforms:
        return out, transforms
    return out


__all__ = ["align", "superpose", "align_principal_axis"]
