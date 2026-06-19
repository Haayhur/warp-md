# Usage:
# from warp_md.analysis.transform import center, translate, rotate, scale, transform
# out = center(traj, system, mask="protein", mode="origin")

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    is_native_traj,
    load_native_symbol,
    native_inputs,
    native_selection,
    native_system_from_atom_count,
    read_all_frames,
)
from .trajectory import ArrayTrajectory


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


def _prepare_source(
    traj,
    chunk_frames: Optional[int],
):
    coords, box, time = read_all_frames(traj, chunk_frames, include_box=True, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    return (
        np.asarray(coords, dtype=np.float32),
        None if box is None else np.asarray(box, dtype=np.float32),
        None if time is None else np.asarray(time, dtype=np.float64),
    )


def _run_native_coord_plan(
    traj,
    system,
    plan,
    chunk_frames: Optional[int],
    *,
    include_box: bool = False,
) -> np.ndarray:
    native_traj, native_system = native_inputs(
        traj,
        system,
        chunk_frames,
        include_box=include_box,
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native transform inputs")
    try:
        return np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native transform plan execution failed") from exc


def _native_atom_count(traj) -> Optional[int]:
    n_atoms = getattr(traj, "n_atoms", None)
    if n_atoms is None:
        return None
    try:
        return int(n_atoms() if callable(n_atoms) else n_atoms)
    except Exception:
        return None


def _trajectory_payload_to_array(payload):
    coords = np.asarray(payload["coords"], dtype=np.float32)
    box = payload.get("box")
    time = payload.get("time_ps")
    return ArrayTrajectory(
        coords,
        box=None if box is None else np.asarray(box, dtype=np.float32),
        time_ps=None if time is None else np.asarray(time, dtype=np.float64),
    )


def _run_native_trajectory_transform(
    traj,
    rotation: np.ndarray,
    translation: np.ndarray,
    chunk_frames: Optional[int],
):
    if not is_native_traj(traj):
        return None
    plan_cls = load_native_symbol("TransformTrajectoryPlan")
    if plan_cls is None:
        return None
    n_atoms = _native_atom_count(traj)
    if n_atoms is None:
        return None
    system = native_system_from_atom_count(n_atoms)
    if system is None:
        return None
    try:
        plan = plan_cls(tuple(rotation.reshape(-1)), tuple(translation))
        payload = plan.run(traj, system, chunk_frames=chunk_frames, device="auto")
    except Exception as exc:
        raise RuntimeError("native trajectory transform plan execution failed") from exc
    return _trajectory_payload_to_array(payload)


def _run_native_center_trajectory(
    traj,
    system,
    mask: str,
    mode: str,
    point_vec,
    mass: bool,
    chunk_frames: Optional[int],
):
    if not is_native_traj(traj):
        return None
    plan_cls = load_native_symbol("CenterTrajectoryOutputPlan")
    if plan_cls is None:
        return None
    native_traj, native_system = native_inputs(
        traj,
        system,
        chunk_frames,
        include_box=True,
        include_time=True,
    )
    if native_traj is None or native_system is None:
        return None
    try:
        plan = plan_cls(
            native_selection(system, native_system, mask),
            mode=mode,
            point=point_vec,
            mass_weighted=mass,
        )
        payload = plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto")
    except Exception as exc:
        raise RuntimeError("native center trajectory plan execution failed") from exc
    return _trajectory_payload_to_array(payload)


def translate(traj, delta: Sequence[float], chunk_frames: Optional[int] = None):
    delta_vec = _as_vec3(delta, "delta")
    native = _run_native_trajectory_transform(
        traj,
        np.eye(3, dtype=np.float64),
        delta_vec,
        chunk_frames,
    )
    if native is not None:
        return native
    coords, box, time = _prepare_source(traj, chunk_frames)
    if coords.size == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)
    plan_cls = load_native_symbol("TranslatePlan")
    if plan_cls is None:
        raise RuntimeError("TranslatePlan binding unavailable")
    source = ArrayTrajectory(coords, box=box, time_ps=time)
    system = native_system_from_atom_count(coords.shape[1], positions0=coords[0])
    if system is None:
        raise RuntimeError("failed to synthesize native transform system")
    values = _run_native_coord_plan(source, system, plan_cls(tuple(delta_vec)), chunk_frames)
    return ArrayTrajectory(values.reshape(coords.shape), box=box, time_ps=time)


def scale(traj, factor: Union[float, Sequence[float]], chunk_frames: Optional[int] = None):
    if isinstance(factor, (float, int)):
        mat = np.eye(3, dtype=np.float64) * float(factor)
        native = _run_native_trajectory_transform(
            traj,
            mat,
            np.zeros(3, dtype=np.float64),
            chunk_frames,
        )
        if native is not None:
            return native
        coords, box, time = _prepare_source(traj, chunk_frames)
        if coords.size == 0:
            return ArrayTrajectory(coords, box=box, time_ps=time)
        plan_cls = load_native_symbol("ScalePlan")
        if plan_cls is None:
            raise RuntimeError("ScalePlan binding unavailable")
        plan = plan_cls(float(factor))
    else:
        vec = _as_vec3(factor, "factor")
        mat = np.diag(vec)
        native = _run_native_trajectory_transform(
            traj,
            mat,
            np.zeros(3, dtype=np.float64),
            chunk_frames,
        )
        if native is not None:
            return native
        coords, box, time = _prepare_source(traj, chunk_frames)
        if coords.size == 0:
            return ArrayTrajectory(coords, box=box, time_ps=time)
        plan_cls = load_native_symbol("TransformPlan")
        if plan_cls is None:
            raise RuntimeError("TransformPlan binding unavailable")
        plan = plan_cls(tuple(mat.reshape(-1)), (0.0, 0.0, 0.0))
    source = ArrayTrajectory(coords, box=box, time_ps=time)
    system = native_system_from_atom_count(coords.shape[1], positions0=coords[0])
    if system is None:
        raise RuntimeError("failed to synthesize native transform system")
    values = _run_native_coord_plan(source, system, plan, chunk_frames)
    return ArrayTrajectory(values.reshape(coords.shape), box=box, time_ps=time)


def rotate(traj, rotation: Sequence[float], chunk_frames: Optional[int] = None):
    mat = _as_mat3(rotation, "rotation")
    native = _run_native_trajectory_transform(
        traj,
        mat,
        np.zeros(3, dtype=np.float64),
        chunk_frames,
    )
    if native is not None:
        return native
    coords, box, time = _prepare_source(traj, chunk_frames)
    if coords.size == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)
    plan_cls = load_native_symbol("RotatePlan")
    if plan_cls is None:
        raise RuntimeError("RotatePlan binding unavailable")
    source = ArrayTrajectory(coords, box=box, time_ps=time)
    system = native_system_from_atom_count(coords.shape[1], positions0=coords[0])
    if system is None:
        raise RuntimeError("failed to synthesize native transform system")
    values = _run_native_coord_plan(
        source,
        system,
        plan_cls(tuple(mat.reshape(-1))),
        chunk_frames,
    )
    return ArrayTrajectory(values.reshape(coords.shape), box=box, time_ps=time)


def transform(
    traj,
    rotation: Optional[Sequence[float]] = None,
    translation: Optional[Sequence[float]] = None,
    chunk_frames: Optional[int] = None,
):
    mat = np.eye(3, dtype=np.float64) if rotation is None else _as_mat3(rotation, "rotation")
    vec = np.zeros(3, dtype=np.float64) if translation is None else _as_vec3(translation, "translation")
    native = _run_native_trajectory_transform(traj, mat, vec, chunk_frames)
    if native is not None:
        return native
    coords, box, time = _prepare_source(traj, chunk_frames)
    if coords.size == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)
    if rotation is None and translation is None:
        return ArrayTrajectory(coords, box=box, time_ps=time)
    plan_cls = load_native_symbol("TransformPlan")
    if plan_cls is None:
        raise RuntimeError("TransformPlan binding unavailable")
    source = ArrayTrajectory(coords, box=box, time_ps=time)
    system = native_system_from_atom_count(coords.shape[1], positions0=coords[0])
    if system is None:
        raise RuntimeError("failed to synthesize native transform system")
    values = _run_native_coord_plan(
        source,
        system,
        plan_cls(tuple(mat.reshape(-1)), tuple(vec)),
        chunk_frames,
    )
    return ArrayTrajectory(values.reshape(coords.shape), box=box, time_ps=time)


def center(
    traj,
    system,
    mask: str = "",
    mode: str = "origin",
    point: Optional[Sequence[float]] = None,
    mass: bool = False,
    chunk_frames: Optional[int] = None,
):
    normalized_mode = str(mode).lower()
    point_vec = None
    if normalized_mode == "point":
        if point is None:
            raise ValueError("point is required when mode='point'")
        point_vec = tuple(_as_vec3(point, "point"))
    elif normalized_mode == "box":
        pass
    elif normalized_mode != "origin":
        raise ValueError("mode must be 'origin', 'point', or 'box'")

    native = _run_native_center_trajectory(
        traj,
        system,
        mask,
        normalized_mode,
        point_vec,
        mass,
        chunk_frames,
    )
    if native is not None:
        return native

    coords, box, time = _prepare_source(traj, chunk_frames)
    if coords.size == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)
    if normalized_mode == "box" and box is None:
        raise ValueError("box lengths required when mode='box'")

    plan_cls = load_native_symbol("CenterTrajectoryPlan")
    if plan_cls is None:
        raise RuntimeError("CenterTrajectoryPlan binding unavailable")
    source = ArrayTrajectory(coords, box=box, time_ps=time)
    native_traj, native_system = native_inputs(
        source,
        system,
        chunk_frames,
        include_box=True,
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native center inputs")
    try:
        plan = plan_cls(
            native_selection(system, native_system, mask),
            mode=normalized_mode,
            point=point_vec,
            mass_weighted=mass,
        )
        values = np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native center plan execution failed") from exc
    return ArrayTrajectory(values.reshape(coords.shape), box=box, time_ps=time)


__all__ = ["center", "translate", "rotate", "scale", "transform"]
