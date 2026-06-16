# Usage:
# from warp_md.analysis.structure import mean_structure, get_average_frame
# avg = mean_structure(traj, system, mask="name CA", frame_indices=[0, 5])

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_inputs,
    native_selection,
    read_all_frames,
    selection_indices,
    subset_frames,
)
from .align import superpose
from .autoimage import autoimage as run_autoimage
from .trajectory import ArrayTrajectory


MaskLike = Union[str, Sequence[int], None]


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _prepare_source(
    traj,
    chunk_frames: Optional[int],
    *,
    frame_indices: Optional[Sequence[int]] = None,
    include_box: bool = False,
    include_time: bool = False,
    length_scale: float = 1.0,
):
    coords, box, time = read_all_frames(
        traj,
        chunk_frames,
        include_box=include_box,
        include_time=include_time,
    )
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords = np.asarray(coords, dtype=np.float32)
    if box is not None:
        box = np.asarray(box, dtype=np.float32)
    if time is not None:
        time = np.asarray(time, dtype=np.float64)
    if length_scale != 1.0:
        scale = np.float32(length_scale)
        coords = coords * scale
        if box is not None:
            box = box * scale
    coords, box, time = subset_frames(coords, frame_indices, box=box, time=time)
    return coords, box, time


def _run_native_structure_plan(
    plan_name: str,
    traj,
    system,
    mask: MaskLike,
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol(plan_name)
    if plan_cls is None:
        raise RuntimeError(f"{plan_name} binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError(f"failed to prepare native inputs for {plan_name}")
    try:
        plan = plan_cls(native_selection(system, native_system, mask))
        return np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(f"native {plan_name} execution failed") from exc


def _run_native_plan_on_live_traj(
    plan_name: str,
    traj,
    system,
    mask: MaskLike,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]],
    **plan_kwargs,
):
    plan_cls = load_native_symbol(plan_name)
    if plan_cls is None:
        raise RuntimeError(f"{plan_name} binding unavailable")
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError(f"failed to prepare native inputs for {plan_name}")
    try:
        plan = plan_cls(native_selection(system, native_system, mask), **plan_kwargs)
        return np.asarray(
            plan.run(
                traj,
                native_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=_frame_indices_arg(frame_indices),
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(f"native {plan_name} execution failed") from exc


def radgyr_tensor(
    traj,
    system,
    mask: MaskLike = "",
    mass: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
    dtype: str = "ndarray",
):
    """Compute radius of gyration tensor (rg + tensor components)."""
    if is_native_traj(traj) and float(length_scale) == 1.0:
        values = _run_native_plan_on_live_traj(
            "RadgyrTensorPlan",
            traj,
            system,
            mask,
            chunk_frames,
            frame_indices,
            mass_weighted=mass,
        )
        if values.ndim != 2 or values.shape[1] != 7:
            raise RuntimeError("native RadgyrTensorPlan returned unexpected output")
        rg_vals = values[:, 0].astype(np.float32, copy=False)
        tensor = values[:, 1:7].astype(np.float32, copy=False)
        if dtype == "dict":
            return {"rg": rg_vals, "tensor": tensor}
        return rg_vals, tensor

    coords, _box, _time = _prepare_source(
        traj,
        chunk_frames,
        frame_indices=frame_indices,
        length_scale=length_scale,
    )
    if coords.size == 0:
        empty_rg = np.empty(0, dtype=np.float32)
        empty_tensor = np.empty((0, 6), dtype=np.float32)
        if dtype == "dict":
            return {"rg": empty_rg, "tensor": empty_tensor}
        return empty_rg, empty_tensor

    plan_cls = load_native_symbol("RadgyrTensorPlan")
    if plan_cls is None:
        raise RuntimeError("RadgyrTensorPlan binding unavailable")
    source = ArrayTrajectory(coords)
    native_traj, native_system = native_inputs(source, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native inputs for RadgyrTensorPlan")
    try:
        plan = plan_cls(native_selection(system, native_system, mask), mass_weighted=mass)
        values = np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native RadgyrTensorPlan execution failed") from exc
    if values.ndim != 2 or values.shape[1] != 7:
        raise RuntimeError("native RadgyrTensorPlan returned unexpected output")
    rg_vals = values[:, 0].astype(np.float32, copy=False)
    tensor = values[:, 1:7].astype(np.float32, copy=False)
    if dtype == "dict":
        return {"rg": rg_vals, "tensor": tensor}
    return rg_vals, tensor


def radgyr(
    traj,
    system,
    mask: MaskLike = "",
    top=None,
    nomax: bool = True,
    mass: bool = True,
    tensor: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
    dtype: str = "ndarray",
):
    """Compute radius of gyration.

    Default behavior returns one Rg value per frame. ``nomax=False`` also
    computes the maximum atom radius from the selected center. ``tensor=True``
    includes the six tensor components returned by ``radgyr_tensor``.
    """
    del top
    if is_native_traj(traj) and float(length_scale) == 1.0:
        values = _run_native_plan_on_live_traj(
            "RadgyrPlan",
            traj,
            system,
            mask,
            chunk_frames,
            frame_indices,
            mass_weighted=mass,
            include_max=not nomax,
            include_tensor=tensor,
        )
        if values.ndim != 2 or values.shape[1] != (1 + int(not nomax) + (6 if tensor else 0)):
            raise RuntimeError("native RadgyrPlan returned unexpected output")
        out = {"rg": values[:, 0].astype(np.float32, copy=False)}
        offset = 1
        if not nomax:
            out["max"] = values[:, offset].astype(np.float32, copy=False)
            offset += 1
        if tensor:
            out["tensor"] = values[:, offset : offset + 6].astype(np.float32, copy=False)

        key = str(dtype).lower()
        if key == "dict":
            return out
        if tensor and not nomax:
            return out["rg"], out["max"], out["tensor"]
        if tensor:
            return out["rg"], out["tensor"]
        if not nomax:
            return np.column_stack([out["rg"], out["max"]]).astype(np.float32)
        return out["rg"]

    coords, _box, _time = _prepare_source(
        traj,
        chunk_frames,
        frame_indices=frame_indices,
        length_scale=length_scale,
    )
    plan_cls = load_native_symbol("RadgyrPlan")
    if plan_cls is None:
        raise RuntimeError("RadgyrPlan binding unavailable")
    source = ArrayTrajectory(coords)
    native_traj, native_system = native_inputs(source, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native inputs for RadgyrPlan")
    try:
        plan = plan_cls(
            native_selection(system, native_system, mask),
            mass_weighted=mass,
            include_max=not nomax,
            include_tensor=tensor,
        )
        values = np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native RadgyrPlan execution failed") from exc
    if values.ndim != 2 or values.shape[1] != (1 + int(not nomax) + (6 if tensor else 0)):
        raise RuntimeError("native RadgyrPlan returned unexpected output")
    out = {"rg": values[:, 0].astype(np.float32, copy=False)}
    offset = 1
    if not nomax:
        out["max"] = values[:, offset].astype(np.float32, copy=False)
        offset += 1
    if tensor:
        out["tensor"] = values[:, offset : offset + 6].astype(np.float32, copy=False)

    key = str(dtype).lower()
    if key == "dict":
        return out
    if tensor and not nomax:
        return out["rg"], out["max"], out["tensor"]
    if tensor:
        return out["rg"], out["tensor"]
    if not nomax:
        return np.column_stack([out["rg"], out["max"]]).astype(np.float32)
    return out["rg"]


def _mean_structure_impl(
    plan_name: str,
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "frame",
    autoimage: bool = False,
    rmsfit: Optional[Union[int, str]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    coords, box, time = _prepare_source(
        traj,
        chunk_frames,
        frame_indices=frame_indices,
        include_box=autoimage,
        include_time=False,
        length_scale=length_scale,
    )
    if coords.size == 0:
        if dtype.lower() in ("traj", "trajectory"):
            return ArrayTrajectory(coords, box=box, time_ps=time)
        return np.empty((0, 3), dtype=np.float32)

    source = ArrayTrajectory(coords, box=box, time_ps=time)
    if autoimage:
        if box is None:
            raise ValueError("autoimage requires box lengths")
        source = run_autoimage(source, system, chunk_frames=chunk_frames)
    if rmsfit is not None:
        source = superpose(
            source,
            system,
            mask=mask if isinstance(mask, str) else "",
            ref=rmsfit,
            chunk_frames=chunk_frames,
        )

    mean = _run_native_structure_plan(plan_name, source, system, mask, chunk_frames).reshape(-1, 3)
    if dtype.lower() in ("traj", "trajectory"):
        return ArrayTrajectory(mean[None, :, :].astype(np.float32))
    return mean.astype(np.float32)


def mean_structure(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "frame",
    autoimage: bool = False,
    rmsfit: Optional[Union[int, str]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Compute average structure for selected atoms."""
    return _mean_structure_impl(
        "MeanStructurePlan",
        traj,
        system,
        mask=mask,
        frame_indices=frame_indices,
        dtype=dtype,
        autoimage=autoimage,
        rmsfit=rmsfit,
        chunk_frames=chunk_frames,
        length_scale=length_scale,
    )


def strip(
    traj,
    system,
    mask: Union[str, Sequence[int]],
    chunk_frames: Optional[int] = None,
):
    """Return trajectory with atoms stripped (keeps inverse selection)."""
    if isinstance(mask, str):
        keep_idx = selection_indices(system, f"!({mask})")
    else:
        mask_idx = set(int(i) for i in mask)
        atoms = system.atom_table()
        n_atoms = len(atoms.get("name", []))
        keep_idx = np.array([i for i in range(n_atoms) if i not in mask_idx], dtype=np.int64)
    if keep_idx.size == 0:
        raise ValueError("strip would remove all atoms")

    coords, box, time = read_all_frames(traj, chunk_frames, include_box=True, include_time=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords = np.asarray(coords, dtype=np.float32)
    if coords.size == 0:
        return ArrayTrajectory(coords, box=box, time_ps=time)
    out = coords[:, keep_idx, :]
    return ArrayTrajectory(out, box=box, time_ps=time)


def get_average_frame(*args, **kwargs):
    return mean_structure(*args, **kwargs)


def make_structure(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "frame",
    autoimage: bool = False,
    rmsfit: Optional[Union[int, str]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Alias for mean_structure."""
    return _mean_structure_impl(
        "MakeStructurePlan",
        traj,
        system,
        mask=mask,
        frame_indices=frame_indices,
        dtype=dtype,
        autoimage=autoimage,
        rmsfit=rmsfit,
        chunk_frames=chunk_frames,
        length_scale=length_scale,
    )


__all__ = [
    "mean_structure",
    "get_average_frame",
    "make_structure",
    "strip",
    "radgyr",
    "radgyr_tensor",
]
