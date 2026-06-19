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
        plan = plan_cls(native_selection(system, native_system, mask, allow_at_indices=True))
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
    device: str = "auto",
    **plan_kwargs,
):
    plan_cls = load_native_symbol(plan_name)
    if plan_cls is None:
        raise RuntimeError(f"{plan_name} binding unavailable")
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError(f"failed to prepare native inputs for {plan_name}")
    try:
        plan = plan_cls(
            native_selection(system, native_system, mask, allow_at_indices=True),
            **plan_kwargs,
        )
        return np.asarray(
            plan.run(
                traj,
                native_system,
                chunk_frames=chunk_frames,
                device=device,
                frame_indices=_frame_indices_arg(frame_indices),
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(f"native {plan_name} execution failed") from exc


def _require_native_radgyr_inputs(traj, name: str, length_scale: float):
    if not is_native_traj(traj):
        raise RuntimeError(
            f"{name} requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    if float(length_scale) != 1.0:
        raise ValueError(
            f"{name} length_scale must be 1.0 on the Rust-backed execution path."
        )


def radgyr_tensor(
    traj,
    system,
    mask: MaskLike = "",
    mass: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
    dtype: str = "ndarray",
    device: str = "auto",
):
    """Compute radius of gyration tensor (rg + tensor components)."""
    _require_native_radgyr_inputs(traj, "radgyr_tensor", length_scale)
    values = _run_native_plan_on_live_traj(
        "RadgyrTensorPlan",
        traj,
        system,
        mask,
        chunk_frames,
        frame_indices,
        device=device,
        mass_weighted=mass,
    )
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
    axes: bool = False,
    tensor: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
    dtype: str = "ndarray",
    device: str = "auto",
):
    """Compute radius of gyration.

    Default behavior returns one Rg value per frame. ``nomax=False`` also
    computes the maximum atom radius from the selected center. ``axes=True``
    includes x/y/z axis radii. ``tensor=True`` includes the six tensor
    components returned by ``radgyr_tensor``.
    """
    del top
    _require_native_radgyr_inputs(traj, "radgyr", length_scale)
    values = _run_native_plan_on_live_traj(
        "RadgyrPlan",
        traj,
        system,
        mask,
        chunk_frames,
        frame_indices,
        device=device,
        mass_weighted=mass,
        include_max=not nomax,
        include_axes=axes,
        include_tensor=tensor,
    )
    expected_cols = 1 + int(not nomax) + (3 if axes else 0) + (6 if tensor else 0)
    if values.ndim != 2 or values.shape[1] != expected_cols:
        raise RuntimeError("native RadgyrPlan returned unexpected output")
    out = {"rg": values[:, 0].astype(np.float32, copy=False)}
    offset = 1
    if not nomax:
        out["max"] = values[:, offset].astype(np.float32, copy=False)
        offset += 1
    if axes:
        out["axes"] = values[:, offset : offset + 3].astype(np.float32, copy=False)
        offset += 3
    if tensor:
        out["tensor"] = values[:, offset : offset + 6].astype(np.float32, copy=False)

    key = str(dtype).lower()
    if key == "dict":
        return out
    if axes and tensor and not nomax:
        return out["rg"], out["max"], out["axes"], out["tensor"]
    if axes and tensor:
        return out["rg"], out["axes"], out["tensor"]
    if axes and not nomax:
        return np.column_stack([out["rg"], out["max"], out["axes"]]).astype(np.float32)
    if axes:
        return np.column_stack([out["rg"], out["axes"]]).astype(np.float32)
    if tensor and not nomax:
        return out["rg"], out["max"], out["tensor"]
    if tensor:
        return out["rg"], out["tensor"]
    if not nomax:
        return np.column_stack([out["rg"], out["max"]]).astype(np.float32)
    return out["rg"]


def gyrate(
    traj,
    system,
    mask: MaskLike = "",
    mass: bool = True,
    axes: bool = True,
    nomax: bool = True,
    tensor: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
    dtype: str = "ndarray",
    device: str = "auto",
):
    """Compute mass-weighted radius of gyration with optional axis radii."""
    return radgyr(
        traj,
        system,
        mask=mask,
        nomax=nomax,
        mass=mass,
        axes=axes,
        tensor=tensor,
        frame_indices=frame_indices,
        chunk_frames=chunk_frames,
        length_scale=length_scale,
        dtype=dtype,
        device=device,
    )


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
    if is_native_traj(traj) and not autoimage and rmsfit is None and float(length_scale) == 1.0:
        mean = _run_native_plan_on_live_traj(
            plan_name,
            traj,
            system,
            mask,
            chunk_frames,
            frame_indices,
        ).reshape(-1, 3)
        if dtype.lower() in ("traj", "trajectory"):
            return ArrayTrajectory(mean[None, :, :].astype(np.float32))
        return mean.astype(np.float32)

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

    if is_native_traj(traj):
        plan_cls = load_native_symbol("StripTrajectoryPlan")
        native_system = coerce_native_system(system)
        if plan_cls is not None and native_system is not None:
            try:
                plan = plan_cls(native_selection(system, native_system, keep_idx))
                payload = plan.run(
                    traj,
                    native_system,
                    chunk_frames=chunk_frames,
                    device="auto",
                )
            except Exception as exc:
                raise RuntimeError("native StripTrajectoryPlan execution failed") from exc
            coords = np.asarray(payload["coords"], dtype=np.float32)
            box = payload.get("box")
            if box is not None:
                box = np.asarray(box, dtype=np.float32)
            time = payload.get("time_ps")
            if time is not None:
                time = np.asarray(time, dtype=np.float64)
            return ArrayTrajectory(coords, box=box, time_ps=time)

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
    "gyrate",
]
