# Usage:
# from warp_md.analysis.fluct import rmsf, atomicfluct, bfactors
# data = rmsf(traj, system, mask="name CA", byres=True)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._runtime import (
    load_native_symbol,
    native_inputs,
    native_selection,
    read_frame_subset,
    selection_indices,
)
from .trajectory import ArrayTrajectory


def _group_by_resid(system, indices: np.ndarray):
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        raise ValueError("system atom table has no resid data")
    resids = np.asarray(resids, dtype=np.int64)
    order = []
    groups = {}
    for pos, atom_idx in enumerate(indices.tolist()):
        if atom_idx >= resids.size:
            continue
        resid = int(resids[atom_idx])
        if resid not in groups:
            groups[resid] = []
            order.append(resid)
        groups[resid].append(pos)
    return order, groups


def _pack_indexed(indices: Sequence[int], values: Sequence[float]) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    vals = np.asarray(values, dtype=np.float64)
    if idx.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack((idx.astype(np.float32), vals.astype(np.float32)))


def _aggregate(values: np.ndarray, indices: np.ndarray, system, byres: bool, bymask: bool) -> np.ndarray:
    if bymask:
        mean_val = float(np.mean(values)) if values.size else 0.0
        return _pack_indexed([0], [mean_val])
    if byres:
        order, groups = _group_by_resid(system, indices)
        out_vals = []
        for resid in order:
            idx = groups.get(resid, [])
            out_vals.append(float(np.mean(values[idx])) if idx else 0.0)
        return _pack_indexed(order, out_vals)
    return _pack_indexed(indices, values)


def _native_fluct_series(
    plan_name: str,
    traj,
    system,
    mask: str,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]],
):
    plan_cls = load_native_symbol(plan_name)
    if plan_cls is None:
        raise RuntimeError(f"{plan_name} binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError(f"failed to prepare native inputs for {plan_name}")
    run_kwargs = {
        "chunk_frames": chunk_frames,
        "device": "auto",
    }
    if frame_indices is not None:
        run_kwargs["frame_indices"] = [int(i) for i in frame_indices]
    try:
        plan = plan_cls(native_selection(system, native_system, mask))
        return np.asarray(plan.run(native_traj, native_system, **run_kwargs), dtype=np.float32)
    except TypeError as exc:
        if frame_indices is None or "frame_indices" not in str(exc):
            raise RuntimeError(f"native {plan_name} execution failed") from exc
        coords, box, time, _source_indices = read_frame_subset(
            traj,
            frame_indices,
            chunk_frames,
            include_box=True,
            include_time=True,
        )
        if coords is None:
            raise RuntimeError(f"failed to prepare native frame subset for {plan_name}") from exc
        source = ArrayTrajectory(
            np.asarray(coords, dtype=np.float32),
            box=None if box is None else np.asarray(box, dtype=np.float32),
            time_ps=None if time is None else np.asarray(time, dtype=np.float32),
        )
        native_traj, native_system = native_inputs(source, system, chunk_frames)
        if native_traj is None or native_system is None:
            raise RuntimeError(f"failed to prepare native frame subset for {plan_name}") from exc
        try:
            plan = plan_cls(native_selection(system, native_system, mask))
            return np.asarray(
                plan.run(
                    native_traj,
                    native_system,
                    chunk_frames=chunk_frames,
                    device="auto",
                ),
                dtype=np.float32,
            )
        except Exception as retry_exc:
            raise RuntimeError(f"native {plan_name} execution failed") from retry_exc
    except Exception as exc:
        raise RuntimeError(f"native {plan_name} execution failed") from exc


def _native_rmsf(
    traj,
    system,
    mask: str,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]],
) -> np.ndarray:
    return _native_fluct_series(
        "AtomicFluctPlan",
        traj,
        system,
        mask,
        chunk_frames,
        frame_indices,
    ).reshape(-1)


def _native_adp(
    traj,
    system,
    mask: str,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]],
    length_scale: float,
    indices: np.ndarray,
) -> np.ndarray:
    values = _native_fluct_series(
        "AtomicAdpPlan",
        traj,
        system,
        mask,
        chunk_frames,
        frame_indices,
    )
    if values.size == 0:
        return np.empty((0, 7), dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 6:
        raise RuntimeError("native AtomicAdpPlan returned unexpected output")
    out = np.empty((values.shape[0], 7), dtype=np.float32)
    out[:, 0] = indices.astype(np.float32)
    out[:, 1:] = values.astype(np.float32, copy=False) * np.float32(length_scale * length_scale)
    return out


def rmsf(
    traj,
    system,
    mask: str = "",
    byres: bool = False,
    bymask: bool = False,
    calcadp: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Compute RMSF for selected atoms."""
    if byres and bymask:
        raise ValueError("byres and bymask are mutually exclusive")
    indices = selection_indices(system, mask)
    if indices.size == 0:
        raise ValueError("selection resolved to empty set")

    if calcadp:
        return _native_adp(
            traj,
            system,
            mask,
            chunk_frames,
            frame_indices,
            length_scale,
            indices,
        )

    rmsf_vals = _native_rmsf(traj, system, mask, chunk_frames, frame_indices) * np.float32(length_scale)
    return _aggregate(rmsf_vals, indices, system, byres, bymask)


def atomicfluct(
    traj,
    system,
    mask: str = "",
    byres: bool = False,
    bymask: bool = False,
    calcadp: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Alias of RMSF with optional ADP output."""
    if calcadp and (byres or bymask):
        raise ValueError("calcadp output is only supported for byatom mode")
    return rmsf(
        traj,
        system,
        mask=mask,
        byres=byres,
        bymask=bymask,
        calcadp=calcadp,
        frame_indices=frame_indices,
        chunk_frames=chunk_frames,
        length_scale=length_scale,
    )


def bfactors(
    traj,
    system,
    mask: str = "",
    byres: bool = True,
    bymask: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Compute pseudo B-factors from RMSF."""
    if byres and bymask:
        raise ValueError("byres and bymask are mutually exclusive")
    rmsf_data = rmsf(
        traj,
        system,
        mask=mask,
        byres=False,
        bymask=False,
        calcadp=False,
        frame_indices=frame_indices,
        chunk_frames=chunk_frames,
        length_scale=length_scale,
    )
    if rmsf_data.size == 0:
        return rmsf_data
    indices = rmsf_data[:, 0].astype(np.int64)
    rmsf_vals = rmsf_data[:, 1]
    factor = 8.0 * np.pi * np.pi / 3.0
    bvals = factor * rmsf_vals * rmsf_vals
    return _aggregate(bvals, indices, system, byres, bymask)


__all__ = ["rmsf", "atomicfluct", "bfactors"]
