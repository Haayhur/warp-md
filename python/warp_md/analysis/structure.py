# Usage:
# from warp_md.analysis.structure import mean_structure, get_average_frame
# avg = mean_structure(traj, system, mask="name CA", frame_indices=[0, 5])

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._chunk_io import read_chunk_fields
from .trajectory import ArrayTrajectory
from .align import superpose

try:  # optional bindings
    from warp_md import MeanStructurePlan, MakeStructurePlan
except Exception:  # pragma: no cover
    MeanStructurePlan = None
    MakeStructurePlan = None


def _all_resid_mask(system) -> str:
    if not hasattr(system, "atom_table"):
        return "all"
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask) -> np.ndarray:
    if mask is None or mask in ("", "*", "all"):
        sel = system.select(_all_resid_mask(system))
        return np.asarray(list(sel.indices), dtype=np.int64)
    if isinstance(mask, str):
        sel = system.select(mask)
        return np.asarray(list(sel.indices), dtype=np.int64)
    return np.asarray([int(i) for i in mask], dtype=np.int64)


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    box_list = []
    time_list = []
    chunk = read_chunk_fields(traj, max_chunk, include_box=True, include_time=True)
    if chunk is None:
        return None, None, None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        box = chunk.get("box")
        if box is not None:
            box_list.append(np.asarray(box, dtype=np.float64))
        time = chunk.get("time")
        if time is None:
            time = chunk.get("time_ps")
        if time is not None:
            time_list.append(np.asarray(time, dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk, include_box=True, include_time=True)
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    box = np.concatenate(box_list, axis=0) if box_list else None
    time = np.concatenate(time_list, axis=0) if time_list else None
    return coords, box, time


def radgyr_tensor(
    traj,
    system,
    mask: Union[str, Sequence[int], None] = "",
    mass: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
    dtype: str = "ndarray",
):
    """Compute radius of gyration tensor (rg + tensor components)."""
    coords, _box, _time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        empty_rg = np.empty(0, dtype=np.float32)
        empty_tensor = np.empty((0, 6), dtype=np.float32)
        if dtype == "dict":
            return {"rg": empty_rg, "tensor": empty_tensor}
        return empty_rg, empty_tensor

    coords = coords * float(length_scale)

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

    idx = _selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    masses = None
    if mass:
        atoms = system.atom_table()
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)
        if masses.size == 0:
            masses = None

    rg_vals = np.empty(coords.shape[0], dtype=np.float64)
    tensor = np.empty((coords.shape[0], 6), dtype=np.float64)
    for f in range(coords.shape[0]):
        sel_coords = coords[f, idx, :]
        if sel_coords.size == 0:
            rg_vals[f] = 0.0
            tensor[f] = 0.0
            continue
        if masses is None:
            w = np.ones(sel_coords.shape[0], dtype=np.float64)
        else:
            w = masses[idx]
        wsum = float(np.sum(w))
        if wsum == 0.0:
            rg_vals[f] = 0.0
            tensor[f] = 0.0
            continue
        com = (sel_coords * w[:, None]).sum(axis=0) / wsum
        disp = sel_coords - com
        g_xx = float(np.sum(w * disp[:, 0] * disp[:, 0]) / wsum)
        g_yy = float(np.sum(w * disp[:, 1] * disp[:, 1]) / wsum)
        g_zz = float(np.sum(w * disp[:, 2] * disp[:, 2]) / wsum)
        g_xy = float(np.sum(w * disp[:, 0] * disp[:, 1]) / wsum)
        g_xz = float(np.sum(w * disp[:, 0] * disp[:, 2]) / wsum)
        g_yz = float(np.sum(w * disp[:, 1] * disp[:, 2]) / wsum)
        rg_vals[f] = np.sqrt(g_xx + g_yy + g_zz)
        tensor[f] = [g_xx, g_yy, g_zz, g_xy, g_xz, g_yz]

    rg_vals = rg_vals.astype(np.float32)
    tensor = tensor.astype(np.float32)
    if dtype == "dict":
        return {"rg": rg_vals, "tensor": tensor}
    return rg_vals, tensor


def _autoimage(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    out = coords.copy()
    for f in range(out.shape[0]):
        lx, ly, lz = box[f]
        if lx > 0:
            out[f, :, 0] -= np.floor(out[f, :, 0] / lx) * lx
        if ly > 0:
            out[f, :, 1] -= np.floor(out[f, :, 1] / ly) * ly
        if lz > 0:
            out[f, :, 2] -= np.floor(out[f, :, 2] / lz) * lz
    return out


def mean_structure(
    traj,
    system,
    mask: Union[str, Sequence[int], None] = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "frame",
    autoimage: bool = False,
    rmsfit: Optional[Union[int, str]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Compute average structure for selected atoms."""
    use_plan = MeanStructurePlan is not None and getattr(MeanStructurePlan, "__name__", "") != "_Missing"
    if use_plan and hasattr(system, "select"):
        try:
            if isinstance(mask, str):
                sel = system.select(mask if mask not in ("", "*", "all", None) else _all_resid_mask(system))
            else:
                sel = system.select_indices(list(_selection_indices(system, mask)))
            plan = MeanStructurePlan(sel)
            out = plan.run(traj, system, chunk_frames=chunk_frames, device="auto")
            mean = np.asarray(out, dtype=np.float32).reshape((-1, 3)) if out is not None else np.empty((0, 3), dtype=np.float32)
            if dtype.lower() in ("traj", "trajectory"):
                return ArrayTrajectory(mean[None, :, :].astype(np.float32))
            return mean.astype(np.float32)
        except TypeError:
            pass

    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        if dtype.lower() in ("traj", "trajectory"):
            return ArrayTrajectory(coords.astype(np.float32), box=box, time_ps=time)
        return np.empty((0, 3), dtype=np.float32)

    coords = coords * float(length_scale)
    if box is not None:
        box = box * float(length_scale)

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
        if time is not None:
            time = time[select]

    if autoimage:
        if box is None:
            raise ValueError("autoimage requires box lengths")
        coords = _autoimage(coords, box)

    if rmsfit is not None:
        ref = rmsfit
        temp = ArrayTrajectory(coords.astype(np.float32), box=box, time_ps=time)
        aligned = superpose(temp, system, mask=mask if isinstance(mask, str) else "", ref=ref)
        chunk = aligned.read_chunk(aligned.n_frames())
        coords = np.asarray(chunk["coords"], dtype=np.float64)

    idx = _selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")
    mean = coords[:, idx, :].mean(axis=0)

    if dtype.lower() in ("traj", "trajectory"):
        return ArrayTrajectory(mean[None, :, :].astype(np.float32))
    return mean.astype(np.float32)


def strip(
    traj,
    system,
    mask: Union[str, Sequence[int]],
    chunk_frames: Optional[int] = None,
):
    """Return trajectory with atoms stripped (keeps inverse selection)."""
    if isinstance(mask, str):
        keep_mask = f"!({mask})"
        keep_idx = _selection_indices(system, keep_mask)
    else:
        mask_idx = set(int(i) for i in mask)
        atoms = system.atom_table()
        n_atoms = len(atoms.get("name", []))
        keep_idx = np.array([i for i in range(n_atoms) if i not in mask_idx], dtype=np.int64)
    if keep_idx.size == 0:
        raise ValueError("strip would remove all atoms")

    coords, box, time = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return ArrayTrajectory(coords.astype(np.float32), box=box, time_ps=time)
    out = coords[:, keep_idx, :]
    return ArrayTrajectory(out.astype(np.float32), box=box, time_ps=time)


def get_average_frame(*args, **kwargs):
    return mean_structure(*args, **kwargs)


def make_structure(
    traj,
    system,
    mask: Union[str, Sequence[int], None] = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "frame",
    autoimage: bool = False,
    rmsfit: Optional[Union[int, str]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
):
    """Alias for mean_structure (cpptraj-style)."""
    use_plan = MakeStructurePlan is not None and getattr(MakeStructurePlan, "__name__", "") != "_Missing"
    if use_plan and hasattr(system, "select"):
        try:
            if isinstance(mask, str):
                sel = system.select(mask if mask not in ("", "*", "all", None) else _all_resid_mask(system))
            else:
                sel = system.select_indices(list(_selection_indices(system, mask)))
            plan = MakeStructurePlan(sel)
            out = plan.run(traj, system, chunk_frames=chunk_frames, device="auto")
            mean = np.asarray(out, dtype=np.float32).reshape((-1, 3)) if out is not None else np.empty((0, 3), dtype=np.float32)
            if dtype.lower() in ("traj", "trajectory"):
                return ArrayTrajectory(mean[None, :, :].astype(np.float32))
            return mean.astype(np.float32)
        except TypeError:
            pass
    return mean_structure(
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


__all__ = ["mean_structure", "get_average_frame", "make_structure", "strip", "radgyr_tensor"]
