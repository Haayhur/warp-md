from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md

from ._runtime import native_inputs
from .h2order import _select

MaskLike = Union[str, Sequence[int], np.ndarray]


def _plan(name: str):
    plan = getattr(warp_md.traj_py, name, None) if getattr(warp_md, "traj_py", None) else None
    if plan is None:
        raise RuntimeError(f"{name} binding unavailable. Rebuild bindings with `maturin develop`.")
    return plan


def _native(traj, system, chunk_frames, *, include_box=True):
    native_traj, native_system = native_inputs(
        traj,
        system,
        chunk_frames,
        include_box=include_box,
        include_time=False,
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("lipid analysis requires Rust-backed or coercible trajectory/system objects")
    return native_traj, native_system


def _matrix_out(out):
    return {
        "values": np.asarray(out["values"], dtype=np.float32),
        "residue_ids": np.asarray(out["residue_ids"], dtype=np.int32),
        "frames": np.asarray(out["frames"], dtype=np.int64),
        "kind": str(out["kind"]),
    }


def _bin_edges(bins, x=None, y=None):
    if np.isscalar(bins):
        if x is None or y is None:
            raise ValueError("x and y are required when bins is an integer")
        n = int(bins)
        if n <= 0:
            raise ValueError("bins must be positive")
        return (
            np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), n + 1, dtype=np.float64),
            np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), n + 1, dtype=np.float64),
        )
    if isinstance(bins, (tuple, list)) and len(bins) == 2:
        bx, by = bins
        if np.isscalar(bx):
            if x is None:
                raise ValueError("x is required when x bins is an integer")
            bx = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), int(bx) + 1, dtype=np.float64)
        else:
            bx = np.asarray(bx, dtype=np.float64)
        if np.isscalar(by):
            if y is None:
                raise ValueError("y is required when y bins is an integer")
            by = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), int(by) + 1, dtype=np.float64)
        else:
            by = np.asarray(by, dtype=np.float64)
        return bx, by
    edges = np.asarray(bins, dtype=np.float64)
    return edges, edges


def _native_binned_statistic_2d(x, y, values, x_edges, y_edges, statistic):
    fn = getattr(warp_md, "binned_statistic_2d_array", None)
    if fn is None:
        return None
    if (
        getattr(fn, "__name__", "") == "binned_statistic_2d_array"
        and getattr(warp_md, "traj_py", None) is None
    ):
        return None
    try:
        grid, counts = fn(
            x.astype(np.float64, copy=False),
            y.astype(np.float64, copy=False),
            values.astype(np.float64, copy=False),
            x_edges.astype(np.float64, copy=False),
            y_edges.astype(np.float64, copy=False),
            str(statistic),
        )
    except RuntimeError:
        return None
    return np.asarray(grid, dtype=np.float32), np.asarray(counts, dtype=np.int64)


def _binned_statistic_2d(x, y, values, bins, statistic):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    if not (x.shape == y.shape == values.shape):
        raise ValueError("x, y, and values must have the same shape")
    x_edges, y_edges = _bin_edges(bins, x, y)
    if x_edges.ndim != 1 or y_edges.ndim != 1 or x_edges.size < 2 or y_edges.size < 2:
        raise ValueError("bins must define at least one bin per dimension")
    native = _native_binned_statistic_2d(x, y, values, x_edges, y_edges, statistic)
    if native is not None:
        grid, counts = native
        return grid, x_edges, y_edges, counts
    nx = x_edges.size - 1
    ny = y_edges.size - 1
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x = x[finite]
    y = y[finite]
    values = values[finite]
    ix = np.searchsorted(x_edges, x, side="right") - 1
    iy = np.searchsorted(y_edges, y, side="right") - 1
    ix[x == x_edges[-1]] = nx - 1
    iy[y == y_edges[-1]] = ny - 1
    keep = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[keep]
    iy = iy[keep]
    values = values[keep]
    count = np.zeros((nx, ny), dtype=np.float64)
    np.add.at(count, (ix, iy), 1.0)
    statistic = str(statistic)
    if statistic == "count":
        return count.astype(np.float32), x_edges, y_edges, count.astype(np.int64)
    if statistic in {"mean", "sum", "std"}:
        total = np.zeros((nx, ny), dtype=np.float64)
        np.add.at(total, (ix, iy), values)
        if statistic == "sum":
            out = total
        elif statistic == "mean":
            out = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        else:
            total2 = np.zeros((nx, ny), dtype=np.float64)
            np.add.at(total2, (ix, iy), values * values)
            mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
            out = np.sqrt(np.divide(total2, count, out=np.full_like(total2, np.nan), where=count > 0) - mean * mean)
        return out.astype(np.float32), x_edges, y_edges, count.astype(np.int64)
    out = np.full((nx, ny), np.nan, dtype=np.float64)
    for bx in range(nx):
        for by in range(ny):
            cell = values[(ix == bx) & (iy == by)]
            if cell.size == 0:
                continue
            if statistic == "median":
                out[bx, by] = np.median(cell)
            elif statistic == "min":
                out[bx, by] = np.min(cell)
            elif statistic == "max":
                out[bx, by] = np.max(cell)
            else:
                raise ValueError("statistic must be mean, std, median, count, sum, min, or max")
    return out.astype(np.float32), x_edges, y_edges, count.astype(np.int64)


def _nearest_fill_grid(values, *, tile=True):
    grid = np.asarray(values, dtype=np.float32)
    if not np.isnan(grid).any():
        return grid
    fn = getattr(warp_md, "nearest_fill_grid_array", None)
    if fn is not None and not (
        getattr(fn, "__name__", "") == "nearest_fill_grid_array"
        and getattr(warp_md, "traj_py", None) is None
    ):
        try:
            return np.asarray(fn(grid, bool(tile)), dtype=np.float32)
        except RuntimeError:
            pass
    work = np.tile(grid, (3, 3)) if tile else grid.copy()
    known = np.argwhere(np.isfinite(work))
    missing = np.argwhere(~np.isfinite(work))
    if known.size == 0:
        return grid
    for point in missing:
        delta = known - point
        idx = int(np.argmin(np.sum(delta * delta, axis=1)))
        work[tuple(point)] = work[tuple(known[idx])]
    if not tile:
        return work
    nx, ny = grid.shape
    return work[nx : 2 * nx, ny : 2 * ny]


def lipid_leaflets(
    traj,
    system,
    selection: MaskLike = "all",
    *,
    midplane_selection: Optional[MaskLike] = None,
    midplane_cutoff: float = 0.0,
    bins: int = 1,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    sel = _select(native_system, selection)
    mid = None if midplane_selection is None else _select(native_system, midplane_selection)
    plan = _plan("PyLipidLeafletPlan")(
        sel,
        midplane_selection=mid,
        midplane_cutoff=float(midplane_cutoff),
        bins=int(bins),
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_curved_leaflets(
    traj,
    system,
    selection: MaskLike = "all",
    *,
    cutoff: float = 15.0,
    midplane_selection: Optional[MaskLike] = None,
    midplane_cutoff: float = 0.0,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    sel = _select(native_system, selection)
    mid = None if midplane_selection is None else _select(native_system, midplane_selection)
    plan = _plan("PyLipidCurvedLeafletPlan")(
        sel,
        cutoff=float(cutoff),
        midplane_selection=mid,
        midplane_cutoff=float(midplane_cutoff),
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_z_positions(
    traj,
    system,
    membrane_selection: MaskLike = "all",
    height_selection: MaskLike = "all",
    *,
    bins: int = 1,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    plan = _plan("PyLipidZPositionPlan")(
        _select(native_system, membrane_selection),
        _select(native_system, height_selection),
        bins=int(bins),
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_z_thickness(
    traj,
    system,
    selection: MaskLike = "all",
    *,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    plan = _plan("PyLipidZThicknessPlan")(_select(native_system, selection), length_scale=length_scale)
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_z_angles(
    traj,
    system,
    atom_a: MaskLike,
    atom_b: MaskLike,
    *,
    degrees: bool = True,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    plan = _plan("PyLipidZAnglePlan")(
        _select(native_system, atom_a),
        _select(native_system, atom_b),
        degrees=bool(degrees),
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_area(
    traj,
    system,
    selection: MaskLike,
    leaflets,
    *,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    leaflets_arr = np.asarray(leaflets, dtype=np.int8)
    if leaflets_arr.ndim == 1:
        leaflets_arr = leaflets_arr[:, np.newaxis]
    if leaflets_arr.ndim != 2:
        raise ValueError("leaflets must be a 1D or 2D array")
    plan = _plan("PyLipidAreaPlan")(
        _select(native_system, selection),
        leaflets_arr,
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_flip_flop(traj, system, leaflets, *, residue_ids=None, frame_cutoff: int = 1, chunk_frames=None, device="auto"):
    native_traj, native_system = _native(traj, system, chunk_frames, include_box=False)
    leaflets_arr = np.asarray(leaflets, dtype=np.int8)
    if leaflets_arr.ndim != 2:
        raise ValueError("leaflets must be a 2D array")
    plan = _plan("PyLipidFlipFlopPlan")(
        leaflets_arr,
        residue_ids=None if residue_ids is None else [int(v) for v in residue_ids],
        frame_cutoff=int(frame_cutoff),
    )
    out = plan.run(native_traj, native_system, chunk_frames=chunk_frames, device=device)
    return {
        "events": np.asarray(out["events"], dtype=np.int32),
        "success": np.asarray(out["success"], dtype=object),
        "residue_ids": np.asarray(out["residue_ids"], dtype=np.int32),
    }


def lipid_neighbours(
    traj,
    system,
    selection: MaskLike,
    *,
    cutoff: float = 10.0,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    plan = _plan("PyLipidNeighbourPlan")(
        _select(native_system, selection),
        cutoff=float(cutoff),
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_neighbour_matrix(
    traj,
    system,
    selection: MaskLike,
    *,
    cutoff: float = 10.0,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    plan = _plan("PyLipidNeighbourMatrixPlan")(
        _select(native_system, selection),
        cutoff=float(cutoff),
        length_scale=length_scale,
    )
    out = _matrix_out(
        plan.run(
            native_traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
        )
    )
    n_res = out["residue_ids"].shape[0]
    values = out["values"].reshape(n_res, n_res, out["values"].shape[1]).transpose(2, 0, 1)
    return {
        "values": values.astype(np.int8),
        "residue_ids": out["residue_ids"],
        "frames": out["frames"],
        "kind": out["kind"],
    }


def lipid_neighbour_composition(neighbour_matrix, labels, *, label_names=None, return_enrichment: bool = False):
    matrix = np.asarray(neighbour_matrix["values"] if isinstance(neighbour_matrix, dict) else neighbour_matrix)
    if matrix.ndim != 3:
        raise ValueError("neighbour_matrix must have shape (n_frames, n_residues, n_residues)")
    labels_arr = np.asarray(labels)
    if labels_arr.ndim == 1:
        labels_arr = np.broadcast_to(labels_arr[:, np.newaxis], (labels_arr.shape[0], matrix.shape[0]))
    if labels_arr.shape != (matrix.shape[1], matrix.shape[0]):
        raise ValueError("labels must have shape (n_residues,) or (n_residues, n_frames)")
    if label_names is None:
        label_values = list(np.unique(labels_arr))
        names = [str(v) for v in label_values]
    else:
        names = list(label_names.keys())
        label_values = [label_names[name] for name in names]
    fn = getattr(warp_md, "lipid_neighbour_composition_array", None)
    if fn is not None and not (
        getattr(fn, "__name__", "") == "lipid_neighbour_composition_array"
        and getattr(warp_md, "traj_py", None) is None
    ):
        try:
            counts = np.asarray(
                fn(
                    matrix.astype(np.int64, copy=False),
                    labels_arr.astype(np.int64, copy=False),
                    np.asarray(label_values, dtype=np.int64),
                ),
                dtype=np.int32,
            )
        except RuntimeError:
            counts = None
    else:
        counts = None
    if counts is None:
        counts = np.zeros((matrix.shape[0], matrix.shape[1], len(label_values)), dtype=np.int32)
        for frame in range(matrix.shape[0]):
            for label_idx, label_value in enumerate(label_values):
                mask = labels_arr[:, frame] == label_value
                counts[frame, :, label_idx] = matrix[frame][:, mask].sum(axis=1)
    out = {
        "counts": counts,
        "labels": np.asarray(names, dtype=object),
        "kind": "neighbour_composition",
    }
    if return_enrichment:
        frame_mean = counts.mean(axis=1, keepdims=True)
        out["enrichment"] = np.divide(
            counts,
            frame_mean,
            out=np.full(counts.shape, np.nan, dtype=np.float32),
            where=frame_mean > 0,
        )
    return out


def lipid_largest_cluster(
    traj,
    system,
    selection: MaskLike,
    *,
    cutoff: float = 10.0,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    plan = _plan("PyLipidLargestClusterPlan")(
        _select(native_system, selection),
        cutoff=float(cutoff),
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_membrane_thickness(
    traj,
    system,
    selection: MaskLike,
    leaflets,
    *,
    bins: int = 1,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    leaflets_arr = np.asarray(leaflets, dtype=np.int8)
    if leaflets_arr.ndim == 1:
        leaflets_arr = leaflets_arr[:, np.newaxis]
    if leaflets_arr.ndim != 2:
        raise ValueError("leaflets must be a 1D or 2D array")
    plan = _plan("PyLipidMembraneThicknessPlan")(
        _select(native_system, selection),
        leaflets_arr,
        bins=int(bins),
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_registration(
    traj,
    system,
    upper_selection: MaskLike,
    lower_selection: MaskLike,
    leaflets,
    *,
    bins: int = 1,
    gaussian_sd: float = 0.0,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    leaflets_arr = np.asarray(leaflets, dtype=np.int8)
    if leaflets_arr.ndim == 1:
        leaflets_arr = leaflets_arr[:, np.newaxis]
    if leaflets_arr.ndim != 2:
        raise ValueError("leaflets must be a 1D or 2D array")
    plan = _plan("PyLipidRegistrationPlan")(
        _select(native_system, upper_selection),
        _select(native_system, lower_selection),
        leaflets_arr,
        bins=int(bins),
        gaussian_sd=float(gaussian_sd),
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_msd(
    traj,
    system,
    selection: MaskLike,
    *,
    com_removal_selection: Optional[MaskLike] = None,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames, include_box=False)
    com_sel = None if com_removal_selection is None else _select(native_system, com_removal_selection)
    plan = _plan("PyLipidMsdPlan")(
        _select(native_system, selection),
        com_removal_selection=com_sel,
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_scc(
    traj,
    system,
    tail_selection: MaskLike,
    *,
    normals=None,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    native_traj, native_system = _native(traj, system, chunk_frames)
    normals_arr = None if normals is None else np.asarray(normals, dtype=np.float32)
    if normals_arr is not None and (normals_arr.ndim != 3 or normals_arr.shape[2] != 3):
        raise ValueError("normals must have shape (n_residues, n_frames, 3)")
    plan = _plan("PyLipidSccPlan")(
        _select(native_system, tail_selection),
        normals=normals_arr,
        length_scale=length_scale,
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
    )
    return _matrix_out(out)


def lipid_scc_weighted_average(sn1, sn2, *, sn1_weight: float = 1.0, sn2_weight: float = 1.0):
    sn1_values = np.asarray(sn1["values"], dtype=np.float32)
    sn2_values = np.asarray(sn2["values"], dtype=np.float32)
    sn1_ids = np.asarray(sn1["residue_ids"], dtype=np.int32)
    sn2_ids = np.asarray(sn2["residue_ids"], dtype=np.int32)
    if sn1_values.shape[1] != sn2_values.shape[1]:
        raise ValueError("SCC inputs must have the same number of frames")
    residue_ids = np.union1d(sn1_ids, sn2_ids).astype(np.int32)
    out = np.full((residue_ids.size, sn1_values.shape[1]), np.nan, dtype=np.float32)
    for row, resid in enumerate(residue_ids):
        weighted = []
        weights = []
        if resid in sn1_ids:
            weighted.append(sn1_values[np.where(sn1_ids == resid)[0][0]] * float(sn1_weight))
            weights.append(float(sn1_weight))
        if resid in sn2_ids:
            weighted.append(sn2_values[np.where(sn2_ids == resid)[0][0]] * float(sn2_weight))
            weights.append(float(sn2_weight))
        out[row] = np.nansum(np.stack(weighted, axis=0), axis=0) / np.sum(weights)
    return {
        "values": out,
        "residue_ids": residue_ids,
        "frames": np.asarray(sn1["frames"], dtype=np.int64),
        "kind": "scc_weighted_average",
    }


def lipid_project_values(
    x,
    y,
    values,
    *,
    bins,
    statistic: str = "mean",
    interpolate: Optional[str] = None,
    tile: bool = True,
):
    grid, x_edges, y_edges, counts = _binned_statistic_2d(x, y, values, bins, statistic)
    if interpolate is not None:
        method = str(interpolate)
        if method != "nearest":
            raise ValueError("only nearest interpolation is supported")
        grid = _nearest_fill_grid(grid, tile=tile)
    return {
        "statistic": grid,
        "x_edges": x_edges.astype(np.float32),
        "y_edges": y_edges.astype(np.float32),
        "counts": counts,
        "kind": "projection",
    }


def lipid_joint_density(observable_x, observable_y, *, bins, temperature: Optional[float] = None):
    x = np.asarray(observable_x, dtype=np.float64).ravel()
    y = np.asarray(observable_y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError("observable_x and observable_y must have the same shape")
    x_edges, y_edges = _bin_edges(bins, x, y)
    finite = np.isfinite(x) & np.isfinite(y)
    hist, x_edges, y_edges = np.histogram2d(x[finite], y[finite], bins=(x_edges, y_edges))
    density = hist / hist.sum() if hist.sum() > 0 else hist
    out = {
        "density": density.astype(np.float32),
        "x_edges": x_edges.astype(np.float32),
        "y_edges": y_edges.astype(np.float32),
        "counts": hist.astype(np.int64),
        "kind": "joint_density",
    }
    if temperature is not None:
        k_b = 0.00831446261815324
        with np.errstate(divide="ignore", invalid="ignore"):
            pmf = -k_b * float(temperature) * np.log(density)
        finite_pmf = np.isfinite(pmf)
        if finite_pmf.any():
            pmf[finite_pmf] -= np.nanmin(pmf[finite_pmf])
        pmf[~finite_pmf] = np.nan
        out["pmf"] = pmf.astype(np.float32)
    return out


__all__ = [
    "lipid_area",
    "lipid_curved_leaflets",
    "lipid_flip_flop",
    "lipid_largest_cluster",
    "lipid_leaflets",
    "lipid_membrane_thickness",
    "lipid_msd",
    "lipid_neighbours",
    "lipid_neighbour_composition",
    "lipid_neighbour_matrix",
    "lipid_joint_density",
    "lipid_project_values",
    "lipid_registration",
    "lipid_scc",
    "lipid_scc_weighted_average",
    "lipid_z_angles",
    "lipid_z_positions",
    "lipid_z_thickness",
]
