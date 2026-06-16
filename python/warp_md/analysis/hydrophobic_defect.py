from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from ._runtime import load_native_symbol, native_inputs, native_selection


def hydrophobic_defects(
    traj,
    system,
    lipid_selection,
    reference_selection,
    voxel_size: float = 1.0,
    z_bounds: Optional[Tuple[float, float]] = None,
    probe_radius: Optional[float] = None,
    defect_radius: Optional[float] = None,
    length_scale: Optional[float] = None,
    grid_mode: str = "voxel_centers",
    leaflet: str = "both",
    midplane_selection=None,
    leaflet_bins: int = 1,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    plan_cls = load_native_symbol("HydrophobicDefectPlan")
    if plan_cls is None:
        raise RuntimeError("HydrophobicDefectPlan binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames, include_box=True)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native inputs for HydrophobicDefectPlan")
    plan = plan_cls(
        native_selection(system, native_system, lipid_selection),
        native_selection(system, native_system, reference_selection),
        float(voxel_size),
        None if z_bounds is None else (float(z_bounds[0]), float(z_bounds[1])),
        probe_radius=None if probe_radius is None else float(probe_radius),
        defect_radius=None if defect_radius is None else float(defect_radius),
        length_scale=None if length_scale is None else float(length_scale),
        grid_mode=str(grid_mode),
        leaflet=str(leaflet),
        midplane_selection=(
            None
            if midplane_selection is None
            else native_selection(system, native_system, midplane_selection)
        ),
        leaflet_bins=int(leaflet_bins),
    )
    out = plan.run(
        native_traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(i) for i in frame_indices],
    )
    return {
        "kind": "hydrophobic_defects",
        "dims": tuple(int(v) for v in out["dims"]),
        "voxel_size": float(out["voxel_size"]),
        "z_bounds": tuple(float(v) for v in out["z_bounds"]),
        "grid_mode": str(grid_mode),
        "leaflet": str(leaflet),
        "mean": np.asarray(out["mean"]),
        "first": np.asarray(out["first"]),
        "last": np.asarray(out["last"]),
        "min": np.asarray(out["min"]),
        "max": np.asarray(out["max"]),
        "frame_counts": np.asarray(out["frame_counts"]),
        "frame_area": np.asarray(out["frame_area"]),
        "frame_volume": np.asarray(out["frame_volume"]),
        "frame_cluster_count": np.asarray(out["frame_cluster_count"]),
        "frame_largest_cluster": np.asarray(out["frame_largest_cluster"]),
        "max_lifetime": np.asarray(out["max_lifetime"]),
    }


def hydrophobic_defect_points(result, *, mask: str = "last", threshold: float = 0.0) -> np.ndarray:
    dims = tuple(int(v) for v in result["dims"])
    values = np.asarray(result[mask])
    if mask == "mean":
        active = values > float(threshold)
    else:
        active = values > 0
    flat = np.flatnonzero(active)
    if flat.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    nx, ny, _nz = dims
    ix = flat % nx
    iy = (flat // nx) % ny
    iz = flat // (nx * ny)
    voxel = float(result["voxel_size"])
    z0 = float(result["z_bounds"][0])
    offset = 0.0 if result.get("grid_mode") == "lattice_nodes" else 0.5
    return np.column_stack(
        [
            (ix.astype(np.float64) + offset) * voxel,
            (iy.astype(np.float64) + offset) * voxel,
            z0 + (iz.astype(np.float64) + offset) * voxel,
        ]
    ).astype(np.float32)


def write_hydrophobic_defect_points(
    result,
    path,
    *,
    mask: str = "last",
    threshold: float = 0.0,
    format: Optional[str] = None,
):
    path = Path(path)
    fmt = (format or path.suffix.lstrip(".") or "xyz").lower()
    points = hydrophobic_defect_points(result, mask=mask, threshold=threshold)
    if fmt == "xyz":
        with path.open("w", encoding="utf-8") as handle:
            for x, y, z in points:
                handle.write(f"1\t {x:.2f}\t {y:.2f}\t {z:.2f}\n")
    elif fmt == "csv":
        np.savetxt(path, points, delimiter=",", header="x,y,z", comments="")
    else:
        raise ValueError("format must be xyz or csv")
    return {"path": str(path), "format": fmt, "points": int(points.shape[0])}


__all__ = [
    "hydrophobic_defects",
    "hydrophobic_defect_points",
    "write_hydrophobic_defect_points",
]
