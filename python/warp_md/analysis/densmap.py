from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md


MaskLike = Union[str, Sequence[int], np.ndarray]

_DensityMapPlan = (
    getattr(warp_md.traj_py, "PyDensityMapPlan", None)
    if getattr(warp_md, "traj_py", None)
    else None
)


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _select(system, selection: Optional[MaskLike]):
    if isinstance(selection, str) or selection is None:
        expr = selection
        if selection in ("", "*", "all", None):
            expr = _all_resid_mask(system) if hasattr(system, "atom_table") else "all"
        return system.select(expr)
    indices = np.asarray(selection, dtype=np.int64).reshape(-1).tolist()
    if hasattr(system, "select_indices"):
        return system.select_indices(indices)
    raise RuntimeError("system.select_indices is required for non-string densmap selections")


def densmap(
    traj,
    system,
    selection: Optional[MaskLike] = "",
    average: str = "z",
    bin: float = 0.25,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    unit: str = "nm-3",
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Planar 2D density map via Rust plan path.

    For GROMACS-style nm output from Angstrom trajectories, pass
    `length_scale=0.1` and choose `bin` in nm.
    """
    if _DensityMapPlan is None:
        raise RuntimeError(
            "PyDensityMapPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    sel = _select(system, selection)
    plan = _DensityMapPlan(
        sel,
        average=average,
        bin=bin,
        n1=n1,
        n2=n2,
        xmin=xmin,
        xmax=xmax,
        unit=unit,
        length_scale=length_scale,
    )
    try:
        out = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
        )
    except TypeError as exc:
        raise RuntimeError(
            "densmap requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "axis1": np.asarray(out["axis1"], dtype=np.float32),
        "axis2": np.asarray(out["axis2"], dtype=np.float32),
        "matrix": np.asarray(out["matrix"], dtype=np.float32),
        "plane_axes": list(out["plane_axes"]),
        "average_axis": str(out["average_axis"]),
        "unit": str(out["unit"]),
        "n_frames": int(out["n_frames"]),
        "bounds": np.asarray(out["bounds"], dtype=np.float32),
        "bin_width": np.asarray(out["bin_width"], dtype=np.float32),
        "used_box": bool(out["used_box"]),
        "length_scale": float(out["length_scale"]),
    }


__all__ = ["densmap"]
