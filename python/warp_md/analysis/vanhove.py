# Usage:
# from warp_md.analysis.vanhove import vanhove
# out = vanhove(traj, system, selection="name OW", r_bin=0.1, r_max=10.0)

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np


MaskLike = Union[str, Sequence[int], np.ndarray]


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _select(system, selection: MaskLike):
    if isinstance(selection, str):
        expr = selection
        if selection in ("", "*", "all", None):
            expr = _all_resid_mask(system) if hasattr(system, "atom_table") else "all"
        return system.select(expr)
    indices = np.asarray(selection, dtype=np.int64).reshape(-1).tolist()
    if hasattr(system, "select_indices"):
        return system.select_indices(indices)
    raise RuntimeError("system.select_indices is required for non-string vanhove selections")


def vanhove(
    traj,
    system,
    selection: MaskLike = "",
    r_bin: float = 0.1,
    r_max: float = 10.0,
    length_scale: Optional[float] = None,
    max_lag: Optional[int] = None,
    sqrt_time_bin: Optional[float] = None,
    integral_radius: Optional[float] = None,
    curve_lags: Optional[Sequence[int]] = None,
    curve_step: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
    scale_to_average_box: bool = True,
    remove_pbc_jumps: bool = True,
    time_scale: Optional[float] = None,
):
    """Self part of the Van Hove displacement function.

    Returns a dict with `time`, `time_sqrt`, `r`, and `matrix` where
    `matrix[i, j] = G(r_j, t_i)`.

    For GROMACS-style nm output from Angstrom trajectories, pass
    `length_scale=0.1`, `r_bin=0.01`, and `r_max=2.0`.
    """
    try:
        from warp_md import VanHovePlan  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "VanHovePlan binding unavailable. Rebuild bindings with `maturin develop`."
        ) from exc
    if getattr(VanHovePlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "VanHovePlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel = _select(system, selection)
    plan = VanHovePlan(
        sel,
        r_bin=r_bin,
        r_max=r_max,
        length_scale=length_scale,
        max_lag=max_lag,
        sqrt_time_bin=sqrt_time_bin,
        scale_to_average_box=scale_to_average_box,
        remove_pbc_jumps=remove_pbc_jumps,
        time_scale=time_scale,
    )
    return plan.run(
        traj,
        system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
        integral_radius=integral_radius,
        curve_lags=None if curve_lags is None else [int(v) for v in curve_lags],
        curve_step=None if curve_step is None else int(curve_step),
    )


__all__ = ["vanhove"]
