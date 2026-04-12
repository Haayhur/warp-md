from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md


MaskLike = Union[str, Sequence[int], np.ndarray]

_CurrentPlan = (
    getattr(warp_md.traj_py, "PyCurrentPlan", None)
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
    raise RuntimeError("system.select_indices is required for non-string current selections")


def _resolve_charges(system, charges: Optional[Sequence[float]]) -> list[float]:
    if charges is not None:
        values = np.asarray(charges, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("charges must contain only finite values")
        return values.tolist()
    atoms = system.atom_table() if hasattr(system, "atom_table") else {}
    values = atoms.get("charge") if isinstance(atoms, dict) else None
    if values is None or len(values) == 0:
        raise RuntimeError(
            "current requires per-atom charges. Pass `charges=` or provide `atom_table()['charge']`."
        )
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("system atom_table()['charge'] must contain only finite values")
    return arr.tolist()


def current(
    traj,
    system,
    selection: Optional[MaskLike] = "",
    charges: Optional[Sequence[float]] = None,
    temperature: float = 300.0,
    group_by: str = "resid",
    length_scale: Optional[float] = None,
    group_types: Optional[Sequence[int]] = None,
    make_whole: bool = True,
    frame_decimation: Optional[tuple[int, int]] = None,
    dt_decimation: Optional[tuple[int, int, int, int]] = None,
    time_binning: Optional[tuple[float, float]] = None,
    lag_mode: Optional[str] = None,
    max_lag: Optional[int] = None,
    memory_budget_bytes: Optional[int] = None,
    multi_tau_m: Optional[int] = None,
    multi_tau_levels: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Current-lane transport and dielectric report via Rust plan path.

    Returns conductivity vs lag time plus per-frame rotational/translational
    dipole statistics. Direct velocity-current ACF outputs from `gmx current`
    (`-caf`/`-mc`) are not exposed yet.
    """
    if _CurrentPlan is None:
        raise RuntimeError(
            "PyCurrentPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    sel = _select(system, selection)
    charge_list = _resolve_charges(system, charges)
    plan = _CurrentPlan(
        sel,
        charge_list,
        temperature=temperature,
        group_by=group_by,
        length_scale=length_scale,
        group_types=None if group_types is None else [int(v) for v in group_types],
        make_whole=make_whole,
        frame_decimation=frame_decimation,
        dt_decimation=dt_decimation,
        time_binning=time_binning,
        lag_mode=lag_mode,
        max_lag=max_lag,
        memory_budget_bytes=memory_budget_bytes,
        multi_tau_m=multi_tau_m,
        multi_tau_levels=multi_tau_levels,
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
            "current requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "conductivity_time": np.asarray(out["conductivity_time"], dtype=np.float32),
        "conductivity": np.asarray(out["conductivity"], dtype=np.float32),
        "time": np.asarray(out["time"], dtype=np.float32),
        "md_sq": np.asarray(out["md_sq"], dtype=np.float32),
        "mj_sq": np.asarray(out["mj_sq"], dtype=np.float32),
        "md_mj": np.asarray(out["md_mj"], dtype=np.float32),
        "dielectric_rot": float(out["dielectric_rot"]),
        "dielectric_total": float(out["dielectric_total"]),
        "mu_avg": float(out["mu_avg"]),
        "conductivity_static": (
            None if out["conductivity_static"] is None else float(out["conductivity_static"])
        ),
    }


__all__ = ["current"]
