# Usage:
# from warp_md.analysis.rotdif import rotdif
# time, data = rotdif(traj, system, mask=":1-10", orientation=[1, 2], group_by="resid")

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

MaskLike = Union[str, Sequence[int], np.ndarray]


def _all_resid_mask(system) -> str:
    if not hasattr(system, "atom_table"):
        return "all"
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_from_mask(system, mask: MaskLike):
    if isinstance(mask, str):
        query = mask if mask not in ("", "*", "all", None) else _all_resid_mask(system)
        return system.select(query)
    return system.select_indices(np.asarray(mask, dtype=np.int64).reshape(-1).tolist())


def _fit_diffusion_rust(
    time: np.ndarray,
    data: np.ndarray,
    fit_component: str,
    fit_window: Optional[Tuple[float, float]],
):
    import warp_md

    fit_fn = getattr(getattr(warp_md, "traj_py", None), "rotdif_fit", None)
    if fit_fn is None:
        raise RuntimeError(
            "rotdif_fit binding unavailable. Rebuild bindings with `maturin develop`."
        )
    d_rot, tau, slope, intercept, n_fit = fit_fn(
        np.asarray(time, dtype=np.float64),
        np.asarray(data, dtype=np.float64),
        fit_component=fit_component,
        fit_window=fit_window,
    )
    return {
        "d_rot": float(d_rot),
        "tau": float(tau),
        "slope": float(slope),
        "intercept": float(intercept),
        "n_fit": int(n_fit),
    }


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be bool")


def _normalize_orientation(orientation: Sequence[int]) -> Sequence[int]:
    if not isinstance(orientation, (list, tuple)):
        raise ValueError("orientation must be a sequence with length 2 or 3")
    values = [int(v) for v in orientation]
    if len(values) not in (2, 3):
        raise ValueError("orientation must have length 2 or 3")
    return values


def _normalize_fit_component(value: str) -> str:
    mode = str(value).lower()
    if mode not in ("p1", "p2"):
        raise ValueError("fit_component must be 'p1' or 'p2'")
    return mode


def _normalize_fit_window(window: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if window is None:
        return None
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError("fit_window must be a 2-item tuple/list (start, stop)")
    start = float(window[0])
    stop = float(window[1])
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError("fit_window values must be finite")
    return (start, stop)


def rotdif(
    traj,
    system,
    mask: MaskLike = "",
    orientation: Optional[Sequence[int]] = None,
    group_by: str = "resid",
    p2_legendre: bool = True,
    length_scale: Optional[float] = None,
    frame_decimation: Optional[Tuple[int, int]] = None,
    dt_decimation: Optional[Tuple[int, int, int, int]] = None,
    time_binning: Optional[Tuple[float, float]] = None,
    group_types: Optional[Sequence[int]] = None,
    lag_mode: Optional[str] = None,
    max_lag: Optional[int] = None,
    memory_budget_bytes: Optional[int] = None,
    multi_tau_m: Optional[int] = None,
    multi_tau_levels: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
    dt: float = 1.0,
    return_fit: bool = False,
    fit_component: str = "p2",
    fit_window: Optional[Tuple[float, float]] = None,
):
    """Rotational autocorrelation using Rust RotAcfPlan path only."""
    del dt
    if orientation is None:
        raise ValueError("orientation indices required")
    orientation_values = _normalize_orientation(orientation)
    p2_legendre = _as_bool(p2_legendre, "p2_legendre")
    fit_component = _normalize_fit_component(fit_component)
    fit_window = _normalize_fit_window(fit_window)

    try:
        from warp_md import RotAcfPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "RotAcfPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(RotAcfPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "RotAcfPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel = _selection_from_mask(system, mask)
    plan = RotAcfPlan(
        sel,
        group_by=group_by,
        orientation=list(orientation_values),
        p2_legendre=p2_legendre,
        length_scale=length_scale,
        frame_decimation=frame_decimation,
        dt_decimation=dt_decimation,
        time_binning=time_binning,
        group_types=list(group_types) if group_types is not None else None,
        lag_mode=lag_mode,
        max_lag=max_lag,
        memory_budget_bytes=memory_budget_bytes,
        multi_tau_m=multi_tau_m,
        multi_tau_levels=multi_tau_levels,
    )
    try:
        time, data = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices,
        )
    except TypeError as exc:
        raise RuntimeError(
            "rotdif requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    time_arr = np.asarray(time, dtype=np.float32)
    data_arr = np.asarray(data, dtype=np.float32)
    if not return_fit:
        return time_arr, data_arr
    fit_stats = _fit_diffusion_rust(time_arr, data_arr, fit_component, fit_window)
    return {"time": time_arr, "data": data_arr, **fit_stats}


__all__ = ["rotdif"]
