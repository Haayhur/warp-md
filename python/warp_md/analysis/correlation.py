# Usage:
# from warp_md.analysis.correlation import acorr, xcorr, timecorr, velocity_autocorrelation
# time, data = velocity_autocorrelation(traj, system, "name CA", lag_mode="ring", max_lag=100)

from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np

from .. import VelocityAutoCorrPlan


def _as_float_array(data: Iterable[float]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    raise ValueError("data must be 1D or 2D array-like")


def _corr_series(
    a: np.ndarray,
    b: np.ndarray,
    normalize: bool,
) -> np.ndarray:
    if a.shape != b.shape:
        raise ValueError("input shapes must match")
    n_frames, n_feat = a.shape
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)
    out = np.zeros(n_frames, dtype=np.float32)
    for lag in range(n_frames):
        prod = (a[lag:] * b[: n_frames - lag]).sum(axis=1)
        out[lag] = prod.mean() / max(n_feat, 1)
    if normalize:
        zero = out[0]
        if zero != 0.0:
            out = out / zero
    return out


def acorr(
    series: Iterable[float],
    normalize: bool = True,
    return_time: bool = False,
    dt: float = 1.0,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Autocorrelation of a scalar or vector series."""
    arr = _as_float_array(series)
    data = _corr_series(arr, arr, normalize)
    if return_time:
        time = np.arange(data.shape[0], dtype=np.float32) * float(dt)
        return time, data
    return data


def xcorr(
    series_a: Iterable[float],
    series_b: Iterable[float],
    normalize: bool = True,
    return_time: bool = False,
    dt: float = 1.0,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Cross-correlation of two scalar or vector series."""
    arr_a = _as_float_array(series_a)
    arr_b = _as_float_array(series_b)
    data = _corr_series(arr_a, arr_b, normalize)
    if return_time:
        time = np.arange(data.shape[0], dtype=np.float32) * float(dt)
        return time, data
    return data


def timecorr(
    vectors: np.ndarray,
    order: int = 1,
    normalize: bool = True,
    return_time: bool = False,
    dt: float = 1.0,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Time correlation of 3D vectors (order 1 or 2)."""
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim == 2:
        if arr.shape[1] != 3:
            raise ValueError("vectors must have shape (n_frames, 3) or (n_frames, n_items, 3)")
        arr = arr[:, None, :]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("vectors must have shape (n_frames, 3) or (n_frames, n_items, 3)")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    n_frames, n_items, _ = arr.shape
    if n_frames == 0:
        return np.empty(0, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=2, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    arr_norm = arr / norms
    out = np.zeros(n_frames, dtype=np.float32)
    for lag in range(n_frames):
        dot = (arr_norm[lag:] * arr_norm[: n_frames - lag]).sum(axis=2)
        if order == 2:
            dot = 1.5 * dot * dot - 0.5
        out[lag] = dot.mean()
    if normalize:
        zero = out[0] if out.size > 0 else 1.0
        if zero != 0.0:
            out = out / zero
    if return_time:
        time = np.arange(out.shape[0], dtype=np.float32) * float(dt)
        return time, out
    return out


def velocity_autocorrelation(
    traj,
    system,
    selection: str,
    normalize: bool = False,
    lag_mode: Optional[str] = None,
    max_lag: Optional[int] = None,
    memory_budget_bytes: Optional[int] = None,
    multi_tau_m: Optional[int] = None,
    multi_tau_levels: Optional[int] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Velocity autocorrelation from a trajectory selection."""
    sel = system.select(selection)
    plan = VelocityAutoCorrPlan(
        sel,
        normalize=normalize,
        lag_mode=lag_mode,
        max_lag=max_lag,
        memory_budget_bytes=memory_budget_bytes,
        multi_tau_m=multi_tau_m,
        multi_tau_levels=multi_tau_levels,
    )
    return plan.run(traj, system, chunk_frames=chunk_frames, device=device)
