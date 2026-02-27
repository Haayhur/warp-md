# Usage:
# from warp_md.analysis.lowestcurve import lowestcurve
# out = lowestcurve(data, points=10, step=0.2)

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_xy(data: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("data must be 2D array-like with shape (2, N) or (N, 2)")
    if arr.shape[0] == 2:
        return arr
    if arr.shape[1] == 2:
        return arr.T
    raise ValueError("data must be 2D array-like with shape (2, N) or (N, 2)")


def lowestcurve(data, points: int = 10, step: float = 0.2) -> np.ndarray:
    """Compute lowest curve for XY data.

    For each x-bin of width ``step`` starting at min(x), compute the mean of the
    lowest ``points`` y-values. Empty bins return y=0.0. Returns a (2, N) array
    with x-bin starts and lowest-curve y values.
    """
    if step <= 0.0:
        raise ValueError("step must be positive")
    if points <= 0:
        raise ValueError("points must be positive")

    xy = _as_xy(data)
    if xy.size == 0:
        return np.empty((2, 0), dtype=np.float32)

    x = xy[0]
    y = xy[1]
    min_x = float(np.min(x))
    max_x = float(np.max(x))
    bins = np.arange(min_x, max_x, step, dtype=np.float64)
    out_x = np.empty(bins.shape[0], dtype=np.float64)
    out_y = np.empty(bins.shape[0], dtype=np.float64)

    for i, b0 in enumerate(bins):
        b1 = b0 + step
        mask = (x >= b0) & (x < b1)
        out_x[i] = b0
        if not np.any(mask):
            out_y[i] = 0.0
            continue
        ys = np.sort(y[mask])
        ys = ys[:points] if ys.size > points else ys
        out_y[i] = float(np.mean(ys))

    return np.vstack([out_x, out_y]).astype(np.float32)


__all__ = ["lowestcurve"]
