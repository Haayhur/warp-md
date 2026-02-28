# Usage:
# from warp_md.analysis.ti import ti
# out = ti("dvdl.dat", x_col=0, y_col=1)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def ti(
    filename: Optional[str] = None,
    x_col: Optional[int] = 0,
    y_col: int = 1,
    skiprows: int = 0,
    comment: str = "#",
    delimiter: Optional[str] = None,
    data: Optional[Sequence[Sequence[float]]] = None,
    method: str = "trapz",
):
    """Compute thermodynamic integration by trapezoidal rule.

    Parameters
    ----------
    filename : str, optional
        Data file with columns (lambda, dV/dlambda, ...).
    data : array-like, optional
        Raw data array (n, m). If provided, filename is ignored.
    x_col : int or None
        Column for lambda. If None, use row index.
    y_col : int
        Column for dV/dlambda.
    method : {'trapz', 'simpson'}
        Integration method.
    """
    if data is None:
        if not filename:
            raise ValueError("filename or data must be provided")
        arr = np.loadtxt(filename, comments=comment, skiprows=skiprows, delimiter=delimiter)
    else:
        arr = np.asarray(data, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if y_col >= arr.shape[1]:
        raise ValueError("y_col is out of range")
    y = arr[:, y_col]
    if x_col is None:
        x = np.arange(len(y), dtype=np.float64)
    else:
        if x_col >= arr.shape[1]:
            raise ValueError("x_col is out of range")
        x = arr[:, x_col]
    method = method.lower()
    if method == "trapz":
        if hasattr(np, "trapezoid"):
            integral = float(np.trapezoid(y, x))
        else:
            integral = float(np.trapz(y, x))
    elif method == "simpson":
        if len(y) < 3 or len(y) % 2 == 0:
            raise ValueError("simpson requires odd number of points >= 3")
        h = (x[-1] - x[0]) / (len(x) - 1)
        integral = float((h / 3.0) * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-2:2].sum()))
    else:
        raise ValueError("method must be 'trapz' or 'simpson'")
    mean = float(np.mean(y)) if len(y) else 0.0
    std = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
    stderr = float(std / np.sqrt(len(y))) if len(y) > 1 else 0.0
    return {
        "lambda": x.astype(np.float64),
        "dvdl": y.astype(np.float64),
        "integral": integral,
        "mean": mean,
        "std": std,
        "stderr": stderr,
    }


__all__ = ["ti"]
