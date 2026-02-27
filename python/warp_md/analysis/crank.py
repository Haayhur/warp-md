from __future__ import annotations

import numpy as np

import warp_md


_crank_kernel = (
    getattr(warp_md.traj_py, "crank_delta", None) if getattr(warp_md, "traj_py", None) else None
)


def crank(data0, data1, mode: str = "distance"):
    """Compute crank-style reaction coordinate via Rust kernel."""
    if _crank_kernel is None:
        raise RuntimeError(
            "crank_delta binding unavailable. Rebuild bindings with `maturin develop`."
        )
    x = np.asarray(data0, dtype=np.float64)
    y = np.asarray(data1, dtype=np.float64)
    try:
        out = _crank_kernel(x, y, mode=mode)
    except TypeError as exc:
        raise RuntimeError(
            "crank requires Rust-backed numpy-compatible inputs."
        ) from exc
    return np.asarray(out, dtype=np.float64)
