from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md

from .h2order import _select


MaskLike = Union[str, Sequence[int], np.ndarray]

_HydOrderPlan = (
    getattr(warp_md.traj_py, "PyHydOrderPlan", None)
    if getattr(warp_md, "traj_py", None)
    else None
)


def hydorder(
    traj,
    system,
    selection: MaskLike,
    axis: str = "z",
    bin: float = 1.0,
    tblock: int = 1,
    sgang1: Optional[float] = None,
    sgang2: Optional[float] = None,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Tetrahedral order parameter grid via Rust plan path.

    `selection` should normally be the water-oxygen or other central-atom set.
    When both `sgang1` and `sgang2` are given, Rust also extracts lower/upper
    interface surfaces from block-averaged angular-order grids.
    """
    if _HydOrderPlan is None:
        raise RuntimeError(
            "PyHydOrderPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    sel = _select(system, selection)
    plan = _HydOrderPlan(
        sel,
        axis=axis,
        bin=bin,
        tblock=int(tblock),
        sgang1=sgang1,
        sgang2=sgang2,
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
            "hydorder requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "sg_mean": float(out["sg_mean"]),
        "sk_mean": float(out["sk_mean"]),
        "sg_grid": np.asarray(out["sg_grid"], dtype=np.float32),
        "sk_grid": np.asarray(out["sk_grid"], dtype=np.float32),
        "counts": np.asarray(out["counts"], dtype=np.uint64),
        "x": np.asarray(out["x"], dtype=np.float32),
        "y": np.asarray(out["y"], dtype=np.float32),
        "z": np.asarray(out["z"], dtype=np.float32),
        "dims": tuple(int(v) for v in out["dims"]),
        "bounds": np.asarray(out["bounds"], dtype=np.float32),
        "bin_width": float(out["bin_width"]),
        "axis": str(out["axis"]),
        "plane_axes": tuple(str(v) for v in out["plane_axes"]),
        "n_frames": int(out["n_frames"]),
        "used_box": bool(out["used_box"]),
        "length_scale": float(out["length_scale"]),
        "interface_lower": np.asarray(out["interface_lower"], dtype=np.float32),
        "interface_upper": np.asarray(out["interface_upper"], dtype=np.float32),
        "interface_blocks": int(out["interface_blocks"]),
        "interface_threshold": (
            None if out["interface_threshold"] is None else float(out["interface_threshold"])
        ),
        "block_size": int(out["block_size"]),
    }


__all__ = ["hydorder"]
