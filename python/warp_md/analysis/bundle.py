from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md

from .h2order import _select


MaskLike = Union[str, Sequence[int], np.ndarray]

_BundlePlan = (
    getattr(warp_md.traj_py, "PyBundlePlan", None) if getattr(warp_md, "traj_py", None) else None
)


def bundle(
    traj,
    system,
    top_selection: MaskLike,
    bottom_selection: MaskLike,
    n_axes: int,
    kink_selection: Optional[MaskLike] = None,
    use_z_reference: bool = False,
    mass_weighted: bool = True,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Analyze bundle axes from top/bottom endpoint groups via Rust plan path."""
    if _BundlePlan is None:
        raise RuntimeError(
            "PyBundlePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    top = _select(system, top_selection)
    bottom = _select(system, bottom_selection)
    kink = None if kink_selection is None else _select(system, kink_selection)
    plan = _BundlePlan(
        top,
        bottom,
        int(n_axes),
        kink_selection=kink,
        use_z_reference=bool(use_z_reference),
        mass_weighted=bool(mass_weighted),
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
            "bundle requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    result = {
        "labels": tuple(str(v) for v in out["labels"]),
        "time": np.asarray(out["time"], dtype=np.float32),
        "reference_axis": np.asarray(out["reference_axis"], dtype=np.float32),
        "top": np.asarray(out["top"], dtype=np.float32),
        "bottom": np.asarray(out["bottom"], dtype=np.float32),
        "mid": np.asarray(out["mid"], dtype=np.float32),
        "direction": np.asarray(out["direction"], dtype=np.float32),
        "length": np.asarray(out["length"], dtype=np.float32),
        "distance": np.asarray(out["distance"], dtype=np.float32),
        "z_shift": np.asarray(out["z_shift"], dtype=np.float32),
        "tilt": np.asarray(out["tilt"], dtype=np.float32),
        "radial_tilt": np.asarray(out["radial_tilt"], dtype=np.float32),
        "lateral_tilt": np.asarray(out["lateral_tilt"], dtype=np.float32),
        "frames": int(out["frames"]),
        "axes": int(out["axes"]),
        "has_kink": bool(out["has_kink"]),
        "use_z_reference": bool(out["use_z_reference"]),
        "mass_weighted": bool(out["mass_weighted"]),
        "used_box": bool(out["used_box"]),
        "length_scale": float(out["length_scale"]),
    }
    if out["kink"] is None:
        result["kink"] = None
        result["kink_angle"] = None
        result["kink_radial"] = None
        result["kink_lateral"] = None
    else:
        result["kink"] = np.asarray(out["kink"], dtype=np.float32)
        result["kink_angle"] = np.asarray(out["kink_angle"], dtype=np.float32)
        result["kink_radial"] = np.asarray(out["kink_radial"], dtype=np.float32)
        result["kink_lateral"] = np.asarray(out["kink_lateral"], dtype=np.float32)
    return result


__all__ = ["bundle"]
