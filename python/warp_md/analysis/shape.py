from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .structure import _require_native_radgyr_inputs, _run_native_plan_on_live_traj


MaskLike = Union[str, Sequence[int], None]


def _shape_dict(values: np.ndarray) -> dict[str, np.ndarray]:
    if values.ndim != 2 or values.shape[1] != 7:
        raise RuntimeError("native ShapeDescriptorsPlan returned unexpected output")
    return {
        "rg": values[:, 0].astype(np.float32, copy=False),
        "principal_moments": values[:, 1:4].astype(np.float32, copy=False),
        "asphericity": values[:, 4].astype(np.float32, copy=False),
        "acylindricity": values[:, 5].astype(np.float32, copy=False),
        "relative_shape_anisotropy": values[:, 6].astype(np.float32, copy=False),
    }


def shape_descriptors(
    traj,
    system,
    mask: MaskLike = "",
    mass: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    length_scale: float = 1.0,
    dtype: str = "dict",
    device: str = "auto",
):
    """Compute gyration-tensor shape descriptors via the Rust plan path."""
    _require_native_radgyr_inputs(traj, "shape_descriptors", length_scale)
    values = _run_native_plan_on_live_traj(
        "ShapeDescriptorsPlan",
        traj,
        system,
        mask,
        chunk_frames,
        frame_indices,
        device=device,
        mass_weighted=mass,
    )
    out = _shape_dict(values)
    key = str(dtype).lower()
    if key == "dict":
        return out
    if key in ("ndarray", "array", "matrix"):
        return values.astype(np.float32, copy=False)
    if key in out:
        return out[key]
    raise ValueError(
        "dtype must be 'dict', 'ndarray', or one of: "
        "rg, principal_moments, asphericity, acylindricity, relative_shape_anisotropy"
    )


def principal_moments(*args, **kwargs):
    kwargs["dtype"] = "principal_moments"
    return shape_descriptors(*args, **kwargs)


def asphericity(*args, **kwargs):
    kwargs["dtype"] = "asphericity"
    return shape_descriptors(*args, **kwargs)


def acylindricity(*args, **kwargs):
    kwargs["dtype"] = "acylindricity"
    return shape_descriptors(*args, **kwargs)


def relative_shape_anisotropy(*args, **kwargs):
    kwargs["dtype"] = "relative_shape_anisotropy"
    return shape_descriptors(*args, **kwargs)


relative_shape_antisotropy = relative_shape_anisotropy


__all__ = [
    "shape_descriptors",
    "principal_moments",
    "asphericity",
    "acylindricity",
    "relative_shape_anisotropy",
    "relative_shape_antisotropy",
]
