from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import coerce_native_system, is_native_traj, load_native_symbol, native_selection


MaskLike = Union[str, Sequence[int], np.ndarray, None]


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _charge_arg(system, charges: Optional[Sequence[float]]):
    if charges is not None:
        return [float(value) for value in np.asarray(charges, dtype=np.float64).reshape(-1)]
    if not hasattr(system, "atom_table"):
        return None
    atoms = system.atom_table()
    values = atoms.get("charge", atoms.get("charges", None))
    if values is None:
        return None
    return [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]


def lineardensity(
    traj,
    system,
    selection: MaskLike = "",
    *,
    axis: str = "z",
    bin: float = 1.0,
    range: Optional[tuple[float, float]] = None,
    weight: str = "number",
    norm: str = "count",
    charges: Optional[Sequence[float]] = None,
    cross_section_area: Optional[float] = None,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "dict",
    device: str = "auto",
):
    """Compute a 1D density/profile along one axis with frame loops in Rust."""
    if not is_native_traj(traj):
        raise RuntimeError(
            "lineardensity requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for lineardensity")
    plan_cls = load_native_symbol("LinearDensityPlan")
    if plan_cls is None:
        raise RuntimeError(
            "LinearDensityPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    sel = native_selection(system, native_system, selection, allow_at_indices=True)
    charge_values = _charge_arg(system, charges)
    if str(weight).lower() == "charge" and charge_values is None:
        raise ValueError("charge-weighted lineardensity requires charges")

    plan = plan_cls(
        sel,
        axis=axis,
        bin=float(bin),
        range=range,
        weight=weight,
        norm=norm,
        charges=charge_values,
        cross_section_area=cross_section_area,
        length_scale=length_scale,
    )
    values = np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )
    if values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError("native LinearDensityPlan returned unexpected output")

    key = str(dtype).lower()
    if key in ("ndarray", "array", "matrix"):
        return values
    out = {
        "axis": values[:, 0].astype(np.float32, copy=False),
        "profile": values[:, 1].astype(np.float32, copy=False),
        "mean_weight": values[:, 2].astype(np.float32, copy=False),
        "axis_name": str(axis).lower(),
        "weight": str(weight).lower(),
        "norm": str(norm).lower(),
        "bin": float(bin),
        "range": None if range is None else (float(range[0]), float(range[1])),
    }
    if key == "dict":
        return out
    if key in ("profile", "density"):
        return out["profile"]
    if key in ("axis", "centers"):
        return out["axis"]
    if key in ("mean_weight", "count", "counts"):
        return out["mean_weight"]
    raise ValueError("dtype must be 'dict', 'ndarray', 'profile', 'axis', or 'mean_weight'")


linear_density = lineardensity


__all__ = ["lineardensity", "linear_density"]
