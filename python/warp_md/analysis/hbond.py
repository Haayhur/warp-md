from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import coerce_native_system, is_native_traj, load_native_symbol, native_selection


MaskLike = Union[str, Sequence[int], np.ndarray, None]


def hbond(
    traj,
    system,
    donors: MaskLike,
    acceptors: MaskLike,
    dist_cutoff: float,
    *,
    hydrogens: MaskLike = None,
    angle_cutoff: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "dict",
    device: str = "auto",
):
    """Count hydrogen bonds per frame with Rust-backed distance/angle loops."""
    if not is_native_traj(traj):
        raise RuntimeError("hbond requires a Rust-backed trajectory so frame/atom loops stay in Rust.")
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for hbond")
    cutoff = float(dist_cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("dist_cutoff must be finite and positive")
    if hydrogens is not None:
        if angle_cutoff is None:
            raise ValueError("angle_cutoff is required when hydrogens are provided")
        angle = float(angle_cutoff)
        if not np.isfinite(angle) or angle <= 0.0 or angle > 180.0:
            raise ValueError("angle_cutoff must be finite and in (0, 180]")
    else:
        angle = None

    plan_cls = load_native_symbol("HbondPlan")
    if plan_cls is None:
        raise RuntimeError("HbondPlan binding unavailable. Rebuild bindings with `maturin develop`.")
    donor_sel = native_selection(system, native_system, donors, allow_at_indices=True)
    acceptor_sel = native_selection(system, native_system, acceptors, allow_at_indices=True)
    hydrogen_sel = (
        None
        if hydrogens is None
        else native_selection(system, native_system, hydrogens, allow_at_indices=True)
    )
    plan = plan_cls(
        donor_sel,
        acceptor_sel,
        cutoff,
        hydrogens=hydrogen_sel,
        angle_cutoff=angle,
    )
    run_kwargs = {"chunk_frames": chunk_frames, "device": device}
    if frame_indices is not None:
        run_kwargs["frame_indices"] = [int(value) for value in frame_indices]
    time, values = plan.run(traj, native_system, **run_kwargs)
    time = np.asarray(time, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 1:
        raise RuntimeError("native HbondPlan returned unexpected output")
    counts = values[:, 0].astype(np.float32, copy=False)

    key = str(dtype).lower()
    if key == "dict":
        return {
            "time": time,
            "count": counts,
            "donors": np.asarray(list(donor_sel.indices), dtype=np.int64),
            "acceptors": np.asarray(list(acceptor_sel.indices), dtype=np.int64),
            "hydrogens": None
            if hydrogen_sel is None
            else np.asarray(list(hydrogen_sel.indices), dtype=np.int64),
            "dist_cutoff": cutoff,
            "angle_cutoff": angle,
        }
    if key in ("ndarray", "array", "counts", "count"):
        return counts
    if key == "tuple":
        return time, counts
    raise ValueError("dtype must be 'dict', 'ndarray', or 'tuple'")


__all__ = ["hbond"]
