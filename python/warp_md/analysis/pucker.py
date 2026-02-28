# Usage:
# from warp_md.analysis.pucker import pucker
# radii = pucker(traj, system, mask=":1-5")

from __future__ import annotations

from typing import Optional, Sequence, Union

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


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be bool")


def _normalize_metric(metric: str) -> str:
    mode = str(metric).lower()
    if mode not in ("amplitude", "max_radius"):
        raise ValueError("metric must be 'amplitude' or 'max_radius'")
    return mode


def pucker(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
    metric: str = "amplitude",
    return_phase: bool = False,
):
    """Ring puckering metric using Rust plan path only."""
    metric = _normalize_metric(metric)
    return_phase = _as_bool(return_phase, "return_phase")

    try:
        from warp_md import PuckerPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "PuckerPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(PuckerPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "PuckerPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel = _selection_from_mask(system, mask)
    if len(sel.indices) == 0:
        if return_phase:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.empty((0,), dtype=np.float32)

    try:
        plan = PuckerPlan(sel, metric=metric, return_phase=return_phase)
    except TypeError as exc:
        raise RuntimeError(
            "pucker requires updated Rust bindings with (metric, return_phase) support. Rebuild with `maturin develop`."
        ) from exc
    try:
        out = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices,
        )
    except TypeError as exc:
        raise RuntimeError(
            "pucker requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    if return_phase:
        try:
            values, phase = out
        except Exception as exc:
            raise RuntimeError(
                "pucker return_phase requires updated Rust bindings. Rebuild with `maturin develop`."
            ) from exc
        values_arr = np.asarray(values, dtype=np.float32)
        phase_arr = np.asarray(phase, dtype=np.float32)
        return values_arr, phase_arr

    out_arr = np.asarray(out, dtype=np.float32)
    return out_arr


__all__ = ["pucker"]
