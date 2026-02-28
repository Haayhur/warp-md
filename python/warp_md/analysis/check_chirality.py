# Usage:
# from warp_md.analysis.check_chirality import check_chirality
# out = check_chirality(traj, system, groups=[(":1@C", ":1@N", ":1@CA", ":1@CB")])

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np

MaskLike = Union[str, Sequence[int], np.ndarray, int]
GroupLike = Sequence[MaskLike]


def _indices_from_item(item: MaskLike) -> np.ndarray:
    if isinstance(item, np.ndarray):
        return np.asarray(item, dtype=np.int64).reshape(-1)
    if isinstance(item, (list, tuple)):
        return np.asarray([int(x) for x in item], dtype=np.int64)
    return np.asarray([int(item)], dtype=np.int64)


def _selection_from_item(system, item: MaskLike):
    if isinstance(item, str):
        return system.select(item)
    indices = _indices_from_item(item).tolist()
    return system.select_indices(indices)


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be bool")


def _normalize_planar_tolerance(value: float) -> float:
    tol = float(value)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("planar_tolerance must be a finite value >= 0")
    return tol


def check_chirality(
    traj,
    system,
    groups: Iterable[GroupLike],
    mass_weighted: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
    planar_tolerance: float = 1e-8,
    return_labels: bool = False,
):
    """Signed volume for chirality groups (A,B,C,D)."""
    mass_weighted = _as_bool(mass_weighted, "mass_weighted")
    return_labels = _as_bool(return_labels, "return_labels")
    planar_tolerance = _normalize_planar_tolerance(planar_tolerance)

    try:
        from warp_md import CheckChiralityPlan  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "CheckChiralityPlan binding unavailable. Rebuild bindings with `maturin develop`."
        ) from exc
    if getattr(CheckChiralityPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "CheckChiralityPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    group_list = list(groups)
    sel_groups = []
    for group in group_list:
        if len(group) != 4:
            raise ValueError("each chirality group must have 4 selections")
        sel_groups.append(
            (
                _selection_from_item(system, group[0]),
                _selection_from_item(system, group[1]),
                _selection_from_item(system, group[2]),
                _selection_from_item(system, group[3]),
            )
        )
    plan = CheckChiralityPlan(sel_groups, mass_weighted=mass_weighted)
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
            "check_chirality requires Rust-backed trajectory/system objects."
        ) from exc
    results = np.asarray(out, dtype=np.float32)
    if not return_labels:
        return results
    labels = np.zeros(results.shape, dtype=np.int8)
    labels[results > planar_tolerance] = 1
    labels[results < -planar_tolerance] = -1
    return results, labels


__all__ = ["check_chirality"]
