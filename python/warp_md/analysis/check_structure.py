# Usage:
# from warp_md.analysis.check_structure import check_structure
# counts, report = check_structure(traj, system)

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_selection,
    native_system_from_atom_count,
)


_MASK_SENTINELS = ("", "*", "all", None)


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _native_system_for_check(traj, system):
    native_system = coerce_native_system(system)
    if native_system is not None:
        return native_system
    if not hasattr(traj, "n_atoms"):
        return None
    try:
        n_atoms = int(traj.n_atoms())
    except Exception:
        return None
    return native_system_from_atom_count(n_atoms)


def _selection_for_check(system, native_system, mask):
    if system is None:
        if mask not in _MASK_SENTINELS:
            raise ValueError("mask requires a system")
        return native_system.select_indices(list(range(int(native_system.n_atoms()))))
    return native_selection(system, native_system, mask, allow_at_indices=True)


def check_structure(
    traj,
    system,
    mask: str = "",
    options: str = "",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "ndarray",
) -> Tuple[np.ndarray, str]:
    """Basic structure checks (NaN/inf)."""
    del options
    if not is_native_traj(traj):
        raise RuntimeError(
            "check_structure requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    plan_cls = load_native_symbol("CheckStructurePlan")
    if plan_cls is None:
        raise RuntimeError(
            "CheckStructurePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    native_system = _native_system_for_check(traj, system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for check_structure")
    selection = _selection_for_check(system, native_system, mask)
    try:
        plan = plan_cls(selection)
        values = plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        )
    except TypeError as exc:
        raise RuntimeError(
            "check_structure requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    counts = np.asarray(values, dtype=np.float32).round().astype(np.int64, copy=False)
    report_lines = [
        f"frame {idx}: {int(count)} atoms with invalid coords"
        for idx, count in enumerate(counts)
        if int(count) > 0
    ]
    report = "\n".join(report_lines)

    if dtype == "ndarray":
        return counts.astype(np.int64), report
    return counts.astype(np.int64), report


__all__ = ["check_structure"]
