# Usage:
# from warp_md.analysis.watershell import watershell
# counts = watershell(traj, system, solute_mask='!:WAT', solvent_mask=':WAT', lower=3.4, upper=5.0)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._runtime import (
    load_native_symbol,
    native_inputs,
    native_selection,
    reset_traj,
)


def _native_watershell(
    traj,
    system,
    solute_mask: str,
    solvent_mask: str,
    lower: float,
    upper: float,
    image: bool,
    frame_indices: Optional[Sequence[int]],
    chunk_frames: Optional[int],
):
    frame_indices_arg = None if frame_indices is None else [int(value) for value in frame_indices]

    def _run_counts(pbc: str) -> np.ndarray:
        upper_plan = plan_cls(solute, solvent, float(upper), pbc=pbc)
        upper_counts = np.asarray(
            upper_plan.run(
                native_traj,
                native_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=frame_indices_arg,
            ),
            dtype=np.float32,
        )
        if lower <= 0.0:
            return upper_counts
        if not reset_traj(native_traj):
            raise RuntimeError("failed to reset native trajectory for lower-shell pass")
        lower_cutoff = float(np.nextafter(lower, -np.inf))
        lower_plan = plan_cls(solute, solvent, lower_cutoff, pbc=pbc)
        lower_counts = np.asarray(
            lower_plan.run(
                native_traj,
                native_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=frame_indices_arg,
            ),
            dtype=np.float32,
        )
        return upper_counts - lower_counts

    plan_cls = load_native_symbol("SearchNeighborsPlan")
    if plan_cls is None:
        raise RuntimeError("SearchNeighborsPlan binding unavailable")
    try:
        native_traj, native_system = native_inputs(
            traj,
            system,
            chunk_frames,
            include_box=image,
        )
        if native_traj is None or native_system is None:
            raise RuntimeError("failed to prepare native watershell inputs")
        solute = native_selection(system, native_system, solute_mask)
        solvent = native_selection(system, native_system, solvent_mask)
        try:
            counts = _run_counts("orthorhombic" if image else "none")
        except Exception as exc:
            if not image or "orthorhombic box required" not in str(exc):
                raise
            if not reset_traj(native_traj):
                raise RuntimeError("failed to reset native trajectory for non-PBC watershell retry") from exc
            counts = _run_counts("none")
        return counts.astype(np.float32, copy=False)
    except Exception as exc:
        raise RuntimeError("native watershell execution failed") from exc


def watershell(
    traj,
    system,
    solute_mask: str,
    solvent_mask: str = ":WAT",
    lower: float = 3.4,
    upper: float = 5.0,
    image: bool = True,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Count solvent atoms in a distance shell around solute atoms."""
    if not solute_mask:
        raise ValueError("solute_mask is required")
    if lower < 0.0 or upper <= 0.0 or upper <= lower:
        raise ValueError("upper must be greater than lower and both positive")

    return _native_watershell(
        traj,
        system,
        solute_mask,
        solvent_mask,
        lower,
        upper,
        image,
        frame_indices,
        chunk_frames,
    )


__all__ = ["watershell"]
