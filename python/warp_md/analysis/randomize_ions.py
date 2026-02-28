# Usage:
# from warp_md.analysis.randomize_ions import randomize_ions
# traj_out = randomize_ions(traj, system, mask='@Na+', around=':1-16', by=5.0, overlap=3.0, seed=1)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._stream import infer_n_atoms
from .trajectory import ArrayTrajectory

def _resolve_randomize_plan():
    try:
        from warp_md import RandomizeIonsPlan  # type: ignore
    except Exception:
        return None
    if getattr(RandomizeIonsPlan, "__name__", "") == "_Missing":
        return None
    return RandomizeIonsPlan


def randomize_ions(
    traj,
    system,
    mask: str,
    around: Optional[str] = None,
    by: float = 0.0,
    overlap: float = 0.0,
    seed: int = 1,
    noimage: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Randomize ions by swapping with solvent residues."""
    if not mask:
        raise ValueError("mask is required")
    sel = system.select(mask)
    around_sel = system.select(around) if around else None

    plan_cls = _resolve_randomize_plan()
    if plan_cls is None:
        raise RuntimeError(
            "RandomizeIonsPlan is unavailable. Build/install warp-md bindings "
            "(e.g. `maturin develop`) before calling randomize_ions."
        )

    plan = plan_cls(sel, seed, around_sel, float(by), float(overlap), bool(noimage))
    flat = plan.run(
        traj,
        system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=frame_indices,
    )
    arr = np.asarray(flat, dtype=np.float32)
    n_atoms = infer_n_atoms(system, traj=traj)
    if n_atoms <= 0:
        return ArrayTrajectory(np.empty((0, 0, 3), dtype=np.float32))
    if arr.size == 0:
        return ArrayTrajectory(np.empty((0, n_atoms, 3), dtype=np.float32))
    if arr.size % (n_atoms * 3) != 0:
        raise ValueError("randomize_ions plan output shape is inconsistent with atom count")

    out_coords = arr.reshape((-1, n_atoms, 3))
    return ArrayTrajectory(out_coords.astype(np.float32, copy=False))


__all__ = ["randomize_ions"]
