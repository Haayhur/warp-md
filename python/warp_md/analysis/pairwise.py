# Usage:
# from warp_md.analysis.pairwise import pairwise_rmsd
# mat = pairwise_rmsd(traj, system, mask="name CA", metric="srmsd")

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .rmsd import pairwise_rmsd as _pairwise_rmsd


def pairwise_rmsd(
    traj,
    system,
    mask: str = "",
    metric: str = "rms",
    mat_type: str = "full",
    pbc: str = "none",
    length_scale: float = 1.0,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Pairwise RMSD with srmsd alias (currently same as rms)."""
    metric = metric.lower()
    if metric == "srmsd":
        metric = "rms"
    return _pairwise_rmsd(
        traj,
        system,
        mask=mask,
        metric=metric,
        mat_type=mat_type,
        pbc=pbc,
        length_scale=length_scale,
        frame_indices=frame_indices,
        chunk_frames=chunk_frames,
    )


__all__ = ["pairwise_rmsd"]
