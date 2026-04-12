# Usage:
# from warp_md.analysis.matrix import covar, correl, dist, mwcovar
# mat = covar(traj, system, mask="name CA")

from __future__ import annotations

from typing import Optional

import numpy as np

from ._runtime import load_native_symbol, native_inputs, native_selection


def _run_native_matrix(traj, system, mask: str, mode: str, chunk_frames: Optional[int]) -> np.ndarray:
    plan_cls = load_native_symbol("MatrixPlan")
    if plan_cls is None:
        raise RuntimeError("MatrixPlan binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native matrix inputs")
    try:
        plan = plan_cls(native_selection(system, native_system, mask), mode, "none")
        return np.asarray(
            plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(f"native MatrixPlan({mode}) execution failed") from exc


def covar(traj, system, mask: str = "", chunk_frames: Optional[int] = None) -> np.ndarray:
    return _run_native_matrix(traj, system, mask, "covar", chunk_frames)


def mwcovar(traj, system, mask: str = "", chunk_frames: Optional[int] = None) -> np.ndarray:
    return _run_native_matrix(traj, system, mask, "mwcovar", chunk_frames)


def dist(traj, system, mask: str = "", chunk_frames: Optional[int] = None) -> np.ndarray:
    return _run_native_matrix(traj, system, mask, "dist", chunk_frames)


def correl(traj, system, mask: str = "", chunk_frames: Optional[int] = None) -> np.ndarray:
    return _run_native_matrix(traj, system, mask, "correl", chunk_frames)


__all__ = ["covar", "mwcovar", "dist", "correl"]
