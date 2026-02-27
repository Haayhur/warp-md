# Usage:
# from warp_md.analysis.clustering import cluster_trajectory
# out = cluster_trajectory(traj, system, mask="name CA", method="dbscan", eps=1.5)

from __future__ import annotations

import os
import ctypes
from pathlib import Path
from typing import Any, Optional, Sequence, Union

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


def _select(system, mask: MaskLike):
    if isinstance(mask, str):
        expr = mask if mask not in ("", "*", "all", None) else _all_resid_mask(system)
        return system.select(expr)
    indices = np.asarray(mask, dtype=np.int64).reshape(-1).tolist()
    if hasattr(system, "select_indices"):
        return system.select_indices(indices)
    raise RuntimeError("system does not support index-based selection (missing select_indices)")


def _normalize_method(value: str) -> str:
    method = str(value).strip().lower()
    if method not in ("dbscan", "kmeans"):
        raise ValueError("method must be 'dbscan' or 'kmeans'")
    return method


def _ensure_nvrtc_library_path() -> None:
    """Best-effort NVRTC runtime path setup for CUDA plan execution."""
    try:
        import nvidia.cuda_nvrtc as cuda_nvrtc  # type: ignore
    except Exception:
        return
    module_path = Path(getattr(cuda_nvrtc, "__file__", "")).resolve()
    lib_dir = module_path.parent / "lib"
    if not lib_dir.exists():
        return
    key = "LD_LIBRARY_PATH"
    current = os.environ.get(key, "")
    paths = [p for p in current.split(os.pathsep) if p]
    lib_str = str(lib_dir)
    if lib_str in paths:
        pass
    else:
        os.environ[key] = os.pathsep.join([lib_str, *paths]) if paths else lib_str

    # On Linux, updating LD_LIBRARY_PATH at runtime is often insufficient.
    # Preload NVRTC symbols into the process so cudarc can resolve them.
    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    candidates = []
    candidates.extend(sorted(lib_dir.glob("libnvrtc.so*")))
    candidates.extend(sorted(lib_dir.glob("libnvrtc-builtins.so*")))
    for path in candidates:
        try:
            ctypes.CDLL(str(path), mode=rtld_global)
        except Exception:
            continue


def cluster_trajectory(
    traj,
    system,
    mask: MaskLike = "",
    method: str = "dbscan",
    eps: float = 2.0,
    min_samples: int = 5,
    n_clusters: int = 8,
    max_iter: int = 100,
    tol: float = 1.0e-4,
    seed: int = 0,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
    memory_budget_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """Cluster trajectory frames using DBSCAN or KMeans on RMSD-aligned coordinates."""
    method = _normalize_method(method)
    _ensure_nvrtc_library_path()
    try:
        from warp_md import TrajectoryClusterPlan  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TrajectoryClusterPlan binding unavailable. Rebuild bindings with `maturin develop`."
        ) from exc
    if getattr(TrajectoryClusterPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "TrajectoryClusterPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    selection = _select(system, mask)
    plan = TrajectoryClusterPlan(
        selection,
        method=method,
        eps=float(eps),
        min_samples=int(min_samples),
        n_clusters=int(n_clusters),
        max_iter=int(max_iter),
        tol=float(tol),
        seed=int(seed),
        memory_budget_bytes=memory_budget_bytes,
    )
    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    try:
        out = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices_list,
        )
    except TypeError as exc:
        raise RuntimeError(
            "cluster_trajectory requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    if not isinstance(out, dict):
        raise RuntimeError("TrajectoryClusterPlan returned unexpected output type")

    return {
        "labels": np.asarray(out.get("labels", []), dtype=np.int32),
        "centroids": np.asarray(out.get("centroids", []), dtype=np.uint32),
        "sizes": np.asarray(out.get("sizes", []), dtype=np.uint32),
        "method": str(out.get("method", method)),
        "n_frames": int(out.get("n_frames", 0)),
    }


__all__ = ["cluster_trajectory"]
