# Usage:
# from warp_md.analysis.surf import surf, molsurf
# areas = surf(traj, system, mask=":1-10")

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


def _normalize_algorithm(value: str) -> str:
    mode = str(value).lower()
    if mode not in ("sasa", "bbox", "auto"):
        raise ValueError("algorithm must be 'sasa', 'bbox', or 'auto'")
    return mode


def _normalize_probe_radius(value: float) -> float:
    radius = float(value)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("probe_radius must be a finite value >= 0")
    return radius


def _normalize_n_sphere_points(value: int) -> int:
    points = int(value)
    if points <= 0:
        raise ValueError("n_sphere_points must be a positive integer")
    return points


def _normalize_radii(radii: Optional[Sequence[float]]) -> Optional[list[float]]:
    if radii is None:
        return None
    arr = np.asarray(radii, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("radii values must be finite and > 0")
    return arr.tolist()


def surf(
    traj,
    system,
    mask: MaskLike = "",
    algorithm: str = "sasa",
    probe_radius: float = 1.4,
    n_sphere_points: int = 64,
    radii: Optional[Sequence[float]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> np.ndarray:
    """Surface area estimate.

    algorithm:
      - "sasa": Shrake-Rupley style SASA approximation
      - "bbox": legacy bounding-box area
      - "auto": use plan default (`sasa`)
    """
    algorithm = _normalize_algorithm(algorithm)
    probe_radius = _normalize_probe_radius(probe_radius)
    n_sphere_points = _normalize_n_sphere_points(n_sphere_points)

    try:
        from warp_md import SurfPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "SurfPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(SurfPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "SurfPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    plan_algorithm = "sasa" if algorithm == "auto" else algorithm
    if isinstance(mask, str):
        sel = system.select(mask if mask not in ("", "*", "all", None) else _all_resid_mask(system))
    else:
        sel = system.select_indices(np.asarray(mask, dtype=np.int64).reshape(-1).tolist())
    radii_list = _normalize_radii(radii)
    plan = SurfPlan(
        sel,
        algorithm=plan_algorithm,
        probe_radius=probe_radius,
        n_sphere_points=n_sphere_points,
        radii=radii_list,
    )
    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    try:
        out = np.asarray(
            plan.run(
                traj,
                system,
                chunk_frames=chunk_frames,
                device=device,
                frame_indices=frame_indices_list,
            ),
            dtype=np.float32,
        )
    except TypeError as exc:
        raise RuntimeError(
            "surf requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return out


def molsurf(
    traj,
    system,
    mask: MaskLike = "",
    algorithm: str = "sasa",
    probe_radius: float = 0.0,
    n_sphere_points: int = 64,
    radii: Optional[Sequence[float]] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> np.ndarray:
    return surf(
        traj,
        system,
        mask=mask,
        algorithm=algorithm,
        probe_radius=probe_radius,
        n_sphere_points=n_sphere_points,
        radii=radii,
        frame_indices=frame_indices,
        chunk_frames=chunk_frames,
        device=device,
    )


__all__ = ["surf", "molsurf"]
