# Usage:
# from warp_md.analysis.multipucker import multipucker
# out = multipucker(traj, system, mask=":1-5", bins=4)

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


def _normalize_bins(value: int) -> int:
    bins = int(value)
    if bins <= 0:
        raise ValueError("bins must be a positive integer")
    return bins


def _normalize_mode(value: str) -> str:
    mode = str(value).lower()
    if mode not in ("histogram", "legacy"):
        raise ValueError("mode must be 'histogram' or 'legacy'")
    return mode


def _as_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be bool")


def _normalize_range_max(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        raise ValueError("range_max must be a finite value > 0 when provided")
    return out


def multipucker(
    traj,
    system,
    mask: MaskLike = "",
    bins: int = 4,
    mode: str = "histogram",
    range_max: Optional[float] = None,
    normalize: bool = True,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> np.ndarray:
    """Multi-bin puckering profile.

    mode:
      - "histogram": per-frame radial distribution around centroid
      - "legacy": legacy one-hot max-radius bin

    Rust-first behavior:
      - histogram supports optional `range_max`; if omitted, Rust auto-estimates.
    """
    bins = _normalize_bins(bins)
    mode = _normalize_mode(mode)
    normalize = _as_bool(normalize, "normalize")
    range_max = _normalize_range_max(range_max)
    try:
        from warp_md import MultiPuckerPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "MultiPuckerPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(MultiPuckerPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "MultiPuckerPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    if isinstance(mask, str):
        sel = system.select(mask if mask not in ("", "*", "all", None) else _all_resid_mask(system))
    else:
        sel = system.select_indices(np.asarray(mask, dtype=np.int64).reshape(-1).tolist())
    plan = MultiPuckerPlan(
        sel,
        bins,
        mode=mode,
        range_max=range_max,
        normalize=normalize,
    )
    try:
        out = np.asarray(
            plan.run(
                traj,
                system,
                chunk_frames=chunk_frames,
                device=device,
                frame_indices=frame_indices,
            ),
            dtype=np.float32,
        )
    except TypeError as exc:
        raise RuntimeError(
            "multipucker requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return out


__all__ = ["multipucker"]
