# Usage:
# from warp_md.analysis.pairdist import pairdist
# out = pairdist(traj, system, mask='@CA', delta=0.1)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_selection,
)


def _format_pairdist(centers, hist, std, counts, n_frames: int, dtype: str):
    out = {
        "bin_centers": np.asarray(centers, dtype=np.float32),
        "hist": np.asarray(hist, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "counts": np.asarray(counts, dtype=np.uint64),
        "n_frames": int(n_frames),
    }
    key = str(dtype).lower()
    if key == "dict":
        return out
    if key in ("tuple", "ndarray"):
        return out["bin_centers"], out["hist"]
    return out


def _format_extrema(values, mode: str, dtype: str):
    out = np.asarray(values, dtype=np.float32)
    key = str(dtype).lower()
    if key == "dict":
        return {"pairdist": out, "mode": mode, "n_frames": int(out.shape[0])}
    return out


def _pairdist_extrema(
    traj,
    system,
    mask: str,
    mask2: str,
    mode: str,
    maxdist: Optional[float],
    frame_indices: Optional[Sequence[int]],
    dtype: str,
    chunk_frames: Optional[int],
    image: bool,
    device: str,
):
    same = not bool(mask2)
    if not is_native_traj(traj):
        raise RuntimeError(
            f"{mode}dist requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError(f"failed to prepare native {mode}dist system")
    maxdist_value = None
    if maxdist is not None:
        maxdist_value = float(maxdist)
        if not np.isfinite(maxdist_value) or maxdist_value <= 0.0:
            raise ValueError("maxdist must be finite and positive")
    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    sel_a = native_selection(system, native_system, mask, allow_at_indices=True)
    sel_b = (
        sel_a
        if same
        else native_selection(system, native_system, mask2, allow_at_indices=True)
    )
    plan_cls = load_native_symbol("PairDistanceExtremaPlan")
    if plan_cls is None:
        raise RuntimeError("PairDistanceExtremaPlan binding unavailable")
    plan_mode = "min" if str(mode).lower() in ("min", "minimum") else "max"
    pbc = "orthorhombic" if image else "none"
    try:
        values = plan_cls(
            sel_a,
            sel_b,
            plan_mode,
            pbc,
            unique_pairs=bool(same),
            cutoff=maxdist_value,
            empty_value=maxdist_value,
        ).run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices_list,
        )
    except Exception as exc:
        raise RuntimeError(f"native {plan_mode}dist execution failed") from exc
    return _format_extrema(values, plan_mode, dtype)


def pairdist(
    traj,
    system,
    mask: str = "*",
    mask2: str = "",
    delta: float = 0.1,
    maxdist: Optional[float] = None,
    mode: str = "hist",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "dict",
    chunk_frames: Optional[int] = None,
    image: bool = False,
    device: str = "auto",
):
    """Compute pair distance histogram or per-frame extrema."""
    mode_value = str(mode).lower()
    histogram_mode = mode_value in ("hist", "histogram", "distribution")
    extrema_mode = mode_value in ("min", "minimum", "max", "maximum")
    if not histogram_mode and not extrema_mode:
        raise ValueError("mode must be 'hist', 'min', or 'max'")
    if histogram_mode and delta <= 0.0:
        raise ValueError("delta must be positive")
    maxdist_value = None
    if maxdist is not None:
        maxdist_value = float(maxdist)
        if not np.isfinite(maxdist_value) or maxdist_value <= 0.0:
            raise ValueError("maxdist must be finite and positive")

    same = not bool(mask2)

    if not is_native_traj(traj):
        raise RuntimeError(
            "pairdist requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native pairdist system")
    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    sel_a = native_selection(system, native_system, mask, allow_at_indices=True)
    sel_b = (
        sel_a
        if same
        else native_selection(system, native_system, mask2, allow_at_indices=True)
    )
    pbc = "orthorhombic" if image else "none"

    if extrema_mode:
        return _pairdist_extrema(
            traj,
            system,
            mask,
            mask2,
            mode_value,
            maxdist,
            frame_indices,
            dtype,
            chunk_frames,
            image,
            device,
        )

    if maxdist_value is not None:
        plan_cls = load_native_symbol("PairDistPlan")
        if plan_cls is None:
            raise RuntimeError("PairDistPlan binding unavailable")
        n_bins = max(1, int(np.ceil(maxdist_value / float(delta))))
        r_max = np.float32(n_bins * float(delta))
        try:
            centers, hist, std, counts, n_frames = plan_cls(
                sel_a,
                sel_b,
                n_bins,
                r_max,
                pbc,
                output_distribution=True,
                unique_pairs=bool(same),
                compact_output=True,
            ).run(
                traj,
                native_system,
                chunk_frames=chunk_frames,
                device=device,
                frame_indices=frame_indices_list,
            )
        except Exception as exc:
            raise RuntimeError("native PairDistPlan execution failed") from exc
        return _format_pairdist(centers, hist, std, counts, n_frames, dtype)

    plan_cls = load_native_symbol("PairDistDynamicPlan")
    if plan_cls is None:
        raise RuntimeError("PairDistDynamicPlan binding unavailable")
    try:
        centers, hist, std, counts, n_frames = plan_cls(
            sel_a,
            sel_b,
            np.float32(delta),
            pbc,
            output_distribution=True,
            unique_pairs=bool(same),
            compact_output=True,
        ).run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices_list,
        )
    except Exception as exc:
        raise RuntimeError("native PairDistDynamicPlan execution failed") from exc

    return _format_pairdist(centers, hist, std, counts, n_frames, dtype)


def mindist(
    traj,
    system,
    mask: str = "*",
    mask2: str = "",
    *,
    maxdist: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
    image: bool = False,
    device: str = "auto",
):
    """Compute the per-frame minimum distance between two selections in Rust."""
    return _pairdist_extrema(
        traj,
        system,
        mask,
        mask2,
        "min",
        maxdist,
        frame_indices,
        dtype,
        chunk_frames,
        image,
        device,
    )


def maxdist(
    traj,
    system,
    mask: str = "*",
    mask2: str = "",
    *,
    maxdist: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
    image: bool = False,
    device: str = "auto",
):
    """Compute the per-frame maximum distance between two selections in Rust."""
    return _pairdist_extrema(
        traj,
        system,
        mask,
        mask2,
        "max",
        maxdist,
        frame_indices,
        dtype,
        chunk_frames,
        image,
        device,
    )


__all__ = ["pairdist", "mindist", "maxdist"]
