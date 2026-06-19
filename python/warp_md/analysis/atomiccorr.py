# Usage:
# from warp_md.analysis.atomiccorr import atomiccorr
# time, data = atomiccorr(traj, system, mask=":1-10", lag_mode="fft", device="cuda")

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._runtime import (
    load_native_symbol,
    native_inputs,
    native_selection,
)
from ._stream import normalize_frame_indices


MaskLike = Union[str, Sequence[int], np.ndarray]


def atomiccorr(
    traj,
    system,
    mask: MaskLike = "",
    reference: str = "frame0",
    lag_mode: Optional[str] = None,
    max_lag: Optional[int] = None,
    memory_budget_bytes: Optional[int] = None,
    multi_tau_m: Optional[int] = None,
    multi_tau_levels: Optional[int] = None,
    normalize: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Atomic displacement autocorrelation."""
    plan_cls = load_native_symbol("AtomicCorrelationPlan")
    if plan_cls is None:
        raise RuntimeError("AtomicCorrelationPlan binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native atomiccorr inputs")
    frame_indices_arg = None
    selected_n_frames = None
    if frame_indices is not None:
        if not hasattr(native_traj, "count_frames"):
            raise RuntimeError("native trajectory frame counting unavailable")
        try:
            total_frames = int(native_traj.count_frames(chunk_frames))
        except Exception as exc:
            raise RuntimeError("failed to count native trajectory frames") from exc
        frame_indices_arg = [int(v) for v in normalize_frame_indices(frame_indices, total_frames)]
        selected_n_frames = len(frame_indices_arg)
        if selected_n_frames < 2:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
    if max_lag is None and hasattr(native_traj, "count_frames"):
        if selected_n_frames is None:
            try:
                n_frames = int(native_traj.count_frames(chunk_frames))
            except Exception:
                n_frames = 0
        else:
            n_frames = selected_n_frames
        if n_frames < 2:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)
        max_lag = n_frames - 1
    ref_mode = str(reference).strip().lower()
    if ref_mode in ("topology", "top", "topology0", "topology_ref"):
        try:
            positions0 = native_system.positions0()
        except Exception:
            positions0 = None
        if positions0 is None:
            reference = "frame0"
    try:
        selection = native_selection(system, native_system, mask)
        plan = plan_cls(
            selection,
            reference=reference,
            lag_mode="ring" if lag_mode is None else lag_mode,
            max_lag=max_lag,
            memory_budget_bytes=memory_budget_bytes,
            multi_tau_m=multi_tau_m,
            multi_tau_levels=multi_tau_levels,
        )
        time, data = plan.run(
            native_traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices_arg,
        )
        time = np.asarray(time, dtype=np.float32).reshape(-1)
        data = np.asarray(data, dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise RuntimeError("native AtomicCorrelationPlan execution failed") from exc
    if normalize and data.size > 0 and data[0] != 0.0:
        data = data / data[0]
    return time, data


__all__ = ["atomiccorr"]
