# Usage:
# from warp_md.analysis.atomiccorr import atomiccorr
# time, data = atomiccorr(traj, system, mask=":1-10", lag_mode="fft", device="cuda")

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields


MaskLike = Union[str, Sequence[int], np.ndarray]


def _all_resid_mask(system) -> str:
    if not hasattr(system, "atom_table"):
        return "all"
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: MaskLike) -> np.ndarray:
    if isinstance(mask, np.ndarray):
        return np.asarray(mask, dtype=np.int64).reshape(-1)
    if isinstance(mask, (list, tuple)):
        return np.asarray([int(x) for x in mask], dtype=np.int64)
    if mask in ("", "*", "all", None):
        sel = system.select(_all_resid_mask(system))
    else:
        sel = system.select(mask)
    return np.asarray(list(sel.indices), dtype=np.int64)


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    chunk = read_chunk_fields(traj, max_chunk)
    if chunk is None:
        return None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk)
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    return coords


def _apply_frame_indices(coords, frame_indices: Optional[Sequence[int]]):
    if frame_indices is None:
        return coords
    n_frames = coords.shape[0]
    select = []
    for i in frame_indices:
        j = int(i)
        if j < 0:
            j = n_frames + j
        if 0 <= j < n_frames:
            select.append(j)
    return coords[select]


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
    use_plan = False
    try:
        from warp_md import AtomicCorrPlan  # type: ignore

        use_plan = hasattr(system, "select") and getattr(AtomicCorrPlan, "__name__", "") != "_Missing"
    except Exception:
        AtomicCorrPlan = None  # type: ignore

    if use_plan and AtomicCorrPlan is not None:
        try:
            if isinstance(mask, str):
                sel = system.select(mask if mask not in ("", "*", "all", None) else _all_resid_mask(system))
            else:
                indices = np.asarray(mask, dtype=np.int64).reshape(-1).tolist()
                sel = system.select_indices(indices)
            plan = AtomicCorrPlan(
                sel,
                reference=reference,
                lag_mode=lag_mode,
                max_lag=max_lag,
                memory_budget_bytes=memory_budget_bytes,
                multi_tau_m=multi_tau_m,
                multi_tau_levels=multi_tau_levels,
            )
            time, data = plan.run(traj, system, chunk_frames=chunk_frames, device=device)
            time = np.asarray(time, dtype=np.float32)
            data = np.asarray(data, dtype=np.float32)
            if normalize and data.size > 0 and data[0] != 0.0:
                data = data / data[0]
            return time, data
        except TypeError:
            pass

    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    coords = _apply_frame_indices(coords, frame_indices)
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    idx = _selection_indices(system, mask)
    if idx.size == 0 or coords.shape[0] < 2:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    ref_mode = reference.lower()
    if ref_mode in ("topology", "top", "topology0", "topology_ref"):
        ref = system.positions0()
        if ref is None:
            raise ValueError("reference=topology requested but system.positions0 is None")
        ref = np.asarray(ref, dtype=np.float64)
        ref_sel = ref[idx]
    else:
        ref_sel = coords[0, idx, :]

    disp = coords[:, idx, :] - ref_sel[None, :, :]
    n_frames = disp.shape[0]
    lag_max = max_lag if max_lag is not None else n_frames - 1
    lag_max = max(1, min(int(lag_max), n_frames - 1))
    lags = np.arange(1, lag_max + 1, dtype=np.float32)
    out = np.zeros(lags.shape[0], dtype=np.float32)
    n_sel = max(1, idx.size)
    for i, lag in enumerate(lags.astype(int)):
        prod = (disp[lag:] * disp[: n_frames - lag]).sum(axis=2)
        out[i] = prod.mean() / n_sel
    if normalize and out.size > 0 and out[0] != 0.0:
        out = out / out[0]
    return lags, out


__all__ = ["atomiccorr"]
