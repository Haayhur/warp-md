# Usage:
# from warp_md.analysis.infraredspec import infraredspec
# freq, intensity = infraredspec(traj, system, "name CA", timestep_fs=1.0)

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ._chunk_io import read_chunk_fields


def infraredspec(
    traj,
    system,
    selection: str,
    timestep_fs: Optional[float] = 1.0,
    timestep_ps: Optional[float] = None,
    freq_unit: str = "cm-1",
    window: Optional[str] = "hann",
    lag_mode: Optional[str] = "fft",
    max_lag: Optional[int] = None,
    memory_budget_bytes: Optional[int] = None,
    multi_tau_m: Optional[int] = None,
    multi_tau_levels: Optional[int] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> Tuple[np.ndarray, np.ndarray]:
    """Infrared spectrum via FFT of velocity autocorrelation.

    Notes
    -----
    - If trajectory provides time (ps), timestep is inferred automatically.
    - Otherwise, `timestep_fs` or `timestep_ps` is required.
    - Parameters related to GPU lag modes are accepted for API compatibility but ignored.
    """
    del device, lag_mode, memory_budget_bytes, multi_tau_m, multi_tau_levels
    coords, times = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    idx = _selection_indices(system, selection)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")
    sel_coords = coords[:, idx, :]

    if times is None or times.size < 2:
        if timestep_ps is None:
            if timestep_fs is None:
                raise ValueError("timestep is required when trajectory has no time data")
            timestep_ps = float(timestep_fs) * 1e-3
        dt_ps = float(timestep_ps)
        dt_steps = np.full(sel_coords.shape[0] - 1, dt_ps, dtype=np.float64)
    else:
        dt_steps = np.diff(times.astype(np.float64))
        dt_steps = dt_steps[np.isfinite(dt_steps) & (dt_steps > 0.0)]
        if dt_steps.size == 0:
            if timestep_ps is None:
                if timestep_fs is None:
                    raise ValueError("timestep is required when trajectory has invalid time data")
                timestep_ps = float(timestep_fs) * 1e-3
            dt_ps = float(timestep_ps)
            dt_steps = np.full(sel_coords.shape[0] - 1, dt_ps, dtype=np.float64)
        else:
            dt_ps = float(np.median(dt_steps))

    if dt_ps <= 0.0:
        raise ValueError("timestep must be positive")

    n_frames = sel_coords.shape[0]
    if n_frames < 2:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    vel = np.zeros((n_frames - 1, sel_coords.shape[1], 3), dtype=np.float64)
    for i in range(n_frames - 1):
        dt = dt_steps[i] if i < dt_steps.size else dt_ps
        if not np.isfinite(dt) or dt <= 0.0:
            dt = dt_ps
        vel[i] = (sel_coords[i + 1] - sel_coords[i]) / dt

    n = vel.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    series = vel.reshape(n, -1)
    n_dof = series.shape[1]
    if n_dof == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    fft = np.fft.rfft(series, axis=0)
    power = (fft * np.conj(fft)).real
    acf = np.fft.irfft(power, n=n, axis=0)
    counts = np.arange(n, 0, -1, dtype=np.float64)
    corr = (acf.sum(axis=1) / counts / float(n_dof)).astype(np.float64)
    if max_lag is not None:
        max_lag = int(max_lag)
        if max_lag >= 0:
            corr = corr[: max_lag + 1]

    if window is not None and window.lower() != "none":
        if window.lower() == "hann":
            corr = corr * np.hanning(corr.size)
        else:
            raise ValueError("window must be None, 'none', or 'hann'")

    dt_s = float(dt_ps) * 1e-12
    spec = np.real(np.fft.rfft(corr)) * dt_s
    freq_hz = np.fft.rfftfreq(corr.size, dt_s)
    unit = freq_unit.lower()
    if unit in ("cm-1", "cm^-1", "wavenumber"):
        c_cm = 2.99792458e10
        freq = freq_hz / c_cm
    elif unit == "thz":
        freq = freq_hz / 1e12
    elif unit == "hz":
        freq = freq_hz
    else:
        raise ValueError("freq_unit must be 'cm-1', 'thz', or 'hz'")

    return freq.astype(np.float32), spec.astype(np.float32)


def _selection_indices(system, mask: str) -> np.ndarray:
    if mask in ("", "*", "all", None):
        atoms = system.atom_table()
        resids = atoms.get("resid", [])
        if not resids:
            sel = system.select("resid 0:0")
        else:
            sel = system.select(f"resid {min(resids)}:{max(resids)}")
    else:
        sel = system.select(mask)
    return np.asarray(list(sel.indices), dtype=np.int64)


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    time_list = []
    chunk = read_chunk_fields(traj, max_chunk, include_time=True)
    if chunk is None:
        return None, None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        time = chunk.get("time_ps")
        if time is None:
            time = chunk.get("time")
        if time is not None:
            time_list.append(np.asarray(time, dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk, include_time=True)
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    times = np.concatenate(time_list, axis=0) if time_list else None
    return coords, times


__all__ = ["infraredspec"]
