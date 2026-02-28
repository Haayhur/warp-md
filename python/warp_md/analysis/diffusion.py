# Usage:
# from warp_md.analysis.diffusion import diffusion
# out = diffusion(traj, system, mask=':WAT@O', tstep=1.0)

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields

def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _linear_fit(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def _traj_has_time_ps(traj) -> bool:
    if not hasattr(traj, "read_chunk"):
        return False
    has_time = False
    reset_ok = hasattr(traj, "reset")
    try:
        chunk = read_chunk_fields(traj, 1, include_time=True)
        if isinstance(chunk, dict):
            has_time = chunk.get("time_ps") is not None
    except Exception:
        has_time = False
    finally:
        if reset_ok:
            try:
                traj.reset()
            except Exception:
                pass
    return has_time


def diffusion(
    traj,
    system,
    mask: str = "",
    tstep: float = 1.0,
    individual: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
):
    """Compute diffusion from mean-squared displacement.

    Returns dict with time, MSD components, and diffusion estimate.
    """
    if tstep <= 0.0:
        raise ValueError("tstep must be positive")
    has_time_ps = _traj_has_time_ps(traj)

    try:
        from warp_md import MsdPlan  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "MsdPlan binding unavailable. Rebuild bindings with `maturin develop`."
        ) from exc
    if getattr(MsdPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "MsdPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    mask_expr = mask
    if mask in ("", "*", "all", None):
        mask_expr = _all_resid_mask(system) if hasattr(system, "atom_table") else "all"
    sel = system.select(mask_expr)
    sel_indices = np.asarray(list(sel.indices), dtype=np.int64)
    if sel_indices.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        out = {"time": empty, "X": empty, "Y": empty, "Z": empty, "MSD": empty, "D": 0.0}
        if individual:
            out["MSD_individual"] = np.empty((0, 0), dtype=np.float32)
        return out

    n_types = int(sel_indices.size) if individual else 1
    group_types = list(range(n_types)) if individual else None
    plan_kwargs = {"group_by": "atom", "lag_mode": "fft"}
    if group_types is not None:
        plan_kwargs["group_types"] = group_types
    plan = MsdPlan(sel, **plan_kwargs)

    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    try:
        time_arr, data = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=frame_indices_list,
        )
    except TypeError as exc:
        raise RuntimeError(
            "diffusion requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    lag_time = np.asarray(time_arr, dtype=np.float32).reshape(-1)
    if not has_time_ps:
        lag_time = lag_time * float(tstep)
    mat = np.asarray(data, dtype=np.float32)
    if mat.ndim != 2:
        mat = np.empty((0, 0), dtype=np.float32)

    block = n_types + 1
    expected_cols = 4 * block
    if mat.shape[1] not in (0, expected_cols):
        raise RuntimeError(
            f"unexpected MsdPlan output columns: got {mat.shape[1]}, expected {expected_cols}"
        )

    if mat.shape[0] == 0:
        msd_x = np.empty((0,), dtype=np.float32)
        msd_y = np.empty((0,), dtype=np.float32)
        msd_z = np.empty((0,), dtype=np.float32)
        msd = np.empty((0,), dtype=np.float32)
    else:
        total_col = n_types
        msd_x = mat[:, total_col]
        msd_y = mat[:, block + total_col]
        msd_z = mat[:, 2 * block + total_col]
        msd = mat[:, 3 * block + total_col]

    if mat.shape[0] > 0:
        time = np.concatenate(([0.0], lag_time.astype(np.float64))).astype(np.float32)
        msd_x = np.concatenate(([0.0], msd_x.astype(np.float64))).astype(np.float32)
        msd_y = np.concatenate(([0.0], msd_y.astype(np.float64))).astype(np.float32)
        msd_z = np.concatenate(([0.0], msd_z.astype(np.float64))).astype(np.float32)
        msd = np.concatenate(([0.0], msd.astype(np.float64))).astype(np.float32)
    else:
        time = lag_time.astype(np.float32)

    start = max(1, msd.shape[0] // 2)
    slope = _linear_fit(time[start:], msd[start:])
    d_total = slope / 6.0 if slope > 0.0 else 0.0

    out = {
        "time": time,
        "X": msd_x,
        "Y": msd_y,
        "Z": msd_z,
        "MSD": msd,
        "D": float(d_total),
    }
    if individual:
        if mat.shape[0] == 0:
            out["MSD_individual"] = np.empty((n_types, 0), dtype=np.float32)
        else:
            per_atom = mat[:, (3 * block) : (3 * block + n_types)].T
            per_atom = np.concatenate(
                (
                    np.zeros((n_types, 1), dtype=np.float32),
                    per_atom.astype(np.float32, copy=False),
                ),
                axis=1,
            )
            out["MSD_individual"] = per_atom
    return out


def _transition_stats(codes: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    counts = np.zeros((4, 4), dtype=np.float64)
    if codes.size == 0:
        return counts, counts.copy()
    lag = int(max(1, lag))
    n_frames, n_torsions = codes.shape
    if n_frames <= lag or n_torsions == 0:
        return counts, counts.copy()
    for f in range(n_frames - lag):
        src = codes[f]
        dst = codes[f + lag]
        for t in range(n_torsions):
            counts[int(src[t]), int(dst[t])] += 1.0
    probs = counts.copy()
    row_sum = probs.sum(axis=1, keepdims=True)
    nz = row_sum[:, 0] > 0.0
    probs[nz] /= row_sum[nz]
    return counts, probs


def _write_tordiff_outputs(
    out_path: Optional[str],
    diffout_path: Optional[str],
    time_arr: np.ndarray,
    data: np.ndarray,
):
    if out_path:
        table = np.column_stack([time_arr, data[:, 0], data[:, 1], data[:, 2], data[:, 3]])
        np.savetxt(out_path, table, header="time trans cis g_plus g_minus")
    if diffout_path:
        if time_arr.size < 2:
            diff_table = np.empty((0, 5), dtype=np.float32)
        else:
            dt = np.diff(time_arr)
            dt = np.where(dt == 0.0, 1.0, dt)
            diff = (data[1:] - data[:-1]) / dt[:, None]
            diff_table = np.column_stack([time_arr[1:], diff[:, 0], diff[:, 1], diff[:, 2], diff[:, 3]])
        np.savetxt(diffout_path, diff_table, header="time dtrans dcis dg_plus dg_minus")


def tordiff(
    traj,
    system,
    mask: Union[str, Sequence[int], np.ndarray] = "",
    mass: bool = False,
    out: Optional[str] = None,
    diffout: Optional[str] = None,
    time: float = 1.0,
    extra_options: str = "",
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
    return_transitions: bool = False,
    transition_lag: int = 1,
):
    """Toroidal diffusion (torsion state fractions)."""
    del extra_options
    try:
        from warp_md import TorsionDiffusionPlan, ToroidalDiffusionPlan  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "TorsionDiffusionPlan/ToroidalDiffusionPlan bindings unavailable. Rebuild bindings with `maturin develop`."
        ) from exc

    has_torsion_plan = getattr(TorsionDiffusionPlan, "__name__", "") != "_Missing"
    has_toroidal_plan = getattr(ToroidalDiffusionPlan, "__name__", "") != "_Missing"
    use_toroidal_plan = bool(mass or return_transitions)
    if use_toroidal_plan and not has_toroidal_plan:
        raise RuntimeError(
            "ToroidalDiffusionPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )
    if not use_toroidal_plan and not has_torsion_plan:
        raise RuntimeError(
            "TorsionDiffusionPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    mask_expr = mask
    if isinstance(mask, str) and mask in ("", "*", "all", None):
        mask_expr = _all_resid_mask(system) if hasattr(system, "atom_table") else "all"
    if isinstance(mask, str):
        sel = system.select(mask_expr)
    else:
        indices = np.asarray(mask, dtype=np.int64).reshape(-1).tolist()
        sel = system.select_indices(indices)

    counts = None
    probs = None
    state_codes = None
    if use_toroidal_plan:
        plan_kwargs = {
            "mass_weighted": bool(mass),
            "transition_lag": int(max(1, transition_lag)),
            "emit_transitions": bool(return_transitions),
            "store_transition_states": bool(return_transitions and frame_indices is not None),
        }
        try:
            plan = ToroidalDiffusionPlan(sel, **plan_kwargs)
        except TypeError:
            plan_kwargs.pop("store_transition_states", None)
            plan = ToroidalDiffusionPlan(sel, **plan_kwargs)
        if return_transitions:
            if frame_indices is not None:
                run_with_states = getattr(plan, "run_full_with_states", None)
                if not callable(run_with_states):
                    raise TypeError(
                        "ToroidalDiffusionPlan.run_full_with_states is required for frame-sliced transitions"
                    )
                mat, counts, probs, _, states = run_with_states(
                    traj,
                    system,
                    chunk_frames=chunk_frames,
                    device=device,
                    frame_indices=frame_indices,
                )
                state_codes = np.asarray(states, dtype=np.int8)
                if state_codes.ndim != 2:
                    state_codes = np.empty((0, 0), dtype=np.int8)
            else:
                mat, counts, probs, _ = plan.run_full(
                    traj,
                    system,
                    chunk_frames=chunk_frames,
                    device=device,
                    frame_indices=frame_indices,
                )
            counts = np.asarray(counts, dtype=np.float32).reshape((4, 4))
            probs = np.asarray(probs, dtype=np.float32).reshape((4, 4))
        else:
            mat = plan.run(
                traj,
                system,
                chunk_frames=chunk_frames,
                device=device,
                frame_indices=frame_indices,
            )
    else:
        plan = TorsionDiffusionPlan(sel)
        mat = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices,
        )

    data = np.asarray(mat, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 4:
        data = np.empty((0, 4), dtype=np.float32)
    time_arr = np.arange(data.shape[0], dtype=np.float32) * float(time)
    out_dict = {
        "time": time_arr,
        "trans": data[:, 0],
        "cis": data[:, 1],
        "g_plus": data[:, 2],
        "g_minus": data[:, 3],
    }
    if return_transitions and state_codes is not None:
        counts, probs = _transition_stats(state_codes.astype(np.int64, copy=False), transition_lag)
    if return_transitions and counts is not None and probs is not None:
        if time_arr.size >= 2:
            dt = np.diff(time_arr.astype(np.float64))
            dt = dt[np.isfinite(dt) & (dt > 0.0)]
            dt_step = float(np.median(dt)) if dt.size > 0 else float(time)
        else:
            dt_step = float(time)
        lag_time = max(dt_step * float(max(1, transition_lag)), 1e-12)
        total_obs = float(np.sum(counts))
        offdiag = float(total_obs - np.trace(counts))
        rate = 0.0 if total_obs == 0.0 else offdiag / (total_obs * lag_time)
        out_dict["transition_counts"] = counts.astype(np.float32)
        out_dict["transition_matrix"] = probs.astype(np.float32)
        out_dict["transition_rate"] = float(rate)
    _write_tordiff_outputs(out, diffout, time_arr, data)
    return out_dict


def toroidal_diffusion(*args, **kwargs):
    return tordiff(*args, **kwargs)


__all__ = ["diffusion", "tordiff", "toroidal_diffusion"]
