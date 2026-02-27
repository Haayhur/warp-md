# Usage:
# from warp_md.analysis.nmr import nh_order_parameters, jcoupling
# s2 = nh_order_parameters(traj, system, nh_pairs, method="tensor")
# j = jcoupling(traj, system, dih_indices, karplus=(6.4, -1.4, 1.9))

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np


PairLike = Union[Tuple[int, int], Sequence[int]]
QuadLike = Union[Tuple[int, int, int, int], Sequence[int]]


def _pairs_from_selection(system, selection: str) -> Sequence[Tuple[int, int]]:
    sel = system.select(selection)
    idx = list(sel.indices)
    if len(idx) < 2:
        raise ValueError("selection must contain at least two atoms")
    return [(int(idx[i]), int(idx[i + 1])) for i in range(len(idx) - 1)]


def _resolve_pairs(system, vector_pairs, selection: Optional[str]):
    if vector_pairs is None:
        if not selection:
            raise ValueError("selection is required when vector_pairs is None")
        return _pairs_from_selection(system, selection)
    if isinstance(vector_pairs, str):
        if vector_pairs.lower() != "sequential":
            raise ValueError("vector_pairs string must be 'sequential'")
        if not selection:
            raise ValueError("selection is required for sequential pairs")
        return _pairs_from_selection(system, selection)

    pairs = []
    for pair in vector_pairs:
        if len(pair) != 2:
            raise ValueError("each vector pair must have length 2")
        a, b = pair
        if isinstance(a, str) and isinstance(b, str):
            sel_a = system.select(a)
            sel_b = system.select(b)
            if len(sel_a.indices) != len(sel_b.indices):
                raise ValueError("selection pairs must match in length")
            pairs.extend(zip((int(i) for i in sel_a.indices), (int(i) for i in sel_b.indices)))
        else:
            pairs.append((int(a), int(b)))
    return pairs


def _resolve_quads(system, dihedral_indices: Sequence[QuadLike]) -> np.ndarray:
    resolved = []
    for quad in dihedral_indices:
        if len(quad) != 4:
            raise ValueError("each dihedral entry must have length 4")
        if all(isinstance(v, str) for v in quad):
            sels = [system.select(v) for v in quad]
            if any(len(s.indices) != 1 for s in sels):
                raise ValueError("each selection in dihedral entry must map to one atom")
            resolved.append(tuple(int(s.indices[0]) for s in sels))
        else:
            resolved.append(tuple(int(v) for v in quad))
    idx = np.asarray(resolved, dtype=np.int64)
    if idx.ndim != 2 or idx.shape[1] != 4:
        raise ValueError("dihedral_indices must be shape (n, 4)")
    return idx


def _read_karplus(kfile: Optional[str], karplus: Tuple[float, float, float]):
    if kfile is None:
        return karplus
    with open(kfile, "r", encoding="utf-8") as handle:
        found = None
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            parts = line.split()
            nums = []
            for part in parts:
                try:
                    nums.append(float(part))
                except ValueError:
                    break
            if len(nums) >= 3:
                found = (nums[0], nums[1], nums[2])
                break
        if found is None:
            raise ValueError("kfile does not contain 3 numeric karplus parameters")
    return found


def ired_vector_and_matrix(
    traj,
    system,
    vector_pairs: Optional[Sequence[PairLike]] = None,
    selection: Optional[str] = None,
    order: int = 2,
    length_scale: float = 0.1,
    pbc: str = "none",
    chunk_frames: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    return_corr: bool = True,
    corr_mode: str = "tensor",
):
    """Return unit vectors and correlation object for IRED-like workflow."""
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if corr_mode not in ("tensor", "timecorr"):
        raise ValueError("corr_mode must be 'tensor' or 'timecorr'")

    pbc_mode = pbc.lower()
    if pbc_mode not in ("none", "orthorhombic"):
        raise ValueError("pbc must be 'none' or 'orthorhombic'")

    pairs = _resolve_pairs(system, vector_pairs, selection)
    n_pairs = len(pairs)
    if n_pairs == 0:
        vecs = np.empty((0, 0, 3), dtype=np.float64)
        if return_corr:
            if corr_mode == "tensor":
                return vecs, np.empty((0, 0), dtype=np.float64)
            return vecs, np.empty((0, 0), dtype=np.float64)
        return vecs

    try:
        from warp_md import NmrIredPlan  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "NmrIredPlan binding unavailable. Rebuild bindings with `maturin develop`."
        ) from exc
    if getattr(NmrIredPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "NmrIredPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    plan = NmrIredPlan(
        [(int(a), int(b)) for a, b in pairs],
        order=order,
        length_scale=float(length_scale),
        pbc=pbc_mode,
        corr_mode=corr_mode,
        return_corr=return_corr,
    )
    try:
        out = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=frame_indices,
        )
    except TypeError as exc:
        raise RuntimeError(
            "ired_vector_and_matrix requires Rust-backed trajectory/system objects."
        ) from exc

    if not return_corr:
        return np.asarray(out, dtype=np.float64)
    vecs, corr = out
    return np.asarray(vecs, dtype=np.float64), np.asarray(corr, dtype=np.float64)


def nh_order_parameters(
    traj,
    system,
    vector_pairs: Optional[Sequence[PairLike]] = None,
    selection: Optional[str] = None,
    order: int = 2,
    tstep: float = 1.0,
    tcorr: float = 10000.0,
    length_scale: float = 0.1,
    pbc: str = "none",
    chunk_frames: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    method: str = "tensor",
):
    """Compute N-H order parameters (S2)."""
    if method not in ("tensor", "timecorr_fit"):
        raise ValueError("method must be 'tensor' or 'timecorr_fit'")
    if method == "timecorr_fit" and int(order) != 2:
        raise ValueError("method='timecorr_fit' currently supports order=2 only")
    if method == "timecorr_fit" and float(tstep) <= 0.0:
        raise ValueError("method='timecorr_fit' requires tstep > 0")
    if method == "timecorr_fit" and tcorr is not None and float(tcorr) <= 0.0:
        raise ValueError("method='timecorr_fit' requires tcorr > 0 when provided")

    if method == "tensor":
        vecs = ired_vector_and_matrix(
            traj,
            system,
            vector_pairs=vector_pairs,
            selection=selection,
            order=order,
            length_scale=length_scale,
            pbc=pbc,
            chunk_frames=chunk_frames,
            frame_indices=frame_indices,
            return_corr=False,
        )
        if vecs.size == 0:
            return np.empty(0, dtype=np.float32)
        n_frames = vecs.shape[0]
        m = np.einsum("fpi,fpj->pij", vecs, vecs) / float(n_frames)
        tr_m2 = np.einsum("pij,pij->p", m, m)
        s2 = 1.5 * tr_m2 - 0.5
        return s2.astype(np.float32)

    vecs, corr = ired_vector_and_matrix(
        traj,
        system,
        vector_pairs=vector_pairs,
        selection=selection,
        order=2,
        length_scale=length_scale,
        pbc=pbc,
        chunk_frames=chunk_frames,
        frame_indices=frame_indices,
        return_corr=True,
        corr_mode="timecorr",
    )
    if vecs.size == 0 or corr.size == 0:
        return np.empty(0, dtype=np.float32)

    n_lags, n_pairs = corr.shape
    times = np.arange(n_lags, dtype=np.float64) * float(tstep)
    out = np.zeros((n_pairs,), dtype=np.float64)
    fit_mask = times > 0.0
    if tcorr is not None and float(tcorr) > 0.0:
        fit_mask &= times <= float(tcorr)

    for p in range(n_pairs):
        c = corr[:, p].astype(np.float64, copy=False)
        tail_start = max(1, int(0.7 * n_lags))
        s2_tail = float(np.mean(c[tail_start:])) if tail_start < n_lags else float(c[-1])
        s2 = float(np.clip(s2_tail, -0.5, 1.0))
        rem = c - s2
        mask = fit_mask & np.isfinite(rem) & (rem > 1e-8)
        if np.count_nonzero(mask) >= 2:
            A = np.vstack([times[mask], np.ones(np.count_nonzero(mask))]).T
            slope, intercept = np.linalg.lstsq(A, np.log(rem[mask]), rcond=None)[0]
            if np.isfinite(intercept):
                amp = float(np.exp(intercept))
                s2_fit = float(np.clip(1.0 - amp, -0.5, 1.0))
                s2 = 0.5 * (s2 + s2_fit)
        out[p] = float(np.clip(s2, -0.5, 1.0))

    return out.astype(np.float32)


def jcoupling(
    traj,
    system,
    dihedral_indices: Sequence[QuadLike],
    karplus: Tuple[float, float, float] = (6.4, -1.4, 1.9),
    kfile: Optional[str] = None,
    phase_deg: float = 0.0,
    length_scale: float = 0.1,
    pbc: str = "none",
    chunk_frames: Optional[int] = None,
    frame_indices: Optional[Sequence[int]] = None,
    device: str = "auto",
    return_dihedral: bool = False,
):
    """Compute scalar J-coupling via a Karplus-like equation.

    J = A*cos^2(theta+phase) + B*cos(theta+phase) + C
    """
    a, b, c = _read_karplus(kfile, karplus)
    phase = np.deg2rad(phase_deg)
    idx = _resolve_quads(system, dihedral_indices)
    n = idx.shape[0]

    pbc_mode = pbc.lower()
    if pbc_mode not in ("none", "orthorhombic"):
        raise ValueError("pbc must be 'none' or 'orthorhombic'")

    try:
        from warp_md import MultiDihedralPlan  # type: ignore

    except Exception as exc:
        raise RuntimeError(
            "MultiDihedralPlan binding unavailable. Rebuild bindings with `maturin develop`."
        ) from exc
    if getattr(MultiDihedralPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "MultiDihedralPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )
    if not hasattr(system, "select_indices"):
        raise RuntimeError("jcoupling requires a Rust-backed `system` with `select_indices`.")

    groups = []
    for a_idx, b_idx, c_idx, d_idx in idx.tolist():
        sel_a = system.select_indices([int(a_idx)])
        sel_b = system.select_indices([int(b_idx)])
        sel_c = system.select_indices([int(c_idx)])
        sel_d = system.select_indices([int(d_idx)])
        groups.append((sel_a, sel_b, sel_c, sel_d))

    plan = MultiDihedralPlan(groups, False, pbc_mode, False, False)
    try:
        angles = np.asarray(
            plan.run(
                traj,
                system,
                chunk_frames=chunk_frames,
                device=device,
                frame_indices=frame_indices,
            ),
            dtype=np.float64,
        )
    except TypeError as exc:
        raise RuntimeError(
            "jcoupling requires Rust-backed trajectory/system objects."
        ) from exc

    if angles.ndim == 1:
        angles = angles.reshape((-1, 1))
    angles = angles + phase
    cos = np.cos(angles)
    jvals = (a * cos * cos + b * cos + c).astype(np.float32)
    if return_dihedral:
        return jvals, angles.astype(np.float32)
    return jvals


calc_ired_vector_and_matrix = ired_vector_and_matrix
calc_nh_order_parameters = nh_order_parameters

__all__ = [
    "jcoupling",
    "ired_vector_and_matrix",
    "nh_order_parameters",
    "calc_ired_vector_and_matrix",
    "calc_nh_order_parameters",
]
