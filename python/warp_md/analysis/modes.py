# Usage:
# from warp_md.analysis.modes import analyze_modes
# out = analyze_modes("fluct", evecs, evals, scalar_type="mwcovar", masses=masses)

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .trajectory import ArrayTrajectory


def analyze_modes(
    mode_type: str,
    eigenvectors: np.ndarray,
    eigenvalues: np.ndarray,
    scalar_type: str = "mwcovar",
    options: str = "",
    dtype: str = "dict",
    *,
    system=None,
    average_coords: Optional[np.ndarray] = None,
    eigenvectors2: Optional[np.ndarray] = None,
    masses: Optional[Sequence[float]] = None,
    mask_pairs: Optional[Sequence[Tuple[int, int]]] = None,
) -> Union[Dict[str, np.ndarray], np.ndarray, float, ArrayTrajectory]:
    """Analyze eigenmodes.

    Supported mode_type: fluct, displ, corr, eigenval, trajout, rmsip.
    """
    mode_type = mode_type.strip().lower()
    opts = _parse_options(options)

    vecs = np.asarray(eigenvectors, dtype=np.float64)
    vals = np.asarray(eigenvalues, dtype=np.float64)
    if vecs.ndim != 2:
        raise ValueError("eigenvectors must be 2D")
    if vals.ndim != 1:
        raise ValueError("eigenvalues must be 1D")
    if vecs.shape[0] != vals.size and vecs.shape[1] != vals.size:
        raise ValueError("eigenvectors shape does not match eigenvalues")
    if vecs.shape[0] == vals.size:
        vecs = vecs.copy()
    else:
        vecs = vecs.T.copy()

    n_modes, n_features = vecs.shape
    if n_features % 3 != 0:
        raise ValueError("eigenvectors length must be multiple of 3")
    n_atoms = n_features // 3

    mode_indices = _select_modes(vals, opts)
    vecs = vecs[mode_indices]
    vals = vals[mode_indices]

    if scalar_type.lower() == "mwcovar":
        if masses is None and system is not None:
            atoms = system.atom_table()
            masses = atoms.get("mass", None)
        if masses is not None:
            masses = np.asarray(masses, dtype=np.float64)
            if masses.size != n_atoms:
                raise ValueError("masses length must match atom count")
            scale = np.repeat(1.0 / np.sqrt(np.clip(masses, 1e-12, None)), 3)
            vecs = vecs * scale[None, :]

    prefix = opts.get("setname")
    if not prefix:
        prefix = mode_type.upper()

    if mode_type == "fluct":
        rmsx, rmsy, rmsz, rms = _fluct(vecs, vals, n_atoms)
        if dtype == "dict":
            return {
                f"{prefix}[rmsX]": rmsx,
                f"{prefix}[rmsY]": rmsy,
                f"{prefix}[rmsZ]": rmsz,
                f"{prefix}[rms]": rms,
            }
        return np.column_stack((rmsx, rmsy, rmsz, rms))

    if mode_type == "displ":
        displx, disply, displz = _displ(vecs, vals, n_atoms)
        if dtype == "dict":
            return {
                f"{prefix}[displX]": displx,
                f"{prefix}[displY]": disply,
                f"{prefix}[displZ]": displz,
            }
        return np.column_stack((displx, disply, displz))

    if mode_type == "eigenval":
        frac, cum, evals = _eigenval(vals)
        if dtype == "dict":
            return {
                f"{prefix}[Frac]": frac,
                f"{prefix}[Cumulative]": cum,
                f"{prefix}[Eigenval]": evals,
            }
        return np.column_stack((frac, cum, evals))

    if mode_type == "rmsip":
        vecs2 = vecs if eigenvectors2 is None else np.asarray(eigenvectors2, dtype=np.float64)
        if vecs2.ndim != 2:
            raise ValueError("eigenvectors2 must be 2D")
        if vecs2.shape[0] == n_features:
            vecs2 = vecs2.T
        if vecs2.shape[1] != n_features:
            raise ValueError("eigenvectors2 feature size mismatch")
        val = _rmsip(vecs, vecs2)
        if dtype == "dict":
            return {prefix: np.array([val], dtype=np.float32)}
        return float(val)

    if mode_type == "corr":
        if average_coords is None:
            raise ValueError("average_coords required for corr")
        pairs = _resolve_pairs(system, mask_pairs, opts)
        corr = _corr(vecs, average_coords, pairs)
        if dtype == "dict":
            return {prefix: corr}
        return corr

    if mode_type == "trajout":
        if average_coords is None:
            raise ValueError("average_coords required for trajout")
        tmode = int(opts.get("tmode", 1)) - 1
        if tmode < 0 or tmode >= vecs.shape[0]:
            raise ValueError("tmode out of range")
        pcmin = float(opts.get("pcmin", -1.0))
        pcmax = float(opts.get("pcmax", 1.0))
        nframes = int(opts.get("nframes", 21))
        factor = float(opts.get("factor", 1.0))
        coords = _trajout(average_coords, vecs[tmode], pcmin, pcmax, nframes, factor)
        return ArrayTrajectory(coords.astype(np.float32))

    raise ValueError("mode_type must be one of: fluct, displ, corr, eigenval, trajout, rmsip")


def _parse_options(options: str) -> Dict[str, str]:
    tokens = options.split()
    out: Dict[str, str] = {}
    i = 0
    while i < len(tokens):
        key = tokens[i].lower()
        if key in ("beg", "end", "factor", "setname", "pcmin", "pcmax", "tmode", "nframes"):
            if i + 1 >= len(tokens):
                break
            out[key] = tokens[i + 1]
            i += 2
        elif key in ("bose", "calcall"):
            out[key] = "true"
            i += 1
        elif key in ("maskp", "mask1", "mask2"):
            out.setdefault("masks", []).append(tokens[i + 1])
            i += 2
        else:
            i += 1
    return out


def _select_modes(vals: np.ndarray, opts: Dict[str, str]) -> np.ndarray:
    n_modes = vals.size
    beg = int(opts.get("beg", 1)) - 1
    end = int(opts.get("end", n_modes)) - 1
    beg = max(0, beg)
    end = min(n_modes - 1, end)
    indices = np.arange(n_modes)
    if "calcall" not in opts:
        indices = indices[vals > 0.0]
    if indices.size == 0:
        return np.array([], dtype=np.int64)
    mask = (indices >= beg) & (indices <= end)
    return indices[mask]


def _fluct(vecs: np.ndarray, vals: np.ndarray, n_atoms: int):
    if vals.size == 0:
        z = np.zeros(n_atoms, dtype=np.float32)
        return z, z, z, z
    vv = (vecs * vecs) * vals[:, None]
    vv = vv.reshape(vals.size, n_atoms, 3).sum(axis=0)
    rmsx = np.sqrt(vv[:, 0])
    rmsy = np.sqrt(vv[:, 1])
    rmsz = np.sqrt(vv[:, 2])
    rms = np.sqrt(rmsx * rmsx + rmsy * rmsy + rmsz * rmsz)
    return rmsx.astype(np.float32), rmsy.astype(np.float32), rmsz.astype(np.float32), rms.astype(np.float32)


def _displ(vecs: np.ndarray, vals: np.ndarray, n_atoms: int):
    if vals.size == 0:
        z = np.zeros(n_atoms, dtype=np.float32)
        return z, z, z
    vv = (vecs * vecs) * vals[:, None]
    vv = vv.reshape(vals.size, n_atoms, 3).sum(axis=0)
    displx = np.sqrt(vv[:, 0])
    disply = np.sqrt(vv[:, 1])
    displz = np.sqrt(vv[:, 2])
    return displx.astype(np.float32), disply.astype(np.float32), displz.astype(np.float32)


def _eigenval(vals: np.ndarray):
    if vals.size == 0:
        z = np.zeros(0, dtype=np.float32)
        return z, z, z
    total = float(np.sum(vals))
    if total == 0.0:
        frac = np.zeros_like(vals)
    else:
        frac = vals / total
    cum = np.cumsum(frac)
    return frac.astype(np.float32), cum.astype(np.float32), vals.astype(np.float32)


def _rmsip(vecs: np.ndarray, vecs2: np.ndarray) -> float:
    a = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    b = vecs2 / np.linalg.norm(vecs2, axis=1, keepdims=True)
    dot = a @ b.T
    val = np.sqrt(np.mean(dot * dot))
    return float(val)


def _resolve_pairs(system, mask_pairs, opts: Dict[str, str]) -> List[Tuple[int, int]]:
    if mask_pairs is not None:
        return [(int(a), int(b)) for a, b in mask_pairs]
    if system is None:
        raise ValueError("system required to resolve mask pairs")
    masks = opts.get("masks", [])
    if len(masks) >= 2:
        if len(masks) % 2 == 0:
            pairs = []
            for i in range(0, len(masks), 2):
                sel_a = system.select(masks[i])
                sel_b = system.select(masks[i + 1])
                a_idx = list(sel_a.indices)
                b_idx = list(sel_b.indices)
                if len(a_idx) != len(b_idx):
                    raise ValueError("mask pairs must have same length")
                pairs.extend(list(zip(a_idx, b_idx)))
            return [(int(a), int(b)) for a, b in pairs]
    raise ValueError("mask pairs not provided")


def _corr(vecs: np.ndarray, average_coords: np.ndarray, pairs: List[Tuple[int, int]]):
    avg = np.asarray(average_coords, dtype=np.float64)
    if avg.ndim != 2 or avg.shape[1] != 3:
        raise ValueError("average_coords must be (n_atoms, 3)")
    n_modes = vecs.shape[0]
    n_pairs = len(pairs)
    out = np.zeros((n_pairs, n_modes), dtype=np.float32)
    for p_idx, (a, b) in enumerate(pairs):
        if a >= avg.shape[0] or b >= avg.shape[0]:
            continue
        base = avg[b] - avg[a]
        norm = np.linalg.norm(base)
        if norm == 0.0:
            continue
        u = base / norm
        for m in range(n_modes):
            v = vecs[m].reshape(-1, 3)
            disp = v[b] - v[a]
            out[p_idx, m] = float(np.dot(disp, u))
    return out


def _trajout(
    average_coords: np.ndarray,
    mode_vec: np.ndarray,
    pcmin: float,
    pcmax: float,
    nframes: int,
    factor: float,
) -> np.ndarray:
    avg = np.asarray(average_coords, dtype=np.float64)
    if avg.ndim != 2 or avg.shape[1] != 3:
        raise ValueError("average_coords must be (n_atoms, 3)")
    mode = mode_vec.reshape(avg.shape[0], 3)
    nframes = max(2, int(nframes))
    amps = np.linspace(pcmin, pcmax, nframes)
    coords = np.empty((nframes, avg.shape[0], 3), dtype=np.float64)
    for i, amp in enumerate(amps):
        coords[i] = avg + factor * amp * mode
    return coords


__all__ = ["analyze_modes"]
