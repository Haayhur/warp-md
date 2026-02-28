# Usage:
# from warp_md.analysis.multidihedral import multidihedral
# out = multidihedral(traj, system, dihedral_types='phi psi', resrange='2-5')

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from .geometry import _dihedral_from_points


MaskLike = Union[str, Sequence[str], np.ndarray]


@dataclass
class DihedralDef:
    label: str
    atoms: Tuple[int, int, int, int]


def _resid_list(system) -> List[int]:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    return list(resids)


def _atoms_by_resid(system) -> dict:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    names = atoms.get("name", [])
    by_res = {}
    for idx, (resid, name) in enumerate(zip(resids, names)):
        by_res.setdefault(resid, {})[str(name).upper()] = idx
    return by_res


def _parse_resrange(resrange: Optional[Union[str, Sequence[int]]]) -> Optional[List[int]]:
    if resrange is None or resrange == "":
        return None
    if isinstance(resrange, str):
        parts = []
        for token in resrange.replace(",", " ").split():
            if "-" in token:
                a, b = token.split("-", 1)
                parts.extend(range(int(a), int(b) + 1))
            else:
                parts.append(int(token))
        return parts
    return [int(x) for x in resrange]


def _phi_psi_defs(system, resrange: Optional[List[int]]) -> List[DihedralDef]:
    by_res = _atoms_by_resid(system)
    resids = sorted(by_res.keys())
    if resrange is not None:
        resids = [r for r in resids if r in resrange]
    defs: List[DihedralDef] = []
    for r in resids:
        prev_res = r - 1
        next_res = r + 1
        cur = by_res.get(r, {})
        prev = by_res.get(prev_res, {})
        nxt = by_res.get(next_res, {})
        # phi: C_{i-1} N_i CA_i C_i
        if prev and {"C", "N", "CA"}.issubset(cur.keys()) and "C" in prev:
            defs.append(
                DihedralDef(
                    label=f"phi:{r}",
                    atoms=(prev["C"], cur["N"], cur["CA"], cur["C"]),
                )
            )
        # psi: N_i CA_i C_i N_{i+1}
        if nxt and {"N", "CA", "C"}.issubset(cur.keys()) and "N" in nxt:
            defs.append(
                DihedralDef(
                    label=f"psi:{r}",
                    atoms=(cur["N"], cur["CA"], cur["C"], nxt["N"]),
                )
            )
    return defs


def _omega_defs(system, resrange: Optional[List[int]]) -> List[DihedralDef]:
    by_res = _atoms_by_resid(system)
    resids = sorted(by_res.keys())
    if resrange is not None:
        resids = [r for r in resids if r in resrange]
    defs: List[DihedralDef] = []
    for r in resids:
        prev_res = r - 1
        cur = by_res.get(r, {})
        prev = by_res.get(prev_res, {})
        if prev and {"N", "CA", "C"}.issubset(prev.keys()) and "N" in cur and "CA" in cur:
            defs.append(
                DihedralDef(
                    label=f"omega:{r}",
                    atoms=(prev["CA"], prev["C"], cur["N"], cur["CA"]),
                )
            )
    return defs


def _chi1_defs(system, resrange: Optional[List[int]]) -> List[DihedralDef]:
    by_res = _atoms_by_resid(system)
    resids = sorted(by_res.keys())
    if resrange is not None:
        resids = [r for r in resids if r in resrange]
    defs: List[DihedralDef] = []
    for r in resids:
        cur = by_res.get(r, {})
        if {"N", "CA", "CB", "CG"}.issubset(cur.keys()):
            defs.append(
                DihedralDef(
                    label=f"chi1:{r}",
                    atoms=(cur["N"], cur["CA"], cur["CB"], cur["CG"]),
                )
            )
    return defs


def _build_defs(system, dihedral_types: Optional[str], resrange: Optional[List[int]]) -> List[DihedralDef]:
    if dihedral_types is None:
        types = ["phi", "psi"]
    else:
        types = [t.strip().lower() for t in dihedral_types.replace(",", " ").split() if t.strip()]
    defs: List[DihedralDef] = []
    for t in types:
        if t == "phi":
            defs.extend(_phi_psi_defs(system, resrange))
        elif t == "psi":
            defs.extend(_phi_psi_defs(system, resrange))
        elif t == "omega":
            defs.extend(_omega_defs(system, resrange))
        elif t in ("chi", "chi1", "chip"):
            defs.extend(_chi1_defs(system, resrange))
        else:
            raise ValueError(f"unsupported dihedral type: {t}")
    # remove duplicates by label
    uniq = {}
    for d in defs:
        uniq[d.label] = d
    return list(uniq.values())


def _compute_dihedral_series(coords: np.ndarray, atoms: Tuple[int, int, int, int], range360: bool) -> np.ndarray:
    a, b, c, d = atoms
    p0 = coords[:, a, :]
    p1 = coords[:, b, :]
    p2 = coords[:, c, :]
    p3 = coords[:, d, :]
    vals = _dihedral_from_points(p0, p1, p2, p3)
    if range360:
        vals = (vals + 360.0) % 360.0
    return vals


def multidihedral(
    traj,
    system,
    dihedral_types: Optional[str] = None,
    resrange: Optional[Union[str, Sequence[int]]] = None,
    range360: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "dict",
    mass: bool = False,
    chunk_frames: Optional[int] = None,
):
    """Compute multiple dihedrals (phi/psi/omega/chi1)."""
    _ = mass
    res_list = _parse_resrange(resrange)
    defs = _build_defs(system, dihedral_types, res_list)
    if not defs:
        return {} if dtype == "dict" else np.empty((0, 0), dtype=np.float32)

    coords = None
    # read coordinates once (avoid exhausting trajectory)
    from .rmsd import _read_all as _read_all_coords
    coords, _ = _read_all_coords(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return {} if dtype == "dict" else np.empty((0, 0), dtype=np.float32)

    if frame_indices is not None:
        n_frames = coords.shape[0]
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        coords = coords[select]

    masks = []
    labels = []
    for d in defs:
        a, b, c, d_idx = d.atoms
        labels.append(d.label)

    n_frames = coords.shape[0]
    values = np.zeros((len(defs), n_frames), dtype=np.float64)
    for i, d in enumerate(defs):
        values[i] = _compute_dihedral_series(coords, d.atoms, range360)
    # values shape (n_defs, n_frames)
    if dtype == "dict":
        return {label: values[i].astype(np.float32) for i, label in enumerate(labels)}
    return values.astype(np.float32)


__all__ = ["multidihedral"]
