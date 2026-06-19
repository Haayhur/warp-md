# Usage:
# from warp_md.analysis.multidihedral import multidihedral
# out = multidihedral(traj, system, dihedral_types='phi psi', resrange='2-5')

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from ._runtime import load_native_symbol, native_inputs


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


def _resnames_by_resid(system) -> dict:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    resnames = atoms.get("resname", [])
    out = {}
    for resid, resname in zip(resids, resnames):
        out.setdefault(resid, str(resname).upper())
    return out


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


def _phi_psi_defs(
    system,
    resrange: Optional[List[int]],
    kind: Optional[str] = None,
) -> List[DihedralDef]:
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
        if (
            kind in (None, "phi")
            and prev
            and {"C", "N", "CA"}.issubset(cur.keys())
            and "C" in prev
        ):
            defs.append(
                DihedralDef(
                    label=f"phi:{r}",
                    atoms=(prev["C"], cur["N"], cur["CA"], cur["C"]),
                )
            )
        # psi: N_i CA_i C_i N_{i+1}
        if (
            kind in (None, "psi")
            and nxt
            and {"N", "CA", "C"}.issubset(cur.keys())
            and "N" in nxt
        ):
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


_CHI_TEMPLATES = {
    "ARG": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD"),
        "chi3": ("CB", "CG", "CD", "NE"),
        "chi4": ("CG", "CD", "NE", "CZ"),
        "chi5": ("CD", "NE", "CZ", "NH1"),
    },
    "ASN": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "OD1"),
    },
    "ASP": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "OD1"),
    },
    "CYS": {"chi1": ("N", "CA", "CB", "SG")},
    "GLN": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD"),
        "chi3": ("CB", "CG", "CD", "OE1"),
    },
    "GLU": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD"),
        "chi3": ("CB", "CG", "CD", "OE1"),
    },
    "HIS": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "ND1"),
    },
    "HIE": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "ND1"),
    },
    "HID": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "ND1"),
    },
    "HIP": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "ND1"),
    },
    "ILE": {
        "chi1": ("N", "CA", "CB", "CG1"),
        "chi2": ("CA", "CB", "CG1", "CD1"),
    },
    "LEU": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD1"),
    },
    "LYS": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD"),
        "chi3": ("CB", "CG", "CD", "CE"),
        "chi4": ("CG", "CD", "CE", "NZ"),
    },
    "MET": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "SD"),
        "chi3": ("CB", "CG", "SD", "CE"),
    },
    "PHE": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD1"),
    },
    "PRO": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD"),
    },
    "SER": {"chi1": ("N", "CA", "CB", "OG")},
    "THR": {"chi1": ("N", "CA", "CB", "OG1")},
    "TRP": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD1"),
    },
    "TYR": {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD1"),
    },
    "VAL": {"chi1": ("N", "CA", "CB", "CG1")},
}

_GENERIC_CHI = {
    "chi1": ("N", "CA", "CB", ("CG", "CG1", "OG", "OG1", "SG")),
    "chi2": ("CA", "CB", ("CG", "CG1"), ("CD", "CD1", "OD1", "ND1", "SD")),
    "chi3": (("CB",), ("CG", "CG1"), ("CD", "SD"), ("NE", "CE", "OE1")),
    "chi4": (("CG", "CG1"), ("CD",), ("NE", "CE"), ("CZ", "NZ")),
    "chi5": (("CD",), ("NE",), ("CZ",), ("NH1",)),
}


def _pick_atom(cur: dict, spec) -> Optional[int]:
    names = spec if isinstance(spec, tuple) else (spec,)
    for name in names:
        if name in cur:
            return cur[name]
    return None


def _chi_defs(system, resrange: Optional[List[int]], chi_name: str) -> List[DihedralDef]:
    by_res = _atoms_by_resid(system)
    resnames = _resnames_by_resid(system)
    resids = sorted(by_res.keys())
    if resrange is not None:
        resids = [r for r in resids if r in resrange]
    defs: List[DihedralDef] = []
    for r in resids:
        cur = by_res.get(r, {})
        resname = resnames.get(r, "")
        template = _CHI_TEMPLATES.get(resname, {}).get(chi_name, _GENERIC_CHI[chi_name])
        atoms = [_pick_atom(cur, spec) for spec in template]
        if all(atom is not None for atom in atoms):
            defs.append(
                DihedralDef(
                    label=f"{chi_name}:{r}",
                    atoms=tuple(int(atom) for atom in atoms),
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
            defs.extend(_phi_psi_defs(system, resrange, "phi"))
        elif t == "psi":
            defs.extend(_phi_psi_defs(system, resrange, "psi"))
        elif t == "omega":
            defs.extend(_omega_defs(system, resrange))
        elif t in ("chi", "chip"):
            for chi_name in ("chi1", "chi2", "chi3", "chi4", "chi5"):
                defs.extend(_chi_defs(system, resrange, chi_name))
        elif t in ("chi1", "chi2", "chi3", "chi4", "chi5"):
            defs.extend(_chi_defs(system, resrange, t))
        else:
            raise ValueError(f"unsupported dihedral type: {t}")
    # remove duplicates by label
    uniq = {}
    for d in defs:
        uniq[d.label] = d
    return list(uniq.values())


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
    """Compute multiple dihedrals (phi/psi/omega/chi1-chi5)."""
    res_list = _parse_resrange(resrange)
    defs = _build_defs(system, dihedral_types, res_list)
    if not defs:
        return {} if dtype == "dict" else np.empty((0, 0), dtype=np.float32)
    plan_cls = load_native_symbol("MultiDihedralPlan")
    if plan_cls is None:
        raise RuntimeError("MultiDihedralPlan binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native multidihedral inputs")
    groups = []
    labels = []
    for d in defs:
        labels.append(d.label)
        a, b, c, d_idx = d.atoms
        groups.append(
            (
                native_system.select_indices([int(a)]),
                native_system.select_indices([int(b)]),
                native_system.select_indices([int(c)]),
                native_system.select_indices([int(d_idx)]),
            )
        )
    try:
        plan = plan_cls(
            groups,
            mass_weighted=mass,
            pbc="none",
            degrees=True,
            range360=range360,
        )
        values = np.asarray(
            plan.run(
                native_traj,
                native_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=None if frame_indices is None else [int(i) for i in frame_indices],
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native MultiDihedralPlan execution failed") from exc
    if values.ndim == 1:
        values = values.reshape(1, -1)
    elif values.ndim == 2 and values.shape[1] == len(labels):
        values = values.T
    if dtype == "dict":
        return {label: values[i].astype(np.float32, copy=False) for i, label in enumerate(labels)}
    return values.astype(np.float32, copy=False)


__all__ = ["multidihedral"]
