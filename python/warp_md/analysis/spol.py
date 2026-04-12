from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md

from .h2order import _DEFAULT_WATER_RESNAMES, _chain_ids, _resolve_charges, _select
from .sorient import _resolve_triplets


MaskLike = Union[str, Sequence[int], np.ndarray]

_SpolPlan = (
    getattr(warp_md.traj_py, "PySpolPlan", None)
    if getattr(warp_md, "traj_py", None)
    else None
)


def spol(
    traj,
    system,
    solute_selection: MaskLike,
    solvent_selection: Optional[MaskLike] = None,
    charges: Optional[Sequence[float]] = None,
    atom1_indices: Optional[Sequence[int]] = None,
    atom2_indices: Optional[Sequence[int]] = None,
    atom3_indices: Optional[Sequence[int]] = None,
    molecules: Optional[Sequence[Sequence[int]]] = None,
    r_min: float = 0.0,
    r_max: float = 0.32,
    bin: float = 0.01,
    use_com: bool = False,
    reference_atom: int = 0,
    direction_atom_offsets: Sequence[int] = (0, 1, 2),
    refdip: float = 0.0,
    r_hist_max: Optional[float] = None,
    length_scale: Optional[float] = None,
    water_resnames: Sequence[str] = _DEFAULT_WATER_RESNAMES,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Solvent dipole and polarization analysis via Rust plan path.

    Auto-groups solvent molecules when `molecules=` is not given, preferring
    `atom_table()['mol_id']` and otherwise falling back to residue grouping.
    Explicit whole-molecule groups can be passed with `molecules=`. Legacy
    explicit 3-atom triplet arguments remain supported. `solvent_selection`,
    when provided, must cover whole molecules.
    """
    if _SpolPlan is None:
        raise RuntimeError(
            "PySpolPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    solute_sel = _select(system, solute_selection)
    direction_offsets = np.asarray(direction_atom_offsets, dtype=np.int64).reshape(-1)
    if direction_offsets.shape != (3,) or np.any(direction_offsets < 0):
        raise ValueError("direction_atom_offsets must contain exactly 3 non-negative offsets")
    if reference_atom < 0:
        raise ValueError("reference_atom must be >= 0")
    explicit_triplets = any(v is not None for v in (atom1_indices, atom2_indices, atom3_indices))
    if molecules is not None and explicit_triplets:
        raise ValueError("pass either `molecules=` or explicit atom1/atom2/atom3 triplets, not both")
    molecule_atoms = None
    molecule_offsets = None
    if molecules is not None:
        atom1 = []
        atom2 = []
        atom3 = []
        molecule_atoms, molecule_offsets = _flatten_molecules(
            molecules,
            int(reference_atom),
            direction_offsets.tolist(),
        )
    elif explicit_triplets:
        atom1, atom2, atom3 = _resolve_triplets(
            system,
            solvent_selection,
            atom1_indices,
            atom2_indices,
            atom3_indices,
            water_resnames,
        )
    else:
        atom1 = []
        atom2 = []
        atom3 = []
        auto_molecules = _molecule_groups(system, solvent_selection, water_resnames)
        molecule_atoms, molecule_offsets = _flatten_molecules(
            auto_molecules,
            int(reference_atom),
            direction_offsets.tolist(),
        )
    charge_list = _resolve_charges(system, charges)
    plan = _SpolPlan(
        solute_sel,
        atom1,
        atom2,
        atom3,
        charge_list,
        r_min=r_min,
        r_max=r_max,
        bin=bin,
        use_com=use_com,
        reference_atom=int(reference_atom),
        direction_atoms=direction_offsets.tolist(),
        refdip=refdip,
        r_hist_max=r_hist_max,
        length_scale=length_scale,
        molecule_atoms=molecule_atoms,
        molecule_offsets=molecule_offsets,
    )
    try:
        out = plan.run(
            traj,
            system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=None if frame_indices is None else [int(v) for v in frame_indices],
        )
    except TypeError as exc:
        raise RuntimeError(
            "spol requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "r": np.asarray(out["r"], dtype=np.float32),
        "cumulative_count": np.asarray(out["cumulative_count"], dtype=np.float32),
        "shell_count": np.asarray(out["shell_count"], dtype=np.uint64),
        "shell_count_per_frame": np.asarray(out["shell_count_per_frame"], dtype=np.float32),
        "average_shell_size": float(out["average_shell_size"]),
        "average_dipole": float(out["average_dipole"]),
        "dipole_std": float(out["dipole_std"]),
        "average_radial_dipole": float(out["average_radial_dipole"]),
        "average_radial_polarization": float(out["average_radial_polarization"]),
        "window_count": int(out["window_count"]),
        "r_window": np.asarray(out["r_window"], dtype=np.float32),
        "bin_width": float(out["bin_width"]),
        "r_hist_max": float(out["r_hist_max"]),
        "use_com": bool(out["use_com"]),
        "reference_atom": int(out["reference_atom"]),
        "refdip": float(out["refdip"]),
        "n_frames": int(out["n_frames"]),
        "used_box": bool(out["used_box"]),
        "length_scale": float(out["length_scale"]),
        "dipole_unit": str(out["dipole_unit"]),
    }


__all__ = ["spol"]


def _flatten_molecules(
    molecules: Sequence[Sequence[int]],
    reference_atom: int,
    direction_atom_offsets: Sequence[int],
) -> tuple[list[int], list[int]]:
    molecule_atoms: list[int] = []
    molecule_offsets = [0]
    max_direction = max(int(v) for v in direction_atom_offsets)
    for molecule in molecules:
        arr = np.asarray(molecule, dtype=np.int64).reshape(-1)
        if arr.size == 0 or np.any(arr < 0):
            raise ValueError("molecules must contain non-empty groups of non-negative atom indices")
        if reference_atom >= arr.size or max_direction >= arr.size:
            raise ValueError("reference_atom/direction_atom_offsets exceed molecule size")
        molecule_atoms.extend(int(v) for v in arr.tolist())
        molecule_offsets.append(len(molecule_atoms))
    if len(molecule_offsets) < 2:
        raise ValueError("spol requires at least one solvent molecule")
    return molecule_atoms, molecule_offsets


def _molecule_groups(system, selection: Optional[MaskLike], water_resnames: Sequence[str]):
    atoms = system.atom_table()
    mol_id = atoms.get("mol_id", [])
    if len(mol_id) == len(atoms.get("resname", [])) and len(mol_id) > 0:
        return _mol_id_molecules(system, selection, water_resnames)
    return _residue_molecules(system, selection, water_resnames)


def _selected_set(system, selection: Optional[MaskLike]) -> Optional[set[int]]:
    if selection is None:
        return None
    return {int(i) for i in _select(system, selection).indices}


def _mol_id_molecules(system, selection: Optional[MaskLike], water_resnames: Sequence[str]):
    atoms = system.atom_table()
    resnames = [str(v).upper() for v in atoms.get("resname", [])]
    mol_ids = [int(v) for v in atoms.get("mol_id", [])]
    n_atoms = len(resnames)
    if len(mol_ids) != n_atoms:
        raise RuntimeError("spol requires atom_table()['mol_id'] to match atom count")
    selected_set = _selected_set(system, selection)
    allowed_resnames = {str(v).upper() for v in water_resnames}
    grouped: dict[int, list[int]] = {}
    for idx, mol in enumerate(mol_ids):
        grouped.setdefault(mol, []).append(idx)
    molecules = []
    for atom_indices in grouped.values():
        if selected_set is None:
            if not all(resnames[idx] in allowed_resnames for idx in atom_indices):
                continue
        else:
            selected_count = sum(idx in selected_set for idx in atom_indices)
            if selected_count == 0:
                continue
            if selected_count != len(atom_indices):
                raise ValueError("spol solvent_selection must contain whole molecules")
        molecules.append([int(idx) for idx in atom_indices])
    return molecules


def _residue_molecules(system, selection: Optional[MaskLike], water_resnames: Sequence[str]):
    atoms = system.atom_table()
    names = atoms.get("name", [])
    resnames = [str(v).upper() for v in atoms.get("resname", [])]
    resids = [int(v) for v in atoms.get("resid", [])]
    n_atoms = len(names)
    if len(resnames) != n_atoms or len(resids) != n_atoms:
        raise RuntimeError("spol requires atom_table() with at least name, resname, and resid")
    chain_ids = _chain_ids(atoms, n_atoms)
    selected_set = _selected_set(system, selection)
    allowed_resnames = {str(v).upper() for v in water_resnames}
    residue_atoms: dict[tuple[int, int, str], list[int]] = {}
    for idx in range(n_atoms):
        key = (chain_ids[idx], resids[idx], resnames[idx])
        residue_atoms.setdefault(key, []).append(idx)
    molecules = []
    for (_, _, resname), atom_indices in residue_atoms.items():
        if selected_set is None:
            if resname not in allowed_resnames:
                continue
        else:
            selected_count = sum(idx in selected_set for idx in atom_indices)
            if selected_count == 0:
                continue
            if selected_count != len(atom_indices):
                raise ValueError("spol solvent_selection must contain whole molecules")
        molecules.append([int(idx) for idx in atom_indices])
    return molecules
