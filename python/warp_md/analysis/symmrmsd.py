# Usage:
# from warp_md.analysis.symmrmsd import symmrmsd
# rms = symmrmsd(traj, system, mask="name CA", ref=0)

from __future__ import annotations

import itertools
from math import factorial
from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    kabsch_rmsd,
    load_native_symbol,
    native_inputs,
    native_selection,
    prepend_reference_frame,
    read_all_frames,
    reset_traj,
    rmsd_raw,
    selection_indices,
)
from .trajectory import ArrayTrajectory


RefLike = Union[int, str]


def _resolve_reference(coords: np.ndarray, system, ref: RefLike):
    n_frames = coords.shape[0]
    ref_coords = None
    if isinstance(ref, str):
        key = ref.strip().lower()
        if key in ("topology", "top", "topo"):
            if hasattr(system, "positions0"):
                ref_coords = system.positions0()
        if ref_coords is None and key in ("frame0", "first", "0", "topology", "top", "topo"):
            ref = 0

    if ref_coords is None:
        if isinstance(ref, int):
            ref_index = ref
            if ref_index < 0:
                ref_index = n_frames + ref_index
            if ref_index < 0 or ref_index >= n_frames:
                raise ValueError("ref index out of range")
            ref_coords = coords[ref_index]
        else:
            ref_coords = coords[0]
    else:
        ref_coords = np.asarray(ref_coords, dtype=np.float64)
    return ref_coords


def _infer_symmetry_groups(system, sel_indices: np.ndarray) -> Sequence[np.ndarray]:
    if not hasattr(system, "atom_table"):
        return []
    atoms = system.atom_table()
    names = atoms.get("name")
    resnames = atoms.get("resname")
    resids = atoms.get("resid")
    chains = atoms.get("chain_id")
    if not names or not resids:
        return []

    if resnames is None:
        resnames = [""] * len(names)
    if chains is None:
        chains = [0] * len(names)

    groups = {}
    for pos, atom_idx in enumerate(sel_indices.tolist()):
        key = (
            resids[int(atom_idx)],
            chains[int(atom_idx)],
            resnames[int(atom_idx)],
            names[int(atom_idx)],
        )
        groups.setdefault(key, []).append(pos)

    out = []
    for positions in groups.values():
        if len(positions) > 1:
            out.append(np.asarray(sorted(positions), dtype=np.int64))
    return out


def _normalize_symmetry_groups(
    symmetry_groups: Sequence[Sequence[int]],
    sel_indices: np.ndarray,
) -> Sequence[np.ndarray]:
    if not symmetry_groups:
        return []

    sel_pos = {int(v): i for i, v in enumerate(sel_indices.tolist())}
    groups = []
    used = set()
    for raw_group in symmetry_groups:
        group = [int(v) for v in raw_group]
        if len(group) < 2:
            continue
        if all(0 <= g < len(sel_indices) for g in group):
            positions = group
        elif all(g in sel_pos for g in group):
            positions = [sel_pos[g] for g in group]
        else:
            raise ValueError(
                "symmetry_groups entries must be selection positions or selected atom indices"
            )
        uniq = sorted(set(positions))
        if len(uniq) < 2:
            continue
        overlap = used.intersection(uniq)
        if overlap:
            raise ValueError(f"symmetry_groups overlap at selection positions: {sorted(overlap)}")
        used.update(uniq)
        groups.append(np.asarray(uniq, dtype=np.int64))
    return groups


def _permutation_budget(groups: Sequence[np.ndarray]) -> int:
    total = 1
    for g in groups:
        total *= factorial(int(g.size))
    return total


def _best_remapped_rmsd(
    cur: np.ndarray,
    ref: np.ndarray,
    groups: Sequence[np.ndarray],
    fit: bool,
    max_permutations: int,
) -> float:
    if not groups:
        return float(kabsch_rmsd(cur, ref) if fit else rmsd_raw(cur, ref))

    total = _permutation_budget(groups)
    if total > max_permutations:
        raise ValueError(
            f"symmetry remap requires {total} permutations, exceeds max_permutations={max_permutations}"
        )

    perm_sets = [list(itertools.permutations(g.tolist())) for g in groups]
    best = np.inf
    for combo in itertools.product(*perm_sets):
        trial = cur.copy()
        for positions, perm in zip(groups, combo):
            trial[positions] = cur[np.asarray(perm, dtype=np.int64)]
        val = float(kabsch_rmsd(trial, ref) if fit else rmsd_raw(trial, ref))
        if val < best:
            best = val
    return best


def _native_symmrmsd(
    traj,
    system,
    mask: str,
    ref: RefLike,
    fit: bool,
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("SymmRmsdPlan")
    if plan_cls is None:
        raise RuntimeError("SymmRmsdPlan binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native symmrmsd inputs")
    try:
        selection = native_selection(system, native_system, mask)
        ref_index = int(ref)
        run_traj = native_traj
        trim_first = False
        if ref_index == 0:
            if not reset_traj(run_traj):
                raise RuntimeError("failed to reset native trajectory")
        else:
            run_traj = prepend_reference_frame(native_traj, ref_index, chunk_frames)
            if run_traj is None:
                raise RuntimeError("failed to build native reference trajectory")
            trim_first = True
        plan = plan_cls(selection, reference="frame0", align=fit)
        values = np.asarray(
            plan.run(run_traj, native_system, chunk_frames=chunk_frames, device="auto"),
            dtype=np.float32,
        )
        if trim_first:
            values = values[1:]
        return values
    except Exception as exc:
        raise RuntimeError("native SymmRmsdPlan execution failed") from exc


def symmrmsd(
    traj,
    system,
    mask: str = "",
    ref: RefLike = 0,
    ref_mask: Optional[str] = None,
    fit: bool = True,
    remap: bool = False,
    symmetry_groups: Optional[Sequence[Sequence[int]]] = None,
    max_permutations: int = 4096,
    mass: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Symmetry-corrected RMSD with optional atom remapping."""
    if mass:
        raise ValueError("mass-weighted symmrmsd not supported yet")
    if max_permutations <= 0:
        raise ValueError("max_permutations must be positive")

    coords, _box, _time = read_all_frames(traj, chunk_frames, include_box=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32)

    idx = selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    if ref_mask is None:
        ref_mask = mask
    ref_idx = selection_indices(system, ref_mask)
    if ref_idx.size != idx.size:
        raise ValueError("mask and ref_mask selections must have same size")

    if (
        not remap
        and ref_mask == mask
        and frame_indices is None
        and isinstance(ref, int)
    ):
        source = ArrayTrajectory(np.asarray(coords, dtype=np.float32))
        return _native_symmrmsd(source, system, mask, ref, fit, chunk_frames)

    n_frames = coords.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        coords = coords[select]
        n_frames = coords.shape[0]

    ref_coords = _resolve_reference(coords, system, ref)
    ref_sel = ref_coords[ref_idx]

    if not remap:
        groups = []
    else:
        if symmetry_groups is not None:
            groups = _normalize_symmetry_groups(symmetry_groups, idx)
        else:
            groups = _infer_symmetry_groups(system, idx)

    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        cur = coords[i][idx].astype(np.float64, copy=False)
        out[i] = _best_remapped_rmsd(cur, ref_sel, groups, fit, max_permutations)
    return out.astype(np.float32)


__all__ = ["symmrmsd"]
