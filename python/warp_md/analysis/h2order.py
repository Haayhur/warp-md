from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md


MaskLike = Union[str, Sequence[int], np.ndarray]

_H2OrderPlan = (
    getattr(warp_md.traj_py, "PyH2OrderPlan", None)
    if getattr(warp_md, "traj_py", None)
    else None
)

_DEFAULT_WATER_RESNAMES = ("HOH", "WAT", "SOL", "TIP3", "TIP3P", "SPC", "SPCE", "OPC")


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _select(system, selection: Optional[MaskLike]):
    if isinstance(selection, str) or selection is None:
        expr = selection
        if selection in ("", "*", "all", None):
            expr = _all_resid_mask(system) if hasattr(system, "atom_table") else "all"
        return system.select(expr)
    indices = np.asarray(selection, dtype=np.int64).reshape(-1).tolist()
    if hasattr(system, "select_indices"):
        return system.select_indices(indices)
    raise RuntimeError("system.select_indices is required for non-string h2order selections")


def _resolve_charges(system, charges: Optional[Sequence[float]]) -> list[float]:
    if charges is not None:
        values = np.asarray(charges, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError("charges must contain only finite values")
        return values.tolist()
    atoms = system.atom_table() if hasattr(system, "atom_table") else {}
    values = atoms.get("charge") if isinstance(atoms, dict) else None
    if values is None or len(values) == 0:
        raise RuntimeError(
            "h2order requires per-atom charges. Pass `charges=` or provide `atom_table()['charge']`."
        )
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError("system atom_table()['charge'] must contain only finite values")
    return arr.tolist()


def _chain_ids(atoms: dict, n_atoms: int) -> list[int]:
    if "chain_id" in atoms and len(atoms["chain_id"]) == n_atoms:
        return [int(v) for v in atoms["chain_id"]]
    if "chain" in atoms and len(atoms["chain"]) == n_atoms:
        chain_map: dict[str, int] = {}
        out = []
        for raw in atoms["chain"]:
            key = str(raw)
            if key not in chain_map:
                chain_map[key] = len(chain_map)
            out.append(chain_map[key])
        return out
    return [0] * n_atoms


def _water_triplets(
    system,
    selection: Optional[MaskLike],
    water_resnames: Sequence[str],
) -> tuple[list[int], list[int], list[int]]:
    atoms = system.atom_table()
    names = [str(v).upper() for v in atoms.get("name", [])]
    resnames = [str(v).upper() for v in atoms.get("resname", [])]
    resids = [int(v) for v in atoms.get("resid", [])]
    elements = [str(v).upper() for v in atoms.get("element", [])]
    n_atoms = len(names)
    if len(resnames) != n_atoms or len(resids) != n_atoms or len(elements) != n_atoms:
        raise RuntimeError("h2order requires atom_table() with name, resname, resid, and element")
    chain_ids = _chain_ids(atoms, n_atoms)
    selected_set = None
    if selection is not None:
        selected_set = {int(i) for i in _select(system, selection).indices}

    residue_atoms: dict[tuple[int, int, str], list[int]] = {}
    for idx in range(n_atoms):
        key = (chain_ids[idx], resids[idx], resnames[idx])
        residue_atoms.setdefault(key, []).append(idx)

    allowed_resnames = {str(v).upper() for v in water_resnames}
    oxy = []
    h1 = []
    h2 = []
    for key, atom_indices in residue_atoms.items():
        _, _, resname = key
        if selected_set is None:
            if resname not in allowed_resnames:
                continue
        elif not any(idx in selected_set for idx in atom_indices):
            continue

        oxygen = []
        hydrogens = []
        for idx in atom_indices:
            name = names[idx]
            element = elements[idx]
            if element == "O" or name in {"O", "OW", "OH2"}:
                oxygen.append(idx)
            elif element == "H" or name.startswith("H"):
                hydrogens.append(idx)
        if len(oxygen) != 1 or len(hydrogens) < 2:
            continue
        oxy.append(int(oxygen[0]))
        h1.append(int(hydrogens[0]))
        h2.append(int(hydrogens[1]))
    return oxy, h1, h2


def h2order(
    traj,
    system,
    selection: Optional[MaskLike] = None,
    charges: Optional[Sequence[float]] = None,
    axis: str = "z",
    bin: float = 0.25,
    n_slices: Optional[int] = None,
    length_scale: Optional[float] = None,
    water_resnames: Sequence[str] = _DEFAULT_WATER_RESNAMES,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Water dipole orientation profile via Rust plan path.

    Waters are auto-detected residue-by-residue from atom_table metadata.
    This v1 contract is planar only and does not expose `gmx h2order -nm`
    spherical micelle mode yet.
    """
    if _H2OrderPlan is None:
        raise RuntimeError(
            "PyH2OrderPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    oxygen_indices, hydrogen1_indices, hydrogen2_indices = _water_triplets(
        system,
        selection,
        water_resnames,
    )
    charge_list = _resolve_charges(system, charges)
    plan = _H2OrderPlan(
        oxygen_indices,
        hydrogen1_indices,
        hydrogen2_indices,
        charge_list,
        axis=axis,
        bin=bin,
        n_slices=n_slices,
        length_scale=length_scale,
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
            "h2order requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "coordinate": np.asarray(out["coordinate"], dtype=np.float32),
        "order": np.asarray(out["order"], dtype=np.float32),
        "dipole": np.asarray(out["dipole"], dtype=np.float32),
        "counts": np.asarray(out["counts"], dtype=np.uint64),
        "axis": str(out["axis"]),
        "bounds": np.asarray(out["bounds"], dtype=np.float32),
        "slice_width": float(out["slice_width"]),
        "n_frames": int(out["n_frames"]),
        "used_box": bool(out["used_box"]),
        "length_scale": float(out["length_scale"]),
        "dipole_unit": str(out["dipole_unit"]),
    }


__all__ = ["h2order"]
