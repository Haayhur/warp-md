from __future__ import annotations

import csv
from typing import Iterable, List, Mapping, Sequence


def group_indices(system, selection, group_by: str) -> List[List[int]]:
    atoms = system.atom_table()
    indices = selection.indices
    resid = atoms["resid"]
    chain_id = atoms["chain_id"]

    if group_by == "resid":
        keys = {idx: resid[idx] for idx in indices}
        grouped = {}
        for idx in indices:
            grouped.setdefault(keys[idx], []).append(idx)
    elif group_by == "chain":
        keys = {idx: chain_id[idx] for idx in indices}
        grouped = {}
        for idx in indices:
            grouped.setdefault(keys[idx], []).append(idx)
    elif group_by in ("resid_chain", "chain_resid"):
        keys = {idx: (chain_id[idx], resid[idx]) for idx in indices}
        grouped = {}
        for idx in indices:
            grouped.setdefault(keys[idx], []).append(idx)
    else:
        raise ValueError("group_by must be resid, chain, or resid_chain")

    return [grouped[key] for key in sorted(grouped.keys())]


def group_types_from_selections(
    system,
    selection,
    group_by: str,
    type_selections: Sequence[str],
) -> List[int]:
    groups = group_indices(system, selection, group_by)
    type_sets = [set(system.select(expr).indices) for expr in type_selections]
    group_types: List[int] = []
    for group in groups:
        assigned = None
        for t, sel_set in enumerate(type_sets):
            if any(idx in sel_set for idx in group):
                assigned = t
                break
        if assigned is None:
            raise ValueError("group has no matching type selection")
        group_types.append(assigned)
    return group_types


def charges_from_selections(
    system,
    selections: Iterable[Mapping[str, float]],
    default: float = 0.0,
) -> List[float]:
    charges = [default] * system.n_atoms()
    for entry in selections:
        expr = entry.get("selection")
        charge = entry.get("charge")
        if expr is None or charge is None:
            raise ValueError("each entry must have 'selection' and 'charge'")
        sel = system.select(expr)
        for idx in sel.indices:
            charges[idx] = float(charge)
    return charges


def charges_from_table(system, path: str, delimiter: str | None = None, default: float = 0.0) -> List[float]:
    with open(path, "r", newline="") as handle:
        sample = handle.read(1024)
        handle.seek(0)
        if delimiter is None:
            delimiter = "\t" if "\t" in sample and "," not in sample else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        headers = {h.lower(): h for h in reader.fieldnames or []}
        resname_key = headers.get("resname") or headers.get("residue") or headers.get("res")
        name_key = headers.get("name") or headers.get("atom") or headers.get("atom_name") or headers.get("atomname")
        charge_key = headers.get("charge") or headers.get("q")
        if resname_key is None or name_key is None or charge_key is None:
            raise ValueError("table must include resname, name/atom, and charge columns")
        rows = list(reader)

    entries = []
    for row in rows:
        resname = row[resname_key].strip()
        name = row[name_key].strip()
        charge = float(row[charge_key])
        expr = f"resname {resname} and name {name}"
        entries.append({"selection": expr, "charge": charge})
    return charges_from_selections(system, entries, default=default)
