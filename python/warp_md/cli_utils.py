from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .builder import charges_from_selections, charges_from_table, group_types_from_selections
from .cli_api import System


def _as_tuple(value: Any, size: int, label: str) -> Optional[tuple[Any, ...]]:
    if value is None:
        return None
    if isinstance(value, tuple):
        if len(value) != size:
            raise ValueError(f"{label} must have length {size}")
        return value
    if isinstance(value, list):
        if len(value) != size:
            raise ValueError(f"{label} must have length {size}")
        return tuple(value)
    raise ValueError(f"{label} must be a list/tuple of length {size}")


def _pick(spec: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {k: spec[k] for k in keys if k in spec and spec[k] is not None}


def _resolve_charges(system: System, spec: Any) -> list[float]:
    if isinstance(spec, list):
        return [float(x) for x in spec]
    if isinstance(spec, dict):
        mode = spec.get("from")
        default = spec.get("default", 0.0)
        if mode == "table":
            path = spec.get("path")
            if not path:
                raise ValueError("charges.from=table requires path")
            return charges_from_table(system, path, delimiter=spec.get("delimiter"), default=default)
        if mode == "selections":
            entries = spec.get("entries")
            if not entries:
                raise ValueError("charges.from=selections requires entries")
            return charges_from_selections(system, entries, default=default)
    raise ValueError("charges must be a list or {from: table|selections}")


def _resolve_group_types(
    system: System,
    selection,
    group_by: str,
    spec: Any,
) -> Optional[list[int]]:
    if spec is None:
        return None
    if isinstance(spec, list):
        return [int(x) for x in spec]
    if isinstance(spec, dict):
        if spec.get("from") != "selections":
            raise ValueError("group_types.from must be selections")
        type_selections = spec.get("type_selections")
        if not type_selections:
            raise ValueError("group_types.type_selections required")
        sel_expr = spec.get("selection")
        sel = selection if sel_expr is None else system.select(sel_expr)
        group_by = spec.get("group_by", group_by)
        return group_types_from_selections(system, sel, group_by, type_selections)
    raise ValueError("group_types must be a list or {from: selections}")
