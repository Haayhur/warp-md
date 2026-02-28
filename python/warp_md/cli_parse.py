from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from .builder import charges_from_selections, charges_from_table, group_types_from_selections
from .cli_api import System, Trajectory, _load_system, _load_trajectory


def _split_values(raw: str) -> list[str]:
    if "," in raw:
        parts = [part.strip() for part in raw.split(",")]
    else:
        parts = raw.split()
    return [part for part in parts if part]


def _parse_float_tuple(raw: str, size: int, label: str) -> tuple[float, ...]:
    values = _split_values(raw)
    if len(values) != size:
        raise ValueError(f"{label} must have {size} values")
    return tuple(float(v) for v in values)


def _parse_int_tuple(raw: str, size: int, label: str) -> tuple[int, ...]:
    values = _split_values(raw)
    if len(values) != size:
        raise ValueError(f"{label} must have {size} values")
    return tuple(int(v) for v in values)


def _parse_int_list(raw: str, label: str) -> list[int]:
    values = _split_values(raw)
    if not values:
        raise ValueError(f"{label} must have at least one value")
    return [int(v) for v in values]


def _parse_json_list(raw: str, label: str) -> list[Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON") from exc
    if not isinstance(data, list):
        raise ValueError(f"{label} must be a JSON list")
    return data


def _parse_charges_arg(raw: str, system: System) -> list[float]:
    if raw.startswith("table:"):
        path = raw[len("table:") :].strip()
        if not path:
            raise ValueError("charges table path is required")
        return charges_from_table(system, path)
    if raw.startswith("selections:"):
        payload = raw[len("selections:") :].strip()
        entries = _parse_json_list(payload, "charges selections")
        return charges_from_selections(system, entries)
    data = _parse_json_list(raw, "charges")
    return [float(x) for x in data]


def _parse_group_types_arg(
    raw: Optional[str],
    system: System,
    selection,
    group_by: str,
) -> Optional[list[int]]:
    if raw is None:
        return None
    if raw.startswith("selections:"):
        payload = raw[len("selections:") :].strip()
        if payload.startswith("["):
            selections = _parse_json_list(payload, "group_types selections")
        else:
            selections = [s.strip() for s in payload.split(",") if s.strip()]
        if not selections:
            raise ValueError("group_types selections cannot be empty")
        return group_types_from_selections(system, selection, group_by, selections)
    data = _parse_json_list(raw, "group_types")
    return [int(x) for x in data]


def _infer_format(path: str) -> str:
    return Path(path).suffix.lower().lstrip(".")


def _load_system_from_args(args: argparse.Namespace) -> System:
    fmt = args.topology_format or _infer_format(args.topology)
    spec = {"path": args.topology, "format": fmt}
    return _load_system(spec)


def _load_traj_from_args(args: argparse.Namespace, system: System) -> Trajectory:
    fmt = args.traj_format or _infer_format(args.traj)
    spec = {
        "path": args.traj,
        "format": fmt,
        "length_scale": args.traj_length_scale,
    }
    return _load_trajectory(spec, system)
