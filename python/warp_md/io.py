from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


_SUPPORTED_TRAJ_FORMATS = {"dcd", "xtc", "pdb", "pdbqt"}


def _resolve_traj_format(path: str, fmt: Optional[str]) -> str:
    token = str(fmt).strip().lower() if fmt is not None else ""
    if token in {"", "auto"}:
        token = Path(path).suffix.lower().lstrip(".")
    if not token:
        raise ValueError(
            "trajectory format could not be inferred; set format explicitly (dcd/xtc/pdb/pdbqt)"
        )
    if token not in _SUPPORTED_TRAJ_FORMATS:
        raise ValueError(
            f"unsupported trajectory format '{token}'; expected one of: dcd, xtc, pdb, pdbqt"
        )
    return token


def open_trajectory_auto(
    path: str,
    system: Any,
    *,
    format: Optional[str] = None,
    length_scale: Optional[float] = None,
    trajectory_cls: Optional[Any] = None,
) -> Any:
    """Open trajectory with automatic format detection from extension.

    If `format` is provided, it overrides extension inference.
    """
    if trajectory_cls is None:
        from . import Trajectory as trajectory_cls

    fmt = _resolve_traj_format(path=path, fmt=format)
    if fmt == "dcd":
        return trajectory_cls.open_dcd(path, system, length_scale=length_scale)
    if fmt == "xtc":
        return trajectory_cls.open_xtc(path, system)
    if fmt in {"pdb", "pdbqt"}:
        return trajectory_cls.open_pdb(path, system)
    raise ValueError(f"unsupported trajectory format '{fmt}'")


def open_trajectory(
    path: str,
    system: Any,
    *,
    format: Optional[str] = None,
    length_scale: Optional[float] = None,
    trajectory_cls: Optional[Any] = None,
) -> Any:
    """Alias for :func:`open_trajectory_auto`."""
    return open_trajectory_auto(
        path=path,
        system=system,
        format=format,
        length_scale=length_scale,
        trajectory_cls=trajectory_cls,
    )

