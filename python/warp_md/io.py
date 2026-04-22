from __future__ import annotations

import importlib
from typing import Any, Optional

try:
    _traj_py = importlib.import_module(f"{__package__}.traj_py")
except Exception as exc:  # pragma: no cover - bindings optional for pure-python tests
    _traj_py = None
    _TRAJ_PY_IMPORT_ERROR = exc
else:
    _TRAJ_PY_IMPORT_ERROR = None


def _require_traj_py() -> Any:
    if _traj_py is None:
        raise RuntimeError(
            "warp-md Python bindings are unavailable. Run `maturin develop` or install warp-md."
        ) from _TRAJ_PY_IMPORT_ERROR
    return _traj_py


def open_trajectory_auto(
    path: str,
    system: Any,
    *,
    format: Optional[str] = None,
    length_scale: Optional[float] = None,
) -> Any:
    """Open trajectory via native Rust format resolution and dispatch."""
    return _require_traj_py().open_trajectory_auto(
        path,
        system,
        format=format,
        length_scale=length_scale,
    )


def open_trajectory(
    path: str,
    system: Any,
    *,
    format: Optional[str] = None,
    length_scale: Optional[float] = None,
) -> Any:
    """Alias for :func:`open_trajectory_auto`."""
    return open_trajectory_auto(
        path=path,
        system=system,
        format=format,
        length_scale=length_scale,
    )
