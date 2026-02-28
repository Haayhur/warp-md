"""Export pack results using Rust backend."""

from __future__ import annotations

from typing import Optional


def export(
    result,
    fmt: str,
    path: str,
    scale: Optional[float] = None,
    *,
    add_box_sides: bool = False,
    box_sides_fix: float = 0.0,
    write_conect: bool = True,
    hexadecimal_indices: bool = False,
):
    fmt = fmt.lower()
    scale = 1.0 if scale is None else float(scale)
    try:
        from .. import traj_py  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "warp-pack export requires Rust bindings. Install warp-md or run `maturin develop`."
        ) from exc
    if not hasattr(traj_py, "pack_write_output"):
        raise RuntimeError(
            "warp-pack export requires Rust bindings. Install warp-md or run `maturin develop`."
        )
    return traj_py.pack_write_output(
        result,
        fmt,
        path,
        scale,
        bool(add_box_sides),
        float(box_sides_fix),
        bool(write_conect),
        bool(hexadecimal_indices),
    )
