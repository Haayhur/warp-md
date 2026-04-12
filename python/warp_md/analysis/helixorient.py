from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md

from .h2order import _select


MaskLike = Union[str, Sequence[int], np.ndarray]

_HelixOrientPlan = (
    getattr(warp_md.traj_py, "PyHelixOrientPlan", None)
    if getattr(warp_md, "traj_py", None)
    else None
)


def helixorient(
    traj,
    system,
    ca_selection: MaskLike,
    sidechain_selection: Optional[MaskLike] = None,
    incremental: bool = False,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Local helix-axis geometry via Rust plan path."""
    if _HelixOrientPlan is None:
        raise RuntimeError(
            "PyHelixOrientPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    ca_sel = _select(system, ca_selection)
    sidechain_sel = None if sidechain_selection is None else _select(system, sidechain_selection)
    plan = _HelixOrientPlan(
        ca_sel,
        sidechain_selection=sidechain_sel,
        incremental=bool(incremental),
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
            "helixorient requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "labels": tuple(str(v) for v in out["labels"]),
        "time": np.asarray(out["time"], dtype=np.float32),
        "axis": np.asarray(out["axis"], dtype=np.float32),
        "center": np.asarray(out["center"], dtype=np.float32),
        "residue_vector": np.asarray(out["residue_vector"], dtype=np.float32),
        "normal": np.asarray(out["normal"], dtype=np.float32),
        "rise": np.asarray(out["rise"], dtype=np.float32),
        "radius": np.asarray(out["radius"], dtype=np.float32),
        "twist": np.asarray(out["twist"], dtype=np.float32),
        "bending": np.asarray(out["bending"], dtype=np.float32),
        "tilt": np.asarray(out["tilt"], dtype=np.float32),
        "rotation": np.asarray(out["rotation"], dtype=np.float32),
        "theta1": np.asarray(out["theta1"], dtype=np.float32),
        "theta2": np.asarray(out["theta2"], dtype=np.float32),
        "theta3": np.asarray(out["theta3"], dtype=np.float32),
        "frames": int(out["frames"]),
        "residues": int(out["residues"]),
        "use_sidechain": bool(out["use_sidechain"]),
        "incremental": bool(out["incremental"]),
        "used_box": bool(out["used_box"]),
        "length_scale": float(out["length_scale"]),
    }


__all__ = ["helixorient"]
