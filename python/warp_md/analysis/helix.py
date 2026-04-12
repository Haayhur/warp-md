from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md

from .h2order import _select


MaskLike = Union[str, Sequence[int], np.ndarray]

_HelixPlan = (
    getattr(warp_md.traj_py, "PyHelixPlan", None) if getattr(warp_md, "traj_py", None) else None
)


def helix(
    traj,
    system,
    selection: MaskLike = "protein and backbone",
    fit: bool = True,
    check_each_frame: bool = False,
    residue_start: Optional[int] = None,
    residue_end: Optional[int] = None,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Helix fragment detection and fitted alpha-helix metrics via Rust plan path."""
    if _HelixPlan is None:
        raise RuntimeError(
            "PyHelixPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    sel = _select(system, selection)
    plan = _HelixPlan(
        sel,
        fit=bool(fit),
        check_each_frame=bool(check_each_frame),
        residue_start=None if residue_start is None else int(residue_start),
        residue_end=None if residue_end is None else int(residue_end),
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
            "helix requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "labels": tuple(str(v) for v in out["labels"]),
        "time": np.asarray(out["time"], dtype=np.float32),
        "fragment_start": np.asarray(out["fragment_start"], dtype=np.int32),
        "fragment_end": np.asarray(out["fragment_end"], dtype=np.int32),
        "radius": np.asarray(out["radius"], dtype=np.float32),
        "twist": np.asarray(out["twist"], dtype=np.float32),
        "rise": np.asarray(out["rise"], dtype=np.float32),
        "length": np.asarray(out["length"], dtype=np.float32),
        "dipole": np.asarray(out["dipole"], dtype=np.float32),
        "rmsd": np.asarray(out["rmsd"], dtype=np.float32),
        "ca_phi": np.asarray(out["ca_phi"], dtype=np.float32),
        "phi": np.asarray(out["phi"], dtype=np.float32),
        "psi": np.asarray(out["psi"], dtype=np.float32),
        "hb3": np.asarray(out["hb3"], dtype=np.float32),
        "hb4": np.asarray(out["hb4"], dtype=np.float32),
        "hb5": np.asarray(out["hb5"], dtype=np.float32),
        "ellipticity": np.asarray(out["ellipticity"], dtype=np.float32),
        "fragment_mask": np.asarray(out["fragment_mask"], dtype=bool),
        "residue_rmsd": np.asarray(out["residue_rmsd"], dtype=np.float32),
        "helicity_fraction": np.asarray(out["helicity_fraction"], dtype=np.float32),
        "jca_ha": np.asarray(out["jca_ha"], dtype=np.float32),
        "frames": int(out["frames"]),
        "residues": int(out["residues"]),
        "fit": bool(out["fit"]),
        "check_each_frame": bool(out["check_each_frame"]),
        "length_scale": float(out["length_scale"]),
        "used_box": bool(out["used_box"]),
    }


__all__ = ["helix"]
