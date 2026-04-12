from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

import warp_md

from .h2order import _DEFAULT_WATER_RESNAMES, _select, _water_triplets


MaskLike = Union[str, Sequence[int], np.ndarray]

_SOrientPlan = (
    getattr(warp_md.traj_py, "PySOrientPlan", None)
    if getattr(warp_md, "traj_py", None)
    else None
)


def _as_index_list(values: Sequence[int], label: str) -> list[int]:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"{label} must contain only non-negative atom indices")
    return arr.tolist()


def _resolve_triplets(
    system,
    solvent_selection: Optional[MaskLike],
    atom1_indices: Optional[Sequence[int]],
    atom2_indices: Optional[Sequence[int]],
    atom3_indices: Optional[Sequence[int]],
    water_resnames: Sequence[str],
) -> tuple[list[int], list[int], list[int]]:
    explicit = [atom1_indices, atom2_indices, atom3_indices]
    if any(value is not None for value in explicit):
        if any(value is None for value in explicit):
            raise ValueError(
                "atom1_indices, atom2_indices, and atom3_indices must be provided together"
            )
        atom1 = _as_index_list(atom1_indices, "atom1_indices")
        atom2 = _as_index_list(atom2_indices, "atom2_indices")
        atom3 = _as_index_list(atom3_indices, "atom3_indices")
        if len(atom1) != len(atom2) or len(atom1) != len(atom3):
            raise ValueError("explicit solvent triplet vectors must have identical length")
        return atom1, atom2, atom3
    return _water_triplets(system, solvent_selection, water_resnames)


def sorient(
    traj,
    system,
    solute_selection: MaskLike,
    solvent_selection: Optional[MaskLike] = None,
    atom1_indices: Optional[Sequence[int]] = None,
    atom2_indices: Optional[Sequence[int]] = None,
    atom3_indices: Optional[Sequence[int]] = None,
    r_min: float = 0.0,
    r_max: float = 0.5,
    cbin: float = 0.02,
    rbin: float = 0.02,
    use_com: bool = False,
    use_vector23: bool = False,
    r_profile_max: Optional[float] = None,
    length_scale: Optional[float] = None,
    water_resnames: Sequence[str] = _DEFAULT_WATER_RESNAMES,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Radial solvent orientation around solute reference positions via Rust plan path.

    v1 auto-detects 3-atom water-like solvent triplets residue-by-residue, or
    accepts explicit `atom1_indices`/`atom2_indices`/`atom3_indices`.
    """
    if _SOrientPlan is None:
        raise RuntimeError(
            "PySOrientPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    solute_sel = _select(system, solute_selection)
    atom1, atom2, atom3 = _resolve_triplets(
        system,
        solvent_selection,
        atom1_indices,
        atom2_indices,
        atom3_indices,
        water_resnames,
    )
    plan = _SOrientPlan(
        solute_sel,
        atom1,
        atom2,
        atom3,
        r_min=r_min,
        r_max=r_max,
        cbin=cbin,
        rbin=rbin,
        use_com=use_com,
        use_vector23=use_vector23,
        r_profile_max=r_profile_max,
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
            "sorient requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return {
        "cos_theta1": np.asarray(out["cos_theta1"], dtype=np.float32),
        "cos_theta1_distribution": np.asarray(
            out["cos_theta1_distribution"], dtype=np.float32
        ),
        "abs_cos_theta2": np.asarray(out["abs_cos_theta2"], dtype=np.float32),
        "abs_cos_theta2_distribution": np.asarray(
            out["abs_cos_theta2_distribution"], dtype=np.float32
        ),
        "r": np.asarray(out["r"], dtype=np.float32),
        "mean_cos_theta1": np.asarray(out["mean_cos_theta1"], dtype=np.float32),
        "mean_p2_theta2": np.asarray(out["mean_p2_theta2"], dtype=np.float32),
        "cumulative_r": np.asarray(out["cumulative_r"], dtype=np.float32),
        "cumulative_cos_theta1": np.asarray(out["cumulative_cos_theta1"], dtype=np.float32),
        "cumulative_p2_theta2": np.asarray(out["cumulative_p2_theta2"], dtype=np.float32),
        "count_density": np.asarray(out["count_density"], dtype=np.float32),
        "counts": np.asarray(out["counts"], dtype=np.uint64),
        "window_count": int(out["window_count"]),
        "average_shell_size": float(out["average_shell_size"]),
        "window_mean_cos_theta1": float(out["window_mean_cos_theta1"]),
        "window_mean_p2_theta2": float(out["window_mean_p2_theta2"]),
        "r_window": np.asarray(out["r_window"], dtype=np.float32),
        "cbin": float(out["cbin"]),
        "rbin": float(out["rbin"]),
        "r_profile_max": float(out["r_profile_max"]),
        "use_vector23": bool(out["use_vector23"]),
        "use_com": bool(out["use_com"]),
        "n_frames": int(out["n_frames"]),
        "n_reference_positions": int(out["n_reference_positions"]),
        "used_box": bool(out["used_box"]),
        "length_scale": float(out["length_scale"]),
    }


__all__ = ["sorient"]
