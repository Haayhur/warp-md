# Usage:
# from warp_md.analysis.surf import surf, molsurf
# areas = surf(traj, system, mask=":1-10")

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import coerce_native_system, is_native_traj, materialize_selection

MaskLike = Union[str, Sequence[int], np.ndarray]
RadiiLike = Union[str, Sequence[float], np.ndarray]


def _normalize_algorithm(value: str) -> str:
    mode = str(value).lower()
    if mode not in ("lcpo", "sasa", "bbox", "auto"):
        raise ValueError("algorithm must be 'lcpo', 'sasa', 'bbox', or 'auto'")
    return mode


def _normalize_probe_radius(value: float) -> float:
    radius = float(value)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError("probe_radius must be a finite value >= 0")
    return radius


def _normalize_offset(value: Optional[float], algorithm: str) -> float:
    if value is None:
        return 1.4 if algorithm == "lcpo" else 0.0
    offset = float(value)
    if not np.isfinite(offset) or offset < 0.0:
        raise ValueError("offset must be a finite value >= 0")
    return offset


def _normalize_nbrcut(value: Optional[float]) -> float:
    if value is None:
        return 2.5
    cutoff = float(value)
    if not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("nbrcut must be a finite value > 0")
    return cutoff


def _normalize_n_sphere_points(value: int) -> int:
    points = int(value)
    if points <= 0:
        raise ValueError("n_sphere_points must be a positive integer")
    return points


def _normalize_radii_mode(value: str) -> str:
    mode = str(value).lower()
    if mode not in ("gb", "parse", "vdw"):
        raise ValueError("radii_mode must be 'gb', 'parse', or 'vdw'")
    return mode


def _normalize_radii(radii: Optional[RadiiLike]) -> Optional[list[float]]:
    if radii is None:
        return None
    if isinstance(radii, str):
        return None
    arr = np.asarray(radii, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return []
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("radii values must be finite and > 0")
    return arr.tolist()


def _normalize_radii_input(
    radii: Optional[RadiiLike], radii_mode: str
) -> tuple[Optional[list[float]], str]:
    if isinstance(radii, str):
        return None, _normalize_radii_mode(radii)
    return _normalize_radii(radii), _normalize_radii_mode(radii_mode)


def _select_mask(system, mask: MaskLike):
    return materialize_selection(system, mask, allow_at_indices=True)


def _require_native_surface_inputs(traj, system, name: str):
    if not is_native_traj(traj):
        raise RuntimeError(
            f"{name} requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError(f"failed to prepare native {name} system")
    return native_system


def _format_surface_output(
    raw,
    key: str,
    dtype: str,
    atom_area_requested: bool,
    volume_requested: bool,
    residue_area_requested: bool = False,
):
    if isinstance(raw, tuple) and len(raw) == 7:
        total, atom_area, volume, residue_area, residue_ids, frames, atoms = raw
        out = np.asarray(total, dtype=np.float32)
        frames = int(frames)
        atoms = int(atoms)
        if (
            str(dtype).lower() == "dict"
            or atom_area_requested
            or volume_requested
            or residue_area_requested
        ):
            result = {key: out, "frames": frames, "atoms": atoms}
            if atom_area is not None:
                result["atom_area"] = np.asarray(atom_area, dtype=np.float32).reshape(
                    (frames, atoms)
                )
            if volume is not None:
                result["volume"] = np.asarray(volume, dtype=np.float32)
            if residue_area is not None:
                residue_ids_arr = np.asarray(residue_ids, dtype=np.int32).reshape(-1)
                result["residue_ids"] = residue_ids_arr
                result["residue_area"] = np.asarray(residue_area, dtype=np.float32).reshape(
                    (frames, residue_ids_arr.size)
                )
            return result
        return out
    if isinstance(raw, tuple) and len(raw) == 5:
        total, atom_area, volume, frames, atoms = raw
        out = np.asarray(total, dtype=np.float32)
        frames = int(frames)
        atoms = int(atoms)
        if str(dtype).lower() == "dict" or atom_area_requested or volume_requested:
            result = {key: out, "frames": frames, "atoms": atoms}
            if atom_area is not None:
                result["atom_area"] = np.asarray(atom_area, dtype=np.float32).reshape(
                    (frames, atoms)
                )
            if volume is not None:
                result["volume"] = np.asarray(volume, dtype=np.float32)
            return result
        return out
    if isinstance(raw, tuple) and len(raw) == 4:
        total, atom_area, frames, atoms = raw
        out = np.asarray(total, dtype=np.float32)
        frames = int(frames)
        atoms = int(atoms)
        atom_area_arr = np.asarray(atom_area, dtype=np.float32).reshape((frames, atoms))
        if str(dtype).lower() == "dict" or atom_area_requested or volume_requested:
            return {
                key: out,
                "atom_area": atom_area_arr,
                "frames": frames,
                "atoms": atoms,
            }
        return out
    out = np.asarray(raw, dtype=np.float32)
    if str(dtype).lower() == "dict":
        return {key: out}
    return out


def surf(
    traj,
    system,
    mask: MaskLike = "",
    dtype: str = "ndarray",
    algorithm: str = "lcpo",
    probe_radius: float = 1.4,
    offset: Optional[float] = None,
    nbrcut: Optional[float] = None,
    solutemask: MaskLike = "",
    n_sphere_points: int = 64,
    radii: Optional[RadiiLike] = None,
    radii_mode: str = "gb",
    frame_indices: Optional[Sequence[int]] = None,
    top=None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
    atom_area: bool = False,
    volume: bool = False,
    residue_area: bool = False,
):
    """Surface area estimate.

    algorithm:
      - "lcpo": LCPO surface
      - "sasa": Shrake-Rupley style SASA approximation
      - "bbox": legacy bounding-box area
      - "auto": use plan default (`lcpo`)
    """
    del top
    algorithm = _normalize_algorithm(algorithm)
    plan_algorithm = "lcpo" if algorithm == "auto" else algorithm
    probe_radius = _normalize_probe_radius(probe_radius)
    offset_value = _normalize_offset(offset, plan_algorithm)
    nbrcut_value = _normalize_nbrcut(nbrcut)
    n_sphere_points = _normalize_n_sphere_points(n_sphere_points)
    atom_area = bool(atom_area)
    volume = bool(volume)
    residue_area = bool(residue_area)
    if (atom_area or volume or residue_area) and plan_algorithm != "sasa":
        raise ValueError("surface detail output requires algorithm='sasa'")
    native_system = _require_native_surface_inputs(traj, system, "surf")

    try:
        from warp_md import SurfPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "SurfPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(SurfPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "SurfPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel = _select_mask(native_system, mask)
    if solutemask in ("", None):
        solute_sel = None
    else:
        solute_sel = _select_mask(native_system, solutemask)
    radii_list, radii_mode_value = _normalize_radii_input(radii, radii_mode)
    plan_kwargs = {
        "algorithm": plan_algorithm,
        "probe_radius": probe_radius,
        "n_sphere_points": n_sphere_points,
        "radii": radii_list,
        "offset": offset_value,
        "nbrcut": nbrcut_value,
        "solute_selection": solute_sel,
        "radii_mode": radii_mode_value,
    }
    if atom_area:
        plan_kwargs["atom_area"] = True
    if volume:
        plan_kwargs["volume"] = True
    if residue_area:
        plan_kwargs["residue_area"] = True
    plan = SurfPlan(sel, **plan_kwargs)
    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    try:
        raw = plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices_list,
        )
    except TypeError as exc:
        raise RuntimeError(
            "surf requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return _format_surface_output(raw, "surf", dtype, atom_area, volume, residue_area)


def molsurf(
    traj,
    system,
    mask: MaskLike = "",
    dtype: str = "ndarray",
    algorithm: str = "sasa",
    probe: float = 1.4,
    probe_radius: Optional[float] = None,
    offset: Optional[float] = 0.0,
    solutemask: MaskLike = "",
    n_sphere_points: int = 64,
    radii: Optional[RadiiLike] = None,
    radii_mode: str = "gb",
    frame_indices: Optional[Sequence[int]] = None,
    top=None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
    atom_area: bool = False,
    volume: bool = False,
    residue_area: bool = False,
):
    radius = float(probe if probe_radius is None else probe_radius)
    del top
    algorithm = _normalize_algorithm(algorithm)
    if algorithm == "auto":
        algorithm = "sasa"
    if algorithm == "lcpo":
        raise ValueError("molsurf algorithm must be 'sasa', 'bbox', or 'auto'")
    radius = _normalize_probe_radius(radius)
    offset_value = _normalize_offset(offset, "sasa")
    n_sphere_points = _normalize_n_sphere_points(n_sphere_points)
    radii_list, radii_mode_value = _normalize_radii_input(radii, radii_mode)
    atom_area = bool(atom_area)
    volume = bool(volume)
    residue_area = bool(residue_area)
    if (atom_area or volume or residue_area) and algorithm != "sasa":
        raise ValueError("surface detail output requires algorithm='sasa'")
    native_system = _require_native_surface_inputs(traj, system, "molsurf")

    try:
        from warp_md import MolSurfPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "MolSurfPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(MolSurfPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "MolSurfPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    sel = _select_mask(native_system, mask)
    solute_sel = None if solutemask in ("", None) else _select_mask(native_system, solutemask)
    plan_kwargs = {
        "algorithm": algorithm,
        "probe_radius": radius,
        "n_sphere_points": n_sphere_points,
        "radii": radii_list,
        "offset": offset_value,
        "radii_mode": radii_mode_value,
    }
    if solute_sel is not None:
        plan_kwargs["solute_selection"] = solute_sel
    if atom_area:
        plan_kwargs["atom_area"] = True
    if volume:
        plan_kwargs["volume"] = True
    if residue_area:
        plan_kwargs["residue_area"] = True
    plan = MolSurfPlan(sel, **plan_kwargs)
    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    try:
        raw = plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=frame_indices_list,
        )
    except TypeError as exc:
        raise RuntimeError(
            "molsurf requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc
    return _format_surface_output(raw, "molsurf", dtype, atom_area, volume, residue_area)


def sasa(*args, **kwargs):
    kwargs.setdefault("algorithm", "sasa")
    kwargs.setdefault("offset", 0.0)
    return surf(*args, **kwargs)


__all__ = ["surf", "molsurf", "sasa"]
