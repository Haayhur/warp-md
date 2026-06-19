from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_selection,
)


def _format(dtype: str, centers, values, counts, integral=None):
    key = str(dtype).lower()
    centers = np.asarray(centers, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.uint64)
    integral_values = None
    if integral is not None:
        integral_values = np.asarray(integral, dtype=np.float32)
    if key == "dict":
        out = {"bin_centers": centers, "rdf": values, "counts": counts}
        if integral_values is not None:
            out["integral_rdf"] = integral_values
        return out
    if key in ("full", "tuple_full"):
        if integral_values is not None:
            return centers, values, counts, integral_values
        return centers, values, counts
    return centers, values


def _normalize_rdf_output_mode(norm, raw_rdf: bool, volume: bool):
    number_density = False
    if norm is None:
        return bool(raw_rdf), bool(volume), number_density
    key = str(norm).strip().lower().replace("-", "_").replace(" ", "_")
    if key in ("rdf", "g_r", "gr"):
        return False, bool(volume), number_density
    if key in ("none", "raw", "count", "counts"):
        return True, False, number_density
    if key in ("volume", "box", "box_volume"):
        return False, True, number_density
    if key in ("density", "number_density", "numberdensity"):
        return False, False, True
    raise ValueError("norm must be 'rdf', 'volume', 'number_density', or 'none'")


def rdf(
    traj,
    system,
    solvent_mask=":WAT@O",
    solute_mask="",
    maximum: float = 10.0,
    bin_spacing: float = 0.5,
    image: bool = True,
    density: Optional[float] = 0.033456,
    volume: bool = False,
    center_solvent: bool = False,
    center_solute: bool = False,
    intramol: bool = True,
    frame_indices: Optional[Sequence[int]] = None,
    top=None,
    raw_rdf: bool = False,
    norm: Optional[str] = None,
    dtype: str = "tuple",
    byres1: bool = False,
    byres2: bool = False,
    bymol1: bool = False,
    bymol2: bool = False,
    mass: bool = True,
    intrdf: bool = False,
    dimension: str = "3d",
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Compute RDF via Rust ``RdfPlan``.

    Rust owns the heavy loop for atom-pair binning and output normalization.
    Unsupported options raise explicitly.
    """
    del top
    maximum = float(maximum)
    bin_spacing = float(bin_spacing)
    if maximum <= 0.0:
        raise ValueError("maximum must be positive")
    if bin_spacing <= 0.0:
        raise ValueError("bin_spacing must be positive")
    density_value = 0.033456 if density is None else float(density)
    if not np.isfinite(density_value) or density_value <= 0.0:
        raise ValueError("density must be finite and positive")
    dimension_value = str(dimension).lower()
    if dimension_value not in ("3d", "xyz", "xy"):
        raise ValueError("dimension must be '3d' or 'xy'")
    raw_rdf_value, volume_value, number_density = _normalize_rdf_output_mode(
        norm, raw_rdf, volume
    )

    plan_cls = load_native_symbol("RdfPlan")
    if plan_cls is None:
        raise RuntimeError(
            "RdfPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )

    bins = max(1, int(np.ceil(maximum / bin_spacing)))
    r_max = np.float32(bins * bin_spacing)
    if not is_native_traj(traj):
        raise RuntimeError(
            "rdf requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native RDF system")

    pbc = "orthorhombic" if image else "none"
    sel_a = native_selection(system, native_system, solvent_mask, allow_at_indices=True)
    sel_b = (
        sel_a
        if solute_mask in ("", None)
        else native_selection(system, native_system, solute_mask, allow_at_indices=True)
    )
    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    plan = plan_cls(
        sel_a,
        sel_b,
        bins=bins,
        r_max=r_max,
        pbc=pbc,
        center1=bool(center_solvent),
        center2=bool(center_solute),
        byres1=bool(byres1),
        byres2=bool(byres2),
        bymol1=bool(bymol1),
        bymol2=bool(bymol2),
        no_intramol=not bool(intramol),
        mass_weighted=bool(mass),
        density=density_value,
        volume=volume_value,
        raw_rdf=raw_rdf_value,
        intrdf=bool(intrdf),
        dimension=dimension_value,
        number_density=number_density,
    )
    result = plan.run(
        traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=frame_indices_list,
    )
    if len(result) == 4:
        centers, values, counts, integral = result
    else:
        centers, values, counts = result
        integral = None
    centers = np.asarray(centers, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    counts = np.asarray(counts, dtype=np.uint64)
    return _format(dtype, centers, values, counts, integral)


def radial(*args, **kwargs):
    return rdf(*args, **kwargs)


__all__ = ["rdf", "radial"]
