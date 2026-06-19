from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_selection,
)


MaskLike = Union[str, Sequence[int], np.ndarray, None]


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def kabsch_sander(
    traj,
    system,
    selection: MaskLike = "protein",
    *,
    energy_cutoff: float = -0.5,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "dict",
    device: str = "auto",
):
    """Compute backbone CO/NH hydrogen-bond energies in Rust."""
    if not is_native_traj(traj):
        raise RuntimeError(
            "kabsch_sander requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for kabsch_sander")
    plan_cls = load_native_symbol("KabschSanderPlan")
    if plan_cls is None:
        raise RuntimeError(
            "KabschSanderPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )

    sel = native_selection(system, native_system, selection, allow_at_indices=True)
    plan = plan_cls(sel)
    result = plan.run(
        traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=_frame_indices_arg(frame_indices),
    )

    labels = np.asarray(result["labels"], dtype="U16")
    flat = np.asarray(result["energies"], dtype=np.float32)
    rows = int(result["rows"])
    cols = int(result["cols"])
    n_res = int(labels.shape[0])
    if cols != n_res * n_res or flat.size != rows * cols:
        raise RuntimeError("native KabschSanderPlan returned unexpected output")

    energy = flat.reshape((rows, n_res, n_res))
    cutoff = float(energy_cutoff)
    hbonds = np.isfinite(energy) & (energy < cutoff)
    out = {
        "residues": labels,
        "energy": energy,
        "hbonds": hbonds,
        "energy_cutoff": cutoff,
    }
    key = str(dtype).lower()
    if key in ("dict", "mapping"):
        return out
    if key in ("energy", "energies", "ndarray", "array"):
        return energy
    if key in ("hbonds", "contacts", "mask"):
        return hbonds
    if key == "tuple":
        return labels, energy, hbonds
    raise ValueError("dtype must be 'dict', 'energy', 'hbonds', or 'tuple'")


__all__ = ["kabsch_sander"]
