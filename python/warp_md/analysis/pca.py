# Usage:
# from warp_md.analysis.pca import pca, projection
# proj, (evals, evecs) = pca(traj, system, mask="name CA", n_vecs=2)

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ._runtime import (
    clone_native_system_with_positions0,
    load_native_symbol,
    mass_weights,
    native_inputs,
    native_selection,
    read_all_frames,
    reset_traj,
    selection_indices,
)
from .align import superpose
from .structure import mean_structure
from .trajectory import ArrayTrajectory


def _normalize_projection_vectors(eigenvectors: np.ndarray, n_features: int) -> np.ndarray:
    vecs = np.asarray(eigenvectors, dtype=np.float64)
    if vecs.ndim != 2:
        raise ValueError("eigenvectors must be 2D")
    if vecs.shape[1] == n_features:
        return vecs
    if vecs.shape[0] == n_features:
        return vecs.T
    raise ValueError("eigenvectors shape does not match features")


def _projection_mean(
    system,
    mask: str,
    average_coords: Optional[np.ndarray],
    mass_weighted: bool,
) -> Optional[np.ndarray]:
    if average_coords is None:
        return None
    avg = np.asarray(average_coords, dtype=np.float64)
    if avg.ndim != 2 or avg.shape[1] != 3:
        raise ValueError("average_coords must be (n_atoms, 3)")
    idx = selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")
    mean = avg[idx, :].reshape(-1)
    if mass_weighted:
        weights = np.sqrt(np.clip(mass_weights(system, idx, True), 0.0, None))
        mean = mean * np.repeat(weights, 3)
    return mean


def _run_native_pca(
    traj,
    system,
    mask: str,
    n_components: int,
    chunk_frames: Optional[int],
):
    plan_cls = load_native_symbol("PcaPlan")
    if plan_cls is None:
        raise RuntimeError("PcaPlan binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native PCA inputs")
    try:
        plan = plan_cls(native_selection(system, native_system, mask), n_components, mass_weighted=False)
        out = plan.run(native_traj, native_system, chunk_frames=chunk_frames, device="auto")
    except Exception as exc:
        raise RuntimeError("native PcaPlan execution failed") from exc
    evals = np.asarray(out["eigenvalues"], dtype=np.float32)
    evecs = np.asarray(out["eigenvectors"], dtype=np.float32)
    return evals, evecs


def projection(
    traj,
    system,
    mask: str = "",
    eigenvectors: Optional[np.ndarray] = None,
    eigenvalues: Optional[np.ndarray] = None,
    scalar_type: str = "covar",
    average_coords: Optional[np.ndarray] = None,
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Project trajectory onto provided eigenvectors.

    Returns array shape (n_vecs, n_frames).
    """
    _ = eigenvalues
    _ = dtype
    if eigenvectors is None:
        raise ValueError("eigenvectors are required")
    mode = scalar_type.lower()
    if mode not in {"covar", "mwcovar"}:
        raise ValueError("scalar_type must be 'covar' or 'mwcovar'")

    idx = selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")

    mass_weighted = mode == "mwcovar"
    mean = _projection_mean(system, mask, average_coords, mass_weighted)
    vecs = _normalize_projection_vectors(np.asarray(eigenvectors), idx.size * 3)

    plan_cls = load_native_symbol("ProjectionPlan")
    if plan_cls is None:
        raise RuntimeError("ProjectionPlan binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native projection inputs")
    try:
        plan = plan_cls(
            native_selection(system, native_system, mask),
            vecs.reshape(-1).tolist(),
            int(vecs.shape[0]),
            int(vecs.shape[1]),
            None if mean is None else mean.tolist(),
            mass_weighted=mass_weighted,
        )
        values = np.asarray(
            plan.run(
                native_traj,
                native_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=None if frame_indices is None else [int(i) for i in frame_indices],
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError("native ProjectionPlan execution failed") from exc
    return values.T.astype(np.float32, copy=False)


def _fit_to_reference(
    traj,
    system,
    align_mask: str,
    ref,
    ref_mask: Optional[str],
    chunk_frames: Optional[int],
):
    if ref is None:
        first_fit = superpose(
            traj,
            system,
            mask=align_mask,
            ref=0,
            ref_mask=align_mask,
            chunk_frames=chunk_frames,
        )
        mean_coords = mean_structure(
            first_fit,
            system,
            mask="all",
            dtype="frame",
            chunk_frames=chunk_frames,
        )
        mean_system = clone_native_system_with_positions0(system, mean_coords)
        if mean_system is None:
            raise RuntimeError("failed to build native mean-reference system")
        reset_traj(first_fit)
        return (
            superpose(
                first_fit,
                mean_system,
                mask=align_mask,
                ref="topology",
                ref_mask=ref_mask,
                chunk_frames=chunk_frames,
            ),
            mean_system,
        )

    if isinstance(ref, (int, str)):
        return (
            superpose(
                traj,
                system,
                mask=align_mask,
                ref=ref,
                ref_mask=ref_mask,
                chunk_frames=chunk_frames,
            ),
            system,
        )

    ref_coords = np.asarray(ref, dtype=np.float32)
    if ref_coords.ndim != 2 or ref_coords.shape[1] != 3:
        raise ValueError("ref coords must be (n_atoms, 3)")
    ref_system = clone_native_system_with_positions0(system, ref_coords)
    if ref_system is None:
        raise RuntimeError("failed to build native reference system")
    return (
        superpose(
            traj,
            ref_system,
            mask=align_mask,
            ref="topology",
            ref_mask=ref_mask,
            chunk_frames=chunk_frames,
        ),
        ref_system,
    )


def _projection_ready_traj(traj, chunk_frames: Optional[int]):
    if hasattr(traj, "reset"):
        return traj
    coords, box, time = read_all_frames(
        traj,
        chunk_frames,
        include_box=True,
        include_time=True,
    )
    if coords is None:
        raise ValueError("trajectory has no frames")
    return ArrayTrajectory(
        np.asarray(coords, dtype=np.float32),
        box=None if box is None else np.asarray(box, dtype=np.float32),
        time_ps=None if time is None else np.asarray(time, dtype=np.float32),
    )


def pca(
    traj,
    system,
    mask: str,
    n_vecs: int = 2,
    fit: bool = True,
    ref: Optional[object] = None,
    ref_mask: Optional[str] = None,
    dtype: str = "ndarray",
    chunk_frames: Optional[int] = None,
):
    """Perform PCA and return projection plus (eigenvalues, eigenvectors)."""
    _ = dtype
    idx = selection_indices(system, mask)
    if idx.size == 0:
        raise ValueError("selection resolved to empty set")
    n_features = idx.size * 3
    n_components = n_features if n_vecs < 0 else max(1, min(int(n_vecs), n_features))

    source_traj = traj
    source_system = system
    if fit:
        align_mask = ref_mask if ref_mask is not None else mask
        if selection_indices(system, align_mask).size == 0:
            raise ValueError("alignment selection resolved to empty set")
        source_traj, source_system = _fit_to_reference(
            traj,
            system,
            align_mask,
            ref,
            ref_mask,
            chunk_frames,
        )
    source_traj = _projection_ready_traj(source_traj, chunk_frames)

    evals, evecs = _run_native_pca(source_traj, source_system, mask, n_components, chunk_frames)
    reset_traj(source_traj)
    proj = projection(
        source_traj,
        source_system,
        mask=mask,
        eigenvectors=evecs,
        eigenvalues=evals,
        chunk_frames=chunk_frames,
    )
    return proj, (evals.astype(np.float32), evecs.astype(np.float32))


__all__ = ["pca", "projection"]
