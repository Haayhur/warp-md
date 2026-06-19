from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from ._runtime import coerce_native_system, is_native_traj, load_native_symbol, selection_indices


MaskLike = Union[str, Sequence[int], np.ndarray, None]


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _pair_indices(
    system,
    pairs: Optional[Sequence[Sequence[int]]],
    tail_indices: Optional[Sequence[int]],
    head_indices: Optional[Sequence[int]],
    selection: MaskLike,
) -> tuple[list[int], list[int]]:
    if pairs is not None:
        if tail_indices is not None or head_indices is not None:
            raise ValueError("pass either pairs or tail_indices/head_indices, not both")
        arr = np.asarray(pairs, dtype=np.int64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("pairs must have shape (n_vectors, 2)")
        tails = arr[:, 0]
        heads = arr[:, 1]
    elif tail_indices is not None or head_indices is not None:
        if tail_indices is None or head_indices is None:
            raise ValueError("tail_indices and head_indices must be provided together")
        tails = np.asarray(tail_indices, dtype=np.int64).reshape(-1)
        heads = np.asarray(head_indices, dtype=np.int64).reshape(-1)
    else:
        indices = selection_indices(system, selection, allow_at_indices=True)
        if indices.size % 2 != 0:
            raise ValueError("selection must contain an even number of atoms for adjacent pairs")
        tails = indices[0::2]
        heads = indices[1::2]

    if tails.shape != heads.shape:
        raise ValueError("tail/head index vectors must have identical length")
    if tails.size == 0:
        raise ValueError("at least one vector pair is required")
    if np.any(tails < 0) or np.any(heads < 0):
        raise ValueError("atom indices must be non-negative")
    return tails.astype(np.uint32).tolist(), heads.astype(np.uint32).tolist()


def _axis_arg(reference_axis: Optional[Sequence[float]]):
    if reference_axis is None:
        return None
    axis = np.asarray(reference_axis, dtype=np.float64).reshape(-1)
    if axis.shape != (3,):
        raise ValueError("reference_axis must contain exactly 3 values")
    norm = float(np.linalg.norm(axis))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("reference_axis must be finite and nonzero")
    return (float(axis[0]), float(axis[1]), float(axis[2]))


def _format_output(
    time: np.ndarray,
    values: np.ndarray,
    tail_indices: Sequence[int],
    head_indices: Sequence[int],
    dtype: str,
):
    if values.ndim != 2 or values.shape[1] != 12:
        raise RuntimeError("native NematicOrderPlan returned unexpected output")
    q = np.zeros((values.shape[0], 3, 3), dtype=np.float32)
    q[:, 0, 0] = values[:, 4]
    q[:, 1, 1] = values[:, 5]
    q[:, 2, 2] = values[:, 6]
    q[:, 0, 1] = q[:, 1, 0] = values[:, 7]
    q[:, 0, 2] = q[:, 2, 0] = values[:, 8]
    q[:, 1, 2] = q[:, 2, 1] = values[:, 9]
    out = {
        "time": time.astype(np.float32, copy=False),
        "order": values[:, 0].astype(np.float32, copy=False),
        "director": values[:, 1:4].astype(np.float32, copy=False),
        "q_tensor": q,
        "axis_order": values[:, 10].astype(np.float32, copy=False),
        "valid_vectors": values[:, 11].astype(np.float32, copy=False),
        "tail_indices": np.asarray(tail_indices, dtype=np.int64),
        "head_indices": np.asarray(head_indices, dtype=np.int64),
    }
    key = str(dtype).lower()
    if key == "dict":
        return out
    if key in ("ndarray", "array", "matrix"):
        return values.astype(np.float32, copy=False)
    if key in ("order", "director", "q_tensor", "axis_order", "valid_vectors"):
        return out[key]
    raise ValueError(
        "dtype must be 'dict', 'ndarray', or one of: "
        "order, director, q_tensor, axis_order, valid_vectors"
    )


def nematic_order(
    traj,
    system,
    pairs: Optional[Sequence[Sequence[int]]] = None,
    *,
    tail_indices: Optional[Sequence[int]] = None,
    head_indices: Optional[Sequence[int]] = None,
    selection: MaskLike = None,
    reference_axis: Optional[Sequence[float]] = None,
    pbc: bool = False,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    dtype: str = "dict",
    device: str = "auto",
):
    """Compute per-frame nematic order and directors from atom-pair vectors in Rust."""
    if not is_native_traj(traj):
        raise RuntimeError(
            "nematic_order requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native system for nematic_order")
    plan_cls = load_native_symbol("NematicOrderPlan")
    if plan_cls is None:
        raise RuntimeError(
            "NematicOrderPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )

    tails, heads = _pair_indices(system, pairs, tail_indices, head_indices, selection)
    plan = plan_cls(
        tails,
        heads,
        reference_axis=_axis_arg(reference_axis),
        pbc=bool(pbc),
        length_scale=length_scale,
    )
    time, values = plan.run(
        traj,
        native_system,
        chunk_frames=chunk_frames,
        device=device,
        frame_indices=_frame_indices_arg(frame_indices),
    )
    return _format_output(
        np.asarray(time, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        tails,
        heads,
        dtype,
    )


def compute_nematic_order(*args, **kwargs):
    return nematic_order(*args, **kwargs)


def compute_directors(*args, **kwargs):
    kwargs["dtype"] = "director"
    return nematic_order(*args, **kwargs)


__all__ = ["nematic_order", "compute_nematic_order", "compute_directors"]
