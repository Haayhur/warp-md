from __future__ import annotations

import importlib
import numbers
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from .. import traj_py as _traj_py
from ._chunk_io import read_chunk_fields
from ._stream import normalize_frame_indices


MaskLike = Union[str, Sequence[int], np.ndarray]
_MASK_SENTINELS = ("", "*", "all", None)

_NATIVE_SYSTEM_TYPES = ()
_NATIVE_TRAJ_TYPES = ()
if _traj_py is not None:
    py_system = getattr(_traj_py, "PySystem", None)
    py_traj = getattr(_traj_py, "PyTrajectory", None)
    if py_system is not None:
        _NATIVE_SYSTEM_TYPES = (py_system,)
    if py_traj is not None:
        _NATIVE_TRAJ_TYPES = (py_traj,)


class _IndexSelection:
    def __init__(self, indices):
        self.indices = list(indices)


def all_resid_mask(system) -> str:
    if not hasattr(system, "atom_table"):
        return "all"
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def is_native_system(system) -> bool:
    return bool(_NATIVE_SYSTEM_TYPES) and isinstance(system, _NATIVE_SYSTEM_TYPES)


def is_native_traj(traj) -> bool:
    return bool(_NATIVE_TRAJ_TYPES) and isinstance(traj, _NATIVE_TRAJ_TYPES)


def _infer_atom_count(atom_table, positions0) -> int:
    if positions0 is not None:
        try:
            arr = np.asarray(positions0)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return int(arr.shape[0])
        except Exception:
            pass
    for key, values in atom_table.items():
        if key in {"bonds"}:
            continue
        if values is not None:
            try:
                return int(len(values))
            except Exception:
                continue
    return 0


def _normalize_atom_table(atom_table, n_atoms: int):
    table = dict(atom_table)
    table.setdefault("name", ["X"] * n_atoms)
    table.setdefault("resname", ["RES"] * n_atoms)
    table.setdefault("resid", list(range(1, n_atoms + 1)))
    table.setdefault("chain_id", [0] * n_atoms)
    table.setdefault("mass", [1.0] * n_atoms)
    return table


def coerce_native_system(system):
    if is_native_system(system):
        return system
    py_system = getattr(_traj_py, "PySystem", None)
    if py_system is None or not hasattr(py_system, "from_arrays"):
        return None
    if not hasattr(system, "atom_table"):
        return None
    try:
        atom_table = system.atom_table()
    except Exception:
        return None
    positions0 = None
    if hasattr(system, "positions0"):
        try:
            positions0 = system.positions0()
        except Exception:
            positions0 = None
    try:
        n_atoms = _infer_atom_count(atom_table, positions0)
        return py_system.from_arrays(
            _normalize_atom_table(atom_table, n_atoms),
            positions0=positions0,
        )
    except Exception:
        return None


def _system_positions0(system):
    if not hasattr(system, "positions0"):
        return None
    try:
        return system.positions0()
    except Exception:
        return None


def native_system_from_atom_count(n_atoms: int, positions0: Optional[np.ndarray] = None):
    py_system = getattr(_traj_py, "PySystem", None)
    if py_system is None or not hasattr(py_system, "from_arrays"):
        return None
    n_atoms = max(0, int(n_atoms))
    try:
        return py_system.from_arrays(
            _normalize_atom_table({}, n_atoms),
            positions0=None if positions0 is None else np.asarray(positions0, dtype=np.float32),
        )
    except Exception:
        return None


def native_system_with_positions0(system, positions0: np.ndarray):
    py_system = getattr(_traj_py, "PySystem", None)
    if py_system is None or not hasattr(py_system, "from_arrays"):
        return None
    atom_table = {}
    if hasattr(system, "atom_table"):
        try:
            atom_table = dict(system.atom_table())
        except Exception:
            atom_table = {}
    n_atoms = _infer_atom_count(atom_table, positions0)
    try:
        return py_system.from_arrays(
            _normalize_atom_table(atom_table, n_atoms),
            positions0=np.asarray(positions0, dtype=np.float32),
        )
    except Exception:
        return None


def coerce_native_traj(
    traj,
    chunk_frames: Optional[int],
    *,
    include_box: bool = False,
    include_time: bool = False,
):
    if is_native_traj(traj):
        return traj
    py_traj = getattr(_traj_py, "PyTrajectory", None)
    if py_traj is None or not hasattr(py_traj, "from_numpy"):
        return None
    coords, box, time = read_all_frames(
        traj,
        chunk_frames,
        include_box=include_box,
        include_time=include_time,
    )
    if coords is None:
        return None
    try:
        return py_traj.from_numpy(
            np.asarray(coords, dtype=np.float32),
            None if box is None else np.asarray(box, dtype=np.float32),
            None if time is None else np.asarray(time, dtype=np.float32),
        )
    except Exception:
        return None


def native_inputs(
    traj,
    system,
    chunk_frames: Optional[int],
    *,
    include_box: bool = False,
    include_time: bool = False,
):
    native_traj = coerce_native_traj(
        traj,
        chunk_frames,
        include_box=include_box,
        include_time=include_time,
    )
    native_system = coerce_native_system(system)
    if native_system is None and native_traj is not None:
        n_atoms = None
        if hasattr(native_traj, "n_atoms"):
            try:
                n_atoms = int(native_traj.n_atoms())
            except Exception:
                n_atoms = None
        if n_atoms is None:
            coords, _box, _time = read_all_frames(
                traj,
                chunk_frames,
                include_box=include_box,
                include_time=include_time,
            )
            if coords is not None and coords.ndim == 3:
                n_atoms = int(coords.shape[1])
        if n_atoms is not None:
            native_system = native_system_from_atom_count(
                n_atoms,
                positions0=_system_positions0(system),
            )
    if native_traj is None or native_system is None:
        return None, None
    return native_traj, native_system


def read_frame_subset(
    traj,
    frame_indices: Sequence[int],
    chunk_frames: Optional[int],
    *,
    include_box: bool = False,
    include_time: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
    indices = [int(idx) for idx in frame_indices]
    if not indices:
        return (
            np.empty((0, 0, 3), dtype=np.float32),
            None,
            None,
            np.empty((0,), dtype=np.int64),
        )
    if hasattr(traj, "read_frames"):
        payload = traj.read_frames(
            indices,
            chunk_frames=chunk_frames,
            include_box=include_box,
            include_box_matrix=False,
            include_time=include_time,
        )
        if payload is None:
            return None, None, None, np.empty((0,), dtype=np.int64)
        coords = np.asarray(payload["coords"], dtype=np.float32)
        box = None
        if include_box:
            box_val = payload.get("box")
            if box_val is not None:
                box = np.asarray(box_val, dtype=np.float32)
        time = None
        if include_time:
            time_val = payload.get("time_ps")
            if time_val is None:
                time_val = payload.get("time")
            if time_val is not None:
                time = np.asarray(time_val, dtype=np.float32)
        source_indices = np.asarray(payload.get("source_indices", ()), dtype=np.int64)
        return coords, box, time, source_indices

    coords, box, time = read_all_frames(
        traj,
        chunk_frames,
        include_box=include_box,
        include_time=include_time,
    )
    if coords is None:
        return None, None, None, np.empty((0,), dtype=np.int64)
    source_indices = np.asarray(
        normalize_frame_indices(indices, coords.shape[0]) or [],
        dtype=np.int64,
    )
    if source_indices.size == 0:
        return (
            np.empty((0, coords.shape[1], coords.shape[2]), dtype=np.float32),
            None if box is None else np.empty((0, box.shape[1]), dtype=np.float32),
            None if time is None else np.empty((0,), dtype=np.float32),
            source_indices,
        )
    coords_sel = np.asarray(coords[source_indices], dtype=np.float32)
    box_sel = None if box is None else np.asarray(box[source_indices], dtype=np.float32)
    time_sel = None if time is None else np.asarray(time[source_indices], dtype=np.float32)
    return coords_sel, box_sel, time_sel, source_indices


def clone_native_system_with_positions0(system, positions0: np.ndarray):
    py_system = getattr(_traj_py, "PySystem", None)
    if py_system is None or not hasattr(py_system, "from_arrays"):
        return None
    base_system = coerce_native_system(system)
    if base_system is None or not hasattr(base_system, "atom_table"):
        return None
    try:
        atom_table = base_system.atom_table()
        return py_system.from_arrays(
            atom_table,
            positions0=np.asarray(positions0, dtype=np.float32),
        )
    except Exception:
        return None


def resolve_reference_coords(
    traj,
    system,
    ref,
    chunk_frames: Optional[int],
) -> np.ndarray:
    if isinstance(ref, numbers.Integral):
        coords, _box, _time, source_indices = read_frame_subset(
            traj,
            [int(ref)],
            chunk_frames,
        )
        if coords is None or coords.shape[0] != 1 or source_indices.size != 1:
            raise ValueError("ref index out of range")
        return np.asarray(coords[0], dtype=np.float32)
    if not isinstance(ref, str):
        raise ValueError("ref must be an integer frame index or a supported string")
    key = ref.strip().lower()
    if key in ("topology", "top", "topo"):
        if hasattr(system, "positions0"):
            try:
                positions0 = system.positions0()
            except Exception:
                positions0 = None
            if positions0 is not None:
                return np.asarray(positions0, dtype=np.float32)
        return resolve_reference_coords(traj, system, 0, chunk_frames)
    if key in ("frame0", "first", "0"):
        return resolve_reference_coords(traj, system, 0, chunk_frames)
    raise ValueError("unsupported ref value")


def native_reference_inputs(
    traj,
    system,
    mask: MaskLike,
    ref,
    ref_mask: MaskLike,
    chunk_frames: Optional[int],
    *,
    include_box: bool = False,
    include_time: bool = False,
):
    native_traj, native_system = native_inputs(
        traj,
        system,
        chunk_frames,
        include_box=include_box,
        include_time=include_time,
    )
    if native_traj is None or native_system is None:
        return None, None
    mask_idx = selection_indices(system, mask)
    ref_idx = selection_indices(system, ref_mask)
    if mask_idx.size != ref_idx.size:
        raise ValueError("mask and ref_mask selections must have same size")
    ref_coords = resolve_reference_coords(native_traj, native_system, ref, chunk_frames)
    n_atoms = int(native_system.n_atoms())
    positions0 = np.zeros((n_atoms, 3), dtype=np.float32)
    if hasattr(native_system, "positions0"):
        try:
            existing = native_system.positions0()
        except Exception:
            existing = None
        if existing is not None:
            positions0[:] = np.asarray(existing, dtype=np.float32)
    positions0[mask_idx] = np.asarray(ref_coords, dtype=np.float32)[ref_idx]
    reference_system = clone_native_system_with_positions0(native_system, positions0)
    if reference_system is None:
        return None, None
    reset_traj(native_traj)
    return native_traj, reference_system


def prepend_reference_frame(
    traj,
    ref: int,
    chunk_frames: Optional[int],
    *,
    include_box: bool = False,
    include_time: bool = False,
):
    py_traj = getattr(_traj_py, "PyTrajectory", None)
    if py_traj is None or not hasattr(py_traj, "from_numpy"):
        return None
    ref_coords, ref_box, ref_time, source_indices = read_frame_subset(
        traj,
        [int(ref)],
        chunk_frames,
        include_box=include_box,
        include_time=include_time,
    )
    if ref_coords is None or ref_coords.shape[0] != 1 or source_indices.size != 1:
        raise ValueError("ref index out of range")
    reset_traj(traj)
    coords, box, time = read_all_frames(
        traj,
        chunk_frames,
        include_box=include_box,
        include_time=include_time,
    )
    if coords is None:
        return None
    coords = np.concatenate(
        [
            np.asarray(ref_coords, dtype=np.float32),
            np.asarray(coords, dtype=np.float32),
        ],
        axis=0,
    )
    if include_box:
        if ref_box is None:
            box = None
        elif box is None:
            box = np.asarray(ref_box, dtype=np.float32)
        else:
            box = np.concatenate(
                [
                    np.asarray(ref_box, dtype=np.float32),
                    np.asarray(box, dtype=np.float32),
                ],
                axis=0,
            )
    if include_time:
        if ref_time is None:
            time = None
        elif time is None:
            time = np.asarray(ref_time, dtype=np.float32)
        else:
            time = np.concatenate(
                [
                    np.asarray(ref_time, dtype=np.float32),
                    np.asarray(time, dtype=np.float32),
                ],
                axis=0,
            )
    try:
        return py_traj.from_numpy(
            np.asarray(coords, dtype=np.float32),
            None if box is None else np.asarray(box, dtype=np.float32),
            None if time is None else np.asarray(time, dtype=np.float32),
        )
    except Exception:
        return None


def reset_traj(traj) -> bool:
    if not hasattr(traj, "reset"):
        return False
    try:
        traj.reset()
    except Exception:
        return False
    return True


def load_native_symbol(name: str):
    try:
        warp_md = importlib.import_module("warp_md")
    except Exception:
        return None
    symbol = getattr(warp_md, name, None)
    symbol_name = getattr(symbol, "__name__", "")
    if symbol is None or symbol_name == "_Missing" or symbol_name.startswith("Missing"):
        return None
    return symbol


def _tuple3(name: str, values, *, positive: bool = False) -> Tuple[float, float, float]:
    if values is None:
        raise ValueError(f"{name} is required")
    if len(values) != 3:
        raise ValueError(f"{name} must have length 3")
    out = (float(values[0]), float(values[1]), float(values[2]))
    if positive and any(value <= 0.0 for value in out):
        raise ValueError(f"{name} values must be positive")
    return out


def run_native_grid_plan(
    plan_name: str,
    traj,
    system,
    selection,
    center_selection,
    box_unit,
    region_size,
    *,
    shift=None,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    plan_cls = load_native_symbol(plan_name)
    if plan_cls is None:
        raise RuntimeError(f"{plan_name} binding unavailable")
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError(f"failed to prepare native inputs for {plan_name}")
    try:
        plan = plan_cls(
            native_selection(system, native_system, selection),
            native_selection(system, native_system, center_selection),
            _tuple3("box_unit", box_unit, positive=True),
            _tuple3("region_size", region_size, positive=True),
            shift=None if shift is None else _tuple3("shift", shift),
            length_scale=None if length_scale is None else float(length_scale),
        )
        output = plan.run(
            native_traj,
            native_system,
            chunk_frames=chunk_frames,
            device=device,
            frame_indices=None if frame_indices is None else [int(i) for i in frame_indices],
        )
    except Exception as exc:
        raise RuntimeError(f"native {plan_name} execution failed") from exc
    return {
        "dims": tuple(int(v) for v in output["dims"]),
        "mean": np.asarray(output["mean"]),
        "std": np.asarray(output["std"]),
        "first": np.asarray(output["first"]),
        "last": np.asarray(output["last"]),
        "min": np.asarray(output["min"]),
        "max": np.asarray(output["max"]),
    }


def native_reference_mode(system, ref) -> Optional[str]:
    if isinstance(ref, int):
        return "frame0" if ref == 0 else None
    if not isinstance(ref, str):
        return None
    key = ref.strip().lower()
    if key in ("frame0", "first", "0"):
        return "frame0"
    if key in ("topology", "top", "topo"):
        if hasattr(system, "positions0"):
            try:
                if system.positions0() is not None:
                    return "topology"
            except Exception:
                pass
        return "frame0"
    return None


def _parse_at_indices(mask: str) -> Optional[np.ndarray]:
    tokens = mask.replace(",", " ").split()
    indices = []
    for token in tokens:
        if token.startswith("@") and token[1:].lstrip("-").isdigit():
            indices.append(int(token[1:]) - 1)
        else:
            return None
    if not indices:
        return None
    return np.asarray(indices, dtype=np.int64)


def materialize_selection(system, mask: MaskLike, *, allow_at_indices: bool = False):
    if isinstance(mask, np.ndarray):
        indices = np.asarray(mask, dtype=np.int64).reshape(-1)
        if hasattr(system, "select_indices"):
            return system.select_indices(indices.tolist())
        return _IndexSelection(indices.tolist())
    if isinstance(mask, (list, tuple)) and not isinstance(mask, str):
        if mask and all(not isinstance(value, str) for value in mask):
            indices = np.asarray(mask, dtype=np.int64).reshape(-1)
            if hasattr(system, "select_indices"):
                return system.select_indices(indices.tolist())
            return _IndexSelection(indices.tolist())
    if isinstance(mask, str) and allow_at_indices:
        parsed = _parse_at_indices(mask.strip())
        if parsed is not None:
            if hasattr(system, "select_indices"):
                return system.select_indices(parsed.tolist())
            return _IndexSelection(parsed.tolist())
    expr = all_resid_mask(system) if mask in _MASK_SENTINELS else mask
    return system.select(expr)


def selection_indices(system, mask: MaskLike, *, allow_at_indices: bool = False) -> np.ndarray:
    sel = materialize_selection(system, mask, allow_at_indices=allow_at_indices)
    return np.asarray(list(sel.indices), dtype=np.int64)


def native_selection(system, native_system, mask: MaskLike, *, allow_at_indices: bool = False):
    indices = selection_indices(system, mask, allow_at_indices=allow_at_indices)
    if hasattr(native_system, "select_indices"):
        return native_system.select_indices(indices.tolist())
    return _IndexSelection(indices.tolist())


def mass_weights(system, indices: np.ndarray, mass: bool) -> np.ndarray:
    if not mass:
        return np.ones(indices.size, dtype=np.float64)
    atoms = system.atom_table()
    masses = atoms.get("mass", [])
    if not masses:
        return np.ones(indices.size, dtype=np.float64)
    return np.asarray(masses, dtype=np.float64)[indices]


def kabsch_rmsd(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.shape != y.shape:
        raise ValueError("frame shapes must match for RMSD")
    cx = x.mean(axis=0)
    cy = y.mean(axis=0)
    x0 = x - cx
    y0 = y - cy
    h = x0.T @ y0
    u, _s, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    x_rot = x0 @ r.T
    diff = x_rot - y0
    return float(np.sqrt((diff * diff).sum() / x.shape[0]))


def rmsd_raw(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return 0.0
    diff = x - y
    return float(np.sqrt((diff * diff).sum() / x.shape[0]))


def read_all_frames(
    traj,
    chunk_frames: Optional[int],
    *,
    include_box: bool = False,
    include_time: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    max_chunk = max(1, int(chunk_frames or 128))
    coords_list = []
    box_list = []
    time_list = []
    chunk = read_chunk_fields(
        traj,
        max_chunk,
        include_box=include_box,
        include_time=include_time,
    )
    if chunk is None:
        return None, None, None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        if include_box:
            box = chunk.get("box")
            if box is not None:
                box_list.append(np.asarray(box, dtype=np.float64))
        if include_time:
            time = chunk.get("time_ps")
            if time is None:
                time = chunk.get("time")
            if time is not None:
                time_list.append(np.asarray(time, dtype=np.float64))
        chunk = read_chunk_fields(
            traj,
            max_chunk,
            include_box=include_box,
            include_time=include_time,
        )
    coords = np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))
    box = np.concatenate(box_list, axis=0) if box_list else None
    time = np.concatenate(time_list, axis=0) if time_list else None
    return coords, box, time


def subset_frames(
    coords: np.ndarray,
    frame_indices: Optional[Sequence[int]],
    *,
    box: Optional[np.ndarray] = None,
    time: Optional[np.ndarray] = None,
):
    if frame_indices is None:
        return coords, box, time
    selected = normalize_frame_indices(frame_indices, coords.shape[0])
    if selected is None:
        return coords, box, time
    coords = coords[selected]
    if box is not None:
        box = box[selected]
    if time is not None:
        time = time[selected]
    return coords, box, time
