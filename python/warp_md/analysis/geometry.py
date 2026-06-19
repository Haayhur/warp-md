# Usage:
# from warp_md.analysis.geometry import angle, dihedral, distance
# vals = angle(traj, system, "@1 @2 @3")

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_inputs,
    reset_traj,
)


MaskLike = Union[str, Sequence[str], np.ndarray]


def _all_resid_mask(system) -> str:
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _selection_indices(system, mask: str) -> np.ndarray:
    if mask in ("", "*", "all", None):
        sel = system.select(_all_resid_mask(system))
        return np.asarray(list(sel.indices), dtype=np.int64)
    if isinstance(mask, str) and mask.strip().startswith("@"):
        raw = mask.strip()
        toks = raw[1:].replace("@", "").replace(",", " ").split()
        idx = []
        for tok in toks:
            if "-" in tok:
                left, right = tok.split("-", 1)
                if left.lstrip("-").isdigit() and right.lstrip("-").isdigit():
                    start = int(left)
                    stop = int(right)
                    step = 1 if stop >= start else -1
                    idx.extend(range(start - 1, stop - 1 + step, step))
                    continue
            if tok.lstrip("-").isdigit():
                idx.append(int(tok) - 1)
        if idx:
            return np.asarray(idx, dtype=np.int64)
    sel = system.select(mask)
    return np.asarray(list(sel.indices), dtype=np.int64)


def _center(coords: np.ndarray, idx: np.ndarray, masses: Optional[np.ndarray], mass: bool) -> np.ndarray:
    sel = coords[:, idx, :]
    if sel.size == 0:
        raise ValueError("selection resolved to empty set")
    if mass and masses is not None and masses.size > 0:
        w = masses[idx]
        wsum = np.sum(w)
        if wsum <= 0.0:
            w = np.ones_like(w)
            wsum = np.sum(w)
        return (sel * w[None, :, None]).sum(axis=1) / wsum
    return sel.mean(axis=1)


def _split_distance_mask(mask: str) -> Tuple[str, str]:
    parts = str(mask).split()
    if len(parts) != 2:
        raise ValueError("distance mask must contain exactly two masks")
    return parts[0], parts[1]


def _distance_dtype(out, dtype: str):
    key = str(dtype).lower()
    arr = np.asarray(out, dtype=np.float32)
    if key in ("stats", "summary"):
        frame_axis = -1 if arr.ndim == 2 else 0
        return _distance_stats(arr, frame_axis=frame_axis)
    if key == "dict":
        return {"distance": arr}
    return arr


def _distance_components_dtype(out, dtype: str):
    arr = np.asarray(out, dtype=np.float32)
    key = str(dtype).lower()
    if key in ("stats", "summary"):
        return _distance_stats(arr, frame_axis=0)
    if key == "dict":
        return {"components": arr}
    return arr


def _stat_value(value):
    arr = np.asarray(value, dtype=np.float32)
    return float(arr) if arr.ndim == 0 else arr


def _distance_stats(values: np.ndarray, frame_axis: int):
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    axis = frame_axis if frame_axis >= 0 else arr.ndim + frame_axis
    n_frames = int(arr.shape[axis]) if arr.ndim > 0 else 0
    if n_frames == 0:
        out_shape = arr.shape[:axis] + arr.shape[axis + 1 :]
        missing = np.full(out_shape, np.nan, dtype=np.float32)
        return {
            "mean": _stat_value(missing),
            "std": _stat_value(missing),
            "min": _stat_value(missing),
            "max": _stat_value(missing),
            "n_frames": 0,
        }
    return {
        "mean": _stat_value(np.mean(arr, axis=axis)),
        "std": _stat_value(np.std(arr, axis=axis)),
        "min": _stat_value(np.min(arr, axis=axis)),
        "max": _stat_value(np.max(arr, axis=axis)),
        "n_frames": n_frames,
    }


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _native_selected(native_system, indices: np.ndarray):
    if not hasattr(native_system, "select_indices"):
        raise RuntimeError("native system does not support index selections")
    return native_system.select_indices(np.asarray(indices, dtype=np.int64).tolist())


def _require_native_geometry_inputs(traj, system, name: str, chunk_frames: Optional[int]):
    if is_native_traj(traj):
        native_system = coerce_native_system(system)
        if native_system is None:
            raise RuntimeError(f"failed to prepare native inputs for {name}")
        return traj, native_system
    native_traj, native_system = native_inputs(traj, system, chunk_frames)
    if native_traj is None or native_system is None:
        raise RuntimeError(f"failed to prepare native inputs for {name}")
    return native_traj, native_system


def _selection_triplet(system, native_system, command):
    parts = str(command).split()
    if len(parts) != 3:
        raise ValueError("angle mask must have 3 parts")
    return tuple(_native_selected(native_system, _selection_indices(system, part)) for part in parts)


def _selection_quartet(system, native_system, command):
    parts = str(command).split()
    if len(parts) != 4:
        raise ValueError("dihedral mask must have 4 parts")
    return tuple(_native_selected(native_system, _selection_indices(system, part)) for part in parts)


def _index_selection(native_system, atom_idx: int):
    return _native_selected(native_system, np.asarray([int(atom_idx)], dtype=np.int64))


def _reset_between_native_runs(traj, name: str):
    if not reset_traj(traj):
        raise RuntimeError(f"{name} requires a resettable Rust-backed trajectory for multiple commands")


def _run_angle_native(
    traj,
    native_system,
    sel_a,
    sel_b,
    sel_c,
    mass,
    chunk_frames,
    frame_indices,
):
    plan_cls = load_native_symbol("AnglePlan")
    if plan_cls is None:
        raise RuntimeError(
            "AnglePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    plan = plan_cls(
        sel_a,
        sel_b,
        sel_c,
        mass_weighted=bool(mass),
        pbc="none",
        degrees=True,
    )
    return np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )


def _run_multi_angle_groups_native(
    traj,
    native_system,
    groups,
    mass,
    chunk_frames,
    frame_indices,
):
    plan_cls = load_native_symbol("MultiAnglePlan")
    if plan_cls is None:
        raise RuntimeError(
            "MultiAnglePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if not groups:
        return np.empty((0, 0), dtype=np.float32)
    sel_a = [np.asarray(group[0], dtype=np.int64).astype(np.uint32).tolist() for group in groups]
    sel_b = [np.asarray(group[1], dtype=np.int64).astype(np.uint32).tolist() for group in groups]
    sel_c = [np.asarray(group[2], dtype=np.int64).astype(np.uint32).tolist() for group in groups]
    plan = plan_cls(
        sel_a,
        sel_b,
        sel_c,
        mass_weighted=bool(mass),
        pbc="none",
        degrees=True,
    )
    values = np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )
    if values.ndim != 2:
        raise RuntimeError("MultiAnglePlan returned unexpected output")
    return values.T


def _multi_angle_command_groups(system, commands):
    groups = []
    for command in commands:
        parts = str(command).split()
        if len(parts) != 3:
            raise ValueError("angle mask must have 3 parts")
        groups.append(
            (
                _selection_indices(system, parts[0]),
                _selection_indices(system, parts[1]),
                _selection_indices(system, parts[2]),
            )
        )
    return groups


def _run_dihedral_native(
    traj,
    native_system,
    sel_a,
    sel_b,
    sel_c,
    sel_d,
    mass,
    range360,
    chunk_frames,
    frame_indices,
):
    plan_cls = load_native_symbol("DihedralPlan")
    if plan_cls is None:
        raise RuntimeError(
            "DihedralPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    plan = plan_cls(
        sel_a,
        sel_b,
        sel_c,
        sel_d,
        mass_weighted=bool(mass),
        pbc="none",
        degrees=True,
        range360=bool(range360),
    )
    return np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )


def _run_multi_dihedral_groups_native(
    traj,
    native_system,
    groups,
    mass,
    range360,
    chunk_frames,
    frame_indices,
):
    plan_cls = load_native_symbol("MultiDihedralPlan")
    if plan_cls is None:
        raise RuntimeError(
            "MultiDihedralPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if not groups:
        return np.empty((0, 0), dtype=np.float32)
    native_groups = []
    for a, b, c, d in groups:
        native_groups.append(
            (
                _native_selected(native_system, np.asarray(a, dtype=np.int64)),
                _native_selected(native_system, np.asarray(b, dtype=np.int64)),
                _native_selected(native_system, np.asarray(c, dtype=np.int64)),
                _native_selected(native_system, np.asarray(d, dtype=np.int64)),
            )
        )
    plan = plan_cls(
        native_groups,
        mass_weighted=bool(mass),
        pbc="none",
        degrees=True,
        range360=bool(range360),
    )
    values = np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )
    if values.ndim != 2:
        raise RuntimeError("MultiDihedralPlan returned unexpected output")
    return values.T


def _multi_dihedral_command_groups(system, commands):
    groups = []
    for command in commands:
        parts = str(command).split()
        if len(parts) != 4:
            raise ValueError("dihedral mask must have 4 parts")
        groups.append(
            (
                _selection_indices(system, parts[0]),
                _selection_indices(system, parts[1]),
                _selection_indices(system, parts[2]),
                _selection_indices(system, parts[3]),
            )
        )
    return groups


def _native_distance_system(system):
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native distance system")
    return native_system


def _run_pair_list_distance_native(traj, native_system, pairs, image, chunk_frames, frame_indices):
    plan_cls = load_native_symbol("PairListDistancePlan")
    if plan_cls is None:
        raise RuntimeError(
            "PairListDistancePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    pbc = "orthorhombic" if image else "none"
    plan = plan_cls(np.asarray(pairs, dtype=np.int64), pbc=pbc)
    values = np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )
    if values.ndim != 2:
        raise RuntimeError("PairListDistancePlan returned unexpected output")
    return values.T


def _adjacent_pair_indices(system, mask: str) -> np.ndarray:
    indices = _selection_indices(system, mask)
    if indices.size == 0:
        raise ValueError("adjacent pair distance selection is empty")
    if indices.size % 2 != 0:
        raise ValueError("adjacent pair distance requires an even number of selected atoms")
    if np.any(indices < 0):
        raise ValueError("adjacent pair distance atom indices must be >= 0")
    return indices.astype(np.int64, copy=False).reshape(-1, 2)


def _run_mask_distance_native(
    traj, native_system, system, mask_a, mask_b, mass, image, chunk_frames, frame_indices
):
    plan_cls = load_native_symbol("DistancePlan")
    if plan_cls is None:
        raise RuntimeError(
            "DistancePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    idx_a = _selection_indices(system, mask_a)
    idx_b = _selection_indices(system, mask_b)
    pbc = "orthorhombic" if image else "none"
    plan = plan_cls(
        _native_selected(native_system, idx_a),
        _native_selected(native_system, idx_b),
        mass_weighted=bool(mass),
        pbc=pbc,
    )
    return np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )


def _run_mask_distance_vector_native(
    traj, native_system, system, mask_a, mask_b, mass, image, chunk_frames, frame_indices
):
    plan_cls = load_native_symbol("DistanceVectorPlan")
    if plan_cls is None:
        raise RuntimeError(
            "DistanceVectorPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    idx_a = _selection_indices(system, mask_a)
    idx_b = _selection_indices(system, mask_b)
    pbc = "orthorhombic" if image else "none"
    plan = plan_cls(
        _native_selected(native_system, idx_a),
        _native_selected(native_system, idx_b),
        mass_weighted=bool(mass),
        pbc=pbc,
    )
    values = np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )
    if values.ndim != 2 or values.shape[1] != 3:
        raise RuntimeError("DistanceVectorPlan returned unexpected output")
    return values


def _run_multi_mask_distance_native(
    traj, native_system, system, commands, mass, image, chunk_frames, frame_indices
):
    plan_cls = load_native_symbol("MultiDistancePlan")
    if plan_cls is None:
        raise RuntimeError(
            "MultiDistancePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    commands = [str(command) for command in commands]
    if not commands:
        return np.empty((0, 0), dtype=np.float32)
    selections_a = []
    selections_b = []
    for command in commands:
        mask_a, mask_b = _split_distance_mask(command)
        selections_a.append(_selection_indices(system, mask_a).astype(np.uint32).tolist())
        selections_b.append(_selection_indices(system, mask_b).astype(np.uint32).tolist())
    pbc = "orthorhombic" if image else "none"
    plan = plan_cls(
        selections_a,
        selections_b,
        mass_weighted=bool(mass),
        pbc=pbc,
    )
    values = np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )
    if values.ndim != 2:
        raise RuntimeError("MultiDistancePlan returned unexpected output")
    return values.T


def _run_center_to_point_distance_native(
    traj, native_system, system, mask, point, mass, image, chunk_frames, frame_indices
):
    plan_cls = load_native_symbol("DistanceCenterToPointPlan")
    if plan_cls is None:
        raise RuntimeError(
            "DistanceCenterToPointPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    target = np.asarray(point, dtype=np.float64).reshape(-1)
    if target.size != 3:
        raise ValueError("point must have 3 coordinates")
    idx = _selection_indices(system, mask)
    pbc = "orthorhombic" if image else "none"
    plan = plan_cls(
        _native_selected(native_system, idx),
        (float(target[0]), float(target[1]), float(target[2])),
        mass_weighted=bool(mass),
        pbc=pbc,
    )
    return np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )


def _run_center_to_reference_distance_native(
    traj, native_system, system, mask, reference, mass, image, chunk_frames, frame_indices
):
    plan_cls = load_native_symbol("DistanceCenterToReferencePlan")
    if plan_cls is None:
        raise RuntimeError(
            "DistanceCenterToReferencePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    idx = _selection_indices(system, mask)
    pbc = "orthorhombic" if image else "none"
    plan = plan_cls(
        _native_selected(native_system, idx),
        reference=str(reference),
        mass_weighted=bool(mass),
        pbc=pbc,
    )
    return np.asarray(
        plan.run(
            traj,
            native_system,
            chunk_frames=chunk_frames,
            device="auto",
            frame_indices=_frame_indices_arg(frame_indices),
        ),
        dtype=np.float32,
    )


def _reference_point_from_value(system, mask, ref, mass):
    ref_arr = np.asarray(ref, dtype=np.float64)
    if ref_arr.shape == (3,):
        return ref_arr
    if ref_arr.ndim != 2 or ref_arr.shape[1] != 3:
        raise ValueError("ref must be frame0/topology, a point, or a coordinate frame")
    idx = _selection_indices(system, mask)
    masses = None
    if mass:
        atoms = system.atom_table()
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)
    return _center(ref_arr[None, :, :], idx, masses, mass)[0]


def angle(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    mass: bool = False,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Compute angle between three masks or index triplets."""
    native_traj, native_system = _require_native_geometry_inputs(
        traj, system, "angle", chunk_frames
    )

    if isinstance(mask, str):
        sel_a, sel_b, sel_c = _selection_triplet(system, native_system, mask)
        return _run_angle_native(
            native_traj, native_system, sel_a, sel_b, sel_c, mass, chunk_frames, frame_indices
        )

    arr = np.asarray(mask)
    if arr.dtype.kind == "i":
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("angle index array must be shape (n, 3)")
        groups = [
            (
                np.asarray([int(a)], dtype=np.int64),
                np.asarray([int(b)], dtype=np.int64),
                np.asarray([int(c)], dtype=np.int64),
            )
            for a, b, c in arr
        ]
        return _run_multi_angle_groups_native(
            native_traj, native_system, groups, mass, chunk_frames, frame_indices
        )

    commands = list(mask)
    groups = _multi_angle_command_groups(system, commands)
    return _run_multi_angle_groups_native(
        native_traj, native_system, groups, mass, chunk_frames, frame_indices
    )


def distance(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    image: bool = False,
    mass: bool = True,
    ref=None,
    point=None,
    chunk_frames: Optional[int] = None,
    pair_mode: Optional[str] = None,
    components: bool = False,
) -> np.ndarray:
    """Compute distance measurements.

    String masks compute one COM/COG distance per frame. Lists of mask strings
    return one row per command. Integer arrays of shape ``(n_pairs, 2)`` return
    explicit atom-pair distances with shape ``(n_pairs, n_frames)``. Set
    ``pair_mode="adjacent"`` with one selection mask to pair atoms as
    ``(1-2, 3-4, ...)``. Set ``dtype="stats"`` for per-command summary
    statistics over frames.
    """
    command = np.asarray(mask)
    if not is_native_traj(traj):
        raise RuntimeError(
            "distance requires a Rust-backed trajectory so frame/atom loops stay in Rust."
        )
    native_traj = traj

    native_system = _native_distance_system(system)
    pair_mode_value = "center" if pair_mode is None else str(pair_mode).strip().lower()
    if pair_mode_value not in ("center", "adjacent"):
        raise ValueError("pair_mode must be None, 'center', or 'adjacent'")
    components = bool(components)

    if command.dtype.kind in ("i", "u"):
        if components:
            raise ValueError("distance components are only supported for mask-pair distances")
        pairs = command.astype(np.int64, copy=False)
        if pairs.ndim == 1:
            if pairs.size != 2:
                raise ValueError("distance index array must be shape (n_pairs, 2)")
            pairs = pairs.reshape(1, 2)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("distance index array must be shape (n_pairs, 2)")
        values = _run_pair_list_distance_native(
            traj, native_system, pairs, image, chunk_frames, frame_indices
        )
        return _distance_dtype(values, dtype)

    if components:
        if ref is not None or point is not None:
            raise ValueError("distance components do not accept point or ref")
        if pair_mode_value != "center":
            raise ValueError("distance components require pair_mode=None")
        if not isinstance(mask, str):
            raise ValueError("distance components require one mask-pair command")
        mask_a, mask_b = _split_distance_mask(mask)
        values = _run_mask_distance_vector_native(
            native_traj,
            native_system,
            system,
            mask_a,
            mask_b,
            mass,
            image,
            chunk_frames,
            frame_indices,
        )
        return _distance_components_dtype(values, dtype)

    if pair_mode_value == "adjacent":
        if ref is not None or point is not None:
            raise ValueError("adjacent pair distance does not accept point or ref")
        if not isinstance(mask, str):
            raise ValueError("adjacent pair distance requires one selection mask")
        pairs = _adjacent_pair_indices(system, mask)
        values = _run_pair_list_distance_native(
            traj, native_system, pairs, image, chunk_frames, frame_indices
        )
        return _distance_dtype(values, dtype)

    if isinstance(mask, (list, tuple, np.ndarray)) and not isinstance(mask, str):
        values = _run_multi_mask_distance_native(
            native_traj,
            native_system,
            system,
            list(mask),
            mass,
            image,
            chunk_frames,
            frame_indices,
        )
        return _distance_dtype(values, dtype)

    if not (isinstance(mask, (list, tuple, np.ndarray)) and not isinstance(mask, str)):
        if point is not None and ref is not None:
            raise ValueError("distance accepts only one of point or ref")
        if point is not None:
            values = _run_center_to_point_distance_native(
                traj,
                native_system,
                system,
                str(mask),
                point,
                mass,
                image,
                chunk_frames,
                frame_indices,
            )
            return _distance_dtype(values, dtype)
        if ref is not None:
            if isinstance(ref, (int, np.integer)):
                if int(ref) != 0:
                    raise ValueError("distance ref frame index currently supports only 0")
                values = _run_center_to_reference_distance_native(
                    traj,
                    native_system,
                    system,
                    str(mask),
                    "frame0",
                    mass,
                    image,
                    chunk_frames,
                    frame_indices,
                )
                return _distance_dtype(values, dtype)
            if isinstance(ref, str) and ref.strip().lower() in ("topology", "top", "topo"):
                values = _run_center_to_reference_distance_native(
                    traj,
                    native_system,
                    system,
                    str(mask),
                    "topology",
                    mass,
                    image,
                    chunk_frames,
                    frame_indices,
                )
                return _distance_dtype(values, dtype)
            target = _reference_point_from_value(system, str(mask), ref, mass)
            values = _run_center_to_point_distance_native(
                traj,
                native_system,
                system,
                str(mask),
                target,
                mass,
                image,
                chunk_frames,
                frame_indices,
            )
            return _distance_dtype(values, dtype)

    mask_a, mask_b = _split_distance_mask(str(mask))
    values = _run_mask_distance_native(
        native_traj,
        native_system,
        system,
        mask_a,
        mask_b,
        mass,
        image,
        chunk_frames,
        frame_indices,
    )
    return _distance_dtype(values, dtype)


def dihedral(
    traj,
    system,
    mask: MaskLike = "",
    frame_indices: Optional[Sequence[int]] = None,
    dtype: str = "ndarray",
    mass: bool = False,
    range360: bool = False,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Compute dihedral angle between four masks or index quartets."""
    native_traj, native_system = _require_native_geometry_inputs(
        traj, system, "dihedral", chunk_frames
    )

    if isinstance(mask, str):
        sel_a, sel_b, sel_c, sel_d = _selection_quartet(system, native_system, mask)
        return _run_dihedral_native(
            native_traj,
            native_system,
            sel_a,
            sel_b,
            sel_c,
            sel_d,
            mass,
            range360,
            chunk_frames,
            frame_indices,
        )

    arr = np.asarray(mask)
    if arr.dtype.kind == "i":
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("dihedral index array must be shape (n, 4)")
        groups = [
            (
                np.asarray([int(a)], dtype=np.int64),
                np.asarray([int(b)], dtype=np.int64),
                np.asarray([int(c)], dtype=np.int64),
                np.asarray([int(d)], dtype=np.int64),
            )
            for a, b, c, d in arr
        ]
        return _run_multi_dihedral_groups_native(
            native_traj,
            native_system,
            groups,
            mass,
            range360,
            chunk_frames,
            frame_indices,
        )

    commands = list(mask)
    groups = _multi_dihedral_command_groups(system, commands)
    return _run_multi_dihedral_groups_native(
        native_traj,
        native_system,
        groups,
        mass,
        range360,
        chunk_frames,
        frame_indices,
    )


__all__ = ["angle", "dihedral", "distance"]
