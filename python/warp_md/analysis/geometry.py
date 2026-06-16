# Usage:
# from warp_md.analysis.geometry import angle, dihedral, distance
# vals = angle(traj, system, "@1 @2 @3")

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._chunk_io import read_chunk_fields
from ._runtime import (
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_inputs,
    read_all_frames,
    subset_frames,
)
from .trajectory import ArrayTrajectory


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


def _frame_subset(coords, box, time, frame_indices):
    coords, box, time = subset_frames(coords, frame_indices, box=box, time=time)
    return np.asarray(coords, dtype=np.float64), box, time


def _read_all(traj, chunk_frames: Optional[int]):
    max_chunk = chunk_frames or 128
    coords_list = []
    chunk = read_chunk_fields(traj, max_chunk)
    if chunk is None:
        return None
    while chunk is not None:
        coords_list.append(np.asarray(chunk["coords"], dtype=np.float64))
        chunk = read_chunk_fields(traj, max_chunk)
    return np.concatenate(coords_list, axis=0) if coords_list else np.empty((0, 0, 3))


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


def _box_lengths_for_distance(box: Optional[np.ndarray], n_frames: int) -> Optional[np.ndarray]:
    if box is None:
        return None
    arr = np.asarray(box, dtype=np.float64)
    if arr.ndim == 1 and arr.size >= 3:
        arr = np.broadcast_to(arr[:3], (n_frames, 3))
    if arr.ndim == 2 and arr.shape[1] >= 3:
        return arr[:, :3]
    return None


def _apply_orthorhombic_image(delta: np.ndarray, box: Optional[np.ndarray]) -> np.ndarray:
    lengths = _box_lengths_for_distance(box, delta.shape[0])
    if lengths is None:
        return delta
    valid = np.all(lengths > 0.0, axis=1)
    if not np.any(valid):
        return delta
    out = delta.copy()
    out[valid] -= np.round(out[valid] / lengths[valid]) * lengths[valid]
    return out


def _distance_dtype(out, dtype: str):
    key = str(dtype).lower()
    if key == "dict":
        return {"distance": np.asarray(out, dtype=np.float32)}
    return np.asarray(out, dtype=np.float32)


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _native_selected(native_system, indices: np.ndarray):
    if not hasattr(native_system, "select_indices"):
        raise RuntimeError("native system does not support index selections")
    return native_system.select_indices(np.asarray(indices, dtype=np.int64).tolist())


def _native_inputs_from_arrays(coords, box, system, chunk_frames):
    source = ArrayTrajectory(coords, box=box)
    native_traj, native_system = native_inputs(
        source,
        system,
        chunk_frames,
        include_box=box is not None,
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native distance inputs")
    return native_traj, native_system


def _native_distance_system(system):
    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native distance system")
    return native_system


def _run_pair_list_distance_native(traj, native_system, pairs, chunk_frames, frame_indices):
    plan_cls = load_native_symbol("PairListDistancePlan")
    if plan_cls is None:
        raise RuntimeError(
            "PairListDistancePlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    plan = plan_cls(np.asarray(pairs, dtype=np.int64), pbc="none")
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


def _run_pair_list_distance(coords, box, system, pairs, chunk_frames):
    native_traj, native_system = _native_inputs_from_arrays(coords, box, system, chunk_frames)
    return _run_pair_list_distance_native(
        native_traj, native_system, pairs, chunk_frames, frame_indices=None
    )


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


def _run_mask_distance(coords, box, system, mask_a, mask_b, mass, image, chunk_frames):
    native_traj, native_system = _native_inputs_from_arrays(coords, box, system, chunk_frames)
    return _run_mask_distance_native(
        native_traj,
        native_system,
        system,
        mask_a,
        mask_b,
        mass,
        image and box is not None,
        chunk_frames,
        frame_indices=None,
    )


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


def _run_center_to_point_distance(coords, box, system, mask, point, mass, image, chunk_frames):
    native_traj, native_system = _native_inputs_from_arrays(coords, box, system, chunk_frames)
    return _run_center_to_point_distance_native(
        native_traj,
        native_system,
        system,
        mask,
        point,
        mass,
        image and box is not None,
        chunk_frames,
        frame_indices=None,
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


def _run_center_to_reference_distance(
    coords, box, system, mask, reference, mass, image, chunk_frames
):
    native_traj, native_system = _native_inputs_from_arrays(coords, box, system, chunk_frames)
    return _run_center_to_reference_distance_native(
        native_traj,
        native_system,
        system,
        mask,
        reference,
        mass,
        image and box is not None,
        chunk_frames,
        frame_indices=None,
    )


def _reference_center(coords, system, mask, ref, mass):
    if isinstance(ref, (int, np.integer)):
        ref_idx = int(ref)
        if ref_idx < 0:
            ref_idx += coords.shape[0]
        if ref_idx < 0 or ref_idx >= coords.shape[0]:
            raise ValueError("ref index out of range")
        ref_arr = coords[ref_idx]
    else:
        ref_arr = np.asarray(ref, dtype=np.float64)
    if ref_arr.shape == (3,):
        return ref_arr
    if ref_arr.ndim != 2 or ref_arr.shape[1] != 3:
        raise ValueError("ref must be a frame index, point, or coordinate frame")
    idx = _selection_indices(system, mask)
    masses = None
    if mass:
        atoms = system.atom_table()
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)
    return _center(ref_arr[None, :, :], idx, masses, mass)[0]


def _angle_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    dot = np.einsum("ij,ij->i", v1, v2)
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    with np.errstate(invalid="ignore", divide="ignore"):
        cosang = np.where(denom > 0.0, dot / denom, 1.0)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _dihedral_from_points(p0, p1, p2, p3):
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1, axis=1)
    b1n = b1 / b1_norm[:, None]
    v = b0 - (np.einsum("ij,ij->i", b0, b1n))[:, None] * b1n
    w = b2 - (np.einsum("ij,ij->i", b2, b1n))[:, None] * b1n
    x = np.einsum("ij,ij->i", v, w)
    y = np.einsum("ij,ij->i", np.cross(b1n, v), w)
    return np.degrees(np.arctan2(y, x))


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
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32)

    n_frames = coords.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        coords = coords[select]
        n_frames = coords.shape[0]

    masses = None
    if mass:
        atoms = system.atom_table()
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)

    if isinstance(mask, str):
        parts = mask.split()
        if len(parts) != 3:
            raise ValueError("angle mask must have 3 parts")
        idx_a = _selection_indices(system, parts[0])
        idx_b = _selection_indices(system, parts[1])
        idx_c = _selection_indices(system, parts[2])
        a = _center(coords, idx_a, masses, mass)
        b = _center(coords, idx_b, masses, mass)
        c = _center(coords, idx_c, masses, mass)
        v1 = a - b
        v2 = c - b
        out = _angle_from_vectors(v1, v2).astype(np.float32)
        return out

    arr = np.asarray(mask)
    if arr.dtype.kind == "i":
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("angle index array must be shape (n, 3)")
        out = np.zeros((arr.shape[0], n_frames), dtype=np.float64)
        for i, (a, b, c) in enumerate(arr):
            pa = coords[:, a, :]
            pb = coords[:, b, :]
            pc = coords[:, c, :]
            v1 = pa - pb
            v2 = pc - pb
            out[i] = _angle_from_vectors(v1, v2)
        return out.astype(np.float32)

    # list of strings
    commands = list(mask)
    out = np.zeros((len(commands), n_frames), dtype=np.float64)
    for i, cmd in enumerate(commands):
        out[i] = angle(traj, system, cmd, frame_indices=frame_indices, mass=mass, chunk_frames=chunk_frames)
    return out.astype(np.float32)


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
) -> np.ndarray:
    """Compute distance measurements.

    String masks compute one COM/COG distance per frame. Lists of mask strings
    return one row per command. Integer arrays of shape ``(n_pairs, 2)`` return
    explicit atom-pair distances with shape ``(n_pairs, n_frames)``.
    """
    command = np.asarray(mask)
    if is_native_traj(traj):
        native_system = _native_distance_system(system)
        if command.dtype.kind in ("i", "u"):
            pairs = command.astype(np.int64, copy=False)
            if pairs.ndim == 1:
                if pairs.size != 2:
                    raise ValueError("distance index array must be shape (n_pairs, 2)")
                pairs = pairs.reshape(1, 2)
            if pairs.ndim != 2 or pairs.shape[1] != 2:
                raise ValueError("distance index array must be shape (n_pairs, 2)")
            values = _run_pair_list_distance_native(
                traj, native_system, pairs, chunk_frames, frame_indices
            )
            return _distance_dtype(values, dtype)

        if isinstance(mask, (list, tuple, np.ndarray)) and not isinstance(mask, str):
            values = _run_multi_mask_distance_native(
                traj,
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
                if isinstance(ref, (int, np.integer)) and int(ref) == 0:
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
                if isinstance(ref, str) and ref.strip().lower() in (
                    "topology",
                    "top",
                    "topo",
                ):
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
            else:
                mask_a, mask_b = _split_distance_mask(str(mask))
                values = _run_mask_distance_native(
                    traj,
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

    coords, box, _time = read_all_frames(traj, chunk_frames, include_box=bool(image))
    if coords is None:
        raise ValueError("trajectory has no frames")
    coords, box, _time = _frame_subset(coords, box, None, frame_indices)
    if coords.size == 0:
        empty = np.empty((0,), dtype=np.float32)
        return _distance_dtype(empty, dtype)

    if command.dtype.kind in ("i", "u"):
        pairs = command.astype(np.int64, copy=False)
        if pairs.ndim == 1:
            if pairs.size != 2:
                raise ValueError("distance index array must be shape (n_pairs, 2)")
            pairs = pairs.reshape(1, 2)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("distance index array must be shape (n_pairs, 2)")
        values = _run_pair_list_distance(coords, box, system, pairs, chunk_frames)
        return _distance_dtype(values, dtype)

    if isinstance(mask, (list, tuple, np.ndarray)) and not isinstance(mask, str):
        rows = [
            np.asarray(
                distance(
                    traj=_ArrayReplay(coords, box),
                    system=system,
                    mask=str(command),
                    dtype="ndarray",
                    image=image,
                    mass=mass,
                    chunk_frames=chunk_frames,
                ),
                dtype=np.float32,
            )
            for command in list(mask)
        ]
        return _distance_dtype(np.vstack(rows) if rows else np.empty((0, coords.shape[0])), dtype)

    if point is not None and ref is not None:
        raise ValueError("distance accepts only one of point or ref")
    if point is not None:
        values = _run_center_to_point_distance(
            coords, box, system, str(mask), point, mass, image, chunk_frames
        )
        return _distance_dtype(values, dtype)
    if ref is not None:
        if isinstance(ref, (int, np.integer)) and int(ref) == 0:
            values = _run_center_to_reference_distance(
                coords, box, system, str(mask), "frame0", mass, image, chunk_frames
            )
        elif isinstance(ref, str) and ref.strip().lower() in ("topology", "top", "topo"):
            values = _run_center_to_reference_distance(
                coords, box, system, str(mask), "topology", mass, image, chunk_frames
            )
        else:
            target = _reference_center(coords, system, str(mask), ref, mass)
            values = _run_center_to_point_distance(
                coords, box, system, str(mask), target, mass, image, chunk_frames
            )
        return _distance_dtype(values, dtype)

    mask_a, mask_b = _split_distance_mask(str(mask))
    values = _run_mask_distance(coords, box, system, mask_a, mask_b, mass, image, chunk_frames)
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
    coords = _read_all(traj, chunk_frames)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32)

    n_frames = coords.shape[0]
    if frame_indices is not None:
        select = []
        for i in frame_indices:
            j = int(i)
            if j < 0:
                j = n_frames + j
            if 0 <= j < n_frames:
                select.append(j)
        coords = coords[select]
        n_frames = coords.shape[0]

    masses = None
    if mass:
        atoms = system.atom_table()
        masses = np.asarray(atoms.get("mass", []), dtype=np.float64)

    if isinstance(mask, str):
        parts = mask.split()
        if len(parts) != 4:
            raise ValueError("dihedral mask must have 4 parts")
        idx_a = _selection_indices(system, parts[0])
        idx_b = _selection_indices(system, parts[1])
        idx_c = _selection_indices(system, parts[2])
        idx_d = _selection_indices(system, parts[3])
        a = _center(coords, idx_a, masses, mass)
        b = _center(coords, idx_b, masses, mass)
        c = _center(coords, idx_c, masses, mass)
        d = _center(coords, idx_d, masses, mass)
        out = _dihedral_from_points(a, b, c, d)
        if range360:
            out = np.where(out < 0.0, out + 360.0, out)
        return out.astype(np.float32)

    arr = np.asarray(mask)
    if arr.dtype.kind == "i":
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError("dihedral index array must be shape (n, 4)")
        out = np.zeros((arr.shape[0], n_frames), dtype=np.float64)
        for i, (a, b, c, d) in enumerate(arr):
            pa = coords[:, a, :]
            pb = coords[:, b, :]
            pc = coords[:, c, :]
            pd = coords[:, d, :]
            vals = _dihedral_from_points(pa, pb, pc, pd)
            if range360:
                vals = np.where(vals < 0.0, vals + 360.0, vals)
            out[i] = vals
        return out.astype(np.float32)

    commands = list(mask)
    out = np.zeros((len(commands), n_frames), dtype=np.float64)
    for i, cmd in enumerate(commands):
        out[i] = dihedral(traj, system, cmd, frame_indices=frame_indices, mass=mass, range360=range360, chunk_frames=chunk_frames)
    return out.astype(np.float32)


class _ArrayReplay:
    def __init__(self, coords, box=None):
        self._coords = np.asarray(coords, dtype=np.float32)
        self._box = None if box is None else np.asarray(box, dtype=np.float32)
        self._used = False

    def read_chunk(self, _max_frames=128, **_kwargs):
        if self._used:
            return None
        self._used = True
        payload = {"coords": self._coords}
        if self._box is not None:
            payload["box"] = self._box
        return payload


__all__ = ["angle", "dihedral", "distance"]
