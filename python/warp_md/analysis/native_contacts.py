# Usage:
# from warp_md.analysis.native_contacts import native_contacts
# frac = native_contacts(traj, system, mask="@CA", mask2="@CB", ref=0)

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np

from ._runtime import (
    clone_native_system_with_positions0,
    coerce_native_system,
    is_native_traj,
    load_native_symbol,
    native_inputs,
    native_selection,
    prepend_reference_frame,
    read_all_frames,
    read_frame_subset,
    reset_traj,
    selection_indices,
)
from .trajectory import ArrayTrajectory


RefLike = Union[int, str]


def _frame_indices_arg(frame_indices: Optional[Sequence[int]]):
    return None if frame_indices is None else [int(value) for value in frame_indices]


def _has_positions0(system) -> bool:
    if not hasattr(system, "positions0"):
        return False
    try:
        return system.positions0() is not None
    except Exception:
        return False


def _run_native_contacts_plan(
    plan_cls,
    traj,
    system,
    native_system,
    mask: str,
    mask2: str,
    reference: str,
    cutoff: float,
    mindist: Optional[float],
    image: bool,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]],
):
    selection = native_selection(system, native_system, mask)
    selection_b = native_selection(system, native_system, mask2) if mask2 else None
    pbc_modes = ["orthorhombic", "none"] if image else ["none"]
    last_exc: Optional[Exception] = None
    for pbc in pbc_modes:
        try:
            if not reset_traj(traj):
                raise RuntimeError("failed to reset native trajectory")
            plan = plan_cls(
                selection,
                selection_b,
                reference=reference,
                cutoff=cutoff,
                pbc=pbc,
                min_cutoff=mindist,
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
        except Exception as exc:
            last_exc = exc
            if pbc == "orthorhombic" and "box" in str(exc).lower():
                continue
            break
    raise RuntimeError("native NativeContactsPlan execution failed") from last_exc


def _native_contacts_live(
    source,
    system,
    mask: str,
    mask2: str,
    ref: RefLike,
    cutoff: float,
    mindist: Optional[float],
    image: bool,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]],
):
    if not is_native_traj(source):
        return None
    plan_cls = load_native_symbol("NativeContactsPlan")
    if plan_cls is None:
        raise RuntimeError("NativeContactsPlan binding unavailable")

    native_system = coerce_native_system(system)
    if native_system is None:
        raise RuntimeError("failed to prepare native contact system")

    if isinstance(ref, str):
        key = ref.strip().lower()
        if key in ("frame0", "first", "0"):
            return _run_native_contacts_plan(
                plan_cls,
                source,
                system,
                native_system,
                mask,
                mask2,
                "frame0",
                cutoff,
                mindist,
                image,
                chunk_frames,
                frame_indices,
            )
        if key in ("topology", "top", "topo") and not image and _has_positions0(native_system):
            return _run_native_contacts_plan(
                plan_cls,
                source,
                system,
                native_system,
                mask,
                mask2,
                "topology",
                cutoff,
                mindist,
                False,
                chunk_frames,
                frame_indices,
            )
        return None

    if isinstance(ref, int) and int(ref) == 0:
        return _run_native_contacts_plan(
            plan_cls,
            source,
            system,
            native_system,
            mask,
            mask2,
            "frame0",
            cutoff,
            mindist,
            image,
            chunk_frames,
            frame_indices,
        )

    if not isinstance(ref, int) or image or frame_indices is not None:
        return None

    ref_coords, _ref_box, _ref_time, source_indices = read_frame_subset(
        source,
        [int(ref)],
        chunk_frames,
    )
    if ref_coords is None or ref_coords.shape[0] != 1 or source_indices.size != 1:
        raise ValueError("ref index out of range")
    ref_system = clone_native_system_with_positions0(system, ref_coords[0])
    if ref_system is None:
        raise RuntimeError("failed to prepare native reference contact system")
    return _run_native_contacts_plan(
        plan_cls,
        source,
        system,
        ref_system,
        mask,
        mask2,
        "topology",
        cutoff,
        mindist,
        False,
        chunk_frames,
        None,
    )


def _native_contacts(
    source,
    system,
    mask: str,
    mask2: str,
    ref: int,
    cutoff: float,
    mindist: Optional[float],
    image: bool,
    chunk_frames: Optional[int],
    frame_indices: Optional[Sequence[int]] = None,
):
    plan_cls = load_native_symbol("NativeContactsPlan")
    if plan_cls is None:
        raise RuntimeError("NativeContactsPlan binding unavailable")
    if is_native_traj(source):
        native_out = _native_contacts_live(
            source,
            system,
            mask,
            mask2,
            ref,
            cutoff,
            mindist,
            image,
            chunk_frames,
            frame_indices,
        )
        if native_out is not None:
            return native_out

    coords, box, _time = read_all_frames(source, chunk_frames, include_box=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    pbc = "orthorhombic" if image and box is not None else "none"
    source = ArrayTrajectory(np.asarray(coords, dtype=np.float32), box=box)
    native_traj, native_system = native_inputs(
        source,
        system,
        chunk_frames,
        include_box=(pbc == "orthorhombic"),
    )
    if native_traj is None or native_system is None:
        raise RuntimeError("failed to prepare native contact inputs")
    try:
        selection = native_selection(system, native_system, mask)
        selection_b = (
            native_selection(system, native_system, mask2) if mask2 else None
        )
        run_traj = native_traj
        trim_first = False
        if ref == 0:
            if not reset_traj(run_traj):
                raise RuntimeError("failed to reset native trajectory")
        else:
            run_traj = prepend_reference_frame(
                native_traj,
                ref,
                chunk_frames,
                include_box=(pbc == "orthorhombic"),
            )
            if run_traj is None:
                raise RuntimeError("failed to build native reference trajectory")
            trim_first = True
        plan = plan_cls(
            selection,
            selection_b,
            reference="frame0",
            cutoff=cutoff,
            pbc=pbc,
            min_cutoff=mindist,
        )
        values = np.asarray(
            plan.run(
                run_traj,
                native_system,
                chunk_frames=chunk_frames,
                device="auto",
                frame_indices=_frame_indices_arg(frame_indices),
            ),
            dtype=np.float32,
        )
        if trim_first:
            values = values[1:]
        return values
    except Exception as exc:
        raise RuntimeError("native NativeContactsPlan execution failed") from exc


def native_contacts(
    traj,
    system,
    mask: str = "",
    mask2: str = "",
    ref: RefLike = 0,
    distance: float = 7.0,
    mindist: Optional[float] = None,
    maxdist: Optional[float] = None,
    image: bool = True,
    include_solvent: bool = False,
    byres: bool = False,
    series: bool = False,
    first: bool = False,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
) -> np.ndarray:
    """Compute native contacts fraction per frame (basic parity)."""
    del include_solvent, byres, series, first
    cutoff = float(maxdist if maxdist is not None else distance)
    mindist_val = None if mindist is None else float(mindist)
    native_out = _native_contacts_live(
        traj,
        system,
        mask,
        mask2,
        ref,
        cutoff,
        mindist_val,
        image,
        chunk_frames,
        frame_indices,
    )
    if native_out is not None:
        return native_out

    if isinstance(ref, int) and int(ref) == 0:
        return _native_contacts(
            traj,
            system,
            mask,
            mask2,
            int(ref),
            cutoff,
            mindist_val,
            image,
            chunk_frames,
            frame_indices=frame_indices,
        )

    coords, box, _time = read_all_frames(traj, chunk_frames, include_box=True)
    if coords is None:
        raise ValueError("trajectory has no frames")
    if coords.size == 0:
        return np.empty((0,), dtype=np.float32)

    if isinstance(ref, int) and frame_indices is None:
        return _native_contacts(
            ArrayTrajectory(np.asarray(coords, dtype=np.float32), box=box),
            system,
            mask,
            mask2,
            int(ref),
            cutoff,
            mindist_val,
            image,
            chunk_frames,
        )

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
        if box is not None:
            box = box[select]
        n_frames = coords.shape[0]

    idx_a = selection_indices(system, mask)
    if mask2:
        idx_b = selection_indices(system, mask2)
    else:
        idx_b = idx_a

    if idx_a.size == 0 or idx_b.size == 0:
        return np.zeros(n_frames, dtype=np.float32)

    ref_coords = None
    if isinstance(ref, str):
        key = ref.strip().lower()
        if key in ("topology", "top", "topo"):
            if hasattr(system, "positions0"):
                ref_coords = system.positions0()
        if ref_coords is None and key in ("frame0", "first", "0", "topology", "top", "topo"):
            ref = 0

    if ref_coords is None:
        if isinstance(ref, int):
            ref_index = ref
            if ref_index < 0:
                ref_index = coords.shape[0] + ref_index
            if ref_index < 0 or ref_index >= coords.shape[0]:
                raise ValueError("ref index out of range")
            ref_coords = coords[ref_index]
        else:
            ref_coords = coords[0]
    else:
        ref_coords = np.asarray(ref_coords, dtype=np.float64)

    ref_pairs = _reference_pairs(ref_coords, idx_a, idx_b, cutoff, mindist_val, image, box[0] if box is not None else None)
    if len(ref_pairs) == 0:
        return np.zeros(n_frames, dtype=np.float32)

    out = np.zeros(n_frames, dtype=np.float64)
    for f in range(n_frames):
        out[f] = _count_pairs(coords[f], ref_pairs, cutoff, mindist_val, image, None if box is None else box[f])
    out /= float(len(ref_pairs))
    return out.astype(np.float32)


def _reference_pairs(
    coords: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    cutoff: float,
    mindist: Optional[float],
    image: bool,
    box: Optional[np.ndarray],
):
    pairs = []
    cutoff_sq = cutoff * cutoff
    mindist_sq = None if mindist is None else mindist * mindist
    same = idx_a is idx_b or np.array_equal(idx_a, idx_b)
    for i, a in enumerate(idx_a):
        pa = coords[a]
        start = i + 1 if same else 0
        for b in idx_b[start:]:
            if a == b:
                continue
            pb = coords[b]
            dx, dy, dz = pb - pa
            if image and box is not None:
                dx -= np.round(dx / box[0]) * box[0]
                dy -= np.round(dy / box[1]) * box[1]
                dz -= np.round(dz / box[2]) * box[2]
            dist2 = dx * dx + dy * dy + dz * dz
            if mindist_sq is not None and dist2 < mindist_sq:
                continue
            if dist2 <= cutoff_sq:
                pairs.append((int(a), int(b)))
    return pairs


def _count_pairs(
    coords: np.ndarray,
    pairs: Sequence[Tuple[int, int]],
    cutoff: float,
    mindist: Optional[float],
    image: bool,
    box: Optional[np.ndarray],
) -> float:
    cutoff_sq = cutoff * cutoff
    mindist_sq = None if mindist is None else mindist * mindist
    count = 0
    for a, b in pairs:
        pa = coords[a]
        pb = coords[b]
        dx, dy, dz = pb - pa
        if image and box is not None:
            dx -= np.round(dx / box[0]) * box[0]
            dy -= np.round(dy / box[1]) * box[1]
            dz -= np.round(dz / box[2]) * box[2]
        dist2 = dx * dx + dy * dy + dz * dz
        if mindist_sq is not None and dist2 < mindist_sq:
            continue
        if dist2 <= cutoff_sq:
            count += 1
    return float(count)


__all__ = ["native_contacts"]
