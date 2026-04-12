from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

import warp_md

from .h2order import _select


MaskLike = Union[str, Sequence[int], np.ndarray]

_MdmatPlan = (
    getattr(warp_md.traj_py, "PyMdmatPlan", None) if getattr(warp_md, "traj_py", None) else None
)


def _chain_ids(atoms: dict, n_atoms: int) -> list[int]:
    if "chain_id" in atoms and len(atoms["chain_id"]) == n_atoms:
        return [int(v) for v in atoms["chain_id"]]
    if "chain" in atoms and len(atoms["chain"]) == n_atoms:
        chain_map: dict[str, int] = {}
        out = []
        for raw in atoms["chain"]:
            key = str(raw)
            if key not in chain_map:
                chain_map[key] = len(chain_map)
            out.append(chain_map[key])
        return out
    return [0] * n_atoms


def _selected_residue_count(system, selection) -> Optional[int]:
    if not hasattr(system, "atom_table"):
        return None
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    resnames = atoms.get("resname", [])
    if not resids or not resnames or len(resids) != len(resnames):
        return None
    n_atoms = len(resids)
    chain_ids = _chain_ids(atoms, n_atoms)
    selected = {int(v) for v in selection.indices}
    if not selected:
        return 0
    count = 0
    start = 0
    while start < n_atoms:
        chain = chain_ids[start]
        resid = int(resids[start])
        resname = str(resnames[start])
        end = start + 1
        while (
            end < n_atoms
            and chain_ids[end] == chain
            and int(resids[end]) == resid
            and str(resnames[end]) == resname
        ):
            end += 1
        if any(idx in selected for idx in range(start, end)):
            count += 1
        start = end
    return count


def _estimate_frame_bytes(
    traj,
    system,
    selection,
    frame_indices: Optional[Sequence[int]],
    chunk_frames: Optional[int],
) -> Optional[int]:
    n_res = _selected_residue_count(system, selection)
    if n_res is None:
        return None
    if frame_indices is not None:
        n_frames = len(frame_indices)
    elif hasattr(traj, "count_frames"):
        try:
            n_frames = int(traj.count_frames(chunk_frames))
        except Exception:
            return None
    else:
        return None
    return int(4 * n_frames * n_res * n_res + 4 * n_frames)


def _write_frames_npz(path: Path, out: dict) -> None:
    np.savez_compressed(
        path,
        time=np.asarray(out["time"], dtype=np.float32),
        frame_matrices=np.asarray(out["frame_matrices"], dtype=np.float32),
        labels=np.asarray(out["labels"]),
        truncate=np.asarray(out["truncate"], dtype=np.float32),
        length_scale=np.asarray(out["length_scale"], dtype=np.float32),
    )


def mdmat(
    traj,
    system,
    selection: MaskLike = "",
    truncate: float = 1.5,
    include_contacts: bool = False,
    include_frames: bool = False,
    frames_mode: str = "auto",
    frames_out: Optional[Union[str, Path]] = None,
    memory_budget_bytes: int = 536870912,
    length_scale: Optional[float] = None,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
):
    """Residue-pair smallest-distance matrix via Rust plan path."""
    if _MdmatPlan is None:
        raise RuntimeError(
            "PyMdmatPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    frames_mode_norm = str(frames_mode).strip().lower()
    if frames_mode_norm not in {"auto", "memory", "artifact"}:
        raise ValueError("frames_mode must be 'auto', 'memory', or 'artifact'")
    if frames_out is not None and not include_frames:
        raise ValueError("frames_out requires include_frames=True")
    if frames_mode_norm == "artifact" and not include_frames:
        raise ValueError("artifact frames_mode requires include_frames=True")
    if frames_mode_norm == "artifact" and frames_out is None:
        raise ValueError("frames_mode='artifact' requires frames_out")
    if frames_mode_norm == "memory" and frames_out is not None:
        raise ValueError("frames_out is incompatible with frames_mode='memory'")

    sel = _select(system, selection)
    if include_frames and frames_out is None and frames_mode_norm == "auto":
        estimate = _estimate_frame_bytes(traj, system, sel, frame_indices, chunk_frames)
        if estimate is not None and estimate > int(memory_budget_bytes):
            raise RuntimeError(
                "mdmat frame matrices exceed memory_budget_bytes; pass frames_out or increase memory_budget_bytes."
            )

    plan = _MdmatPlan(
        sel,
        truncate=float(truncate),
        include_contacts=bool(include_contacts),
        include_frames=bool(include_frames),
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
            "mdmat requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    result = {
        "labels": tuple(str(v) for v in out["labels"]),
        "mean_matrix": np.asarray(out["mean_matrix"], dtype=np.float32),
        "frames": int(out["frames"]),
        "residues": int(out["residues"]),
        "truncate": float(out["truncate"]),
        "used_box": bool(out["used_box"]),
        "length_scale": float(out["length_scale"]),
    }
    if out.get("distinct_contact_atoms") is not None:
        result["distinct_contact_atoms"] = np.asarray(out["distinct_contact_atoms"], dtype=np.uint32)
        result["mean_contact_atoms"] = np.asarray(out["mean_contact_atoms"], dtype=np.float32)
        result["contact_ratio"] = np.asarray(out["contact_ratio"], dtype=np.float32)
        result["residue_atom_counts"] = np.asarray(out["residue_atom_counts"], dtype=np.uint32)
        result["mean_contact_atoms_per_residue_atom"] = np.asarray(
            out["mean_contact_atoms_per_residue_atom"], dtype=np.float32
        )
    if out.get("frame_matrices") is not None:
        result["time"] = np.asarray(out["time"], dtype=np.float32)
        result["frame_matrices"] = np.asarray(out["frame_matrices"], dtype=np.float32)
        if frames_out is not None:
            out_path = Path(frames_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _write_frames_npz(out_path, result)
            result["frames_artifact"] = {
                "path": str(out_path),
                "format": "npz",
                "kind": "timeseries",
            }
            del result["time"]
            del result["frame_matrices"]
    return result


__all__ = ["mdmat"]
