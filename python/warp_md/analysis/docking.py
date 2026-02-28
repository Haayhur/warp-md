from __future__ import annotations

import math
from collections import defaultdict
from html import escape
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

MaskLike = Union[str, Sequence[int], np.ndarray]

_INTERACTION_CODES: Dict[int, str] = {
    1: "hydrogen_bond",
    2: "hydrophobic_contact",
    3: "close_contact",
    4: "clash",
    5: "salt_bridge",
    6: "halogen_bond",
    7: "metal_coordination",
    8: "cation_pi",
    9: "pi_pi_stacking",
}

_INTERACTION_COLORS: Dict[str, str] = {
    "hydrogen_bond": "#1f77b4",
    "hydrophobic_contact": "#2ca02c",
    "close_contact": "#7f7f7f",
    "clash": "#d62728",
    "salt_bridge": "#ff7f0e",
    "halogen_bond": "#17becf",
    "metal_coordination": "#8c564b",
    "cation_pi": "#bcbd22",
    "pi_pi_stacking": "#e377c2",
}


def _all_resid_mask(system) -> str:
    if not hasattr(system, "atom_table"):
        return "all"
    atoms = system.atom_table()
    resids = atoms.get("resid", [])
    if not resids:
        return "resid 0:0"
    return f"resid {min(resids)}:{max(resids)}"


def _to_selection(system, mask: MaskLike):
    if isinstance(mask, str):
        normalized = mask if mask not in ("", "*", "all", None) else _all_resid_mask(system)
        return system.select(normalized)
    indices = np.asarray(mask, dtype=np.int64).reshape(-1).tolist()
    return system.select_indices(indices)


def _require_bool(value, name: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be bool")


def docking(
    traj,
    system,
    receptor_mask: MaskLike,
    ligand_mask: MaskLike,
    close_contact_cutoff: float = 4.0,
    hydrophobic_cutoff: float = 4.0,
    hydrogen_bond_cutoff: float = 3.5,
    clash_cutoff: float = 2.5,
    salt_bridge_cutoff: float = 5.5,
    halogen_bond_cutoff: float = 5.5,
    metal_coordination_cutoff: float = 3.5,
    cation_pi_cutoff: float = 6.0,
    pi_pi_cutoff: float = 7.5,
    hbond_min_angle_deg: float = 120.0,
    donor_hydrogen_cutoff: float = 1.25,
    allow_missing_hydrogen: bool = True,
    length_scale: float = 1.0,
    max_events_per_frame: int = 20_000,
    frame_indices: Optional[Sequence[int]] = None,
    chunk_frames: Optional[int] = None,
    device: str = "auto",
) -> dict:
    """Analyze docking-simulation interactions in a LigPlot/BINANA-inspired contract."""
    if not np.isfinite(close_contact_cutoff) or float(close_contact_cutoff) <= 0.0:
        raise ValueError("close_contact_cutoff must be finite and > 0")
    if not np.isfinite(hydrophobic_cutoff) or float(hydrophobic_cutoff) <= 0.0:
        raise ValueError("hydrophobic_cutoff must be finite and > 0")
    if not np.isfinite(hydrogen_bond_cutoff) or float(hydrogen_bond_cutoff) <= 0.0:
        raise ValueError("hydrogen_bond_cutoff must be finite and > 0")
    if not np.isfinite(clash_cutoff) or float(clash_cutoff) < 0.0:
        raise ValueError("clash_cutoff must be finite and >= 0")
    if float(hydrophobic_cutoff) > float(close_contact_cutoff):
        raise ValueError("hydrophobic_cutoff must be <= close_contact_cutoff")
    if float(hydrogen_bond_cutoff) > float(close_contact_cutoff):
        raise ValueError("hydrogen_bond_cutoff must be <= close_contact_cutoff")
    if float(clash_cutoff) > float(close_contact_cutoff):
        raise ValueError("clash_cutoff must be <= close_contact_cutoff")
    if not np.isfinite(salt_bridge_cutoff) or float(salt_bridge_cutoff) <= 0.0:
        raise ValueError("salt_bridge_cutoff must be finite and > 0")
    if not np.isfinite(halogen_bond_cutoff) or float(halogen_bond_cutoff) <= 0.0:
        raise ValueError("halogen_bond_cutoff must be finite and > 0")
    if not np.isfinite(metal_coordination_cutoff) or float(metal_coordination_cutoff) <= 0.0:
        raise ValueError("metal_coordination_cutoff must be finite and > 0")
    if not np.isfinite(cation_pi_cutoff) or float(cation_pi_cutoff) <= 0.0:
        raise ValueError("cation_pi_cutoff must be finite and > 0")
    if not np.isfinite(pi_pi_cutoff) or float(pi_pi_cutoff) <= 0.0:
        raise ValueError("pi_pi_cutoff must be finite and > 0")
    if not np.isfinite(hbond_min_angle_deg) or float(hbond_min_angle_deg) <= 0.0 or float(
        hbond_min_angle_deg
    ) > 180.0:
        raise ValueError("hbond_min_angle_deg must be finite and in (0, 180]")
    if not np.isfinite(donor_hydrogen_cutoff) or float(donor_hydrogen_cutoff) <= 0.0:
        raise ValueError("donor_hydrogen_cutoff must be finite and > 0")
    allow_missing_hydrogen = _require_bool(allow_missing_hydrogen, "allow_missing_hydrogen")
    if not np.isfinite(length_scale) or float(length_scale) <= 0.0:
        raise ValueError("length_scale must be finite and > 0")
    if int(max_events_per_frame) < 1:
        raise ValueError("max_events_per_frame must be >= 1")

    try:
        from warp_md import DockingPlan  # type: ignore
    except Exception:
        raise RuntimeError(
            "DockingPlan binding unavailable. Rebuild bindings with `maturin develop`."
        )
    if getattr(DockingPlan, "__name__", "") == "_Missing":
        raise RuntimeError(
            "DockingPlan binding unavailable in this build. Rebuild bindings with `maturin develop`."
        )

    receptor = _to_selection(system, receptor_mask)
    ligand = _to_selection(system, ligand_mask)
    frame_indices_list = (
        None if frame_indices is None else [int(value) for value in frame_indices]
    )
    try:
        plan = DockingPlan(
            receptor,
            ligand,
            close_contact_cutoff=float(close_contact_cutoff),
            hydrophobic_cutoff=float(hydrophobic_cutoff),
            hydrogen_bond_cutoff=float(hydrogen_bond_cutoff),
            clash_cutoff=float(clash_cutoff),
            salt_bridge_cutoff=float(salt_bridge_cutoff),
            halogen_bond_cutoff=float(halogen_bond_cutoff),
            metal_coordination_cutoff=float(metal_coordination_cutoff),
            cation_pi_cutoff=float(cation_pi_cutoff),
            pi_pi_cutoff=float(pi_pi_cutoff),
            hbond_min_angle_deg=float(hbond_min_angle_deg),
            donor_hydrogen_cutoff=float(donor_hydrogen_cutoff),
            allow_missing_hydrogen=allow_missing_hydrogen,
            length_scale=float(length_scale),
            max_events_per_frame=int(max_events_per_frame),
        )
        out = np.asarray(
            plan.run(
                traj,
                system,
                chunk_frames=chunk_frames,
                device=device,
                frame_indices=frame_indices_list,
            ),
            dtype=np.float32,
        )
    except TypeError as exc:
        raise RuntimeError(
            "docking requires Rust-backed trajectory/system objects when using the Rust plan path."
        ) from exc

    if out.ndim != 2 or out.shape[1] != 6:
        raise RuntimeError("docking output shape mismatch: expected (n_events, 6)")
    atoms = system.atom_table() if hasattr(system, "atom_table") else {}
    names = atoms.get("name", []) if isinstance(atoms, dict) else []
    resnames = atoms.get("resname", []) if isinstance(atoms, dict) else []
    resids = atoms.get("resid", []) if isinstance(atoms, dict) else []
    chain_ids = atoms.get("chain_id", []) if isinstance(atoms, dict) else []

    def _atom_meta(atom_index: int) -> dict:
        return {
            "atom_index": atom_index,
            "atom_name": names[atom_index] if atom_index < len(names) else "",
            "resname": resnames[atom_index] if atom_index < len(resnames) else "",
            "resid": int(resids[atom_index]) if atom_index < len(resids) else -1,
            "chain_id": int(chain_ids[atom_index]) if atom_index < len(chain_ids) else 0,
        }

    events: List[dict] = []
    counts_by_type: Dict[str, int] = {name: 0 for name in _INTERACTION_CODES.values()}
    per_frame_counts = defaultdict(lambda: {name: 0 for name in _INTERACTION_CODES.values()})
    residue_counts = defaultdict(lambda: {name: 0 for name in _INTERACTION_CODES.values()})
    frame_set = set()

    for row in out:
        frame_index = int(round(float(row[0])))
        receptor_atom_idx = int(round(float(row[1])))
        ligand_atom_idx = int(round(float(row[2])))
        interaction_code = int(round(float(row[3])))
        distance = float(row[4])
        strength = float(row[5])
        interaction_name = _INTERACTION_CODES.get(interaction_code, "unknown")
        frame_set.add(frame_index)
        if interaction_name in counts_by_type:
            counts_by_type[interaction_name] += 1
            per_frame_counts[frame_index][interaction_name] += 1
            receptor_meta = _atom_meta(receptor_atom_idx)
            residue_key = (
                receptor_meta["chain_id"],
                receptor_meta["resid"],
                receptor_meta["resname"],
            )
            residue_counts[residue_key][interaction_name] += 1
        else:
            receptor_meta = _atom_meta(receptor_atom_idx)
        ligand_meta = _atom_meta(ligand_atom_idx)
        events.append(
            {
                "frame_index": frame_index,
                "interaction_code": interaction_code,
                "interaction_type": interaction_name,
                "distance": distance,
                "strength": strength,
                "receptor_atom": receptor_meta,
                "ligand_atom": ligand_meta,
            }
        )

    frames = []
    for frame_index in sorted(frame_set):
        frame_counts = per_frame_counts[frame_index]
        frames.append(
            {
                "frame_index": frame_index,
                "interaction_count": int(sum(frame_counts.values())),
                "counts_by_type": {k: int(v) for k, v in frame_counts.items()},
            }
        )

    residues = []
    for chain_id, resid, resname in sorted(residue_counts.keys()):
        counts = residue_counts[(chain_id, resid, resname)]
        residues.append(
            {
                "chain_id": int(chain_id),
                "resid": int(resid),
                "resname": str(resname),
                "interaction_count": int(sum(counts.values())),
                "counts_by_type": {k: int(v) for k, v in counts.items()},
            }
        )

    return {
        "schema_version": "warp_md.docking.interactions.v1",
        "interaction_codes": {v: k for k, v in _INTERACTION_CODES.items()},
        "parameters": {
            "close_contact_cutoff": float(close_contact_cutoff),
            "hydrophobic_cutoff": float(hydrophobic_cutoff),
            "hydrogen_bond_cutoff": float(hydrogen_bond_cutoff),
            "clash_cutoff": float(clash_cutoff),
            "salt_bridge_cutoff": float(salt_bridge_cutoff),
            "halogen_bond_cutoff": float(halogen_bond_cutoff),
            "metal_coordination_cutoff": float(metal_coordination_cutoff),
            "cation_pi_cutoff": float(cation_pi_cutoff),
            "pi_pi_cutoff": float(pi_pi_cutoff),
            "hbond_min_angle_deg": float(hbond_min_angle_deg),
            "donor_hydrogen_cutoff": float(donor_hydrogen_cutoff),
            "allow_missing_hydrogen": bool(allow_missing_hydrogen),
            "length_scale": float(length_scale),
            "max_events_per_frame": int(max_events_per_frame),
        },
        "summary": {
            "n_events": int(out.shape[0]),
            "n_frames": int(len(frame_set)),
            "counts_by_type": {k: int(v) for k, v in counts_by_type.items()},
        },
        "frames": frames,
        "residues": residues,
        "events": events,
        "raw": out.astype(np.float32, copy=False),
    }


def docking_ligplot_svg(
    result: dict,
    max_residues: int = 24,
    width: int = 960,
    height: int = 700,
    title: str = "Docking Interaction Map",
    path: Optional[str] = None,
) -> str:
    """Render a compact LigPlot-style interaction map from `docking()` output."""
    residues = list(result.get("residues", []))
    if not residues:
        raise ValueError("result has no residues to render")
    if int(max_residues) < 1:
        raise ValueError("max_residues must be >= 1")
    width = int(width)
    height = int(height)
    if width < 320 or height < 240:
        raise ValueError("width/height too small for readable SVG")

    residues = sorted(
        residues,
        key=lambda row: (
            -int(row.get("interaction_count", 0)),
            int(row.get("chain_id", 0)),
            int(row.get("resid", 0)),
            str(row.get("resname", "")),
        ),
    )[: int(max_residues)]
    n_nodes = len(residues)
    cx = width * 0.5
    cy = height * 0.52
    radius = min(width, height) * 0.34

    def _dominant_kind(counts: dict) -> str:
        if not counts:
            return "close_contact"
        return max(
            _INTERACTION_CODES.values(),
            key=lambda kind: (int(counts.get(kind, 0)), kind),
        )

    lines: List[str] = []
    circles: List[str] = []
    labels: List[str] = []
    for idx, residue in enumerate(residues):
        angle = -math.pi / 2.0 + 2.0 * math.pi * idx / max(n_nodes, 1)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        counts = residue.get("counts_by_type", {})
        dominant = _dominant_kind(counts)
        color = _INTERACTION_COLORS.get(dominant, "#7f7f7f")
        count = int(residue.get("interaction_count", 0))
        stroke_width = 1.5 + min(4.0, 0.1 * float(count))
        lines.append(
            f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="{stroke_width:.2f}" stroke-opacity="0.85" />'
        )
        circles.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="12" fill="{color}" fill-opacity="0.2" '
            f'stroke="{color}" stroke-width="1.5" />'
        )
        label = (
            f'{residue.get("resname", "")} {residue.get("resid", "")}'
            f' (chain {residue.get("chain_id", 0)}, n={count})'
        )
        anchor = "start" if x >= cx else "end"
        lx = x + (18 if x >= cx else -18)
        labels.append(
            f'<text x="{lx:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" '
            f'font-family="monospace" font-size="12" fill="#222">{escape(label)}</text>'
        )

    legend_items = []
    used_kinds = [kind for kind in _INTERACTION_CODES.values() if kind in _INTERACTION_COLORS]
    for idx, kind in enumerate(used_kinds):
        y = 46 + idx * 18
        color = _INTERACTION_COLORS[kind]
        legend_items.append(
            f'<rect x="28" y="{y - 9}" width="12" height="12" fill="{color}" fill-opacity="0.25" '
            f'stroke="{color}" stroke-width="1" />'
        )
        legend_items.append(
            f'<text x="46" y="{y}" font-family="monospace" font-size="11" fill="#333">{escape(kind)}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">'
        '<rect x="0" y="0" width="100%" height="100%" fill="#fbfbf8" />'
        f'<text x="{width * 0.5:.1f}" y="28" text-anchor="middle" font-family="monospace" '
        f'font-size="18" fill="#222">{escape(title)}</text>'
        f'<text x="{width * 0.5:.1f}" y="{cy - 26:.1f}" text-anchor="middle" font-family="monospace" '
        'font-size="12" fill="#444">Ligand</text>'
        f'<rect x="{cx - 42:.1f}" y="{cy - 16:.1f}" width="84" height="32" rx="8" ry="8" '
        'fill="#fff5db" stroke="#444" stroke-width="1.5" />'
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="monospace" '
        'font-size="13" font-weight="bold" fill="#222">LIG</text>'
        f'{"".join(lines)}'
        f'{"".join(circles)}'
        f'{"".join(labels)}'
        f'{"".join(legend_items)}'
        "</svg>"
    )
    if path is not None:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(svg)
    return svg


__all__ = ["docking", "docking_ligplot_svg"]
