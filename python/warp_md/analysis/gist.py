from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from ._chunk_io import read_chunk_fields

K_BOLTZ = 0.008314462618  # kJ/mol/K


@dataclass
class GistConfig:
    grid_spacing: float = 0.1
    padding: float = 0.5
    temperature: float = 300.0
    length_scale: float = 0.1
    cutoff: float = 1.0
    orientation_bins: int = 12
    chunk_frames: Optional[int] = None
    frame_indices: Optional[Sequence[int]] = None
    solute_selection: Optional[str] = None
    max_frames: Optional[int] = None
    water_resnames: Sequence[str] = ("HOH", "WAT", "SOL", "TIP3", "OPC")
    bulk_density: Optional[float] = None
    energy_method: str = "direct"  # "direct", "pme", or "none"
    pme_totals_source: str = "openmm"  # "openmm", "native", or "direct_approx"


@dataclass
class GistResult:
    density: np.ndarray
    energy_sw: np.ndarray
    energy_ww: np.ndarray
    ts_trans: np.ndarray
    ts_orient: np.ndarray
    counts: np.ndarray
    origin: np.ndarray
    spacing: float
    n_frames: int
    voxel_volume: float
    bulk_density: float

    def as_dict(self):
        return {
            "density": self.density,
            "energy_sw": self.energy_sw,
            "energy_ww": self.energy_ww,
            "ts_trans": self.ts_trans,
            "ts_orient": self.ts_orient,
            "counts": self.counts,
            "origin": self.origin,
            "spacing": self.spacing,
            "n_frames": self.n_frames,
            "voxel_volume": self.voxel_volume,
            "bulk_density": self.bulk_density,
        }


def _require_openmm():
    try:
        from openmm import unit, openmm as mm
    except Exception as exc:  # pragma: no cover
        raise ImportError("openmm is required for gist") from exc
    return mm, unit


def _find_nonbonded(system):
    mm, _ = _require_openmm()
    for force in system.getForces():
        if isinstance(force, mm.NonbondedForce):
            return force
    raise ValueError("openmm_system has no NonbondedForce")


def _nonbonded_parameters(openmm_system):
    _, unit = _require_openmm()
    nb = _find_nonbonded(openmm_system)
    n_particles = nb.getNumParticles()
    charges = np.zeros(n_particles, dtype=np.float64)
    sigmas = np.zeros(n_particles, dtype=np.float64)
    epsilons = np.zeros(n_particles, dtype=np.float64)
    for i in range(nb.getNumParticles()):
        charge, sigma, epsilon = nb.getParticleParameters(i)
        charges[i] = float(charge.value_in_unit(unit.elementary_charge))
        sigmas[i] = float(sigma.value_in_unit(unit.nanometer))
        epsilons[i] = float(epsilon.value_in_unit(unit.kilojoule_per_mole))

    exceptions = []
    for i in range(nb.getNumExceptions()):
        p1, p2, q, sigma, epsilon = nb.getExceptionParameters(i)
        exceptions.append(
            (
                int(p1),
                int(p2),
                float(q.value_in_unit(unit.elementary_charge**2)),
                float(sigma.value_in_unit(unit.nanometer)),
                float(epsilon.value_in_unit(unit.kilojoule_per_mole)),
            )
        )
    return charges, sigmas, epsilons, exceptions, bool(openmm_system.usesPeriodicBoundaryConditions())


def _water_groups(topology, water_resnames: Iterable[str]):
    resnames = {r.upper() for r in water_resnames}
    waters = []
    atom_indices = {atom: idx for idx, atom in enumerate(topology.atoms())}
    for res in topology.residues():
        if res.name.upper() not in resnames:
            continue
        atoms = list(res.atoms())
        indices = [atom_indices[a] for a in atoms]
        oxy = None
        hydrogens = []
        for atom in atoms:
            name = atom.name.upper()
            if atom.element is not None and atom.element.symbol.upper() == "O":
                oxy = atom_indices[atom]
            elif name.startswith("O"):
                oxy = atom_indices[atom]
            elif atom.element is not None and atom.element.symbol.upper() == "H":
                hydrogens.append(atom_indices[atom])
            elif name.startswith("H"):
                hydrogens.append(atom_indices[atom])
        waters.append((indices, oxy, hydrogens))
    return waters


def _water_layout(water_atoms):
    offsets = [0]
    flat = []
    for atoms in water_atoms:
        for atom_idx in atoms:
            flat.append(int(atom_idx))
        offsets.append(len(flat))
    return offsets, flat


class _PmeEnergyEstimator:
    def __init__(self, openmm_system, cutoff, platform_name=None):
        mm, unit = _require_openmm()
        if not openmm_system.usesPeriodicBoundaryConditions():
            raise ValueError("PME requires periodic boundary conditions")
        self.system = mm.System()
        for i in range(openmm_system.getNumParticles()):
            self.system.addParticle(openmm_system.getParticleMass(i))
        nb = _find_nonbonded(openmm_system)
        self.force = mm.NonbondedForce()
        self.force.setNonbondedMethod(mm.NonbondedForce.PME)
        self.force.setCutoffDistance(cutoff * unit.nanometer)
        self.force.setUseDispersionCorrection(False)
        self.base_params = []
        for i in range(nb.getNumParticles()):
            charge, sigma, epsilon = nb.getParticleParameters(i)
            self.force.addParticle(charge, sigma, epsilon)
            self.base_params.append((charge, sigma, epsilon))
        self.base_exceptions = []
        for i in range(nb.getNumExceptions()):
            p1, p2, q, sig, eps = nb.getExceptionParameters(i)
            self.force.addException(p1, p2, q, sig, eps)
            self.base_exceptions.append((p1, p2, q, sig, eps))
        self.system.addForce(self.force)
        self.integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
        if platform_name:
            platform = mm.Platform.getPlatformByName(platform_name)
            self.context = mm.Context(self.system, self.integrator, platform)
        else:
            self.context = mm.Context(self.system, self.integrator)
        self.unit = unit
        self.active_mask = None

    def set_positions(self, positions_nm, box):
        box_arr = np.asarray(box, dtype=np.float64)
        if box_arr.shape == (9,):
            box_arr = box_arr.reshape(3, 3)
        if box_arr.shape == (3,):
            a = self.unit.Vec3(float(box_arr[0]), 0.0, 0.0)
            b = self.unit.Vec3(0.0, float(box_arr[1]), 0.0)
            c = self.unit.Vec3(0.0, 0.0, float(box_arr[2]))
        elif box_arr.shape == (3, 3):
            a = self.unit.Vec3(float(box_arr[0, 0]), float(box_arr[0, 1]), float(box_arr[0, 2]))
            b = self.unit.Vec3(float(box_arr[1, 0]), float(box_arr[1, 1]), float(box_arr[1, 2]))
            c = self.unit.Vec3(float(box_arr[2, 0]), float(box_arr[2, 1]), float(box_arr[2, 2]))
        else:
            raise ValueError("PME box must be shape (3,), (9,), or (3, 3)")
        self.context.setPeriodicBoxVectors(a, b, c)
        self.context.setPositions(positions_nm * self.unit.nanometer)

    def set_active_mask(self, active_mask):
        if self.active_mask is not None and np.array_equal(self.active_mask, active_mask):
            return
        self.active_mask = np.array(active_mask, dtype=bool)
        for i, (charge, sigma, epsilon) in enumerate(self.base_params):
            if self.active_mask[i]:
                self.force.setParticleParameters(i, charge, sigma, epsilon)
            else:
                self.force.setParticleParameters(i, charge * 0.0, sigma, epsilon * 0.0)
        for idx, (p1, p2, q, sig, eps) in enumerate(self.base_exceptions):
            if self.active_mask[p1] and self.active_mask[p2]:
                self.force.setExceptionParameters(idx, p1, p2, q, sig, eps)
            else:
                self.force.setExceptionParameters(idx, p1, p2, 0.0, sig, 0.0)
        self.force.updateParametersInContext(self.context)

    def energy(self):
        state = self.context.getState(getEnergy=True)
        return state.getPotentialEnergy().value_in_unit(self.unit.kilojoule_per_mole)


def _run_rust_gist_grid(
    traj,
    system,
    water_oxygens,
    water_hydrogens,
    solute_atoms,
    config: GistConfig,
    orient_bins: int,
):
    if system is None or not hasattr(system, "select_indices"):
        raise ValueError(
            "energy_method='none' requires a Rust-backed `system` with `select`/`select_indices`."
        )
    from warp_md import GistGridPlan  # type: ignore

    oxy_idx = []
    h1_idx = []
    h2_idx = []
    orient_valid = []
    for oxy, hydrogens in zip(water_oxygens, water_hydrogens):
        if oxy is None:
            continue
        oxy_i = int(oxy)
        oxy_idx.append(oxy_i)
        if len(hydrogens) >= 2:
            h1_idx.append(int(hydrogens[0]))
            h2_idx.append(int(hydrogens[1]))
            orient_valid.append(True)
        elif len(hydrogens) == 1:
            h1_idx.append(int(hydrogens[0]))
            h2_idx.append(int(hydrogens[0]))
            orient_valid.append(True)
        else:
            h1_idx.append(oxy_i)
            h2_idx.append(oxy_i)
            orient_valid.append(False)

    if not oxy_idx:
        raise ValueError("no water oxygen atoms available for Rust GIST grid")
    if solute_atoms:
        solute_sel = system.select_indices([int(i) for i in solute_atoms])
    else:
        solute_sel = system.select("all")

    frame_indices = None
    if config.frame_indices is not None:
        frame_indices = [int(i) for i in config.frame_indices if int(i) >= 0]

    plan = GistGridPlan(
        oxy_idx,
        h1_idx,
        h2_idx,
        orient_valid,
        solute_sel,
        spacing=float(config.grid_spacing),
        orientation_bins=int(orient_bins),
        length_scale=float(config.length_scale),
        padding=float(config.padding),
        origin=None,
        dims=None,
        frame_indices=frame_indices,
        max_frames=config.max_frames,
    )
    counts, orient_counts, origin, n_frames = plan.run(
        traj,
        system,
        chunk_frames=config.chunk_frames,
        device="auto",
    )

    counts = np.asarray(counts, dtype=np.int64)
    orient_counts = np.asarray(orient_counts, dtype=np.int64)
    origin = np.asarray(origin, dtype=np.float64)
    return counts, orient_counts, origin, int(n_frames)


def _run_rust_gist_direct(
    traj,
    system,
    water_atoms,
    water_oxygens,
    water_hydrogens,
    solute_atoms,
    config: GistConfig,
    orient_bins: int,
    charges,
    sigmas,
    epsilons,
    exceptions,
    periodic: bool,
    record_frame_energies: bool = False,
    record_pme_frame_totals: bool = False,
):
    if system is None or not hasattr(system, "select_indices"):
        raise ValueError(
            "energy_method='direct' and 'pme' require a Rust-backed `system` with `select_indices`."
        )
    from warp_md import GistDirectPlan  # type: ignore

    oxy_idx = []
    h1_idx = []
    h2_idx = []
    orient_valid = []
    kept_water_atoms = []
    for atoms, oxy, hydrogens in zip(water_atoms, water_oxygens, water_hydrogens):
        if oxy is None:
            continue
        oxy_i = int(oxy)
        oxy_idx.append(oxy_i)
        if len(hydrogens) >= 2:
            h1_idx.append(int(hydrogens[0]))
            h2_idx.append(int(hydrogens[1]))
            orient_valid.append(True)
        elif len(hydrogens) == 1:
            h1_idx.append(int(hydrogens[0]))
            h2_idx.append(int(hydrogens[0]))
            orient_valid.append(True)
        else:
            h1_idx.append(oxy_i)
            h2_idx.append(oxy_i)
            orient_valid.append(False)
        kept_water_atoms.append([int(i) for i in atoms])

    if not oxy_idx:
        raise ValueError("no water oxygen atoms available for Rust GIST direct")
    water_offsets, water_flat = _water_layout(kept_water_atoms)

    frame_indices = None
    if config.frame_indices is not None:
        frame_indices = [int(i) for i in config.frame_indices if int(i) >= 0]

    plan = GistDirectPlan(
        oxy_idx,
        h1_idx,
        h2_idx,
        orient_valid,
        water_offsets,
        water_flat,
        [int(i) for i in solute_atoms],
        np.asarray(charges, dtype=np.float64).tolist(),
        np.asarray(sigmas, dtype=np.float64).tolist(),
        np.asarray(epsilons, dtype=np.float64).tolist(),
        list(exceptions),
        spacing=float(config.grid_spacing),
        cutoff=float(config.cutoff),
        periodic=bool(periodic),
        orientation_bins=int(orient_bins),
        length_scale=float(config.length_scale),
        padding=float(config.padding),
        frame_indices=frame_indices,
        max_frames=config.max_frames,
        record_frame_energies=bool(record_frame_energies),
        record_pme_frame_totals=bool(record_pme_frame_totals),
    )
    out = plan.run(
        traj,
        system,
        chunk_frames=config.chunk_frames,
        device="auto",
    )
    if len(out) != 16:
        raise RuntimeError(f"unexpected GistDirectPlan output arity: {len(out)}; expected 16")
    (
        counts,
        orient_counts,
        energy_sw,
        energy_ww,
        origin,
        n_frames,
        direct_sw_total,
        direct_ww_total,
        frame_direct_sw,
        frame_direct_ww,
        frame_offsets,
        frame_cells,
        frame_sw,
        frame_ww,
        frame_pme_sw,
        frame_pme_ww,
    ) = out
    counts_arr = np.asarray(counts, dtype=np.int64)
    n_frames = int(n_frames)
    frame_data = {
        "direct_sw": np.asarray(frame_direct_sw, dtype=np.float64),
        "direct_ww": np.asarray(frame_direct_ww, dtype=np.float64),
        "offsets": np.asarray(frame_offsets, dtype=np.int64),
        "cells": np.asarray(frame_cells, dtype=np.int64),
        "sw": np.asarray(frame_sw, dtype=np.float64),
        "ww": np.asarray(frame_ww, dtype=np.float64),
        "native_sw": np.asarray(frame_pme_sw, dtype=np.float64),
        "native_ww": np.asarray(frame_pme_ww, dtype=np.float64),
    }
    offsets = frame_data["offsets"]
    cells = frame_data["cells"]
    if offsets.size == 0:
        if (
            frame_data["direct_sw"].size != 0
            or frame_data["direct_ww"].size != 0
            or cells.size != 0
            or frame_data["sw"].size != 0
            or frame_data["ww"].size != 0
        ):
            raise RuntimeError("invalid GIST frame payload: sparse vectors require frame_offsets")
    else:
        if offsets[0] != 0:
            raise RuntimeError("invalid GIST frame payload: frame_offsets must start at 0")
        if np.any(offsets[1:] < offsets[:-1]):
            raise RuntimeError("invalid GIST frame payload: frame_offsets must be non-decreasing")
        sparse_len = int(offsets[-1])
        if (
            frame_data["direct_sw"].size != n_frames
            or frame_data["direct_ww"].size != n_frames
            or cells.size != sparse_len
            or frame_data["sw"].size != sparse_len
            or frame_data["ww"].size != sparse_len
        ):
            raise RuntimeError("invalid GIST frame payload: sparse vector lengths are inconsistent")
        if np.any(cells < 0) or np.any(cells >= counts_arr.size):
            raise RuntimeError("invalid GIST frame payload: frame cell index out of bounds")
    if frame_data["native_sw"].size not in (0, n_frames) or frame_data["native_ww"].size not in (
        0,
        n_frames,
    ):
        raise RuntimeError("invalid GIST frame payload: native PME totals length mismatch")
    if frame_data["native_sw"].size != frame_data["native_ww"].size:
        raise RuntimeError("invalid GIST frame payload: native PME vectors length mismatch")
    return (
        counts_arr,
        np.asarray(orient_counts, dtype=np.int64),
        np.asarray(energy_sw, dtype=np.float64),
        np.asarray(energy_ww, dtype=np.float64),
        np.asarray(origin, dtype=np.float64),
        n_frames,
        float(direct_sw_total),
        float(direct_ww_total),
        frame_data,
    )


def _frame_filter(config: GistConfig):
    if config.frame_indices is not None:
        frame_set = {int(i) for i in config.frame_indices if int(i) >= 0}
        max_frame_idx = max(frame_set) if frame_set else -1
    else:
        frame_set = None
        max_frame_idx = None
    return frame_set, max_frame_idx


def _pme_totals(
    traj,
    openmm_system,
    solute_atoms,
    all_water_atoms,
    config: GistConfig,
    platform_name: Optional[str],
):
    if not hasattr(traj, "reset"):
        raise ValueError(
            "energy_method='pme' requires a rewindable trajectory (`traj.reset()`)."
        )
    traj.reset()
    pme_estimator = _PmeEnergyEstimator(openmm_system, config.cutoff, platform_name=platform_name)
    n_particles = openmm_system.getNumParticles()
    active_all = np.ones(n_particles, dtype=bool)
    active_solute = np.zeros(n_particles, dtype=bool)
    if solute_atoms:
        active_solute[solute_atoms] = True
    active_water = np.zeros(n_particles, dtype=bool)
    if all_water_atoms:
        active_water[list(all_water_atoms)] = True

    frame_set, max_frame_idx = _frame_filter(config)
    max_chunk = config.chunk_frames or 128
    per_frame_sw = []
    per_frame_ww = []
    kept = 0
    global_frame = 0

    chunk = read_chunk_fields(
        traj,
        max_chunk,
        include_box=True,
        include_box_matrix=True,
    )
    while chunk is not None:
        coords = np.asarray(chunk["coords"], dtype=np.float64) * config.length_scale
        box = chunk.get("box")
        box_matrix = chunk.get("box_matrix")
        if box_matrix is not None:
            box_matrix = np.asarray(box_matrix, dtype=np.float64) * config.length_scale
        elif box is not None:
            box = np.asarray(box, dtype=np.float64) * config.length_scale
        frames = coords.shape[0]
        for f in range(frames):
            if frame_set is not None and global_frame not in frame_set:
                global_frame += 1
                continue
            if box_matrix is not None:
                frame_box = box_matrix[f]
            elif box is not None:
                frame_box = box[f]
            else:
                frame_box = None
            if frame_box is None:
                raise ValueError("PME requires periodic box")
            pme_estimator.set_positions(coords[f], frame_box)
            pme_estimator.set_active_mask(active_all)
            e_all = pme_estimator.energy()
            pme_estimator.set_active_mask(active_solute)
            e_ss = pme_estimator.energy()
            pme_estimator.set_active_mask(active_water)
            e_ww = pme_estimator.energy()
            per_frame_sw.append(float(e_all - e_ss - e_ww))
            per_frame_ww.append(float(e_ww))
            kept += 1
            global_frame += 1
            if config.max_frames is not None and kept >= config.max_frames:
                return (
                    np.asarray(per_frame_sw, dtype=np.float64),
                    np.asarray(per_frame_ww, dtype=np.float64),
                    kept,
                )
        if max_frame_idx is not None and global_frame > max_frame_idx:
            break
        if config.max_frames is not None and kept >= config.max_frames:
            break
        chunk = read_chunk_fields(
            traj,
            max_chunk,
            include_box=True,
            include_box_matrix=True,
        )
    return np.asarray(per_frame_sw, dtype=np.float64), np.asarray(per_frame_ww, dtype=np.float64), kept


def _finalize_gist_result(
    counts: np.ndarray,
    orient_counts: np.ndarray,
    energy_sw: np.ndarray,
    energy_ww: np.ndarray,
    origin: np.ndarray,
    config: GistConfig,
    n_frames: int,
):
    voxel_volume = config.grid_spacing ** 3
    density = counts / (max(n_frames, 1) * voxel_volume)
    bulk_density = (
        float(config.bulk_density)
        if config.bulk_density is not None
        else counts.sum() / (max(n_frames, 1) * voxel_volume * density.size)
    )
    ratio = np.clip(density / max(bulk_density, 1e-12), 1e-12, None)
    ts_trans = -K_BOLTZ * config.temperature * density * np.log(ratio)

    dims = counts.shape
    ts_orient = np.zeros_like(density)
    with np.errstate(divide="ignore", invalid="ignore"):
        for idx in np.ndindex(*dims):
            total = counts[idx]
            if total <= 0:
                continue
            probs = orient_counts[idx] / total
            probs = probs[probs > 0]
            ts_orient[idx] = -K_BOLTZ * config.temperature * float(np.sum(probs * np.log(probs)))

    energy_sw = np.divide(
        energy_sw,
        counts,
        out=np.zeros_like(energy_sw, dtype=np.float64),
        where=counts > 0,
    )
    energy_ww = np.divide(
        energy_ww,
        counts,
        out=np.zeros_like(energy_ww, dtype=np.float64),
        where=counts > 0,
    )

    return GistResult(
        density=density,
        energy_sw=energy_sw,
        energy_ww=energy_ww,
        ts_trans=ts_trans,
        ts_orient=ts_orient,
        counts=counts,
        origin=origin,
        spacing=config.grid_spacing,
        n_frames=n_frames,
        voxel_volume=voxel_volume,
        bulk_density=bulk_density,
    )


def gist(
    traj,
    system,
    openmm_system,
    openmm_topology,
    config: Optional[GistConfig] = None,
    platform_name: Optional[str] = None,
):
    config = config or GistConfig()
    if config.grid_spacing <= 0:
        raise ValueError("grid_spacing must be > 0")
    energy_method = config.energy_method.lower()
    if energy_method not in ("direct", "pme", "none"):
        raise ValueError("energy_method must be 'direct', 'pme', or 'none'")
    pme_totals_source = config.pme_totals_source.lower()
    if pme_totals_source not in ("openmm", "native", "direct_approx"):
        raise ValueError("pme_totals_source must be 'openmm', 'native', or 'direct_approx'")

    waters = _water_groups(openmm_topology, config.water_resnames)
    if not waters:
        raise ValueError("no water residues found for GIST")
    water_atoms = [w[0] for w in waters]
    water_oxygens = [w[1] for w in waters]
    water_hydrogens = [w[2] for w in waters]

    all_water_atoms = set()
    for atoms in water_atoms:
        all_water_atoms.update(atoms)
    if config.solute_selection is not None:
        if system is None:
            raise ValueError("system is required when solute_selection is provided")
        solute_sel = system.select(config.solute_selection)
        solute_atoms = [int(i) for i in solute_sel.indices]
        if not solute_atoms:
            raise ValueError("solute_selection resolved to empty selection")
    else:
        solute_atoms = [i for i in range(openmm_system.getNumParticles()) if i not in all_water_atoms]

    orient_bins = max(int(config.orientation_bins), 1)
    if energy_method == "none":
        counts, orient_counts, origin, n_frames = _run_rust_gist_grid(
            traj,
            system,
            water_oxygens,
            water_hydrogens,
            solute_atoms,
            config,
            orient_bins,
        )
        energy_sw = np.zeros_like(counts, dtype=np.float64)
        energy_ww = np.zeros_like(counts, dtype=np.float64)
        return _finalize_gist_result(
            counts=counts,
            orient_counts=orient_counts,
            energy_sw=energy_sw,
            energy_ww=energy_ww,
            origin=origin,
            config=config,
            n_frames=n_frames,
        )
    charges, sigmas, epsilons, exceptions, periodic = _nonbonded_parameters(openmm_system)
    (
        counts,
        orient_counts,
        energy_sw_direct,
        energy_ww_direct,
        origin,
        n_frames,
        direct_sw_total,
        direct_ww_total,
        frame_data,
    ) = _run_rust_gist_direct(
        traj=traj,
        system=system,
        water_atoms=water_atoms,
        water_oxygens=water_oxygens,
        water_hydrogens=water_hydrogens,
        solute_atoms=solute_atoms,
        config=config,
        orient_bins=orient_bins,
        charges=charges,
        sigmas=sigmas,
        epsilons=epsilons,
        exceptions=exceptions,
        periodic=periodic,
        record_frame_energies=(energy_method == "pme"),
        record_pme_frame_totals=(energy_method == "pme" and pme_totals_source == "native"),
    )

    if energy_method == "direct":
        energy_sw = energy_sw_direct
        energy_ww = energy_ww_direct
    else:
        if pme_totals_source == "openmm":
            pme_sw_frame, pme_ww_frame, pme_frames = _pme_totals(
                traj=traj,
                openmm_system=openmm_system,
                solute_atoms=solute_atoms,
                all_water_atoms=all_water_atoms,
                config=config,
                platform_name=platform_name,
            )
            if pme_frames != n_frames:
                raise ValueError(
                    f"PME pass frame mismatch: Rust direct kept {n_frames} frames, PME pass kept {pme_frames}."
                )
        elif pme_totals_source == "native":
            if (
                frame_data is not None
                and frame_data["native_sw"].size == n_frames
                and frame_data["native_ww"].size == n_frames
            ):
                pme_sw_frame = np.asarray(frame_data["native_sw"], dtype=np.float64)
                pme_ww_frame = np.asarray(frame_data["native_ww"], dtype=np.float64)
                pme_frames = n_frames
            else:
                raise RuntimeError(
                    "native PME totals unavailable from Rust bindings; rebuild warp_md with updated bindings"
                )
        else:
            # Rust-first approximation mode: reuse direct frame totals for unit scaling.
            if (
                frame_data is not None
                and frame_data["direct_sw"].size == n_frames
                and frame_data["direct_ww"].size == n_frames
            ):
                pme_sw_frame = np.asarray(frame_data["direct_sw"], dtype=np.float64)
                pme_ww_frame = np.asarray(frame_data["direct_ww"], dtype=np.float64)
                pme_frames = n_frames
            else:
                pme_sw_frame = np.asarray([direct_sw_total], dtype=np.float64)
                pme_ww_frame = np.asarray([direct_ww_total], dtype=np.float64)
                pme_frames = 1
        if frame_data is not None and frame_data["offsets"].size == (n_frames + 1):
            direct_sw_frame = frame_data["direct_sw"]
            direct_ww_frame = frame_data["direct_ww"]
            offsets = frame_data["offsets"].astype(np.uint64, copy=False)
            cells = frame_data["cells"].astype(np.uint32, copy=False)
            vals_sw = frame_data["sw"].astype(np.float64, copy=False)
            vals_ww = frame_data["ww"].astype(np.float64, copy=False)
        else:
            direct_sw_frame = np.asarray([], dtype=np.float64)
            direct_ww_frame = np.asarray([], dtype=np.float64)
            offsets = np.asarray([], dtype=np.uint64)
            cells = np.asarray([], dtype=np.uint32)
            vals_sw = np.asarray([], dtype=np.float64)
            vals_ww = np.asarray([], dtype=np.float64)
        try:
            from warp_md import gist_apply_pme_scaling  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "gist_apply_pme_scaling binding unavailable; rebuild warp_md with Rust bindings"
            ) from exc
        energy_sw, energy_ww = gist_apply_pme_scaling(
            np.asarray(energy_sw_direct, dtype=np.float64),
            np.asarray(energy_ww_direct, dtype=np.float64),
            float(direct_sw_total),
            float(direct_ww_total),
            np.asarray(direct_sw_frame, dtype=np.float64),
            np.asarray(direct_ww_frame, dtype=np.float64),
            np.asarray(pme_sw_frame, dtype=np.float64),
            np.asarray(pme_ww_frame, dtype=np.float64),
            offsets,
            cells,
            vals_sw,
            vals_ww,
        )
        energy_sw = np.asarray(energy_sw, dtype=np.float64)
        energy_ww = np.asarray(energy_ww, dtype=np.float64)

    return _finalize_gist_result(
        counts=counts,
        orient_counts=orient_counts,
        energy_sw=energy_sw,
        energy_ww=energy_ww,
        origin=origin,
        config=config,
        n_frames=n_frames,
    )
