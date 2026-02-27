# Usage:
# from warp_md.analysis.energy_analysis import energy_analysis, lie
# energies = energy_analysis(traj, system, openmm_system, length_scale=0.1)
# lie_out = lie(traj, system, openmm_system, "resname LIG", length_scale=0.1)

from __future__ import annotations

from typing import Optional

import numpy as np

from ._chunk_io import read_chunk_fields


def _require_openmm():
    try:
        from openmm import unit, openmm as mm
    except Exception as exc:  # pragma: no cover
        raise ImportError("openmm is required for energy analysis") from exc
    return mm, unit


def _find_nonbonded(openmm_system):
    mm, _ = _require_openmm()
    for force in openmm_system.getForces():
        if isinstance(force, mm.NonbondedForce):
            return force
    raise ValueError("openmm_system has no NonbondedForce")


class _LieEstimator:
    def __init__(self, openmm_system, cutoff: Optional[float], platform_name: Optional[str]):
        mm, unit = _require_openmm()
        nb = _find_nonbonded(openmm_system)
        self.system = mm.System()
        for i in range(openmm_system.getNumParticles()):
            self.system.addParticle(openmm_system.getParticleMass(i))

        lj_expr = (
            "4*sqrt(epsilon1*epsilon2)"
            "*((0.5*(sigma1+sigma2)/r)^12 - (0.5*(sigma1+sigma2)/r)^6)"
        )
        coul_expr = "k_e*charge1*charge2/r"
        self.force_lj = mm.CustomNonbondedForce(lj_expr)
        self.force_coul = mm.CustomNonbondedForce(coul_expr)
        self.force_coul.addGlobalParameter("k_e", 138.935456)
        self.charge = []
        self.sigma = []
        self.epsilon = []
        for force in (self.force_lj, self.force_coul):
            force.addPerParticleParameter("charge")
            force.addPerParticleParameter("sigma")
            force.addPerParticleParameter("epsilon")
            if openmm_system.usesPeriodicBoundaryConditions():
                force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffPeriodic)
            else:
                force.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic)
            if cutoff is None:
                cutoff_nm = nb.getCutoffDistance().value_in_unit(unit.nanometer)
            else:
                cutoff_nm = float(cutoff)
            force.setCutoffDistance(cutoff_nm * unit.nanometer)
            force.setUseLongRangeCorrection(False)

        for i in range(nb.getNumParticles()):
            charge, sigma, epsilon = nb.getParticleParameters(i)
            self.charge.append(float(charge.value_in_unit(unit.elementary_charge)))
            self.sigma.append(float(sigma.value_in_unit(unit.nanometer)))
            self.epsilon.append(float(epsilon.value_in_unit(unit.kilojoule_per_mole)))
            params = [charge, sigma, epsilon]
            self.force_lj.addParticle(params)
            self.force_coul.addParticle(params)
        self.charge = np.asarray(self.charge, dtype=np.float64)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
        self.epsilon = np.asarray(self.epsilon, dtype=np.float64)

        self.exceptions = []
        self.exception_map = {}
        for i in range(nb.getNumExceptions()):
            p1, p2, _q, _sig, _eps = nb.getExceptionParameters(i)
            self.force_lj.addExclusion(p1, p2)
            self.force_coul.addExclusion(p1, p2)
            q, sig, eps = nb.getExceptionParameters(i)[2:]
            key = (min(int(p1), int(p2)), max(int(p1), int(p2)))
            self.exceptions.append(
                (
                    int(p1),
                    int(p2),
                    float(q.value_in_unit(unit.elementary_charge**2)),
                    float(sig.value_in_unit(unit.nanometer)),
                    float(eps.value_in_unit(unit.kilojoule_per_mole)),
                )
            )
            self.exception_map[key] = (
                float(q.value_in_unit(unit.elementary_charge**2)),
                float(sig.value_in_unit(unit.nanometer)),
                float(eps.value_in_unit(unit.kilojoule_per_mole)),
            )

        self.force_lj.setForceGroup(1)
        self.force_coul.setForceGroup(2)
        self.system.addForce(self.force_lj)
        self.system.addForce(self.force_coul)

        self.integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
        if platform_name:
            platform = mm.Platform.getPlatformByName(platform_name)
            self.context = mm.Context(self.system, self.integrator, platform)
        else:
            self.context = mm.Context(self.system, self.integrator)
        self.unit = unit
        self.coulomb_const = 138.935456
        self._positions = None
        self._box = None

    def set_positions(self, positions_nm, box=None):
        if box is not None:
            lx, ly, lz = box
            a = self.unit.Vec3(lx, 0.0, 0.0)
            b = self.unit.Vec3(0.0, ly, 0.0)
            c = self.unit.Vec3(0.0, 0.0, lz)
            self.context.setPeriodicBoxVectors(a, b, c)
        self.context.setPositions(positions_nm * self.unit.nanometer)
        self._positions = np.asarray(positions_nm, dtype=np.float64)
        self._box = np.asarray(box, dtype=np.float64) if box is not None else None

    def energy_between(self, group_a, group_b):
        if not group_a or not group_b:
            return 0.0, 0.0
        group_a = set(int(i) for i in group_a)
        group_b = set(int(i) for i in group_b)
        if hasattr(self.force_lj, "clearInteractionGroups") and hasattr(self.force_coul, "clearInteractionGroups"):
            for force in (self.force_lj, self.force_coul):
                force.clearInteractionGroups()
                force.addInteractionGroup(group_a, group_b)
                force.updateParametersInContext(self.context)
            lj = self.context.getState(getEnergy=True, groups=1 << 1).getPotentialEnergy()
            coul = self.context.getState(getEnergy=True, groups=1 << 2).getPotentialEnergy()
            lj_val = lj.value_in_unit(self.unit.kilojoule_per_mole)
            coul_val = coul.value_in_unit(self.unit.kilojoule_per_mole)
            if self._positions is not None and self.exceptions:
                exc_lj, exc_coul = self._exception_energy(group_a, group_b)
                lj_val += exc_lj
                coul_val += exc_coul
            return (lj_val, coul_val)
        return self._pairwise_energy(group_a, group_b)

    def _pairwise_energy(self, group_a, group_b):
        if self._positions is None:
            return 0.0, 0.0
        pos = self._positions
        box = self._box
        e_lj = 0.0
        e_coul = 0.0
        for i in group_a:
            for j in group_b:
                if i == j:
                    continue
                key = (min(i, j), max(i, j))
                if key in self.exception_map:
                    q_prod, sig, eps = self.exception_map[key]
                else:
                    q_prod = float(self.charge[i] * self.charge[j])
                    sig = float(0.5 * (self.sigma[i] + self.sigma[j]))
                    eps = float(np.sqrt(self.epsilon[i] * self.epsilon[j]))

                rij = pos[i] - pos[j]
                if box is not None:
                    rij = rij - np.round(rij / box) * box
                r = float(np.linalg.norm(rij))
                if r == 0.0:
                    continue
                if eps != 0.0 and sig != 0.0:
                    sr6 = (sig / r) ** 6
                    e_lj += 4.0 * eps * (sr6 * sr6 - sr6)
                if q_prod != 0.0:
                    e_coul += self.coulomb_const * q_prod / r
        return e_lj, e_coul

    def _exception_energy(self, group_a, group_b):
        pos = self._positions
        box = self._box
        e_lj = 0.0
        e_coul = 0.0
        for p1, p2, q, sig, eps in self.exceptions:
            in_a = p1 in group_a and p2 in group_b
            in_b = p2 in group_a and p1 in group_b
            if not (in_a or in_b):
                continue
            rij = pos[p1] - pos[p2]
            if box is not None:
                rij = rij - np.round(rij / box) * box
            r = np.linalg.norm(rij)
            if r == 0.0:
                continue
            if eps != 0.0 and sig != 0.0:
                sr6 = (sig / r) ** 6
                e_lj += 4.0 * eps * (sr6 * sr6 - sr6)
            if q != 0.0:
                e_coul += self.coulomb_const * q / r
        return e_lj, e_coul


def energy_analysis(
    traj,
    system,
    openmm_system,
    length_scale: float = 0.1,
    platform_name: Optional[str] = None,
    chunk_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
    energy_groups: bool = False,
    summarize: bool = False,
):
    """Compute potential energy per frame with OpenMM."""
    mm, unit = _require_openmm()
    integrator = mm.VerletIntegrator(1.0 * unit.femtosecond)
    if platform_name:
        platform = mm.Platform.getPlatformByName(platform_name)
        context = mm.Context(openmm_system, integrator, platform)
    else:
        context = mm.Context(openmm_system, integrator)
    if energy_groups:
        for idx, force in enumerate(openmm_system.getForces()):
            force.setForceGroup(idx)

    _ = system
    energies = []
    group_energy = [] if energy_groups else None
    group_names = None
    if energy_groups:
        group_names = [f.__class__.__name__ for f in openmm_system.getForces()]
    n_frames = 0
    max_chunk = chunk_frames or 128
    chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    while chunk is not None:
        coords = np.asarray(chunk["coords"], dtype=np.float64) * length_scale
        box = chunk.get("box")
        if box is not None:
            box = np.asarray(box, dtype=np.float64) * length_scale
        frames = coords.shape[0]
        for f in range(frames):
            if box is not None:
                lx, ly, lz = box[f]
                a = unit.Vec3(lx, 0.0, 0.0)
                b = unit.Vec3(0.0, ly, 0.0)
                c = unit.Vec3(0.0, 0.0, lz)
                context.setPeriodicBoxVectors(a, b, c)
            context.setPositions(coords[f] * unit.nanometer)
            state = context.getState(getEnergy=True)
            energies.append(state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
            if energy_groups:
                frame_group = []
                for idx, _force in enumerate(openmm_system.getForces()):
                    st = context.getState(getEnergy=True, groups=1 << idx)
                    frame_group.append(st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole))
                group_energy.append(frame_group)
            n_frames += 1
            if max_frames is not None and n_frames >= max_frames:
                out = {"potential": np.asarray(energies, dtype=np.float64)}
                if energy_groups:
                    out["groups"] = np.asarray(group_energy, dtype=np.float64)
                    out["group_names"] = list(group_names)
                if summarize:
                    out.update(_summarize_energy(out))
                return out
        chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    out = {"potential": np.asarray(energies, dtype=np.float64)}
    if energy_groups:
        out["groups"] = np.asarray(group_energy, dtype=np.float64)
        out["group_names"] = list(group_names)
    if summarize:
        out.update(_summarize_energy(out))
    return out


def esander(
    traj,
    system,
    openmm_system,
    length_scale: float = 0.1,
    platform_name: Optional[str] = None,
    chunk_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
    energy_groups: bool = False,
    summarize: bool = False,
):
    """Alias to energy_analysis (OpenMM-based)."""
    return energy_analysis(
        traj,
        system,
        openmm_system,
        length_scale=length_scale,
        platform_name=platform_name,
        chunk_frames=chunk_frames,
        max_frames=max_frames,
        energy_groups=energy_groups,
        summarize=summarize,
    )


def ene_decomp(
    traj,
    system,
    openmm_system,
    length_scale: float = 0.1,
    platform_name: Optional[str] = None,
    chunk_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
    energy_groups: bool = False,
    summarize: bool = False,
):
    """Alias to energy_analysis (OpenMM-based)."""
    return energy_analysis(
        traj,
        system,
        openmm_system,
        length_scale=length_scale,
        platform_name=platform_name,
        chunk_frames=chunk_frames,
        max_frames=max_frames,
        energy_groups=energy_groups,
        summarize=summarize,
    )


def _summarize_energy(out):
    pot = out.get("potential")
    summary = {}
    if pot is not None and len(pot):
        summary["potential_mean"] = float(np.mean(pot))
        summary["potential_std"] = float(np.std(pot, ddof=1)) if len(pot) > 1 else 0.0
        summary["potential_stderr"] = (
            float(np.std(pot, ddof=1) / np.sqrt(len(pot))) if len(pot) > 1 else 0.0
        )
    groups = out.get("groups")
    if groups is not None and len(groups):
        summary["groups_mean"] = np.mean(groups, axis=0)
        summary["groups_std"] = np.std(groups, axis=0, ddof=1) if groups.shape[0] > 1 else np.zeros(groups.shape[1])
        summary["groups_stderr"] = (
            summary["groups_std"] / np.sqrt(groups.shape[0]) if groups.shape[0] > 1 else np.zeros(groups.shape[1])
        )
    return summary


def lie(
    traj,
    system,
    openmm_system,
    selection,
    length_scale: float = 0.1,
    cutoff: Optional[float] = None,
    platform_name: Optional[str] = None,
    chunk_frames: Optional[int] = None,
    max_frames: Optional[int] = None,
):
    """Linear interaction energy (LJ + Coulomb) between selection and rest."""
    if hasattr(selection, "indices"):
        indices = list(selection.indices)
    elif isinstance(selection, (list, tuple, np.ndarray)):
        indices = [int(i) for i in np.asarray(selection).reshape(-1).tolist()]
    else:
        if system is None:
            raise ValueError("system is required when selection is not an index list")
        indices = list(system.select(selection).indices)
    n_particles = openmm_system.getNumParticles()
    group_a = [int(i) for i in indices]
    group_b = [i for i in range(n_particles) if i not in set(group_a)]
    estimator = _LieEstimator(openmm_system, cutoff, platform_name)

    vdw = []
    ele = []
    total = []
    n_frames = 0
    max_chunk = chunk_frames or 128
    chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    while chunk is not None:
        coords = np.asarray(chunk["coords"], dtype=np.float64) * length_scale
        box = chunk.get("box")
        if box is not None:
            box = np.asarray(box, dtype=np.float64) * length_scale
        frames = coords.shape[0]
        for f in range(frames):
            frame_box = box[f] if box is not None else None
            estimator.set_positions(coords[f], box=frame_box)
            e_vdw, e_ele = estimator.energy_between(group_a, group_b)
            vdw.append(e_vdw)
            ele.append(e_ele)
            total.append(e_vdw + e_ele)
            n_frames += 1
            if max_frames is not None and n_frames >= max_frames:
                return {
                    "vdw": np.asarray(vdw, dtype=np.float64),
                    "ele": np.asarray(ele, dtype=np.float64),
                    "total": np.asarray(total, dtype=np.float64),
                }
        chunk = read_chunk_fields(traj, max_chunk, include_box=True)
    return {
        "vdw": np.asarray(vdw, dtype=np.float64),
        "ele": np.asarray(ele, dtype=np.float64),
        "total": np.asarray(total, dtype=np.float64),
    }


__all__ = ["energy_analysis", "esander", "ene_decomp", "lie"]
