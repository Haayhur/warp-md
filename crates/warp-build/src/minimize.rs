use std::collections::{BTreeMap, BTreeSet};

use traj_core::elements::{mass_for_element, normalize_element};
use warp_structure::io::AmberTopology;
use warp_structure::{center_of_geometry, PackOutput, Vec3};

const COULOMB_KCAL_PER_MOL_ANG_E2: f32 = 332.0637;
const ENERGY_EPS: f32 = 1.0e-8;
const DIST_EPS: f32 = 1.0e-6;
const FORCE_TOLERANCE: f32 = 5.0e-2;
const LINE_SEARCH_MAX_BACKTRACKS: usize = 8;

#[derive(Clone, Debug)]
pub struct MinimizationTelemetry {
    pub steps_executed: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub initial_energy: f32,
    pub final_energy: f32,
    pub initial_max_force: f32,
    pub final_max_force: f32,
    pub moved_atom_count: usize,
    pub termination_reason: String,
}

#[derive(Clone, Copy, Debug)]
struct PairNonbondedParam {
    acoef: f32,
    bcoef: f32,
    coulomb_scale: f32,
}

#[derive(Clone, Debug)]
struct NonbondedPair {
    left: usize,
    right: usize,
    param: PairNonbondedParam,
}

#[derive(Clone, Debug)]
struct EnergyEvaluation {
    total_energy: f32,
    max_force: f32,
    gradient: Vec<Vec3>,
}

fn ordered_pair(a: usize, b: usize) -> (usize, usize) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn wrap_dihedral(mut value: f32) -> f32 {
    let period = 2.0 * std::f32::consts::PI;
    while value <= -std::f32::consts::PI {
        value += period;
    }
    while value > std::f32::consts::PI {
        value -= period;
    }
    value
}

fn vec_norm_or_zero(value: Vec3) -> f32 {
    let norm = value.norm();
    if norm.is_finite() {
        norm
    } else {
        0.0
    }
}

fn accumulate(target: &mut Vec3, delta: Vec3) {
    *target = target.add(delta);
}

fn subtract(target: &mut Vec3, delta: Vec3) {
    *target = target.sub(delta);
}

fn bond_term(topology: &AmberTopology, bond_idx: usize) -> Option<(f32, f32)> {
    let type_idx = topology
        .bond_type_indices
        .get(bond_idx)
        .copied()?
        .saturating_sub(1);
    Some((
        *topology.bond_force_constants.get(type_idx)?,
        *topology.bond_equil_values.get(type_idx)?,
    ))
}

fn angle_term(topology: &AmberTopology, angle_idx: usize) -> Option<(f32, f32)> {
    let type_idx = topology
        .angle_type_indices
        .get(angle_idx)
        .copied()?
        .saturating_sub(1);
    Some((
        *topology.angle_force_constants.get(type_idx)?,
        *topology.angle_equil_values.get(type_idx)?,
    ))
}

fn dihedral_term(
    topology: &AmberTopology,
    type_idx_1based: usize,
) -> Option<(f32, f32, f32, f32, f32)> {
    let type_idx = type_idx_1based.saturating_sub(1);
    Some((
        *topology.dihedral_force_constants.get(type_idx)?,
        *topology.dihedral_periodicities.get(type_idx)?,
        *topology.dihedral_phases.get(type_idx)?,
        *topology.scee_scale_factors.get(type_idx).unwrap_or(&1.2),
        *topology.scnb_scale_factors.get(type_idx).unwrap_or(&2.0),
    ))
}

fn dihedral_angle(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) -> Option<f32> {
    let b1 = p1.sub(p0);
    let b2 = p2.sub(p1);
    let b3 = p3.sub(p2);
    let n1 = b1.cross(b2);
    let n2 = b2.cross(b3);
    let n1_norm = n1.norm();
    let n2_norm = n2.norm();
    let b2_norm = b2.norm();
    if n1_norm <= DIST_EPS || n2_norm <= DIST_EPS || b2_norm <= DIST_EPS {
        return None;
    }
    let n1_unit = n1.scale(1.0 / n1_norm);
    let n2_unit = n2.scale(1.0 / n2_norm);
    let m1 = n1_unit.cross(b2.scale(1.0 / b2_norm));
    Some(m1.dot(n2_unit).atan2(n1_unit.dot(n2_unit)))
}

fn torsion_gradient_forces(
    p0: Vec3,
    p1: Vec3,
    p2: Vec3,
    p3: Vec3,
    d_e_d_phi: f32,
) -> Option<[Vec3; 4]> {
    let b1 = p1.sub(p0);
    let b2 = p2.sub(p1);
    let b3 = p3.sub(p2);
    let a = b1.cross(b2);
    let b = b3.cross(b2);
    let a_norm_sq = a.dot(a);
    let b_norm_sq = b.dot(b);
    let b2_norm = b2.norm();
    let b2_norm_sq = b2.dot(b2);
    if a_norm_sq <= DIST_EPS
        || b_norm_sq <= DIST_EPS
        || b2_norm <= DIST_EPS
        || b2_norm_sq <= DIST_EPS
    {
        return None;
    }
    let f0 = a.scale(-d_e_d_phi * b2_norm / a_norm_sq);
    let f3 = b.scale(d_e_d_phi * b2_norm / b_norm_sq);
    let s1 = b1.dot(b2) / b2_norm_sq;
    let s2 = b3.dot(b2) / b2_norm_sq;
    let f1 = f0.scale(s1 - 1.0).sub(f3.scale(s2));
    let f2 = f0.scale(-s1).sub(f3.scale(1.0 - s2));
    Some([f0, f1, f2, f3])
}

fn atom_masses(output: &PackOutput, topology: &AmberTopology) -> Vec<f32> {
    output
        .atoms
        .iter()
        .enumerate()
        .map(|(idx, atom)| {
            topology
                .masses
                .get(idx)
                .copied()
                .filter(|value| *value > 0.0)
                .or_else(|| {
                    normalize_element(&atom.element)
                        .map(|element| mass_for_element(&element))
                        .filter(|value| *value > 0.0)
                })
                .unwrap_or(12.0)
        })
        .collect()
}

fn build_one_four_scales(topology: &AmberTopology) -> BTreeMap<(usize, usize), (f32, f32)> {
    let mut scales = BTreeMap::new();
    for (idx, dihedral) in topology.dihedrals.iter().enumerate() {
        let Some((_, _, _, scee, scnb)) = dihedral_term(
            topology,
            topology
                .dihedral_type_indices
                .get(idx)
                .copied()
                .unwrap_or(1),
        ) else {
            continue;
        };
        scales
            .entry(ordered_pair(dihedral[0], dihedral[3]))
            .or_insert((scee, scnb));
    }
    scales
}

fn exclusion_pairs(topology: &AmberTopology) -> BTreeSet<(usize, usize)> {
    let mut exclusions = BTreeSet::new();
    for (left, items) in topology.excluded_atoms.iter().enumerate() {
        for &right_1based in items {
            if right_1based == 0 {
                continue;
            }
            let right = right_1based.saturating_sub(1);
            if right < topology.atom_names.len() && right != left {
                exclusions.insert(ordered_pair(left, right));
            }
        }
    }
    exclusions
}

fn nonbonded_param_for_pair(
    topology: &AmberTopology,
    left: usize,
    right: usize,
    one_four_scales: &BTreeMap<(usize, usize), (f32, f32)>,
) -> Option<PairNonbondedParam> {
    let n_types = topology
        .atom_type_indices
        .iter()
        .copied()
        .max()
        .unwrap_or(1);
    let left_type = topology
        .atom_type_indices
        .get(left)
        .copied()
        .unwrap_or(1)
        .saturating_sub(1);
    let right_type = topology
        .atom_type_indices
        .get(right)
        .copied()
        .unwrap_or(1)
        .saturating_sub(1);
    let parm_slot = left_type.checked_mul(n_types)?.checked_add(right_type)?;
    let parm_idx = topology
        .nonbonded_parm_index
        .get(parm_slot)
        .copied()
        .unwrap_or(parm_slot + 1)
        .saturating_sub(1);
    let regular_a = *topology.lennard_jones_acoef.get(parm_idx).unwrap_or(&0.0);
    let regular_b = *topology.lennard_jones_bcoef.get(parm_idx).unwrap_or(&0.0);
    let is_one_four = ordered_pair(left, right);
    if let Some(&(scee, scnb)) = one_four_scales.get(&is_one_four) {
        let a14 = *topology
            .lennard_jones_14_acoef
            .get(parm_idx)
            .unwrap_or(&regular_a);
        let b14 = *topology
            .lennard_jones_14_bcoef
            .get(parm_idx)
            .unwrap_or(&regular_b);
        let scnb = if scnb.abs() <= ENERGY_EPS { 1.0 } else { scnb };
        let use_direct_14 = (a14 - regular_a).abs() > 1.0e-6 || (b14 - regular_b).abs() > 1.0e-6;
        let (acoef, bcoef) = if use_direct_14 {
            (a14, b14)
        } else {
            (a14 / scnb, b14 / scnb)
        };
        return Some(PairNonbondedParam {
            acoef,
            bcoef,
            coulomb_scale: if scee.abs() <= ENERGY_EPS {
                1.0
            } else {
                1.0 / scee
            },
        });
    }
    Some(PairNonbondedParam {
        acoef: regular_a,
        bcoef: regular_b,
        coulomb_scale: 1.0,
    })
}

fn build_nonbonded_pairs(topology: &AmberTopology) -> Vec<NonbondedPair> {
    let one_four_scales = build_one_four_scales(topology);
    let exclusions = exclusion_pairs(topology);
    let mut pairs = Vec::new();
    for left in 0..topology.atom_names.len() {
        for right in (left + 1)..topology.atom_names.len() {
            let pair = ordered_pair(left, right);
            if exclusions.contains(&pair) && !one_four_scales.contains_key(&pair) {
                continue;
            }
            if let Some(param) = nonbonded_param_for_pair(topology, left, right, &one_four_scales) {
                pairs.push(NonbondedPair { left, right, param });
            }
        }
    }
    pairs
}

fn evaluate_energy(
    positions: &[Vec3],
    topology: &AmberTopology,
    nonbonded_pairs: &[NonbondedPair],
) -> EnergyEvaluation {
    let mut gradient = vec![Vec3::new(0.0, 0.0, 0.0); positions.len()];
    let mut total_energy = 0.0f32;

    for (bond_idx, &(left, right)) in topology.bonds.iter().enumerate() {
        let Some((force_constant, rest_length)) = bond_term(topology, bond_idx) else {
            continue;
        };
        let delta = positions[left].sub(positions[right]);
        let distance = delta.norm().max(DIST_EPS);
        let stretch = distance - rest_length;
        total_energy += 0.5 * force_constant * stretch * stretch;
        let coeff = force_constant * stretch / distance;
        let grad = delta.scale(coeff);
        accumulate(&mut gradient[left], grad);
        subtract(&mut gradient[right], grad);
    }

    for (angle_idx, angle) in topology.angles.iter().enumerate() {
        let Some((force_constant, theta0)) = angle_term(topology, angle_idx) else {
            continue;
        };
        let left_vec = positions[angle[0]].sub(positions[angle[1]]);
        let right_vec = positions[angle[2]].sub(positions[angle[1]]);
        let left_norm = left_vec.norm();
        let right_norm = right_vec.norm();
        if left_norm <= DIST_EPS || right_norm <= DIST_EPS {
            continue;
        }
        let left_unit = left_vec.scale(1.0 / left_norm);
        let right_unit = right_vec.scale(1.0 / right_norm);
        let cos_theta = left_unit.dot(right_unit).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1.0e-4);
        let delta_theta = theta - theta0;
        total_energy += 0.5 * force_constant * delta_theta * delta_theta;
        let d_e_d_theta = force_constant * delta_theta;
        let grad_left = left_unit
            .scale(cos_theta)
            .sub(right_unit)
            .scale(d_e_d_theta / (left_norm * sin_theta));
        let grad_right = right_unit
            .scale(cos_theta)
            .sub(left_unit)
            .scale(d_e_d_theta / (right_norm * sin_theta));
        accumulate(&mut gradient[angle[0]], grad_left);
        accumulate(&mut gradient[angle[2]], grad_right);
        subtract(&mut gradient[angle[1]], grad_left.add(grad_right));
    }

    for (dihedral_idx, dihedral) in topology.dihedrals.iter().enumerate() {
        let Some((force_constant, periodicity, phase_rad, _, _)) = dihedral_term(
            topology,
            topology
                .dihedral_type_indices
                .get(dihedral_idx)
                .copied()
                .unwrap_or(1),
        ) else {
            continue;
        };
        let Some(phi) = dihedral_angle(
            positions[dihedral[0]],
            positions[dihedral[1]],
            positions[dihedral[2]],
            positions[dihedral[3]],
        ) else {
            continue;
        };
        let arg = periodicity * phi - phase_rad;
        total_energy += 0.5 * force_constant * (1.0 + arg.cos());
        let d_e_d_phi = -0.5 * force_constant * periodicity * arg.sin();
        if let Some(forces) = torsion_gradient_forces(
            positions[dihedral[0]],
            positions[dihedral[1]],
            positions[dihedral[2]],
            positions[dihedral[3]],
            d_e_d_phi,
        ) {
            subtract(&mut gradient[dihedral[0]], forces[0]);
            subtract(&mut gradient[dihedral[1]], forces[1]);
            subtract(&mut gradient[dihedral[2]], forces[2]);
            subtract(&mut gradient[dihedral[3]], forces[3]);
        }
    }

    for (improper_idx, improper) in topology.impropers.iter().enumerate() {
        let Some((force_constant, _periodicity, phase_rad, _, _)) = dihedral_term(
            topology,
            topology
                .improper_type_indices
                .get(improper_idx)
                .copied()
                .unwrap_or(1),
        ) else {
            continue;
        };
        let Some(phi) = dihedral_angle(
            positions[improper[0]],
            positions[improper[1]],
            positions[improper[2]],
            positions[improper[3]],
        ) else {
            continue;
        };
        let delta_phi = wrap_dihedral(phi - phase_rad);
        total_energy += 0.5 * force_constant * delta_phi * delta_phi;
        let d_e_d_phi = force_constant * delta_phi;
        if let Some(forces) = torsion_gradient_forces(
            positions[improper[0]],
            positions[improper[1]],
            positions[improper[2]],
            positions[improper[3]],
            d_e_d_phi,
        ) {
            subtract(&mut gradient[improper[0]], forces[0]);
            subtract(&mut gradient[improper[1]], forces[1]);
            subtract(&mut gradient[improper[2]], forces[2]);
            subtract(&mut gradient[improper[3]], forces[3]);
        }
    }

    let has_trustworthy_charges = topology.charges.iter().any(|charge| charge.abs() > 1.0e-4);
    for pair in nonbonded_pairs {
        let delta = positions[pair.left].sub(positions[pair.right]);
        let distance = delta.norm().max(0.6);
        let distance_sq = distance * distance;
        let inv_r2 = 1.0 / distance_sq;
        let inv_r6 = inv_r2 * inv_r2 * inv_r2;
        let inv_r12 = inv_r6 * inv_r6;
        if pair.param.acoef.abs() > ENERGY_EPS || pair.param.bcoef.abs() > ENERGY_EPS {
            total_energy += pair.param.acoef * inv_r12 - pair.param.bcoef * inv_r6;
            let coeff = -12.0 * pair.param.acoef * inv_r12 * inv_r2
                + 6.0 * pair.param.bcoef * inv_r6 * inv_r2;
            let grad = delta.scale(coeff);
            accumulate(&mut gradient[pair.left], grad);
            subtract(&mut gradient[pair.right], grad);
        }
        if has_trustworthy_charges {
            let left_charge = *topology.charges.get(pair.left).unwrap_or(&0.0);
            let right_charge = *topology.charges.get(pair.right).unwrap_or(&0.0);
            if left_charge.abs() > ENERGY_EPS || right_charge.abs() > ENERGY_EPS {
                let scaled_k = COULOMB_KCAL_PER_MOL_ANG_E2 * pair.param.coulomb_scale;
                total_energy += scaled_k * left_charge * right_charge / distance;
                let coeff = -scaled_k * left_charge * right_charge / (distance_sq * distance);
                let grad = delta.scale(coeff);
                accumulate(&mut gradient[pair.left], grad);
                subtract(&mut gradient[pair.right], grad);
            }
        }
    }

    let max_force = gradient
        .iter()
        .map(|value| vec_norm_or_zero(*value))
        .fold(0.0f32, f32::max);
    EnergyEvaluation {
        total_energy,
        max_force,
        gradient,
    }
}

pub fn minimize_synthetic_topology(
    output: &mut PackOutput,
    topology: &AmberTopology,
    steps_requested: usize,
    step_scale: f32,
) -> MinimizationTelemetry {
    let masses = atom_masses(output, topology);
    let nonbonded_pairs = build_nonbonded_pairs(topology);
    let initial_positions = output
        .atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let initial_center = center_of_geometry(&initial_positions);
    let mut current_positions = initial_positions.clone();
    let mut current = evaluate_energy(&current_positions, topology, &nonbonded_pairs);
    let initial_energy = current.total_energy;
    let initial_max_force = current.max_force;
    let mut accepted_steps = 0usize;
    let mut rejected_steps = 0usize;
    let mut steps_executed = 0usize;
    let mut termination_reason = "max_steps".to_string();
    let max_displacement = (0.20 * step_scale.clamp(0.15, 1.0)).clamp(0.02, 0.12);

    for _ in 0..steps_requested {
        if current.max_force <= FORCE_TOLERANCE {
            termination_reason = "converged_force".into();
            break;
        }
        let direction = current
            .gradient
            .iter()
            .zip(masses.iter())
            .map(|(grad, mass)| grad.scale(-1.0 / mass.max(1.0)))
            .collect::<Vec<_>>();
        let max_direction = direction
            .iter()
            .map(|value| vec_norm_or_zero(*value))
            .fold(0.0f32, f32::max);
        if max_direction <= 1.0e-8 {
            termination_reason = "flat_direction".into();
            break;
        }
        let mut alpha = max_displacement / max_direction;
        let mut accepted = None;
        for _ in 0..LINE_SEARCH_MAX_BACKTRACKS {
            let mut trial_positions = current_positions
                .iter()
                .zip(direction.iter())
                .map(|(position, dir)| position.add(dir.scale(alpha)))
                .collect::<Vec<_>>();
            let current_center = center_of_geometry(&trial_positions);
            let shift = current_center.sub(initial_center);
            for position in &mut trial_positions {
                *position = position.sub(shift);
            }
            let trial = evaluate_energy(&trial_positions, topology, &nonbonded_pairs);
            if trial.total_energy + 1.0e-4 < current.total_energy {
                accepted = Some((trial_positions, trial));
                break;
            }
            rejected_steps = rejected_steps.saturating_add(1);
            alpha *= 0.5;
        }
        let Some((next_positions, next_eval)) = accepted else {
            termination_reason = "line_search_failed".into();
            break;
        };
        current_positions = next_positions;
        current = next_eval;
        accepted_steps = accepted_steps.saturating_add(1);
        steps_executed = steps_executed.saturating_add(1);
    }

    for (atom, position) in output.atoms.iter_mut().zip(current_positions.iter()) {
        atom.position = *position;
    }
    let moved_atom_count = initial_positions
        .iter()
        .zip(current_positions.iter())
        .filter(|(initial, current)| current.sub(**initial).norm() > 1.0e-4)
        .count();
    MinimizationTelemetry {
        steps_executed,
        accepted_steps,
        rejected_steps,
        initial_energy,
        final_energy: current.total_energy,
        initial_max_force,
        final_max_force: current.max_force,
        moved_atom_count,
        termination_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use warp_structure::{AtomRecord, AtomRecordKind};

    fn atom(name: &str, element: &str, position: Vec3) -> AtomRecord {
        AtomRecord {
            record_kind: AtomRecordKind::Atom,
            name: name.into(),
            element: element.into(),
            resname: "TST".into(),
            resid: 1,
            chain: 'A',
            segid: String::new(),
            charge: 0.0,
            position,
            mol_id: 1,
            pdb_metadata: None,
        }
    }

    fn chain_topology() -> AmberTopology {
        AmberTopology {
            atom_names: vec!["C1".into(), "C2".into(), "C3".into(), "C4".into()],
            residue_labels: vec!["TST".into()],
            residue_pointers: vec![1],
            atomic_numbers: vec![6, 6, 6, 6],
            masses: vec![12.011; 4],
            charges: vec![0.0; 4],
            atom_type_indices: vec![1; 4],
            amber_atom_types: vec!["CT".into(); 4],
            radii: vec![1.7; 4],
            screen: vec![0.72; 4],
            bonds: vec![(0, 1), (1, 2), (2, 3)],
            bond_type_indices: vec![1, 1, 1],
            bond_force_constants: vec![320.0],
            bond_equil_values: vec![1.54],
            angles: vec![[0, 1, 2], [1, 2, 3]],
            angle_type_indices: vec![1, 1],
            angle_force_constants: vec![70.0],
            angle_equil_values: vec![1.9106332],
            dihedrals: vec![[0, 1, 2, 3]],
            dihedral_type_indices: vec![1],
            dihedral_force_constants: vec![1.0],
            dihedral_periodicities: vec![3.0],
            dihedral_phases: vec![0.0],
            scee_scale_factors: vec![1.2],
            scnb_scale_factors: vec![2.0],
            solty: vec![0.0],
            impropers: Vec::new(),
            improper_type_indices: Vec::new(),
            excluded_atoms: vec![vec![2, 3], vec![1, 3, 4], vec![1, 2, 4], vec![2, 3]],
            nonbonded_parm_index: vec![1],
            lennard_jones_acoef: vec![10_000.0],
            lennard_jones_bcoef: vec![100.0],
            lennard_jones_14_acoef: vec![10_000.0],
            lennard_jones_14_bcoef: vec![100.0],
            hbond_acoef: vec![0.0],
            hbond_bcoef: vec![0.0],
            hbcut: vec![0.0],
            tree_chain_classification: vec!["M".into(); 4],
            join_array: vec![0; 4],
            irotat: vec![0; 4],
            solvent_pointers: Vec::new(),
            atoms_per_molecule: vec![4],
            box_dimensions: Vec::new(),
            radius_set: Some("modified Bondi radii".into()),
            ipol: 0,
        }
    }

    #[test]
    fn minimizer_reduces_simple_chain_energy() {
        let mut output = PackOutput {
            atoms: vec![
                atom("C1", "C", Vec3::new(0.0, 0.0, 0.0)),
                atom("C2", "C", Vec3::new(1.54, 0.0, 0.0)),
                atom("C3", "C", Vec3::new(3.08, 0.0, 0.0)),
                atom("C4", "C", Vec3::new(0.2, 0.2, 0.0)),
            ],
            bonds: vec![(0, 1), (1, 2), (2, 3)],
            box_size: [0.0, 0.0, 0.0],
            ter_after: vec![3],
            box_vectors: None,
        };
        let topology = chain_topology();
        let telemetry = minimize_synthetic_topology(&mut output, &topology, 32, 0.5);
        assert!(telemetry.steps_executed > 0, "{telemetry:#?}");
        assert!(
            telemetry.final_energy < telemetry.initial_energy,
            "{telemetry:#?}"
        );
        assert!(
            telemetry.final_max_force < telemetry.initial_max_force,
            "{telemetry:#?}"
        );
    }

    #[test]
    fn one_four_pairs_survive_exclusion_table() {
        let topology = chain_topology();
        let pairs = build_nonbonded_pairs(&topology);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, 0);
        assert_eq!(pairs[0].right, 3);
        assert!(pairs[0].param.coulomb_scale > 0.0);
    }
}
