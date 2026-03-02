#[cfg(feature = "cuda")]
use std::cell::RefCell;

use rand::{rngs::StdRng, Rng};

use crate::atom_params::AtomParams;
use crate::config::{PackConfig, StructureSpec};
use crate::constraint_penalty::penalty_and_grad;
#[cfg(feature = "cuda")]
use crate::constraints::ConstraintSpec;
use crate::error::{PackError, PackResult};
use crate::gencan::optimize_gencan;
use crate::geom::{Quaternion, Vec3};
use crate::pack::AtomRecord;
use crate::pack_ops::{
    random_center, satisfies_structure_constraints, transform_positions_into, TemplateEntry,
};
use crate::pbc::PbcBox;
use crate::placement::PlacementRecord;
use crate::spatial_hash::SpatialHashV2;

#[cfg(feature = "cuda")]
use crate::gpu_cells::build_gpu_cell_list;
#[cfg(feature = "cuda")]
use crate::gpu_constraints::build_constraint_gpu_data;
#[cfg(feature = "cuda")]
use traj_gpu::{Float4, GpuContext};

#[cfg(feature = "cuda")]
thread_local! {
    static GPU_CTX: RefCell<Option<GpuContext>> = RefCell::new(None);
}

#[cfg(feature = "cuda")]
const GPU_NAIVE_MAX_ATOMS: usize = 2048;

#[cfg(feature = "cuda")]
fn gpu_enabled() -> bool {
    match std::env::var("WARP_PACK_USE_GPU") {
        Ok(val) => matches!(
            val.as_str(),
            "1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON"
        ),
        Err(_) => false,
    }
}

#[cfg(feature = "cuda")]
fn compute_atom_overlap_gpu(
    positions: &[Vec3],
    params: &[AtomParams],
    atoms: &[AtomRecord],
    pbc: Option<PbcBox>,
    cell_size: f32,
) -> Option<Vec<f32>> {
    if !gpu_enabled() {
        return None;
    }
    if positions.len() != params.len() || positions.len() != atoms.len() {
        return None;
    }
    let ctx = GPU_CTX.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = GpuContext::new(0).ok();
        }
        slot.clone()
    })?;
    let coords: Vec<Float4> = positions
        .iter()
        .map(|p| Float4 {
            x: p.x,
            y: p.y,
            z: p.z,
            w: 0.0,
        })
        .collect();
    let radii: Vec<f32> = params.iter().map(|p| p.radius).collect();
    let mol_id: Vec<i32> = atoms.iter().map(|a| a.mol_id).collect();
    if let Some(cells) = build_gpu_cell_list(positions, cell_size, pbc) {
        let box_min = [cells.origin.x, cells.origin.y, cells.origin.z];
        let box_len = pbc
            .map(|b| [b.length.x, b.length.y, b.length.z])
            .unwrap_or([0.0, 0.0, 0.0]);
        if let Ok(res) = ctx.pack_overlap_max_cells(
            &coords,
            &radii,
            &mol_id,
            &cells.offsets,
            &cells.atoms,
            cells.dims,
            box_min,
            box_len,
            cells.cell_size,
        ) {
            return Some(res);
        }
    }
    if positions.len() <= GPU_NAIVE_MAX_ATOMS {
        let box_l = pbc.map(|b| [b.length.x, b.length.y, b.length.z]);
        return ctx.pack_overlap_max(&coords, &radii, &mol_id, box_l).ok();
    }
    None
}

pub(crate) struct MovebadIndex {
    per_spec: Vec<Vec<i32>>,
    mol_atom_indices: Vec<Vec<usize>>,
}

pub(crate) fn build_movebad_index(
    atoms: &[AtomRecord],
    mol_spec: &[usize],
    n_specs: usize,
) -> MovebadIndex {
    MovebadIndex {
        per_spec: collect_molecules_per_spec(mol_spec, n_specs),
        mol_atom_indices: collect_mol_atom_indices(atoms, mol_spec.len()),
    }
}

pub(crate) fn run_movebad_pass(
    cfg: &PackConfig,
    templates: &[TemplateEntry],
    atoms: &mut Vec<AtomRecord>,
    positions: &mut Vec<Vec3>,
    params: &mut Vec<AtomParams>,
    hash: &mut SpatialHashV2,
    placements: &mut [PlacementRecord],
    box_size: [f32; 3],
    box_origin: Vec3,
    pbc: Option<PbcBox>,
    rng: &mut StdRng,
    mol_spec: &[usize],
    index: &MovebadIndex,
    active_spec: Option<usize>,
    include_overlap: bool,
    cell_size: f32,
) -> PackResult<()> {
    let movefrac = cfg.movefrac.unwrap_or(0.05);
    if movefrac <= 0.0 {
        return Ok(());
    }
    let precision = cfg.precision.unwrap_or(1.0e-2);
    let (bounds_min, bounds_max) = compute_bounds(cfg, positions, params, box_origin, box_size);
    let per_spec = &index.per_spec;
    let total_mols = mol_spec.len();
    let mol_atom_indices = &index.mol_atom_indices;
    let atom_overlap = if include_overlap {
        compute_atom_overlap(
            positions,
            params,
            atoms,
            mol_spec,
            pbc,
            cell_size,
            active_spec,
            hash,
        )
    } else {
        vec![0.0f32; positions.len()]
    };
    let atom_constraint = compute_atom_constraint(
        atoms,
        positions,
        cfg,
        mol_spec,
        mol_atom_indices,
        active_spec,
    );
    let (mol_fdist, mol_frest) =
        compute_mol_scores(total_mols, atoms, &atom_overlap, &atom_constraint);
    let mut moved_any = false;
    for (spec_index, mol_ids) in per_spec.iter().enumerate() {
        if active_spec.is_some() && active_spec != Some(spec_index) {
            continue;
        }
        if mol_ids.is_empty() {
            continue;
        }
        let spec = &cfg.structures[spec_index];
        let entry = &templates[spec_index];
        let mut scores = Vec::with_capacity(mol_ids.len());
        let mut nbad = 0usize;
        for &mol_id in mol_ids {
            let mol_idx = mol_id.max(1) as usize - 1;
            let fdist = *mol_fdist.get(mol_idx).unwrap_or(&0.0);
            let frest = *mol_frest.get(mol_idx).unwrap_or(&0.0);
            let bad = fdist > precision || frest > precision;
            if bad {
                nbad += 1;
            }
            let score = if bad { fdist + frest } else { 0.0 };
            scores.push((mol_id, score));
        }
        if nbad == 0 {
            continue;
        }
        let frac = (nbad as f32 / mol_ids.len() as f32).min(movefrac);
        let good_pool_len = ((mol_ids.len() as f32) * frac).floor().max(1.0) as usize;
        let mut nmove = ((mol_ids.len() as f32) * frac).floor().max(1.0) as usize;
        nmove = nmove.min(entry.maxmove.max(1)).min(mol_ids.len());
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let good_pool_len = good_pool_len.min(scores.len());
        let bad_start = scores.len().saturating_sub(nmove);
        let bad_list: Vec<i32> = scores[bad_start..].iter().map(|v| v.0).collect();

        for target in bad_list {
            let target_index = (target.max(1) - 1) as usize;
            let Some(target_atom_indices) = mol_atom_indices.get(target_index) else {
                continue;
            };
            if target_atom_indices.is_empty() {
                continue;
            }

            // Store old positions for incremental hash updates
            let mut old_positions: Vec<(usize, Vec3)> =
                Vec::with_capacity(target_atom_indices.len());
            for &atom_idx in target_atom_indices {
                old_positions.push((atom_idx, positions[atom_idx]));
            }

            let good_idx = rng.gen_range(0..good_pool_len);
            let good_mol = scores[good_idx].0;
            let good_index = (good_mol.max(1) - 1) as usize;
            let anchor_center = if cfg.movebadrandom {
                None
            } else {
                placements.get(good_index).map(|p| p.center)
            };
            let anchor_euler = placements.get(good_index).map(|p| p.euler);
            let bounds = if cfg.movebadrandom {
                Some((bounds_min, bounds_max))
            } else {
                None
            };
            let (record, _needs_constraints) = relocate_molecule_in_place(
                cfg,
                spec,
                entry,
                atoms,
                positions,
                params,
                hash,
                box_size,
                box_origin,
                pbc,
                rng,
                target_atom_indices,
                anchor_center,
                anchor_euler,
                bounds,
                false,
                cell_size,
            )?;
            if let Some(slot) = placements.get_mut(target_index) {
                *slot = record;
            }

            // Incremental hash update: update only moved atoms
            for (atom_idx, old_pos) in old_positions {
                let new_pos = positions[atom_idx];
                hash.update(atom_idx, old_pos, new_pos);
            }

            if _needs_constraints
                && (!spec.constraints.is_empty() || !spec.atom_constraints.is_empty())
            {
                optimize_target_constraints(
                    cfg,
                    spec_index,
                    target_index,
                    target_atom_indices,
                    mol_spec,
                    templates,
                    atoms,
                    positions,
                    params,
                    placements,
                    box_size,
                    box_origin,
                    pbc,
                )?;
            }
            moved_any = true;
        }
    }

    // No longer need to rebuild hash - incremental updates already done
    Ok(())
}

fn collect_molecules_per_spec(mol_spec: &[usize], n_specs: usize) -> Vec<Vec<i32>> {
    let mut per_spec: Vec<Vec<i32>> = vec![Vec::new(); n_specs];
    for (mol_idx, &spec_index) in mol_spec.iter().enumerate() {
        if spec_index < per_spec.len() {
            per_spec[spec_index].push((mol_idx + 1) as i32);
        }
    }
    per_spec
}

fn collect_mol_atom_indices(atoms: &[AtomRecord], total_mols: usize) -> Vec<Vec<usize>> {
    let mut per_mol = vec![Vec::new(); total_mols];
    for (atom_idx, atom) in atoms.iter().enumerate() {
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        if let Some(list) = per_mol.get_mut(mol_idx) {
            list.push(atom_idx);
        }
    }
    per_mol
}

fn compute_atom_overlap(
    positions: &[Vec3],
    params: &[AtomParams],
    atoms: &[AtomRecord],
    mol_spec: &[usize],
    pbc: Option<PbcBox>,
    _cell_size: f32,
    active_spec: Option<usize>,
    hash: &SpatialHashV2,
) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    if active_spec.is_none() {
        if let Some(overlap) = compute_atom_overlap_gpu(positions, params, atoms, pbc, _cell_size) {
            return overlap;
        }
    }
    let mut overlap = vec![0.0f32; positions.len()];
    if let Some(active) = active_spec {
        let mut selected_mask = vec![false; positions.len()];
        let mut selected_indices = Vec::new();
        for (atom_idx, atom) in atoms.iter().enumerate() {
            let mol_idx = atom.mol_id.max(1) as usize - 1;
            if mol_spec.get(mol_idx).copied() == Some(active) {
                if !selected_mask[atom_idx] {
                    selected_mask[atom_idx] = true;
                    selected_indices.push(atom_idx);
                }
            }
        }
        if selected_indices.is_empty() {
            return overlap;
        }
        selected_indices.sort_unstable();
        for &i in selected_indices.iter() {
            let pos_i = positions[i];
            hash.for_each_neighbor(pos_i, |j| {
                if j == i {
                    return;
                }
                let j_selected = selected_mask.get(j).copied().unwrap_or(false);
                if !j_selected || j <= i {
                    return;
                }
                if atoms[i].mol_id == atoms[j].mol_id {
                    return;
                }
                let delta = if let Some(pbc_box) = pbc {
                    pbc_box.delta(pos_i, positions[j])
                } else {
                    pos_i.sub(positions[j])
                };
                let target = params[i].radius + params[j].radius;
                let d2 = delta.dot(delta);
                let tol2 = target * target;
                if d2 < tol2 {
                    let diff = tol2 - d2;
                    overlap[i] = overlap[i].max(diff);
                    overlap[j] = overlap[j].max(diff);
                }
            });
        }
        return overlap;
    }
    for (i, pos_i) in positions.iter().enumerate() {
        hash.for_each_neighbor(*pos_i, |j| {
            if j <= i {
                return;
            }
            let mol_i_id = atoms[i].mol_id;
            let mol_j_id = atoms[j].mol_id;
            if mol_i_id == mol_j_id {
                return;
            }
            let delta = if let Some(pbc_box) = pbc {
                pbc_box.delta(*pos_i, positions[j])
            } else {
                pos_i.sub(positions[j])
            };
            let target = params[i].radius + params[j].radius;
            let d2 = delta.dot(delta);
            let tol2 = target * target;
            if d2 < tol2 {
                let diff = tol2 - d2;
                overlap[i] = overlap[i].max(diff);
                overlap[j] = overlap[j].max(diff);
            }
        });
    }
    overlap
}

#[cfg(feature = "cuda")]
fn compute_atom_constraint_gpu(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    cfg: &PackConfig,
    mol_spec: &[usize],
    active_spec: Option<usize>,
) -> Option<Vec<f32>> {
    if !gpu_enabled() {
        return None;
    }
    if active_spec.is_some() {
        return None;
    }
    if atoms.len() != positions.len() || atoms.is_empty() {
        return None;
    }
    let total_mols = mol_spec.len();
    if total_mols == 0 {
        return None;
    }
    if mol_spec.iter().any(|&spec_idx| {
        cfg.structures
            .get(spec_idx)
            .map(|s| !s.atom_constraints.is_empty())
            .unwrap_or(false)
    }) {
        return None;
    }
    let mut mol_constraints: Vec<&[ConstraintSpec]> = Vec::with_capacity(total_mols);
    for &spec_idx in mol_spec {
        let constraints = cfg
            .structures
            .get(spec_idx)
            .map(|s| s.constraints.as_slice())
            .unwrap_or(&[]);
        mol_constraints.push(constraints);
    }
    let data = build_constraint_gpu_data(atoms, &mol_constraints)?;
    let ctx = GPU_CTX.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = GpuContext::new(0).ok();
        }
        slot.clone()
    })?;

    let coords: Vec<Float4> = positions
        .iter()
        .map(|p| Float4 {
            x: p.x,
            y: p.y,
            z: p.z,
            w: 0.0,
        })
        .collect();

    let (sum, _grad, _max_val, _max_violation) = ctx
        .pack_constraint_penalty(
            &coords,
            &data.types,
            &data.modes,
            &data.data0,
            &data.data1,
            &data.atom_offsets,
            &data.atom_indices,
        )
        .ok()?;

    Some(sum)
}

fn compute_atom_constraint(
    _atoms: &[AtomRecord],
    positions: &[Vec3],
    cfg: &PackConfig,
    mol_spec: &[usize],
    mol_atom_indices: &[Vec<usize>],
    active_spec: Option<usize>,
) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    if let Some(violations) =
        compute_atom_constraint_gpu(_atoms, positions, cfg, mol_spec, active_spec)
    {
        return violations;
    }
    let mut violations = vec![0.0f32; positions.len()];
    for (mol_idx, &spec_idx) in mol_spec.iter().enumerate() {
        if active_spec.is_some() && active_spec != Some(spec_idx) {
            continue;
        }
        let spec = &cfg.structures[spec_idx];
        if spec.constraints.is_empty() && spec.atom_constraints.is_empty() {
            continue;
        }
        let Some(atom_indices) = mol_atom_indices.get(mol_idx) else {
            continue;
        };
        for (local_zero_idx, &atom_idx) in atom_indices.iter().enumerate() {
            if atom_idx >= positions.len() {
                continue;
            }
            let mut sum_v = 0.0f32;
            for constraint in &spec.constraints {
                let pen = penalty_and_grad(positions[atom_idx], constraint);
                if pen.value > 0.0 {
                    sum_v += pen.value;
                }
            }
            let local_idx = local_zero_idx + 1;
            for atom_constraint in &spec.atom_constraints {
                if atom_constraint.indices.iter().any(|&idx| idx == local_idx) {
                    let pen = penalty_and_grad(positions[atom_idx], &atom_constraint.constraint);
                    if pen.value > 0.0 {
                        sum_v += pen.value;
                    }
                }
            }
            violations[atom_idx] = sum_v;
        }
    }
    violations
}

fn compute_mol_scores(
    total_mols: usize,
    atoms: &[AtomRecord],
    atom_overlap: &[f32],
    atom_constraint: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut fdist = vec![0.0f32; total_mols];
    let mut frest = vec![0.0f32; total_mols];
    for (i, atom) in atoms.iter().enumerate() {
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        if mol_idx >= total_mols {
            continue;
        }
        fdist[mol_idx] = fdist[mol_idx].max(atom_overlap[i]);
        frest[mol_idx] = frest[mol_idx].max(atom_constraint[i]);
    }
    (fdist, frest)
}

fn optimize_target_constraints(
    cfg: &PackConfig,
    spec_index: usize,
    target_index: usize,
    target_atom_indices: &[usize],
    mol_spec: &[usize],
    templates: &[TemplateEntry],
    atoms: &mut Vec<AtomRecord>,
    positions: &mut Vec<Vec3>,
    params: &[AtomParams],
    placements: &mut [PlacementRecord],
    box_size: [f32; 3],
    box_origin: Vec3,
    pbc: Option<PbcBox>,
) -> PackResult<()> {
    if target_atom_indices.is_empty() {
        return Ok(());
    }
    let spec = &cfg.structures[spec_index];
    let mut fast_cfg = cfg.clone();
    let fast_maxit = cfg.gencan_maxit.unwrap_or(20).min(8).max(1);
    fast_cfg.gencan_maxit = Some(fast_maxit);
    let _ = optimize_gencan(
        &fast_cfg,
        atoms,
        positions,
        params,
        placements,
        mol_spec,
        templates,
        box_size,
        box_origin.to_array(),
        pbc,
        false,
        Some(spec_index),
        Some(target_index),
        None,
    )?;
    if !molecule_satisfies_constraints(spec, positions, target_atom_indices, pbc) {
        let _ = optimize_gencan(
            cfg,
            atoms,
            positions,
            params,
            placements,
            mol_spec,
            templates,
            box_size,
            box_origin.to_array(),
            pbc,
            false,
            Some(spec_index),
            Some(target_index),
            None,
        )?;
    }
    Ok(())
}

fn molecule_satisfies_constraints(
    spec: &StructureSpec,
    positions: &[Vec3],
    target_atom_indices: &[usize],
    pbc: Option<PbcBox>,
) -> bool {
    let mut points = Vec::with_capacity(target_atom_indices.len());
    for &idx in target_atom_indices {
        points.push(positions[idx]);
    }
    satisfies_structure_constraints(&points, spec, pbc)
}

fn compute_bounds(
    cfg: &PackConfig,
    positions: &[Vec3],
    params: &[AtomParams],
    origin: Vec3,
    box_size: [f32; 3],
) -> (Vec3, Vec3) {
    if let Some(sidemax) = cfg.sidemax {
        let center = Vec3::new(
            origin.x + box_size[0] * 0.5,
            origin.y + box_size[1] * 0.5,
            origin.z + box_size[2] * 0.5,
        );
        let min = Vec3::new(center.x - sidemax, center.y - sidemax, center.z - sidemax);
        let max = Vec3::new(center.x + sidemax, center.y + sidemax, center.z + sidemax);
        return (min, max);
    }
    if positions.is_empty() {
        let max = Vec3::new(
            origin.x + box_size[0],
            origin.y + box_size[1],
            origin.z + box_size[2],
        );
        return (origin, max);
    }
    let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for pos in positions {
        min.x = min.x.min(pos.x);
        min.y = min.y.min(pos.y);
        min.z = min.z.min(pos.z);
        max.x = max.x.max(pos.x);
        max.y = max.y.max(pos.y);
        max.z = max.z.max(pos.z);
    }
    let radmax = params.iter().fold(0.0f32, |acc, p| acc.max(p.radius));
    let pad = 1.1 * radmax;
    min = Vec3::new(min.x - pad, min.y - pad, min.z - pad);
    max = Vec3::new(max.x + pad, max.y + pad, max.z + pad);
    (min, max)
}

fn overlaps_except_indices(
    hash: &SpatialHashV2,
    candidate_positions: &[Vec3],
    candidate_params: &[AtomParams],
    existing_positions: &[Vec3],
    existing_params: &[AtomParams],
    skipped_indices: &[usize],
) -> bool {
    for (cand_idx, cand_pos) in candidate_positions.iter().enumerate() {
        let mut overlap = false;
        hash.for_each_neighbor(*cand_pos, |idx| {
            if overlap || skipped_indices.binary_search(&idx).is_ok() {
                return;
            }
            let delta = cand_pos.sub(existing_positions[idx]);
            let d2 = delta.dot(delta);
            let min_r = candidate_params[cand_idx].radius + existing_params[idx].radius;
            if d2 < min_r * min_r {
                overlap = true;
            }
        });
        if overlap {
            return true;
        }
    }
    false
}

fn relocate_molecule_in_place(
    cfg: &PackConfig,
    spec: &StructureSpec,
    entry: &TemplateEntry,
    atoms: &mut [AtomRecord],
    positions: &mut [Vec3],
    params: &mut [AtomParams],
    hash: &mut SpatialHashV2,
    box_size: [f32; 3],
    box_origin: Vec3,
    pbc: Option<PbcBox>,
    rng: &mut StdRng,
    target_atom_indices: &[usize],
    anchor: Option<Vec3>,
    anchor_euler: Option<[f32; 3]>,
    bounds: Option<(Vec3, Vec3)>,
    check_overlap: bool,
    cell_size: f32,
) -> PackResult<(PlacementRecord, bool)> {
    if target_atom_indices.len() != entry.template.atoms.len() {
        return Err(PackError::Placement(
            "movebad atom mapping mismatch for target molecule".into(),
        ));
    }
    let has_constraints = !spec.constraints.is_empty() || !spec.atom_constraints.is_empty();
    let direct_constraint_attempts = 1usize;
    let mut sorted_target_indices: Vec<usize> = target_atom_indices.to_vec();
    sorted_target_indices.sort_unstable();
    let mut candidate_positions = Vec::with_capacity(entry.template.atoms.len());

    // Store old positions for incremental hash updates
    let mut old_positions = Vec::with_capacity(target_atom_indices.len());
    for &atom_idx in target_atom_indices {
        old_positions.push(positions[atom_idx]);
    }

    for attempt in 0..cfg.max_attempts.unwrap_or(10000) {
        let (rotation, euler) = if spec.rotate {
            if let Some(anchor_euler) = anchor_euler {
                (
                    Quaternion::from_packmol_euler(
                        anchor_euler[0],
                        anchor_euler[1],
                        anchor_euler[2],
                    ),
                    anchor_euler,
                )
            } else if let Some(bounds) = spec.rot_bounds {
                let bx = rng.gen_range(bounds[0][0]..=bounds[0][1]);
                let gy = rng.gen_range(bounds[1][0]..=bounds[1][1]);
                let tz = rng.gen_range(bounds[2][0]..=bounds[2][1]);
                (Quaternion::from_packmol_euler(bx, gy, tz), [bx, gy, tz])
            } else {
                let rot = Quaternion::random(rng);
                let (b, g, t) = rot.to_packmol_euler();
                (rot, [b, g, t])
            }
        } else {
            (Quaternion::identity(), [0.0, 0.0, 0.0])
        };
        let center = if let Some(anchor) = anchor {
            let local_anchor = anchor.sub(box_origin);
            let jitter = (0.3 * entry.template.dmax).max(1.0e-3);
            let local_center = random_center_near(rng, local_anchor, jitter)?;
            local_center.add(box_origin)
        } else if let Some((min, max)) = bounds {
            random_center_bounds(rng, min, max)
        } else {
            let mut center = random_center(rng, box_size, entry.template.radius)?;
            center = center.add(box_origin);
            center
        };
        transform_positions_into(&entry.template, rotation, center, &mut candidate_positions);
        let constraints_ok = satisfies_structure_constraints(&candidate_positions, spec, pbc);
        if !constraints_ok && !has_constraints {
            continue;
        }
        if !constraints_ok && has_constraints && attempt + 1 < direct_constraint_attempts {
            continue;
        }
        if check_overlap
            && overlaps_except_indices(
                hash,
                &candidate_positions,
                &entry.atom_params,
                positions,
                params,
                &sorted_target_indices,
            )
        {
            continue;
        }
        for (offset, &atom_idx) in target_atom_indices.iter().enumerate() {
            let candidate_pos = candidate_positions[offset];
            positions[atom_idx] = candidate_pos;
            params[atom_idx] = entry.atom_params[offset];
            atoms[atom_idx].position = candidate_pos;
        }
        if check_overlap {
            // Incremental hash update: update only moved atoms
            for (offset, &atom_idx) in target_atom_indices.iter().enumerate() {
                let old_pos = old_positions[offset];
                let new_pos = positions[atom_idx];
                hash.update(atom_idx, old_pos, new_pos);
            }
        }
        return Ok((PlacementRecord::new(center, euler), !constraints_ok));
    }
    Err(PackError::Placement("movebad failed to re-place".into()))
}

fn random_center_near<R: Rng + ?Sized>(rng: &mut R, anchor: Vec3, jitter: f32) -> PackResult<Vec3> {
    let range = jitter.max(1.0e-3);
    let center = Vec3::new(
        anchor.x + rng.gen_range(-range..range),
        anchor.y + rng.gen_range(-range..range),
        anchor.z + rng.gen_range(-range..range),
    );
    Ok(center)
}

fn random_center_bounds<R: Rng + ?Sized>(rng: &mut R, min: Vec3, max: Vec3) -> Vec3 {
    Vec3::new(
        rng.gen_range(min.x..max.x),
        rng.gen_range(min.y..max.y),
        rng.gen_range(min.z..max.z),
    )
}
