use crate::atom_params::AtomParams;
use crate::config::PackConfig;
use crate::error::PackResult;
use crate::geom::Vec3;
use crate::pack::AtomRecord;
use crate::pack_ops::{satisfies_structure_constraints, TemplateEntry};
use crate::pbc::PbcBox;
use crate::placement::PlacementRecord;
use crate::spatial_hash::SpatialHash;

#[cfg(feature = "cuda")]
use crate::gpu_cells::build_gpu_cell_list;
#[cfg(feature = "cuda")]
use std::cell::RefCell;
#[cfg(feature = "cuda")]
use traj_gpu::{Float4, GpuContext};

#[cfg(feature = "cuda")]
thread_local! {
    static GPU_CTX: RefCell<Option<GpuContext>> = RefCell::new(None);
}

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
fn compute_relax_gpu(
    positions: &[Vec3],
    atom_params: &[AtomParams],
    atoms: &[AtomRecord],
    mol_movable: &[bool],
    cell_size: f32,
) -> Option<(Vec<Vec3>, f32)> {
    if !gpu_enabled() {
        return None;
    }
    if positions.len() != atom_params.len() || positions.len() != atoms.len() {
        return None;
    }
    let n_atoms = positions.len();
    let n_mols = mol_movable.len();
    if n_atoms == 0 || n_mols == 0 {
        return None;
    }
    let ctx = GPU_CTX.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = GpuContext::new(0).ok();
        }
        slot.clone()
    })?;

    let cells = build_gpu_cell_list(positions, cell_size, None)?;
    let coords: Vec<Float4> = positions
        .iter()
        .map(|p| Float4 {
            x: p.x,
            y: p.y,
            z: p.z,
            w: 0.0,
        })
        .collect();
    let radii: Vec<f32> = atom_params.iter().map(|p| p.radius).collect();
    let mol_id: Vec<i32> = atoms.iter().map(|a| a.mol_id).collect();
    let mol_movable_u8: Vec<u8> = mol_movable
        .iter()
        .map(|m| if *m { 1u8 } else { 0u8 })
        .collect();

    let box_min = [cells.origin.x, cells.origin.y, cells.origin.z];
    let (disp, max_violation) = ctx
        .pack_relax_accum(
            &coords,
            &radii,
            &mol_id,
            &mol_movable_u8,
            &cells.offsets,
            &cells.atoms,
            cells.dims,
            box_min,
            n_mols,
            cells.cell_size,
        )
        .ok()?;

    let mut disp_vec = Vec::with_capacity(disp.len());
    for d in disp {
        disp_vec.push(Vec3::new(d.x, d.y, d.z));
    }
    Some((disp_vec, max_violation))
}

pub(crate) fn relax_overlaps(
    cfg: &PackConfig,
    atoms: &mut Vec<AtomRecord>,
    positions: &mut Vec<Vec3>,
    atom_params: &mut Vec<AtomParams>,
    placements: &mut [PlacementRecord],
    mol_spec: &[usize],
    templates: &[TemplateEntry],
    box_size: [f32; 3],
    pbc: Option<PbcBox>,
    dist_scale: f32,
) -> PackResult<()> {
    let steps = cfg.relax_steps.unwrap_or(0);
    if steps == 0 || atoms.is_empty() {
        return Ok(());
    }
    let step_scale = cfg.relax_step.unwrap_or(0.5);
    if step_scale <= 0.0 {
        return Ok(());
    }
    let precision = cfg.precision.unwrap_or(1.0e-2);
    let total_mols = placements.len();
    if total_mols == 0 {
        return Ok(());
    }

    let mut mol_atoms: Vec<Vec<usize>> = vec![Vec::new(); total_mols];
    for (idx, atom) in atoms.iter().enumerate() {
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        if mol_idx < total_mols {
            mol_atoms[mol_idx].push(idx);
        }
    }

    let mut mol_radius = vec![0.0f32; total_mols];
    let mut mol_movable = vec![true; total_mols];
    for (mol_idx, &spec_index) in mol_spec.iter().enumerate() {
        if let Some(entry) = templates.get(spec_index) {
            mol_radius[mol_idx] = entry.template.radius;
            mol_movable[mol_idx] = !cfg.structures[spec_index].fixed;
        }
    }

    let fbins = cfg.fbins.unwrap_or(3.0_f32.sqrt());
    let max_radius = atom_params.iter().fold(0.0f32, |acc, p| acc.max(p.radius));
    if max_radius <= 0.0 {
        return Ok(());
    }
    let cell_size = (2.0 * max_radius * fbins * dist_scale).max(1.0e-6);

    for _ in 0..steps {
        let mut disp = vec![Vec3::default(); total_mols];
        let mut max_violation = 0.0f32;
        #[cfg(feature = "cuda")]
        let mut used_gpu = false;
        #[cfg(not(feature = "cuda"))]
        let used_gpu = false;

        #[cfg(feature = "cuda")]
        if let Some((gpu_disp, gpu_max)) =
            compute_relax_gpu(positions, atom_params, atoms, &mol_movable, cell_size)
        {
            disp = gpu_disp;
            max_violation = gpu_max;
            used_gpu = true;
        }

        if !used_gpu {
            let mut hash = SpatialHash::new(cell_size);
            for (i, pos_i) in positions.iter().enumerate() {
                hash.for_each_neighbor(*pos_i, |j| {
                    if j >= i {
                        return;
                    }
                    let mol_i = atoms[i].mol_id.max(1) as usize - 1;
                    let mol_j = atoms[j].mol_id.max(1) as usize - 1;
                    if mol_i == mol_j || mol_i >= total_mols || mol_j >= total_mols {
                        return;
                    }
                    let delta = pos_i.sub(positions[j]);
                    let dist2 = delta.dot(delta);
                    let rij = atom_params[i].radius + atom_params[j].radius;
                    let min2 = rij * rij;
                    if dist2 >= min2 {
                        return;
                    }
                    let dist = dist2.sqrt();
                    let overlap = rij - dist;
                    max_violation = max_violation.max(overlap);
                    let dir = if dist > 1.0e-6 {
                        delta.scale(1.0 / dist)
                    } else {
                        fallback_dir(i, j)
                    };
                    if mol_movable[mol_i] && mol_movable[mol_j] {
                        disp[mol_i] = disp[mol_i].add(dir.scale(overlap * 0.5));
                        disp[mol_j] = disp[mol_j].sub(dir.scale(overlap * 0.5));
                    } else if mol_movable[mol_i] {
                        disp[mol_i] = disp[mol_i].add(dir.scale(overlap));
                    } else if mol_movable[mol_j] {
                        disp[mol_j] = disp[mol_j].sub(dir.scale(overlap));
                    }
                });
                hash.insert(i, *pos_i);
            }
        }

        if max_violation <= precision {
            break;
        }

        for mol_idx in 0..total_mols {
            if !mol_movable[mol_idx] {
                continue;
            }
            let atom_count = mol_atoms[mol_idx].len().max(1) as f32;
            let delta = disp[mol_idx].scale(step_scale / atom_count);
            if delta.norm() <= 1.0e-6 {
                continue;
            }
            let radius = mol_radius[mol_idx];
            let mut new_center = placements[mol_idx].center.add(delta);
            if box_size[0] > 2.0 * radius {
                new_center.x = new_center.x.clamp(radius, box_size[0] - radius);
            }
            if box_size[1] > 2.0 * radius {
                new_center.y = new_center.y.clamp(radius, box_size[1] - radius);
            }
            if box_size[2] > 2.0 * radius {
                new_center.z = new_center.z.clamp(radius, box_size[2] - radius);
            }
            let delta = new_center.sub(placements[mol_idx].center);
            if delta.norm() <= 1.0e-6 {
                continue;
            }
            let spec_index = mol_spec[mol_idx];
            let spec = &cfg.structures[spec_index];
            let mut cand_positions = Vec::with_capacity(mol_atoms[mol_idx].len());
            for &atom_idx in mol_atoms[mol_idx].iter() {
                cand_positions.push(positions[atom_idx].add(delta));
            }
            if !satisfies_structure_constraints(&cand_positions, spec, pbc) {
                continue;
            }
            for (offset, &atom_idx) in mol_atoms[mol_idx].iter().enumerate() {
                let new_pos = cand_positions[offset];
                positions[atom_idx] = new_pos;
                atoms[atom_idx].position = new_pos;
            }
            placements[mol_idx].center = placements[mol_idx].center.add(delta);
        }
    }

    Ok(())
}

fn fallback_dir(i: usize, j: usize) -> Vec3 {
    let mut seed = (i as u64).wrapping_mul(6364136223846793005);
    seed = seed
        .wrapping_add(j as u64)
        .wrapping_add(1442695040888963407);
    let x = ((seed & 0xFF) as f32 / 255.0) * 2.0 - 1.0;
    let y = (((seed >> 8) & 0xFF) as f32 / 255.0) * 2.0 - 1.0;
    let z = (((seed >> 16) & 0xFF) as f32 / 255.0) * 2.0 - 1.0;
    let v = Vec3::new(x, y, z);
    let norm = v.norm();
    if norm > 1.0e-6 {
        v.scale(1.0 / norm)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    }
}
