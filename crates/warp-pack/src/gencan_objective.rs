use crate::atom_params::AtomParams;
use crate::constraint_penalty::penalty_and_grad;
use crate::constraints::ConstraintSpec;
use crate::error::{PackError, PackResult};
use crate::gencan_math::{mat_vec, rotation_matrix, rotation_with_derivatives};
use crate::geom::Vec3;
use crate::pack::AtomRecord;
use crate::pbc::PbcBox;
use crate::placement::PlacementRecord;

#[cfg(feature = "cuda")]
use std::cell::RefCell;

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
fn compute_overlap_gpu(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    atom_params: &[AtomParams],
    mols: &[MolInfo],
    pbc: Option<PbcBox>,
    cell_size: f32,
    want_grad: bool,
) -> Option<(f32, f32, Option<Vec<Vec3>>)> {
    if !gpu_enabled() {
        return None;
    }
    if atoms.len() != positions.len() || atoms.len() != atom_params.len() {
        return None;
    }
    let n_atoms = atoms.len();
    if n_atoms == 0 {
        return None;
    }
    let ctx = GPU_CTX.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = GpuContext::new(0).ok();
        }
        slot.clone()
    })?;

    let cells = build_gpu_cell_list(positions, cell_size, pbc)?;

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
    let fscale: Vec<f32> = atom_params.iter().map(|p| p.fscale).collect();
    let mol_id: Vec<i32> = atoms.iter().map(|a| a.mol_id).collect();
    let mut movable = vec![1u8; n_atoms];
    for (idx, atom) in atoms.iter().enumerate() {
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        let is_movable = mols.get(mol_idx).map(|m| m.movable).unwrap_or(true);
        movable[idx] = if is_movable { 1 } else { 0 };
    }

    let box_min = [cells.origin.x, cells.origin.y, cells.origin.z];
    let box_len = pbc
        .map(|b| [b.length.x, b.length.y, b.length.z])
        .unwrap_or([0.0, 0.0, 0.0]);

    let per_atom = ctx
        .pack_overlap_penalty_cells(
            &coords,
            &radii,
            &fscale,
            &mol_id,
            &movable,
            &cells.offsets,
            &cells.atoms,
            cells.dims,
            box_min,
            box_len,
            cells.cell_size,
        )
        .ok()?;

    let per_atom_max = ctx
        .pack_overlap_max_cells_movable(
            &coords,
            &radii,
            &mol_id,
            &movable,
            &cells.offsets,
            &cells.atoms,
            cells.dims,
            box_min,
            box_len,
            cells.cell_size,
        )
        .ok()?;

    let mut value = 0.0f32;
    let mut max_overlap = 0.0f32;
    for i in 0..n_atoms {
        value += per_atom[i];
        let ov = per_atom_max[i];
        if ov > max_overlap {
            max_overlap = ov;
        }
    }
    let mut grad_out = None;
    if want_grad {
        if let Ok(grads) = ctx.pack_overlap_grad_cells(
            &coords,
            &radii,
            &fscale,
            &mol_id,
            &movable,
            &cells.offsets,
            &cells.atoms,
            cells.dims,
            box_min,
            box_len,
            cells.cell_size,
        ) {
            let mut grad_pos = Vec::with_capacity(grads.len());
            for g in grads {
                grad_pos.push(Vec3::new(g.x, g.y, g.z));
            }
            grad_out = Some(grad_pos);
        }
    }

    Some((value, max_overlap, grad_out))
}

#[cfg(feature = "cuda")]
fn compute_short_tol_gpu(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    atom_params: &[AtomParams],
    mols: &[MolInfo],
    pbc: Option<PbcBox>,
    cell_size: f32,
    want_grad: bool,
) -> Option<(f32, Option<Vec<Vec3>>)> {
    if !gpu_enabled() {
        return None;
    }
    if atoms.len() != positions.len() || atoms.len() != atom_params.len() {
        return None;
    }
    let n_atoms = atoms.len();
    if n_atoms == 0 {
        return None;
    }
    let ctx = GPU_CTX.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            *slot = GpuContext::new(0).ok();
        }
        slot.clone()
    })?;

    let cells = build_gpu_cell_list(positions, cell_size, pbc)?;

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
    let short_radius: Vec<f32> = atom_params.iter().map(|p| p.short_radius).collect();
    let fscale: Vec<f32> = atom_params.iter().map(|p| p.fscale).collect();
    let short_scale: Vec<f32> = atom_params.iter().map(|p| p.short_scale).collect();
    let use_short: Vec<u8> = atom_params
        .iter()
        .map(|p| if p.use_short { 1u8 } else { 0u8 })
        .collect();
    let mol_id: Vec<i32> = atoms.iter().map(|a| a.mol_id).collect();
    let mut movable = vec![1u8; n_atoms];
    for (idx, atom) in atoms.iter().enumerate() {
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        let is_movable = mols.get(mol_idx).map(|m| m.movable).unwrap_or(true);
        movable[idx] = if is_movable { 1 } else { 0 };
    }

    let box_min = [cells.origin.x, cells.origin.y, cells.origin.z];
    let box_len = pbc
        .map(|b| [b.length.x, b.length.y, b.length.z])
        .unwrap_or([0.0, 0.0, 0.0]);

    let (penalty, grad) = ctx
        .pack_short_tol_penalty_grad_cells(
            &coords,
            &radii,
            &short_radius,
            &fscale,
            &short_scale,
            &use_short,
            &mol_id,
            &movable,
            &cells.offsets,
            &cells.atoms,
            cells.dims,
            box_min,
            box_len,
            cells.cell_size,
        )
        .ok()?;

    let mut value = 0.0f32;
    for v in penalty {
        value += v;
    }
    let grad_out = if want_grad {
        let mut out = Vec::with_capacity(grad.len());
        for g in grad {
            out.push(Vec3::new(g.x, g.y, g.z));
        }
        Some(out)
    } else {
        None
    };

    Some((value, grad_out))
}

#[cfg(feature = "cuda")]
fn compute_constraints_gpu(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    mols: &[MolInfo],
    want_grad: bool,
) -> Option<(f32, f32, Option<Vec<Vec3>>)> {
    if !gpu_enabled() {
        return None;
    }
    if atoms.len() != positions.len() {
        return None;
    }
    let n_atoms = atoms.len();
    if n_atoms == 0 {
        return None;
    }
    if mols.iter().any(|m| !m.atom_constraints.is_empty()) {
        return None;
    }
    let mol_constraints: Vec<&[ConstraintSpec]> =
        mols.iter().map(|m| m.constraints.as_slice()).collect();
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

    let (sum, grad, _max_val, _max_violation) = ctx
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

    let mut value = 0.0f32;
    let mut max_constraint = 0.0f32;
    for i in 0..n_atoms {
        value += sum[i];
        if sum[i] > max_constraint {
            max_constraint = sum[i];
        }
    }

    let grad_out = if want_grad {
        let mut out = Vec::with_capacity(grad.len());
        for g in grad {
            out.push(Vec3::new(g.x, g.y, g.z));
        }
        Some(out)
    } else {
        None
    };

    Some((value, max_constraint, grad_out))
}

const MAX_CPU_CELLS: usize = 5_000_000;
const FORWARD_NEIGHBOR_DELTAS: [[i32; 3]; 13] = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, -1, 0],
    [1, 0, -1],
    [0, 1, -1],
    [0, 1, 1],
    [1, 1, 0],
    [1, 0, 1],
    [1, -1, -1],
    [1, -1, 1],
    [1, 1, -1],
    [1, 1, 1],
];

pub(crate) enum CpuCellList<'a> {
    Dense(CpuDenseCellList<'a>),
    Sparse(CpuSparseCellList<'a>),
}

pub(crate) struct CpuDenseCellList<'a> {
    pub(crate) dims: [usize; 3],
    pub(crate) offsets: &'a [usize],
    pub(crate) atoms: &'a [usize],
    pub(crate) active_cells: &'a [usize],
    pub(crate) periodic: bool,
}

pub(crate) struct CpuSparseCellList<'a> {
    pub(crate) dims: [usize; 3],
    pub(crate) cell_ids: &'a [usize],
    pub(crate) offsets: &'a [usize],
    pub(crate) atoms: &'a [usize],
    pub(crate) periodic: bool,
}

#[derive(Default)]
pub(crate) struct CpuCellBuffer {
    pub(crate) counts: Vec<usize>,
    pub(crate) cell_ids: Vec<usize>,
    pub(crate) atom_order: Vec<usize>,
    pub(crate) active_cells: Vec<usize>,
    pub(crate) offsets: Vec<usize>,
    pub(crate) cursor: Vec<usize>,
    pub(crate) sorted_atoms: Vec<usize>,
    pub(crate) pairs: Vec<(usize, usize)>,
}

#[derive(Default)]
pub(crate) struct ObjectiveBuffer {
    pub(crate) grad_pos: Vec<Vec3>,
    pub(crate) cpu_cells: CpuCellBuffer,
    pub(crate) atom_mol_idx: Vec<usize>,
    pub(crate) atom_movable: Vec<bool>,
    atom_meta_atoms_ptr: usize,
    atom_meta_atoms_len: usize,
    atom_meta_mols_ptr: usize,
    atom_meta_mols_len: usize,
}

#[inline]
fn wrap_axis(mut value: f32, length: f32) -> f32 {
    if length > 0.0 {
        value -= (value / length).floor() * length;
    }
    value
}

#[inline]
fn linear_cell_id(ix: usize, iy: usize, iz: usize, dims: [usize; 3]) -> usize {
    ((iz * dims[1] + iy) * dims[0]) + ix
}

#[inline]
fn decode_cell_id(cell_id: usize, dims: [usize; 3]) -> (usize, usize, usize) {
    let nx = dims[0];
    let ny = dims[1];
    let z = cell_id / (nx * ny);
    let rem = cell_id - z * nx * ny;
    let y = rem / nx;
    let x = rem - y * nx;
    (x, y, z)
}

#[inline]
fn shifted_index(base: usize, shift: i32, dim: usize, periodic: bool) -> Option<usize> {
    let base = base as i32;
    let dim_i32 = dim as i32;
    if periodic {
        Some((base + shift).rem_euclid(dim_i32) as usize)
    } else {
        let idx = base + shift;
        if idx < 0 || idx >= dim_i32 {
            None
        } else {
            Some(idx as usize)
        }
    }
}

#[inline]
fn position_to_cell_id(
    p: Vec3,
    origin: Vec3,
    extent: Vec3,
    cell_size: f32,
    dims: [usize; 3],
    periodic: bool,
) -> usize {
    let mut x = p.x - origin.x;
    let mut y = p.y - origin.y;
    let mut z = p.z - origin.z;
    if periodic {
        x = wrap_axis(x, extent.x);
        y = wrap_axis(y, extent.y);
        z = wrap_axis(z, extent.z);
    }
    let mut ix = (x / cell_size).floor() as i32;
    let mut iy = (y / cell_size).floor() as i32;
    let mut iz = (z / cell_size).floor() as i32;
    if periodic {
        ix = ix.rem_euclid(dims[0] as i32);
        iy = iy.rem_euclid(dims[1] as i32);
        iz = iz.rem_euclid(dims[2] as i32);
    } else {
        ix = ix.clamp(0, dims[0] as i32 - 1);
        iy = iy.clamp(0, dims[1] as i32 - 1);
        iz = iz.clamp(0, dims[2] as i32 - 1);
    }
    linear_cell_id(ix as usize, iy as usize, iz as usize, dims)
}

fn build_cpu_cell_list<'a>(
    positions: &[Vec3],
    atom_indices: Option<&[usize]>,
    cell_size: f32,
    pbc: Option<PbcBox>,
    buffer: &'a mut CpuCellBuffer,
) -> Option<CpuCellList<'a>> {
    let atom_count = atom_indices.map_or(positions.len(), |indices| indices.len());
    if atom_count == 0 || cell_size <= 0.0 {
        return None;
    }

    let periodic = pbc.is_some();
    let (origin, extent) = if let Some(box3) = pbc {
        (box3.min, box3.length)
    } else {
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        if let Some(indices) = atom_indices {
            for &idx in indices {
                let p = positions[idx];
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                min.z = min.z.min(p.z);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
                max.z = max.z.max(p.z);
            }
        } else {
            for p in positions {
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                min.z = min.z.min(p.z);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
                max.z = max.z.max(p.z);
            }
        }
        let mut extent = max.sub(min);
        extent.x = extent.x.max(cell_size);
        extent.y = extent.y.max(cell_size);
        extent.z = extent.z.max(cell_size);
        (min, extent)
    };

    let nx = ((extent.x / cell_size).ceil() as usize).max(1);
    let ny = ((extent.y / cell_size).ceil() as usize).max(1);
    let nz = ((extent.z / cell_size).ceil() as usize).max(1);
    let n_cells = nx.checked_mul(ny)?.checked_mul(nz)?;
    if n_cells == 0 {
        return None;
    }

    let dims = [nx, ny, nz];
    let dense_target_cells = atom_count.saturating_mul(8).max(1024);
    let use_sparse = n_cells > MAX_CPU_CELLS || n_cells > dense_target_cells;

    if use_sparse {
        buffer.pairs.clear();
        let pairs = &mut buffer.pairs;
        if let Some(indices) = atom_indices {
            for &atom_idx in indices {
                let cell_id = position_to_cell_id(
                    positions[atom_idx],
                    origin,
                    extent,
                    cell_size,
                    dims,
                    periodic,
                );
                pairs.push((cell_id, atom_idx));
            }
        } else {
            for atom_idx in 0..positions.len() {
                let cell_id = position_to_cell_id(
                    positions[atom_idx],
                    origin,
                    extent,
                    cell_size,
                    dims,
                    periodic,
                );
                pairs.push((cell_id, atom_idx));
            }
        }
        pairs.sort_unstable_by_key(|(cell_id, _)| *cell_id);

        buffer.cell_ids.clear();
        buffer.offsets.clear();
        buffer.sorted_atoms.clear();
        let mut last_cell = None;
        for (idx, &(cell_id, atom_idx)) in pairs.iter().enumerate() {
            if last_cell != Some(cell_id) {
                buffer.cell_ids.push(cell_id);
                buffer.offsets.push(idx);
                last_cell = Some(cell_id);
            }
            buffer.sorted_atoms.push(atom_idx);
        }
        buffer.offsets.push(buffer.sorted_atoms.len());

        Some(CpuCellList::Sparse(CpuSparseCellList {
            dims,
            cell_ids: &buffer.cell_ids,
            offsets: &buffer.offsets,
            atoms: &buffer.sorted_atoms,
            periodic,
        }))
    } else {
        buffer.counts.clear();
        buffer.counts.resize(n_cells, 0usize);
        buffer.cell_ids.clear();
        buffer.atom_order.clear();
        buffer.active_cells.clear();

        if let Some(indices) = atom_indices {
            for &atom_idx in indices {
                let cell_id = position_to_cell_id(
                    positions[atom_idx],
                    origin,
                    extent,
                    cell_size,
                    dims,
                    periodic,
                );
                if buffer.counts[cell_id] == 0 {
                    buffer.active_cells.push(cell_id);
                }
                buffer.counts[cell_id] += 1;
                buffer.cell_ids.push(cell_id);
                buffer.atom_order.push(atom_idx);
            }
        } else {
            for atom_idx in 0..positions.len() {
                let cell_id = position_to_cell_id(
                    positions[atom_idx],
                    origin,
                    extent,
                    cell_size,
                    dims,
                    periodic,
                );
                if buffer.counts[cell_id] == 0 {
                    buffer.active_cells.push(cell_id);
                }
                buffer.counts[cell_id] += 1;
                buffer.cell_ids.push(cell_id);
                buffer.atom_order.push(atom_idx);
            }
        }

        buffer.offsets.clear();
        buffer.offsets.resize(n_cells + 1, 0usize);
        for idx in 0..n_cells {
            buffer.offsets[idx + 1] = buffer.offsets[idx] + buffer.counts[idx];
        }
        if buffer.offsets[n_cells] != atom_count {
            return None;
        }

        if buffer.cursor.len() != buffer.offsets.len() {
            buffer.cursor.resize(buffer.offsets.len(), 0usize);
        }
        buffer.cursor.copy_from_slice(&buffer.offsets);
        buffer.sorted_atoms.clear();
        buffer.sorted_atoms.resize(atom_count, 0usize);
        for (slot, &cell_id) in buffer.cell_ids.iter().enumerate() {
            let dst = buffer.cursor[cell_id];
            buffer.sorted_atoms[dst] = buffer.atom_order[slot];
            buffer.cursor[cell_id] += 1;
        }

        Some(CpuCellList::Dense(CpuDenseCellList {
            dims,
            offsets: &buffer.offsets,
            atoms: &buffer.sorted_atoms,
            active_cells: &buffer.active_cells,
            periodic,
        }))
    }
}

#[inline]
fn sparse_cell_range(cells: &CpuSparseCellList, cell_id: usize) -> Option<(usize, usize)> {
    match cells.cell_ids.binary_search(&cell_id) {
        Ok(idx) => Some((cells.offsets[idx], cells.offsets[idx + 1])),
        Err(_) => None,
    }
}

#[inline]
fn refresh_atom_metadata(atoms: &[AtomRecord], mols: &[MolInfo], buffer: &mut ObjectiveBuffer) {
    let atoms_ptr = atoms.as_ptr() as usize;
    let mols_ptr = mols.as_ptr() as usize;
    let cache_stale = buffer.atom_meta_atoms_ptr != atoms_ptr
        || buffer.atom_meta_atoms_len != atoms.len()
        || buffer.atom_meta_mols_ptr != mols_ptr
        || buffer.atom_meta_mols_len != mols.len();
    if !cache_stale {
        return;
    }

    if buffer.atom_mol_idx.len() != atoms.len() {
        buffer.atom_mol_idx.resize(atoms.len(), usize::MAX);
    }
    if buffer.atom_movable.len() != atoms.len() {
        buffer.atom_movable.resize(atoms.len(), false);
    }
    for (idx, atom) in atoms.iter().enumerate() {
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        if mol_idx < mols.len() {
            buffer.atom_mol_idx[idx] = mol_idx;
            buffer.atom_movable[idx] = mols[mol_idx].movable;
        } else {
            buffer.atom_mol_idx[idx] = usize::MAX;
            buffer.atom_movable[idx] = false;
        }
    }
    buffer.atom_meta_atoms_ptr = atoms_ptr;
    buffer.atom_meta_atoms_len = atoms.len();
    buffer.atom_meta_mols_ptr = mols_ptr;
    buffer.atom_meta_mols_len = mols.len();
}
pub(crate) struct MolInfo {
    pub(crate) atom_indices: Vec<usize>,
    pub(crate) local_positions: Vec<Vec3>,
    pub(crate) radius: f32,
    pub(crate) selected: bool,
    pub(crate) movable: bool,
    pub(crate) rotatable: bool,
    pub(crate) constraints: Vec<ConstraintSpec>,
    pub(crate) atom_constraints: Vec<(usize, ConstraintSpec)>,
    pub(crate) rot_bounds: Option<[[f32; 2]; 3]>,
}

pub(crate) struct Bounds {
    pub(crate) lower: Vec<f32>,
    pub(crate) upper: Vec<f32>,
}

pub(crate) struct Objective {
    pub(crate) value: f32,
    pub(crate) max_overlap: f32,
    pub(crate) max_constraint: f32,
    pub(crate) grad_pos: Vec<Vec3>,
}

pub(crate) fn pack_variables(placements: &[PlacementRecord], mols: &[MolInfo]) -> Vec<f32> {
    let mut x = Vec::with_capacity(mols.len() * 6);
    for (idx, mol) in mols.iter().enumerate() {
        if !mol.selected {
            continue;
        }
        let placement = placements[idx];
        x.push(placement.center.x);
        x.push(placement.center.y);
        x.push(placement.center.z);
        x.extend_from_slice(&placement.euler);
    }
    x
}

pub(crate) fn build_bounds(
    mols: &[MolInfo],
    placements: &[PlacementRecord],
    _box_size: [f32; 3],
    _box_origin: [f32; 3],
) -> Bounds {
    let mut lower = Vec::with_capacity(mols.len() * 6);
    let mut upper = Vec::with_capacity(mols.len() * 6);
    const PACKMOL_INF: f32 = 1.0e20;
    for (idx, mol) in mols.iter().enumerate() {
        if !mol.selected {
            continue;
        }
        let placement = placements[idx];
        if mol.movable {
            lower.extend_from_slice(&[-PACKMOL_INF, -PACKMOL_INF, -PACKMOL_INF]);
            upper.extend_from_slice(&[PACKMOL_INF, PACKMOL_INF, PACKMOL_INF]);
        } else {
            lower.extend_from_slice(&[placement.center.x, placement.center.y, placement.center.z]);
            upper.extend_from_slice(&[placement.center.x, placement.center.y, placement.center.z]);
        }
        if mol.rotatable {
            if let Some(bounds) = mol.rot_bounds {
                for axis in 0..3 {
                    lower.push(bounds[axis][0]);
                    upper.push(bounds[axis][1]);
                }
            } else {
                lower.extend_from_slice(&[-PACKMOL_INF, -PACKMOL_INF, -PACKMOL_INF]);
                upper.extend_from_slice(&[PACKMOL_INF, PACKMOL_INF, PACKMOL_INF]);
            }
        } else {
            lower.extend_from_slice(&placement.euler);
            upper.extend_from_slice(&placement.euler);
        }
    }
    Bounds { lower, upper }
}

pub(crate) fn update_positions(
    x: &[f32],
    placements: &mut [PlacementRecord],
    mols: &[MolInfo],
    atoms: &mut [AtomRecord],
    positions: &mut [Vec3],
) {
    let mut offset = 0;
    for (mol_idx, mol) in mols.iter().enumerate() {
        if !mol.selected {
            continue;
        }
        if offset + 5 >= x.len() {
            break;
        }
        let center = Vec3::new(x[offset], x[offset + 1], x[offset + 2]);
        let beta = x[offset + 3];
        let gamma = x[offset + 4];
        let teta = x[offset + 5];
        placements[mol_idx].center = center;
        placements[mol_idx].euler = [beta, gamma, teta];
        let rot = rotation_matrix(beta, gamma, teta);
        for (local_idx, &atom_idx) in mol.atom_indices.iter().enumerate() {
            let local = mol.local_positions[local_idx];
            let rotated = mat_vec(rot, local);
            let pos = rotated.add(center);
            positions[atom_idx] = pos;
            atoms[atom_idx].position = pos;
        }
        offset += 6;
    }
}

pub(crate) fn compute_objective(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    atom_params: &[AtomParams],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    short_dist: f32,
    short_scale: f32,
    fbins: f32,
    radmax: f32,
    buffer: &mut ObjectiveBuffer,
) -> Objective {
    let (value, max_overlap, max_constraint) = compute_objective_with_buffer(
        atoms,
        positions,
        atom_params,
        mols,
        overlap_atoms,
        pbc,
        use_short_tol,
        include_overlap,
        short_dist,
        short_scale,
        fbins,
        radmax,
        buffer,
    );
    Objective {
        value,
        max_overlap,
        max_constraint,
        grad_pos: buffer.grad_pos.clone(),
    }
}

pub(crate) fn compute_objective_with_buffer(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    atom_params: &[AtomParams],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    short_dist: f32,
    short_scale: f32,
    fbins: f32,
    radmax: f32,
    buffer: &mut ObjectiveBuffer,
) -> (f32, f32, f32) {
    if buffer.grad_pos.len() != atoms.len() {
        buffer.grad_pos.resize(atoms.len(), Vec3::default());
    } else {
        buffer.grad_pos.fill(Vec3::default());
    }
    compute_penalty(
        atoms,
        positions,
        atom_params,
        mols,
        overlap_atoms,
        pbc,
        use_short_tol,
        include_overlap,
        short_dist,
        short_scale,
        fbins,
        radmax,
        buffer,
        true,
    )
}

pub(crate) fn compute_objective_only(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    atom_params: &[AtomParams],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    short_dist: f32,
    short_scale: f32,
    fbins: f32,
    radmax: f32,
    buffer: &mut ObjectiveBuffer,
) -> f32 {
    let (value, _, _) = compute_penalty(
        atoms,
        positions,
        atom_params,
        mols,
        overlap_atoms,
        pbc,
        use_short_tol,
        include_overlap,
        short_dist,
        short_scale,
        fbins,
        radmax,
        buffer,
        false,
    );
    value
}

pub(crate) fn compute_penalty(
    atoms: &[AtomRecord],
    positions: &[Vec3],
    atom_params: &[AtomParams],
    mols: &[MolInfo],
    overlap_atoms: Option<&[usize]>,
    pbc: Option<PbcBox>,
    use_short_tol: bool,
    include_overlap: bool,
    _short_dist: f32,
    _short_scale: f32,
    fbins: f32,
    radmax: f32,
    buffer: &mut ObjectiveBuffer,
    want_grad: bool,
) -> (f32, f32, f32) {
    let mut value = 0.0f32;
    let mut max_overlap = 0.0f32;
    let mut max_constraint = 0.0f32;
    let _selected_subset = overlap_atoms.is_some();

    #[cfg(feature = "cuda")]
    let mut used_gpu = false;
    #[cfg(not(feature = "cuda"))]
    let used_gpu = false;
    if include_overlap {
        #[cfg(feature = "cuda")]
        {
            if !_selected_subset {
                let cell_size = (2.0 * radmax * fbins).max(1.0e-6);
                if let Some((gpu_value, gpu_max, gpu_grad)) = compute_overlap_gpu(
                    atoms,
                    positions,
                    atom_params,
                    mols,
                    pbc,
                    cell_size,
                    want_grad,
                ) {
                    if use_short_tol {
                        if let Some((short_value, short_grad)) = compute_short_tol_gpu(
                            atoms,
                            positions,
                            atom_params,
                            mols,
                            pbc,
                            cell_size,
                            want_grad,
                        ) {
                            value += gpu_value + short_value;
                            max_overlap = max_overlap.max(gpu_max);
                            if want_grad {
                                if let Some(gpu_grad) = gpu_grad {
                                    buffer.grad_pos.clone_from(&gpu_grad);
                                }
                                if let Some(short_grad) = short_grad {
                                    for (idx, g) in short_grad.iter().enumerate() {
                                        buffer.grad_pos[idx] = buffer.grad_pos[idx].add(*g);
                                    }
                                }
                            }
                            used_gpu = true;
                        }
                    } else {
                        value += gpu_value;
                        max_overlap = max_overlap.max(gpu_max);
                        if want_grad {
                            if let Some(gpu_grad) = gpu_grad {
                                buffer.grad_pos.clone_from(&gpu_grad);
                            }
                        }
                        used_gpu = true;
                    }
                }
            }
        }
    }

    if include_overlap && !used_gpu {
        let cell_size = (2.0 * radmax * fbins).max(1.0e-6);
        let overlap_count = overlap_atoms.map_or(positions.len(), |indices| indices.len());

        if overlap_count > 1 {
            refresh_atom_metadata(atoms, mols, buffer);
            let atom_mol_idx = &buffer.atom_mol_idx;
            let atom_movable = &buffer.atom_movable;

            macro_rules! eval_pair {
                ($i:expr, $j:expr) => {{
                    let i = $i;
                    let j = $j;
                    let mol_i = atom_mol_idx[i];
                    let mol_j = atom_mol_idx[j];
                    if mol_i != mol_j
                        && mol_i != usize::MAX
                        && mol_j != usize::MAX
                        && (atom_movable[i] || atom_movable[j])
                    {
                        let delta = if let Some(pbc_box) = pbc {
                            pbc_box.delta(positions[i], positions[j])
                        } else {
                            positions[i].sub(positions[j])
                        };
                        let d2 = delta.dot(delta);
                        let pi = atom_params[i];
                        let pj = atom_params[j];
                        let tol = pi.radius + pj.radius;
                        let tol2 = tol * tol;
                        if d2 < tol2 {
                            let diff = d2 - tol2;
                            let weight = pi.fscale * pj.fscale;
                            value += weight * diff * diff;
                            let overlap = tol - d2.sqrt();
                            if overlap > max_overlap {
                                max_overlap = overlap;
                            }
                            if want_grad {
                                let coeff = weight * 4.0 * diff;
                                let g = delta.scale(coeff);
                                buffer.grad_pos[i] = buffer.grad_pos[i].add(g);
                                buffer.grad_pos[j] = buffer.grad_pos[j].sub(g);
                            }
                        } else if use_short_tol && (pi.use_short || pj.use_short) {
                            let short_r = pi.short_radius + pj.short_radius;
                            let short2 = short_r * short_r;
                            if d2 < short2 {
                                let diff2 = d2 - short2;
                                let scale = (pi.short_scale * pj.short_scale).sqrt();
                                let weight = pi.fscale * pj.fscale * scale;
                                value += weight * diff2 * diff2;
                                if want_grad {
                                    let coeff = weight * 4.0 * diff2;
                                    let g = delta.scale(coeff);
                                    buffer.grad_pos[i] = buffer.grad_pos[i].add(g);
                                    buffer.grad_pos[j] = buffer.grad_pos[j].sub(g);
                                }
                            }
                        }
                    }
                }};
            }

            if let Some(cells) = build_cpu_cell_list(
                positions,
                overlap_atoms,
                cell_size,
                pbc,
                &mut buffer.cpu_cells,
            ) {
                match cells {
                    CpuCellList::Dense(cells) => {
                        let dims = cells.dims;
                        let need_neighbor_dedupe =
                            cells.periodic && (dims[0] <= 2 || dims[1] <= 2 || dims[2] <= 2);
                        for &cell_id in cells.active_cells {
                            let start = cells.offsets[cell_id];
                            let end = cells.offsets[cell_id + 1];
                            if start >= end {
                                continue;
                            }

                            for a in start..end {
                                let i = cells.atoms[a];
                                for b in (a + 1)..end {
                                    let j = cells.atoms[b];
                                    eval_pair!(i, j);
                                }
                            }

                            let (cx, cy, cz) = decode_cell_id(cell_id, dims);
                            let mut seen_neighbors = [usize::MAX; FORWARD_NEIGHBOR_DELTAS.len()];
                            let mut seen_count = 0usize;

                            for delta in FORWARD_NEIGHBOR_DELTAS {
                                let nx = match shifted_index(cx, delta[0], dims[0], cells.periodic)
                                {
                                    Some(v) => v,
                                    None => continue,
                                };
                                let ny = match shifted_index(cy, delta[1], dims[1], cells.periodic)
                                {
                                    Some(v) => v,
                                    None => continue,
                                };
                                let nz = match shifted_index(cz, delta[2], dims[2], cells.periodic)
                                {
                                    Some(v) => v,
                                    None => continue,
                                };
                                let neighbor_id = linear_cell_id(nx, ny, nz, dims);
                                if neighbor_id == cell_id {
                                    continue;
                                }
                                if need_neighbor_dedupe {
                                    let mut duplicate = false;
                                    for &seen in &seen_neighbors[..seen_count] {
                                        if seen == neighbor_id {
                                            duplicate = true;
                                            break;
                                        }
                                    }
                                    if duplicate {
                                        continue;
                                    }
                                    seen_neighbors[seen_count] = neighbor_id;
                                    seen_count += 1;
                                }

                                let nstart = cells.offsets[neighbor_id];
                                let nend = cells.offsets[neighbor_id + 1];
                                if nstart >= nend {
                                    continue;
                                }
                                for a in start..end {
                                    let i = cells.atoms[a];
                                    for b in nstart..nend {
                                        let j = cells.atoms[b];
                                        eval_pair!(i, j);
                                    }
                                }
                            }
                        }
                    }
                    CpuCellList::Sparse(cells) => {
                        let dims = cells.dims;
                        let need_neighbor_dedupe =
                            cells.periodic && (dims[0] <= 2 || dims[1] <= 2 || dims[2] <= 2);
                        for (cell_idx, &cell_id) in cells.cell_ids.iter().enumerate() {
                            let start = cells.offsets[cell_idx];
                            let end = cells.offsets[cell_idx + 1];
                            if start >= end {
                                continue;
                            }

                            for a in start..end {
                                let i = cells.atoms[a];
                                for b in (a + 1)..end {
                                    let j = cells.atoms[b];
                                    eval_pair!(i, j);
                                }
                            }

                            let (cx, cy, cz) = decode_cell_id(cell_id, dims);
                            let mut seen_neighbors = [usize::MAX; FORWARD_NEIGHBOR_DELTAS.len()];
                            let mut seen_count = 0usize;

                            for delta in FORWARD_NEIGHBOR_DELTAS {
                                let nx = match shifted_index(cx, delta[0], dims[0], cells.periodic)
                                {
                                    Some(v) => v,
                                    None => continue,
                                };
                                let ny = match shifted_index(cy, delta[1], dims[1], cells.periodic)
                                {
                                    Some(v) => v,
                                    None => continue,
                                };
                                let nz = match shifted_index(cz, delta[2], dims[2], cells.periodic)
                                {
                                    Some(v) => v,
                                    None => continue,
                                };
                                let neighbor_id = linear_cell_id(nx, ny, nz, dims);
                                if neighbor_id == cell_id {
                                    continue;
                                }
                                if need_neighbor_dedupe {
                                    let mut duplicate = false;
                                    for &seen in &seen_neighbors[..seen_count] {
                                        if seen == neighbor_id {
                                            duplicate = true;
                                            break;
                                        }
                                    }
                                    if duplicate {
                                        continue;
                                    }
                                    seen_neighbors[seen_count] = neighbor_id;
                                    seen_count += 1;
                                }

                                let (nstart, nend) = match sparse_cell_range(&cells, neighbor_id) {
                                    Some(range) => range,
                                    None => continue,
                                };
                                if nstart >= nend {
                                    continue;
                                }
                                for a in start..end {
                                    let i = cells.atoms[a];
                                    for b in nstart..nend {
                                        let j = cells.atoms[b];
                                        eval_pair!(i, j);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                if let Some(indices) = overlap_atoms {
                    for a in 0..indices.len() {
                        let i = indices[a];
                        for b in (a + 1)..indices.len() {
                            let j = indices[b];
                            eval_pair!(i, j);
                        }
                    }
                } else {
                    for i in 0..positions.len() {
                        for j in (i + 1)..positions.len() {
                            eval_pair!(i, j);
                        }
                    }
                }
            }
        }
    }

    #[cfg(feature = "cuda")]
    let mut used_constraints_gpu = false;
    #[cfg(not(feature = "cuda"))]
    let used_constraints_gpu = false;
    #[cfg(feature = "cuda")]
    {
        if !_selected_subset {
            if let Some((c_value, c_max, c_grad)) =
                compute_constraints_gpu(atoms, positions, mols, want_grad)
            {
                value += c_value;
                max_constraint = max_constraint.max(c_max);
                if want_grad {
                    if let Some(c_grad) = c_grad {
                        for (idx, g) in c_grad.iter().enumerate() {
                            buffer.grad_pos[idx] = buffer.grad_pos[idx].add(*g);
                        }
                    }
                }
                used_constraints_gpu = true;
            }
        }
    }

    if !used_constraints_gpu {
        for mol in mols {
            if !mol.selected {
                continue;
            }
            for (local_idx, &atom_idx) in mol.atom_indices.iter().enumerate() {
                let pos = positions[atom_idx];
                let mut atom_penalty_sum = 0.0f32;
                for constraint in &mol.constraints {
                    let pen = penalty_and_grad(pos, constraint);
                    if pen.value > 0.0 {
                        value += pen.value;
                        atom_penalty_sum += pen.value;
                        if want_grad {
                            buffer.grad_pos[atom_idx] = buffer.grad_pos[atom_idx].add(pen.grad);
                        }
                    }
                }
                for (constraint_local_idx, constraint) in mol.atom_constraints.iter() {
                    if *constraint_local_idx != local_idx {
                        continue;
                    }
                    let pen = penalty_and_grad(pos, constraint);
                    if pen.value > 0.0 {
                        value += pen.value;
                        atom_penalty_sum += pen.value;
                        if want_grad {
                            buffer.grad_pos[atom_idx] = buffer.grad_pos[atom_idx].add(pen.grad);
                        }
                    }
                }
                max_constraint = max_constraint.max(atom_penalty_sum);
            }
        }
    }

    (value, max_overlap, max_constraint)
}

pub(crate) fn compute_gradient(objective: &Objective, x: &[f32], mols: &[MolInfo]) -> Vec<f32> {
    compute_gradient_from_grad_pos(&objective.grad_pos, x, mols)
}

pub(crate) fn compute_gradient_from_grad_pos(
    grad_pos: &[Vec3],
    x: &[f32],
    mols: &[MolInfo],
) -> Vec<f32> {
    let mut grad_x = Vec::new();
    compute_gradient_from_grad_pos_into(grad_pos, x, mols, &mut grad_x);
    grad_x
}

pub(crate) fn compute_gradient_from_grad_pos_into(
    grad_pos: &[Vec3],
    x: &[f32],
    mols: &[MolInfo],
    grad_x: &mut Vec<f32>,
) {
    if grad_x.len() != x.len() {
        grad_x.resize(x.len(), 0.0);
    } else {
        grad_x.fill(0.0);
    }
    let mut offset = 0usize;
    for mol in mols.iter() {
        if !mol.selected {
            continue;
        }
        if offset + 5 >= x.len() {
            break;
        }
        let beta = x[offset + 3];
        let gamma = x[offset + 4];
        let teta = x[offset + 5];
        let (_mat, db, dg, dt) = rotation_with_derivatives(beta, gamma, teta);
        let mut grad_center = Vec3::default();
        let mut grad_beta = 0.0f32;
        let mut grad_gamma = 0.0f32;
        let mut grad_teta = 0.0f32;
        for (local_idx, &atom_idx) in mol.atom_indices.iter().enumerate() {
            let gpos = grad_pos[atom_idx];
            grad_center = grad_center.add(gpos);
            if mol.rotatable {
                let local = mol.local_positions[local_idx];
                let dpos_db = mat_vec(db, local);
                let dpos_dg = mat_vec(dg, local);
                let dpos_dt = mat_vec(dt, local);
                grad_beta += gpos.dot(dpos_db);
                grad_gamma += gpos.dot(dpos_dg);
                grad_teta += gpos.dot(dpos_dt);
            }
        }
        if mol.movable {
            grad_x[offset] = grad_center.x;
            grad_x[offset + 1] = grad_center.y;
            grad_x[offset + 2] = grad_center.z;
        }
        if mol.rotatable {
            grad_x[offset + 3] = grad_beta;
            grad_x[offset + 4] = grad_gamma;
            grad_x[offset + 5] = grad_teta;
        }
        offset += 6;
    }
}

pub(crate) fn grad_norm(grad: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for g in grad {
        sum += g * g;
    }
    sum.sqrt()
}

pub(crate) fn grad_sup_norm(grad: &[f32]) -> f32 {
    let mut max_val = 0.0f32;
    for g in grad {
        let v = g.abs();
        if v > max_val {
            max_val = v;
        }
    }
    max_val
}

pub(crate) fn project_step(x: &[f32], direction: &[f32], step: f32, bounds: &Bounds) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    let n = x
        .len()
        .min(direction.len())
        .min(bounds.lower.len())
        .min(bounds.upper.len());
    for i in 0..n {
        let mut v = x[i] + step * direction[i];
        let lo = bounds.lower[i];
        let hi = bounds.upper[i];
        if v < lo {
            v = lo;
        }
        if v > hi {
            v = hi;
        }
        out.push(v);
    }
    if n < x.len() {
        out.extend_from_slice(&x[n..]);
    }
    out
}

pub(crate) fn projected_gradient(x: &[f32], grad: &[f32], bounds: &Bounds) -> Vec<f32> {
    let mut pg = Vec::with_capacity(x.len());
    let n = x
        .len()
        .min(grad.len())
        .min(bounds.lower.len())
        .min(bounds.upper.len());
    for i in 0..n {
        let mut v = x[i] - grad[i];
        let lo = bounds.lower[i];
        let hi = bounds.upper[i];
        if v < lo {
            v = lo;
        }
        if v > hi {
            v = hi;
        }
        pg.push(x[i] - v);
    }
    if n < x.len() {
        pg.resize(x.len(), 0.0);
    }
    pg
}

pub(crate) fn validate_molecule_counts(
    atom_indices: &[usize],
    template_len: usize,
) -> PackResult<()> {
    if atom_indices.len() != template_len {
        return Err(PackError::Placement(
            "molecule atom count does not match template".into(),
        ));
    }
    Ok(())
}
