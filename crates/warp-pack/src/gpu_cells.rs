#[cfg(feature = "cuda")]
use crate::geom::Vec3;
#[cfg(feature = "cuda")]
use crate::pbc::PbcBox;

#[cfg(feature = "cuda")]
pub(crate) const MAX_GPU_CELLS: usize = 5_000_000;

#[cfg(feature = "cuda")]
pub(crate) struct GpuCellList {
    pub(crate) origin: Vec3,
    pub(crate) cell_size: f32,
    pub(crate) dims: [i32; 3],
    pub(crate) offsets: Vec<i32>,
    pub(crate) atoms: Vec<i32>,
}

#[cfg(feature = "cuda")]
pub(crate) fn build_gpu_cell_list(
    positions: &[Vec3],
    cell_size: f32,
    pbc: Option<PbcBox>,
) -> Option<GpuCellList> {
    if positions.is_empty() || cell_size <= 0.0 {
        return None;
    }
    if positions.len() > i32::MAX as usize {
        return None;
    }
    let (origin, extent) = if let Some(b) = pbc {
        (b.min, b.length)
    } else {
        let mut min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for p in positions {
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }
        let mut extent = max.sub(min);
        extent.x = extent.x.max(cell_size);
        extent.y = extent.y.max(cell_size);
        extent.z = extent.z.max(cell_size);
        (min, extent)
    };

    let nx = ((extent.x / cell_size).ceil() as i32).max(1);
    let ny = ((extent.y / cell_size).ceil() as i32).max(1);
    let nz = ((extent.z / cell_size).ceil() as i32).max(1);
    let n_cells = (nx as i64) * (ny as i64) * (nz as i64);
    if n_cells <= 0 || n_cells > i32::MAX as i64 {
        return None;
    }
    let n_cells = n_cells as usize;
    if n_cells > MAX_GPU_CELLS {
        return None;
    }

    let mut counts = vec![0i32; n_cells];
    let mut cell_ids = Vec::with_capacity(positions.len());

    for p in positions {
        let mut x = p.x - origin.x;
        let mut y = p.y - origin.y;
        let mut z = p.z - origin.z;
        if let Some(b) = pbc {
            let lx = b.length.x;
            let ly = b.length.y;
            let lz = b.length.z;
            if lx > 0.0 {
                x -= (x / lx).floor() * lx;
            }
            if ly > 0.0 {
                y -= (y / ly).floor() * ly;
            }
            if lz > 0.0 {
                z -= (z / lz).floor() * lz;
            }
        }
        let mut ix = (x / cell_size).floor() as i32;
        let mut iy = (y / cell_size).floor() as i32;
        let mut iz = (z / cell_size).floor() as i32;
        if pbc.is_none() {
            ix = ix.clamp(0, nx - 1);
            iy = iy.clamp(0, ny - 1);
            iz = iz.clamp(0, nz - 1);
        } else {
            ix = ((ix % nx) + nx) % nx;
            iy = ((iy % ny) + ny) % ny;
            iz = ((iz % nz) + nz) % nz;
        }
        let cell_id = ((iz * ny + iy) * nx + ix) as usize;
        counts[cell_id] += 1;
        cell_ids.push(cell_id);
    }

    let mut offsets = vec![0i32; n_cells + 1];
    for i in 0..n_cells {
        offsets[i + 1] = offsets[i] + counts[i];
    }
    if offsets[n_cells] as usize != positions.len() {
        return None;
    }
    let mut cursor = offsets.clone();
    let mut atoms = vec![0i32; positions.len()];
    for (idx, &cell_id) in cell_ids.iter().enumerate() {
        let pos = cursor[cell_id] as usize;
        atoms[pos] = idx as i32;
        cursor[cell_id] += 1;
    }

    Some(GpuCellList {
        origin,
        cell_size,
        dims: [nx, ny, nz],
        offsets,
        atoms,
    })
}
