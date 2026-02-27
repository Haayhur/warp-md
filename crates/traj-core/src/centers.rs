use crate::frame::FrameChunk;

pub fn center_of_selection(
    chunk: &FrameChunk,
    frame: usize,
    indices: &[u32],
    masses: &[f32],
    mass_weighted: bool,
) -> [f64; 3] {
    let n_atoms = chunk.n_atoms;
    let mut sum = [0.0f64; 3];
    let mut mass_sum = 0.0f64;
    for &idx in indices.iter() {
        let atom_idx = idx as usize;
        let p = chunk.coords[frame * n_atoms + atom_idx];
        let m = if mass_weighted {
            masses[atom_idx] as f64
        } else {
            1.0
        };
        sum[0] += p[0] as f64 * m;
        sum[1] += p[1] as f64 * m;
        sum[2] += p[2] as f64 * m;
        mass_sum += m;
    }
    if mass_sum == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [sum[0] / mass_sum, sum[1] / mass_sum, sum[2] / mass_sum]
}

pub fn center_of_coords(
    coords: &[[f32; 4]],
    indices: &[u32],
    masses: &[f32],
    mass_weighted: bool,
) -> [f64; 3] {
    let mut sum = [0.0f64; 3];
    let mut mass_sum = 0.0f64;
    for &idx in indices.iter() {
        let atom_idx = idx as usize;
        let p = coords[atom_idx];
        let m = if mass_weighted {
            masses[atom_idx] as f64
        } else {
            1.0
        };
        sum[0] += p[0] as f64 * m;
        sum[1] += p[1] as f64 * m;
        sum[2] += p[2] as f64 * m;
        mass_sum += m;
    }
    if mass_sum == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    [sum[0] / mass_sum, sum[1] / mass_sum, sum[2] / mass_sum]
}
