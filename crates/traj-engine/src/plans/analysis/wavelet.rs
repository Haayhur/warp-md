use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::PbcMode;

pub struct WaveletPlan {
    sel_a: Selection,
    sel_b: Selection,
    mass_weighted: bool,
    pbc: PbcMode,
    series: Vec<f32>,
}

impl WaveletPlan {
    pub fn new(sel_a: Selection, sel_b: Selection, mass_weighted: bool, pbc: PbcMode) -> Self {
        Self {
            sel_a,
            sel_b,
            mass_weighted,
            pbc,
            series: Vec::new(),
        }
    }
}

impl Plan for WaveletPlan {
    fn name(&self) -> &'static str {
        "wavelet"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.series.clear();
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let _ = chunk.n_atoms;
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let a = center_of_selection(
                chunk,
                frame,
                &self.sel_a.indices,
                masses,
                self.mass_weighted,
            );
            let b = center_of_selection(
                chunk,
                frame,
                &self.sel_b.indices,
                masses,
                self.mass_weighted,
            );
            let mut dx = b[0] - a[0];
            let mut dy = b[1] - a[1];
            let mut dz = b[2] - a[2];
            if matches!(self.pbc, PbcMode::Orthorhombic) {
                let (lx, ly, lz) = box_lengths(chunk, frame)?;
                apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
            }
            let dist = (dx * dx + dy * dy + dz * dz).sqrt() as f32;
            self.series.push(dist);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.series.len() < 2 {
            return Ok(PlanOutput::Matrix {
                data: Vec::new(),
                rows: 0,
                cols: 0,
            });
        }
        let mut current = self.series.clone();
        let max_cols = current.len() / 2;
        let mut details: Vec<Vec<f32>> = Vec::new();
        while current.len() >= 2 {
            let n = current.len() / 2;
            let mut next = Vec::with_capacity(n);
            let mut detail = Vec::with_capacity(n);
            for i in 0..n {
                let a = current[2 * i];
                let b = current[2 * i + 1];
                next.push(0.5 * (a + b));
                detail.push(0.5 * (a - b));
            }
            details.push(detail);
            current = next;
        }
        let rows = details.len();
        let cols = max_cols;
        let mut data = vec![0.0f32; rows * cols];
        for (r, detail) in details.iter().enumerate() {
            for (c, &val) in detail.iter().enumerate() {
                data[r * cols + c] = val;
            }
        }
        Ok(PlanOutput::Matrix { data, rows, cols })
    }
}

fn center_of_selection(
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

fn box_lengths(chunk: &FrameChunk, frame: usize) -> TrajResult<(f64, f64, f64)> {
    match chunk.box_[frame] {
        Box3::Orthorhombic { lx, ly, lz } => Ok((lx as f64, ly as f64, lz as f64)),
        _ => Err(TrajError::Mismatch(
            "orthorhombic box required for PBC".into(),
        )),
    }
}

fn apply_pbc(dx: &mut f64, dy: &mut f64, dz: &mut f64, lx: f64, ly: f64, lz: f64) {
    if lx > 0.0 {
        *dx -= (*dx / lx).round() * lx;
    }
    if ly > 0.0 {
        *dy -= (*dy / ly).round() * ly;
    }
    if lz > 0.0 {
        *dz -= (*dz / lz).round() * lz;
    }
}
