use std::collections::HashSet;

use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

pub struct DridPlan {
    selection: Selection,
    exclude_bonds: bool,
    partners: Vec<Vec<usize>>,
    results: Vec<f32>,
    frames: usize,
}

impl DridPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            exclude_bonds: true,
            partners: Vec::new(),
            results: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_exclude_bonds(mut self, exclude_bonds: bool) -> Self {
        self.exclude_bonds = exclude_bonds;
        self
    }

    fn cols(&self) -> usize {
        self.selection.indices.len() * 3
    }
}

impl Plan for DridPlan {
    fn name(&self) -> &'static str {
        "drid"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        self.partners = build_partners(system, &self.selection, self.exclude_bonds);
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        Some(&self.selection.indices)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = self.selection.indices.clone();
        for frame in 0..chunk.n_frames {
            let frame_base = frame * n_atoms;
            self.push_frame(|local| chunk.coords[frame_base + sel[local] as usize]);
            self.frames += 1;
        }
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        for frame in 0..chunk.n_frames {
            let frame_base = frame * chunk.n_atoms;
            self.push_frame(|local| chunk.coords[frame_base + local]);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.cols(),
        })
    }
}

impl DridPlan {
    fn push_frame<P>(&mut self, point: P)
    where
        P: Fn(usize) -> [f32; 4],
    {
        for atom in 0..self.partners.len() {
            let moments = drid_moments(&point, atom, &self.partners[atom]);
            self.results.extend_from_slice(&moments);
        }
    }
}

fn build_partners(system: &System, selection: &Selection, exclude_bonds: bool) -> Vec<Vec<usize>> {
    let n_selected = selection.indices.len();
    let mut local_by_global = vec![usize::MAX; system.n_atoms()];
    for (local, &global) in selection.indices.iter().enumerate() {
        let global = global as usize;
        if global < local_by_global.len() {
            local_by_global[global] = local;
        }
    }

    let mut bonded = HashSet::new();
    if exclude_bonds {
        for &(a, b) in system.bonds.iter() {
            let a = a as usize;
            let b = b as usize;
            if a >= local_by_global.len() || b >= local_by_global.len() {
                continue;
            }
            let local_a = local_by_global[a];
            let local_b = local_by_global[b];
            if local_a == usize::MAX || local_b == usize::MAX {
                continue;
            }
            let pair = if local_a < local_b {
                (local_a, local_b)
            } else {
                (local_b, local_a)
            };
            bonded.insert(pair);
        }
    }

    let mut partners = Vec::with_capacity(n_selected);
    for atom in 0..n_selected {
        let mut row = Vec::with_capacity(n_selected.saturating_sub(1));
        for other in 0..n_selected {
            if atom == other {
                continue;
            }
            let pair = if atom < other {
                (atom, other)
            } else {
                (other, atom)
            };
            if bonded.contains(&pair) {
                continue;
            }
            row.push(other);
        }
        partners.push(row);
    }
    partners
}

fn drid_moments<P>(point: &P, atom: usize, partners: &[usize]) -> [f32; 3]
where
    P: Fn(usize) -> [f32; 4],
{
    if partners.is_empty() {
        return [0.0; 3];
    }
    let mut count = 0usize;
    let mut sum = 0.0f64;
    for &partner in partners {
        if let Some(value) = reciprocal_distance(point, atom, partner) {
            count += 1;
            sum += value;
        }
    }
    if count == 0 {
        return [0.0; 3];
    }
    let n = count as f64;
    let mean = sum / n;
    let mut second = 0.0;
    let mut third = 0.0;
    for &partner in partners {
        if let Some(value) = reciprocal_distance(point, atom, partner) {
            let centered = value - mean;
            second += centered * centered;
            third += centered * centered * centered;
        }
    }
    second /= n;
    third /= n;
    let third_root = if third.abs() < 1.0e-15 {
        0.0
    } else {
        third.cbrt()
    };
    [mean as f32, second.sqrt() as f32, third_root as f32]
}

fn reciprocal_distance<P>(point: &P, atom: usize, partner: usize) -> Option<f64>
where
    P: Fn(usize) -> [f32; 4],
{
    let origin = point(atom);
    let other = point(partner);
    let dx = origin[0] as f64 - other[0] as f64;
    let dy = origin[1] as f64 - other[1] as f64;
    let dz = origin[2] as f64 - other[2] as f64;
    let d2 = dx * dx + dy * dy + dz * dz;
    (d2 > f64::EPSILON).then(|| 1.0 / d2.sqrt())
}
