use std::collections::HashSet;

use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

const CODE_C: u8 = 0;
const CODE_H: u8 = 1;
const CODE_E: u8 = 2;

const HELIX_PHI: (f64, f64) = (-100.0, -30.0);
const HELIX_PSI: (f64, f64) = (-80.0, -10.0);
const SHEET_PHI: (f64, f64) = (-180.0, -100.0);
const SHEET_PSI: (f64, f64) = (90.0, 180.0);

#[derive(Clone, Debug)]
struct ResidueBackbone {
    n_idx: Option<usize>,
    ca_idx: Option<usize>,
    c_idx: Option<usize>,
}

pub struct DsspPlan {
    selection: Selection,
    residues: Vec<ResidueBackbone>,
    labels: Vec<String>,
    codes: Vec<u8>,
    frames: usize,
}

impl DsspPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            residues: Vec::new(),
            labels: Vec::new(),
            codes: Vec::new(),
            frames: 0,
        }
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    fn build_residues(&mut self, system: &System) {
        self.residues.clear();
        self.labels.clear();
        let n_atoms = system.n_atoms();
        if n_atoms == 0 {
            return;
        }
        let selected: HashSet<usize> = self
            .selection
            .indices
            .iter()
            .map(|&idx| idx as usize)
            .collect();
        let use_all = selected.is_empty();
        let atoms = &system.atoms;

        let mut start = 0usize;
        while start < n_atoms {
            let chain = atoms.chain_id[start];
            let resid = atoms.resid[start];
            let resname_id = atoms.resname_id[start];
            let mut end = start + 1;
            while end < n_atoms
                && atoms.chain_id[end] == chain
                && atoms.resid[end] == resid
                && atoms.resname_id[end] == resname_id
            {
                end += 1;
            }

            let mut include = use_all;
            if !include {
                for idx in start..end {
                    if selected.contains(&idx) {
                        include = true;
                        break;
                    }
                }
            }
            if include {
                let mut n_idx = None;
                let mut ca_idx = None;
                let mut c_idx = None;
                for idx in start..end {
                    let atom_name = system.interner.resolve(atoms.name_id[idx]).unwrap_or("");
                    if atom_name.eq_ignore_ascii_case("N") {
                        n_idx = Some(idx);
                    } else if atom_name.eq_ignore_ascii_case("CA") {
                        ca_idx = Some(idx);
                    } else if atom_name.eq_ignore_ascii_case("C") {
                        c_idx = Some(idx);
                    }
                }
                let resname = system
                    .interner
                    .resolve(resname_id)
                    .unwrap_or("RES")
                    .to_string();
                self.residues.push(ResidueBackbone {
                    n_idx,
                    ca_idx,
                    c_idx,
                });
                self.labels.push(format!("{resname}:{resid}"));
            }
            start = end;
        }
    }
}

impl Plan for DsspPlan {
    fn name(&self) -> &'static str {
        "dssp"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.codes.clear();
        self.frames = 0;
        self.build_residues(system);
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_res = self.residues.len();
        if n_res == 0 {
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            for i in 0..n_res {
                let (phi, psi) = phi_psi(&self.residues, chunk, frame, i);
                self.codes.push(classify(phi, psi));
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_res = self.residues.len();
        if self.frames == 0 || n_res == 0 {
            return Ok(PlanOutput::Matrix {
                data: Vec::new(),
                rows: 0,
                cols: n_res,
            });
        }
        let data: Vec<f32> = self.codes.iter().map(|&v| v as f32).collect();
        Ok(PlanOutput::Matrix {
            data,
            rows: self.frames,
            cols: n_res,
        })
    }
}

fn classify(phi: Option<f64>, psi: Option<f64>) -> u8 {
    let (Some(phi), Some(psi)) = (phi, psi) else {
        return CODE_C;
    };
    if HELIX_PHI.0 <= phi && phi <= HELIX_PHI.1 && HELIX_PSI.0 <= psi && psi <= HELIX_PSI.1 {
        return CODE_H;
    }
    if SHEET_PHI.0 <= phi && phi <= SHEET_PHI.1 && SHEET_PSI.0 <= psi && psi <= SHEET_PSI.1 {
        return CODE_E;
    }
    CODE_C
}

fn phi_psi(
    residues: &[ResidueBackbone],
    chunk: &FrameChunk,
    frame: usize,
    idx: usize,
) -> (Option<f64>, Option<f64>) {
    let res = &residues[idx];
    let (Some(n_idx), Some(ca_idx), Some(c_idx)) = (res.n_idx, res.ca_idx, res.c_idx) else {
        return (None, None);
    };
    let mut phi = None;
    let mut psi = None;
    if idx > 0 {
        if let Some(prev_c_idx) = residues[idx - 1].c_idx {
            phi = dihedral(
                chunk,
                frame,
                prev_c_idx as u32,
                n_idx as u32,
                ca_idx as u32,
                c_idx as u32,
            );
        }
    }
    if idx + 1 < residues.len() {
        if let Some(next_n_idx) = residues[idx + 1].n_idx {
            psi = dihedral(
                chunk,
                frame,
                n_idx as u32,
                ca_idx as u32,
                c_idx as u32,
                next_n_idx as u32,
            );
        }
    }
    (phi, psi)
}

fn dihedral(chunk: &FrameChunk, frame: usize, a: u32, b: u32, c: u32, d: u32) -> Option<f64> {
    let n_atoms = chunk.n_atoms;
    let base = frame * n_atoms;
    let p0 = point(chunk, base + a as usize);
    let p1 = point(chunk, base + b as usize);
    let p2 = point(chunk, base + c as usize);
    let p3 = point(chunk, base + d as usize);

    let b0 = sub(p1, p0);
    let b1 = sub(p2, p1);
    let b2 = sub(p3, p2);

    let norm_b1 = norm(b1);
    if norm_b1 == 0.0 {
        return None;
    }
    let b1n = mul(b1, 1.0 / norm_b1);

    let v = sub(b0, mul(b1n, dot(b0, b1n)));
    let w = sub(b2, mul(b1n, dot(b2, b1n)));

    let norm_v = norm(v);
    let norm_w = norm(w);
    if norm_v == 0.0 || norm_w == 0.0 {
        return None;
    }
    let vn = mul(v, 1.0 / norm_v);
    let wn = mul(w, 1.0 / norm_w);

    let x = dot(vn, wn);
    let y = dot(cross(b1n, vn), wn);
    Some(y.atan2(x).to_degrees())
}

#[inline]
fn point(chunk: &FrameChunk, idx: usize) -> [f64; 3] {
    let p = chunk.coords[idx];
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn mul(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
