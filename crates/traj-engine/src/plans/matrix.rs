use nalgebra::{DMatrix, SymmetricEigen};

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, PcaOutput, Plan, PlanOutput};
use crate::plans::PbcMode;

#[derive(Debug, Clone, Copy)]
pub enum MatrixMode {
    Distance,
    Covariance,
    MwCovariance,
}

pub struct MatrixPlan {
    selection: Selection,
    mode: MatrixMode,
    pbc: PbcMode,
    sum: Vec<f64>,
    sum_outer: Vec<f64>,
    sum_dist: Vec<f64>,
    frames: usize,
}

impl MatrixPlan {
    pub fn new(selection: Selection, mode: MatrixMode, pbc: PbcMode) -> Self {
        Self {
            selection,
            mode,
            pbc,
            sum: Vec::new(),
            sum_outer: Vec::new(),
            sum_dist: Vec::new(),
            frames: 0,
        }
    }
}

impl Plan for MatrixPlan {
    fn name(&self) -> &'static str {
        "matrix"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        let n_sel = self.selection.indices.len();
        let n_features = n_sel * 3;
        self.frames = 0;
        self.sum.clear();
        self.sum_outer.clear();
        self.sum_dist.clear();
        match self.mode {
            MatrixMode::Distance => {
                self.sum_dist.resize(n_sel * n_sel, 0.0);
            }
            MatrixMode::Covariance | MatrixMode::MwCovariance => {
                self.sum.resize(n_features, 0.0);
                self.sum_outer.resize(n_features * n_features, 0.0);
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        match self.mode {
            MatrixMode::Distance => {
                let n_sel = sel.len();
                if n_sel == 0 {
                    self.frames += chunk.n_frames;
                    return Ok(());
                }
                for frame in 0..chunk.n_frames {
                    let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                        box_lengths(chunk, frame)?
                    } else {
                        (0.0, 0.0, 0.0)
                    };
                    for i in 0..n_sel {
                        let a_idx = sel[i] as usize;
                        let pa = chunk.coords[frame * n_atoms + a_idx];
                        let base_i = i * n_sel;
                        for j in i..n_sel {
                            let b_idx = sel[j] as usize;
                            let pb = chunk.coords[frame * n_atoms + b_idx];
                            let mut dx = (pb[0] - pa[0]) as f64;
                            let mut dy = (pb[1] - pa[1]) as f64;
                            let mut dz = (pb[2] - pa[2]) as f64;
                            if matches!(self.pbc, PbcMode::Orthorhombic) {
                                apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
                            }
                            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                            self.sum_dist[base_i + j] += dist;
                            if i != j {
                                self.sum_dist[j * n_sel + i] += dist;
                            }
                        }
                    }
                    self.frames += 1;
                }
            }
            MatrixMode::Covariance | MatrixMode::MwCovariance => {
                let n_sel = sel.len();
                if n_sel == 0 {
                    self.frames += chunk.n_frames;
                    return Ok(());
                }
                let n_features = n_sel * 3;
                let masses = &system.atoms.mass;
                let use_mass = matches!(self.mode, MatrixMode::MwCovariance);
                let mut frame_vec = vec![0.0f64; n_features];
                for frame in 0..chunk.n_frames {
                    for (i, &idx) in sel.iter().enumerate() {
                        let atom_idx = idx as usize;
                        let p = chunk.coords[frame * n_atoms + atom_idx];
                        let weight = if use_mass {
                            masses[atom_idx].max(0.0) as f64
                        } else {
                            1.0
                        };
                        let w = if use_mass { weight.sqrt() } else { 1.0 };
                        let base = i * 3;
                        frame_vec[base] = p[0] as f64 * w;
                        frame_vec[base + 1] = p[1] as f64 * w;
                        frame_vec[base + 2] = p[2] as f64 * w;
                    }
                    for i in 0..n_features {
                        self.sum[i] += frame_vec[i];
                    }
                    for i in 0..n_features {
                        let v_i = frame_vec[i];
                        let row = i * n_features;
                        for j in 0..n_features {
                            self.sum_outer[row + j] += v_i * frame_vec[j];
                        }
                    }
                    self.frames += 1;
                }
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        match self.mode {
            MatrixMode::Distance => {
                let n_sel = self.selection.indices.len();
                if self.frames == 0 || n_sel == 0 {
                    return Ok(PlanOutput::Matrix {
                        data: Vec::new(),
                        rows: n_sel,
                        cols: n_sel,
                    });
                }
                let inv = 1.0 / self.frames as f64;
                let mut out = Vec::with_capacity(self.sum_dist.len());
                for val in self.sum_dist.iter() {
                    out.push((val * inv) as f32);
                }
                Ok(PlanOutput::Matrix {
                    data: out,
                    rows: n_sel,
                    cols: n_sel,
                })
            }
            MatrixMode::Covariance | MatrixMode::MwCovariance => {
                let n_features = self.selection.indices.len() * 3;
                if self.frames == 0 || n_features == 0 {
                    return Ok(PlanOutput::Matrix {
                        data: Vec::new(),
                        rows: n_features,
                        cols: n_features,
                    });
                }
                let inv = 1.0 / self.frames as f64;
                let mut mean = vec![0.0f64; n_features];
                for i in 0..n_features {
                    mean[i] = self.sum[i] * inv;
                }
                let mut out = Vec::with_capacity(n_features * n_features);
                for i in 0..n_features {
                    let row = i * n_features;
                    for j in 0..n_features {
                        let cov = self.sum_outer[row + j] * inv - mean[i] * mean[j];
                        out.push(cov as f32);
                    }
                }
                Ok(PlanOutput::Matrix {
                    data: out,
                    rows: n_features,
                    cols: n_features,
                })
            }
        }
    }
}

pub struct PcaPlan {
    selection: Selection,
    n_components: usize,
    mass_weighted: bool,
    sum: Vec<f64>,
    sum_outer: Vec<f64>,
    frames: usize,
}

impl PcaPlan {
    pub fn new(selection: Selection, n_components: usize, mass_weighted: bool) -> Self {
        Self {
            selection,
            n_components,
            mass_weighted,
            sum: Vec::new(),
            sum_outer: Vec::new(),
            frames: 0,
        }
    }
}

pub struct AnalyzeModesPlan {
    inner: PcaPlan,
}

impl AnalyzeModesPlan {
    pub fn new(selection: Selection, n_components: usize, mass_weighted: bool) -> Self {
        Self {
            inner: PcaPlan::new(selection, n_components, mass_weighted),
        }
    }
}

impl Plan for PcaPlan {
    fn name(&self) -> &'static str {
        "pca"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        let n_features = self.selection.indices.len() * 3;
        self.frames = 0;
        self.sum.clear();
        self.sum_outer.clear();
        self.sum.resize(n_features, 0.0);
        self.sum_outer.resize(n_features * n_features, 0.0);
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        let n_features = sel.len() * 3;
        if n_features == 0 {
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let masses = &system.atoms.mass;
        let mut frame_vec = vec![0.0f64; n_features];
        for frame in 0..chunk.n_frames {
            for (i, &idx) in sel.iter().enumerate() {
                let atom_idx = idx as usize;
                let p = chunk.coords[frame * n_atoms + atom_idx];
                let weight = if self.mass_weighted {
                    masses[atom_idx].max(0.0) as f64
                } else {
                    1.0
                };
                let w = if self.mass_weighted {
                    weight.sqrt()
                } else {
                    1.0
                };
                let base = i * 3;
                frame_vec[base] = p[0] as f64 * w;
                frame_vec[base + 1] = p[1] as f64 * w;
                frame_vec[base + 2] = p[2] as f64 * w;
            }
            for i in 0..n_features {
                self.sum[i] += frame_vec[i];
            }
            for i in 0..n_features {
                let v_i = frame_vec[i];
                let row = i * n_features;
                for j in 0..n_features {
                    self.sum_outer[row + j] += v_i * frame_vec[j];
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_features = self.selection.indices.len() * 3;
        if self.frames == 0 || n_features == 0 {
            return Ok(PlanOutput::Pca(PcaOutput {
                eigenvalues: Vec::new(),
                eigenvectors: Vec::new(),
                n_components: 0,
                n_features,
            }));
        }
        let inv = 1.0 / self.frames as f64;
        let mut mean = vec![0.0f64; n_features];
        for i in 0..n_features {
            mean[i] = self.sum[i] * inv;
        }
        let mut cov = vec![0.0f64; n_features * n_features];
        for i in 0..n_features {
            let row = i * n_features;
            for j in 0..n_features {
                cov[row + j] = self.sum_outer[row + j] * inv - mean[i] * mean[j];
            }
        }
        let mat = DMatrix::from_row_slice(n_features, n_features, &cov);
        let eigen = SymmetricEigen::new(mat);
        let mut order: Vec<usize> = (0..n_features).collect();
        order.sort_by(|&a, &b| {
            eigen.eigenvalues[b]
                .partial_cmp(&eigen.eigenvalues[a])
                .unwrap()
        });
        let n_components = self.n_components.min(n_features).max(1);
        let mut eigenvalues = Vec::with_capacity(n_components);
        let mut eigenvectors = Vec::with_capacity(n_components * n_features);
        for k in 0..n_components {
            let col = order[k];
            eigenvalues.push(eigen.eigenvalues[col] as f32);
            for i in 0..n_features {
                eigenvectors.push(eigen.eigenvectors[(i, col)] as f32);
            }
        }
        Ok(PlanOutput::Pca(PcaOutput {
            eigenvalues,
            eigenvectors,
            n_components,
            n_features,
        }))
    }
}

impl Plan for AnalyzeModesPlan {
    fn name(&self) -> &'static str {
        "analyze_modes"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.inner.init(system, device)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.inner.finalize()
    }
}

pub struct ProjectionPlan {
    selection: Selection,
    n_components: usize,
    n_features: usize,
    eigenvectors: Vec<f64>,
    mean: Option<Vec<f64>>,
    results: Vec<f32>,
    frames: usize,
}

impl ProjectionPlan {
    pub fn new(
        selection: Selection,
        eigenvectors: Vec<f64>,
        n_components: usize,
        n_features: usize,
        mean: Option<Vec<f64>>,
    ) -> TrajResult<Self> {
        if n_components * n_features != eigenvectors.len() {
            return Err(TrajError::Mismatch("eigenvector length mismatch".into()));
        }
        if let Some(ref m) = mean {
            if m.len() != n_features {
                return Err(TrajError::Mismatch("mean length mismatch".into()));
            }
        }
        Ok(Self {
            selection,
            n_components,
            n_features,
            eigenvectors,
            mean,
            results: Vec::new(),
            frames: 0,
        })
    }
}

impl Plan for ProjectionPlan {
    fn name(&self) -> &'static str {
        "projection"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        if self.n_features != sel.len() * 3 {
            return Err(TrajError::Mismatch("selection size mismatch".into()));
        }
        if self.n_components == 0 {
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let mut frame_vec = vec![0.0f64; self.n_features];
        let mean = self.mean.as_ref();
        for frame in 0..chunk.n_frames {
            for (i, &idx) in sel.iter().enumerate() {
                let atom_idx = idx as usize;
                let p = chunk.coords[frame * n_atoms + atom_idx];
                let base = i * 3;
                frame_vec[base] = p[0] as f64;
                frame_vec[base + 1] = p[1] as f64;
                frame_vec[base + 2] = p[2] as f64;
            }
            for comp in 0..self.n_components {
                let mut dot = 0.0f64;
                let row = comp * self.n_features;
                for i in 0..self.n_features {
                    let v = frame_vec[i] - mean.map(|m| m[i]).unwrap_or(0.0);
                    dot += v * self.eigenvectors[row + i];
                }
                self.results.push(dot as f32);
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.n_components,
        })
    }
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
