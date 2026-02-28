use std::f64::consts::PI;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, Float4, GpuSelection};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfAlgorithm {
    Bbox,
    Sasa,
}

pub struct SurfPlan {
    selection: Selection,
    algorithm: SurfAlgorithm,
    probe_radius: f32,
    n_sphere_points: usize,
    radii: Option<Vec<f32>>,
    resolved_radii: Vec<f64>,
    sphere_points: Vec<[f64; 3]>,
    results: Vec<f32>,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuSelection>,
}

impl SurfPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            algorithm: SurfAlgorithm::Bbox,
            probe_radius: 1.4,
            n_sphere_points: 64,
            radii: None,
            resolved_radii: Vec::new(),
            sphere_points: Vec::new(),
            results: Vec::new(),
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_algorithm(mut self, algorithm: SurfAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    pub fn with_probe_radius(mut self, probe_radius: f32) -> Self {
        self.probe_radius = probe_radius;
        self
    }

    pub fn with_n_sphere_points(mut self, n_sphere_points: usize) -> Self {
        self.n_sphere_points = n_sphere_points.max(8);
        self
    }

    pub fn with_radii(mut self, radii: Option<Vec<f32>>) -> Self {
        self.radii = radii;
        self
    }
}

impl Plan for SurfPlan {
    fn name(&self) -> &'static str {
        "surf"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.sphere_points.clear();
        self.resolved_radii.clear();
        if matches!(self.algorithm, SurfAlgorithm::Sasa) {
            self.sphere_points = fibonacci_sphere(self.n_sphere_points);
            self.resolved_radii =
                resolve_radii(system, &self.selection.indices, self.radii.as_ref())?;
        }
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let selection = ctx.selection(&self.selection.indices, None)?;
                self.gpu = Some(selection);
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
        #[cfg(feature = "cuda")]
        if matches!(self.algorithm, SurfAlgorithm::Bbox) {
            if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
                let coords = convert_coords(&chunk.coords);
                let out = ctx.bbox_area(&coords, chunk.n_atoms, chunk.n_frames, gpu)?;
                self.results.extend(out);
                return Ok(());
            }
        }
        #[cfg(feature = "cuda")]
        if matches!(self.algorithm, SurfAlgorithm::Sasa) {
            if let (Device::Cuda(ctx), Some(gpu)) = (_device, &self.gpu) {
                let sel = &self.selection.indices;
                if sel.is_empty() {
                    self.results
                        .extend(std::iter::repeat(0.0f32).take(chunk.n_frames));
                    return Ok(());
                }
                if self.resolved_radii.len() != sel.len() {
                    self.resolved_radii = resolve_radii(system, sel, self.radii.as_ref())?;
                }
                let probe = self.probe_radius.max(0.0) as f64;
                let expanded: Vec<f32> = self
                    .resolved_radii
                    .iter()
                    .map(|r| (*r + probe).max(0.0) as f32)
                    .collect();
                let sphere: Vec<Float4> = self
                    .sphere_points
                    .iter()
                    .map(|p| Float4 {
                        x: p[0] as f32,
                        y: p[1] as f32,
                        z: p[2] as f32,
                        w: 0.0,
                    })
                    .collect();
                let coords = convert_coords(&chunk.coords);
                let out = ctx.sasa_approx(
                    &coords,
                    chunk.n_atoms,
                    chunk.n_frames,
                    gpu,
                    &expanded,
                    &sphere,
                )?;
                self.results.extend(out);
                return Ok(());
            }
        }

        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        if sel.is_empty() {
            self.results
                .extend(std::iter::repeat(0.0f32).take(chunk.n_frames));
            return Ok(());
        }

        match self.algorithm {
            SurfAlgorithm::Bbox => {
                for frame in 0..chunk.n_frames {
                    let mut min = [f64::INFINITY; 3];
                    let mut max = [f64::NEG_INFINITY; 3];
                    for &idx in sel.iter() {
                        let p = chunk.coords[frame * n_atoms + idx as usize];
                        let x = p[0] as f64;
                        let y = p[1] as f64;
                        let z = p[2] as f64;
                        if x < min[0] {
                            min[0] = x;
                        }
                        if y < min[1] {
                            min[1] = y;
                        }
                        if z < min[2] {
                            min[2] = z;
                        }
                        if x > max[0] {
                            max[0] = x;
                        }
                        if y > max[1] {
                            max[1] = y;
                        }
                        if z > max[2] {
                            max[2] = z;
                        }
                    }
                    let dx = (max[0] - min[0]).max(0.0);
                    let dy = (max[1] - min[1]).max(0.0);
                    let dz = (max[2] - min[2]).max(0.0);
                    let area = 2.0 * (dx * dy + dy * dz + dx * dz);
                    self.results.push(area as f32);
                }
            }
            SurfAlgorithm::Sasa => {
                if self.resolved_radii.len() != sel.len() {
                    self.resolved_radii = resolve_radii(system, sel, self.radii.as_ref())?;
                }
                let probe = self.probe_radius.max(0.0) as f64;
                let n_points = self.sphere_points.len().max(1) as f64;
                let expanded: Vec<f64> = self
                    .resolved_radii
                    .iter()
                    .map(|r| (*r + probe).max(0.0))
                    .collect();
                let cutoff2: Vec<f64> = expanded.iter().map(|r| r * r).collect();
                let area_const: Vec<f64> = expanded.iter().map(|r| 4.0 * PI * r * r).collect();
                let mut coords_sel = vec![[0.0f64; 3]; sel.len()];
                for frame in 0..chunk.n_frames {
                    let frame_offset = frame * n_atoms;
                    for (i, &idx) in sel.iter().enumerate() {
                        let p = chunk.coords[frame_offset + idx as usize];
                        coords_sel[i] = [p[0] as f64, p[1] as f64, p[2] as f64];
                    }
                    let mut total = 0.0f64;
                    for i in 0..coords_sel.len() {
                        let radius = expanded[i];
                        if radius <= 0.0 {
                            continue;
                        }
                        let mut exposed = 0usize;
                        let center = coords_sel[i];
                        for dir in self.sphere_points.iter() {
                            let px = center[0] + dir[0] * radius;
                            let py = center[1] + dir[1] * radius;
                            let pz = center[2] + dir[2] * radius;
                            let mut occluded = false;
                            for j in 0..coords_sel.len() {
                                if j == i {
                                    continue;
                                }
                                let other = coords_sel[j];
                                let dx = px - other[0];
                                let dy = py - other[1];
                                let dz = pz - other[2];
                                if dx * dx + dy * dy + dz * dz < cutoff2[j] {
                                    occluded = true;
                                    break;
                                }
                            }
                            if !occluded {
                                exposed += 1;
                            }
                        }
                        total += (exposed as f64 / n_points) * area_const[i];
                    }
                    self.results.push(total as f32);
                }
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

pub struct MolSurfPlan {
    inner: SurfPlan,
}

impl MolSurfPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            inner: SurfPlan::new(selection)
                .with_algorithm(SurfAlgorithm::Sasa)
                .with_probe_radius(0.0),
        }
    }

    pub fn with_algorithm(mut self, algorithm: SurfAlgorithm) -> Self {
        self.inner = self.inner.with_algorithm(algorithm);
        self
    }

    pub fn with_probe_radius(mut self, probe_radius: f32) -> Self {
        self.inner = self.inner.with_probe_radius(probe_radius);
        self
    }

    pub fn with_n_sphere_points(mut self, n_sphere_points: usize) -> Self {
        self.inner = self.inner.with_n_sphere_points(n_sphere_points);
        self
    }

    pub fn with_radii(mut self, radii: Option<Vec<f32>>) -> Self {
        self.inner = self.inner.with_radii(radii);
        self
    }
}

impl Plan for MolSurfPlan {
    fn name(&self) -> &'static str {
        "molsurf"
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

fn fibonacci_sphere(n_points: usize) -> Vec<[f64; 3]> {
    let n = n_points.max(8);
    let phi = (1.0 + 5.0_f64.sqrt()) * 0.5;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let z = 1.0 - 2.0 * ((i as f64) + 0.5) / (n as f64);
        let r = (1.0 - z * z).max(0.0).sqrt();
        let theta = 2.0 * PI * (i as f64) / phi;
        points.push([r * theta.cos(), r * theta.sin(), z]);
    }
    points
}

fn resolve_radii(
    system: &System,
    selection: &[u32],
    radii: Option<&Vec<f32>>,
) -> TrajResult<Vec<f64>> {
    if let Some(custom) = radii {
        if custom.len() != selection.len() {
            return Err(TrajError::Parse(
                "surf radii length must match selected atom count".into(),
            ));
        }
        return Ok(custom.iter().map(|r| *r as f64).collect());
    }
    let mut out = Vec::with_capacity(selection.len());
    for &idx in selection.iter() {
        out.push(default_radius(system, idx as usize));
    }
    Ok(out)
}

fn default_radius(system: &System, atom_idx: usize) -> f64 {
    let default = 1.7f64;
    let atoms = &system.atoms;
    if atom_idx >= atoms.name_id.len() {
        return default;
    }
    let mut symbol = String::new();
    if let Some(&elem_id) = atoms.element_id.get(atom_idx) {
        if let Some(elem) = system.interner.resolve(elem_id) {
            symbol = elem.trim().to_ascii_uppercase();
        }
    }
    if symbol.is_empty() {
        if let Some(&name_id) = atoms.name_id.get(atom_idx) {
            if let Some(name) = system.interner.resolve(name_id) {
                let up = name.trim().to_ascii_uppercase();
                let two = up.chars().take(2).collect::<String>();
                symbol = if radius_from_symbol(&two).is_some() {
                    two
                } else {
                    up.chars().take(1).collect::<String>()
                };
            }
        }
    }
    radius_from_symbol(&symbol).unwrap_or(default)
}

fn radius_from_symbol(symbol: &str) -> Option<f64> {
    match symbol {
        "H" => Some(1.20),
        "C" => Some(1.70),
        "N" => Some(1.55),
        "O" => Some(1.52),
        "S" => Some(1.80),
        "P" => Some(1.80),
        "F" => Some(1.47),
        "CL" => Some(1.75),
        _ => None,
    }
}
