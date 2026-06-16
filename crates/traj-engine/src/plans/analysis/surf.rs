use std::f64::consts::PI;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, SurfaceOutput};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, Float4, GpuSelection};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfAlgorithm {
    Bbox,
    Lcpo,
    Sasa,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceRadiiMode {
    Gb,
    Parse,
    Vdw,
}

pub struct SurfPlan {
    selection: Selection,
    algorithm: SurfAlgorithm,
    probe_radius: f32,
    radius_offset: f32,
    neighbor_cutoff: f32,
    n_sphere_points: usize,
    radii: Option<Vec<f32>>,
    radii_mode: SurfaceRadiiMode,
    solute_selection: Option<Selection>,
    resolved_radii: Vec<f64>,
    lcpo: LcpoWorkspace,
    sphere_points: Vec<[f64; 3]>,
    results: Vec<f32>,
    record_atom_area: bool,
    record_volume: bool,
    record_residue_area: bool,
    atom_area_results: Vec<f32>,
    volume_results: Vec<f32>,
    residue_area_results: Vec<f32>,
    residue_ids: Vec<i32>,
    residue_slot_by_selected_atom: Vec<usize>,
    use_selected_input: bool,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuSelection>,
}

impl SurfPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            algorithm: SurfAlgorithm::Bbox,
            probe_radius: 1.4,
            radius_offset: 0.0,
            neighbor_cutoff: 2.5,
            n_sphere_points: 64,
            radii: None,
            radii_mode: SurfaceRadiiMode::Gb,
            solute_selection: None,
            resolved_radii: Vec::new(),
            lcpo: LcpoWorkspace::default(),
            sphere_points: Vec::new(),
            results: Vec::new(),
            record_atom_area: false,
            record_volume: false,
            record_residue_area: false,
            atom_area_results: Vec::new(),
            volume_results: Vec::new(),
            residue_area_results: Vec::new(),
            residue_ids: Vec::new(),
            residue_slot_by_selected_atom: Vec::new(),
            use_selected_input: true,
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

    pub fn with_radius_offset(mut self, radius_offset: f32) -> Self {
        self.radius_offset = radius_offset;
        self
    }

    pub fn with_neighbor_cutoff(mut self, neighbor_cutoff: f32) -> Self {
        self.neighbor_cutoff = neighbor_cutoff;
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

    pub fn with_radii_mode(mut self, radii_mode: SurfaceRadiiMode) -> Self {
        self.radii_mode = radii_mode;
        self
    }

    pub fn with_solute_selection(mut self, solute_selection: Option<Selection>) -> Self {
        self.solute_selection = solute_selection;
        self
    }

    pub fn with_atom_area(mut self, enabled: bool) -> Self {
        self.record_atom_area = enabled;
        self
    }

    pub fn with_volume(mut self, enabled: bool) -> Self {
        self.record_volume = enabled;
        self
    }

    pub fn with_residue_area(mut self, enabled: bool) -> Self {
        self.record_residue_area = enabled;
        self
    }
}

impl Plan for SurfPlan {
    fn name(&self) -> &'static str {
        "surf"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.atom_area_results.clear();
        self.volume_results.clear();
        self.residue_area_results.clear();
        self.residue_ids.clear();
        self.residue_slot_by_selected_atom.clear();
        self.sphere_points.clear();
        self.resolved_radii.clear();
        self.lcpo = LcpoWorkspace::default();
        self.use_selected_input =
            matches!(_device, Device::Cpu) && !matches!(self.algorithm, SurfAlgorithm::Lcpo);
        if (self.record_atom_area || self.record_volume || self.record_residue_area)
            && !matches!(self.algorithm, SurfAlgorithm::Sasa)
        {
            return Err(TrajError::Parse(
                "surface detail output requires the sasa algorithm".into(),
            ));
        }
        if matches!(self.algorithm, SurfAlgorithm::Sasa) {
            self.sphere_points = fibonacci_sphere(self.n_sphere_points);
            self.resolved_radii = resolve_radii_with_offset(
                system,
                &self.selection.indices,
                self.radii.as_ref(),
                self.radii_mode,
                self.radius_offset as f64,
            )?;
            if self.record_residue_area {
                let (residue_ids, slots) = build_residue_area_map(system, &self.selection.indices)?;
                self.residue_ids = residue_ids;
                self.residue_slot_by_selected_atom = slots;
            }
        } else if matches!(self.algorithm, SurfAlgorithm::Lcpo) {
            if self.radii.is_some() {
                return Err(TrajError::Parse(
                    "lcpo surface uses LCPO atom parameters; custom radii are only supported for sasa"
                        .into(),
                ));
            }
            self.lcpo = build_lcpo_workspace(
                system,
                &self.selection.indices,
                self.solute_selection
                    .as_ref()
                    .map(|sel| sel.indices.as_slice()),
                self.radius_offset as f64,
                self.neighbor_cutoff as f64,
            )?;
        }
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                self.use_selected_input = false;
                let selection = ctx.selection(&self.selection.indices, None)?;
                self.gpu = Some(selection);
            }
        }
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input {
            Some(self.selection.indices.as_slice())
        } else {
            None
        }
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        if matches!(self.algorithm, SurfAlgorithm::Lcpo) {
            None
        } else {
            Some(self.selection.indices.as_slice())
        }
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
                if !(self.record_atom_area || self.record_volume || self.record_residue_area) {
                    let sel = &self.selection.indices;
                    if sel.is_empty() {
                        self.results
                            .extend(std::iter::repeat(0.0f32).take(chunk.n_frames));
                        return Ok(());
                    }
                    if self.resolved_radii.len() != sel.len() {
                        self.resolved_radii = resolve_radii_with_offset(
                            system,
                            sel,
                            self.radii.as_ref(),
                            self.radii_mode,
                            self.radius_offset as f64,
                        )?;
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
        }

        let n_atoms = chunk.n_atoms;
        let sel = &self.selection.indices;
        if sel.is_empty() {
            self.results
                .extend(std::iter::repeat(0.0f32).take(chunk.n_frames));
            if self.record_volume {
                self.volume_results
                    .extend(std::iter::repeat(0.0f32).take(chunk.n_frames));
            }
            if self.record_residue_area {
                self.residue_area_results.extend(
                    std::iter::repeat(0.0f32).take(chunk.n_frames * self.residue_ids.len()),
                );
            }
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
            SurfAlgorithm::Lcpo => {
                if self.lcpo.needs_init {
                    self.lcpo = build_lcpo_workspace(
                        system,
                        &self.selection.indices,
                        self.solute_selection
                            .as_ref()
                            .map(|sel| sel.indices.as_slice()),
                        self.radius_offset as f64,
                        self.neighbor_cutoff as f64,
                    )?;
                }
                process_lcpo_chunk(chunk, &self.lcpo, &mut self.results);
            }
            SurfAlgorithm::Sasa => {
                if self.resolved_radii.len() != sel.len() {
                    self.resolved_radii = resolve_radii_with_offset(
                        system,
                        sel,
                        self.radii.as_ref(),
                        self.radii_mode,
                        self.radius_offset as f64,
                    )?;
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
                    let mut center_sum = [0.0f64; 3];
                    for (i, &idx) in sel.iter().enumerate() {
                        let p = chunk.coords[frame_offset + idx as usize];
                        coords_sel[i] = [p[0] as f64, p[1] as f64, p[2] as f64];
                        center_sum[0] += coords_sel[i][0];
                        center_sum[1] += coords_sel[i][1];
                        center_sum[2] += coords_sel[i][2];
                    }
                    let inv_atoms = 1.0 / coords_sel.len() as f64;
                    let volume_center = [
                        center_sum[0] * inv_atoms,
                        center_sum[1] * inv_atoms,
                        center_sum[2] * inv_atoms,
                    ];
                    let mut total = 0.0f64;
                    let mut volume_sum = 0.0f64;
                    let mut frame_residue_area = if self.record_residue_area {
                        vec![0.0f32; self.residue_ids.len()]
                    } else {
                        Vec::new()
                    };
                    for i in 0..coords_sel.len() {
                        let radius = expanded[i];
                        let (atom_area, atom_volume_raw) = if radius <= 0.0 {
                            (0.0, 0.0)
                        } else {
                            let mut exposed = 0usize;
                            let mut exposed_dir_sum = [0.0f64; 3];
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
                                    exposed_dir_sum[0] += dir[0];
                                    exposed_dir_sum[1] += dir[1];
                                    exposed_dir_sum[2] += dir[2];
                                }
                            }
                            let atom_area = (exposed as f64 / n_points) * area_const[i];
                            let dx = center[0] - volume_center[0];
                            let dy = center[1] - volume_center[1];
                            let dz = center[2] - volume_center[2];
                            let atom_volume_raw = radius
                                * radius
                                * (exposed_dir_sum[0] * dx
                                    + exposed_dir_sum[1] * dy
                                    + exposed_dir_sum[2] * dz
                                    + radius * exposed as f64);
                            (atom_area, atom_volume_raw)
                        };
                        if self.record_atom_area {
                            self.atom_area_results.push(atom_area as f32);
                        }
                        if self.record_residue_area {
                            let slot = self.residue_slot_by_selected_atom[i];
                            if let Some(value) = frame_residue_area.get_mut(slot) {
                                *value += atom_area as f32;
                            }
                        }
                        total += atom_area;
                        volume_sum += atom_volume_raw;
                    }
                    self.results.push(total as f32);
                    if self.record_residue_area {
                        self.residue_area_results.extend(frame_residue_area);
                    }
                    if self.record_volume {
                        self.volume_results
                            .push((volume_sum * 4.0 * PI / (3.0 * n_points)) as f32);
                    }
                }
            }
        }
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if !self.use_selected_input {
            return Err(TrajError::Mismatch(
                "surface selected chunk received while selected IO is disabled".into(),
            ));
        }
        if source_selection != self.selection.indices.as_slice() {
            return Err(TrajError::Mismatch(
                "surface selected chunk does not match expected selection".into(),
            ));
        }
        match self.algorithm {
            SurfAlgorithm::Bbox => process_bbox_selected_chunk(chunk, &mut self.results),
            SurfAlgorithm::Sasa => self.process_sasa_selected_chunk(chunk)?,
            SurfAlgorithm::Lcpo => {
                return Err(TrajError::Mismatch(
                    "lcpo surface requires full-frame coordinates".into(),
                ));
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.record_atom_area || self.record_volume || self.record_residue_area {
            let total = std::mem::take(&mut self.results);
            let atom_area = std::mem::take(&mut self.atom_area_results);
            let volume = std::mem::take(&mut self.volume_results);
            let residue_area = std::mem::take(&mut self.residue_area_results);
            let residue_ids = std::mem::take(&mut self.residue_ids);
            let frames = total.len();
            let atoms = self.selection.indices.len();
            let residues = residue_ids.len();
            return Ok(PlanOutput::Surface(SurfaceOutput {
                total,
                atom_area,
                volume,
                residue_area,
                residue_ids,
                frames,
                atoms,
                residues,
            }));
        }
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}

impl SurfPlan {
    fn process_sasa_selected_chunk(&mut self, chunk: &FrameChunk) -> TrajResult<()> {
        let n_selected = self.selection.indices.len();
        if chunk.n_atoms != n_selected {
            return Err(TrajError::Mismatch(
                "surface selected chunk atom count does not match selection".into(),
            ));
        }
        if n_selected == 0 {
            self.results
                .extend(std::iter::repeat(0.0f32).take(chunk.n_frames));
            if self.record_volume {
                self.volume_results
                    .extend(std::iter::repeat(0.0f32).take(chunk.n_frames));
            }
            if self.record_residue_area {
                self.residue_area_results.extend(
                    std::iter::repeat(0.0f32).take(chunk.n_frames * self.residue_ids.len()),
                );
            }
            return Ok(());
        }
        if self.resolved_radii.len() != n_selected {
            return Err(TrajError::Mismatch(
                "surface selected radii length does not match selection".into(),
            ));
        }
        if self.sphere_points.is_empty() {
            self.sphere_points = fibonacci_sphere(self.n_sphere_points);
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
        let mut coords_sel = vec![[0.0f64; 3]; n_selected];

        for frame in 0..chunk.n_frames {
            let frame_offset = frame * chunk.n_atoms;
            let mut center_sum = [0.0f64; 3];
            for (i, coord) in coords_sel.iter_mut().enumerate() {
                let p = chunk.coords[frame_offset + i];
                *coord = [p[0] as f64, p[1] as f64, p[2] as f64];
                center_sum[0] += coord[0];
                center_sum[1] += coord[1];
                center_sum[2] += coord[2];
            }
            let inv_atoms = 1.0 / n_selected as f64;
            let volume_center = [
                center_sum[0] * inv_atoms,
                center_sum[1] * inv_atoms,
                center_sum[2] * inv_atoms,
            ];
            let mut total = 0.0f64;
            let mut volume_sum = 0.0f64;
            let mut frame_residue_area = if self.record_residue_area {
                vec![0.0f32; self.residue_ids.len()]
            } else {
                Vec::new()
            };

            for i in 0..n_selected {
                let radius = expanded[i];
                let (atom_area, atom_volume_raw) = if radius <= 0.0 {
                    (0.0, 0.0)
                } else {
                    let mut exposed = 0usize;
                    let mut exposed_dir_sum = [0.0f64; 3];
                    let center = coords_sel[i];
                    for dir in self.sphere_points.iter() {
                        let px = center[0] + dir[0] * radius;
                        let py = center[1] + dir[1] * radius;
                        let pz = center[2] + dir[2] * radius;
                        let mut occluded = false;
                        for j in 0..n_selected {
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
                            exposed_dir_sum[0] += dir[0];
                            exposed_dir_sum[1] += dir[1];
                            exposed_dir_sum[2] += dir[2];
                        }
                    }
                    let atom_area = (exposed as f64 / n_points) * area_const[i];
                    let dx = center[0] - volume_center[0];
                    let dy = center[1] - volume_center[1];
                    let dz = center[2] - volume_center[2];
                    let atom_volume_raw = radius
                        * radius
                        * (exposed_dir_sum[0] * dx
                            + exposed_dir_sum[1] * dy
                            + exposed_dir_sum[2] * dz
                            + radius * exposed as f64);
                    (atom_area, atom_volume_raw)
                };
                if self.record_atom_area {
                    self.atom_area_results.push(atom_area as f32);
                }
                if self.record_residue_area {
                    let slot = self.residue_slot_by_selected_atom[i];
                    if let Some(value) = frame_residue_area.get_mut(slot) {
                        *value += atom_area as f32;
                    }
                }
                total += atom_area;
                volume_sum += atom_volume_raw;
            }
            self.results.push(total as f32);
            if self.record_residue_area {
                self.residue_area_results.extend(frame_residue_area);
            }
            if self.record_volume {
                self.volume_results
                    .push((volume_sum * 4.0 * PI / (3.0 * n_points)) as f32);
            }
        }
        Ok(())
    }
}

fn process_bbox_selected_chunk(chunk: &FrameChunk, results: &mut Vec<f32>) {
    if chunk.n_atoms == 0 {
        results.extend(std::iter::repeat(0.0f32).take(chunk.n_frames));
        return;
    }
    for frame in 0..chunk.n_frames {
        let frame_offset = frame * chunk.n_atoms;
        let frame_coords = &chunk.coords[frame_offset..frame_offset + chunk.n_atoms];
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        for p in frame_coords {
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
        results.push((2.0 * (dx * dy + dy * dz + dx * dz)) as f32);
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
                .with_probe_radius(1.4),
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

    pub fn with_radius_offset(mut self, radius_offset: f32) -> Self {
        self.inner = self.inner.with_radius_offset(radius_offset);
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

    pub fn with_radii_mode(mut self, radii_mode: SurfaceRadiiMode) -> Self {
        self.inner = self.inner.with_radii_mode(radii_mode);
        self
    }

    pub fn with_atom_area(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_atom_area(enabled);
        self
    }

    pub fn with_volume(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_volume(enabled);
        self
    }

    pub fn with_residue_area(mut self, enabled: bool) -> Self {
        self.inner = self.inner.with_residue_area(enabled);
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

    fn preferred_selection(&self) -> Option<&[u32]> {
        self.inner.preferred_selection()
    }

    fn preferred_selection_hint(&self, system: &System) -> Option<&[u32]> {
        self.inner.preferred_selection_hint(system)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner
            .process_chunk_selected(chunk, source_selection, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.inner.finalize()
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct LcpoInfo {
    atom: usize,
    vdw_radius: f64,
    p1: f64,
    p2: f64,
    p3: f64,
    p4: f64,
}

#[derive(Debug)]
struct LcpoWorkspace {
    needs_init: bool,
    heavy_atoms: Vec<usize>,
    heavy_radii: Vec<f64>,
    sa_params: Vec<LcpoInfo>,
    no_neighbor_term: f64,
}

impl Default for LcpoWorkspace {
    fn default() -> Self {
        Self {
            needs_init: true,
            heavy_atoms: Vec::new(),
            heavy_radii: Vec::new(),
            sa_params: Vec::new(),
            no_neighbor_term: 0.0,
        }
    }
}

fn build_lcpo_workspace(
    system: &System,
    selection: &[u32],
    solute_selection: Option<&[u32]>,
    radius_offset: f64,
    neighbor_cutoff: f64,
) -> TrajResult<LcpoWorkspace> {
    let n_atoms = system.n_atoms();
    let mut selected = vec![false; n_atoms];
    for &idx in selection.iter() {
        let idx = idx as usize;
        if idx >= n_atoms {
            return Err(TrajError::Mismatch(
                "surf selection contains atom index outside system".into(),
            ));
        }
        selected[idx] = true;
    }

    let solute_atoms: Vec<usize> = if let Some(indices) = solute_selection {
        let mut atoms = Vec::with_capacity(indices.len());
        for &idx in indices.iter() {
            let idx = idx as usize;
            if idx >= n_atoms {
                return Err(TrajError::Mismatch(
                    "surf solutemask contains atom index outside system".into(),
                ));
            }
            atoms.push(idx);
        }
        atoms
    } else {
        (0..n_atoms).collect()
    };

    let mut workspace = LcpoWorkspace {
        needs_init: false,
        heavy_atoms: Vec::new(),
        heavy_radii: Vec::new(),
        sa_params: Vec::new(),
        no_neighbor_term: 0.0,
    };
    let cutoff = neighbor_cutoff.max(0.0);

    for atom_idx in solute_atoms {
        let info = lcpo_info_for_atom(system, atom_idx, radius_offset);
        if info.vdw_radius > cutoff {
            workspace.heavy_atoms.push(atom_idx);
            workspace.heavy_radii.push(info.vdw_radius);
            if selected[atom_idx] {
                workspace.sa_params.push(info);
            }
        } else if selected[atom_idx] {
            let si = 4.0 * PI * info.vdw_radius * info.vdw_radius;
            workspace.no_neighbor_term += info.p1 * si;
        }
    }

    Ok(workspace)
}

fn build_residue_area_map(
    system: &System,
    selection: &[u32],
) -> TrajResult<(Vec<i32>, Vec<usize>)> {
    let n_atoms = system.n_atoms();
    let mut keys: Vec<(u32, i32, u32)> = Vec::new();
    let mut residue_ids: Vec<i32> = Vec::new();
    let mut slots = Vec::with_capacity(selection.len());
    for &idx in selection.iter() {
        let atom_idx = idx as usize;
        if atom_idx >= n_atoms {
            return Err(TrajError::Mismatch(
                "surf selection contains atom index outside system".into(),
            ));
        }
        let key = (
            system.atoms.chain_id[atom_idx],
            system.atoms.resid[atom_idx],
            system.atoms.resname_id[atom_idx],
        );
        let slot = if let Some(pos) = keys.iter().position(|existing| *existing == key) {
            pos
        } else {
            let pos = keys.len();
            keys.push(key);
            residue_ids.push(system.atoms.resid[atom_idx]);
            pos
        };
        slots.push(slot);
    }
    Ok((residue_ids, slots))
}

fn process_lcpo_chunk(chunk: &FrameChunk, workspace: &LcpoWorkspace, results: &mut Vec<f32>) {
    let n_atoms = chunk.n_atoms;

    let mut heavy_coords = vec![[0.0f64; 3]; workspace.heavy_atoms.len()];
    let mut neighbor_indices: Vec<usize> = Vec::new();
    let mut neighbor_distances: Vec<f64> = Vec::new();

    for frame in 0..chunk.n_frames {
        let frame_offset = frame * n_atoms;
        let mut total = workspace.no_neighbor_term;
        if workspace.sa_params.is_empty() {
            results.push(total as f32);
            continue;
        }

        for (slot, &atom_idx) in workspace.heavy_atoms.iter().enumerate() {
            let p = chunk.coords[frame_offset + atom_idx];
            heavy_coords[slot] = [p[0] as f64, p[1] as f64, p[2] as f64];
        }

        for info in workspace.sa_params.iter() {
            let atomi = info.atom;
            let p = chunk.coords[frame_offset + atomi];
            let center_i = [p[0] as f64, p[1] as f64, p[2] as f64];
            let vdwi = info.vdw_radius;
            let vdwi2 = vdwi * vdwi;
            let si = 4.0 * PI * vdwi2;

            neighbor_indices.clear();
            neighbor_distances.clear();
            for (jdx, &atomj) in workspace.heavy_atoms.iter().enumerate() {
                if atomj == atomi {
                    continue;
                }
                let center_j = heavy_coords[jdx];
                let dij2 = dist2(center_i, center_j);
                let touch = vdwi + workspace.heavy_radii[jdx];
                if dij2 > f64::EPSILON && touch * touch > dij2 {
                    neighbor_indices.push(jdx);
                    neighbor_distances.push(dij2.sqrt());
                }
            }

            let mut sum_aij = 0.0f64;
            let mut sum_ajk = 0.0f64;
            let mut sum_aij_ajk = 0.0f64;

            for (mm, &jdx) in neighbor_indices.iter().enumerate() {
                let dij = neighbor_distances[mm];
                let vdwj = workspace.heavy_radii[jdx];
                let vdwj2 = vdwj * vdwj;
                let tmp_aij = vdwi - (dij * 0.5) - ((vdwi2 - vdwj2) / (2.0 * dij));
                let aij = 2.0 * PI * vdwi * tmp_aij;
                sum_aij += aij;

                let mut sum_ajk_for_j = 0.0f64;
                for (nn, &kdx) in neighbor_indices.iter().enumerate() {
                    if mm == nn {
                        continue;
                    }
                    let center_j = heavy_coords[jdx];
                    let center_k = heavy_coords[kdx];
                    let djk2 = dist2(center_j, center_k);
                    let vdwk = workspace.heavy_radii[kdx];
                    let touch = vdwj + vdwk;
                    if djk2 <= f64::EPSILON || touch * touch <= djk2 {
                        continue;
                    }
                    let djk = djk2.sqrt();
                    let vdw2_dif = vdwj2 - (vdwk * vdwk);
                    let tmp_ajk = (2.0 * vdwj) - djk - (vdw2_dif / djk);
                    let ajk = PI * vdwj * tmp_ajk;
                    sum_ajk += ajk;
                    sum_ajk_for_j += ajk;
                }
                sum_aij_ajk += aij * sum_ajk_for_j;
            }

            total += (info.p1 * si)
                + (info.p2 * sum_aij)
                + (info.p3 * sum_ajk)
                + (info.p4 * sum_aij_ajk);
        }
        results.push(total as f32);
    }
}

fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    dx * dx + dy * dy + dz * dz
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

pub(crate) fn resolve_radii(
    system: &System,
    selection: &[u32],
    radii: Option<&Vec<f32>>,
) -> TrajResult<Vec<f64>> {
    resolve_radii_with_offset(system, selection, radii, SurfaceRadiiMode::Gb, 0.0)
}

fn resolve_radii_with_offset(
    system: &System,
    selection: &[u32],
    radii: Option<&Vec<f32>>,
    radii_mode: SurfaceRadiiMode,
    offset: f64,
) -> TrajResult<Vec<f64>> {
    if let Some(custom) = radii {
        if custom.len() != selection.len() {
            return Err(TrajError::Parse(
                "surf radii length must match selected atom count".into(),
            ));
        }
        return Ok(custom
            .iter()
            .map(|r| ((*r as f64) + offset).max(0.0))
            .collect());
    }
    let mut out = Vec::with_capacity(selection.len());
    for &idx in selection.iter() {
        out.push((radius_for_mode(system, idx as usize, radii_mode)? + offset).max(0.0));
    }
    Ok(out)
}

fn radius_for_mode(
    system: &System,
    atom_idx: usize,
    radii_mode: SurfaceRadiiMode,
) -> TrajResult<f64> {
    match radii_mode {
        SurfaceRadiiMode::Gb => {
            if let Some(radius) = system.gb_radius_for_atom(atom_idx) {
                Ok(radius as f64)
            } else {
                Ok(element_radius(system, atom_idx))
            }
        }
        SurfaceRadiiMode::Parse => {
            if let Some(radius) = system.parse_radius_for_atom(atom_idx) {
                Ok(radius as f64)
            } else {
                Ok(parse_radius(system, atom_idx))
            }
        }
        SurfaceRadiiMode::Vdw => system
            .vdw_radius_for_atom(atom_idx)
            .map(|radius| radius as f64)
            .ok_or_else(|| {
                TrajError::Parse(
                    "vdw radii requested but system does not carry nonbonded vdW radii".into(),
                )
            }),
    }
}

fn element_radius(system: &System, atom_idx: usize) -> f64 {
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

fn parse_radius(system: &System, atom_idx: usize) -> f64 {
    match atom_symbol(system, atom_idx).as_str() {
        "H" => 1.0,
        "C" => 1.7,
        "N" => 1.5,
        "O" => 1.4,
        "P" => 2.0,
        "S" => 1.85,
        _ => 0.0,
    }
}

fn lcpo_info_for_atom(system: &System, atom_idx: usize, radius_offset: f64) -> LcpoInfo {
    let symbol = atom_symbol(system, atom_idx);
    let atom_type = atom_type_upper(system, atom_idx);
    let mut chars = atom_type.chars();
    let atype0 = chars.next().unwrap_or('\0');
    let atype1 = chars.next().unwrap_or('\0');
    let total_bonds = inferred_bond_count(system, atom_idx, false);
    let non_h_bonds = inferred_bond_count(system, atom_idx, true);
    let (vdw, p1, p2, p3, p4) = match symbol.as_str() {
        "C" => {
            if total_bonds == 4 {
                match non_h_bonds {
                    1 => (1.70, 0.77887, -0.28063, -0.0012968, 0.00039328),
                    2 => (1.70, 0.56482, -0.19608, -0.0010219, 0.0002658),
                    3 => (1.70, 0.23348, -0.072627, -0.00020079, 0.00007967),
                    4 => (1.70, 0.0, 0.0, 0.0, 0.0),
                    _ => (1.70, 0.77887, -0.28063, -0.0012968, 0.00039328),
                }
            } else {
                match non_h_bonds {
                    2 => (1.70, 0.51245, -0.15966, -0.00019781, 0.00016392),
                    3 => (1.70, 0.070344, -0.019015, -0.000022009, 0.000016875),
                    _ => (1.70, 0.77887, -0.28063, -0.0012968, 0.00039328),
                }
            }
        }
        "O" => {
            if atype0 == 'O' && atype1 == '\0' {
                (1.60, 0.68563, -0.1868, -0.00135573, 0.00023743)
            } else if atype0 == 'O' && atype1 == '2' {
                (1.60, 0.88857, -0.33421, -0.0018683, 0.00049372)
            } else {
                match non_h_bonds {
                    1 => (1.60, 0.77914, -0.25262, -0.0016056, 0.00035071),
                    2 => (1.60, 0.49392, -0.16038, -0.00015512, 0.00016453),
                    _ => (1.60, 0.77914, -0.25262, -0.0016056, 0.00035071),
                }
            }
        }
        "N" => {
            if atype0 == 'N' && atype1 == '3' {
                match non_h_bonds {
                    1 => (1.65, 0.078602, -0.29198, -0.0006537, 0.00036247),
                    2 => (1.65, 0.22599, -0.036648, -0.0012297, 0.000080038),
                    3 => (1.65, 0.051481, -0.012603, -0.00032006, 0.000024774),
                    _ => (1.65, 0.078602, -0.29198, -0.0006537, 0.00036247),
                }
            } else {
                match non_h_bonds {
                    1 => (1.65, 0.73511, -0.22116, -0.00089148, 0.0002523),
                    2 => (1.65, 0.41102, -0.12254, -0.000075448, 0.00011804),
                    3 => (1.65, 0.062577, -0.017874, -0.00008312, 0.000019849),
                    _ => (1.65, 0.078602, -0.29198, -0.0006537, 0.00036247),
                }
            }
        }
        "S" => {
            if atype0 == 'S' && atype1 == 'H' {
                (1.90, 0.7722, -0.26393, 0.0010629, 0.0002179)
            } else {
                (1.90, 0.54581, -0.19477, -0.0012873, 0.00029247)
            }
        }
        "P" => match non_h_bonds {
            3 => (1.90, 0.3865, -0.18249, -0.0036598, 0.0004264),
            4 => (1.90, 0.03873, -0.0089339, 0.0000083582, 0.0000030381),
            _ => (1.90, 0.3865, -0.18249, -0.0036598, 0.0004264),
        },
        "H" => (0.0, 0.0, 0.0, 0.0, 0.0),
        "MG" => (1.18, 0.49392, -0.16038, -0.00015512, 0.00016453),
        "F" => (1.47, 0.68563, -0.1868, -0.00135573, 0.00023743),
        "Z" => (0.0, 0.0, 0.0, 0.0, 0.0),
        _ => (1.70, 0.51245, -0.15966, -0.00019781, 0.00016392),
    };
    LcpoInfo {
        atom: atom_idx,
        vdw_radius: (vdw + radius_offset).max(0.0),
        p1,
        p2,
        p3,
        p4,
    }
}

fn atom_symbol(system: &System, atom_idx: usize) -> String {
    let atoms = &system.atoms;
    if let Some(&elem_id) = atoms.element_id.get(atom_idx) {
        if let Some(elem) = system.interner.resolve(elem_id) {
            let symbol = elem.trim().to_ascii_uppercase();
            if !symbol.is_empty() && radius_from_symbol(&symbol).is_some() {
                return symbol;
            }
        }
    }
    let atom_type = atom_type_upper(system, atom_idx);
    let two = atom_type.chars().take(2).collect::<String>();
    if radius_from_symbol(&two).is_some() {
        two
    } else {
        atom_type.chars().take(1).collect::<String>()
    }
}

fn atom_type_upper(system: &System, atom_idx: usize) -> String {
    system
        .atoms
        .name_id
        .get(atom_idx)
        .and_then(|id| system.interner.resolve(*id))
        .unwrap_or("")
        .trim()
        .to_ascii_uppercase()
}

fn inferred_bond_count(system: &System, atom_idx: usize, non_h_only: bool) -> usize {
    if let Some(count) = system.bonded_neighbor_count(atom_idx, non_h_only) {
        return count;
    }
    let Some(positions) = system.positions0.as_ref() else {
        return 0;
    };
    if atom_idx >= positions.len() {
        return 0;
    }
    let symbol_i = atom_symbol(system, atom_idx);
    let ri = covalent_radius_from_symbol(&symbol_i).unwrap_or(0.77);
    let pi = positions[atom_idx];
    let mut count = 0usize;
    for (other_idx, pj) in positions.iter().enumerate() {
        if other_idx == atom_idx {
            continue;
        }
        let symbol_j = atom_symbol(system, other_idx);
        if non_h_only && symbol_j == "H" {
            continue;
        }
        let rj = covalent_radius_from_symbol(&symbol_j).unwrap_or(0.77);
        let cutoff = ri + rj + 0.45;
        let dx = (pi[0] - pj[0]) as f64;
        let dy = (pi[1] - pj[1]) as f64;
        let dz = (pi[2] - pj[2]) as f64;
        if dx * dx + dy * dy + dz * dz <= cutoff * cutoff {
            count += 1;
        }
    }
    count
}

fn covalent_radius_from_symbol(symbol: &str) -> Option<f64> {
    match symbol {
        "H" => Some(0.31),
        "C" => Some(0.76),
        "N" => Some(0.71),
        "O" => Some(0.66),
        "S" => Some(1.05),
        "P" => Some(1.07),
        "F" => Some(0.57),
        "CL" => Some(1.02),
        "MG" => Some(1.30),
        _ => None,
    }
}

pub(crate) fn radius_from_symbol(symbol: &str) -> Option<f64> {
    match symbol {
        "H" => Some(1.20),
        "C" => Some(1.70),
        "N" => Some(1.55),
        "O" => Some(1.52),
        "S" => Some(1.80),
        "P" => Some(1.80),
        "F" => Some(1.47),
        "CL" => Some(1.75),
        "MG" => Some(1.18),
        _ => None,
    }
}
