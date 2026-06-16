use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, HydrophobicDefectOutput, Plan, PlanOutput, PlanRequirements};

const DEFAULT_PROBE_RADIUS_A: f64 = 1.4;
const DEFAULT_DEFECT_RADIUS_A: f64 = 7.3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HydrophobicDefectGridMode {
    VoxelCenters,
    LatticeNodes,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HydrophobicDefectLeaflet {
    Both,
    Upper,
    Lower,
}

#[derive(Clone, Copy, Debug)]
struct MembraneCell {
    a: [f64; 2],
    b: [f64; 2],
    inv: [[f64; 2]; 2],
    a_len: f64,
    b_len: f64,
}

pub struct HydrophobicDefectPlan {
    lipid_selection: Selection,
    reference_selection: Selection,
    midplane_selection: Option<Selection>,
    voxel_size: f64,
    z_bounds: Option<[f64; 2]>,
    length_scale: f64,
    probe_radius: f64,
    defect_radius: f64,
    grid_mode: HydrophobicDefectGridMode,
    leaflet: HydrophobicDefectLeaflet,
    leaflet_bins: usize,
    atom_radii: Vec<f64>,
    dims: [usize; 3],
    resolved_z_bounds: [f64; 2],
    sum: Vec<f64>,
    first: Vec<u32>,
    last: Vec<u32>,
    min: Vec<u32>,
    max: Vec<u32>,
    frame_counts: Vec<u32>,
    frame_cluster_count: Vec<u32>,
    frame_largest_cluster: Vec<u32>,
    current_lifetime: Vec<u32>,
    max_lifetime: Vec<u32>,
    frames: usize,
}

impl HydrophobicDefectPlan {
    pub fn new(
        lipid_selection: Selection,
        reference_selection: Selection,
        voxel_size: f64,
        z_bounds: Option<[f64; 2]>,
    ) -> Self {
        Self {
            lipid_selection,
            reference_selection,
            midplane_selection: None,
            voxel_size,
            z_bounds,
            length_scale: 1.0,
            probe_radius: DEFAULT_PROBE_RADIUS_A,
            defect_radius: DEFAULT_DEFECT_RADIUS_A,
            grid_mode: HydrophobicDefectGridMode::VoxelCenters,
            leaflet: HydrophobicDefectLeaflet::Both,
            leaflet_bins: 1,
            atom_radii: Vec::new(),
            dims: [1, 1, 1],
            resolved_z_bounds: [0.0, 1.0],
            sum: Vec::new(),
            first: Vec::new(),
            last: Vec::new(),
            min: Vec::new(),
            max: Vec::new(),
            frame_counts: Vec::new(),
            frame_cluster_count: Vec::new(),
            frame_largest_cluster: Vec::new(),
            current_lifetime: Vec::new(),
            max_lifetime: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    pub fn with_probe_radius(mut self, probe_radius: f64) -> Self {
        self.probe_radius = probe_radius;
        self
    }

    pub fn with_defect_radius(mut self, defect_radius: f64) -> Self {
        self.defect_radius = defect_radius;
        self
    }

    pub fn with_grid_mode(mut self, grid_mode: HydrophobicDefectGridMode) -> Self {
        self.grid_mode = grid_mode;
        self
    }

    pub fn with_leaflet(mut self, leaflet: HydrophobicDefectLeaflet) -> Self {
        self.leaflet = leaflet;
        self
    }

    pub fn with_midplane_selection(mut self, selection: Selection) -> Self {
        self.midplane_selection = Some(selection);
        self
    }

    pub fn with_leaflet_bins(mut self, bins: usize) -> Self {
        self.leaflet_bins = bins.max(1);
        self
    }

    fn infer_z_bounds(&self, system: &System) -> TrajResult<[f64; 2]> {
        if let Some(bounds) = self.z_bounds {
            return Ok(bounds);
        }
        let positions = system.positions0.as_ref().ok_or_else(|| {
            TrajError::Mismatch(
                "hydrophobic_defects requires z_bounds when topology has no initial coordinates"
                    .into(),
            )
        })?;
        let mut z_min = f64::INFINITY;
        let mut z_max = f64::NEG_INFINITY;
        for &idx in self
            .lipid_selection
            .indices
            .iter()
            .chain(self.reference_selection.indices.iter())
        {
            let pos = positions.get(idx as usize).ok_or_else(|| {
                TrajError::Mismatch("hydrophobic_defects selection index out of bounds".into())
            })?;
            let z = pos[2] as f64 * self.length_scale;
            z_min = z_min.min(z);
            z_max = z_max.max(z);
        }
        if !z_min.is_finite() || !z_max.is_finite() {
            return Err(TrajError::Mismatch(
                "hydrophobic_defects selections are empty".into(),
            ));
        }
        let pad = (self.probe_radius + self.voxel_size).max(self.voxel_size);
        Ok([z_min - pad, z_max + pad])
    }

    fn radius_for_atom(system: &System, atom_idx: usize) -> f64 {
        if let Some(radius) = system.vdw_radius_for_atom(atom_idx) {
            if radius.is_finite() && radius > 0.0 {
                return radius as f64;
            }
        }
        match system.atom_symbol(atom_idx).as_str() {
            "H" => 1.20,
            "C" => 1.70,
            "N" => 1.55,
            "O" => 1.52,
            "P" => 1.80,
            "S" => 1.80,
            _ => 1.70,
        }
    }

    fn sample_offset(&self) -> f64 {
        match self.grid_mode {
            HydrophobicDefectGridMode::VoxelCenters => 0.5,
            HydrophobicDefectGridMode::LatticeNodes => 0.0,
        }
    }

    fn z_dim(&self, bounds: [f64; 2]) -> usize {
        let base = ((bounds[1] - bounds[0]) / self.voxel_size).ceil().max(1.0) as usize;
        base + usize::from(self.grid_mode == HydrophobicDefectGridMode::LatticeNodes)
    }

    fn flat(&self, ix: usize, iy: usize, iz: usize) -> usize {
        ix + self.dims[0] * (iy + self.dims[1] * iz)
    }

    fn cell_z(&self, iz: usize) -> f64 {
        self.resolved_z_bounds[0] + (iz as f64 + self.sample_offset()) * self.voxel_size
    }

    fn atom_z(chunk: &FrameChunk, frame: usize, atom: usize, scale: f64) -> f64 {
        chunk.coords[frame * chunk.n_atoms + atom][2] as f64 * scale
    }

    fn mean_z_for_selection(
        chunk: &FrameChunk,
        frame: usize,
        selection: &Selection,
        scale: f64,
    ) -> TrajResult<f64> {
        if selection.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "hydrophobic_defects midplane selection is empty".into(),
            ));
        }
        let mut sum = 0.0;
        for &idx in selection.indices.iter() {
            sum += Self::atom_z(chunk, frame, idx as usize, scale);
        }
        Ok(sum / selection.indices.len() as f64)
    }

    fn frame_midplane(
        &self,
        chunk: &FrameChunk,
        frame: usize,
        cell: MembraneCell,
    ) -> TrajResult<Option<Vec<f64>>> {
        if self.leaflet == HydrophobicDefectLeaflet::Both {
            return Ok(None);
        }
        let selection = self
            .midplane_selection
            .as_ref()
            .unwrap_or(&self.lipid_selection);
        if self.leaflet_bins <= 1 {
            return Self::mean_z_for_selection(chunk, frame, selection, self.length_scale)
                .map(|mid| Some(vec![mid]));
        }
        let bins = self.leaflet_bins;
        let mut sum = vec![0.0; bins * bins];
        let mut count = vec![0usize; bins * bins];
        for &idx in selection.indices.iter() {
            let atom = idx as usize;
            let p = chunk.coords[frame * chunk.n_atoms + atom];
            let xy = Self::wrap_position(
                cell,
                [
                    p[0] as f64 * self.length_scale,
                    p[1] as f64 * self.length_scale,
                ],
            );
            let frac = Self::position_to_fractional(cell, xy);
            let ix = ((frac[0].rem_euclid(1.0) * bins as f64).floor() as usize).min(bins - 1);
            let iy = ((frac[1].rem_euclid(1.0) * bins as f64).floor() as usize).min(bins - 1);
            let flat = ix * bins + iy;
            sum[flat] += p[2] as f64 * self.length_scale;
            count[flat] += 1;
        }
        let total_count: usize = count.iter().sum();
        if total_count == 0 {
            return Err(TrajError::Mismatch(
                "hydrophobic_defects midplane selection is empty".into(),
            ));
        }
        let global = sum.iter().sum::<f64>() / total_count as f64;
        for idx in 0..sum.len() {
            sum[idx] = if count[idx] == 0 {
                global
            } else {
                sum[idx] / count[idx] as f64
            };
        }
        Ok(Some(sum))
    }

    fn midplane_for_xy(
        &self,
        midplane: Option<&[f64]>,
        cell: MembraneCell,
        xy: [f64; 2],
    ) -> Option<f64> {
        let midplane = midplane?;
        if self.leaflet_bins <= 1 {
            return midplane.first().copied();
        }
        let bins = self.leaflet_bins;
        let frac = Self::position_to_fractional(cell, xy);
        let ix = ((frac[0].rem_euclid(1.0) * bins as f64).floor() as usize).min(bins - 1);
        let iy = ((frac[1].rem_euclid(1.0) * bins as f64).floor() as usize).min(bins - 1);
        midplane.get(ix * bins + iy).copied()
    }

    fn include_leaflet(&self, z: f64, midplane: Option<f64>) -> bool {
        match (self.leaflet, midplane) {
            (HydrophobicDefectLeaflet::Both, _) | (_, None) => true,
            (HydrophobicDefectLeaflet::Upper, Some(mid)) => z >= mid,
            (HydrophobicDefectLeaflet::Lower, Some(mid)) => z <= mid,
        }
    }

    fn cluster_summary(&self, defect_mask: &[u32]) -> (u32, u32) {
        let mut visited = vec![false; defect_mask.len()];
        let mut clusters = 0u32;
        let mut largest = 0u32;
        let mut stack = Vec::new();
        for idx in 0..defect_mask.len() {
            if defect_mask[idx] == 0 || visited[idx] {
                continue;
            }
            clusters += 1;
            visited[idx] = true;
            stack.push(idx);
            let mut size = 0u32;
            while let Some(flat) = stack.pop() {
                size += 1;
                let ix = flat % self.dims[0];
                let iy = (flat / self.dims[0]) % self.dims[1];
                let iz = flat / (self.dims[0] * self.dims[1]);
                let neighbors = [
                    ((ix + self.dims[0] - 1) % self.dims[0], iy, iz),
                    ((ix + 1) % self.dims[0], iy, iz),
                    (ix, (iy + self.dims[1] - 1) % self.dims[1], iz),
                    (ix, (iy + 1) % self.dims[1], iz),
                    (ix, iy, iz.saturating_sub(1)),
                    (ix, iy, (iz + 1).min(self.dims[2] - 1)),
                ];
                for (nx, ny, nz) in neighbors {
                    let nflat = self.flat(nx, ny, nz);
                    if defect_mask[nflat] != 0 && !visited[nflat] {
                        visited[nflat] = true;
                        stack.push(nflat);
                    }
                }
            }
            largest = largest.max(size);
        }
        (clusters, largest)
    }

    fn position_to_fractional(cell: MembraneCell, position: [f64; 2]) -> [f64; 2] {
        [
            cell.inv[0][0] * position[0] + cell.inv[0][1] * position[1],
            cell.inv[1][0] * position[0] + cell.inv[1][1] * position[1],
        ]
    }

    fn wrap_position(cell: MembraneCell, position: [f64; 2]) -> [f64; 2] {
        let frac = Self::position_to_fractional(cell, position);
        let fx = frac[0].rem_euclid(1.0);
        let fy = frac[1].rem_euclid(1.0);
        [
            fx * cell.a[0] + fy * cell.b[0],
            fx * cell.a[1] + fy * cell.b[1],
        ]
    }

    fn grid_xy(&self, cell: MembraneCell, ix: usize, iy: usize) -> [f64; 2] {
        let fx = ((ix as f64 + self.sample_offset()) * self.voxel_size) / cell.a_len;
        let fy = ((iy as f64 + self.sample_offset()) * self.voxel_size) / cell.b_len;
        [
            fx * cell.a[0] + fy * cell.b[0],
            fx * cell.a[1] + fy * cell.b[1],
        ]
    }

    fn fractional_index(&self, value: f64, length: f64) -> isize {
        let scaled = value.rem_euclid(1.0) * length / self.voxel_size - self.sample_offset();
        scaled.round() as isize
    }

    fn pbc_delta_xy(cell: MembraneCell, point: [f64; 2], center: [f64; 2]) -> [f64; 2] {
        let point_frac = Self::position_to_fractional(cell, point);
        let center_frac = Self::position_to_fractional(cell, center);
        let dfx = point_frac[0] - center_frac[0] - (point_frac[0] - center_frac[0]).round();
        let dfy = point_frac[1] - center_frac[1] - (point_frac[1] - center_frac[1]).round();
        [
            dfx * cell.a[0] + dfy * cell.b[0],
            dfx * cell.a[1] + dfy * cell.b[1],
        ]
    }

    fn mark_occupied(&self, mask: &mut [u32], center: [f64; 3], radius: f64, cell: MembraneCell) {
        if self.grid_mode == HydrophobicDefectGridMode::LatticeNodes {
            self.mark_occupied_clamped_lattice(mask, center, radius);
            return;
        }
        let r2 = radius * radius;
        let r_cells = (radius / self.voxel_size).ceil() as isize;
        let frac = Self::position_to_fractional(cell, [center[0], center[1]]);
        let ix0 = self.fractional_index(frac[0], cell.a_len);
        let iy0 = self.fractional_index(frac[1], cell.b_len);
        let iz0 = ((center[2] - self.resolved_z_bounds[0]) / self.voxel_size).floor() as isize;

        for dx_i in -r_cells..=r_cells {
            let ix = (ix0 + dx_i).rem_euclid(self.dims[0] as isize) as usize;
            for dy_i in -r_cells..=r_cells {
                let iy = (iy0 + dy_i).rem_euclid(self.dims[1] as isize) as usize;
                let xy = self.grid_xy(cell, ix, iy);
                let delta = Self::pbc_delta_xy(cell, xy, [center[0], center[1]]);
                let dxy2 = delta[0] * delta[0] + delta[1] * delta[1];
                if dxy2 > r2 {
                    continue;
                }
                for iz_i in (iz0 - r_cells).max(0)..=(iz0 + r_cells).min(self.dims[2] as isize - 1)
                {
                    let iz = iz_i as usize;
                    let dz = self.cell_z(iz) - center[2];
                    if dxy2 + dz * dz <= r2 {
                        let flat = self.flat(ix, iy, iz);
                        mask[flat] = 0;
                    }
                }
            }
        }
    }

    fn clamped_lattice_range(&self, center: f64, radius: f64, dim: usize) -> (usize, usize) {
        let lo = ((center - radius) / self.voxel_size).trunc().max(0.0) as usize;
        let hi = ((center + radius) / self.voxel_size)
            .trunc()
            .min(dim as f64 - 1.0)
            .max(0.0) as usize;
        (lo.min(hi), hi)
    }

    fn mark_occupied_clamped_lattice(&self, mask: &mut [u32], center: [f64; 3], radius: f64) {
        let r2 = radius * radius;
        let (lo_x, hi_x) = self.clamped_lattice_range(center[0], radius, self.dims[0]);
        let (lo_y, hi_y) = self.clamped_lattice_range(center[1], radius, self.dims[1]);
        let (lo_z, hi_z) =
            self.clamped_lattice_range(center[2] - self.resolved_z_bounds[0], radius, self.dims[2]);
        for ix in lo_x..=hi_x {
            let dx = ix as f64 * self.voxel_size - center[0];
            for iy in lo_y..=hi_y {
                let dy = iy as f64 * self.voxel_size - center[1];
                let dxy2 = dx * dx + dy * dy;
                if dxy2 > r2 {
                    continue;
                }
                for iz in lo_z..=hi_z {
                    let dz = self.cell_z(iz) - center[2];
                    if dxy2 + dz * dz <= r2 {
                        let flat = self.flat(ix, iy, iz);
                        mask[flat] = 0;
                    }
                }
            }
        }
    }

    fn mark_defects(
        &self,
        free_mask: &[u32],
        defect_mask: &mut [u32],
        reference: [f64; 3],
        cell: MembraneCell,
    ) {
        if self.grid_mode == HydrophobicDefectGridMode::LatticeNodes {
            self.mark_defects_clamped_lattice(free_mask, defect_mask, reference);
            return;
        }
        let r2 = self.defect_radius * self.defect_radius;
        let r_cells = (self.defect_radius / self.voxel_size).ceil() as isize;
        let frac = Self::position_to_fractional(cell, [reference[0], reference[1]]);
        let ix0 = self.fractional_index(frac[0], cell.a_len);
        let iy0 = self.fractional_index(frac[1], cell.b_len);
        let iz_max =
            ((reference[2] - self.resolved_z_bounds[0]) / self.voxel_size).floor() as isize;
        if iz_max < 0 {
            return;
        }
        for dx_i in -r_cells..=r_cells {
            let ix = (ix0 + dx_i).rem_euclid(self.dims[0] as isize) as usize;
            for dy_i in -r_cells..=r_cells {
                let iy = (iy0 + dy_i).rem_euclid(self.dims[1] as isize) as usize;
                let xy = self.grid_xy(cell, ix, iy);
                let delta = Self::pbc_delta_xy(cell, xy, [reference[0], reference[1]]);
                if delta[0] * delta[0] + delta[1] * delta[1] >= r2 {
                    continue;
                }
                for iz_i in 0..=iz_max.min(self.dims[2] as isize - 1) {
                    let iz = iz_i as usize;
                    let flat = self.flat(ix, iy, iz);
                    if self.cell_z(iz) < reference[2] && free_mask[flat] != 0 {
                        defect_mask[flat] = 1;
                    }
                }
            }
        }
    }

    fn mark_defects_clamped_lattice(
        &self,
        free_mask: &[u32],
        defect_mask: &mut [u32],
        reference: [f64; 3],
    ) {
        let r2 = self.defect_radius * self.defect_radius;
        let (lo_x, hi_x) =
            self.clamped_lattice_range(reference[0], self.defect_radius, self.dims[0]);
        let (lo_y, hi_y) =
            self.clamped_lattice_range(reference[1], self.defect_radius, self.dims[1]);
        let (lo_z, hi_z) = self.clamped_lattice_range(
            reference[2] - self.resolved_z_bounds[0],
            self.defect_radius,
            self.dims[2],
        );
        for ix in lo_x..=hi_x {
            let dx = ix as f64 * self.voxel_size - reference[0];
            for iy in lo_y..=hi_y {
                let dy = iy as f64 * self.voxel_size - reference[1];
                if dx * dx + dy * dy >= r2 {
                    continue;
                }
                for iz in lo_z..=hi_z {
                    let flat = self.flat(ix, iy, iz);
                    if self.cell_z(iz) < reference[2] && free_mask[flat] != 0 {
                        defect_mask[flat] = 1;
                    }
                }
            }
        }
    }

    fn membrane_cell(box_: Box3, length_scale: f64) -> TrajResult<MembraneCell> {
        match box_ {
            Box3::Orthorhombic { lx, ly, .. } => {
                let a_len = lx as f64 * length_scale;
                let b_len = ly as f64 * length_scale;
                if a_len <= 0.0 || b_len <= 0.0 {
                    return Err(TrajError::Mismatch(
                        "hydrophobic_defects requires positive XY box lengths".into(),
                    ));
                }
                Ok(MembraneCell {
                    a: [a_len, 0.0],
                    b: [0.0, b_len],
                    inv: [[1.0 / a_len, 0.0], [0.0, 1.0 / b_len]],
                    a_len,
                    b_len,
                })
            }
            Box3::Triclinic { m } => {
                let a = [m[0] as f64 * length_scale, m[1] as f64 * length_scale];
                let b = [m[3] as f64 * length_scale, m[4] as f64 * length_scale];
                let det = a[0] * b[1] - a[1] * b[0];
                if det.abs() < 1.0e-12 {
                    return Err(TrajError::Mismatch(
                        "hydrophobic_defects triclinic XY cell is singular".into(),
                    ));
                }
                let a_len = (a[0] * a[0] + a[1] * a[1]).sqrt();
                let b_len = (b[0] * b[0] + b[1] * b[1]).sqrt();
                if a_len <= 0.0 || b_len <= 0.0 {
                    return Err(TrajError::Mismatch(
                        "hydrophobic_defects requires positive XY box lengths".into(),
                    ));
                }
                Ok(MembraneCell {
                    a,
                    b,
                    inv: [[b[1] / det, -b[0] / det], [-a[1] / det, a[0] / det]],
                    a_len,
                    b_len,
                })
            }
            Box3::None => Err(TrajError::Mismatch(
                "hydrophobic_defects requires box metadata".into(),
            )),
        }
    }
}

impl Plan for HydrophobicDefectPlan {
    fn name(&self) -> &'static str {
        "hydrophobic_defects"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(true, false)
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.lipid_selection.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "hydrophobic_defects lipid selection is empty".into(),
            ));
        }
        if self.reference_selection.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "hydrophobic_defects reference selection is empty".into(),
            ));
        }
        if let Some(selection) = &self.midplane_selection {
            if selection.indices.is_empty() {
                return Err(TrajError::Mismatch(
                    "hydrophobic_defects midplane selection is empty".into(),
                ));
            }
        }
        for (name, value) in [
            ("voxel_size", self.voxel_size),
            ("length_scale", self.length_scale),
            ("probe_radius", self.probe_radius),
            ("defect_radius", self.defect_radius),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(TrajError::Mismatch(format!(
                    "hydrophobic_defects {name} must be finite and > 0"
                )));
            }
        }
        let bounds = self.infer_z_bounds(system)?;
        if !bounds[0].is_finite() || !bounds[1].is_finite() || bounds[1] <= bounds[0] {
            return Err(TrajError::Mismatch(
                "hydrophobic_defects z_bounds must be finite and increasing".into(),
            ));
        }
        self.resolved_z_bounds = bounds;
        self.dims = [1, 1, self.z_dim(bounds)];
        self.atom_radii = (0..system.n_atoms())
            .map(|idx| Self::radius_for_atom(system, idx))
            .collect();
        self.sum.clear();
        self.first.clear();
        self.last.clear();
        self.min.clear();
        self.max.clear();
        self.frame_counts.clear();
        self.frame_cluster_count.clear();
        self.frame_largest_cluster.clear();
        self.current_lifetime.clear();
        self.max_lifetime.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if chunk.box_.len() != chunk.n_frames {
            return Err(TrajError::Mismatch(
                "hydrophobic_defects requires box metadata for every frame".into(),
            ));
        }
        for frame in 0..chunk.n_frames {
            let cell = Self::membrane_cell(chunk.box_[frame], self.length_scale)?;
            let extra_node = usize::from(self.grid_mode == HydrophobicDefectGridMode::LatticeNodes);
            let dims = [
                (cell.a_len / self.voxel_size).floor().max(1.0) as usize + extra_node,
                (cell.b_len / self.voxel_size).floor().max(1.0) as usize + extra_node,
                self.dims[2],
            ];
            let n_cells = dims[0]
                .checked_mul(dims[1])
                .and_then(|v| v.checked_mul(dims[2]))
                .ok_or_else(|| TrajError::Mismatch("hydrophobic_defects grid too large".into()))?;
            if self.frames == 0 {
                self.dims = dims;
                self.sum = vec![0.0; n_cells];
                self.first = vec![0; n_cells];
                self.last = vec![0; n_cells];
                self.min = vec![u32::MAX; n_cells];
                self.max = vec![0; n_cells];
                self.current_lifetime = vec![0; n_cells];
                self.max_lifetime = vec![0; n_cells];
            } else if dims != self.dims {
                return Err(TrajError::Mismatch(
                    "hydrophobic_defects box dimensions changed grid shape across frames".into(),
                ));
            }

            let mut free_mask = vec![1u32; n_cells];
            let mut defect_mask = vec![0u32; n_cells];
            let offset = frame * chunk.n_atoms;
            let midplane = self.frame_midplane(chunk, frame, cell)?;
            for &idx in self.lipid_selection.indices.iter() {
                let atom_idx = idx as usize;
                let p = chunk.coords[offset + atom_idx];
                let z = p[2] as f64 * self.length_scale;
                let xy = Self::wrap_position(
                    cell,
                    [
                        p[0] as f64 * self.length_scale,
                        p[1] as f64 * self.length_scale,
                    ],
                );
                if !self.include_leaflet(z, self.midplane_for_xy(midplane.as_deref(), cell, xy)) {
                    continue;
                }
                let center = [xy[0], xy[1], z];
                let radius = self.atom_radii[atom_idx] + self.probe_radius;
                self.mark_occupied(&mut free_mask, center, radius, cell);
            }
            for &idx in self.reference_selection.indices.iter() {
                let atom_idx = idx as usize;
                let p = chunk.coords[offset + atom_idx];
                let z = p[2] as f64 * self.length_scale;
                let xy = Self::wrap_position(
                    cell,
                    [
                        p[0] as f64 * self.length_scale,
                        p[1] as f64 * self.length_scale,
                    ],
                );
                if !self.include_leaflet(z, self.midplane_for_xy(midplane.as_deref(), cell, xy)) {
                    continue;
                }
                let reference = [xy[0], xy[1], z];
                self.mark_defects(&free_mask, &mut defect_mask, reference, cell);
            }
            let count = defect_mask.iter().copied().sum::<u32>();
            let (cluster_count, largest_cluster) = self.cluster_summary(&defect_mask);
            if self.frames == 0 {
                self.first.copy_from_slice(&defect_mask);
            }
            self.last.copy_from_slice(&defect_mask);
            for (i, &value) in defect_mask.iter().enumerate() {
                self.sum[i] += value as f64;
                self.min[i] = self.min[i].min(value);
                self.max[i] = self.max[i].max(value);
                if value != 0 {
                    self.current_lifetime[i] += 1;
                    self.max_lifetime[i] = self.max_lifetime[i].max(self.current_lifetime[i]);
                } else {
                    self.current_lifetime[i] = 0;
                }
            }
            self.frame_counts.push(count);
            self.frame_cluster_count.push(cluster_count);
            self.frame_largest_cluster.push(largest_cluster);
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.frames == 0 {
            return Err(TrajError::Mismatch("no frames processed".into()));
        }
        let frames_f = self.frames as f64;
        let mean = self
            .sum
            .iter()
            .map(|value| (value / frames_f) as f32)
            .collect();
        let voxel_volume = self.voxel_size * self.voxel_size * self.voxel_size;
        let voxel_area = self.voxel_size * self.voxel_size;
        let frame_area = self
            .frame_counts
            .iter()
            .map(|count| (*count as f64 * voxel_area) as f32)
            .collect();
        let frame_volume = self
            .frame_counts
            .iter()
            .map(|count| (*count as f64 * voxel_volume) as f32)
            .collect();
        Ok(PlanOutput::HydrophobicDefect(HydrophobicDefectOutput {
            dims: self.dims,
            voxel_size: self.voxel_size as f32,
            z_bounds: [
                self.resolved_z_bounds[0] as f32,
                self.resolved_z_bounds[1] as f32,
            ],
            mean,
            first: self.first.clone(),
            last: self.last.clone(),
            min: self
                .min
                .iter()
                .map(|v| if *v == u32::MAX { 0 } else { *v })
                .collect(),
            max: self.max.clone(),
            frame_counts: self.frame_counts.clone(),
            frame_area,
            frame_volume,
            frame_cluster_count: self.frame_cluster_count.clone(),
            frame_largest_cluster: self.frame_largest_cluster.clone(),
            max_lifetime: self.max_lifetime.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use traj_core::frame::FrameChunkBuilder;
    use traj_core::interner::StringInterner;
    use traj_core::system::AtomTable;

    fn test_system() -> System {
        let mut interner = StringInterner::new();
        let lipid = interner.intern_upper("C1");
        let reference = interner.intern_upper("R1");
        let res = interner.intern_upper("MEM");
        let carbon = interner.intern_upper("C");
        let atoms = AtomTable {
            name_id: vec![lipid, reference],
            resname_id: vec![res, res],
            resid: vec![1, 1],
            chain_id: vec![0, 0],
            element_id: vec![carbon, carbon],
            mass: vec![12.0, 12.0],
        };
        let positions0 = Some(vec![[5.0, 5.0, 0.0, 1.0], [5.0, 5.0, 2.0, 1.0]]);
        System::with_atoms(atoms, interner, positions0)
    }

    #[test]
    fn hydrophobic_defects_reports_free_voxels_below_reference() {
        let mut system = test_system();
        let lipid_sel = system.select("name C1").unwrap();
        let ref_sel = system.select("name R1").unwrap();
        let mut plan = HydrophobicDefectPlan::new(lipid_sel, ref_sel, 1.0, Some([0.0, 3.0]))
            .with_probe_radius(0.1)
            .with_defect_radius(1.1);
        let mut builder = FrameChunkBuilder::new(2, 1);
        let coords = builder.start_frame(
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            None,
        );
        coords.copy_from_slice(&[[0.0, 0.0, 0.0, 1.0], [5.0, 5.0, 2.0, 1.0]]);
        let chunk = builder.finish().unwrap();
        plan.init(&system, &Device::Cpu).unwrap();
        plan.process_chunk(&chunk, &system, &Device::Cpu).unwrap();
        match plan.finalize().unwrap() {
            PlanOutput::HydrophobicDefect(out) => {
                assert_eq!(out.dims, [10, 10, 3]);
                assert_eq!(out.frame_counts.len(), 1);
                assert!(out.frame_counts[0] > 0);
                assert_eq!(out.frame_volume[0], out.frame_counts[0] as f32);
                assert!(out.mean.iter().any(|value| *value > 0.0));
            }
            _ => panic!("unexpected output"),
        }
    }

    #[test]
    fn hydrophobic_defects_supports_lattice_nodes() {
        let mut system = test_system();
        let lipid_sel = system.select("name C1").unwrap();
        let ref_sel = system.select("name R1").unwrap();
        let mut plan = HydrophobicDefectPlan::new(lipid_sel, ref_sel, 1.0, Some([0.0, 3.0]))
            .with_grid_mode(HydrophobicDefectGridMode::LatticeNodes)
            .with_probe_radius(0.1)
            .with_defect_radius(1.1);
        let mut builder = FrameChunkBuilder::new(2, 1);
        let coords = builder.start_frame(
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 10.0,
                lz: 10.0,
            },
            None,
        );
        coords.copy_from_slice(&[[0.0, 0.0, 0.0, 1.0], [5.0, 5.0, 2.0, 1.0]]);
        let chunk = builder.finish().unwrap();
        plan.init(&system, &Device::Cpu).unwrap();
        plan.process_chunk(&chunk, &system, &Device::Cpu).unwrap();
        match plan.finalize().unwrap() {
            PlanOutput::HydrophobicDefect(out) => {
                assert_eq!(out.dims, [11, 11, 4]);
                assert!(out.frame_counts[0] > 0);
            }
            _ => panic!("unexpected output"),
        }
    }

    #[test]
    fn hydrophobic_defects_accepts_triclinic_xy_cell() {
        let mut system = test_system();
        let lipid_sel = system.select("name C1").unwrap();
        let ref_sel = system.select("name R1").unwrap();
        let mut plan = HydrophobicDefectPlan::new(lipid_sel, ref_sel, 1.0, Some([0.0, 3.0]))
            .with_probe_radius(0.1)
            .with_defect_radius(1.1);
        let mut builder = FrameChunkBuilder::new(2, 1);
        let coords = builder.start_frame(
            Box3::Triclinic {
                m: [10.0, 0.0, 0.0, 2.0, 10.0, 0.0, 0.0, 0.0, 10.0],
            },
            None,
        );
        coords.copy_from_slice(&[[0.0, 0.0, 0.0, 1.0], [5.0, 5.0, 2.0, 1.0]]);
        let chunk = builder.finish().unwrap();
        plan.init(&system, &Device::Cpu).unwrap();
        plan.process_chunk(&chunk, &system, &Device::Cpu).unwrap();
        assert!(matches!(
            plan.finalize().unwrap(),
            PlanOutput::HydrophobicDefect(_)
        ));
    }

    #[test]
    fn hydrophobic_defects_requires_box_metadata() {
        let mut system = test_system();
        let lipid_sel = system.select("name C1").unwrap();
        let ref_sel = system.select("name R1").unwrap();
        let mut plan = HydrophobicDefectPlan::new(lipid_sel, ref_sel, 1.0, Some([0.0, 3.0]));
        let mut builder = FrameChunkBuilder::new(2, 1);
        let coords = builder.start_frame(Box3::None, None);
        coords.copy_from_slice(&[[0.0, 0.0, 0.0, 1.0], [5.0, 5.0, 2.0, 1.0]]);
        let chunk = builder.finish().unwrap();
        plan.init(&system, &Device::Cpu).unwrap();
        let err = plan
            .process_chunk(&chunk, &system, &Device::Cpu)
            .unwrap_err();
        assert!(err.to_string().contains("box metadata"));
    }
}
