use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, GridOutput, Plan, PlanOutput};

const DEFAULT_BOX_UNIT: f64 = 1.0;

pub struct FreeVolumePlan {
    selection: Selection,
    center_selection: Selection,
    box_unit: Option<[f64; 3]>,
    region_size: Option<[f64; 3]>,
    shift: [f64; 3],
    shift_explicit: bool,
    length_scale: f64,
    probe_radius: f64,
    dims: [usize; 3],
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    first: Vec<u32>,
    last: Vec<u32>,
    min: Vec<u32>,
    max: Vec<u32>,
    frames: usize,
}

impl FreeVolumePlan {
    pub fn new(
        selection: Selection,
        center_selection: Selection,
        box_unit: Option<[f64; 3]>,
        region_size: Option<[f64; 3]>,
    ) -> Self {
        let n_cells = 1; // Will be computed in init()
        Self {
            selection,
            center_selection,
            box_unit,
            region_size,
            shift: [0.0, 0.0, 0.0],
            shift_explicit: false,
            length_scale: 1.0,
            probe_radius: 0.0,
            dims: [1, 1, 1],
            sum: vec![0.0; n_cells],
            sum_sq: vec![0.0; n_cells],
            first: vec![0; n_cells],
            last: vec![0; n_cells],
            min: vec![0; n_cells],
            max: vec![0; n_cells],
            frames: 0,
        }
    }

    /// Compute bounding box from selection positions
    fn compute_bounding_box(
        system: &System,
        selection: &Selection,
    ) -> Option<(f64, f64, f64, f64, f64, f64)> {
        let positions = system.positions0.as_ref()?;
        if positions.is_empty() {
            return None;
        }

        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut min_z = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;
        let mut max_z = f64::MIN;
        let mut found = false;

        for &idx in selection.indices.iter() {
            let pos = positions[idx as usize];
            let x = pos[0] as f64;
            let y = pos[1] as f64;
            let z = pos[2] as f64;

            min_x = min_x.min(x);
            min_y = min_y.min(y);
            min_z = min_z.min(z);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            max_z = max_z.max(z);
            found = true;
        }

        if !found {
            return None;
        }

        // Add 10% padding on each side
        let padding_x = (max_x - min_x) * 0.1;
        let padding_y = (max_y - min_y) * 0.1;
        let padding_z = (max_z - min_z) * 0.1;

        Some((
            min_x - padding_x,
            min_y - padding_y,
            min_z - padding_z,
            max_x + padding_x,
            max_y + padding_y,
            max_z + padding_z,
        ))
    }

    pub fn with_shift(mut self, shift: [f64; 3]) -> Self {
        self.shift = shift;
        self.shift_explicit = true;
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }

    pub fn with_probe_radius(mut self, probe_radius: f64) -> Self {
        self.probe_radius = probe_radius.max(0.0);
        self
    }

    fn mark_occupied_with_probe(
        &self,
        box_unit: [f64; 3],
        x: f64,
        y: f64,
        z: f64,
        ix: usize,
        iy: usize,
        iz: usize,
        free_mask: &mut [u32],
    ) {
        if self.probe_radius <= 0.0 {
            let flat = ix + self.dims[0] * (iy + self.dims[1] * iz);
            free_mask[flat] = 0;
            return;
        }

        let rx = (self.probe_radius / box_unit[0]).ceil() as isize;
        let ry = (self.probe_radius / box_unit[1]).ceil() as isize;
        let rz = (self.probe_radius / box_unit[2]).ceil() as isize;
        let r2 = self.probe_radius * self.probe_radius;

        let x0 = ix as isize - rx;
        let x1 = ix as isize + rx;
        let y0 = iy as isize - ry;
        let y1 = iy as isize + ry;
        let z0 = iz as isize - rz;
        let z1 = iz as isize + rz;

        for nx in x0.max(0)..=x1.min(self.dims[0] as isize - 1) {
            let cx = (nx as f64 + 0.5) * box_unit[0];
            let dx = cx - x;
            let dx2 = dx * dx;
            if dx2 > r2 {
                continue;
            }
            for ny in y0.max(0)..=y1.min(self.dims[1] as isize - 1) {
                let cy = (ny as f64 + 0.5) * box_unit[1];
                let dy = cy - y;
                let dxy2 = dx2 + dy * dy;
                if dxy2 > r2 {
                    continue;
                }
                for nz in z0.max(0)..=z1.min(self.dims[2] as isize - 1) {
                    let cz = (nz as f64 + 0.5) * box_unit[2];
                    let dz = cz - z;
                    if dxy2 + dz * dz <= r2 {
                        let flat =
                            nx as usize + self.dims[0] * (ny as usize + self.dims[1] * nz as usize);
                        free_mask[flat] = 0;
                    }
                }
            }
        }
    }
}

impl Plan for FreeVolumePlan {
    fn name(&self) -> &'static str {
        "free_volume"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        // Auto-detect box_unit (default to 1.0 Å)
        let box_unit = self.box_unit.unwrap_or([DEFAULT_BOX_UNIT; 3]);

        // Auto-detect region_size from selection bounding box
        let region_size = if let Some(size) = self.region_size {
            size
        } else {
            // Compute bounding box from center_selection
            match Self::compute_bounding_box(system, &self.center_selection) {
                Some((min_x, min_y, min_z, max_x, max_y, max_z)) => {
                    let size = [max_x - min_x, max_y - min_y, max_z - min_z];
                    // Ensure minimum size of 1.0 Å in each dimension
                    [size[0].max(1.0), size[1].max(1.0), size[2].max(1.0)]
                }
                None => {
                    return Err(TrajError::Mismatch(
                        "free_volume could not auto-detect region_size: no positions available. \
                         Please provide region_size explicitly or ensure the system has initial coordinates.".into()
                    ));
                }
            }
        };

        // Validate box_unit
        if box_unit[0] <= 0.0 || box_unit[1] <= 0.0 || box_unit[2] <= 0.0 {
            return Err(TrajError::Mismatch(
                "free_volume requires positive box_unit".into(),
            ));
        }

        // Validate region_size
        if region_size[0] <= 0.0 || region_size[1] <= 0.0 || region_size[2] <= 0.0 {
            return Err(TrajError::Mismatch(
                "free_volume requires positive region_size".into(),
            ));
        }

        // Compute grid dimensions
        let dims = [
            (region_size[0] / box_unit[0]).floor().max(1.0) as usize + 1,
            (region_size[1] / box_unit[1]).floor().max(1.0) as usize + 1,
            (region_size[2] / box_unit[2]).floor().max(1.0) as usize + 1,
        ];

        let n_cells = dims[0] * dims[1] * dims[2];

        // Store computed values
        self.box_unit = Some(box_unit);
        self.region_size = Some(region_size);
        if !self.shift_explicit {
            // By default the free-volume region is centered on center_selection.
            self.shift = [
                region_size[0] * 0.5,
                region_size[1] * 0.5,
                region_size[2] * 0.5,
            ];
        }
        self.dims = dims;

        // Reallocate if size changed
        if self.sum.len() != n_cells {
            self.sum = vec![0.0; n_cells];
            self.sum_sq = vec![0.0; n_cells];
            self.first = vec![0; n_cells];
            self.last = vec![0; n_cells];
            self.min = vec![0; n_cells];
            self.max = vec![0; n_cells];
        } else {
            self.sum.fill(0.0);
            self.sum_sq.fill(0.0);
            self.first.fill(0);
            self.last.fill(0);
            self.min.fill(0);
            self.max.fill(0);
        }
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let box_unit = self.box_unit.expect("box_unit must be set in init");
        let region_size = self.region_size.expect("region_size must be set in init");

        let n_atoms = chunk.n_atoms;
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        let mut free_mask = vec![1u32; n_cells];

        for frame in 0..chunk.n_frames {
            free_mask.fill(1);

            let mut center = [0.0f64; 3];
            let mut count = 0.0f64;
            for &idx in self.center_selection.indices.iter() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                center[0] += p[0] as f64;
                center[1] += p[1] as f64;
                center[2] += p[2] as f64;
                count += 1.0;
            }
            if count > 0.0 {
                center[0] = (center[0] / count - self.shift[0]) * self.length_scale;
                center[1] = (center[1] / count - self.shift[1]) * self.length_scale;
                center[2] = (center[2] / count - self.shift[2]) * self.length_scale;
            }

            for &idx in self.selection.indices.iter() {
                let p = chunk.coords[frame * n_atoms + idx as usize];
                let x = p[0] as f64 * self.length_scale - center[0];
                let y = p[1] as f64 * self.length_scale - center[1];
                let z = p[2] as f64 * self.length_scale - center[2];
                if x < 0.0 || y < 0.0 || z < 0.0 {
                    continue;
                }
                if x > region_size[0] || y > region_size[1] || z > region_size[2] {
                    continue;
                }
                let ix = (x / box_unit[0]).floor() as usize;
                let iy = (y / box_unit[1]).floor() as usize;
                let iz = (z / box_unit[2]).floor() as usize;
                if ix < self.dims[0] && iy < self.dims[1] && iz < self.dims[2] {
                    self.mark_occupied_with_probe(box_unit, x, y, z, ix, iy, iz, &mut free_mask);
                }
            }

            if self.frames == 0 {
                self.first.copy_from_slice(&free_mask);
                self.min.copy_from_slice(&free_mask);
                self.max.copy_from_slice(&free_mask);
            }
            self.last.copy_from_slice(&free_mask);

            for i in 0..n_cells {
                let val = free_mask[i] as f64;
                self.sum[i] += val;
                self.sum_sq[i] += val * val;
                if free_mask[i] < self.min[i] {
                    self.min[i] = free_mask[i];
                }
                if free_mask[i] > self.max[i] {
                    self.max[i] = free_mask[i];
                }
            }
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        if self.frames == 0 {
            return Err(TrajError::Mismatch("no frames processed".into()));
        }
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        let frames_f = self.frames as f64;
        let mut mean = vec![0.0f32; n_cells];
        let mut std = vec![0.0f32; n_cells];
        for i in 0..n_cells {
            let avg = self.sum[i] / frames_f;
            let var = (self.sum_sq[i] / frames_f) - avg * avg;
            mean[i] = avg as f32;
            std[i] = var.max(0.0).sqrt() as f32;
        }
        Ok(PlanOutput::Grid(GridOutput {
            dims: self.dims,
            mean,
            std,
            first: self.first.clone(),
            last: self.last.clone(),
            min: self.min.clone(),
            max: self.max.clone(),
        }))
    }
}
