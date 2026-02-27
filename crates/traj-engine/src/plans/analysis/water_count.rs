use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, GridOutput, Plan, PlanOutput};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuContext, GpuCoords, GpuCountsU32, GpuSelection};

pub struct WaterCountPlan {
    water_selection: Selection,
    center_selection: Selection,
    box_unit: [f64; 3],
    region_size: [f64; 3],
    shift: [f64; 3],
    length_scale: f64,
    dims: [usize; 3],
    sum: Vec<f64>,
    sum_sq: Vec<f64>,
    first: Vec<u32>,
    last: Vec<u32>,
    min: Vec<u32>,
    max: Vec<u32>,
    frames: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<WaterCountGpuState>,
}

#[cfg(feature = "cuda")]
struct WaterCountGpuState {
    ctx: GpuContext,
    water_sel: GpuSelection,
    counts: GpuCountsU32,
}

impl WaterCountPlan {
    pub fn new(
        water_selection: Selection,
        center_selection: Selection,
        box_unit: [f64; 3],
        region_size: [f64; 3],
    ) -> Self {
        let dims = [
            (region_size[0] / box_unit[0]).floor().max(1.0) as usize + 1,
            (region_size[1] / box_unit[1]).floor().max(1.0) as usize + 1,
            (region_size[2] / box_unit[2]).floor().max(1.0) as usize + 1,
        ];
        let n_cells = dims[0] * dims[1] * dims[2];
        Self {
            water_selection,
            center_selection,
            box_unit,
            region_size,
            shift: [0.0, 0.0, 0.0],
            length_scale: 1.0,
            dims,
            sum: vec![0.0; n_cells],
            sum_sq: vec![0.0; n_cells],
            first: vec![0; n_cells],
            last: vec![0; n_cells],
            min: vec![0; n_cells],
            max: vec![0; n_cells],
            frames: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_shift(mut self, shift: [f64; 3]) -> Self {
        self.shift = shift;
        self
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }
}

impl Plan for WaterCountPlan {
    fn name(&self) -> &'static str {
        "water_count"
    }

    fn init(&mut self, _system: &System, device: &Device) -> TrajResult<()> {
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        self.sum.fill(0.0);
        self.sum_sq.fill(0.0);
        self.first.fill(0);
        self.last.fill(0);
        self.min.fill(0);
        self.max.fill(0);
        if self.sum.len() != n_cells {
            self.sum = vec![0.0; n_cells];
            self.sum_sq = vec![0.0; n_cells];
            self.first = vec![0; n_cells];
            self.last = vec![0; n_cells];
            self.min = vec![0; n_cells];
            self.max = vec![0; n_cells];
        }
        self.frames = 0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let water_sel = ctx.selection(&self.water_selection.indices, None)?;
                let counts = ctx.alloc_counts_u32(n_cells)?;
                self.gpu = Some(WaterCountGpuState {
                    ctx: ctx.clone(),
                    water_sel,
                    counts,
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = _system;
            let _ = device;
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        let mut counts = vec![0u32; n_cells];
        #[cfg(feature = "cuda")]
        let coords_gpu: Option<GpuCoords> = if let Some(gpu) = &self.gpu {
            Some(gpu.ctx.upload_coords(&convert_coords(&chunk.coords))?)
        } else {
            None
        };

        for frame in 0..chunk.n_frames {
            counts.fill(0);
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
            let used_gpu = {
                #[cfg(feature = "cuda")]
                {
                    if let (Some(gpu), Some(coords_gpu)) = (self.gpu.as_mut(), &coords_gpu) {
                        gpu.ctx.reset_counts_u32(&mut gpu.counts)?;
                        gpu.ctx.water_count_frame(
                            coords_gpu,
                            n_atoms,
                            frame,
                            &gpu.water_sel,
                            [center[0] as f32, center[1] as f32, center[2] as f32],
                            [
                                self.box_unit[0] as f32,
                                self.box_unit[1] as f32,
                                self.box_unit[2] as f32,
                            ],
                            [
                                self.region_size[0] as f32,
                                self.region_size[1] as f32,
                                self.region_size[2] as f32,
                            ],
                            [
                                self.dims[0] as i32,
                                self.dims[1] as i32,
                                self.dims[2] as i32,
                            ],
                            self.length_scale as f32,
                            &mut gpu.counts,
                        )?;
                        let counts_gpu = gpu.ctx.read_counts_u32(&gpu.counts)?;
                        counts.copy_from_slice(&counts_gpu);
                        true
                    } else {
                        false
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    false
                }
            };
            if !used_gpu {
                for &idx in self.water_selection.indices.iter() {
                    let p = chunk.coords[frame * n_atoms + idx as usize];
                    let x = p[0] as f64 * self.length_scale - center[0];
                    let y = p[1] as f64 * self.length_scale - center[1];
                    let z = p[2] as f64 * self.length_scale - center[2];
                    if x < 0.0 || y < 0.0 || z < 0.0 {
                        continue;
                    }
                    if x > self.region_size[0] || y > self.region_size[1] || z > self.region_size[2]
                    {
                        continue;
                    }
                    let ix = (x / self.box_unit[0]).floor() as usize;
                    let iy = (y / self.box_unit[1]).floor() as usize;
                    let iz = (z / self.box_unit[2]).floor() as usize;
                    if ix < self.dims[0] && iy < self.dims[1] && iz < self.dims[2] {
                        let flat = ix + self.dims[0] * (iy + self.dims[1] * iz);
                        counts[flat] += 1;
                    }
                }
            }
            if self.frames == 0 {
                self.first.copy_from_slice(&counts);
                self.min.copy_from_slice(&counts);
                self.max.copy_from_slice(&counts);
            }
            self.last.copy_from_slice(&counts);
            for i in 0..n_cells {
                let val = counts[i] as f64;
                self.sum[i] += val;
                self.sum_sq[i] += val * val;
                if counts[i] < self.min[i] {
                    self.min[i] = counts[i];
                }
                if counts[i] > self.max[i] {
                    self.max[i] = counts[i];
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
