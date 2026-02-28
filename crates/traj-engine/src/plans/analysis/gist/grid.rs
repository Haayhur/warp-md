use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

use super::common::{
    dims_from_bounds, keep_frame_internal, mean_center_all_atoms, mean_center_indices,
    orientation_bin, sorted_unique_indices, validate_water_vectors, voxel_flat,
};
use super::scaling::finalize_counts_orientation;

#[cfg(feature = "cuda")]
use super::gpu::ensure_gist_gpu_hist_buffers;
#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuBufferU32, GpuContext, GpuCoords};

pub struct GistGridPlan {
    oxygen_indices: Vec<u32>,
    hydrogen1_indices: Vec<u32>,
    hydrogen2_indices: Vec<u32>,
    orientation_valid: Vec<u8>,
    solute_selection: Selection,
    origin: [f64; 3],
    dims: [usize; 3],
    spacing: f64,
    padding: f64,
    orientation_bins: usize,
    length_scale: f64,
    auto_grid: bool,
    frame_filter: Option<Vec<usize>>,
    frame_filter_pos: usize,
    max_frames: Option<usize>,
    counts: Vec<u32>,
    orient_counts: Vec<u32>,
    n_frames: usize,
    global_frame: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<GistGridGpuState>,
}

#[cfg(feature = "cuda")]
struct GistGridGpuState {
    ctx: GpuContext,
    oxygen_idx: GpuBufferU32,
    h1_idx: GpuBufferU32,
    h2_idx: GpuBufferU32,
    orient_valid: GpuBufferU32,
    counts: Option<GpuBufferU32>,
    orient_counts: Option<GpuBufferU32>,
    n_cells: usize,
}

impl GistGridPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        oxygen_indices: Vec<u32>,
        hydrogen1_indices: Vec<u32>,
        hydrogen2_indices: Vec<u32>,
        orientation_valid: Vec<u8>,
        solute_selection: Selection,
        origin: [f64; 3],
        dims: [usize; 3],
        spacing: f64,
        orientation_bins: usize,
    ) -> TrajResult<Self> {
        if spacing <= 0.0 {
            return Err(TrajError::Parse("gist spacing must be > 0".into()));
        }
        if dims.iter().any(|&d| d == 0) {
            return Err(TrajError::Parse("gist dims must be positive".into()));
        }
        validate_water_vectors(
            &oxygen_indices,
            &hydrogen1_indices,
            &hydrogen2_indices,
            &orientation_valid,
        )?;
        let orientation_bins = orientation_bins.max(1);
        let n_cells = dims[0] * dims[1] * dims[2];
        Ok(Self {
            oxygen_indices,
            hydrogen1_indices,
            hydrogen2_indices,
            orientation_valid,
            solute_selection,
            origin,
            dims,
            spacing,
            padding: 0.0,
            orientation_bins,
            length_scale: 1.0,
            auto_grid: false,
            frame_filter: None,
            frame_filter_pos: 0,
            max_frames: None,
            counts: vec![0u32; n_cells],
            orient_counts: vec![0u32; n_cells * orientation_bins],
            n_frames: 0,
            global_frame: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        })
    }

    pub fn new_auto(
        oxygen_indices: Vec<u32>,
        hydrogen1_indices: Vec<u32>,
        hydrogen2_indices: Vec<u32>,
        orientation_valid: Vec<u8>,
        solute_selection: Selection,
        spacing: f64,
        padding: f64,
        orientation_bins: usize,
    ) -> TrajResult<Self> {
        if spacing <= 0.0 {
            return Err(TrajError::Parse("gist spacing must be > 0".into()));
        }
        validate_water_vectors(
            &oxygen_indices,
            &hydrogen1_indices,
            &hydrogen2_indices,
            &orientation_valid,
        )?;
        if padding < 0.0 {
            return Err(TrajError::Parse("gist padding must be non-negative".into()));
        }
        let orientation_bins = orientation_bins.max(1);
        Ok(Self {
            oxygen_indices,
            hydrogen1_indices,
            hydrogen2_indices,
            orientation_valid,
            solute_selection,
            origin: [0.0, 0.0, 0.0],
            dims: [0, 0, 0],
            spacing,
            padding,
            orientation_bins,
            length_scale: 1.0,
            auto_grid: true,
            frame_filter: None,
            frame_filter_pos: 0,
            max_frames: None,
            counts: Vec::new(),
            orient_counts: Vec::new(),
            n_frames: 0,
            global_frame: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        })
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }

    pub fn with_max_frames(mut self, max_frames: Option<usize>) -> Self {
        self.max_frames = max_frames;
        self
    }

    pub fn with_frame_indices(mut self, frame_indices: Option<Vec<usize>>) -> Self {
        self.frame_filter = frame_indices.map(sorted_unique_indices);
        self
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    pub fn orientation_bins(&self) -> usize {
        self.orientation_bins
    }

    pub fn n_frames(&self) -> usize {
        self.n_frames
    }

    fn keep_frame(&mut self, abs_frame: usize) -> bool {
        keep_frame_internal(
            self.max_frames,
            self.n_frames,
            self.frame_filter.as_ref(),
            &mut self.frame_filter_pos,
            abs_frame,
        )
    }

    fn ensure_grid_for_frame(&mut self, chunk: &FrameChunk, frame: usize) -> TrajResult<()> {
        if !self.auto_grid || !self.counts.is_empty() {
            return Ok(());
        }
        let n_atoms = chunk.n_atoms;
        if n_atoms == 0 {
            return Err(TrajError::Mismatch(
                "gist grid requires at least one atom".into(),
            ));
        }
        let base = frame * n_atoms;
        let use_all = self.solute_selection.indices.is_empty();
        let mut min_xyz = [f64::INFINITY; 3];
        let mut max_xyz = [f64::NEG_INFINITY; 3];
        if use_all {
            for atom_idx in 0..n_atoms {
                let p = chunk.coords[base + atom_idx];
                let x = p[0] as f64 * self.length_scale;
                let y = p[1] as f64 * self.length_scale;
                let z = p[2] as f64 * self.length_scale;
                min_xyz[0] = min_xyz[0].min(x);
                min_xyz[1] = min_xyz[1].min(y);
                min_xyz[2] = min_xyz[2].min(z);
                max_xyz[0] = max_xyz[0].max(x);
                max_xyz[1] = max_xyz[1].max(y);
                max_xyz[2] = max_xyz[2].max(z);
            }
        } else {
            for &idx in self.solute_selection.indices.iter() {
                let atom_idx = idx as usize;
                if atom_idx >= n_atoms {
                    return Err(TrajError::Mismatch(
                        "gist solute selection index out of bounds".into(),
                    ));
                }
                let p = chunk.coords[base + atom_idx];
                let x = p[0] as f64 * self.length_scale;
                let y = p[1] as f64 * self.length_scale;
                let z = p[2] as f64 * self.length_scale;
                min_xyz[0] = min_xyz[0].min(x);
                min_xyz[1] = min_xyz[1].min(y);
                min_xyz[2] = min_xyz[2].min(z);
                max_xyz[0] = max_xyz[0].max(x);
                max_xyz[1] = max_xyz[1].max(y);
                max_xyz[2] = max_xyz[2].max(z);
            }
        }
        self.origin = [
            min_xyz[0] - self.padding,
            min_xyz[1] - self.padding,
            min_xyz[2] - self.padding,
        ];
        self.dims = dims_from_bounds(min_xyz, max_xyz, self.padding, self.spacing);
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        self.counts = vec![0u32; n_cells];
        self.orient_counts = vec![0u32; n_cells * self.orientation_bins];
        Ok(())
    }
}

impl Plan for GistGridPlan {
    fn name(&self) -> &'static str {
        "gist_grid"
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        if self.auto_grid {
            self.counts.clear();
            self.orient_counts.clear();
            self.dims = [0, 0, 0];
            self.origin = [0.0, 0.0, 0.0];
        } else {
            self.counts.fill(0);
            self.orient_counts.fill(0);
        }
        self.n_frames = 0;
        self.global_frame = 0;
        self.frame_filter_pos = 0;
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = _device {
                let orient_valid: Vec<u32> = self
                    .orientation_valid
                    .iter()
                    .map(|&v| if v != 0 { 1u32 } else { 0u32 })
                    .collect();
                let gpu = GistGridGpuState {
                    ctx: ctx.clone(),
                    oxygen_idx: ctx.upload_u32(&self.oxygen_indices)?,
                    h1_idx: ctx.upload_u32(&self.hydrogen1_indices)?,
                    h2_idx: ctx.upload_u32(&self.hydrogen2_indices)?,
                    orient_valid: ctx.upload_u32(&orient_valid)?,
                    counts: None,
                    orient_counts: None,
                    n_cells: 0,
                };
                self.gpu = Some(gpu);
            }
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
        let n_waters = self.oxygen_indices.len();
        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
        if self.counts.len() != n_cells
            || self.orient_counts.len() != n_cells * self.orientation_bins
        {
            return Err(TrajError::Mismatch(
                "gist grid buffer shape mismatch".into(),
            ));
        }

        #[cfg(feature = "cuda")]
        let coords_gpu: Option<GpuCoords> = if let Some(gpu) = &self.gpu {
            Some(gpu.ctx.upload_coords(&convert_coords(&chunk.coords))?)
        } else {
            None
        };

        for local_frame in 0..chunk.n_frames {
            let abs_frame = self.global_frame + local_frame;
            if !self.keep_frame(abs_frame) {
                continue;
            }
            self.ensure_grid_for_frame(chunk, local_frame)?;
            let base = local_frame * n_atoms;
            let center = if self.solute_selection.indices.is_empty() {
                mean_center_all_atoms(chunk, local_frame, self.length_scale)?
            } else {
                mean_center_indices(
                    chunk,
                    local_frame,
                    self.length_scale,
                    self.solute_selection.indices.as_ref(),
                )?
            };

            let used_gpu = {
                #[cfg(feature = "cuda")]
                {
                    if let (Some(gpu), Some(coords_gpu), Device::Cuda(_)) =
                        (self.gpu.as_mut(), coords_gpu.as_ref(), _device)
                    {
                        let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
                        ensure_gist_gpu_hist_buffers(
                            &gpu.ctx,
                            &mut gpu.counts,
                            &mut gpu.orient_counts,
                            &mut gpu.n_cells,
                            n_cells,
                            self.orientation_bins,
                        )?;
                        let (cells_dev, bins_dev) = gpu.ctx.gist_counts_orient_frame_device(
                            coords_gpu,
                            n_atoms,
                            local_frame,
                            &gpu.oxygen_idx,
                            &gpu.h1_idx,
                            &gpu.h2_idx,
                            &gpu.orient_valid,
                            n_waters,
                            [center[0] as f32, center[1] as f32, center[2] as f32],
                            [
                                self.origin[0] as f32,
                                self.origin[1] as f32,
                                self.origin[2] as f32,
                            ],
                            self.spacing as f32,
                            self.dims,
                            self.orientation_bins,
                            self.length_scale as f32,
                        )?;
                        let (Some(counts_dev), Some(orient_counts_dev)) =
                            (gpu.counts.as_mut(), gpu.orient_counts.as_mut())
                        else {
                            return Err(TrajError::Mismatch(
                                "gist cuda histogram buffers not initialized".into(),
                            ));
                        };
                        gpu.ctx.gist_accumulate_hist(
                            &cells_dev,
                            &bins_dev,
                            n_waters,
                            self.orientation_bins,
                            counts_dev,
                            orient_counts_dev,
                        )?;
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
                for i in 0..n_waters {
                    let oxy_idx = self.oxygen_indices[i] as usize;
                    if oxy_idx >= n_atoms {
                        return Err(TrajError::Mismatch(
                            "gist oxygen index out of bounds".into(),
                        ));
                    }
                    let po = chunk.coords[base + oxy_idx];
                    let ox = po[0] as f64 * self.length_scale;
                    let oy = po[1] as f64 * self.length_scale;
                    let oz = po[2] as f64 * self.length_scale;
                    let Some(flat) = voxel_flat([ox, oy, oz], self.origin, self.spacing, self.dims)
                    else {
                        continue;
                    };
                    self.counts[flat] = self.counts[flat].saturating_add(1);
                    if self.orientation_valid[i] == 0 {
                        continue;
                    }
                    let h1_idx = self.hydrogen1_indices[i] as usize;
                    let h2_idx = self.hydrogen2_indices[i] as usize;
                    if h1_idx >= n_atoms || h2_idx >= n_atoms {
                        continue;
                    }
                    let ph1 = chunk.coords[base + h1_idx];
                    let ph2 = chunk.coords[base + h2_idx];
                    let hmid = [
                        0.5 * ((ph1[0] as f64 + ph2[0] as f64) * self.length_scale),
                        0.5 * ((ph1[1] as f64 + ph2[1] as f64) * self.length_scale),
                        0.5 * ((ph1[2] as f64 + ph2[2] as f64) * self.length_scale),
                    ];
                    let hvec = [hmid[0] - ox, hmid[1] - oy, hmid[2] - oz];
                    let rvec = [ox - center[0], oy - center[1], oz - center[2]];
                    let Some(bin) = orientation_bin(hvec, rvec, self.orientation_bins) else {
                        continue;
                    };
                    let orient_flat = flat * self.orientation_bins + bin;
                    self.orient_counts[orient_flat] =
                        self.orient_counts[orient_flat].saturating_add(1);
                }
            }
            self.n_frames += 1;
        }
        self.global_frame += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        #[cfg(feature = "cuda")]
        {
            if let Some(gpu) = self.gpu.as_ref() {
                let n_cells = self.dims[0] * self.dims[1] * self.dims[2];
                if n_cells != 0 {
                    let orient_len =
                        n_cells.checked_mul(self.orientation_bins).ok_or_else(|| {
                            TrajError::Mismatch("gist orientation histogram size overflow".into())
                        })?;
                    if let (Some(counts_dev), Some(orient_counts_dev)) =
                        (gpu.counts.as_ref(), gpu.orient_counts.as_ref())
                    {
                        self.counts = gpu.ctx.download_u32(counts_dev, n_cells)?;
                        self.orient_counts = gpu.ctx.download_u32(orient_counts_dev, orient_len)?;
                    }
                }
            }
        }
        finalize_counts_orientation(
            &self.counts,
            &self.orient_counts,
            self.dims,
            self.orientation_bins,
        )
    }
}
