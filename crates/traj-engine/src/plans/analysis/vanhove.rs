use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, PlanRequirements, VanHoveOutput};

pub struct VanHovePlan {
    selection: Selection,
    r_bin: f64,
    r_max: f64,
    length_scale: f64,
    max_lag: Option<usize>,
    sqrt_time_bin: Option<f64>,
    scale_to_average_box: bool,
    remove_pbc_jumps: bool,
    time_scale: f64,
    n_sel: usize,
    n_frames: usize,
    coords: Vec<[f64; 3]>,
    boxes: Vec<Box3>,
    first_time_ps: Option<f64>,
    last_time_ps: Option<f64>,
    have_time_ps: bool,
}

impl VanHovePlan {
    pub fn new(selection: Selection, r_bin: f64, r_max: f64) -> Self {
        Self {
            selection,
            r_bin,
            r_max,
            length_scale: 1.0,
            max_lag: None,
            sqrt_time_bin: None,
            scale_to_average_box: true,
            remove_pbc_jumps: true,
            time_scale: 1.0,
            n_sel: 0,
            n_frames: 0,
            coords: Vec::new(),
            boxes: Vec::new(),
            first_time_ps: None,
            last_time_ps: None,
            have_time_ps: false,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }

    pub fn with_max_lag(mut self, max_lag: usize) -> Self {
        self.max_lag = Some(max_lag);
        self
    }

    pub fn with_sqrt_time_bin(mut self, width: f64) -> Self {
        self.sqrt_time_bin = Some(width);
        self
    }

    pub fn with_scale_to_average_box(mut self, enabled: bool) -> Self {
        self.scale_to_average_box = enabled;
        self
    }

    pub fn with_remove_pbc_jumps(mut self, enabled: bool) -> Self {
        self.remove_pbc_jumps = enabled;
        self
    }

    pub fn with_time_scale(mut self, scale: f64) -> Self {
        self.time_scale = scale;
        self
    }

    fn needs_box(&self) -> bool {
        self.scale_to_average_box || self.remove_pbc_jumps
    }

    fn dt_ps(&self) -> f64 {
        if self.have_time_ps && self.n_frames > 1 {
            let raw = (self.last_time_ps.unwrap_or(0.0) - self.first_time_ps.unwrap_or(0.0))
                / (self.n_frames.saturating_sub(1) as f64);
            let dt = if raw.is_finite() && raw > 0.0 {
                raw
            } else {
                1.0
            };
            (dt * 10_000.0).round() / 10_000.0 * self.time_scale
        } else {
            self.time_scale
        }
    }

    fn processed_coords(&self) -> TrajResult<Vec<[f64; 3]>> {
        let mut coords = self.coords.clone();
        if self.n_frames == 0 || self.n_sel == 0 || !self.needs_box() {
            return Ok(coords);
        }
        if self.boxes.len() != self.n_frames {
            return Err(TrajError::Mismatch(
                "vanhove requires per-frame box metadata when PBC correction is enabled".into(),
            ));
        }
        if self.boxes.iter().all(|box_| matches!(box_, Box3::None)) {
            return Ok(coords);
        }

        let mut scaled_boxes = Vec::with_capacity(self.n_frames);
        for box_ in &self.boxes {
            match box_ {
                Box3::Orthorhombic { lx, ly, lz } => scaled_boxes.push([
                    *lx as f64 * self.length_scale,
                    *ly as f64 * self.length_scale,
                    *lz as f64 * self.length_scale,
                ]),
                Box3::None => return Err(TrajError::Mismatch(
                    "vanhove requires either orthorhombic boxes for all frames or no box metadata"
                        .into(),
                )),
                Box3::Triclinic { .. } => {
                    return Err(TrajError::Unsupported(
                        "vanhove currently supports only orthorhombic boxes for PBC correction"
                            .into(),
                    ))
                }
            }
        }

        let mut average_box = [0.0f64; 3];
        if self.scale_to_average_box {
            for box_ in &scaled_boxes {
                average_box[0] += box_[0];
                average_box[1] += box_[1];
                average_box[2] += box_[2];
            }
            let inv = 1.0 / self.n_frames as f64;
            average_box[0] *= inv;
            average_box[1] *= inv;
            average_box[2] *= inv;
        }

        for frame in 0..self.n_frames {
            let frame_box = scaled_boxes[frame];
            let jump_box = if self.scale_to_average_box {
                average_box
            } else {
                frame_box
            };
            for atom in 0..self.n_sel {
                let idx = frame * self.n_sel + atom;
                if self.scale_to_average_box {
                    coords[idx][0] *= average_box[0] / frame_box[0];
                    coords[idx][1] *= average_box[1] / frame_box[1];
                    coords[idx][2] *= average_box[2] / frame_box[2];
                }
                if self.remove_pbc_jumps && frame > 0 {
                    let prev = (frame - 1) * self.n_sel + atom;
                    for axis in 0..3 {
                        let box_len = jump_box[axis];
                        if box_len <= 0.0 {
                            continue;
                        }
                        while coords[idx][axis] - coords[prev][axis] > 0.5 * box_len {
                            coords[idx][axis] -= box_len;
                        }
                        while coords[idx][axis] - coords[prev][axis] <= -0.5 * box_len {
                            coords[idx][axis] += box_len;
                        }
                    }
                }
            }
        }

        Ok(coords)
    }
}

impl Plan for VanHovePlan {
    fn name(&self) -> &'static str {
        "vanhove"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(self.needs_box(), true)
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        Some(self.selection.indices.as_ref())
    }

    fn init(&mut self, _system: &System, _device: &Device) -> TrajResult<()> {
        if !self.r_bin.is_finite() || self.r_bin <= 0.0 {
            return Err(TrajError::Parse("vanhove requires r_bin > 0".into()));
        }
        if !self.r_max.is_finite() || self.r_max <= 0.0 {
            return Err(TrajError::Parse("vanhove requires r_max > 0".into()));
        }
        if let Some(width) = self.sqrt_time_bin {
            if !width.is_finite() || width <= 0.0 {
                return Err(TrajError::Parse(
                    "vanhove sqrt_time_bin must be finite and > 0".into(),
                ));
            }
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "vanhove length_scale must be finite and > 0".into(),
            ));
        }
        if !self.time_scale.is_finite() || self.time_scale <= 0.0 {
            return Err(TrajError::Parse(
                "vanhove time_scale must be finite and > 0".into(),
            ));
        }

        self.n_sel = self.selection.indices.len();
        self.n_frames = 0;
        self.coords.clear();
        self.boxes.clear();
        self.first_time_ps = None;
        self.last_time_ps = None;
        self.have_time_ps = false;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.n_sel == 0 || chunk.n_frames == 0 {
            return Ok(());
        }
        if chunk.n_atoms != self.n_sel {
            return Err(TrajError::Mismatch(
                "vanhove expected selected-atom chunks matching the analysis selection".into(),
            ));
        }
        if self.needs_box() && chunk.box_.len() != chunk.n_frames {
            return Err(TrajError::Mismatch(
                "vanhove requires box metadata for every frame".into(),
            ));
        }

        self.coords
            .reserve(chunk.n_frames.saturating_mul(chunk.n_atoms));
        if self.needs_box() {
            self.boxes.reserve(chunk.n_frames);
        }

        for frame in 0..chunk.n_frames {
            if let Some(times) = &chunk.time_ps {
                if let Some(&time_ps) = times.get(frame) {
                    let time_ps = time_ps as f64;
                    if self.first_time_ps.is_none() {
                        self.first_time_ps = Some(time_ps);
                    }
                    self.last_time_ps = Some(time_ps);
                    self.have_time_ps = true;
                }
            }
            if self.needs_box() {
                self.boxes.push(chunk.box_[frame]);
            }
            let base = frame * chunk.n_atoms;
            for atom in 0..chunk.n_atoms {
                let p = chunk.coords[base + atom];
                self.coords.push([
                    p[0] as f64 * self.length_scale,
                    p[1] as f64 * self.length_scale,
                    p[2] as f64 * self.length_scale,
                ]);
            }
            self.n_frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let nbin = (self.r_max / self.r_bin).ceil() as usize;
        if self.n_sel == 0 || self.n_frames == 0 || nbin == 0 {
            return Ok(PlanOutput::VanHove(VanHoveOutput {
                time: Vec::new(),
                time_sqrt: Vec::new(),
                r: Vec::new(),
                matrix: Vec::new(),
                rows: 0,
                cols: 0,
                counts: Vec::new(),
                r_bin: self.r_bin as f32,
                r_max: self.r_max as f32,
            }));
        }

        let max_lag = self
            .max_lag
            .unwrap_or_else(|| self.n_frames.saturating_sub(1))
            .min(self.n_frames.saturating_sub(1));
        let dt_ps = self.dt_ps();
        let sqrt_width = self.sqrt_time_bin;
        let rows = if let Some(width) = sqrt_width {
            (((max_lag as f64 * dt_ps).sqrt() / width).round() as usize).saturating_add(1)
        } else {
            max_lag.saturating_add(1)
        };

        let mut counts = vec![0u64; rows];
        let mut accum = vec![0.0f64; rows.saturating_mul(nbin)];
        let coords = self.processed_coords()?;
        let r_limit = self.r_max;
        let r_limit2 = r_limit * r_limit;

        counts[0] = self.n_frames as u64;
        accum[0] = (self.n_frames * self.n_sel) as f64;

        for frame in 1..self.n_frames {
            for prev in 0..frame {
                let lag = frame - prev;
                if lag > max_lag {
                    continue;
                }
                let row = if let Some(width) = sqrt_width {
                    ((lag as f64 * dt_ps).sqrt() / width).round() as usize
                } else {
                    lag
                };
                if row >= rows {
                    continue;
                }
                for atom in 0..self.n_sel {
                    let idx = frame * self.n_sel + atom;
                    let prev_idx = prev * self.n_sel + atom;
                    let dx = coords[idx][0] - coords[prev_idx][0];
                    let dy = coords[idx][1] - coords[prev_idx][1];
                    let dz = coords[idx][2] - coords[prev_idx][2];
                    let d2 = dx * dx + dy * dy + dz * dz;
                    if d2 >= r_limit2 {
                        continue;
                    }
                    let bin = (d2.sqrt() / self.r_bin).floor() as usize;
                    if bin < nbin {
                        accum[row * nbin + bin] += 1.0;
                    }
                }
                counts[row] = counts[row].saturating_add(1);
            }
        }

        let mut matrix = vec![0.0f32; rows * nbin];
        let norm_n_sel = self.n_sel as f64 * self.r_bin;
        for row in 0..rows {
            let pairs = counts[row] as f64;
            if pairs == 0.0 {
                continue;
            }
            let norm = pairs * norm_n_sel;
            for bin in 0..nbin {
                matrix[row * nbin + bin] = (accum[row * nbin + bin] / norm) as f32;
            }
        }

        let r = (0..nbin)
            .map(|bin| (bin as f64 * self.r_bin) as f32)
            .collect();
        let mut time = Vec::with_capacity(rows);
        let mut time_sqrt = Vec::with_capacity(rows);
        for row in 0..rows {
            let sqrt_t = if let Some(width) = sqrt_width {
                row as f64 * width
            } else {
                (row as f64 * dt_ps).sqrt()
            };
            let t = if let Some(width) = sqrt_width {
                let _ = width;
                sqrt_t * sqrt_t
            } else {
                row as f64 * dt_ps
            };
            time.push(t as f32);
            time_sqrt.push(sqrt_t as f32);
        }

        Ok(PlanOutput::VanHove(VanHoveOutput {
            time,
            time_sqrt,
            r,
            matrix,
            rows,
            cols: nbin,
            counts,
            r_bin: self.r_bin as f32,
            r_max: self.r_max as f32,
        }))
    }
}
