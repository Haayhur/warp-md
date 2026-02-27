use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuContext, GpuSelection};

pub struct HbondPlan {
    donors: Selection,
    acceptors: Selection,
    hydrogens: Option<Selection>,
    dist_cutoff: f64,
    angle_cutoff_deg: Option<f64>,
    coords: Vec<[f32; 4]>,
    times: Vec<f64>,
    n_atoms: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<HbondGpuState>,
}

#[cfg(feature = "cuda")]
struct HbondGpuState {
    ctx: GpuContext,
    donors: GpuSelection,
    acceptors: GpuSelection,
    hydrogens: Option<GpuSelection>,
}

impl HbondPlan {
    pub fn new(donors: Selection, acceptors: Selection, dist_cutoff: f64) -> Self {
        Self {
            donors,
            acceptors,
            hydrogens: None,
            dist_cutoff,
            angle_cutoff_deg: None,
            coords: Vec::new(),
            times: Vec::new(),
            n_atoms: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn with_hydrogens(mut self, hydrogens: Selection, angle_cutoff_deg: f64) -> Self {
        self.hydrogens = Some(hydrogens);
        self.angle_cutoff_deg = Some(angle_cutoff_deg);
        self
    }
}

impl Plan for HbondPlan {
    fn name(&self) -> &'static str {
        "hbond"
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.coords.clear();
        self.times.clear();
        self.n_atoms = system.n_atoms();
        if let Some(h_sel) = &self.hydrogens {
            if h_sel.indices.len() != self.donors.indices.len() {
                return Err(TrajError::Mismatch(
                    "hydrogen selection must match donor selection length".into(),
                ));
            }
        }
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let donors = ctx.selection(&self.donors.indices, None)?;
                let acceptors = ctx.selection(&self.acceptors.indices, None)?;
                let hydrogens = if let Some(h_sel) = &self.hydrogens {
                    Some(ctx.selection(&h_sel.indices, None)?)
                } else {
                    None
                };
                self.gpu = Some(HbondGpuState {
                    ctx: ctx.clone(),
                    donors,
                    acceptors,
                    hydrogens,
                });
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = device;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        self.coords.extend_from_slice(&chunk.coords);
        let base = self.times.len();
        if let Some(times) = &chunk.time_ps {
            for &t in times {
                self.times.push(t as f64);
            }
        } else {
            for i in 0..chunk.n_frames {
                self.times.push((base + i) as f64);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_frames = self.times.len();
        if n_frames == 0 {
            return Ok(PlanOutput::TimeSeries {
                time: Vec::new(),
                data: Vec::new(),
                rows: 0,
                cols: 0,
            });
        }
        let mut counts = vec![0.0f32; n_frames];
        let dist2 = self.dist_cutoff * self.dist_cutoff;
        let use_angle = self.hydrogens.is_some() && self.angle_cutoff_deg.is_some();
        let cos_cutoff = self.angle_cutoff_deg.map(|deg| (deg.to_radians()).cos());
        let donors = &self.donors.indices;
        let acceptors = &self.acceptors.indices;
        let hydrogens = self.hydrogens.as_ref().map(|h| &h.indices);

        let used_gpu = {
            #[cfg(feature = "cuda")]
            {
                if let Some(gpu) = &self.gpu {
                    let coords = convert_coords(&self.coords);
                    let counts_gpu = if use_angle {
                        let hydrogens = gpu.hydrogens.as_ref().ok_or_else(|| {
                            TrajError::Mismatch("hydrogen selection missing on GPU".into())
                        })?;
                        let cos_cutoff = cos_cutoff.unwrap_or(-1.0) as f32;
                        gpu.ctx.hbond_counts_angle(
                            &coords,
                            self.n_atoms,
                            n_frames,
                            &gpu.donors,
                            hydrogens,
                            &gpu.acceptors,
                            dist2 as f32,
                            cos_cutoff,
                        )?
                    } else {
                        gpu.ctx.hbond_counts(
                            &coords,
                            self.n_atoms,
                            n_frames,
                            &gpu.donors,
                            &gpu.acceptors,
                            dist2 as f32,
                        )?
                    };
                    for (i, c) in counts_gpu.iter().enumerate() {
                        counts[i] = *c as f32;
                    }
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
            for frame in 0..n_frames {
                let frame_offset = frame * self.n_atoms;
                let mut count = 0u32;
                for (d_idx, &donor) in donors.iter().enumerate() {
                    let d = self.coords[frame_offset + donor as usize];
                    let h = hydrogens.map(|h| self.coords[frame_offset + h[d_idx] as usize]);
                    for &acceptor in acceptors.iter() {
                        if acceptor == donor {
                            continue;
                        }
                        let a = self.coords[frame_offset + acceptor as usize];
                        let dx = a[0] as f64 - d[0] as f64;
                        let dy = a[1] as f64 - d[1] as f64;
                        let dz = a[2] as f64 - d[2] as f64;
                        let r2 = dx * dx + dy * dy + dz * dz;
                        if r2 > dist2 {
                            continue;
                        }
                        if use_angle {
                            let h = h.unwrap();
                            let v1 = [
                                d[0] as f64 - h[0] as f64,
                                d[1] as f64 - h[1] as f64,
                                d[2] as f64 - h[2] as f64,
                            ];
                            let v2 = [
                                a[0] as f64 - h[0] as f64,
                                a[1] as f64 - h[1] as f64,
                                a[2] as f64 - h[2] as f64,
                            ];
                            let norm1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
                            let norm2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
                            if norm1 == 0.0 || norm2 == 0.0 {
                                continue;
                            }
                            let cos_val =
                                (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (norm1 * norm2);
                            if let Some(cutoff) = cos_cutoff {
                                if cos_val < cutoff {
                                    continue;
                                }
                            }
                        }
                        count += 1;
                    }
                }
                counts[frame] = count as f32;
            }
        }

        Ok(PlanOutput::TimeSeries {
            time: self.times.iter().map(|t| *t as f32).collect(),
            data: counts,
            rows: n_frames,
            cols: 1,
        })
    }
}
