use nalgebra::{Matrix3, SymmetricEigen};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::pbc_math::minimum_image_vector;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

pub struct NematicOrderPlan {
    tail_indices: Vec<u32>,
    head_indices: Vec<u32>,
    reference_axis: Option<[f64; 3]>,
    use_pbc: bool,
    length_scale: f64,
    results: Vec<f32>,
    time: Vec<f32>,
    frames: usize,
}

impl NematicOrderPlan {
    pub fn new(tail_indices: Vec<u32>, head_indices: Vec<u32>) -> Self {
        Self {
            tail_indices,
            head_indices,
            reference_axis: None,
            use_pbc: false,
            length_scale: 1.0,
            results: Vec::new(),
            time: Vec::new(),
            frames: 0,
        }
    }

    pub fn with_reference_axis(mut self, axis: Option<[f64; 3]>) -> Self {
        self.reference_axis = axis.and_then(normalize_axis);
        self
    }

    pub fn with_pbc(mut self, use_pbc: bool) -> Self {
        self.use_pbc = use_pbc;
        self
    }

    pub fn with_length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }
}

impl Plan for NematicOrderPlan {
    fn name(&self) -> &'static str {
        "nematic_order"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.tail_indices.len() != self.head_indices.len() {
            return Err(TrajError::Mismatch(
                "nematic_order tail/head index vectors must have identical length".into(),
            ));
        }
        if self.tail_indices.is_empty() {
            return Err(TrajError::Mismatch(
                "nematic_order requires at least one vector pair".into(),
            ));
        }
        let n_atoms = system.n_atoms();
        for &idx in self.tail_indices.iter().chain(self.head_indices.iter()) {
            if idx as usize >= n_atoms {
                return Err(TrajError::Mismatch(
                    "nematic_order atom index out of bounds".into(),
                ));
            }
        }
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(TrajError::Parse(
                "nematic_order length_scale must be finite and > 0".into(),
            ));
        }
        self.results.clear();
        self.time.clear();
        self.frames = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let base_time = self.frames;
        for frame in 0..chunk.n_frames {
            let base = frame * chunk.n_atoms;
            let coords = &chunk.coords[base..base + chunk.n_atoms];
            let stats = frame_nematic_stats(
                coords,
                chunk.box_[frame],
                &self.tail_indices,
                &self.head_indices,
                self.reference_axis,
                self.use_pbc,
                self.length_scale,
            )?;
            self.results.push(stats.order);
            self.results.extend_from_slice(&stats.director);
            self.results.extend_from_slice(&stats.q_tensor);
            self.results.push(stats.axis_order);
            self.results.push(stats.valid_vectors as f32);
            let time = chunk
                .time_ps
                .as_ref()
                .and_then(|times| times.get(frame).copied())
                .unwrap_or((base_time + frame) as f32);
            self.time.push(time);
        }
        self.frames += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::TimeSeries {
            time: std::mem::take(&mut self.time),
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: 12,
        })
    }
}

#[derive(Clone, Copy)]
struct NematicStats {
    order: f32,
    director: [f32; 3],
    q_tensor: [f32; 6],
    axis_order: f32,
    valid_vectors: usize,
}

fn frame_nematic_stats(
    coords: &[[f32; 4]],
    box_: traj_core::frame::Box3,
    tail_indices: &[u32],
    head_indices: &[u32],
    reference_axis: Option<[f64; 3]>,
    use_pbc: bool,
    length_scale: f64,
) -> TrajResult<NematicStats> {
    let mut q = [[0.0f64; 3]; 3];
    let mut axis_sum = 0.0f64;
    let mut valid = 0usize;
    for (&tail, &head) in tail_indices.iter().zip(head_indices.iter()) {
        let from = coords[tail as usize];
        let to = coords[head as usize];
        let delta = if use_pbc {
            minimum_image_vector(
                [from[0] as f64, from[1] as f64, from[2] as f64],
                [to[0] as f64, to[1] as f64, to[2] as f64],
                box_,
                length_scale,
            )?
        } else {
            [
                (to[0] - from[0]) as f64 * length_scale,
                (to[1] - from[1]) as f64 * length_scale,
                (to[2] - from[2]) as f64 * length_scale,
            ]
        };
        let norm2 = dot(delta, delta);
        if norm2 <= f64::EPSILON {
            continue;
        }
        let inv_norm = norm2.sqrt().recip();
        let u = [
            delta[0] * inv_norm,
            delta[1] * inv_norm,
            delta[2] * inv_norm,
        ];
        for row in 0..3 {
            for col in 0..3 {
                q[row][col] += 1.5 * u[row] * u[col];
            }
            q[row][row] -= 0.5;
        }
        if let Some(axis) = reference_axis {
            let c = dot(u, axis);
            axis_sum += 0.5 * (3.0 * c * c - 1.0);
        }
        valid += 1;
    }
    if valid == 0 {
        return Ok(NematicStats {
            order: 0.0,
            director: [0.0; 3],
            q_tensor: [0.0; 6],
            axis_order: f32::NAN,
            valid_vectors: 0,
        });
    }
    let inv_valid = 1.0 / valid as f64;
    for row in 0..3 {
        for col in 0..3 {
            q[row][col] *= inv_valid;
        }
    }
    let tensor = Matrix3::new(
        q[0][0], q[0][1], q[0][2], q[1][0], q[1][1], q[1][2], q[2][0], q[2][1], q[2][2],
    );
    let eigen = SymmetricEigen::new(tensor);
    let mut max_idx = 0usize;
    if eigen.eigenvalues[1] > eigen.eigenvalues[max_idx] {
        max_idx = 1;
    }
    if eigen.eigenvalues[2] > eigen.eigenvalues[max_idx] {
        max_idx = 2;
    }
    let mut director = [
        eigen.eigenvectors[(0, max_idx)],
        eigen.eigenvectors[(1, max_idx)],
        eigen.eigenvectors[(2, max_idx)],
    ];
    orient_director(&mut director, reference_axis);
    let axis_order = reference_axis
        .map(|_| (axis_sum * inv_valid) as f32)
        .unwrap_or(f32::NAN);
    Ok(NematicStats {
        order: eigen.eigenvalues[max_idx] as f32,
        director: [director[0] as f32, director[1] as f32, director[2] as f32],
        q_tensor: [
            q[0][0] as f32,
            q[1][1] as f32,
            q[2][2] as f32,
            q[0][1] as f32,
            q[0][2] as f32,
            q[1][2] as f32,
        ],
        axis_order,
        valid_vectors: valid,
    })
}

fn normalize_axis(axis: [f64; 3]) -> Option<[f64; 3]> {
    let norm2 = dot(axis, axis);
    if !norm2.is_finite() || norm2 <= f64::EPSILON {
        return None;
    }
    let inv = norm2.sqrt().recip();
    Some([axis[0] * inv, axis[1] * inv, axis[2] * inv])
}

fn orient_director(director: &mut [f64; 3], reference_axis: Option<[f64; 3]>) {
    if let Some(axis) = reference_axis {
        if dot(*director, axis) < 0.0 {
            director[0] = -director[0];
            director[1] = -director[1];
            director[2] = -director[2];
        }
        return;
    }
    let mut idx = 0usize;
    if director[1].abs() > director[idx].abs() {
        idx = 1;
    }
    if director[2].abs() > director[idx].abs() {
        idx = 2;
    }
    if director[idx] < 0.0 {
        director[0] = -director[0];
        director[1] = -director[1];
        director[2] = -director[2];
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
