use crate::error::{TrajError, TrajResult};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Box3 {
    None,
    Orthorhombic { lx: f32, ly: f32, lz: f32 },
    Triclinic { m: [f32; 9] },
}

#[derive(Debug, Clone)]
pub struct FrameChunk {
    pub n_atoms: usize,
    pub n_frames: usize,
    pub coords: Vec<[f32; 4]>,
    pub box_: Vec<Box3>,
    pub time_ps: Option<Vec<f32>>,
    pub velocities: Option<Vec<[f32; 3]>>,
    pub forces: Option<Vec<[f32; 3]>>,
    pub lambda_values: Option<Vec<f32>>,
}

#[derive(Debug)]
pub struct FrameChunkBuilder {
    n_atoms: usize,
    n_frames: usize,
    coords_buf: Vec<[f32; 4]>,
    box_buf: Vec<Box3>,
    time_buf: Vec<f32>,
    velocity_buf: Vec<[f32; 3]>,
    force_buf: Vec<[f32; 3]>,
    lambda_buf: Vec<f32>,
    time_enabled: bool,
    velocities_enabled: bool,
    forces_enabled: bool,
    lambda_enabled: bool,
    store_box: bool,
    store_time: bool,
    store_velocities: bool,
    store_forces: bool,
    store_lambda: bool,
}

impl FrameChunkBuilder {
    pub fn new(n_atoms: usize, max_frames: usize) -> Self {
        Self {
            n_atoms,
            n_frames: 0,
            coords_buf: Vec::with_capacity(n_atoms * max_frames),
            box_buf: Vec::with_capacity(max_frames),
            time_buf: Vec::with_capacity(max_frames),
            velocity_buf: Vec::with_capacity(n_atoms * max_frames),
            force_buf: Vec::with_capacity(n_atoms * max_frames),
            lambda_buf: Vec::with_capacity(max_frames),
            time_enabled: false,
            velocities_enabled: false,
            forces_enabled: false,
            lambda_enabled: false,
            store_box: true,
            store_time: true,
            store_velocities: false,
            store_forces: false,
            store_lambda: false,
        }
    }

    pub fn set_requirements(&mut self, needs_box: bool, needs_time: bool) {
        self.store_box = needs_box;
        self.store_time = needs_time;
    }

    pub fn set_optional_requirements(
        &mut self,
        needs_velocities: bool,
        needs_forces: bool,
        needs_lambda: bool,
    ) {
        self.store_velocities = needs_velocities;
        self.store_forces = needs_forces;
        self.store_lambda = needs_lambda;
    }

    pub fn needs_box(&self) -> bool {
        self.store_box
    }

    pub fn needs_time(&self) -> bool {
        self.store_time
    }

    pub fn needs_velocities(&self) -> bool {
        self.store_velocities
    }

    pub fn needs_forces(&self) -> bool {
        self.store_forces
    }

    pub fn needs_lambda(&self) -> bool {
        self.store_lambda
    }

    pub fn reset(&mut self, n_atoms: usize, max_frames: usize) {
        self.n_atoms = n_atoms;
        self.n_frames = 0;
        self.coords_buf.clear();
        self.box_buf.clear();
        self.time_buf.clear();
        self.velocity_buf.clear();
        self.force_buf.clear();
        self.lambda_buf.clear();
        self.time_enabled = false;
        self.velocities_enabled = false;
        self.forces_enabled = false;
        self.lambda_enabled = false;
        self.coords_buf.reserve(n_atoms * max_frames);
        if self.store_box {
            self.box_buf.reserve(max_frames);
        }
        if self.store_time {
            self.time_buf.reserve(max_frames);
        }
        if self.store_velocities {
            self.velocity_buf.reserve(n_atoms * max_frames);
        }
        if self.store_forces {
            self.force_buf.reserve(n_atoms * max_frames);
        }
        if self.store_lambda {
            self.lambda_buf.reserve(max_frames);
        }
    }

    pub fn start_frame(&mut self, box_: Box3, time_ps: Option<f32>) -> &mut [[f32; 4]] {
        let frame_index = self.n_frames;
        self.n_frames += 1;
        if self.store_box {
            self.box_buf.push(box_);
        }
        if self.store_time {
            let t = time_ps.unwrap_or(0.0);
            self.time_buf.push(t);
            if time_ps.is_some() {
                self.time_enabled = true;
            }
        }
        let start = frame_index * self.n_atoms;
        let end = start + self.n_atoms;
        if self.coords_buf.len() < end {
            self.coords_buf.resize(end, [0.0; 4]);
        }
        &mut self.coords_buf[start..end]
    }

    pub fn set_frame_extras(
        &mut self,
        velocities: Option<&[[f32; 3]]>,
        forces: Option<&[[f32; 3]]>,
        lambda_value: Option<f32>,
    ) -> TrajResult<()> {
        if self.n_frames == 0 {
            return Err(TrajError::Parse(
                "frame extras require an active frame in the chunk builder".into(),
            ));
        }
        if self.store_velocities {
            match velocities {
                Some(data) if data.len() == self.n_atoms => {
                    self.velocity_buf.extend_from_slice(data);
                    self.velocities_enabled = true;
                }
                Some(_) => {
                    return Err(TrajError::Parse(
                        "frame chunk velocity buffer size mismatch".into(),
                    ));
                }
                None => self
                    .velocity_buf
                    .resize(self.velocity_buf.len() + self.n_atoms, [0.0; 3]),
            }
        }
        if self.store_forces {
            match forces {
                Some(data) if data.len() == self.n_atoms => {
                    self.force_buf.extend_from_slice(data);
                    self.forces_enabled = true;
                }
                Some(_) => {
                    return Err(TrajError::Parse(
                        "frame chunk force buffer size mismatch".into(),
                    ));
                }
                None => self
                    .force_buf
                    .resize(self.force_buf.len() + self.n_atoms, [0.0; 3]),
            }
        }
        if self.store_lambda {
            self.lambda_buf.push(lambda_value.unwrap_or(0.0));
            if lambda_value.is_some() {
                self.lambda_enabled = true;
            }
        }
        Ok(())
    }

    pub fn finish(&mut self) -> TrajResult<FrameChunk> {
        let n_frames = self.n_frames;
        if self.coords_buf.len() != n_frames * self.n_atoms {
            return Err(TrajError::Parse("frame chunk buffer size mismatch".into()));
        }
        if self.store_box && self.box_buf.len() != n_frames {
            return Err(TrajError::Parse(
                "frame chunk box buffer size mismatch".into(),
            ));
        }
        if self.store_time && self.time_enabled && self.time_buf.len() != n_frames {
            return Err(TrajError::Parse(
                "frame chunk time buffer size mismatch".into(),
            ));
        }
        if self.store_velocities && self.velocity_buf.len() != n_frames * self.n_atoms {
            return Err(TrajError::Parse(
                "frame chunk velocity buffer size mismatch".into(),
            ));
        }
        if self.store_forces && self.force_buf.len() != n_frames * self.n_atoms {
            return Err(TrajError::Parse(
                "frame chunk force buffer size mismatch".into(),
            ));
        }
        if self.store_lambda && self.lambda_buf.len() != n_frames {
            return Err(TrajError::Parse(
                "frame chunk lambda buffer size mismatch".into(),
            ));
        }
        let coords = self.coords_buf.clone();
        let box_ = if self.store_box {
            self.box_buf.clone()
        } else {
            Vec::new()
        };
        let time_ps = if self.store_time && self.time_enabled {
            Some(self.time_buf.clone())
        } else {
            None
        };
        let velocities = if self.store_velocities && self.velocities_enabled {
            Some(self.velocity_buf.clone())
        } else {
            None
        };
        let forces = if self.store_forces && self.forces_enabled {
            Some(self.force_buf.clone())
        } else {
            None
        };
        let lambda_values = if self.store_lambda && self.lambda_enabled {
            Some(self.lambda_buf.clone())
        } else {
            None
        };
        Ok(FrameChunk {
            n_atoms: self.n_atoms,
            n_frames,
            coords,
            box_,
            time_ps,
            velocities,
            forces,
            lambda_values,
        })
    }

    pub fn finish_take(&mut self) -> TrajResult<FrameChunk> {
        let n_frames = self.n_frames;
        if self.coords_buf.len() != n_frames * self.n_atoms {
            return Err(TrajError::Parse("frame chunk buffer size mismatch".into()));
        }
        if self.store_box && self.box_buf.len() != n_frames {
            return Err(TrajError::Parse(
                "frame chunk box buffer size mismatch".into(),
            ));
        }
        if self.store_time && self.time_enabled && self.time_buf.len() != n_frames {
            return Err(TrajError::Parse(
                "frame chunk time buffer size mismatch".into(),
            ));
        }
        if self.store_velocities && self.velocity_buf.len() != n_frames * self.n_atoms {
            return Err(TrajError::Parse(
                "frame chunk velocity buffer size mismatch".into(),
            ));
        }
        if self.store_forces && self.force_buf.len() != n_frames * self.n_atoms {
            return Err(TrajError::Parse(
                "frame chunk force buffer size mismatch".into(),
            ));
        }
        if self.store_lambda && self.lambda_buf.len() != n_frames {
            return Err(TrajError::Parse(
                "frame chunk lambda buffer size mismatch".into(),
            ));
        }
        let coords = std::mem::take(&mut self.coords_buf);
        let box_ = if self.store_box {
            std::mem::take(&mut self.box_buf)
        } else {
            Vec::new()
        };
        let time_ps = if self.store_time && self.time_enabled {
            Some(std::mem::take(&mut self.time_buf))
        } else {
            None
        };
        let velocities = if self.store_velocities && self.velocities_enabled {
            Some(std::mem::take(&mut self.velocity_buf))
        } else {
            None
        };
        let forces = if self.store_forces && self.forces_enabled {
            Some(std::mem::take(&mut self.force_buf))
        } else {
            None
        };
        let lambda_values = if self.store_lambda && self.lambda_enabled {
            Some(std::mem::take(&mut self.lambda_buf))
        } else {
            None
        };
        self.n_frames = 0;
        self.time_enabled = false;
        self.velocities_enabled = false;
        self.forces_enabled = false;
        self.lambda_enabled = false;
        Ok(FrameChunk {
            n_atoms: self.n_atoms,
            n_frames,
            coords,
            box_,
            time_ps,
            velocities,
            forces,
            lambda_values,
        })
    }

    pub fn reclaim(&mut self, chunk: FrameChunk) {
        self.n_atoms = chunk.n_atoms;
        self.coords_buf = chunk.coords;
        if !chunk.box_.is_empty() {
            self.box_buf = chunk.box_;
        } else {
            self.box_buf.clear();
        }
        if let Some(time_ps) = chunk.time_ps {
            self.time_buf = time_ps;
        } else {
            self.time_buf.clear();
        }
        if let Some(velocities) = chunk.velocities {
            self.velocity_buf = velocities;
        } else {
            self.velocity_buf.clear();
        }
        if let Some(forces) = chunk.forces {
            self.force_buf = forces;
        } else {
            self.force_buf.clear();
        }
        if let Some(lambda_values) = chunk.lambda_values {
            self.lambda_buf = lambda_values;
        } else {
            self.lambda_buf.clear();
        }
        self.n_frames = 0;
        self.time_enabled = false;
        self.velocities_enabled = false;
        self.forces_enabled = false;
        self.lambda_enabled = false;
        self.coords_buf.clear();
        self.box_buf.clear();
        self.time_buf.clear();
        self.velocity_buf.clear();
        self.force_buf.clear();
        self.lambda_buf.clear();
    }
}
