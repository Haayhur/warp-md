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
}

#[derive(Debug)]
pub struct FrameChunkBuilder {
    n_atoms: usize,
    n_frames: usize,
    coords_buf: Vec<[f32; 4]>,
    box_buf: Vec<Box3>,
    time_buf: Vec<f32>,
    time_enabled: bool,
    store_box: bool,
    store_time: bool,
}

impl FrameChunkBuilder {
    pub fn new(n_atoms: usize, max_frames: usize) -> Self {
        Self {
            n_atoms,
            n_frames: 0,
            coords_buf: Vec::with_capacity(n_atoms * max_frames),
            box_buf: Vec::with_capacity(max_frames),
            time_buf: Vec::with_capacity(max_frames),
            time_enabled: false,
            store_box: true,
            store_time: true,
        }
    }

    pub fn set_requirements(&mut self, needs_box: bool, needs_time: bool) {
        self.store_box = needs_box;
        self.store_time = needs_time;
    }

    pub fn needs_box(&self) -> bool {
        self.store_box
    }

    pub fn needs_time(&self) -> bool {
        self.store_time
    }

    pub fn reset(&mut self, n_atoms: usize, max_frames: usize) {
        self.n_atoms = n_atoms;
        self.n_frames = 0;
        self.coords_buf.clear();
        self.box_buf.clear();
        self.time_buf.clear();
        self.time_enabled = false;
        self.coords_buf.reserve(n_atoms * max_frames);
        if self.store_box {
            self.box_buf.reserve(max_frames);
        }
        if self.store_time {
            self.time_buf.reserve(max_frames);
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
        Ok(FrameChunk {
            n_atoms: self.n_atoms,
            n_frames,
            coords,
            box_,
            time_ps,
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
        self.n_frames = 0;
        self.time_enabled = false;
        Ok(FrameChunk {
            n_atoms: self.n_atoms,
            n_frames,
            coords,
            box_,
            time_ps,
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
        self.n_frames = 0;
        self.time_enabled = false;
        self.coords_buf.clear();
        self.box_buf.clear();
        self.time_buf.clear();
    }
}
