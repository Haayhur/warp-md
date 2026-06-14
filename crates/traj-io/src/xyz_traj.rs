use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::{validate_and_materialize_selection, TrajReader};

pub struct XyzTrajReader {
    frames: Vec<Vec<[f32; 3]>>,
    index: usize,
    n_atoms: usize,
}

impl XyzTrajReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let frames = parse_xyz_frames(reader)?;
        let n_atoms = frames.get(0).map(|frame| frame.len()).unwrap_or(0);
        if n_atoms == 0 {
            return Err(TrajError::Parse("no atoms found in XYZ".into()));
        }
        for (idx, frame) in frames.iter().enumerate() {
            if frame.len() != n_atoms {
                return Err(TrajError::Parse(format!(
                    "frame {idx} atom count mismatch in XYZ (expected {n_atoms}, found {})",
                    frame.len()
                )));
            }
        }
        Ok(Self {
            frames,
            index: 0,
            n_atoms,
        })
    }

    pub fn reset(&mut self) {
        self.index = 0;
    }
}

impl TrajReader for XyzTrajReader {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.frames.len())
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        let max_frames = max_frames.max(1);
        out.reset(self.n_atoms, max_frames);
        let mut frames_read = 0;
        while frames_read < max_frames && self.index < self.frames.len() {
            let coords_src = &self.frames[self.index];
            let coords_dst = out.start_frame(Box3::None, None);
            for (i, coord) in coords_src.iter().enumerate() {
                coords_dst[i] = [coord[0], coord[1], coord[2], 1.0];
            }
            self.index += 1;
            frames_read += 1;
        }
        Ok(frames_read)
    }

    fn read_chunk_selected(
        &mut self,
        max_frames: usize,
        selection: &[u32],
        out: &mut FrameChunkBuilder,
    ) -> TrajResult<usize> {
        let selection = validate_and_materialize_selection(selection, self.n_atoms)?;

        let max_frames = max_frames.max(1);
        out.reset(selection.len(), max_frames);
        let mut frames_read = 0;
        while frames_read < max_frames && self.index < self.frames.len() {
            let coords_src = &self.frames[self.index];
            let coords_dst = out.start_frame(Box3::None, None);
            for (i, &src_idx) in selection.iter().enumerate() {
                let coord = coords_src[src_idx];
                coords_dst[i] = [coord[0], coord[1], coord[2], 1.0];
            }
            self.index += 1;
            frames_read += 1;
        }
        Ok(frames_read)
    }

    fn skip_frames(&mut self, n_frames: usize) -> TrajResult<usize> {
        let remaining = self.frames.len().saturating_sub(self.index);
        let skipped = remaining.min(n_frames);
        self.index += skipped;
        Ok(skipped)
    }
}

fn parse_xyz_frames<R: BufRead>(reader: R) -> TrajResult<Vec<Vec<[f32; 3]>>> {
    let mut frames: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut lines = reader.lines();

    loop {
        // Read number of atoms
        let Some(atom_count_line) = lines.next() else {
            break; // EOF
        };
        let atom_count_str = atom_count_line?;
        let atom_count_trimmed = atom_count_str.trim();
        if atom_count_trimmed.is_empty() {
            continue; // Skip empty lines between frames if any
        }
        let n_atoms = atom_count_trimmed.parse::<usize>().map_err(|_| {
            TrajError::Parse(format!("invalid atom count in XYZ: {}", atom_count_trimmed))
        })?;

        // Read comment line (skip)
        let Some(comment_line) = lines.next() else {
            return Err(TrajError::Parse(
                "unexpected EOF reading XYZ comment line".into(),
            ));
        };
        let _comment = comment_line?;

        let mut frame_coords = Vec::with_capacity(n_atoms);
        for i in 0..n_atoms {
            let Some(atom_line) = lines.next() else {
                return Err(TrajError::Parse(format!(
                    "unexpected EOF reading atom coordinate line {}/{}",
                    i + 1,
                    n_atoms
                )));
            };
            let atom_str = atom_line?;
            let mut parts = atom_str.split_whitespace();
            let _atom_name = parts.next().ok_or_else(|| {
                TrajError::Parse(format!("missing atom element on coordinate line {}", i + 1))
            })?;
            let x = parts
                .next()
                .ok_or_else(|| TrajError::Parse(format!("missing x coordinate on line {}", i + 1)))?
                .parse::<f32>()
                .map_err(|_| TrajError::Parse(format!("invalid x coordinate on line {}", i + 1)))?;
            let y = parts
                .next()
                .ok_or_else(|| TrajError::Parse(format!("missing y coordinate on line {}", i + 1)))?
                .parse::<f32>()
                .map_err(|_| TrajError::Parse(format!("invalid y coordinate on line {}", i + 1)))?;
            let z = parts
                .next()
                .ok_or_else(|| TrajError::Parse(format!("missing z coordinate on line {}", i + 1)))?
                .parse::<f32>()
                .map_err(|_| TrajError::Parse(format!("invalid z coordinate on line {}", i + 1)))?;

            frame_coords.push([x, y, z]);
        }
        frames.push(frame_coords);
    }

    Ok(frames)
}
