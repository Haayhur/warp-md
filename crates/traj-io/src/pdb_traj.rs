use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::TrajReader;

pub struct PdbTrajReader {
    frames: Vec<Vec<[f32; 3]>>,
    index: usize,
    n_atoms: usize,
}

impl PdbTrajReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let frames = parse_pdb_frames(reader)?;
        let n_atoms = frames.get(0).map(|frame| frame.len()).unwrap_or(0);
        if n_atoms == 0 {
            return Err(TrajError::Parse("no atoms found in PDB/PDBQT".into()));
        }
        for (idx, frame) in frames.iter().enumerate() {
            if frame.len() != n_atoms {
                return Err(TrajError::Parse(format!(
                    "frame {idx} atom count mismatch in PDB/PDBQT"
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

impl TrajReader for PdbTrajReader {
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
        for &idx in selection {
            if (idx as usize) >= self.n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "selection index {idx} out of bounds for trajectory with {} atoms",
                    self.n_atoms
                )));
            }
        }

        let max_frames = max_frames.max(1);
        out.reset(selection.len(), max_frames);
        let mut frames_read = 0;
        while frames_read < max_frames && self.index < self.frames.len() {
            let coords_src = &self.frames[self.index];
            let coords_dst = out.start_frame(Box3::None, None);
            for (i, &src_idx) in selection.iter().enumerate() {
                let coord = coords_src[src_idx as usize];
                coords_dst[i] = [coord[0], coord[1], coord[2], 1.0];
            }
            self.index += 1;
            frames_read += 1;
        }
        Ok(frames_read)
    }
}

fn parse_pdb_frames<R: BufRead>(reader: R) -> TrajResult<Vec<Vec<[f32; 3]>>> {
    let mut frames: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut current: Vec<[f32; 3]> = Vec::new();
    let mut saw_model = false;
    let mut in_model = false;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("MODEL") {
            if !current.is_empty() {
                frames.push(current);
                current = Vec::new();
            }
            saw_model = true;
            in_model = true;
            continue;
        }
        if line.starts_with("ENDMDL") {
            if saw_model {
                if !current.is_empty() {
                    frames.push(current);
                    current = Vec::new();
                }
                in_model = false;
            }
            continue;
        }
        if saw_model && !in_model {
            continue;
        }
        if !(line.starts_with("ATOM") || line.starts_with("HETATM")) {
            continue;
        }
        let alt_loc = line.chars().nth(16).unwrap_or(' ');
        if alt_loc != ' ' && alt_loc != 'A' {
            continue;
        }
        let x = parse_float(slice_trim_opt(&line, 30, 38), "x")?;
        let y = parse_float(slice_trim_opt(&line, 38, 46), "y")?;
        let z = parse_float(slice_trim_opt(&line, 46, 54), "z")?;
        current.push([x, y, z]);
    }

    if !current.is_empty() {
        frames.push(current);
    }
    if frames.is_empty() {
        return Err(TrajError::Parse("no atoms found in PDB/PDBQT".into()));
    }
    Ok(frames)
}

fn slice_trim_opt<'a>(line: &'a str, start: usize, end: usize) -> &'a str {
    line.get(start..end).unwrap_or("").trim()
}

fn parse_float(token: &str, label: &str) -> TrajResult<f32> {
    token
        .parse::<f32>()
        .map_err(|_| TrajError::Parse(format!("invalid {label} coordinate: {token}")))
}
