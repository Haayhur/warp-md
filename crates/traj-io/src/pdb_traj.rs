use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::TrajReader;

pub struct PdbTrajReader {
    frames: Vec<Vec<[f32; 3]>>,
    boxes: Vec<Box3>,
    index: usize,
    n_atoms: usize,
}

impl PdbTrajReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let (frames, boxes) = parse_pdb_frames(reader)?;
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
            boxes,
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
            let box_ = self.boxes.get(self.index).copied().unwrap_or(Box3::None);
            let coords_dst = out.start_frame(box_, None);
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
            let box_ = self.boxes.get(self.index).copied().unwrap_or(Box3::None);
            let coords_dst = out.start_frame(box_, None);
            for (i, &src_idx) in selection.iter().enumerate() {
                let coord = coords_src[src_idx as usize];
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

fn parse_pdb_frames<R: BufRead>(reader: R) -> TrajResult<(Vec<Vec<[f32; 3]>>, Vec<Box3>)> {
    let mut frames: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut boxes: Vec<Box3> = Vec::new();
    let mut current: Vec<[f32; 3]> = Vec::new();
    let mut saw_model = false;
    let mut in_model = false;
    let mut current_box = Box3::None;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("CRYST1") {
            current_box = parse_cryst1_box(&line)?;
            continue;
        }
        if line.starts_with("MODEL") {
            if !current.is_empty() {
                frames.push(current);
                boxes.push(current_box);
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
                    boxes.push(current_box);
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
        boxes.push(current_box);
    }
    if frames.is_empty() {
        return Err(TrajError::Parse("no atoms found in PDB/PDBQT".into()));
    }
    Ok((frames, boxes))
}

fn slice_trim_opt<'a>(line: &'a str, start: usize, end: usize) -> &'a str {
    line.get(start..end).unwrap_or("").trim()
}

fn parse_float(token: &str, label: &str) -> TrajResult<f32> {
    token
        .parse::<f32>()
        .map_err(|_| TrajError::Parse(format!("invalid {label} coordinate: {token}")))
}

fn parse_cryst1_box(line: &str) -> TrajResult<Box3> {
    let a = parse_float(slice_trim_opt(line, 6, 15), "CRYST1 a")? as f64;
    let b = parse_float(slice_trim_opt(line, 15, 24), "CRYST1 b")? as f64;
    let c = parse_float(slice_trim_opt(line, 24, 33), "CRYST1 c")? as f64;
    let alpha = parse_float(slice_trim_opt(line, 33, 40), "CRYST1 alpha")? as f64;
    let beta = parse_float(slice_trim_opt(line, 40, 47), "CRYST1 beta")? as f64;
    let gamma = parse_float(slice_trim_opt(line, 47, 54), "CRYST1 gamma")? as f64;
    let ninety = 90.0f64;
    if (alpha - ninety).abs() < 1.0e-3
        && (beta - ninety).abs() < 1.0e-3
        && (gamma - ninety).abs() < 1.0e-3
    {
        return Ok(Box3::Orthorhombic {
            lx: a as f32,
            ly: b as f32,
            lz: c as f32,
        });
    }

    let alpha = alpha.to_radians();
    let beta = beta.to_radians();
    let gamma = gamma.to_radians();
    let cos_alpha = alpha.cos();
    let cos_beta = beta.cos();
    let cos_gamma = gamma.cos();
    let sin_gamma = gamma.sin();
    if sin_gamma.abs() < 1.0e-8 {
        return Err(TrajError::Parse(
            "invalid CRYST1 box: gamma angle produces singular cell".into(),
        ));
    }
    let bx = b * cos_gamma;
    let by = b * sin_gamma;
    let cx = c * cos_beta;
    let cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    let cz2 = c * c - cx * cx - cy * cy;
    if cz2 < -1.0e-6 {
        return Err(TrajError::Parse(
            "invalid CRYST1 box: negative triclinic z component".into(),
        ));
    }
    let cz = cz2.max(0.0).sqrt();
    Ok(Box3::Triclinic {
        m: [
            a as f32, 0.0, 0.0, bx as f32, by as f32, 0.0, cx as f32, cy as f32, cz as f32,
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn pdb_traj_preserves_cryst1_box() {
        let content = "CRYST1   36.192   36.192   36.192  90.00  90.00  90.00 P 1           1\n\
ATOM      1  C   UNK A   1       1.000   2.000   3.000  1.00  0.00           C\n\
END\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("box.pdb");
        let mut file = File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();

        let mut reader = PdbTrajReader::open(&path).unwrap();
        let mut builder = FrameChunkBuilder::new(1, 1);
        let read = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(read, 1);
        let chunk = builder.finish().unwrap();
        assert_eq!(
            chunk.box_[0],
            Box3::Orthorhombic {
                lx: 36.192,
                ly: 36.192,
                lz: 36.192,
            }
        );
    }
}
