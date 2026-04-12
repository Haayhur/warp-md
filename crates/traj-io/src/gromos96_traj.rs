use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::TrajReader;

const NM_TO_ANGSTROM: f32 = 10.0;
const ANGSTROM_TO_NM: f32 = 0.1;
const BOX_TOL: f32 = 1.0e-6;

pub struct Gromos96TrajReader {
    path: PathBuf,
    reader: BufReader<File>,
    pending_line: Option<String>,
    pending: Option<Gromos96Frame>,
    n_atoms: usize,
}

pub struct Gromos96TrajWriter {
    writer: BufWriter<File>,
    n_atoms: usize,
}

#[derive(Clone)]
struct Gromos96Frame {
    coords: Vec<[f32; 3]>,
    box_: Box3,
    time_ps: Option<f32>,
    velocities: Option<Vec<[f32; 3]>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    None,
    Title,
    TimeStep,
    Position,
    PositionReduced,
    Velocity,
    VelocityReduced,
    Box,
}

impl Gromos96TrajReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let file = File::open(&path)?;
        let mut reader = BufReader::new(file);
        let mut pending_line = None;
        let pending = parse_next_gromos96_frame(&mut reader, &mut pending_line)?;
        let Some(frame) = pending else {
            return Err(TrajError::Parse(
                "no frames found in Gromos96 trajectory".into(),
            ));
        };
        let n_atoms = frame.coords.len();
        if n_atoms == 0 {
            return Err(TrajError::Parse(
                "no atoms found in Gromos96 trajectory".into(),
            ));
        }
        Ok(Self {
            path,
            reader,
            pending_line,
            pending: Some(frame),
            n_atoms,
        })
    }

    pub fn reset(&mut self) -> TrajResult<()> {
        let reopened = Self::open(self.path.clone())?;
        *self = reopened;
        Ok(())
    }
}

impl TrajReader for Gromos96TrajReader {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        None
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        let max_frames = max_frames.max(1);
        out.reset(self.n_atoms, max_frames);
        let mut frames = 0usize;
        while frames < max_frames {
            let frame = if let Some(frame) = self.pending.take() {
                Some(frame)
            } else {
                parse_next_gromos96_frame(&mut self.reader, &mut self.pending_line)?
            };
            let Some(frame) = frame else {
                break;
            };
            if frame.coords.len() != self.n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "Gromos96 frame atom count {} does not match first frame {}",
                    frame.coords.len(),
                    self.n_atoms
                )));
            }
            let box_ = if out.needs_box() {
                frame.box_
            } else {
                Box3::None
            };
            let time_ps = if out.needs_time() {
                frame.time_ps
            } else {
                None
            };
            let dst = out.start_frame(box_, time_ps);
            for (atom, src) in dst.iter_mut().zip(frame.coords.iter()) {
                atom[0] = src[0];
                atom[1] = src[1];
                atom[2] = src[2];
                atom[3] = 1.0;
            }
            let velocities = if out.needs_velocities() {
                frame.velocities.as_deref()
            } else {
                None
            };
            out.set_frame_extras(velocities, None, None)?;
            frames += 1;
        }
        Ok(frames)
    }
}

impl Gromos96TrajWriter {
    pub fn create(path: impl Into<PathBuf>, n_atoms: usize) -> TrajResult<Self> {
        if n_atoms == 0 {
            return Err(TrajError::Invalid(
                "Gromos96 trajectory writer requires at least one atom".into(),
            ));
        }
        let path = path.into();
        let file = File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            n_atoms,
        })
    }

    pub fn write_frame(
        &mut self,
        coords: &[[f32; 3]],
        box_: Box3,
        step: usize,
        time_ps: Option<f32>,
        velocities: Option<&[[f32; 3]]>,
    ) -> TrajResult<()> {
        if coords.len() != self.n_atoms {
            return Err(TrajError::Mismatch(format!(
                "frame atom count {} does not match writer atom count {}",
                coords.len(),
                self.n_atoms
            )));
        }
        if let Some(velocities) = velocities {
            if velocities.len() != self.n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "velocity atom count {} does not match writer atom count {}",
                    velocities.len(),
                    self.n_atoms
                )));
            }
        }

        writeln!(self.writer, "TITLE")?;
        writeln!(
            self.writer,
            "Generated by warp_md t={:10.5} step= {}",
            time_ps.unwrap_or(step as f32),
            step
        )?;
        writeln!(self.writer, "END")?;
        writeln!(self.writer, "TIMESTEP")?;
        writeln!(
            self.writer,
            "{step:>15}{:>15.6}",
            time_ps.unwrap_or(step as f32)
        )?;
        writeln!(self.writer, "END")?;
        writeln!(self.writer, "POSITIONRED")?;
        for coord in coords {
            writeln!(
                self.writer,
                "{:>15.9}{:>15.9}{:>15.9}",
                coord[0] * ANGSTROM_TO_NM,
                coord[1] * ANGSTROM_TO_NM,
                coord[2] * ANGSTROM_TO_NM
            )?;
        }
        writeln!(self.writer, "END")?;
        if let Some(velocities) = velocities {
            writeln!(self.writer, "VELOCITYRED")?;
            for velocity in velocities {
                writeln!(
                    self.writer,
                    "{:>15.9}{:>15.9}{:>15.9}",
                    velocity[0] * ANGSTROM_TO_NM,
                    velocity[1] * ANGSTROM_TO_NM,
                    velocity[2] * ANGSTROM_TO_NM
                )?;
            }
            writeln!(self.writer, "END")?;
        }
        if let Some(fields) = gromos96_box_fields(box_) {
            writeln!(self.writer, "BOX")?;
            for value in fields {
                write!(self.writer, "{value:>15.9}")?;
            }
            writeln!(self.writer)?;
            writeln!(self.writer, "END")?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> TrajResult<()> {
        self.writer.flush()?;
        Ok(())
    }
}

fn parse_next_gromos96_frame<R: BufRead>(
    reader: &mut R,
    pending_line: &mut Option<String>,
) -> TrajResult<Option<Gromos96Frame>> {
    let mut frame = Gromos96Frame {
        coords: Vec::new(),
        box_: Box3::None,
        time_ps: None,
        velocities: None,
    };
    let mut saw_any = false;
    let mut title_line: Option<String> = None;
    let mut section = Section::None;
    loop {
        let Some(line) = next_line(reader, pending_line)? else {
            if !saw_any {
                return Ok(None);
            }
            return finalize_gromos96_frame(frame, title_line);
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match section {
            Section::None => match trimmed {
                "TITLE" => {
                    if !frame.coords.is_empty() {
                        *pending_line = Some(line);
                        return finalize_gromos96_frame(frame, title_line);
                    }
                    saw_any = true;
                    section = Section::Title;
                }
                "TIMESTEP" => {
                    if !frame.coords.is_empty() {
                        *pending_line = Some(line);
                        return finalize_gromos96_frame(frame, title_line);
                    }
                    saw_any = true;
                    section = Section::TimeStep;
                }
                "POSITION" => {
                    if !frame.coords.is_empty() {
                        *pending_line = Some(line);
                        return finalize_gromos96_frame(frame, title_line);
                    }
                    saw_any = true;
                    section = Section::Position;
                }
                "POSITIONRED" => {
                    if !frame.coords.is_empty() {
                        *pending_line = Some(line);
                        return finalize_gromos96_frame(frame, title_line);
                    }
                    saw_any = true;
                    section = Section::PositionReduced;
                }
                "VELOCITY" => {
                    saw_any = true;
                    frame.velocities.get_or_insert_with(Vec::new);
                    section = Section::Velocity;
                }
                "VELOCITYRED" => {
                    saw_any = true;
                    frame.velocities.get_or_insert_with(Vec::new);
                    section = Section::VelocityReduced;
                }
                "BOX" => {
                    saw_any = true;
                    section = Section::Box;
                }
                "END" => {}
                _ => {}
            },
            Section::Title => {
                if trimmed == "END" {
                    section = Section::None;
                } else if title_line.is_none() {
                    title_line = Some(trimmed.to_string());
                }
            }
            Section::TimeStep => {
                if trimmed == "END" {
                    section = Section::None;
                } else {
                    frame.time_ps = parse_gromos96_time_ps(trimmed)?;
                }
            }
            Section::Position => {
                if trimmed == "END" {
                    section = Section::None;
                } else {
                    frame
                        .coords
                        .push(parse_gromos96_triplet(trimmed, "POSITION")?);
                }
            }
            Section::PositionReduced => {
                if trimmed == "END" {
                    section = Section::None;
                } else {
                    frame
                        .coords
                        .push(parse_gromos96_triplet(trimmed, "POSITIONRED")?);
                }
            }
            Section::Velocity => {
                if trimmed == "END" {
                    section = Section::None;
                } else {
                    frame
                        .velocities
                        .get_or_insert_with(Vec::new)
                        .push(parse_gromos96_triplet(trimmed, "VELOCITY")?);
                }
            }
            Section::VelocityReduced => {
                if trimmed == "END" {
                    section = Section::None;
                } else {
                    frame
                        .velocities
                        .get_or_insert_with(Vec::new)
                        .push(parse_gromos96_triplet(trimmed, "VELOCITYRED")?);
                }
            }
            Section::Box => {
                if trimmed == "END" {
                    section = Section::None;
                } else {
                    frame.box_ = parse_gromos96_box_line(trimmed)?;
                }
            }
        }
    }
}

fn finalize_gromos96_frame(
    mut frame: Gromos96Frame,
    title_line: Option<String>,
) -> TrajResult<Option<Gromos96Frame>> {
    if frame.coords.is_empty() {
        return if title_line.is_none() && frame.time_ps.is_none() {
            Ok(None)
        } else {
            Err(TrajError::Parse(
                "Gromos96 frame missing POSITION/POSITIONRED section".into(),
            ))
        };
    }
    if frame.time_ps.is_none() {
        frame.time_ps = title_line.as_deref().and_then(parse_title_time_ps);
    }
    if let Some(velocities) = frame.velocities.as_ref() {
        if velocities.len() != frame.coords.len() {
            return Err(TrajError::Parse(format!(
                "Gromos96 velocity count {} does not match coordinate count {}",
                velocities.len(),
                frame.coords.len()
            )));
        }
    }
    Ok(Some(frame))
}

fn next_line<R: BufRead>(
    reader: &mut R,
    pending_line: &mut Option<String>,
) -> TrajResult<Option<String>> {
    if let Some(line) = pending_line.take() {
        return Ok(Some(line));
    }
    let mut line = String::new();
    let read = reader.read_line(&mut line)?;
    if read == 0 {
        return Ok(None);
    }
    Ok(Some(line.trim_end_matches(['\r', '\n']).to_string()))
}

fn parse_gromos96_time_ps(line: &str) -> TrajResult<Option<f32>> {
    let Some(token) = line.split_whitespace().last() else {
        return Ok(None);
    };
    let value = token
        .parse::<f32>()
        .map_err(|_| TrajError::Parse(format!("invalid Gromos96 time '{token}'")))?;
    Ok(Some(value))
}

fn parse_gromos96_triplet(line: &str, label: &str) -> TrajResult<[f32; 3]> {
    let parts = line.split_whitespace().collect::<Vec<_>>();
    if parts.len() < 3 {
        return Err(TrajError::Parse(format!(
            "invalid Gromos96 {label} line: {line}"
        )));
    }
    let tail = &parts[parts.len() - 3..];
    Ok([
        parse_gromos96_scalar(tail[0], label)?,
        parse_gromos96_scalar(tail[1], label)?,
        parse_gromos96_scalar(tail[2], label)?,
    ])
}

fn parse_gromos96_scalar(token: &str, label: &str) -> TrajResult<f32> {
    token
        .parse::<f32>()
        .map(|value| value * NM_TO_ANGSTROM)
        .map_err(|_| TrajError::Parse(format!("invalid Gromos96 {label} value '{token}'")))
}

fn parse_gromos96_box_line(line: &str) -> TrajResult<Box3> {
    let values = line
        .split_whitespace()
        .map(|token| {
            token
                .parse::<f32>()
                .map(|value| value * NM_TO_ANGSTROM)
                .map_err(|_| TrajError::Parse(format!("invalid Gromos96 box value '{token}'")))
        })
        .collect::<Result<Vec<_>, _>>()?;
    match values.as_slice() {
        [xx, yy, zz] => Ok(Box3::Orthorhombic {
            lx: *xx,
            ly: *yy,
            lz: *zz,
        }),
        [xx, yy, zz, xy, xz, yx, yz, zx, zy] => Ok(Box3::Triclinic {
            m: [*xx, *xy, *xz, *yx, *yy, *yz, *zx, *zy, *zz],
        }),
        _ => Err(TrajError::Parse(
            "Gromos96 BOX section must contain 3 or 9 values".into(),
        )),
    }
}

fn gromos96_box_fields(box_: Box3) -> Option<Vec<f32>> {
    match box_ {
        Box3::None => None,
        Box3::Orthorhombic { lx, ly, lz } => Some(vec![
            lx * ANGSTROM_TO_NM,
            ly * ANGSTROM_TO_NM,
            lz * ANGSTROM_TO_NM,
        ]),
        Box3::Triclinic { m } => {
            if is_orthorhombic(&m) {
                Some(vec![
                    m[0] * ANGSTROM_TO_NM,
                    m[4] * ANGSTROM_TO_NM,
                    m[8] * ANGSTROM_TO_NM,
                ])
            } else {
                Some(vec![
                    m[0] * ANGSTROM_TO_NM,
                    m[4] * ANGSTROM_TO_NM,
                    m[8] * ANGSTROM_TO_NM,
                    m[1] * ANGSTROM_TO_NM,
                    m[2] * ANGSTROM_TO_NM,
                    m[3] * ANGSTROM_TO_NM,
                    m[5] * ANGSTROM_TO_NM,
                    m[6] * ANGSTROM_TO_NM,
                    m[7] * ANGSTROM_TO_NM,
                ])
            }
        }
    }
}

fn is_orthorhombic(matrix: &[f32; 9]) -> bool {
    matrix[1].abs() <= BOX_TOL
        && matrix[2].abs() <= BOX_TOL
        && matrix[3].abs() <= BOX_TOL
        && matrix[5].abs() <= BOX_TOL
        && matrix[6].abs() <= BOX_TOL
        && matrix[7].abs() <= BOX_TOL
}

fn parse_title_time_ps(title: &str) -> Option<f32> {
    let (_, tail) = title.split_once("t=")?;
    tail.trim_start()
        .split_whitespace()
        .next()
        .and_then(|value| value.parse::<f32>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xtc::XtcReader;
    use std::path::Path;

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../python/warp_md/tests/fixtures/gro_g96")
            .join(name)
    }

    #[test]
    fn gromos96_reader_matches_xtc_fixture() {
        let mut g96 = Gromos96TrajReader::open(fixture("spc2-traj.g96")).unwrap();
        let mut xtc = XtcReader::open(fixture("spc2-traj.xtc")).unwrap();
        let mut g96_builder = FrameChunkBuilder::new(6, 4);
        g96_builder.set_requirements(true, true);
        g96_builder.set_optional_requirements(true, false, false);
        let mut xtc_builder = FrameChunkBuilder::new(6, 4);
        xtc_builder.set_requirements(true, true);
        let g96_read = g96.read_chunk(4, &mut g96_builder).unwrap();
        let xtc_read = xtc.read_chunk(4, &mut xtc_builder).unwrap();
        assert_eq!(g96_read, 2);
        assert_eq!(xtc_read, 2);
        let g96_chunk = g96_builder.finish().unwrap();
        let xtc_chunk = xtc_builder.finish().unwrap();
        assert_eq!(g96_chunk.coords.len(), xtc_chunk.coords.len());
        for (lhs, rhs) in g96_chunk.coords.iter().zip(xtc_chunk.coords.iter()) {
            assert!((lhs[0] - rhs[0]).abs() < 1.0e-4);
            assert!((lhs[1] - rhs[1]).abs() < 1.0e-4);
            assert!((lhs[2] - rhs[2]).abs() < 1.0e-4);
        }
        assert_eq!(g96_chunk.box_, xtc_chunk.box_);
        assert_eq!(g96_chunk.time_ps, xtc_chunk.time_ps);
        let velocities = g96_chunk.velocities.unwrap();
        assert_eq!(velocities.len(), 12);
        assert!((velocities[0][0] - 5.69).abs() < 1.0e-4);
        assert!((velocities[11][2] - 84.65).abs() < 1.0e-3);
    }

    #[test]
    fn gromos96_writer_roundtrips_frames() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.g96");
        let mut writer = Gromos96TrajWriter::create(&path, 2).unwrap();
        writer
            .write_frame(
                &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                Box3::Orthorhombic {
                    lx: 20.0,
                    ly: 21.0,
                    lz: 22.0,
                },
                0,
                Some(0.25),
                Some(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            )
            .unwrap();
        writer
            .write_frame(
                &[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                Box3::Triclinic {
                    m: [23.0, 1.0, 2.0, 3.0, 24.0, 4.0, 5.0, 6.0, 25.0],
                },
                1,
                Some(1.75),
                None,
            )
            .unwrap();
        writer.flush().unwrap();

        let mut reader = Gromos96TrajReader::open(&path).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 2);
        builder.set_requirements(true, true);
        builder.set_optional_requirements(true, false, false);
        assert_eq!(reader.read_chunk(2, &mut builder).unwrap(), 2);
        let chunk = builder.finish().unwrap();
        assert_eq!(chunk.n_frames, 2);
        assert!((chunk.coords[0][0] - 1.0).abs() < 1.0e-5);
        assert!((chunk.coords[3][2] - 12.0).abs() < 1.0e-5);
        assert_eq!(chunk.time_ps.unwrap(), vec![0.25, 1.75]);
        let velocities = chunk.velocities.unwrap();
        assert!((velocities[0][0] - 0.1).abs() < 1.0e-5);
        assert_eq!(velocities[2], [0.0, 0.0, 0.0]);
        assert_eq!(
            chunk.box_[1],
            Box3::Triclinic {
                m: [23.0, 1.0, 2.0, 3.0, 24.0, 4.0, 5.0, 6.0, 25.0],
            }
        );
    }
}
