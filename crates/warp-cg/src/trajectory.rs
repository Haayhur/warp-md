use anyhow::{anyhow, Result};
use chemfiles::{Atom, Frame, Trajectory};
use std::collections::HashMap;
use std::path::Path;

use traj_core::frame::{Box3, FrameChunkBuilder};
use traj_io::cpt::{CptReader, CptWriter};
use traj_io::dcd::{DcdReader, DcdWriter};
use traj_io::gro_traj::{GroTrajReader, GroTrajWriter};
use traj_io::gromos96_traj::{Gromos96TrajReader, Gromos96TrajWriter};
use traj_io::h5md::{H5mdReader, H5mdWriter};
use traj_io::pdb_traj::PdbTrajReader;
use traj_io::tng::TngReader;
use traj_io::trr::{TrrReader, TrrWriter};
use traj_io::xtc::{XtcReader, XtcWriter};
use traj_io::TrajReader;
use warp_structure::io::read_system_auto;

use crate::parameters::{
    angle_deg, bonded_stats_from_values, bonded_term_definitions, dihedral_deg, AngleStats,
    BondStats, DihedralStats,
};

pub struct BeadMapping {
    pub bead_names: Vec<String>,
    pub atom_indices: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Default)]
pub struct NativeTrajectoryOptions {
    pub topology: Option<String>,
    pub topology_format: Option<String>,
    pub format: Option<String>,
    pub start: Option<usize>,
    pub stop: Option<usize>,
    pub stride: Option<usize>,
    pub length_scale: Option<f32>,
    pub target_selection: Option<String>,
    pub atom_indices: Option<Vec<usize>>,
    pub mass_weighted: bool,
    pub chunk_frames: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct TrajectoryMapReport {
    pub frames_read: usize,
    pub frames_written: usize,
    pub first_cg_coords: Option<Vec<[f32; 3]>>,
    pub bond_stats: Vec<BondStats>,
    pub angle_stats: Vec<AngleStats>,
    pub dihedral_stats: Vec<DihedralStats>,
}

pub fn map_trajectory(input_path: &str, output_path: &str, mapping: &BeadMapping) -> Result<()> {
    map_trajectory_first_frame(input_path, output_path, mapping).map(|_| ())
}

pub fn map_trajectory_first_frame(
    input_path: &str,
    output_path: &str,
    mapping: &BeadMapping,
) -> Result<Option<Vec<[f32; 3]>>> {
    let mut input = Trajectory::open(input_path, 'r')?;
    let mut output = Trajectory::open(output_path, 'w')?;

    let mut frame = Frame::new();
    let mut first_cg_coords = None;
    while let Ok(_) = input.read(&mut frame) {
        let mut cg_frame = Frame::new();

        let positions = frame.positions();
        let mut cg_coords = Vec::new();

        for (i, bead_atoms) in mapping.atom_indices.iter().enumerate() {
            let mut cog = [0.0; 3];
            for &atom_idx in bead_atoms {
                if atom_idx < positions.len() {
                    let pos = positions[atom_idx];
                    cog[0] += pos[0];
                    cog[1] += pos[1];
                    cog[2] += pos[2];
                }
            }
            let count = bead_atoms.len() as f64;
            if count > 0.0 {
                cog[0] /= count;
                cog[1] /= count;
                cog[2] /= count;
            }

            let atom = Atom::new(mapping.bead_names[i].as_str());
            cg_frame.add_atom(&atom, cog, None);
            cg_coords.push([cog[0] as f32, cog[1] as f32, cog[2] as f32]);
        }

        if first_cg_coords.is_none() {
            first_cg_coords = Some(cg_coords);
        }
        output.write(&cg_frame)?;
    }

    Ok(first_cg_coords)
}

pub fn map_native_trajectory(
    input_path: &Path,
    output_path: Option<&Path>,
    mapping: &BeadMapping,
    connections: &[(usize, usize)],
    options: &NativeTrajectoryOptions,
) -> Result<TrajectoryMapReport> {
    let mut reader = open_reader(input_path, options.format.as_deref(), options.length_scale)?;
    let source_atom_count = reader.n_atoms();
    let source_indices = resolve_source_indices(source_atom_count, options)?;
    let translated_mapping = translate_mapping(mapping, &source_indices)?;
    let bead_weights = if options.mass_weighted {
        Some(resolve_bead_weights(
            options,
            source_atom_count,
            &translated_mapping,
        )?)
    } else {
        None
    };
    let chunk_frames = options.chunk_frames.unwrap_or(128).max(1);
    let mut builder = FrameChunkBuilder::new(source_atom_count, chunk_frames);
    builder.set_requirements(true, true);

    let mut writer = match output_path {
        Some(path) => Some(open_writer(path, mapping.bead_names.len())?),
        None => None,
    };
    let mut bond_values: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    for &(i, j) in connections {
        let key = if i < j { (i, j) } else { (j, i) };
        bond_values.insert(key, Vec::new());
    }
    let (angle_defs, dihedral_defs) =
        bonded_term_definitions(mapping.bead_names.len(), connections);
    let mut angle_values: HashMap<(usize, usize, usize), Vec<f64>> = angle_defs
        .iter()
        .copied()
        .map(|key| (key, Vec::new()))
        .collect();
    let mut dihedral_values: HashMap<(usize, usize, usize, usize), Vec<f64>> = dihedral_defs
        .iter()
        .copied()
        .map(|key| (key, Vec::new()))
        .collect();

    let start = options.start.unwrap_or(0);
    let stop = options.stop.unwrap_or(usize::MAX);
    let stride = options.stride.unwrap_or(1).max(1);
    let mut frames_read = 0usize;
    let mut frames_written = 0usize;
    let mut first_cg_coords = None;

    loop {
        let read = reader.read_chunk(chunk_frames, &mut builder)?;
        if read == 0 {
            break;
        }
        let chunk = builder.finish_take()?;
        for frame_idx in 0..chunk.n_frames {
            let global_frame = frames_read + frame_idx;
            if global_frame < start || global_frame >= stop || (global_frame - start) % stride != 0
            {
                continue;
            }
            let source_frame =
                &chunk.coords[frame_idx * chunk.n_atoms..(frame_idx + 1) * chunk.n_atoms];
            let cg_coords = map_frame(source_frame, &translated_mapping, bead_weights.as_deref());
            if first_cg_coords.is_none() {
                first_cg_coords = Some(cg_coords.clone());
            }
            accumulate_bond_values(&mut bond_values, &cg_coords);
            accumulate_angle_values(&mut angle_values, &cg_coords);
            accumulate_dihedral_values(&mut dihedral_values, &cg_coords);
            if let Some(writer) = writer.as_mut() {
                if writer.is_single_frame() && frames_written > 0 {
                    return Err(anyhow!(
                        "mapped trajectory output format cpt supports exactly one selected frame; use start/stop/stride to select one frame"
                    ));
                }
                let box_ = chunk.box_.get(frame_idx).copied().unwrap_or(Box3::None);
                let time_ps = chunk
                    .time_ps
                    .as_ref()
                    .and_then(|values| values.get(frame_idx).copied());
                writer.write_frame(&cg_coords, box_, global_frame, time_ps)?;
                frames_written += 1;
            }
        }
        frames_read += read;
        if frames_read >= stop {
            break;
        }
        builder.reset(source_atom_count, chunk_frames);
    }
    if let Some(writer) = writer.as_mut() {
        writer.flush()?;
    }

    let bonded_stats = bonded_stats_from_values(bond_values, angle_values, dihedral_values);
    Ok(TrajectoryMapReport {
        frames_read,
        frames_written,
        first_cg_coords,
        bond_stats: bonded_stats.bonds,
        angle_stats: bonded_stats.angles,
        dihedral_stats: bonded_stats.dihedrals,
    })
}

fn resolve_source_indices(
    source_atom_count: usize,
    options: &NativeTrajectoryOptions,
) -> Result<Vec<usize>> {
    if let Some(indices) = &options.atom_indices {
        if indices.is_empty() {
            return Err(anyhow!("trajectory_source.atom_indices must not be empty"));
        }
        validate_indices(indices, source_atom_count)?;
        return Ok(indices.clone());
    }

    let Some(selection_expr) = options.target_selection.as_deref() else {
        return Ok((0..source_atom_count).collect());
    };
    let topology = options
        .topology
        .as_deref()
        .ok_or_else(|| anyhow!("trajectory_source.target_selection requires topology"))?;
    let mut system = read_system_auto(Path::new(topology), options.topology_format.as_deref())
        .map_err(|err| anyhow!("failed to read topology for trajectory selection: {err}"))?;
    if system.n_atoms() != source_atom_count {
        return Err(anyhow!(
            "topology atom count {} does not match trajectory atom count {}",
            system.n_atoms(),
            source_atom_count
        ));
    }
    let selection = system
        .select(selection_expr)
        .map_err(|err| anyhow!("invalid trajectory_source.target_selection: {err}"))?;
    let indices: Vec<usize> = selection.indices.iter().map(|idx| *idx as usize).collect();
    if indices.is_empty() {
        return Err(anyhow!(
            "trajectory_source.target_selection selected no atoms"
        ));
    }
    Ok(indices)
}

fn translate_mapping(mapping: &BeadMapping, source_indices: &[usize]) -> Result<Vec<Vec<usize>>> {
    mapping
        .atom_indices
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|&relative_idx| {
                    source_indices.get(relative_idx).copied().ok_or_else(|| {
                        anyhow!(
                            "mapping atom index {} is outside selected target atom count {}",
                            relative_idx,
                            source_indices.len()
                        )
                    })
                })
                .collect()
        })
        .collect()
}

fn resolve_bead_weights(
    options: &NativeTrajectoryOptions,
    source_atom_count: usize,
    translated_mapping: &[Vec<usize>],
) -> Result<Vec<Vec<f32>>> {
    let topology = options
        .topology
        .as_deref()
        .ok_or_else(|| anyhow!("mass-weighted bead centers require topology"))?;
    let system = read_system_auto(Path::new(topology), options.topology_format.as_deref())
        .map_err(|err| anyhow!("failed to read topology for mass-weighted mapping: {err}"))?;
    if system.n_atoms() != source_atom_count {
        return Err(anyhow!(
            "topology atom count {} does not match trajectory atom count {}",
            system.n_atoms(),
            source_atom_count
        ));
    }
    Ok(translated_mapping
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|&atom_idx| {
                    let mass = system.atoms.mass.get(atom_idx).copied().unwrap_or(0.0);
                    if mass.is_finite() && mass > 0.0 {
                        mass
                    } else {
                        1.0
                    }
                })
                .collect()
        })
        .collect())
}

fn validate_indices(indices: &[usize], atom_count: usize) -> Result<()> {
    for &idx in indices {
        if idx >= atom_count {
            return Err(anyhow!(
                "trajectory_source.atom_indices contains index {} but trajectory has {} atoms",
                idx,
                atom_count
            ));
        }
    }
    Ok(())
}

fn map_frame(
    source_frame: &[[f32; 4]],
    mapping: &[Vec<usize>],
    weights: Option<&[Vec<f32>]>,
) -> Vec<[f32; 3]> {
    mapping
        .iter()
        .enumerate()
        .map(|(bead_idx, bead_atoms)| {
            let mut cog = [0.0f32; 3];
            let mut weight_sum = 0.0f32;
            for (local_idx, &atom_idx) in bead_atoms.iter().enumerate() {
                if let Some(pos) = source_frame.get(atom_idx) {
                    let weight = weights
                        .and_then(|values| values.get(bead_idx))
                        .and_then(|values| values.get(local_idx))
                        .copied()
                        .unwrap_or(1.0);
                    cog[0] += pos[0] * weight;
                    cog[1] += pos[1] * weight;
                    cog[2] += pos[2] * weight;
                    weight_sum += weight;
                }
            }
            if weight_sum > 0.0 {
                cog[0] /= weight_sum;
                cog[1] /= weight_sum;
                cog[2] /= weight_sum;
            }
            cog
        })
        .collect()
}

fn accumulate_bond_values(
    bond_values: &mut HashMap<(usize, usize), Vec<f64>>,
    positions: &[[f32; 3]],
) {
    for (&(i, j), values) in bond_values.iter_mut() {
        if i < positions.len() && j < positions.len() {
            let dx = f64::from(positions[i][0] - positions[j][0]);
            let dy = f64::from(positions[i][1] - positions[j][1]);
            let dz = f64::from(positions[i][2] - positions[j][2]);
            values.push((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
}

fn accumulate_angle_values(
    angle_values: &mut HashMap<(usize, usize, usize), Vec<f64>>,
    positions: &[[f32; 3]],
) {
    for (&(i, j, k), values) in angle_values.iter_mut() {
        if i < positions.len() && j < positions.len() && k < positions.len() {
            values.push(angle_deg(positions[i], positions[j], positions[k]));
        }
    }
}

fn accumulate_dihedral_values(
    dihedral_values: &mut HashMap<(usize, usize, usize, usize), Vec<f64>>,
    positions: &[[f32; 3]],
) {
    for (&(i, j, k, l), values) in dihedral_values.iter_mut() {
        if i < positions.len() && j < positions.len() && k < positions.len() && l < positions.len()
        {
            values.push(dihedral_deg(
                positions[i],
                positions[j],
                positions[k],
                positions[l],
            ));
        }
    }
}

fn open_reader(
    path: &Path,
    format: Option<&str>,
    length_scale: Option<f32>,
) -> Result<Box<dyn TrajReader>> {
    let format = format_token(path, format)?;
    let reader: Box<dyn TrajReader> = match format.as_str() {
        "dcd" => Box::new(DcdReader::open(path, length_scale.unwrap_or(1.0))?),
        "xtc" => Box::new(XtcReader::open(path)?),
        "gro" => Box::new(GroTrajReader::open(path)?),
        "g96" => Box::new(Gromos96TrajReader::open(path)?),
        "h5md" => Box::new(H5mdReader::open(path)?),
        "tng" => Box::new(TngReader::open(path)?),
        "cpt" => Box::new(CptReader::open(path)?),
        "trr" => Box::new(TrrReader::open(path)?),
        "pdb" | "pdbqt" => Box::new(PdbTrajReader::open(path)?),
        other => return Err(anyhow!("unsupported trajectory format: {other}")),
    };
    Ok(reader)
}

fn open_writer(path: &Path, n_atoms: usize) -> Result<NativeWriter> {
    let format = format_token(path, None)?;
    match format.as_str() {
        "xtc" => Ok(NativeWriter::Xtc(XtcWriter::create(path, n_atoms)?)),
        "dcd" => Ok(NativeWriter::Dcd(DcdWriter::create(path, n_atoms, 0)?)),
        "gro" => Ok(NativeWriter::Gro(GroTrajWriter::create(path, n_atoms)?)),
        "g96" => Ok(NativeWriter::G96(Gromos96TrajWriter::create(path, n_atoms)?)),
        "cpt" => Ok(NativeWriter::Cpt(CptWriter::create(path, n_atoms)?)),
        "trr" => Ok(NativeWriter::Trr(TrrWriter::create(path, n_atoms)?)),
        "h5md" => Ok(NativeWriter::H5md(H5mdWriter::create(path, n_atoms)?)),
        other => Err(anyhow!(
            "unsupported mapped trajectory output format: {other}; expected xtc, dcd, gro, g96, cpt, trr, or h5md"
        )),
    }
}

fn format_token(path: &Path, format: Option<&str>) -> Result<String> {
    let token = format
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| {
            path.extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
        });
    if token.is_empty() {
        return Err(anyhow!("trajectory format could not be inferred from path"));
    }
    Ok(token)
}

enum NativeWriter {
    Xtc(XtcWriter),
    Dcd(DcdWriter),
    Gro(GroTrajWriter),
    G96(Gromos96TrajWriter),
    Cpt(CptWriter),
    Trr(TrrWriter),
    H5md(H5mdWriter),
}

impl NativeWriter {
    fn is_single_frame(&self) -> bool {
        matches!(self, NativeWriter::Cpt(_))
    }

    fn write_frame(
        &mut self,
        coords: &[[f32; 3]],
        box_: Box3,
        step: usize,
        time_ps: Option<f32>,
    ) -> Result<()> {
        match self {
            NativeWriter::Xtc(writer) => writer.write_frame(coords, box_, step, time_ps)?,
            NativeWriter::Dcd(writer) => writer.write_frame(coords, box_)?,
            NativeWriter::Gro(writer) => writer.write_frame(coords, box_, step, time_ps, None)?,
            NativeWriter::G96(writer) => writer.write_frame(coords, box_, step, time_ps, None)?,
            NativeWriter::Cpt(writer) => {
                writer.write_frame(coords, box_, step, time_ps, None, None)?
            }
            NativeWriter::Trr(writer) => {
                writer.write_frame(coords, box_, step, time_ps, None, None, None)?
            }
            NativeWriter::H5md(writer) => {
                writer.write_frame(coords, box_, step, time_ps, None, None)?
            }
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        if let NativeWriter::Xtc(writer) = self {
            writer.flush()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn map_frame_can_use_mass_weighted_centers() {
        let source = vec![[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]];
        let mapping = vec![vec![0, 1]];
        let weights = vec![vec![1.0, 3.0]];

        let cog = map_frame(&source, &mapping, None);
        let com = map_frame(&source, &mapping, Some(&weights));

        assert!((cog[0][0] - 5.0).abs() < 1.0e-6);
        assert!((com[0][0] - 7.5).abs() < 1.0e-6);
    }

    #[test]
    fn native_writer_supports_text_and_checkpoint_formats() {
        let dir = tempfile::tempdir().unwrap();
        let coords = [[1.0, 2.0, 3.0]];
        for ext in ["gro", "g96", "cpt"] {
            let path = dir.path().join(format!("mapped.{ext}"));
            let mut writer = open_writer(&path, 1).unwrap();
            writer
                .write_frame(&coords, Box3::None, 0, Some(0.0))
                .unwrap();
            assert!(path.is_file());
        }
    }

    #[test]
    fn cpt_writer_reports_single_frame_contract() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mapped.cpt");
        let writer = open_writer(&path, 1).unwrap();

        assert!(writer.is_single_frame());
    }

    #[test]
    fn native_mapping_uses_target_selection_in_solvated_trajectory() {
        let dir = tempfile::tempdir().unwrap();
        let top = dir.path().join("solvated.gro");
        fs::write(
            &top,
            concat!(
                "solvated ethanol\n",
                "5\n",
                "    1EOH     C1    1   0.000   0.000   0.000\n",
                "    1EOH     C2    2   0.100   0.000   0.000\n",
                "    1EOH     O1    3   0.200   0.000   0.000\n",
                "    2SOL     OW    4   0.500   0.500   0.500\n",
                "    2SOL    HW1    5   0.600   0.500   0.500\n",
                "   1.00000 1.00000 1.00000\n",
            ),
        )
        .unwrap();
        let traj = dir.path().join("solvated.xtc");
        let mut writer = XtcWriter::create(&traj, 5).unwrap();
        writer
            .write_frame(
                &[
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [5.0, 5.0, 5.0],
                    [6.0, 5.0, 5.0],
                ],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 10.0,
                    lz: 10.0,
                },
                0,
                Some(0.0),
            )
            .unwrap();
        writer
            .write_frame(
                &[
                    [1.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [3.0, 0.0, 0.0],
                    [5.0, 5.0, 5.0],
                    [6.0, 5.0, 5.0],
                ],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 10.0,
                    lz: 10.0,
                },
                1,
                Some(1.0),
            )
            .unwrap();
        writer.flush().unwrap();

        let out = dir.path().join("ethanol_cg.xtc");
        let report = map_native_trajectory(
            &traj,
            Some(&out),
            &BeadMapping {
                bead_names: vec!["SP1".to_string()],
                atom_indices: vec![vec![0, 1, 2]],
            },
            &[],
            &NativeTrajectoryOptions {
                topology: Some(top.to_string_lossy().to_string()),
                topology_format: Some("gro".to_string()),
                format: Some("xtc".to_string()),
                target_selection: Some("resname EOH".to_string()),
                ..NativeTrajectoryOptions::default()
            },
        )
        .unwrap();

        assert_eq!(report.frames_read, 2);
        assert_eq!(report.frames_written, 2);
        assert!(out.is_file());
    }
}
