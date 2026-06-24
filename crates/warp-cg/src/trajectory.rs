use anyhow::{anyhow, Result};
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
use traj_io::xyz_traj::XyzTrajReader;
use traj_io::TrajReader;
use warp_structure::io::read_system_auto;

use crate::bonded_terms::BondedTermSet;
use crate::parameters::{
    bonded_stats_from_values, AngleStats, BondStats, BondedValueSeries, DihedralStats,
};
use crate::trajectory_bonded::{
    accumulate_angle_group_values, accumulate_bond_group_values, accumulate_dihedral_group_values,
    angle_group_series, bond_group_series, dihedral_group_series, empty_angle_group_values,
    empty_bond_group_values, empty_dihedral_group_values, single_angle_values, single_bond_values,
    single_dihedral_values,
};

#[path = "trajectory_metrics.rs"]
mod trajectory_metrics;
#[path = "trajectory_virtual_sites.rs"]
mod trajectory_virtual_sites;
#[path = "trajectory_whole.rs"]
mod trajectory_whole;
use trajectory_metrics::{
    radius_of_gyration, sasa_sphere_points, shrake_rupley_sasa_with_points, MetricStats,
    RunningMetricStats, DEFAULT_SASA_FALLBACK_RADIUS_NM, DEFAULT_SASA_PROBE_RADIUS_NM,
    DEFAULT_SASA_SPHERE_POINTS,
};
use trajectory_virtual_sites::{apply_virtual_sites, coordinate_count_with_virtual_sites};
use trajectory_whole::{
    bonded_whole_connections, make_source_whole_by_bonded_connectivity,
    make_whole_by_bonded_connectivity, resolve_source_whole_connections,
};

#[derive(Debug, Clone)]
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
    pub make_whole: bool,
    pub chunk_frames: Option<usize>,
    pub sasa: NativeSasaOptions,
}

#[derive(Debug, Clone)]
pub struct NativeSasaOptions {
    pub probe_radius_nm: f64,
    pub n_sphere_points: usize,
    pub radii_nm: Option<Vec<f64>>,
    pub fallback_radius_nm: f64,
}

impl Default for NativeSasaOptions {
    fn default() -> Self {
        Self {
            probe_radius_nm: DEFAULT_SASA_PROBE_RADIUS_NM,
            n_sphere_points: DEFAULT_SASA_SPHERE_POINTS,
            radii_nm: None,
            fallback_radius_nm: DEFAULT_SASA_FALLBACK_RADIUS_NM,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrajectoryMapReport {
    pub frames_read: usize,
    pub frames_written: usize,
    pub first_cg_coords: Option<Vec<[f32; 3]>>,
    pub bond_stats: Vec<BondStats>,
    pub angle_stats: Vec<AngleStats>,
    pub dihedral_stats: Vec<DihedralStats>,
    pub bonded_values: BondedValueSeries,
    pub rg_stats: Option<MetricStats>,
    pub sasa_stats: Option<MetricStats>,
}

pub fn map_trajectory(input_path: &str, output_path: &str, mapping: &BeadMapping) -> Result<()> {
    map_trajectory_first_frame(input_path, output_path, mapping).map(|_| ())
}

pub fn map_trajectory_first_frame(
    input_path: &str,
    output_path: &str,
    mapping: &BeadMapping,
) -> Result<Option<Vec<[f32; 3]>>> {
    let report = map_native_trajectory(
        Path::new(input_path),
        Some(Path::new(output_path)),
        mapping,
        &[],
        &NativeTrajectoryOptions::default(),
    )?;
    Ok(report.first_cg_coords)
}

pub fn map_native_trajectory(
    input_path: &Path,
    output_path: Option<&Path>,
    mapping: &BeadMapping,
    connections: &[(usize, usize)],
    options: &NativeTrajectoryOptions,
) -> Result<TrajectoryMapReport> {
    let terms = BondedTermSet::from_connections(mapping.bead_names.len(), connections);
    map_native_trajectory_with_terms(input_path, output_path, mapping, &terms, options)
}

pub fn map_native_trajectory_with_terms(
    input_path: &Path,
    output_path: Option<&Path>,
    mapping: &BeadMapping,
    terms: &BondedTermSet,
    options: &NativeTrajectoryOptions,
) -> Result<TrajectoryMapReport> {
    let mut reader = open_reader(input_path, options.format.as_deref(), options.length_scale)?;
    let source_atom_count = reader.n_atoms();
    let source_indices = resolve_source_indices(source_atom_count, options)?;
    let translated_mapping = translate_mapping(mapping, &source_indices)?;
    let physical_bead_count = mapping.bead_names.len();
    let coordinate_count = coordinate_count_with_virtual_sites(mapping.bead_names.len(), terms);
    let sasa_radii = resolve_sasa_radii(
        options,
        source_atom_count,
        &translated_mapping,
        &mapping.bead_names,
        coordinate_count,
    )?;
    let sasa_sphere_points = sasa_sphere_points(options.sasa.n_sphere_points);
    let bead_weights = if options.mass_weighted {
        Some(resolve_bead_weights(
            options,
            source_atom_count,
            &translated_mapping,
        )?)
    } else {
        None
    };
    let bead_masses = bead_weights.as_deref().map(bead_masses_from_weights);
    let source_whole_connections = if options.make_whole {
        resolve_source_whole_connections(options, source_atom_count)?
    } else {
        Vec::new()
    };
    let chunk_frames = options.chunk_frames.unwrap_or(128).max(1);
    let mut builder = FrameChunkBuilder::new(source_atom_count, chunk_frames);
    builder.set_requirements(true, true);

    let mut writer = match output_path {
        Some(path) => Some(open_writer(path, coordinate_count)?),
        None => None,
    };
    let mut constraint_values = empty_bond_group_values(&terms.constraints);
    let mut bond_group_values = empty_bond_group_values(&terms.bonds);
    let mut angle_group_values = empty_angle_group_values(&terms.angles);
    let mut dihedral_group_values = empty_dihedral_group_values(&terms.dihedrals);
    let mut rg_values = RunningMetricStats::default();
    let mut sasa_values = RunningMetricStats::default();
    let whole_connections = if options.make_whole {
        bonded_whole_connections(terms)
    } else {
        Vec::new()
    };

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
            let box_ = chunk.box_.get(frame_idx).copied().unwrap_or(Box3::None);
            let repaired_source_frame;
            let mapping_source_frame = if options.make_whole && !source_whole_connections.is_empty()
            {
                repaired_source_frame = make_source_whole_by_bonded_connectivity(
                    source_frame,
                    &source_whole_connections,
                    box_,
                )?;
                repaired_source_frame.as_slice()
            } else {
                source_frame
            };
            let mut cg_coords = map_frame(
                mapping_source_frame,
                &translated_mapping,
                bead_weights.as_deref(),
            );
            apply_virtual_sites(&mut cg_coords, terms, bead_masses.as_deref())?;
            if options.make_whole {
                cg_coords =
                    make_whole_by_bonded_connectivity(&cg_coords, &whole_connections, box_)?;
            }
            if first_cg_coords.is_none() {
                first_cg_coords = Some(cg_coords.clone());
            }
            if let Some(rg) = radius_of_gyration(&cg_coords, bead_masses.as_deref()) {
                rg_values.push(rg);
            }
            if let Some(sphere_points) = sasa_sphere_points.as_deref() {
                let sasa_coord_count = physical_bead_count
                    .min(cg_coords.len())
                    .min(sasa_radii.len());
                if let Some(sasa) = shrake_rupley_sasa_with_points(
                    &cg_coords[..sasa_coord_count],
                    &sasa_radii[..sasa_coord_count],
                    options.sasa.probe_radius_nm,
                    sphere_points,
                ) {
                    sasa_values.push(sasa);
                }
            }
            accumulate_bond_group_values(&mut constraint_values, &terms.constraints, &cg_coords);
            accumulate_bond_group_values(&mut bond_group_values, &terms.bonds, &cg_coords);
            accumulate_angle_group_values(&mut angle_group_values, &terms.angles, &cg_coords);
            accumulate_dihedral_group_values(
                &mut dihedral_group_values,
                &terms.dihedrals,
                &cg_coords,
            );
            if let Some(writer) = writer.as_mut() {
                if writer.is_single_frame() && frames_written > 0 {
                    return Err(anyhow!(
                        "mapped trajectory output format cpt supports exactly one selected frame; use start/stop/stride to select one frame"
                    ));
                }
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

    let bonded_values = BondedValueSeries {
        constraints: bond_group_series(&terms.constraints, constraint_values),
        bonds: bond_group_series(&terms.bonds, bond_group_values.clone()),
        angles: angle_group_series(&terms.angles, angle_group_values.clone()),
        dihedrals: dihedral_group_series(&terms.dihedrals, dihedral_group_values.clone()),
    };
    let bond_values = single_bond_values(&terms.bonds, bond_group_values);
    let angle_values = single_angle_values(&terms.angles, angle_group_values);
    let dihedral_values = single_dihedral_values(&terms.dihedrals, dihedral_group_values);
    let bonded_stats = bonded_stats_from_values(bond_values, angle_values, dihedral_values);
    Ok(TrajectoryMapReport {
        frames_read,
        frames_written,
        first_cg_coords,
        bond_stats: bonded_stats.bonds,
        angle_stats: bonded_stats.angles,
        dihedral_stats: bonded_stats.dihedrals,
        bonded_values,
        rg_stats: rg_values.finish(),
        sasa_stats: sasa_values.finish(),
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
    let occurrence_counts = atom_mapping_occurrence_counts(translated_mapping, source_atom_count);
    Ok(translated_mapping
        .iter()
        .map(|group| {
            group
                .iter()
                .map(|&atom_idx| {
                    let mass = system.atoms.mass.get(atom_idx).copied().unwrap_or(0.0);
                    let mass = if mass.is_finite() && mass > 0.0 {
                        mass
                    } else {
                        1.0
                    };
                    let occurrence_count =
                        occurrence_counts.get(atom_idx).copied().unwrap_or(1).max(1);
                    mass / occurrence_count as f32
                })
                .collect()
        })
        .collect())
}

fn atom_mapping_occurrence_counts(
    translated_mapping: &[Vec<usize>],
    source_atom_count: usize,
) -> Vec<usize> {
    let mut occurrence_counts = vec![0; source_atom_count];
    for group in translated_mapping {
        for &atom_idx in group {
            if let Some(count) = occurrence_counts.get_mut(atom_idx) {
                *count += 1;
            }
        }
    }
    occurrence_counts
}

fn bead_masses_from_weights(weights: &[Vec<f32>]) -> Vec<f64> {
    weights
        .iter()
        .map(|bead_weights| bead_weights.iter().map(|value| f64::from(*value)).sum())
        .collect()
}

fn resolve_sasa_radii(
    options: &NativeTrajectoryOptions,
    source_atom_count: usize,
    translated_mapping: &[Vec<usize>],
    bead_names: &[String],
    coordinate_count: usize,
) -> Result<Vec<f64>> {
    if let Some(radii) = options.sasa.radii_nm.as_ref() {
        return validate_configured_sasa_radii(
            radii,
            coordinate_count,
            bead_names.len(),
            options.sasa.fallback_radius_nm,
        );
    }

    let fallback_radius = validated_fallback_sasa_radius(options.sasa.fallback_radius_nm);
    let mut radii: Vec<f64> = if let Some(radii) = topology_sasa_radii(
        options,
        source_atom_count,
        translated_mapping,
        fallback_radius,
    ) {
        radii
    } else {
        bead_names
            .iter()
            .map(|name| radius_from_exact_element_name_nm(name).unwrap_or(fallback_radius))
            .collect()
    };

    radii.resize(coordinate_count, fallback_radius);
    Ok(radii)
}

fn topology_sasa_radii(
    options: &NativeTrajectoryOptions,
    source_atom_count: usize,
    translated_mapping: &[Vec<usize>],
    fallback_radius_nm: f64,
) -> Option<Vec<f64>> {
    let topology = options.topology.as_deref()?;
    let system = read_system_auto(Path::new(topology), options.topology_format.as_deref()).ok()?;
    if system.n_atoms() != source_atom_count {
        return None;
    }
    Some(
        translated_mapping
            .iter()
            .map(|group| bead_radius_from_atom_group(&system, group, fallback_radius_nm))
            .collect(),
    )
}

fn validate_configured_sasa_radii(
    radii: &[f64],
    coordinate_count: usize,
    physical_bead_count: usize,
    fallback_radius_nm: f64,
) -> Result<Vec<f64>> {
    if radii.len() != coordinate_count && radii.len() != physical_bead_count {
        return Err(anyhow!(
            "trajectory_source.sasa.radii_nm length must match mapped coordinate count {} or physical bead count {}",
            coordinate_count,
            physical_bead_count
        ));
    }
    let mut out = radii.to_vec();
    for (idx, radius) in out.iter().enumerate() {
        if !radius.is_finite() || *radius <= 0.0 {
            return Err(anyhow!(
                "trajectory_source.sasa.radii_nm[{idx}] must be finite and greater than zero"
            ));
        }
    }
    out.resize(
        coordinate_count,
        validated_fallback_sasa_radius(fallback_radius_nm),
    );
    Ok(out)
}

fn validated_fallback_sasa_radius(radius_nm: f64) -> f64 {
    if radius_nm.is_finite() && radius_nm > 0.0 {
        radius_nm
    } else {
        DEFAULT_SASA_FALLBACK_RADIUS_NM
    }
}

fn bead_radius_from_atom_group(
    system: &traj_core::system::System,
    atom_indices: &[usize],
    fallback_radius_nm: f64,
) -> f64 {
    let sum_r3 = atom_indices
        .iter()
        .map(|&atom_idx| system_atom_vdw_radius_nm(system, atom_idx).unwrap_or(fallback_radius_nm))
        .filter(|radius| radius.is_finite() && *radius > 0.0)
        .map(|radius| radius.powi(3))
        .sum::<f64>();
    if sum_r3 > 0.0 {
        sum_r3.cbrt()
    } else {
        fallback_radius_nm
    }
}

fn system_atom_vdw_radius_nm(system: &traj_core::system::System, atom_idx: usize) -> Option<f64> {
    let atoms = &system.atoms;
    if atom_idx >= atoms.len() {
        return None;
    }
    atoms
        .element_id
        .get(atom_idx)
        .and_then(|id| system.interner.resolve(*id))
        .and_then(radius_from_symbol_nm)
        .or_else(|| {
            atoms
                .name_id
                .get(atom_idx)
                .and_then(|id| system.interner.resolve(*id))
                .and_then(radius_from_atom_name_nm)
        })
}

fn radius_from_exact_element_name_nm(name: &str) -> Option<f64> {
    let trimmed = name.trim();
    let symbol: String = trimmed
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .collect();
    if symbol.eq_ignore_ascii_case(trimmed) {
        radius_from_symbol_nm(&symbol)
    } else {
        None
    }
}

fn radius_from_atom_name_nm(name: &str) -> Option<f64> {
    let upper: String = name
        .trim()
        .chars()
        .filter(|ch| ch.is_ascii_alphabetic())
        .map(|ch| ch.to_ascii_uppercase())
        .collect();
    if upper.is_empty() {
        return None;
    }
    let two = upper.chars().take(2).collect::<String>();
    radius_from_symbol_nm(&two).or_else(|| {
        upper
            .chars()
            .next()
            .map(|ch| ch.to_string())
            .and_then(|symbol| radius_from_symbol_nm(&symbol))
    })
}

fn radius_from_symbol_nm(symbol: &str) -> Option<f64> {
    match symbol.trim().to_ascii_uppercase().as_str() {
        "H" => Some(0.120),
        "C" => Some(0.170),
        "N" => Some(0.155),
        "O" => Some(0.152),
        "F" => Some(0.147),
        "P" => Some(0.180),
        "S" => Some(0.180),
        "CL" => Some(0.175),
        "BR" => Some(0.185),
        "I" => Some(0.198),
        "NA" => Some(0.227),
        "K" => Some(0.275),
        "MG" => Some(0.173),
        "CA" => Some(0.231),
        "ZN" => Some(0.139),
        "FE" => Some(0.194),
        _ => None,
    }
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

pub(crate) fn open_reader(
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
        "xyz" => Box::new(XyzTrajReader::open(path)?),
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
#[path = "trajectory_tests.rs"]
mod tests;
