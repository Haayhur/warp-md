use std::path::{Path, PathBuf};

use crate::config::OutputSpec;
use crate::error::{StructureError as PackError, StructureResult as PackResult};
use crate::model::BoxVectors;
use crate::pack::{AtomRecord, PackOutput};

mod amber;
mod crd;
mod gro;
mod gromos96;
mod lammps;
mod mol2;
mod pdb;
mod pdbx;
mod pqr;
mod tinker;
mod xyz;

pub use amber::read_amber_inpcrd;
pub use amber::read_prmtop_atom_charges;
pub use amber::read_prmtop_topology;
pub use amber::read_prmtop_total_charge;
pub use amber::write_amber_inpcrd;
pub use amber::write_minimal_prmtop;
pub use amber::AmberTopology;
pub use crd::read_crd;
pub use crd::write_crd;
pub use gro::read_gro;
pub use gro::write_gro;
pub use gromos96::read_gromos96;
pub use gromos96::write_gromos96;
pub use lammps::read_lammps_data;
pub use lammps::write_lammps;
pub use mol2::write_mol2;
pub use pdb::write_pdb;
pub use pdbx::read_pdbx;
pub use pdbx::write_pdbx;
pub use pqr::read_pqr;
pub use pqr::write_pqr;
pub use tinker::read_tinker_xyz;
pub use xyz::write_xyz;

#[derive(Clone, Debug)]
pub struct MoleculeData {
    pub atoms: Vec<AtomRecord>,
    pub bonds: Vec<(usize, usize)>,
    pub box_vectors: Option<BoxVectors>,
    pub ter_after: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct OutputWriteResult {
    pub path: String,
    pub format: String,
    pub fallback_applied: bool,
}

pub fn infer_output_format_from_path(path: &str) -> String {
    OutputSpec::infer_format_from_path(path)
}

fn resolved_output_format(spec: &OutputSpec) -> String {
    spec.resolved_format()
}

fn default_output_scale(format: &str) -> f32 {
    OutputSpec::default_scale_for_format(format)
}

fn should_fallback_to_mmcif(err: &PackError) -> bool {
    match err {
        PackError::Invalid(message) => message.contains("use mmcif"),
        _ => false,
    }
}

fn fallback_mmcif_path(path: &str) -> String {
    let mut fallback = PathBuf::from(path);
    fallback.set_extension("cif");
    fallback.to_string_lossy().to_string()
}

pub fn read_molecule(
    path: &Path,
    format: Option<&str>,
    ignore_conect: bool,
    non_standard_conect: bool,
    topology: Option<&Path>,
) -> PackResult<MoleculeData> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    let fmt = format.unwrap_or(&ext).to_lowercase();
    match fmt.as_str() {
        "pdb" | "brk" | "ent" => pdb::read_pdb(path, ignore_conect, non_standard_conect),
        "pqr" => pqr::read_pqr(path),
        "xyz" => xyz::read_xyz(path),
        "mol2" => mol2::read_mol2(path),
        "tinker" | "txyz" => tinker::read_tinker_xyz(path),
        "amber" | "inpcrd" | "rst" | "rst7" => amber::read_amber_inpcrd(path, topology),
        "pdbx" | "cif" | "mmcif" => pdbx::read_pdbx(path),
        "gro" => gro::read_gro(path),
        "g96" | "gromos96" => gromos96::read_gromos96(path),
        "lammps" | "lammps-data" | "lmp" => lammps::read_lammps_data(path),
        "crd" => crd::read_crd(path),
        _ => Err(PackError::Parse(format!("unsupported input format: {fmt}"))),
    }
}

pub fn write_output(
    out: &PackOutput,
    spec: &OutputSpec,
    add_box_sides: bool,
    box_sides_fix: f32,
    write_conect: bool,
    hexadecimal_indices: bool,
) -> PackResult<OutputWriteResult> {
    spec.validate()?;
    let format = resolved_output_format(spec);
    let scale = spec.scale.unwrap_or_else(|| default_output_scale(&format));
    let box_fix = if add_box_sides { box_sides_fix } else { 0.0 };
    match format.as_str() {
        "pdb" | "pdb-strict" | "brk" | "ent" => match write_pdb(
            out,
            &spec.path,
            scale,
            add_box_sides,
            box_fix,
            write_conect,
            hexadecimal_indices,
            format == "pdb-strict",
        ) {
            Ok(()) => Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            }),
            Err(err) if should_fallback_to_mmcif(&err) => {
                let fallback_path = fallback_mmcif_path(&spec.path);
                write_pdbx(out, &fallback_path, scale, box_fix)?;
                Ok(OutputWriteResult {
                    path: fallback_path,
                    format: "mmcif".into(),
                    fallback_applied: true,
                })
            }
            Err(err) => Err(err),
        },
        "pqr" => {
            write_pqr(out, &spec.path, scale, add_box_sides, box_fix)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        "xyz" => {
            write_xyz(out, &spec.path, scale)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        "pdbx" | "cif" | "mmcif" => {
            write_pdbx(out, &spec.path, scale, box_fix)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        "gro" => {
            write_gro(out, &spec.path, scale, box_fix)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        "g96" | "gromos96" => {
            write_gromos96(out, &spec.path, scale, box_fix)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        "lammps" | "lammps-data" | "lmp" => {
            write_lammps(out, &spec.path, scale, box_fix)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        "mol2" => {
            write_mol2(out, &spec.path, scale)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        "crd" => {
            write_crd(out, &spec.path, scale, box_fix)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        "inpcrd" | "rst" | "rst7" => {
            write_amber_inpcrd(out, &spec.path, scale)?;
            Ok(OutputWriteResult {
                path: spec.path.clone(),
                format,
                fallback_applied: false,
            })
        }
        _ => Err(PackError::Invalid(format!(
            "unsupported output format: {format}"
        ))),
    }
}
