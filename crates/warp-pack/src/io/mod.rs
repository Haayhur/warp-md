use std::path::Path;

use crate::config::OutputSpec;
use crate::error::{PackError, PackResult};
use crate::pack::{AtomRecord, PackOutput};

mod amber;
mod crd;
mod gro;
mod lammps;
mod mol2;
mod pdb;
mod pdbx;
mod tinker;
mod xyz;

pub use amber::read_amber_inpcrd;
pub use crd::read_crd;
pub use crd::write_crd;
pub use gro::read_gro;
pub use gro::write_gro;
pub use lammps::read_lammps_data;
pub use lammps::write_lammps;
pub use mol2::write_mol2;
pub use pdb::write_pdb;
pub use pdbx::read_pdbx;
pub use pdbx::write_pdbx;
pub use tinker::read_tinker_xyz;
pub use xyz::write_xyz;

#[derive(Clone, Debug)]
pub struct MoleculeData {
    pub atoms: Vec<AtomRecord>,
    pub bonds: Vec<(usize, usize)>,
    pub ter_after: Vec<usize>,
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
        "pdb" => pdb::read_pdb(path, ignore_conect, non_standard_conect),
        "xyz" => xyz::read_xyz(path),
        "mol2" => mol2::read_mol2(path),
        "tinker" | "txyz" => tinker::read_tinker_xyz(path),
        "amber" | "inpcrd" | "rst" | "rst7" => amber::read_amber_inpcrd(path, topology),
        "pdbx" | "cif" | "mmcif" => pdbx::read_pdbx(path),
        "gro" => gro::read_gro(path),
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
) -> PackResult<()> {
    let format = spec.format.to_lowercase();
    let scale = spec.scale.unwrap_or(1.0);
    let box_fix = if add_box_sides { box_sides_fix } else { 0.0 };
    match format.as_str() {
        "pdb" => write_pdb(
            out,
            &spec.path,
            scale,
            add_box_sides,
            box_fix,
            write_conect,
            hexadecimal_indices,
        ),
        "xyz" => write_xyz(out, &spec.path, scale),
        "pdbx" | "cif" | "mmcif" => write_pdbx(out, &spec.path, scale, box_fix),
        "gro" => write_gro(out, &spec.path, scale, box_fix),
        "lammps" | "lammps-data" | "lmp" => write_lammps(out, &spec.path, scale, box_fix),
        "mol2" => write_mol2(out, &spec.path, scale),
        "crd" => write_crd(out, &spec.path, scale, box_fix),
        _ => Err(PackError::Invalid(format!(
            "unsupported output format: {format}"
        ))),
    }
}
