use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};
use traj_core::elements::infer_element_from_atom_name;
use traj_core::pdb_gro::parse_gro_reader;

pub fn read_gro(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let parsed = parse_gro_reader(reader, false).map_err(|e| PackError::Parse(e.to_string()))?;
    let atoms = parsed
        .atoms
        .into_iter()
        .map(|atom| {
            let element = if atom.element.is_empty() {
                infer_element_from_atom_name(&atom.name).unwrap_or_else(|| "X".into())
            } else {
                atom.element
            };
            AtomRecord {
                record_kind: AtomRecordKind::Atom,
                name: atom.name,
                element,
                resname: if atom.resname.is_empty() {
                    "MOL".into()
                } else {
                    atom.resname
                },
                resid: atom.resid,
                chain: 'A',
                segid: String::new(),
                charge: 0.0,
                position: Vec3::new(atom.position[0], atom.position[1], atom.position[2]),
                mol_id: 1,
            }
        })
        .collect::<Vec<_>>();
    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in gro".into()));
    }
    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        ter_after: Vec::new(),
    })
}

pub fn write_gro(out: &PackOutput, path: &str, scale: f32, box_sides_fix: f32) -> PackResult<()> {
    let mut file = File::create(path)?;
    writeln!(file, "warp_pack")?;
    writeln!(file, "{:>5}", out.atoms.len())?;
    for (i, atom) in out.atoms.iter().enumerate() {
        let idx = i + 1;
        let p = atom.position.scale(scale);
        let resid = (atom.resid.rem_euclid(100000)) as usize;
        let line = format!(
            "{resid:>5}{resname:<5}{name:>5}{idx:>5}{x:>8.3}{y:>8.3}{z:>8.3}\n",
            resid = resid,
            resname = atom.resname,
            name = atom.name,
            idx = idx,
            x = p.x,
            y = p.y,
            z = p.z
        );
        file.write_all(line.as_bytes())?;
    }
    let b = out.box_size;
    writeln!(
        file,
        "{:.5} {:.5} {:.5}",
        (b[0] + box_sides_fix) * scale,
        (b[1] + box_sides_fix) * scale,
        (b[2] + box_sides_fix) * scale
    )?;
    Ok(())
}
