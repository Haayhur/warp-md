use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecordKind, PackOutput};

pub fn read_xyz(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let count_line = lines
        .next()
        .ok_or_else(|| PackError::Parse("xyz missing atom count".into()))??;
    let n_atoms: usize = count_line
        .trim()
        .parse()
        .map_err(|_| PackError::Parse("invalid xyz count".into()))?;
    let _ = lines.next();
    let mut atoms = Vec::with_capacity(n_atoms);
    for line in lines {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }
        let element = parts[0].to_string();
        let x: f32 = parts[1]
            .parse()
            .map_err(|_| PackError::Parse("bad xyz x".into()))?;
        let y: f32 = parts[2]
            .parse()
            .map_err(|_| PackError::Parse("bad xyz y".into()))?;
        let z: f32 = parts[3]
            .parse()
            .map_err(|_| PackError::Parse("bad xyz z".into()))?;
        atoms.push(crate::pack::AtomRecord {
            record_kind: AtomRecordKind::Atom,
            name: element.clone(),
            element,
            resname: "MOL".into(),
            resid: 1,
            chain: 'A',
            segid: String::new(),
            charge: 0.0,
            position: Vec3::new(x, y, z),
            mol_id: 1,
        });
    }
    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in xyz".into()));
    }
    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        ter_after: Vec::new(),
    })
}

pub fn write_xyz(out: &PackOutput, path: &str, scale: f32) -> PackResult<()> {
    let file = File::create(path)?;
    let mut file = BufWriter::new(file);
    writeln!(file, "{}", out.atoms.len())?;
    writeln!(file, "warp_pack")?;
    for atom in &out.atoms {
        let p = atom.position.scale(scale);
        writeln!(file, "{} {:.6} {:.6} {:.6}", atom.element, p.x, p.y, p.z)?;
    }
    Ok(())
}
