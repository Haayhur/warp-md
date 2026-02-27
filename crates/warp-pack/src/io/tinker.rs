use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind};

pub fn read_tinker_xyz(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| PackError::Parse("tinker xyz missing header".into()))??;
    let mut header_parts = header.split_whitespace();
    let n_atoms: usize = header_parts
        .next()
        .ok_or_else(|| PackError::Parse("tinker xyz missing atom count".into()))?
        .parse()
        .map_err(|_| PackError::Parse("invalid tinker xyz count".into()))?;

    let mut atoms = Vec::with_capacity(n_atoms);
    for line in lines {
        if atoms.len() >= n_atoms {
            break;
        }
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 5 {
            continue;
        }
        let name = parts[1].to_string();
        let element = parts[1].to_string();
        let x: f32 = parts[2]
            .parse()
            .map_err(|_| PackError::Parse("bad tinker x".into()))?;
        let y: f32 = parts[3]
            .parse()
            .map_err(|_| PackError::Parse("bad tinker y".into()))?;
        let z: f32 = parts[4]
            .parse()
            .map_err(|_| PackError::Parse("bad tinker z".into()))?;
        atoms.push(AtomRecord {
            record_kind: AtomRecordKind::Atom,
            name,
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
    if atoms.len() != n_atoms {
        return Err(PackError::Parse(format!(
            "tinker xyz expected {n_atoms} atoms, found {}",
            atoms.len()
        )));
    }
    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        ter_after: Vec::new(),
    })
}
