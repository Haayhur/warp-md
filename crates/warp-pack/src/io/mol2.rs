use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};
use traj_core::elements::{infer_element_from_atom_name, normalize_element};

pub fn read_mol2(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut atoms = Vec::new();
    let mut bonds = Vec::new();
    let mut in_atoms = false;
    let mut in_bonds = false;
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.starts_with("@<TRIPOS>ATOM") {
            in_atoms = true;
            in_bonds = false;
            continue;
        }
        if trimmed.starts_with("@<TRIPOS>BOND") {
            in_atoms = false;
            in_bonds = true;
            continue;
        }
        if trimmed.starts_with("@<TRIPOS>") {
            in_atoms = false;
            in_bonds = false;
            continue;
        }
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if in_atoms {
            if parts.len() < 6 {
                continue;
            }
            let name = parts[1].to_string();
            let x: f32 = parts[2]
                .parse()
                .map_err(|_| PackError::Parse("bad mol2 x".into()))?;
            let y: f32 = parts[3]
                .parse()
                .map_err(|_| PackError::Parse("bad mol2 y".into()))?;
            let z: f32 = parts[4]
                .parse()
                .map_err(|_| PackError::Parse("bad mol2 z".into()))?;
            let resname = parts
                .get(7)
                .map(|s| s.to_string())
                .unwrap_or_else(|| "MOL".into());
            let resid = parts
                .get(6)
                .and_then(|s| s.parse::<i32>().ok())
                .unwrap_or(1);
            let element = normalize_element(parts[5])
                .or_else(|| infer_element_from_atom_name(&name))
                .unwrap_or_else(|| "X".into());
            let charge = parts
                .get(8)
                .and_then(|s| s.parse::<f32>().ok())
                .unwrap_or(0.0);
            atoms.push(AtomRecord {
                record_kind: AtomRecordKind::Atom,
                name,
                element,
                resname,
                resid,
                chain: 'A',
                segid: String::new(),
                charge,
                position: Vec3::new(x, y, z),
                mol_id: 1,
            });
            continue;
        }
        if in_bonds {
            if parts.len() < 4 {
                continue;
            }
            let a = parts[1]
                .parse::<usize>()
                .map_err(|_| PackError::Parse("bad mol2 bond atom id".into()))?;
            let b = parts[2]
                .parse::<usize>()
                .map_err(|_| PackError::Parse("bad mol2 bond atom id".into()))?;
            if a == 0 || b == 0 {
                return Err(PackError::Parse("mol2 bond atom ids must be >= 1".into()));
            }
            bonds.push((a - 1, b - 1));
        }
    }
    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in mol2".into()));
    }
    if bonds
        .iter()
        .any(|&(a, b)| a >= atoms.len() || b >= atoms.len())
    {
        return Err(PackError::Parse(
            "mol2 bond references atom outside atom table".into(),
        ));
    }
    Ok(MoleculeData {
        atoms,
        bonds,
        ter_after: Vec::new(),
    })
}

pub fn write_mol2(out: &PackOutput, path: &str, scale: f32) -> PackResult<()> {
    let mut file = File::create(path)?;
    writeln!(file, "@<TRIPOS>MOLECULE")?;
    writeln!(file, "warp_pack")?;
    writeln!(file, "{} 0 0 0 0", out.atoms.len())?;
    writeln!(file, "SMALL")?;
    writeln!(file, "USER_CHARGES")?;
    writeln!(file, "@<TRIPOS>ATOM")?;
    for (i, atom) in out.atoms.iter().enumerate() {
        let idx = i + 1;
        let p = atom.position.scale(scale);
        writeln!(
            file,
            "{:>7} {:<4} {:>10.4} {:>10.4} {:>10.4} {:<6} {:>4} {:<6} {:>7.4}",
            idx, atom.name, p.x, p.y, p.z, atom.element, atom.resid, atom.resname, atom.charge
        )?;
    }
    Ok(())
}
