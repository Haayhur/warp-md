use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};
use traj_core::elements::infer_element_from_atom_name;

pub fn read_crd(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut atoms = Vec::new();
    let mut in_atoms = false;
    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.starts_with('*') {
            continue;
        }
        if !in_atoms {
            if trimmed
                .split_whitespace()
                .next()
                .and_then(|s| s.parse::<usize>().ok())
                .is_some()
            {
                in_atoms = true;
            }
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 7 {
            continue;
        }
        let resid = parts[1].parse::<i32>().unwrap_or(1);
        let resname = parts[2].to_string();
        let name = parts[3].to_string();
        let x: f32 = parts[4]
            .parse()
            .map_err(|_| PackError::Parse("bad crd x".into()))?;
        let y: f32 = parts[5]
            .parse()
            .map_err(|_| PackError::Parse("bad crd y".into()))?;
        let z: f32 = parts[6]
            .parse()
            .map_err(|_| PackError::Parse("bad crd z".into()))?;
        let segid = parts.get(7).map(|s| s.to_string()).unwrap_or_default();
        let charge = parts
            .last()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.0);
        let element = infer_element_from_atom_name(&name).unwrap_or_else(|| "X".into());
        atoms.push(AtomRecord {
            record_kind: AtomRecordKind::Atom,
            name,
            element,
            resname,
            resid,
            chain: 'A',
            segid,
            charge,
            position: Vec3::new(x, y, z),
            mol_id: 1,
        });
    }
    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in crd".into()));
    }
    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        ter_after: Vec::new(),
    })
}

pub fn write_crd(out: &PackOutput, path: &str, scale: f32, _box_sides_fix: f32) -> PackResult<()> {
    let mut file = File::create(path)?;
    writeln!(file, "* TITLE warp-pack")?;
    writeln!(file, "* Packmol-style CHARMM CRD export")?;
    writeln!(file, "*")?;
    writeln!(file, "{:>10} EXT", out.atoms.len())?;

    for (i, atom) in out.atoms.iter().enumerate() {
        let idx = i + 1;
        let resid = atom.resid;
        let resname = trim_len(&atom.resname, 8);
        let name = trim_len(&atom.name, 8);
        let pos = atom.position.scale(scale);
        let segid = if atom.segid.is_empty() {
            trim_len(&atom.resname, 8)
        } else {
            trim_len(&atom.segid, 8)
        };
        let resid_str = trim_len(&format!("{resid}"), 8);
        writeln!(
            file,
            "{idx:>10}{resid:>10}  {resname:<8}  {name:<8}{x:>20.10}{y:>20.10}{z:>20.10}  {segid:<8}  {resid_str:<8}{charge:>20.10}",
            idx = idx,
            resid = resid,
            resname = resname,
            name = name,
            x = pos.x,
            y = pos.y,
            z = pos.z,
            segid = segid,
            resid_str = resid_str,
            charge = atom.charge
        )?;
    }
    Ok(())
}

fn trim_len(value: &str, max_len: usize) -> String {
    let mut out = value.trim().to_string();
    if out.len() > max_len {
        out.truncate(max_len);
    }
    out
}
