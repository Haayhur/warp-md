use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};
use traj_core::elements::infer_element_from_mass;

pub fn read_lammps_data(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut atoms = Vec::new();
    let mut bonds_raw: Vec<(i32, i32)> = Vec::new();
    let mut atom_index: HashMap<i32, usize> = HashMap::new();
    let mut type_elements: HashMap<i32, String> = HashMap::new();
    let mut section = Section::None;
    let mut atom_style: Option<String> = None;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(next) = section_header(trimmed) {
            section = next;
            if section == Section::Atoms {
                atom_style = parse_atom_style(trimmed);
            }
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        match section {
            Section::Masses => {
                if let Some((atom_type, element)) = parse_mass_line(trimmed) {
                    type_elements.insert(atom_type, element);
                }
            }
            Section::Atoms => {
                if let Some((atom_id, atom)) =
                    parse_atom_line(trimmed, atom_style.as_deref(), &type_elements)?
                {
                    let idx = atoms.len();
                    atoms.push(atom);
                    atom_index.insert(atom_id, idx);
                }
            }
            Section::Bonds => {
                if let Some((a, b)) = parse_bond_line(trimmed) {
                    bonds_raw.push((a, b));
                }
            }
            Section::Other | Section::None => {}
        }
    }

    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in lammps data".into()));
    }
    let mut bonds = Vec::new();
    for (a_id, b_id) in bonds_raw {
        if let (Some(&a_idx), Some(&b_idx)) = (atom_index.get(&a_id), atom_index.get(&b_id)) {
            bonds.push((a_idx, b_idx));
        }
    }
    Ok(MoleculeData {
        atoms,
        bonds,
        ter_after: Vec::new(),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    None,
    Masses,
    Atoms,
    Bonds,
    Other,
}

fn section_header(line: &str) -> Option<Section> {
    let token = line.split_whitespace().next()?;
    match token {
        "Masses" => Some(Section::Masses),
        "Atoms" => Some(Section::Atoms),
        "Bonds" => Some(Section::Bonds),
        "Angles" | "Dihedrals" | "Impropers" | "Velocities" => Some(Section::Other),
        _ => None,
    }
}

fn parse_atom_style(line: &str) -> Option<String> {
    line.split('#').nth(1).map(|s| s.trim().to_lowercase())
}

fn parse_mass_line(line: &str) -> Option<(i32, String)> {
    let mut parts = line.split('#');
    let left = parts.next()?.trim();
    let comment = parts.next().map(|s| s.trim());
    let fields: Vec<&str> = left.split_whitespace().collect();
    if fields.len() < 2 {
        return None;
    }
    let atom_type = fields[0].parse::<i32>().ok()?;
    let mass = fields[1].parse::<f32>().ok()?;
    let element = comment
        .and_then(|c| c.split_whitespace().next())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            infer_element_from_mass(mass).unwrap_or_else(|| format!("T{atom_type}"))
        });
    Some((atom_type, element))
}

fn parse_bond_line(line: &str) -> Option<(i32, i32)> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 4 {
        return None;
    }
    let a = parts[2].parse::<i32>().ok()?;
    let b = parts[3].parse::<i32>().ok()?;
    Some((a, b))
}

fn parse_atom_line(
    line: &str,
    style: Option<&str>,
    type_elements: &HashMap<i32, String>,
) -> PackResult<Option<(i32, AtomRecord)>> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 5 {
        return Ok(None);
    }
    let style = style.unwrap_or("").trim();
    let parsed = match style {
        "full" => parse_full(&parts),
        "atomic" => parse_atomic(&parts),
        "charge" => parse_charge(&parts),
        "molecular" => parse_molecular(&parts),
        _ => parse_unknown(&parts),
    }?;
    let (atom_id, mol_id, atom_type, charge, x, y, z) = parsed;
    let element = type_elements
        .get(&atom_type)
        .cloned()
        .unwrap_or_else(|| format!("T{atom_type}"));
    Ok(Some((
        atom_id,
        AtomRecord {
            record_kind: AtomRecordKind::Atom,
            name: element.clone(),
            element,
            resname: "MOL".into(),
            resid: mol_id,
            chain: 'A',
            segid: String::new(),
            charge,
            position: Vec3::new(x, y, z),
            mol_id,
        },
    )))
}

fn parse_full(parts: &[&str]) -> PackResult<(i32, i32, i32, f32, f32, f32, f32)> {
    if parts.len() < 7 {
        return Err(PackError::Parse("bad lammps full atom line".into()));
    }
    let id = parts[0]
        .parse::<i32>()
        .map_err(|_| PackError::Parse("bad atom id".into()))?;
    let mol = parts[1].parse::<i32>().unwrap_or(1);
    let atom_type = parts[2].parse::<i32>().unwrap_or(1);
    let charge = parts[3].parse::<f32>().unwrap_or(0.0);
    let x = parts[4]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps x".into()))?;
    let y = parts[5]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps y".into()))?;
    let z = parts[6]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps z".into()))?;
    Ok((id, mol, atom_type, charge, x, y, z))
}

fn parse_atomic(parts: &[&str]) -> PackResult<(i32, i32, i32, f32, f32, f32, f32)> {
    if parts.len() < 5 {
        return Err(PackError::Parse("bad lammps atomic atom line".into()));
    }
    let id = parts[0]
        .parse::<i32>()
        .map_err(|_| PackError::Parse("bad atom id".into()))?;
    let atom_type = parts[1].parse::<i32>().unwrap_or(1);
    let x = parts[2]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps x".into()))?;
    let y = parts[3]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps y".into()))?;
    let z = parts[4]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps z".into()))?;
    Ok((id, 1, atom_type, 0.0, x, y, z))
}

fn parse_charge(parts: &[&str]) -> PackResult<(i32, i32, i32, f32, f32, f32, f32)> {
    if parts.len() < 6 {
        return Err(PackError::Parse("bad lammps charge atom line".into()));
    }
    let id = parts[0]
        .parse::<i32>()
        .map_err(|_| PackError::Parse("bad atom id".into()))?;
    let atom_type = parts[1].parse::<i32>().unwrap_or(1);
    let charge = parts[2].parse::<f32>().unwrap_or(0.0);
    let x = parts[3]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps x".into()))?;
    let y = parts[4]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps y".into()))?;
    let z = parts[5]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps z".into()))?;
    Ok((id, 1, atom_type, charge, x, y, z))
}

fn parse_molecular(parts: &[&str]) -> PackResult<(i32, i32, i32, f32, f32, f32, f32)> {
    if parts.len() < 6 {
        return Err(PackError::Parse("bad lammps molecular atom line".into()));
    }
    let id = parts[0]
        .parse::<i32>()
        .map_err(|_| PackError::Parse("bad atom id".into()))?;
    let mol = parts[1].parse::<i32>().unwrap_or(1);
    let atom_type = parts[2].parse::<i32>().unwrap_or(1);
    let x = parts[3]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps x".into()))?;
    let y = parts[4]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps y".into()))?;
    let z = parts[5]
        .parse::<f32>()
        .map_err(|_| PackError::Parse("bad lammps z".into()))?;
    Ok((id, mol, atom_type, 0.0, x, y, z))
}

fn parse_unknown(parts: &[&str]) -> PackResult<(i32, i32, i32, f32, f32, f32, f32)> {
    if parts.len() >= 7 {
        return parse_full(parts);
    }
    if parts.len() == 6 {
        if parts[2].parse::<i32>().is_ok() {
            return parse_molecular(parts);
        }
        return parse_charge(parts);
    }
    parse_atomic(parts)
}

pub fn write_lammps(
    out: &PackOutput,
    path: &str,
    scale: f32,
    box_sides_fix: f32,
) -> PackResult<()> {
    let mut file = File::create(path)?;
    let mut type_map: HashMap<String, i32> = HashMap::new();
    let mut next_type = 1i32;
    for atom in &out.atoms {
        if !type_map.contains_key(&atom.element) {
            type_map.insert(atom.element.clone(), next_type);
            next_type += 1;
        }
    }
    let n_types = type_map.len();
    writeln!(file, "warp_pack lammps data")?;
    writeln!(file)?;
    writeln!(file, "{} atoms", out.atoms.len())?;
    writeln!(file, "{} atom types", n_types)?;
    writeln!(file)?;
    let b = out.box_size;
    writeln!(file, "0.0 {:.6} xlo xhi", (b[0] + box_sides_fix) * scale)?;
    writeln!(file, "0.0 {:.6} ylo yhi", (b[1] + box_sides_fix) * scale)?;
    writeln!(file, "0.0 {:.6} zlo zhi", (b[2] + box_sides_fix) * scale)?;
    writeln!(file)?;
    writeln!(file, "Masses")?;
    writeln!(file)?;
    let mut elements: Vec<_> = type_map.iter().collect();
    elements.sort_by_key(|(_, t)| **t);
    for (elem, t) in elements {
        writeln!(file, "{} 1.0 # {}", t, elem)?;
    }
    writeln!(file)?;
    writeln!(file, "Atoms # full")?;
    writeln!(file)?;
    for (i, atom) in out.atoms.iter().enumerate() {
        let id = i + 1;
        let mol = atom.mol_id.max(1);
        let atom_type = *type_map.get(&atom.element).unwrap_or(&1);
        let p = atom.position.scale(scale);
        writeln!(
            file,
            "{} {} {} {:.4} {:.6} {:.6} {:.6}",
            id, mol, atom_type, atom.charge, p.x, p.y, p.z
        )?;
    }
    Ok(())
}
