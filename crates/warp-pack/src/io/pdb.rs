use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};
use traj_core::pdb_gro::{parse_pdb_reader, PdbParseOptions, PdbRecordKind};

pub fn read_pdb(
    path: &Path,
    ignore_conect: bool,
    non_standard_conect: bool,
) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let options = PdbParseOptions {
        include_conect: !ignore_conect,
        non_standard_conect,
        include_ter: true,
        strict: false,
        only_first_model: false,
    };
    let parsed = parse_pdb_reader(reader, &options).map_err(|e| PackError::Parse(e.to_string()))?;
    let atoms = parsed
        .atoms
        .into_iter()
        .map(|atom| {
            let record_kind = match atom.record_kind {
                PdbRecordKind::HetAtom => AtomRecordKind::HetAtom,
                PdbRecordKind::Atom => AtomRecordKind::Atom,
            };
            let resname = if atom.resname.trim().is_empty() {
                "MOL".into()
            } else {
                atom.resname
            };
            let chain = if atom.chain == ' ' { 'A' } else { atom.chain };
            AtomRecord {
                record_kind,
                name: atom.name,
                element: atom.element,
                resname,
                resid: atom.resid,
                chain,
                segid: atom.segid,
                charge: 0.0,
                position: Vec3::new(atom.position[0], atom.position[1], atom.position[2]),
                mol_id: 1,
            }
        })
        .collect::<Vec<_>>();
    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in pdb".into()));
    }
    Ok(MoleculeData {
        atoms,
        bonds: parsed.bonds,
        ter_after: parsed.ter_after,
    })
}

pub fn write_pdb(
    out: &PackOutput,
    path: &str,
    scale: f32,
    add_box_sides: bool,
    box_sides_fix: f32,
    write_conect: bool,
    hexadecimal_indices: bool,
    strict_fields: bool,
) -> PackResult<()> {
    if strict_fields {
        validate_strict_pdb_output(out, hexadecimal_indices)?;
    }
    let file = File::create(path)?;
    let mut file = BufWriter::new(file);
    if add_box_sides {
        let b = out.box_size;
        writeln!(
            file,
            "CRYST1{:>9.2}{:>9.2}{:>9.2}{:>7.2}{:>7.2}{:>7.2} P 1           1",
            (b[0] + box_sides_fix) * scale,
            (b[1] + box_sides_fix) * scale,
            (b[2] + box_sides_fix) * scale,
            90.0,
            90.0,
            90.0
        )?;
    }
    let mut ter_iter = out.ter_after.iter().copied().peekable();
    for (i, atom) in out.atoms.iter().enumerate() {
        let idx = i + 1;
        let pos = atom.position.scale(scale);
        let record = match atom.record_kind {
            AtomRecordKind::Atom => "ATOM  ",
            AtomRecordKind::HetAtom => "HETATM",
        };
        let idx_str = format_atom_id(idx, hexadecimal_indices);
        let resid_str = format_res_id(atom.resid, hexadecimal_indices);
        let atom_name = format_atom_name(&atom.name);
        let resname = truncate_right(&atom.resname, 3);
        let chain = if atom.chain == ' ' { 'A' } else { atom.chain };
        let element = format_element(&atom.element);
        let line = format!(
            "{record}{idx:>5} {name}{alt}{resname:>3} {chain}{resid:>4}{icode}   {x:>8.3}{y:>8.3}{z:>8.3}{occ:>6.2}{temp:>6.2}          {element:>2}  \n",
            record = record,
            idx = idx_str,
            name = atom_name,
            alt = " ",
            resname = resname,
            chain = chain,
            resid = resid_str,
            icode = " ",
            x = pos.x,
            y = pos.y,
            z = pos.z,
            occ = 1.0,
            temp = 0.0,
            element = element,
        );
        file.write_all(line.as_bytes())?;
        if matches!(ter_iter.peek(), Some(next) if *next == i) {
            let ter_line = format!(
                "TER   {:>5}      {:>3} {}{:>4}\n",
                format_atom_id(idx + 1, hexadecimal_indices),
                resname,
                chain,
                resid_str
            );
            file.write_all(ter_line.as_bytes())?;
            ter_iter.next();
        }
    }
    if write_conect && !out.bonds.is_empty() {
        write_conect_lines(&mut file, &out.bonds, hexadecimal_indices)?;
    }
    file.write_all(b"END\n")?;
    Ok(())
}

fn validate_strict_pdb_output(out: &PackOutput, hexadecimal_indices: bool) -> PackResult<()> {
    if hexadecimal_indices {
        return Err(PackError::Invalid(
            "pdb-strict does not support hexadecimal atom or residue indices; use mmcif for larger systems".into(),
        ));
    }
    for atom in &out.atoms {
        let resname = atom.resname.trim();
        if resname.len() > 3 {
            return Err(PackError::Invalid(format!(
                "pdb-strict residue name '{}' exceeds 3 characters; preserve residue templates or use mmcif",
                resname
            )));
        }
        let name = atom.name.trim();
        if name.len() > 4 {
            return Err(PackError::Invalid(format!(
                "pdb-strict atom name '{}' exceeds 4 characters; use mmcif",
                name
            )));
        }
        let element = atom.element.trim();
        if element.len() > 2 {
            return Err(PackError::Invalid(format!(
                "pdb-strict element '{}' exceeds 2 characters; use mmcif",
                element
            )));
        }
        if atom.resid < -999 || atom.resid > 9999 {
            return Err(PackError::Invalid(format!(
                "pdb-strict residue id '{}' exceeds fixed-column width; use mmcif",
                atom.resid
            )));
        }
        let pos = atom.position;
        for (axis, value) in [("x", pos.x), ("y", pos.y), ("z", pos.z)] {
            if value < -999.999 || value > 9999.999 {
                return Err(PackError::Invalid(format!(
                    "pdb-strict {axis} coordinate '{}' exceeds fixed-column width; use mmcif",
                    value
                )));
            }
        }
    }
    Ok(())
}

fn write_conect_lines<W: Write>(
    file: &mut W,
    bonds: &[(usize, usize)],
    hexadecimal_indices: bool,
) -> PackResult<()> {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in bonds {
        adjacency.entry(a).or_default().push(b);
        adjacency.entry(b).or_default().push(a);
    }
    for (atom, mut neighbors) in adjacency {
        neighbors.sort_unstable();
        let atom_id = atom + 1;
        let mut chunk = Vec::new();
        for n in neighbors {
            chunk.push(n + 1);
            if chunk.len() == 4 {
                write_conect_line(file, atom_id, &chunk, hexadecimal_indices)?;
                chunk.clear();
            }
        }
        if !chunk.is_empty() {
            write_conect_line(file, atom_id, &chunk, hexadecimal_indices)?;
        }
    }
    Ok(())
}

fn write_conect_line<W: Write>(
    file: &mut W,
    atom_id: usize,
    neighbors: &[usize],
    hexadecimal_indices: bool,
) -> PackResult<()> {
    let mut line = format!("CONECT{:>5}", format_atom_id(atom_id, hexadecimal_indices));
    for n in neighbors {
        line.push_str(&format!("{:>5}", format_atom_id(*n, hexadecimal_indices)));
    }
    line.push('\n');
    file.write_all(line.as_bytes())?;
    Ok(())
}

fn format_atom_id(idx: usize, hex: bool) -> String {
    if hex || idx > 99999 {
        format!("{:X}", idx)
    } else {
        format!("{idx}")
    }
}

fn format_res_id(resid: i32, hex: bool) -> String {
    if resid >= 0 && (hex || resid > 9999) {
        format!("{:X}", resid)
    } else {
        format!("{resid}")
    }
}

fn truncate_right(value: &str, width: usize) -> String {
    value.chars().take(width).collect::<String>()
}

fn format_atom_name(name: &str) -> String {
    let trimmed = name.trim();
    let compact = trimmed.chars().take(4).collect::<String>();
    format!("{compact:>4}")
}

fn format_element(element: &str) -> String {
    let trimmed = element.trim();
    let mut chars = trimmed.chars();
    match (chars.next(), chars.next()) {
        (Some(first), Some(second)) => {
            format!(
                "{}{}",
                first.to_ascii_uppercase(),
                second.to_ascii_lowercase()
            )
        }
        (Some(first), None) => first.to_ascii_uppercase().to_string(),
        _ => "  ".into(),
    }
}
