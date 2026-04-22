use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{StructureError as PackError, StructureResult as PackResult};
use crate::geometry::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput, PdbAtomMetadata};
use traj_core::pdb_gro::{parse_pdb_reader, PdbParseOptions, PdbParseResult, PdbRecordKind};

pub fn read_pdb(
    path: &Path,
    ignore_conect: bool,
    non_standard_conect: bool,
) -> PackResult<MoleculeData> {
    let options = PdbParseOptions {
        include_conect: !ignore_conect,
        non_standard_conect,
        include_ter: true,
        strict: false,
        only_first_model: false,
    };
    read_pdb_with_options(path, &options)
}

pub(crate) fn read_pdb_with_options(
    path: &Path,
    options: &PdbParseOptions,
) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let parsed = parse_pdb_reader(reader, options).map_err(|e| PackError::Parse(e.to_string()))?;
    molecule_from_parsed(parsed)
}

pub(crate) fn molecule_from_parsed(parsed: PdbParseResult) -> PackResult<MoleculeData> {
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
                pdb_metadata: Some(PdbAtomMetadata {
                    occupancy: Some(atom.occupancy),
                    temp_factor: Some(atom.temp_factor),
                    altloc: Some(atom.altloc),
                    insertion_code: Some(atom.icode),
                    formal_charge: if atom.charge.trim().is_empty() {
                        None
                    } else {
                        Some(atom.charge)
                    },
                    pqr_radius: None,
                }),
            }
        })
        .collect::<Vec<_>>();
    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in pdb".into()));
    }
    Ok(MoleculeData {
        atoms,
        bonds: parsed.bonds,
        box_vectors: None,
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
    validate_pdb_output(out, hexadecimal_indices, strict_fields)?;
    let file = File::create(path)?;
    let mut file = BufWriter::new(file);
    if add_box_sides {
        let ((a, b, c), (alpha, beta, gamma)) = pdb_box_parameters(out, box_sides_fix, scale);
        writeln!(
            file,
            "CRYST1{:>9.3}{:>9.3}{:>9.3}{:>7.2}{:>7.2}{:>7.2} P 1           1",
            a, b, c, alpha, beta, gamma
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
        let segid = format_segid(&atom.segid);
        let metadata = atom.pdb_metadata.as_ref();
        let alt = metadata.and_then(|value| sanitize_pdb_char(value.altloc));
        let insertion_code = metadata.and_then(|value| sanitize_pdb_char(value.insertion_code));
        let occupancy = metadata.and_then(|value| value.occupancy).unwrap_or(1.0);
        let temp_factor = metadata.and_then(|value| value.temp_factor).unwrap_or(0.0);
        let formal_charge =
            format_pdb_charge(metadata.and_then(|value| value.formal_charge.as_deref()));
        let line = format!(
            "{record}{idx:>5} {name}{alt}{resname:>3} {chain}{resid:>4}{icode}   {x:>8.3}{y:>8.3}{z:>8.3}{occ:>6.2}{temp:>6.2}      {segid:<4}{element:>2}{charge:>2}\n",
            record = record,
            idx = idx_str,
            name = atom_name,
            alt = alt.unwrap_or(' '),
            resname = resname,
            chain = chain,
            resid = resid_str,
            icode = insertion_code.unwrap_or(' '),
            x = pos.x,
            y = pos.y,
            z = pos.z,
            occ = occupancy,
            temp = temp_factor,
            segid = segid,
            element = element,
            charge = formal_charge,
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

fn validate_pdb_output(
    out: &PackOutput,
    hexadecimal_indices: bool,
    strict_fields: bool,
) -> PackResult<()> {
    if strict_fields && hexadecimal_indices {
        return Err(PackError::Invalid(
            "pdb-strict does not support hexadecimal atom or residue indices; use mmcif for larger systems".into(),
        ));
    }
    if !hexadecimal_indices && out.atoms.len() > 99_999 {
        return Err(PackError::Invalid(
            "pdb atom count exceeds fixed-column width; use mmcif".into(),
        ));
    }
    for atom in &out.atoms {
        if !hexadecimal_indices && (atom.resid < -999 || atom.resid > 9_999) {
            return Err(PackError::Invalid(format!(
                "pdb residue id '{}' exceeds fixed-column width; use mmcif",
                atom.resid
            )));
        }
        if !strict_fields {
            continue;
        }
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
        if atom.segid.trim().len() > 4 {
            return Err(PackError::Invalid(format!(
                "pdb-strict segid '{}' exceeds 4 characters; use mmcif",
                atom.segid.trim()
            )));
        }
        if atom
            .pdb_metadata
            .as_ref()
            .and_then(|value| value.formal_charge.as_deref())
            .map(|value| value.trim().len() > 2)
            .unwrap_or(false)
        {
            return Err(PackError::Invalid(
                "pdb-strict formal charge exceeds 2 characters; use mmcif".into(),
            ));
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

fn pdb_box_parameters(
    out: &PackOutput,
    box_sides_fix: f32,
    scale: f32,
) -> ((f32, f32, f32), (f32, f32, f32)) {
    let box_vectors = out.box_vectors.unwrap_or([
        [out.box_size[0] + box_sides_fix, 0.0, 0.0],
        [0.0, out.box_size[1] + box_sides_fix, 0.0],
        [0.0, 0.0, out.box_size[2] + box_sides_fix],
    ]);
    let a = vector_norm(box_vectors[0]) * scale;
    let b = vector_norm(box_vectors[1]) * scale;
    let c = vector_norm(box_vectors[2]) * scale;
    let alpha = angle_deg(box_vectors[1], box_vectors[2]);
    let beta = angle_deg(box_vectors[0], box_vectors[2]);
    let gamma = angle_deg(box_vectors[0], box_vectors[1]);
    ((a, b, c), (alpha, beta, gamma))
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
    if hex {
        format!("{:X}", idx)
    } else {
        format!("{idx}")
    }
}

fn format_res_id(resid: i32, hex: bool) -> String {
    if resid >= 0 && hex {
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

fn format_segid(segid: &str) -> String {
    truncate_right(segid.trim(), 4)
}

fn sanitize_pdb_char(value: Option<char>) -> Option<char> {
    value.filter(|value| value.is_ascii_alphanumeric())
}

fn format_pdb_charge(charge: Option<&str>) -> String {
    let trimmed = charge.unwrap_or("").trim();
    if trimmed.is_empty() {
        "  ".into()
    } else {
        let mut value = trimmed.chars().take(2).collect::<String>();
        if value.len() == 1 {
            value.insert(0, ' ');
        }
        value
    }
}

fn vector_norm(vector: [f32; 3]) -> f32 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn angle_deg(a: [f32; 3], b: [f32; 3]) -> f32 {
    let norm_a = vector_norm(a);
    let norm_b = vector_norm(b);
    if norm_a <= 1.0e-8 || norm_b <= 1.0e-8 {
        return 90.0;
    }
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    (dot / (norm_a * norm_b))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
}
