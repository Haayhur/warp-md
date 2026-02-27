use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};

pub fn read_pdbx(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

    let mut in_loop = false;
    let mut columns: Vec<String> = Vec::new();
    let mut atom_site = false;
    let mut tokens: Vec<String> = Vec::new();
    let mut atoms = Vec::new();

    let mut i = 0usize;
    while i < lines.len() {
        let raw = lines[i].trim_end();
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            i += 1;
            continue;
        }
        if line.starts_with("loop_") {
            in_loop = true;
            columns.clear();
            atom_site = false;
            tokens.clear();
            i += 1;
            continue;
        }
        if in_loop && line.starts_with('_') {
            let col = line.split_whitespace().next().unwrap_or("").to_string();
            if col.starts_with("_atom_site.") {
                atom_site = true;
            }
            columns.push(col);
            i += 1;
            continue;
        }
        if in_loop {
            if line.starts_with("loop_") {
                in_loop = false;
                continue;
            }
            let mut row_tokens = tokenize_cif_line(&lines, &mut i)?;
            tokens.append(&mut row_tokens);
            while atom_site && tokens.len() >= columns.len() {
                let row: Vec<String> = tokens.drain(0..columns.len()).collect();
                if let Some(atom) = atom_from_row(&columns, &row) {
                    atoms.push(atom);
                }
            }
            i += 1;
            continue;
        }
        i += 1;
    }

    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in pdbx".into()));
    }
    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        ter_after: Vec::new(),
    })
}

pub fn write_pdbx(out: &PackOutput, path: &str, scale: f32, _box_sides_fix: f32) -> PackResult<()> {
    let mut file = File::create(path)?;
    writeln!(file, "data_warp_pack")?;
    writeln!(file, "loop_")?;
    writeln!(file, "_atom_site.group_PDB")?;
    writeln!(file, "_atom_site.id")?;
    writeln!(file, "_atom_site.type_symbol")?;
    writeln!(file, "_atom_site.label_atom_id")?;
    writeln!(file, "_atom_site.label_comp_id")?;
    writeln!(file, "_atom_site.label_asym_id")?;
    writeln!(file, "_atom_site.label_seq_id")?;
    writeln!(file, "_atom_site.Cartn_x")?;
    writeln!(file, "_atom_site.Cartn_y")?;
    writeln!(file, "_atom_site.Cartn_z")?;
    for (i, atom) in out.atoms.iter().enumerate() {
        let idx = i + 1;
        let p = atom.position.scale(scale);
        writeln!(
            file,
            "ATOM {id} {elem} {name} {res} {chain} {resid} {:.3} {:.3} {:.3}",
            p.x,
            p.y,
            p.z,
            id = idx,
            elem = atom.element,
            name = atom.name,
            res = atom.resname,
            chain = atom.chain,
            resid = atom.resid
        )?;
    }
    Ok(())
}

fn atom_from_row(columns: &[String], row: &[String]) -> Option<AtomRecord> {
    let col_idx = |name: &str| columns.iter().position(|c| c == name);
    let pick = |name: &str| col_idx(name).and_then(|i| row.get(i)).map(|s| s.as_str());

    let x = pick("_atom_site.Cartn_x")?.parse::<f32>().ok()?;
    let y = pick("_atom_site.Cartn_y")?.parse::<f32>().ok()?;
    let z = pick("_atom_site.Cartn_z")?.parse::<f32>().ok()?;

    let name = pick("_atom_site.label_atom_id")
        .or_else(|| pick("_atom_site.auth_atom_id"))
        .unwrap_or("X");
    let element = pick("_atom_site.type_symbol")
        .or_else(|| pick("_atom_site.type_symbol"))
        .unwrap_or(name);
    let resname = pick("_atom_site.label_comp_id")
        .or_else(|| pick("_atom_site.auth_comp_id"))
        .unwrap_or("MOL");
    let chain = pick("_atom_site.label_asym_id")
        .or_else(|| pick("_atom_site.auth_asym_id"))
        .and_then(|s| s.chars().next())
        .unwrap_or('A');
    let resid = pick("_atom_site.label_seq_id")
        .or_else(|| pick("_atom_site.auth_seq_id"))
        .and_then(|s| s.parse::<i32>().ok())
        .unwrap_or(1);
    let kind = pick("_atom_site.group_PDB")
        .map(|s| s.to_ascii_uppercase())
        .map(|s| {
            if s == "HETATM" {
                AtomRecordKind::HetAtom
            } else {
                AtomRecordKind::Atom
            }
        })
        .unwrap_or(AtomRecordKind::Atom);

    Some(AtomRecord {
        record_kind: kind,
        name: name.to_string(),
        element: element.to_string(),
        resname: resname.to_string(),
        resid,
        chain,
        segid: String::new(),
        charge: 0.0,
        position: Vec3::new(x, y, z),
        mol_id: 1,
    })
}

fn tokenize_cif_line(lines: &[String], idx: &mut usize) -> PackResult<Vec<String>> {
    let line = lines
        .get(*idx)
        .ok_or_else(|| PackError::Parse("pdbx line index out of bounds".into()))?;
    let trimmed = line.trim_end();
    if trimmed.starts_with(';') {
        let mut text = String::new();
        let mut line_idx = *idx + 1;
        while line_idx < lines.len() {
            let l = &lines[line_idx];
            if l.starts_with(';') {
                *idx = line_idx;
                break;
            }
            if !text.is_empty() {
                text.push('\n');
            }
            text.push_str(l.trim_end());
            line_idx += 1;
        }
        return Ok(vec![text]);
    }
    Ok(tokenize_simple(trimmed))
}

fn tokenize_simple(line: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut quote: Option<char> = None;
    while let Some(ch) = chars.next() {
        if let Some(q) = quote {
            if ch == q {
                quote = None;
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
            } else {
                current.push(ch);
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            quote = Some(ch);
            continue;
        }
        if ch.is_whitespace() {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}
