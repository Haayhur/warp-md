use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{StructureError as PackError, StructureResult as PackResult};
use crate::geometry::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput, PdbAtomMetadata};
use traj_core::elements::infer_element_from_atom_name;

pub fn read_pqr(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut atoms = Vec::new();
    let mut box_vectors = None;
    let mut ter_after = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("CRYST1") {
            box_vectors = Some(parse_cryst1_line(&line)?);
            continue;
        }
        if line.starts_with("TER") {
            if let Some(last) = atoms.len().checked_sub(1) {
                ter_after.push(last);
            }
            continue;
        }
        if !(line.starts_with("ATOM") || line.starts_with("HETATM")) {
            continue;
        }
        atoms.push(parse_pqr_atom_line(&line)?);
    }

    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in pqr".into()));
    }

    ter_after.sort_unstable();
    ter_after.dedup();
    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        box_vectors,
        ter_after,
    })
}

pub fn write_pqr(
    out: &PackOutput,
    path: &str,
    scale: f32,
    add_box_sides: bool,
    box_sides_fix: f32,
) -> PackResult<()> {
    let file = File::create(path)?;
    let mut file = BufWriter::new(file);
    if add_box_sides {
        let ((a, b, c), (alpha, beta, gamma)) = pqr_box_parameters(out, box_sides_fix, scale);
        writeln!(
            file,
            "CRYST1{:>9.3}{:>9.3}{:>9.3}{:>7.2}{:>7.2}{:>7.2} P 1           1",
            a, b, c, alpha, beta, gamma
        )?;
    }
    let mut ter_iter = out.ter_after.iter().copied().peekable();
    for (i, atom) in out.atoms.iter().enumerate() {
        let idx = (i + 1) % 100_000;
        let position = atom.position.scale(scale);
        let atom_name = truncate_left(atom.name.trim(), 4);
        let resname = truncate_left(atom.resname.trim(), 4);
        let chain = if atom.chain == ' ' { 'A' } else { atom.chain };
        let resid = atom.resid % 10_000;
        let radius = atom
            .pdb_metadata
            .as_ref()
            .and_then(|metadata| metadata.pqr_radius)
            .unwrap_or(0.0);
        writeln!(
            file,
            "{record:<6}{idx:>5} {atom_name:<4}{resname:>4}{chain}{resid:>4} {x:>8.3} {y:>8.3} {z:>8.3} {charge:>6.2} {radius:>6.2}",
            record = match atom.record_kind {
                AtomRecordKind::Atom => "ATOM",
                AtomRecordKind::HetAtom => "HETATM",
            },
            x = position.x,
            y = position.y,
            z = position.z,
            charge = atom.charge,
        )?;
        if matches!(ter_iter.peek(), Some(next) if *next == i) {
            writeln!(
                file,
                "TER   {:>5}      {:>4} {}{:>4}",
                (idx + 1) % 100_000,
                resname,
                chain,
                resid
            )?;
            ter_iter.next();
        }
    }
    writeln!(file, "END")?;
    Ok(())
}

fn parse_pqr_atom_line(line: &str) -> PackResult<AtomRecord> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    if fields.len() < 10 {
        return Err(PackError::Parse(format!("invalid pqr atom line: {line}")));
    }
    let coord_start = fields.len() - 5;
    let prefix = &fields[..coord_start];
    let (record_kind, name, resname, chain, resid) = match prefix {
        [record, serial, name, resname, resid] => {
            parse_pqr_serial(serial)?;
            (
                parse_record_kind(record)?,
                (*name).to_string(),
                (*resname).to_string(),
                'A',
                parse_pqr_resid(resid)?,
            )
        }
        [record, serial, name, resname, chain, resid] => {
            parse_pqr_serial(serial)?;
            (
                parse_record_kind(record)?,
                (*name).to_string(),
                (*resname).to_string(),
                chain.chars().next().unwrap_or('A'),
                parse_pqr_resid(resid)?,
            )
        }
        _ => {
            return Err(PackError::Parse(format!(
                "unsupported pqr atom layout: {line}"
            )))
        }
    };
    let x = parse_pqr_float(fields[coord_start], "x")?;
    let y = parse_pqr_float(fields[coord_start + 1], "y")?;
    let z = parse_pqr_float(fields[coord_start + 2], "z")?;
    let charge = parse_pqr_float(fields[coord_start + 3], "charge")?;
    let radius = parse_pqr_float(fields[coord_start + 4], "radius")?;
    let element = infer_element_from_atom_name(&name).unwrap_or_else(|| "X".into());
    Ok(AtomRecord {
        record_kind,
        name,
        element,
        resname,
        resid,
        chain,
        segid: String::new(),
        charge,
        position: Vec3::new(x, y, z),
        mol_id: 1,
        pdb_metadata: Some(PdbAtomMetadata {
            occupancy: None,
            temp_factor: None,
            altloc: None,
            insertion_code: None,
            formal_charge: None,
            pqr_radius: Some(radius),
        }),
    })
}

fn parse_record_kind(value: &str) -> PackResult<AtomRecordKind> {
    match value {
        "ATOM" => Ok(AtomRecordKind::Atom),
        "HETATM" => Ok(AtomRecordKind::HetAtom),
        _ => Err(PackError::Parse(format!(
            "unsupported pqr record type: {value}"
        ))),
    }
}

fn parse_pqr_serial(value: &str) -> PackResult<i32> {
    value
        .parse::<i32>()
        .map_err(|_| PackError::Parse(format!("invalid pqr serial '{value}'")))
}

fn parse_pqr_resid(value: &str) -> PackResult<i32> {
    value
        .parse::<i32>()
        .map_err(|_| PackError::Parse(format!("invalid pqr residue id '{value}'")))
}

fn parse_pqr_float(value: &str, label: &str) -> PackResult<f32> {
    value
        .parse::<f32>()
        .map_err(|_| PackError::Parse(format!("invalid pqr {label} '{value}'")))
}

fn parse_cryst1_line(line: &str) -> PackResult<[[f32; 3]; 3]> {
    let a = parse_cryst1_field(line, 6, 15, "a")?;
    let b = parse_cryst1_field(line, 15, 24, "b")?;
    let c = parse_cryst1_field(line, 24, 33, "c")?;
    let alpha = parse_cryst1_field(line, 33, 40, "alpha")?;
    let beta = parse_cryst1_field(line, 40, 47, "beta")?;
    let gamma = parse_cryst1_field(line, 47, 54, "gamma")?;
    Ok(cryst1_to_box_vectors(a, b, c, alpha, beta, gamma))
}

fn parse_cryst1_field(line: &str, start: usize, end: usize, label: &str) -> PackResult<f32> {
    line.get(start..end)
        .unwrap_or("")
        .trim()
        .parse::<f32>()
        .map_err(|_| PackError::Parse(format!("invalid CRYST1 {label} field")))
}

fn cryst1_to_box_vectors(
    a: f32,
    b: f32,
    c: f32,
    alpha_deg: f32,
    beta_deg: f32,
    gamma_deg: f32,
) -> [[f32; 3]; 3] {
    let alpha = alpha_deg.to_radians();
    let beta = beta_deg.to_radians();
    let gamma = gamma_deg.to_radians();
    let cos_alpha = alpha.cos();
    let cos_beta = beta.cos();
    let cos_gamma = gamma.cos();
    let sin_gamma = gamma.sin();
    if sin_gamma.abs() <= 1.0e-8 {
        return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, c]];
    }
    let cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    let cz_sq = (c * c - (c * cos_beta) * (c * cos_beta) - cy * cy).max(0.0);
    [
        [a, 0.0, 0.0],
        [b * cos_gamma, b * sin_gamma, 0.0],
        [c * cos_beta, cy, cz_sq.sqrt()],
    ]
}

fn pqr_box_parameters(
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

fn truncate_left(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
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
