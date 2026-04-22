use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::error::{StructureError as PackError, StructureResult as PackResult};
use crate::geometry::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};
use traj_core::elements::infer_element_from_atom_name;

const G96_TO_INTERNAL_LENGTH_SCALE: f32 = 10.0;

pub fn read_gromos96(path: &Path) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut atoms = Vec::new();
    let mut box_vectors = None;
    let mut section = Section::Other;

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "TITLE" | "TIMESTEP" | "VELOCITY" | "VELOCITYRED" => {
                section = Section::Other;
                continue;
            }
            "POSITION" => {
                section = Section::Position;
                continue;
            }
            "POSITIONRED" => {
                section = Section::PositionReduced;
                continue;
            }
            "BOX" => {
                section = Section::Box;
                continue;
            }
            "END" => {
                section = Section::Other;
                continue;
            }
            _ => {}
        }

        match section {
            Section::Position => atoms.push(parse_position_line(trimmed)?),
            Section::PositionReduced => {
                atoms.push(parse_reduced_position_line(trimmed, atoms.len())?)
            }
            Section::Box => {
                box_vectors = Some(parse_box_line(trimmed)?);
            }
            Section::Other => {}
        }
    }

    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in gromos96".into()));
    }

    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        box_vectors,
        ter_after: Vec::new(),
    })
}

pub fn write_gromos96(
    out: &PackOutput,
    path: &str,
    scale: f32,
    box_sides_fix: f32,
) -> PackResult<()> {
    let file = File::create(path)?;
    let mut file = BufWriter::new(file);
    writeln!(file, "TITLE")?;
    writeln!(file, "warp_pack")?;
    writeln!(file, "END")?;
    writeln!(file, "POSITION")?;
    for (i, atom) in out.atoms.iter().enumerate() {
        let resid = atom.resid.rem_euclid(100_000);
        let atom_id = (i + 1) % 10_000_000;
        let position = atom.position.scale(scale);
        let resname = truncate_left(&atom.resname, 5);
        let atom_name = truncate_left(&atom.name, 5);
        writeln!(
            file,
            "{resid:>5} {resname:<5} {atom_name:<5}{atom_id:>7}{x:>15.9}{y:>15.9}{z:>15.9}",
            x = position.x,
            y = position.y,
            z = position.z,
        )?;
    }
    writeln!(file, "END")?;
    write_box_section(&mut file, out, scale, box_sides_fix)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Section {
    Other,
    Position,
    PositionReduced,
    Box,
}

fn parse_position_line(line: &str) -> PackResult<AtomRecord> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 7 {
        return Err(PackError::Parse(format!(
            "invalid gromos96 POSITION line: {line}"
        )));
    }
    let resid = parts[0]
        .parse::<i32>()
        .map_err(|_| PackError::Parse("bad gromos96 residue id".into()))?;
    let resname = parts[1].to_string();
    let name = parts[2].to_string();
    let x = parse_g96_coord(parts[4], "x")?;
    let y = parse_g96_coord(parts[5], "y")?;
    let z = parse_g96_coord(parts[6], "z")?;
    let element = infer_element_from_atom_name(&name).unwrap_or_else(|| "X".into());
    Ok(AtomRecord {
        record_kind: AtomRecordKind::Atom,
        name,
        element,
        resname,
        resid,
        chain: 'A',
        segid: String::new(),
        charge: 0.0,
        position: Vec3::new(x, y, z),
        mol_id: 1,
        pdb_metadata: None,
    })
}

fn parse_reduced_position_line(line: &str, atom_index: usize) -> PackResult<AtomRecord> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 3 {
        return Err(PackError::Parse(format!(
            "invalid gromos96 POSITIONRED line: {line}"
        )));
    }
    let x = parse_g96_coord(parts[0], "x")?;
    let y = parse_g96_coord(parts[1], "y")?;
    let z = parse_g96_coord(parts[2], "z")?;
    Ok(AtomRecord {
        record_kind: AtomRecordKind::Atom,
        name: format!("X{}", atom_index + 1),
        element: "X".into(),
        resname: "MOL".into(),
        resid: 1,
        chain: 'A',
        segid: String::new(),
        charge: 0.0,
        position: Vec3::new(x, y, z),
        mol_id: 1,
        pdb_metadata: None,
    })
}

fn parse_g96_coord(value: &str, axis: &str) -> PackResult<f32> {
    value
        .parse::<f32>()
        .map(|v| v * G96_TO_INTERNAL_LENGTH_SCALE)
        .map_err(|_| PackError::Parse(format!("bad gromos96 {axis}")))
}

fn parse_box_line(line: &str) -> PackResult<[[f32; 3]; 3]> {
    let values = line
        .split_whitespace()
        .map(|value| {
            value
                .parse::<f32>()
                .map(|v| v * G96_TO_INTERNAL_LENGTH_SCALE)
                .map_err(|_| PackError::Parse("bad gromos96 box value".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    match values.as_slice() {
        [xx, yy, zz] => Ok([[*xx, 0.0, 0.0], [0.0, *yy, 0.0], [0.0, 0.0, *zz]]),
        [xx, yy, zz, xy, xz, yx, yz, zx, zy] => {
            Ok([[*xx, *xy, *xz], [*yx, *yy, *yz], [*zx, *zy, *zz]])
        }
        _ => Err(PackError::Parse(
            "gromos96 BOX section must have 3 or 9 values".into(),
        )),
    }
}

fn write_box_section<W: Write>(
    file: &mut W,
    out: &PackOutput,
    scale: f32,
    box_sides_fix: f32,
) -> PackResult<()> {
    let box_vectors = out
        .box_vectors
        .map(|vectors| {
            if is_orthorhombic_box(&vectors) {
                [
                    [vectors[0][0] + box_sides_fix, 0.0, 0.0],
                    [0.0, vectors[1][1] + box_sides_fix, 0.0],
                    [0.0, 0.0, vectors[2][2] + box_sides_fix],
                ]
            } else {
                vectors
            }
        })
        .unwrap_or([
            [out.box_size[0] + box_sides_fix, 0.0, 0.0],
            [0.0, out.box_size[1] + box_sides_fix, 0.0],
            [0.0, 0.0, out.box_size[2] + box_sides_fix],
        ]);
    let box_vectors = [
        box_vectors[0].map(|value| value * scale),
        box_vectors[1].map(|value| value * scale),
        box_vectors[2].map(|value| value * scale),
    ];
    writeln!(file, "BOX")?;
    write!(
        file,
        "{:>15.9}{:>15.9}{:>15.9}",
        box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]
    )?;
    if !is_orthorhombic_box(&box_vectors) {
        write!(
            file,
            "{:>15.9}{:>15.9}{:>15.9}{:>15.9}{:>15.9}{:>15.9}",
            box_vectors[0][1],
            box_vectors[0][2],
            box_vectors[1][0],
            box_vectors[1][2],
            box_vectors[2][0],
            box_vectors[2][1]
        )?;
    }
    writeln!(file)?;
    writeln!(file, "END")?;
    Ok(())
}

fn truncate_left(value: &str, width: usize) -> String {
    value.trim().chars().take(width).collect()
}

fn is_orthorhombic_box(box_vectors: &[[f32; 3]; 3]) -> bool {
    box_vectors[0][1].abs() <= 1.0e-8
        && box_vectors[0][2].abs() <= 1.0e-8
        && box_vectors[1][0].abs() <= 1.0e-8
        && box_vectors[1][2].abs() <= 1.0e-8
        && box_vectors[2][0].abs() <= 1.0e-8
        && box_vectors[2][1].abs() <= 1.0e-8
}
