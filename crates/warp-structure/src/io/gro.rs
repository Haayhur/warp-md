use std::fs::File;
use std::io::{BufReader, Write};
use std::path::Path;

use crate::error::{StructureError as PackError, StructureResult as PackResult};
use crate::geometry::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};
use traj_core::elements::infer_element_from_atom_name;
use traj_core::pdb_gro::parse_gro_reader;

const GRO_TO_INTERNAL_LENGTH_SCALE: f32 = 10.0;

pub fn read_gro(path: &Path) -> PackResult<MoleculeData> {
    read_gro_with_strict(path, false)
}

pub(crate) fn read_gro_with_strict(path: &Path, strict: bool) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let parsed = parse_gro_reader(reader, strict).map_err(|e| PackError::Parse(e.to_string()))?;
    let atoms = parsed
        .atoms
        .into_iter()
        .map(|atom| {
            let element = if atom.element.is_empty() {
                infer_element_from_atom_name(&atom.name).unwrap_or_else(|| "X".into())
            } else {
                atom.element
            };
            AtomRecord {
                record_kind: AtomRecordKind::Atom,
                name: atom.name,
                element,
                resname: if atom.resname.is_empty() {
                    "MOL".into()
                } else {
                    atom.resname
                },
                resid: atom.resid,
                chain: 'A',
                segid: String::new(),
                charge: 0.0,
                position: Vec3::new(
                    atom.position[0] * GRO_TO_INTERNAL_LENGTH_SCALE,
                    atom.position[1] * GRO_TO_INTERNAL_LENGTH_SCALE,
                    atom.position[2] * GRO_TO_INTERNAL_LENGTH_SCALE,
                ),
                mol_id: 1,
                pdb_metadata: None,
            }
        })
        .collect::<Vec<_>>();
    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in gro".into()));
    }
    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        box_vectors: parsed.box_vectors.map(scale_box_vectors),
        ter_after: Vec::new(),
    })
}

pub fn write_gro(out: &PackOutput, path: &str, scale: f32, box_sides_fix: f32) -> PackResult<()> {
    let mut file = File::create(path)?;
    writeln!(file, "warp_pack")?;
    writeln!(file, "{:>5}", out.atoms.len())?;
    for (i, atom) in out.atoms.iter().enumerate() {
        let idx = i + 1;
        let p = atom.position.scale(scale);
        let resid = atom.resid.rem_euclid(100000);
        let idx = idx % 100000;
        let resname = gro_fixed_field(&atom.resname);
        let name = gro_fixed_field(&atom.name);
        let line = format!(
            "{resid:>5}{resname:<5}{name:>5}{idx:>5}{x:>8.3}{y:>8.3}{z:>8.3}\n",
            resid = resid,
            resname = resname,
            name = name,
            idx = idx,
            x = p.x,
            y = p.y,
            z = p.z
        );
        file.write_all(line.as_bytes())?;
    }
    write_gro_box(&mut file, out, scale, box_sides_fix)?;
    Ok(())
}

fn gro_fixed_field(value: &str) -> String {
    value.trim().chars().take(5).collect()
}

fn scale_box_vectors(box_vectors: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    [
        box_vectors[0].map(|value| value * GRO_TO_INTERNAL_LENGTH_SCALE),
        box_vectors[1].map(|value| value * GRO_TO_INTERNAL_LENGTH_SCALE),
        box_vectors[2].map(|value| value * GRO_TO_INTERNAL_LENGTH_SCALE),
    ]
}

fn write_gro_box(
    file: &mut File,
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
        .unwrap_or_else(|| {
            let b = out.box_size;
            [
                [(b[0] + box_sides_fix), 0.0, 0.0],
                [0.0, (b[1] + box_sides_fix), 0.0],
                [0.0, 0.0, (b[2] + box_sides_fix)],
            ]
        });
    let box_vectors = [
        box_vectors[0].map(|value| value * scale),
        box_vectors[1].map(|value| value * scale),
        box_vectors[2].map(|value| value * scale),
    ];
    if is_orthorhombic_box(&box_vectors) {
        writeln!(
            file,
            "{:.5} {:.5} {:.5}",
            box_vectors[0][0], box_vectors[1][1], box_vectors[2][2]
        )?;
    } else {
        writeln!(
            file,
            "{:.5} {:.5} {:.5} {:.5} {:.5} {:.5} {:.5} {:.5} {:.5}",
            box_vectors[0][0],
            box_vectors[1][1],
            box_vectors[2][2],
            box_vectors[0][1],
            box_vectors[0][2],
            box_vectors[1][0],
            box_vectors[1][2],
            box_vectors[2][0],
            box_vectors[2][1]
        )?;
    }
    Ok(())
}

fn is_orthorhombic_box(box_vectors: &[[f32; 3]; 3]) -> bool {
    box_vectors[0][1].abs() <= 1.0e-8
        && box_vectors[0][2].abs() <= 1.0e-8
        && box_vectors[1][0].abs() <= 1.0e-8
        && box_vectors[1][2].abs() <= 1.0e-8
        && box_vectors[2][0].abs() <= 1.0e-8
        && box_vectors[2][1].abs() <= 1.0e-8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gro_writer_truncates_fixed_width_names() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("long_names.gro");
        let out = PackOutput {
            atoms: vec![AtomRecord {
                record_kind: AtomRecordKind::Atom,
                name: "LONGATOM".to_string(),
                element: "X".to_string(),
                resname: "SUCROSE".to_string(),
                resid: 1,
                chain: 'A',
                segid: String::new(),
                charge: 0.0,
                position: Vec3::new(1.0, 2.0, 3.0),
                mol_id: 1,
                pdb_metadata: None,
            }],
            bonds: Vec::new(),
            box_size: [10.0, 10.0, 10.0],
            box_vectors: None,
            ter_after: Vec::new(),
        };
        write_gro(&out, path.to_str().unwrap(), 0.1, 0.0).unwrap();

        let text = std::fs::read_to_string(path).unwrap();
        let atom_line = text.lines().nth(2).unwrap();
        assert_eq!(&atom_line[5..10], "SUCRO");
        assert_eq!(&atom_line[10..15], "LONGA");
        assert_eq!(atom_line[20..28].trim().parse::<f32>().unwrap(), 0.1);
    }
}
