//! Conversions between warp-pep's Structure and warp-pack's PackOutput / MoleculeData.

use std::collections::HashMap;
use std::path::Path;

use warp_pack::io::{read_molecule, write_output, MoleculeData};
use warp_pack::{AtomRecord, OutputSpec, PackOutput};

use crate::coord::Vec3 as PepVec3;
use crate::residue::{parse_amber_name, Atom, Chain, Residue, Structure};

/// Convert warp-pep Structure → warp-pack PackOutput for multi-format writing.
pub fn structure_to_pack_output(struc: &Structure) -> PackOutput {
    let mut atoms = Vec::new();
    let mut ter_after = Vec::new();

    for chain in &struc.chains {
        for res in &chain.residues {
            for atom in &res.atoms {
                atoms.push(AtomRecord {
                    record_kind: warp_pack::pack::AtomRecordKind::Atom,
                    name: atom.name.clone(),
                    element: atom.element.clone(),
                    resname: res.amber_name().to_string(),
                    resid: res.seq_id,
                    chain: chain.id,
                    segid: String::new(),
                    charge: 0.0,
                    position: warp_pack::geom::Vec3::new(
                        atom.coord.x as f32,
                        atom.coord.y as f32,
                        atom.coord.z as f32,
                    ),
                    mol_id: 0,
                });
            }
        }
        // TER after last atom of each chain
        if !atoms.is_empty() {
            ter_after.push(atoms.len() - 1);
        }
    }

    // Map disulfide bonds to atom-index pairs (SG—SG)
    let ss_bonds: Vec<(usize, usize)> = struc
        .ssbonds
        .iter()
        .filter_map(|ss| {
            let i1 = atoms
                .iter()
                .position(|a| a.chain == ss.chain1 && a.resid == ss.resid1 && a.name == "SG")?;
            let i2 = atoms
                .iter()
                .position(|a| a.chain == ss.chain2 && a.resid == ss.resid2 && a.name == "SG")?;
            Some((i1, i2))
        })
        .collect();

    PackOutput {
        atoms,
        bonds: ss_bonds,
        box_size: [0.0; 3],
        ter_after,
    }
}

/// Convert warp-pack MoleculeData → warp-pep Structure (for reading input files).
pub fn molecule_data_to_structure(mol: &MoleculeData) -> Result<Structure, String> {
    let mut struc = Structure::new_empty();
    let mut chain_index_by_id: HashMap<char, usize> = HashMap::new();

    for arec in &mol.atoms {
        let chain_idx = if let Some(idx) = chain_index_by_id.get(&arec.chain).copied() {
            idx
        } else {
            struc.chains.push(Chain::new(arec.chain));
            let idx = struc.chains.len() - 1;
            chain_index_by_id.insert(arec.chain, idx);
            idx
        };
        let chain = &mut struc.chains[chain_idx];
        let res_changed = chain
            .residues
            .last()
            .map_or(true, |res| res.seq_id != arec.resid);

        if res_changed {
            // Check for non-standard residues before Amber parsing.
            let nsr = crate::non_standard::NonStdResidue::from_str(&arec.resname);
            let (resname, variant) = if let Some(ns) = nsr {
                (ns.canonical(), None)
            } else {
                parse_amber_name(&arec.resname).ok_or_else(|| {
                    format!(
                        "unknown residue name '{}' at resid {}",
                        arec.resname, arec.resid
                    )
                })?
            };
            let mut res = Residue::new(resname, arec.resid);
            res.variant = variant;
            res.non_std = nsr;
            chain.residues.push(res);
        }

        let atom = Atom::new(
            &arec.name,
            &arec.element,
            PepVec3::new(
                arec.position.x as f64,
                arec.position.y as f64,
                arec.position.z as f64,
            ),
        );
        chain.residues.last_mut().unwrap().add_atom(atom);
    }

    if struc.chains.is_empty() {
        return Err("no atoms in input file".into());
    }

    Ok(struc)
}

/// Read a structure from any supported file format via warp-pack.
pub fn read_structure(path: &str) -> Result<Structure, String> {
    let p = Path::new(path);
    let mol = read_molecule(p, None, false, false, None)
        .map_err(|e| format!("failed to read '{}': {}", path, e))?;
    molecule_data_to_structure(&mol)
}

/// Infer output format from file extension.
pub fn infer_format(path: &str) -> &str {
    let lower = path.to_lowercase();
    if lower.ends_with(".pdb") {
        "pdb"
    } else if lower.ends_with(".cif") || lower.ends_with(".mmcif") {
        "pdbx"
    } else if lower.ends_with(".xyz") {
        "xyz"
    } else if lower.ends_with(".gro") {
        "gro"
    } else if lower.ends_with(".mol2") {
        "mol2"
    } else if lower.ends_with(".crd") {
        "crd"
    } else if lower.ends_with(".lmp") || lower.ends_with(".lammps") {
        "lammps"
    } else {
        "pdb"
    }
}

/// Write a warp-pep Structure to a file in the given format (or auto-detect from extension).
pub fn write_structure(struc: &Structure, path: &str, format: Option<&str>) -> Result<(), String> {
    let pack_out = structure_to_pack_output(struc);
    let fmt = format.unwrap_or_else(|| infer_format(path));
    let write_conect = !pack_out.bonds.is_empty();
    let spec = OutputSpec {
        path: path.to_string(),
        format: fmt.to_string(),
        scale: None,
    };
    write_output(&pack_out, &spec, false, 0.0, write_conect, false)
        .map_err(|e| format!("failed to write '{}': {}", path, e))
}

/// Write a warp-pep Structure to stdout in the given format (default: pdb).
pub fn write_structure_stdout(struc: &Structure, format: &str) -> Result<(), String> {
    let pack_out = structure_to_pack_output(struc);
    let write_conect = !pack_out.bonds.is_empty();
    let tmp = std::env::temp_dir().join(format!("warp_pep_stdout_{}.tmp", std::process::id()));
    let tmp_path = tmp.to_string_lossy().to_string();
    let spec = OutputSpec {
        path: tmp_path.clone(),
        format: format.to_string(),
        scale: None,
    };
    write_output(&pack_out, &spec, false, 0.0, write_conect, false)
        .map_err(|e| format!("write failed: {}", e))?;
    let contents =
        std::fs::read_to_string(&tmp_path).map_err(|e| format!("read tmp failed: {}", e))?;
    let _ = std::fs::remove_file(&tmp_path);
    print!("{}", contents);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder;
    use crate::residue::ResName;

    #[test]
    fn test_round_trip_pdb() {
        // Build → write PDB → read back → check residue count + names
        let struc = builder::make_extended_structure("ACDE").unwrap();
        let tmp = std::env::temp_dir().join("warp_pep_rt_test.pdb");
        let path = tmp.to_string_lossy().to_string();
        write_structure(&struc, &path, None).unwrap();
        let back = read_structure(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(back.chain_a().residues.len(), 4);
        assert_eq!(back.chain_a().residues[0].name, ResName::ALA);
        assert_eq!(back.chain_a().residues[1].name, ResName::CYS);
        assert_eq!(back.chain_a().residues[2].name, ResName::ASP);
        assert_eq!(back.chain_a().residues[3].name, ResName::GLU);

        // Coordinates should survive within f32 roundtrip tolerance
        let orig_ca = struc.chain_a().residues[0].atom_coord("CA").unwrap();
        let back_ca = back.chain_a().residues[0].atom_coord("CA").unwrap();
        let delta = orig_ca.sub(back_ca).length();
        assert!(delta < 0.01, "CA coord drift {delta} > 0.01");
    }

    #[test]
    fn test_round_trip_amber_naming() {
        // CYX/HID must survive write→read round-trip
        let specs = builder::parse_three_letter_sequence("ALA-CYX-HID").unwrap();
        let struc = builder::make_extended_structure_from_specs(&specs).unwrap();
        let tmp = std::env::temp_dir().join("warp_pep_rt_amber.pdb");
        let path = tmp.to_string_lossy().to_string();
        write_structure(&struc, &path, None).unwrap();
        let back = read_structure(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(back.chain_a().residues[1].amber_name(), "CYX");
        assert_eq!(back.chain_a().residues[2].amber_name(), "HID");
    }

    #[test]
    fn test_multi_chain_pack_output() {
        let chains = vec![
            builder::ChainSpec {
                id: 'A',
                residues: builder::parse_three_letter_sequence("ALA-GLY").unwrap(),
                preset: None,
            },
            builder::ChainSpec {
                id: 'B',
                residues: builder::parse_three_letter_sequence("VAL-TRP").unwrap(),
                preset: None,
            },
        ];
        let struc = builder::make_multi_chain_structure(&chains).unwrap();
        let pack = structure_to_pack_output(&struc);
        // Should have TER records after each chain
        assert_eq!(pack.ter_after.len(), 2);
        // Chain B atoms should have chain='B'
        let last = pack.atoms.last().unwrap();
        assert_eq!(last.chain, 'B');
    }

    #[test]
    fn test_multi_chain_round_trip() {
        // Build multi-chain → write → read → verify chain IDs preserved
        let chains = vec![
            builder::ChainSpec {
                id: 'A',
                residues: builder::parse_three_letter_sequence("ALA-GLY").unwrap(),
                preset: None,
            },
            builder::ChainSpec {
                id: 'B',
                residues: builder::parse_three_letter_sequence("VAL-TRP").unwrap(),
                preset: None,
            },
        ];
        let struc = builder::make_multi_chain_structure(&chains).unwrap();
        let tmp = std::env::temp_dir().join("warp_pep_rt_mc.pdb");
        let path = tmp.to_string_lossy().to_string();
        write_structure(&struc, &path, None).unwrap();
        let back = read_structure(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(back.chains.len(), 2);
        assert_eq!(back.chains[0].id, 'A');
        assert_eq!(back.chains[1].id, 'B');
        assert_eq!(back.chains[0].residues.len(), 2);
        assert_eq!(back.chains[1].residues.len(), 2);
        assert_eq!(back.chains[1].residues[0].name, ResName::VAL);
    }

    #[test]
    fn test_unknown_residue_rejected() {
        use warp_pack::geom::Vec3 as PackVec3;
        // Fabricate MoleculeData with an unknown residue name
        let mol = MoleculeData {
            atoms: vec![AtomRecord {
                record_kind: warp_pack::pack::AtomRecordKind::Atom,
                name: "CA".into(),
                element: "C".into(),
                resname: "ZZZ".into(),
                resid: 1,
                chain: 'A',
                segid: String::new(),
                charge: 0.0,
                position: PackVec3::new(0.0, 0.0, 0.0),
                mol_id: 0,
            }],
            bonds: vec![],
            ter_after: vec![],
        };
        let result = molecule_data_to_structure(&mol);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("ZZZ"));
    }

    #[test]
    fn test_non_contiguous_chain_segments_reuse_chain_ids() {
        use warp_pack::geom::Vec3 as PackVec3;
        let mk =
            |name: &str, elem: &str, resname: &str, resid: i32, chain: char, x: f32| AtomRecord {
                record_kind: warp_pack::pack::AtomRecordKind::Atom,
                name: name.into(),
                element: elem.into(),
                resname: resname.into(),
                resid,
                chain,
                segid: String::new(),
                charge: 0.0,
                position: PackVec3::new(x, 0.0, 0.0),
                mol_id: 0,
            };
        let mol = MoleculeData {
            atoms: vec![
                mk("N", "N", "ALA", 1, 'A', 0.0),
                mk("N", "N", "GLY", 1, 'B', 1.0),
                mk("N", "N", "ALA", 2, 'A', 2.0),
            ],
            bonds: vec![],
            ter_after: vec![],
        };

        let struc = molecule_data_to_structure(&mol).unwrap();
        assert_eq!(struc.chains.len(), 2);
        assert_eq!(struc.chains.iter().filter(|c| c.id == 'A').count(), 1);
        assert_eq!(struc.chain_by_id('A').unwrap().residues.len(), 2);
        assert_eq!(struc.chain_by_id('B').unwrap().residues.len(), 1);
    }
}
