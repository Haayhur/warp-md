use std::path::Path;

use traj_core::elements::mass_for_element;
use traj_core::interner::StringInterner;
use traj_core::pdb_gro::PdbParseOptions;
use traj_core::system::{AtomTable, System};

use crate::error::{StructureError, StructureResult};

use super::gro::read_gro_with_strict;
use super::pdb::read_pdb_with_options;
use super::pdbqt::read_pdbqt;
use super::MoleculeData;

const SUPPORTED_SYSTEM_FORMATS_TEXT: &str = "pdb, pdbqt, gro, prmtop, top";

fn path_format_token(path: &Path, format: Option<&str>) -> String {
    format
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| {
            path.extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
        })
}

fn resolved_system_format(path: &Path, format: Option<&str>) -> StructureResult<String> {
    let token = path_format_token(path, format);
    if matches!(token.as_str(), "pdb" | "pdbqt" | "gro" | "prmtop" | "top") {
        Ok(token)
    } else {
        Err(StructureError::Parse(format!(
            "system.format must be {SUPPORTED_SYSTEM_FORMATS_TEXT}"
        )))
    }
}

pub fn read_pdb_system(path: &Path, strict: bool) -> StructureResult<System> {
    let options = PdbParseOptions {
        include_conect: false,
        include_ter: false,
        non_standard_conect: false,
        strict,
        only_first_model: true,
    };
    let molecule = read_pdb_with_options(path, &options)?;
    system_from_molecule(&molecule)
}

pub fn read_pdbqt_system(path: &Path) -> StructureResult<System> {
    let molecule = read_pdbqt(path)?;
    system_from_molecule(&molecule)
}

pub fn read_gro_system(path: &Path) -> StructureResult<System> {
    let molecule = read_gro_with_strict(path, true)?;
    system_from_molecule(&molecule)
}

pub fn read_system_auto(path: &Path, format: Option<&str>) -> StructureResult<System> {
    let format = resolved_system_format(path, format)?;
    match format.as_str() {
        "pdb" => match read_pdb_system(path, true) {
            Ok(system) => Ok(system),
            Err(err) => {
                if err
                    .to_string()
                    .to_ascii_lowercase()
                    .contains("invalid resid")
                {
                    read_pdb_system(path, false)
                } else {
                    Err(err)
                }
            }
        },
        "pdbqt" => read_pdbqt_system(path),
        "gro" => read_gro_system(path),
        "prmtop" | "top" => {
            let topo = super::amber::read_prmtop_topology(path)?;
            system_from_prmtop(&topo)
        }
        _ => unreachable!("validated system format"),
    }
}

pub fn system_from_prmtop(topo: &super::amber::AmberTopology) -> StructureResult<System> {
    if topo.atom_names.is_empty() {
        return Err(StructureError::Invalid(
            "cannot project empty prmtop topology into system".into(),
        ));
    }

    let mut interner = StringInterner::new();
    let mut atoms = AtomTable::default();
    let n_atoms = topo.atom_names.len();

    for i in 0..n_atoms {
        let name = topo.atom_names[i].trim();
        let atomic_number = topo.atomic_numbers.get(i).copied().unwrap_or(0);
        let element = super::amber::atomic_number_to_symbol(atomic_number)
            .unwrap_or_else(|| name.chars().next().unwrap_or('X').to_string());
        let (resid, resname) = super::amber::residue_for_atom(i, topo);
        let mass = topo.masses.get(i).copied().unwrap_or(0.0);

        atoms.name_id.push(interner.intern_upper(name));
        atoms.resname_id.push(interner.intern_upper(resname.trim()));
        atoms.resid.push(resid);
        atoms.chain_id.push(interner.intern_upper("A"));
        atoms.element_id.push(interner.intern_upper(&element));
        atoms.mass.push(mass);
    }

    let system = System::with_atoms(atoms, interner, None);
    Ok(system)
}

pub fn system_from_molecule(molecule: &MoleculeData) -> StructureResult<System> {
    if molecule.atoms.is_empty() {
        return Err(StructureError::Invalid(
            "cannot project empty molecule into system".into(),
        ));
    }

    let mut interner = StringInterner::new();
    let mut atoms = AtomTable::default();
    let mut positions = Vec::with_capacity(molecule.atoms.len());

    for atom in &molecule.atoms {
        let name = atom.name.trim();
        let resname = atom.resname.trim();
        let chain = if atom.chain == ' ' {
            String::new()
        } else {
            atom.chain.to_string()
        };
        let element = atom.element.trim();
        let mass = if element.is_empty() {
            0.0
        } else {
            mass_for_element(element)
        };

        atoms.name_id.push(interner.intern_upper(name));
        atoms.resname_id.push(interner.intern_upper(resname));
        atoms.resid.push(atom.resid);
        atoms.chain_id.push(interner.intern_upper(&chain));
        atoms.element_id.push(interner.intern_upper(element));
        atoms.mass.push(mass);
        positions.push([atom.position.x, atom.position.y, atom.position.z, 1.0]);
    }

    let system = System::with_atoms(atoms, interner, Some(positions));
    system.validate_positions0().map_err(StructureError::from)?;
    Ok(system)
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;

    use tempfile::tempdir;

    use super::*;
    use crate::geometry::Vec3;
    use crate::model::{AtomRecord, AtomRecordKind};

    #[test]
    fn system_from_molecule_projects_atom_fields() {
        let molecule = MoleculeData {
            atoms: vec![AtomRecord {
                record_kind: AtomRecordKind::Atom,
                name: " CA ".into(),
                element: "C".into(),
                resname: " GLY ".into(),
                resid: 7,
                chain: 'A',
                segid: String::new(),
                charge: 0.0,
                position: Vec3::new(1.0, 2.0, 3.0),
                mol_id: 1,
                pdb_metadata: None,
            }],
            bonds: Vec::new(),
            box_vectors: None,
            ter_after: Vec::new(),
        };

        let system = system_from_molecule(&molecule).expect("project molecule");
        let position = system.positions0.as_ref().expect("positions0");
        assert_eq!(system.n_atoms(), 1);
        assert!((position[0][0] - 1.0).abs() < 1.0e-6);
        assert_eq!(system.interner.resolve(system.atoms.name_id[0]), Some("CA"));
        assert_eq!(
            system.interner.resolve(system.atoms.resname_id[0]),
            Some("GLY")
        );
        assert_eq!(system.interner.resolve(system.atoms.chain_id[0]), Some("A"));
    }

    #[test]
    fn read_pdb_system_uses_first_model_only() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("multi_model.pdb");
        let mut file = File::create(&path).expect("create pdb");
        file.write_all(
            b"MODEL        1\n\
ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00 20.00           N\n\
ENDMDL\n\
MODEL        2\n\
ATOM      1  N   ALA A   1       2.000   0.000   0.000  1.00 20.00           N\n\
ENDMDL\n",
        )
        .expect("write pdb");

        let system = read_pdb_system(&path, true).expect("read system");
        let position = system.positions0.as_ref().expect("positions0");
        assert_eq!(system.n_atoms(), 1);
        assert!((position[0][0] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn read_pdb_system_permissive_accepts_alphanumeric_resid() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("alpha_resid.pdb");
        let mut file = File::create(&path).expect("create pdb");
        file.write_all(
            b"ATOM      1  N   ALA AA000      11.104  13.207  14.099  1.00 20.00           N\n",
        )
        .expect("write pdb");

        let system = read_pdb_system(&path, false).expect("read permissive system");
        assert_eq!(system.n_atoms(), 1);
    }

    #[test]
    fn read_system_auto_retries_invalid_resid_pdb() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("alpha_resid.pdb");
        let mut file = File::create(&path).expect("create pdb");
        file.write_all(
            b"ATOM      1  N   ALA AA000      11.104  13.207  14.099  1.00 20.00           N\n",
        )
        .expect("write pdb");

        let system = read_system_auto(&path, None).expect("read auto system");
        assert_eq!(system.n_atoms(), 1);
    }

    #[test]
    fn read_system_auto_rejects_unsupported_format() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("alpha_resid.xyz");

        let err = read_system_auto(&path, None).expect_err("unsupported format");
        assert!(err.to_string().contains(SUPPORTED_SYSTEM_FORMATS_TEXT));
    }

    #[test]
    fn read_pdbqt_system_projects_positions() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("ligand.pdbqt");
        let mut file = File::create(&path).expect("create pdbqt");
        file.write_all(
            b"ATOM      1  C1  LIG A   1       1.500   2.500   3.500  0.00  0.00           C\n",
        )
        .expect("write pdbqt");

        let system = read_pdbqt_system(&path).expect("read pdbqt system");
        let position = system.positions0.as_ref().expect("positions0");
        assert_eq!(system.n_atoms(), 1);
        assert!((position[0][1] - 2.5).abs() < 1.0e-6);
    }

    #[test]
    fn read_system_auto_parses_prmtop_correctly() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("test.prmtop");
        let mut file = File::create(&path).expect("create prmtop");
        file.write_all(
            b"%FLAG TITLE\n\
              %FORMAT(20a4)\n\
              TEST PRMTOP\n\
              %FLAG POINTERS\n\
              %FORMAT(10I8)\n\
                     2       1       0       0       0       0       0       0       0       0\n\
              %FLAG ATOM_NAME\n\
              %FORMAT(20a4)\n\
              H1  H2  \n\
              %FLAG CHARGE\n\
              %FORMAT(5E16.8)\n\
                4.20000000E-01  4.20000000E-01\n\
              %FLAG MASS\n\
              %FORMAT(5E16.8)\n\
                1.00800000E+00  1.00800000E+00\n\
              %FLAG RESIDUE_LABEL\n\
              %FORMAT(20a4)\n\
              SOL \n\
              %FLAG RESIDUE_POINTER\n\
              %FORMAT(10I8)\n\
                     1\n\
              %FLAG ATOMIC_NUMBER\n\
              %FORMAT(10I8)\n\
                     1       1\n\
              %FLAG ATOM_TYPE_INDEX\n\
              %FORMAT(10I8)\n\
                     1       1\n"
        ).expect("write prmtop");

        let system = read_system_auto(&path, None).expect("read auto system prmtop");
        assert_eq!(system.n_atoms(), 2);
        assert_eq!(system.interner.resolve(system.atoms.name_id[0]), Some("H1"));
        assert_eq!(system.interner.resolve(system.atoms.resname_id[0]), Some("SOL"));
        assert_eq!(system.atoms.resid[0], 1);
        assert_eq!(system.atoms.mass[0], 1.008);
    }
}
