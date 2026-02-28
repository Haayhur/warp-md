use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use traj_core::elements::mass_for_element;
use traj_core::error::{TrajError, TrajResult};
use traj_core::pdb_gro::{parse_pdb_reader, PdbParseOptions};
use traj_core::system::{AtomTable, System};

use crate::TopologyReader;

pub struct PdbReader {
    path: PathBuf,
}

impl PdbReader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

pub struct PdbqtReader {
    path: PathBuf,
}

impl PdbqtReader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl TopologyReader for PdbReader {
    fn read_system(&mut self) -> TrajResult<System> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut system = System::new();
        let mut positions = Vec::new();
        let mut atoms = AtomTable::default();
        let options = PdbParseOptions {
            include_conect: false,
            include_ter: false,
            non_standard_conect: false,
            strict: true,
            only_first_model: true,
        };
        let parsed = parse_pdb_reader(reader, &options)?;
        for atom in parsed.atoms {
            let name_id = system.interner.intern_upper(atom.name.trim());
            let resname_id = system.interner.intern_upper(atom.resname.trim());
            let chain_id = system.interner.intern_upper(atom.chain.to_string().trim());
            let element_id = system.interner.intern_upper(&atom.element);
            let mass = if atom.element.is_empty() {
                0.0
            } else {
                mass_for_element(&atom.element)
            };
            atoms.name_id.push(name_id);
            atoms.resname_id.push(resname_id);
            atoms.resid.push(atom.resid);
            atoms.chain_id.push(chain_id);
            atoms.element_id.push(element_id);
            atoms.mass.push(mass);
            positions.push([atom.position[0], atom.position[1], atom.position[2], 1.0]);
        }

        if atoms.is_empty() {
            return Err(TrajError::Parse("no atoms found in PDB".into()));
        }

        system.atoms = atoms;
        system.positions0 = Some(positions);
        system.validate_positions0()?;
        Ok(system)
    }
}

impl TopologyReader for PdbqtReader {
    fn read_system(&mut self) -> TrajResult<System> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut system = System::new();
        let mut positions = Vec::new();
        let mut atoms = AtomTable::default();
        let options = PdbParseOptions {
            include_conect: false,
            include_ter: false,
            non_standard_conect: false,
            strict: false,
            only_first_model: true,
        };
        let parsed = parse_pdb_reader(reader, &options)?;
        for atom in parsed.atoms {
            let name_id = system.interner.intern_upper(atom.name.trim());
            let resname_id = system.interner.intern_upper(atom.resname.trim());
            let chain_id = system.interner.intern_upper(atom.chain.to_string().trim());
            let element_id = system.interner.intern_upper(&atom.element);
            let mass = if atom.element.is_empty() {
                0.0
            } else {
                mass_for_element(&atom.element)
            };
            atoms.name_id.push(name_id);
            atoms.resname_id.push(resname_id);
            atoms.resid.push(atom.resid);
            atoms.chain_id.push(chain_id);
            atoms.element_id.push(element_id);
            atoms.mass.push(mass);
            positions.push([atom.position[0], atom.position[1], atom.position[2], 1.0]);
        }

        if atoms.is_empty() {
            return Err(TrajError::Parse("no atoms found in PDBQT".into()));
        }

        system.atoms = atoms;
        system.positions0 = Some(positions);
        system.validate_positions0()?;
        Ok(system)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn read_simple_pdb() {
        let content =
            "ATOM      1  N   ALA A   1      11.104  13.207  14.099  1.00 20.00           N\n\
ATOM      2  CA  ALA A   1      12.560  13.207  14.099  1.00 20.00           C\n\
TER\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pdb");
        let mut file = File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        let mut reader = PdbReader::new(&path);
        let system = reader.read_system().unwrap();
        assert_eq!(system.n_atoms(), 2);
        let pos = system.positions0.unwrap();
        assert_eq!(pos.len(), 2);
        assert!((pos[0][0] - 11.104).abs() < 1e-3);
    }
}
