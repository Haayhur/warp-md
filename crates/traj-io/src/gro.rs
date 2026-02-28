use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

use traj_core::elements::mass_for_element;
use traj_core::error::TrajResult;
use traj_core::pdb_gro::parse_gro_reader;
use traj_core::system::{AtomTable, System};

use crate::TopologyReader;

pub struct GroReader {
    path: PathBuf,
}

impl GroReader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

impl TopologyReader for GroReader {
    fn read_system(&mut self) -> TrajResult<System> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let parsed = parse_gro_reader(reader, true)?;

        let mut system = System::new();
        let mut positions = Vec::with_capacity(parsed.atoms.len());
        let mut atoms = AtomTable::default();

        for atom in parsed.atoms {
            let name_id = system.interner.intern_upper(atom.name.trim());
            let resname_id = system.interner.intern_upper(atom.resname.trim());
            let chain_id = system.interner.intern_upper("");
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
            positions.push([
                atom.position[0] * 10.0,
                atom.position[1] * 10.0,
                atom.position[2] * 10.0,
                1.0,
            ]);
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
    fn read_simple_gro() {
        let content = "Test\n   2\n    1ALA      N    1   0.000   0.000   0.000\n    1ALA     CA    2   0.100   0.000   0.000\n   1.0 1.0 1.0\n";
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.gro");
        let mut file = File::create(&path).unwrap();
        file.write_all(content.as_bytes()).unwrap();
        let mut reader = GroReader::new(&path);
        let system = reader.read_system().unwrap();
        assert_eq!(system.n_atoms(), 2);
        let pos = system.positions0.unwrap();
        assert!((pos[1][0] - 1.0).abs() < 1e-6);
    }
}
