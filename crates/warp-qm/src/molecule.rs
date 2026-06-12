use std::path::Path;

use serde_json::json;
use warp_structure::io::read_molecule;
use warp_structure::AtomRecord;

use crate::contract::{QmMoleculeSpec, QmRequest};

#[derive(Clone, Debug)]
pub struct QmMolecule {
    pub atoms: Vec<AtomRecord>,
    pub charge: i32,
    pub multiplicity: u32,
    pub units: String,
    pub source_path: Option<String>,
    pub source_format: Option<String>,
}

impl QmMolecule {
    pub fn from_request(request: &QmRequest) -> Result<Self, String> {
        Self::from_spec(&request.molecule)
    }

    pub fn from_spec(spec: &QmMoleculeSpec) -> Result<Self, String> {
        if spec.source.kind != "file" {
            return Err(format!(
                "molecule source kind '{}' is not implemented for QM adapter execution",
                spec.source.kind
            ));
        }
        let path = spec
            .source
            .path
            .as_deref()
            .ok_or_else(|| "file molecule source requires path".to_string())?;
        let molecule = read_molecule(
            Path::new(path),
            spec.source.format.as_deref(),
            true,
            true,
            None,
        )
        .map_err(|err| err.to_string())?;
        Ok(Self {
            atoms: molecule.atoms,
            charge: spec.charge,
            multiplicity: spec.multiplicity,
            units: spec.units.clone(),
            source_path: Some(path.into()),
            source_format: spec.source.format.clone(),
        })
    }

    pub fn atom_count(&self) -> usize {
        self.atoms.len()
    }

    pub fn to_orca_xyz_block(&self) -> String {
        let mut out = format!("* xyz {} {}\n", self.charge, self.multiplicity);
        for atom in &self.atoms {
            let p = atom.position;
            out.push_str(&format!(
                "{} {:.8} {:.8} {:.8}\n",
                normalized_element(atom),
                p.x,
                p.y,
                p.z
            ));
        }
        out.push('*');
        out
    }

    pub fn provenance(&self) -> serde_json::Value {
        json!({
            "source_path": self.source_path,
            "source_format": self.source_format,
            "units": self.units,
            "atom_count": self.atom_count()
        })
    }
}

fn normalized_element(atom: &AtomRecord) -> String {
    let element = atom.element.trim();
    if !element.is_empty() {
        element.to_string()
    } else {
        atom.name.trim().to_string()
    }
}
