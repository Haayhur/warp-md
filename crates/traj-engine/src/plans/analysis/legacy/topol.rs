use std::fs;
use std::path::Path;

use traj_core::error::{TrajError, TrajResult};

#[derive(Debug, Clone)]
pub struct LegacyMolType {
    pub name: String,
    pub num_atoms: usize,
    pub atom_names: Vec<String>,
    pub masses: Vec<f64>,
    pub charges: Vec<f64>,
    pub drude_pairs: Option<Vec<usize>>,
    pub plane_atoms: Option<[usize; 3]>,
    pub mol_charge: f64,
    pub mol_mass: f64,
}

#[derive(Debug, Clone)]
pub struct LegacyTopology {
    pub moltypes: Vec<LegacyMolType>,
    pub total_molecules: usize,
    pub mol_counts: Vec<usize>,
    pub pair_list: Option<Vec<(String, String)>>,
}

impl LegacyTopology {
    pub fn from_file(path: impl AsRef<Path>) -> TrajResult<Self> {
        let text = fs::read_to_string(&path).map_err(|err| {
            TrajError::Parse(format!(
                "failed to read legacy topol file {}: {err}",
                path.as_ref().display()
            ))
        })?;
        let lines: Vec<String> = text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| {
                !l.is_empty() && !l.starts_with('#') && !l.starts_with('!') && !l.starts_with("//")
            })
            .collect();
        let mut idx = 0usize;
        let next_line = |lines: &Vec<String>, idx: &mut usize| -> TrajResult<String> {
            if *idx >= lines.len() {
                return Err(TrajError::Parse("unexpected end of topol file".into()));
            }
            let line = lines[*idx].clone();
            *idx += 1;
            Ok(line)
        };

        let header = next_line(&lines, &mut idx)?;
        if header.to_ascii_lowercase() != "nmoltype" {
            return Err(TrajError::Parse(format!(
                "expected 'nmoltype', got '{header}'"
            )));
        }
        let nmoltype_line = next_line(&lines, &mut idx)?;
        let nmoltype: usize = nmoltype_line
            .split_whitespace()
            .next()
            .ok_or_else(|| TrajError::Parse("missing nmoltype value".into()))?
            .parse()
            .map_err(|_| TrajError::Parse("invalid nmoltype value".into()))?;
        let mut moltypes = Vec::with_capacity(nmoltype);
        for _ in 0..nmoltype {
            let name = next_line(&lines, &mut idx)?;
            let natom_line = next_line(&lines, &mut idx)?;
            let num_atoms: usize = natom_line
                .split_whitespace()
                .next()
                .ok_or_else(|| TrajError::Parse("missing atom count".into()))?
                .parse()
                .map_err(|_| TrajError::Parse("invalid atom count".into()))?;
            let mut atom_names = Vec::with_capacity(num_atoms);
            let mut masses = Vec::with_capacity(num_atoms);
            let mut charges = Vec::with_capacity(num_atoms);
            let mut drude_pairs: Option<Vec<usize>> = None;
            let mut mol_charge = 0.0f64;
            let mut mol_mass = 0.0f64;
            for atom_idx in 0..num_atoms {
                let line = next_line(&lines, &mut idx)?;
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 3 {
                    return Err(TrajError::Parse(format!("invalid atom line '{line}'")));
                }
                let atom_name = parts[0].to_string();
                let mass: f64 = parts[1]
                    .parse()
                    .map_err(|_| TrajError::Parse(format!("invalid mass in line '{line}'")))?;
                let charge: f64 = parts[2]
                    .parse()
                    .map_err(|_| TrajError::Parse(format!("invalid charge in line '{line}'")))?;
                atom_names.push(atom_name);
                masses.push(mass);
                charges.push(charge);
                mol_mass += mass;
                mol_charge += charge;
                if parts.len() >= 4 {
                    let idx_val: usize = parts[3].parse().map_err(|_| {
                        TrajError::Parse(format!("invalid drude index in line '{line}'"))
                    })?;
                    drude_pairs.get_or_insert_with(|| vec![0; num_atoms])[atom_idx] = idx_val;
                }
            }
            moltypes.push(LegacyMolType {
                name,
                num_atoms,
                atom_names,
                masses,
                charges,
                drude_pairs,
                plane_atoms: None,
                mol_charge,
                mol_mass,
            });
        }

        let system_line = next_line(&lines, &mut idx)?;
        if system_line.to_ascii_lowercase() != "system" {
            return Err(TrajError::Parse(format!(
                "expected 'system', got '{system_line}'"
            )));
        }
        let total_line = next_line(&lines, &mut idx)?;
        let total_molecules: usize = total_line
            .split_whitespace()
            .next()
            .ok_or_else(|| TrajError::Parse("missing total molecule count".into()))?
            .parse()
            .map_err(|_| TrajError::Parse("invalid total molecule count".into()))?;
        let mut mol_counts = Vec::with_capacity(nmoltype);
        for i in 0..nmoltype {
            let line = next_line(&lines, &mut idx)?;
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                return Err(TrajError::Parse(format!("invalid mol count line '{line}'")));
            }
            let count: usize = parts[1]
                .parse()
                .map_err(|_| TrajError::Parse(format!("invalid mol count in line '{line}'")))?;
            mol_counts.push(count);
            moltypes[i].name = parts[0].to_string();
        }

        let mut pair_list = None;
        while idx < lines.len() {
            let line = next_line(&lines, &mut idx)?;
            let lower = line.to_ascii_lowercase();
            if lower.starts_with("mol plane") {
                for i in 0..nmoltype {
                    let line = next_line(&lines, &mut idx)?;
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() < 4 {
                        return Err(TrajError::Parse(format!(
                            "invalid plane atom line '{line}'"
                        )));
                    }
                    let a: usize = parts[1].parse().map_err(|_| {
                        TrajError::Parse(format!("invalid plane index in '{line}'"))
                    })?;
                    let b: usize = parts[2].parse().map_err(|_| {
                        TrajError::Parse(format!("invalid plane index in '{line}'"))
                    })?;
                    let c: usize = parts[3].parse().map_err(|_| {
                        TrajError::Parse(format!("invalid plane index in '{line}'"))
                    })?;
                    moltypes[i].plane_atoms = Some([a, b, c]);
                }
            } else if lower.starts_with("gr pair list") {
                let count_line = next_line(&lines, &mut idx)?;
                let n_pairs: usize = count_line
                    .split_whitespace()
                    .next()
                    .ok_or_else(|| TrajError::Parse("missing gr pair count".into()))?
                    .parse()
                    .map_err(|_| TrajError::Parse("invalid gr pair count".into()))?;
                let mut pairs = Vec::with_capacity(n_pairs);
                for _ in 0..n_pairs {
                    let line = next_line(&lines, &mut idx)?;
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() < 2 {
                        return Err(TrajError::Parse(format!("invalid gr pair line '{line}'")));
                    }
                    pairs.push((parts[0].to_string(), parts[1].to_string()));
                }
                pair_list = Some(pairs);
            }
        }

        Ok(Self {
            moltypes,
            total_molecules,
            mol_counts,
            pair_list,
        })
    }

    pub fn atom_count(&self) -> usize {
        let mut total = 0usize;
        for (i, mt) in self.moltypes.iter().enumerate() {
            total += mt.num_atoms * self.mol_counts[i];
        }
        total
    }

    pub fn molecule_map(&self) -> (Vec<Vec<usize>>, Vec<usize>) {
        let mut groups = Vec::with_capacity(self.total_molecules);
        let mut types = Vec::with_capacity(self.total_molecules);
        let mut atom_idx = 0usize;
        for (type_idx, mt) in self.moltypes.iter().enumerate() {
            for _ in 0..self.mol_counts[type_idx] {
                let mut group = Vec::with_capacity(mt.num_atoms);
                for _ in 0..mt.num_atoms {
                    group.push(atom_idx);
                    atom_idx += 1;
                }
                groups.push(group);
                types.push(type_idx);
            }
        }
        (groups, types)
    }

    pub fn atom_properties(&self) -> (Vec<f64>, Vec<f64>, Vec<String>) {
        let mut masses = Vec::with_capacity(self.atom_count());
        let mut charges = Vec::with_capacity(self.atom_count());
        let mut types = Vec::with_capacity(self.atom_count());
        for (type_idx, mt) in self.moltypes.iter().enumerate() {
            for _ in 0..self.mol_counts[type_idx] {
                masses.extend(mt.masses.iter().copied());
                charges.extend(mt.charges.iter().copied());
                types.extend(mt.atom_names.iter().cloned());
            }
        }
        (masses, charges, types)
    }
}
