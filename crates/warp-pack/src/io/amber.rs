use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind};

const AMBER_CHARGE_SCALE: f32 = 18.2223;

pub fn read_amber_inpcrd(path: &Path, topology: Option<&Path>) -> PackResult<MoleculeData> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    let _title = lines.next();
    let second = lines
        .next()
        .ok_or_else(|| PackError::Parse("amber inpcrd missing atom count".into()))??;
    let mut coords_tokens: Vec<String> = Vec::new();
    let mut n_atoms: Option<usize> = None;
    let mut second_parts = second.split_whitespace().collect::<Vec<_>>();
    if let Some(first) = second_parts.first() {
        if let Ok(n) = first.parse::<usize>() {
            n_atoms = Some(n);
            second_parts.remove(0);
        }
    }
    for part in second_parts {
        coords_tokens.push(part.to_string());
    }
    for line in lines {
        let line = line?;
        for part in line.split_whitespace() {
            coords_tokens.push(part.to_string());
        }
    }
    let mut floats = Vec::with_capacity(coords_tokens.len());
    for tok in coords_tokens {
        let val = tok
            .parse::<f32>()
            .map_err(|_| PackError::Parse(format!("invalid coordinate value: {tok}")))?;
        floats.push(val);
    }
    let n_atoms = if let Some(n) = n_atoms {
        if floats.len() < n * 3 {
            return Err(PackError::Parse(
                "amber inpcrd does not contain enough coordinates".into(),
            ));
        }
        n
    } else {
        if floats.len() % 3 != 0 {
            return Err(PackError::Parse(
                "amber inpcrd coordinate count is not divisible by 3".into(),
            ));
        }
        floats.len() / 3
    };

    let topology = if let Some(path) = topology {
        Some(parse_prmtop(path, n_atoms)?)
    } else {
        None
    };
    let mut atoms = Vec::with_capacity(n_atoms);
    for i in 0..n_atoms {
        let x = floats[i * 3];
        let y = floats[i * 3 + 1];
        let z = floats[i * 3 + 2];
        let (name, element, resname, resid, charge) = topology_data(i, &topology);
        atoms.push(AtomRecord {
            record_kind: AtomRecordKind::Atom,
            name,
            element,
            resname,
            resid,
            chain: 'A',
            segid: String::new(),
            charge,
            position: Vec3::new(x, y, z),
            mol_id: 1,
        });
    }
    if atoms.is_empty() {
        return Err(PackError::Parse("no atoms found in amber inpcrd".into()));
    }
    Ok(MoleculeData {
        atoms,
        bonds: Vec::new(),
        ter_after: Vec::new(),
    })
}

struct AmberTopology {
    atom_names: Vec<String>,
    residue_labels: Vec<String>,
    residue_pointers: Vec<usize>,
    atomic_numbers: Vec<i32>,
    charges: Vec<f32>,
}

fn parse_prmtop(path: &Path, n_atoms: usize) -> PackResult<AmberTopology> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut current_flag: Option<String> = None;
    let mut atom_names = Vec::new();
    let mut residue_labels = Vec::new();
    let mut residue_pointers = Vec::new();
    let mut atomic_numbers = Vec::new();
    let mut charges = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("%FLAG") {
            let name = line.split_whitespace().nth(1).unwrap_or("").to_string();
            current_flag = Some(name);
            continue;
        }
        if line.starts_with("%FORMAT") {
            continue;
        }
        let Some(flag) = current_flag.as_deref() else {
            continue;
        };
        let tokens = line.split_whitespace();
        match flag {
            "ATOM_NAME" => {
                for tok in tokens {
                    atom_names.push(tok.to_string());
                }
            }
            "RESIDUE_LABEL" => {
                for tok in tokens {
                    residue_labels.push(tok.to_string());
                }
            }
            "RESIDUE_POINTER" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        residue_pointers.push(val);
                    }
                }
            }
            "ATOMIC_NUMBER" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<i32>() {
                        atomic_numbers.push(val);
                    }
                }
            }
            "CHARGE" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        charges.push(val);
                    }
                }
            }
            _ => {}
        }
    }

    if atom_names.is_empty() {
        atom_names = vec!["X".into(); n_atoms];
    }
    if residue_labels.is_empty() {
        residue_labels = vec!["MOL".into()];
    }
    if residue_pointers.is_empty() {
        residue_pointers = vec![1];
    }
    if atomic_numbers.is_empty() {
        atomic_numbers = vec![0; n_atoms];
    }
    if charges.is_empty() {
        charges = vec![0.0; n_atoms];
    }

    if atom_names.len() < n_atoms {
        atom_names.resize(n_atoms, "X".into());
    }
    if atomic_numbers.len() < n_atoms {
        atomic_numbers.resize(n_atoms, 0);
    }
    if charges.len() < n_atoms {
        charges.resize(n_atoms, 0.0);
    }

    Ok(AmberTopology {
        atom_names,
        residue_labels,
        residue_pointers,
        atomic_numbers,
        charges,
    })
}

fn topology_data(
    atom_idx: usize,
    topo: &Option<AmberTopology>,
) -> (String, String, String, i32, f32) {
    let Some(topo) = topo else {
        return ("X".into(), "X".into(), "MOL".into(), 1, 0.0);
    };
    let name = topo
        .atom_names
        .get(atom_idx)
        .cloned()
        .unwrap_or_else(|| "X".into());
    let atomic_number = topo.atomic_numbers.get(atom_idx).copied().unwrap_or(0);
    let element = atomic_number_to_symbol(atomic_number)
        .unwrap_or_else(|| name.chars().next().unwrap_or('X').to_string());
    let charge_raw = topo.charges.get(atom_idx).copied().unwrap_or(0.0);
    let charge = charge_raw / AMBER_CHARGE_SCALE;
    let (resid, resname) = residue_for_atom(atom_idx, topo);
    (name, element, resname, resid, charge)
}

fn residue_for_atom(atom_idx: usize, topo: &AmberTopology) -> (i32, String) {
    let mut resid = 1usize;
    for (i, start) in topo.residue_pointers.iter().enumerate() {
        if *start <= atom_idx + 1 {
            resid = i + 1;
        } else {
            break;
        }
    }
    let resname = topo
        .residue_labels
        .get(resid - 1)
        .cloned()
        .unwrap_or_else(|| "MOL".into());
    (resid as i32, resname)
}

fn atomic_number_to_symbol(z: i32) -> Option<String> {
    const TABLE: [&str; 119] = [
        "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
        "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
        "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
        "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
        "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
        "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg",
        "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    ];
    if z > 0 && (z as usize) < TABLE.len() {
        Some(TABLE[z as usize].to_string())
    } else {
        None
    }
}
