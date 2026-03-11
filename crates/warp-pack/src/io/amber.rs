use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::geom::Vec3;
use crate::io::MoleculeData;
use crate::pack::{AtomRecord, AtomRecordKind, PackOutput};

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
        bonds: topology.map(|value| value.bonds).unwrap_or_default(),
        ter_after: Vec::new(),
    })
}

#[derive(Clone, Debug)]
pub struct AmberTopology {
    pub atom_names: Vec<String>,
    pub residue_labels: Vec<String>,
    pub residue_pointers: Vec<usize>,
    pub atomic_numbers: Vec<i32>,
    pub masses: Vec<f32>,
    /// Partial charges in electron units.
    pub charges: Vec<f32>,
    pub atom_type_indices: Vec<usize>,
    pub amber_atom_types: Vec<String>,
    pub radii: Vec<f32>,
    pub screen: Vec<f32>,
    pub bonds: Vec<(usize, usize)>,
    pub bond_type_indices: Vec<usize>,
    pub bond_force_constants: Vec<f32>,
    pub bond_equil_values: Vec<f32>,
    pub angles: Vec<[usize; 3]>,
    pub angle_type_indices: Vec<usize>,
    pub angle_force_constants: Vec<f32>,
    pub angle_equil_values: Vec<f32>,
    pub dihedrals: Vec<[usize; 4]>,
    pub dihedral_type_indices: Vec<usize>,
    pub dihedral_force_constants: Vec<f32>,
    pub dihedral_periodicities: Vec<f32>,
    pub dihedral_phases: Vec<f32>,
    pub scee_scale_factors: Vec<f32>,
    pub scnb_scale_factors: Vec<f32>,
    pub solty: Vec<f32>,
    pub impropers: Vec<[usize; 4]>,
    pub improper_type_indices: Vec<usize>,
    pub excluded_atoms: Vec<Vec<usize>>,
    pub nonbonded_parm_index: Vec<usize>,
    pub lennard_jones_acoef: Vec<f32>,
    pub lennard_jones_bcoef: Vec<f32>,
    pub lennard_jones_14_acoef: Vec<f32>,
    pub lennard_jones_14_bcoef: Vec<f32>,
    pub hbond_acoef: Vec<f32>,
    pub hbond_bcoef: Vec<f32>,
    pub hbcut: Vec<f32>,
    pub tree_chain_classification: Vec<String>,
    pub join_array: Vec<usize>,
    pub irotat: Vec<usize>,
    pub solvent_pointers: Vec<usize>,
    pub atoms_per_molecule: Vec<usize>,
    pub box_dimensions: Vec<f32>,
    pub radius_set: Option<String>,
    pub ipol: usize,
}

pub(crate) fn parse_prmtop(path: &Path, n_atoms: usize) -> PackResult<AmberTopology> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut current_flag: Option<String> = None;
    let mut atom_names = Vec::new();
    let mut residue_labels = Vec::new();
    let mut residue_pointers = Vec::new();
    let mut atomic_numbers = Vec::new();
    let mut masses = Vec::new();
    let mut charges = Vec::new();
    let mut atom_type_indices = Vec::new();
    let mut amber_atom_types = Vec::new();
    let mut radii = Vec::new();
    let mut screen = Vec::new();
    let mut bonds = Vec::new();
    let mut bond_type_indices = Vec::new();
    let mut bond_force_constants = Vec::new();
    let mut bond_equil_values = Vec::new();
    let mut angles = Vec::new();
    let mut angle_type_indices = Vec::new();
    let mut angle_force_constants = Vec::new();
    let mut angle_equil_values = Vec::new();
    let mut dihedrals = Vec::new();
    let mut dihedral_type_indices = Vec::new();
    let mut dihedral_force_constants = Vec::new();
    let mut dihedral_periodicities = Vec::new();
    let mut dihedral_phases = Vec::new();
    let mut scee_scale_factors = Vec::new();
    let mut scnb_scale_factors = Vec::new();
    let mut solty = Vec::new();
    let mut impropers = Vec::new();
    let mut improper_type_indices = Vec::new();
    let mut number_excluded_atoms = Vec::new();
    let mut excluded_atoms_list = Vec::new();
    let mut nonbonded_parm_index = Vec::new();
    let mut lennard_jones_acoef = Vec::new();
    let mut lennard_jones_bcoef = Vec::new();
    let mut lennard_jones_14_acoef = Vec::new();
    let mut lennard_jones_14_bcoef = Vec::new();
    let mut hbond_acoef = Vec::new();
    let mut hbond_bcoef = Vec::new();
    let mut hbcut = Vec::new();
    let mut tree_chain_classification = Vec::new();
    let mut join_array = Vec::new();
    let mut irotat = Vec::new();
    let mut solvent_pointers = Vec::new();
    let mut atoms_per_molecule = Vec::new();
    let mut box_dimensions = Vec::new();
    let mut radius_set: Option<String> = None;
    let mut ipol = 0usize;
    let mut pointers = Vec::new();

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
            "MASS" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        masses.push(val);
                    }
                }
            }
            "CHARGE" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        charges.push(val / AMBER_CHARGE_SCALE);
                    }
                }
            }
            "POINTERS" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        pointers.push(val);
                    }
                }
            }
            "ATOM_TYPE_INDEX" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        atom_type_indices.push(val);
                    }
                }
            }
            "AMBER_ATOM_TYPE" => {
                for tok in tokens {
                    amber_atom_types.push(tok.to_string());
                }
            }
            "RADII" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        radii.push(val);
                    }
                }
            }
            "SCREEN" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        screen.push(val);
                    }
                }
            }
            "BOND_FORCE_CONSTANT" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        bond_force_constants.push(val);
                    }
                }
            }
            "BOND_EQUIL_VALUE" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        bond_equil_values.push(val);
                    }
                }
            }
            "BONDS_INC_HYDROGEN" | "BONDS_WITHOUT_HYDROGEN" => {
                let values = tokens
                    .filter_map(|tok| tok.parse::<usize>().ok())
                    .collect::<Vec<_>>();
                for chunk in values.chunks(3) {
                    if chunk.len() < 2 {
                        continue;
                    }
                    let a = chunk[0] / 3;
                    let b = chunk[1] / 3;
                    let (i, j) = if a <= b { (a, b) } else { (b, a) };
                    if i != j {
                        bonds.push((i, j));
                        bond_type_indices.push(chunk.get(2).copied().unwrap_or(1));
                    }
                }
            }
            "ANGLE_FORCE_CONSTANT" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        angle_force_constants.push(val);
                    }
                }
            }
            "ANGLE_EQUIL_VALUE" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        angle_equil_values.push(val);
                    }
                }
            }
            "ANGLES_INC_HYDROGEN" | "ANGLES_WITHOUT_HYDROGEN" => {
                let values = tokens
                    .filter_map(|tok| tok.parse::<usize>().ok())
                    .collect::<Vec<_>>();
                for chunk in values.chunks(4) {
                    if chunk.len() < 3 {
                        continue;
                    }
                    angles.push([chunk[0] / 3, chunk[1] / 3, chunk[2] / 3]);
                    angle_type_indices.push(chunk.get(3).copied().unwrap_or(1));
                }
            }
            "DIHEDRAL_FORCE_CONSTANT" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        dihedral_force_constants.push(val);
                    }
                }
            }
            "DIHEDRAL_PERIODICITY" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        dihedral_periodicities.push(val);
                    }
                }
            }
            "DIHEDRAL_PHASE" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        dihedral_phases.push(val);
                    }
                }
            }
            "SCEE_SCALE_FACTOR" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        scee_scale_factors.push(val);
                    }
                }
            }
            "SCNB_SCALE_FACTOR" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        scnb_scale_factors.push(val);
                    }
                }
            }
            "SOLTY" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        solty.push(val);
                    }
                }
            }
            "DIHEDRALS_INC_HYDROGEN" | "DIHEDRALS_WITHOUT_HYDROGEN" => {
                let values = tokens
                    .filter_map(|tok| tok.parse::<isize>().ok())
                    .collect::<Vec<_>>();
                for chunk in values.chunks(5) {
                    if chunk.len() < 4 {
                        continue;
                    }
                    let entry = [
                        chunk[0].unsigned_abs() / 3,
                        chunk[1].unsigned_abs() / 3,
                        chunk[2].unsigned_abs() / 3,
                        chunk[3].unsigned_abs() / 3,
                    ];
                    if chunk[2] < 0 || chunk[3] < 0 {
                        impropers.push(entry);
                        improper_type_indices.push(chunk.get(4).copied().unwrap_or(1) as usize);
                    } else {
                        dihedrals.push(entry);
                        dihedral_type_indices.push(chunk.get(4).copied().unwrap_or(1) as usize);
                    }
                }
            }
            "NUMBER_EXCLUDED_ATOMS" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        number_excluded_atoms.push(val);
                    }
                }
            }
            "EXCLUDED_ATOMS_LIST" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        excluded_atoms_list.push(val);
                    }
                }
            }
            "NONBONDED_PARM_INDEX" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        nonbonded_parm_index.push(val);
                    }
                }
            }
            "LENNARD_JONES_ACOEF" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        lennard_jones_acoef.push(val);
                    }
                }
            }
            "LENNARD_JONES_BCOEF" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        lennard_jones_bcoef.push(val);
                    }
                }
            }
            "LENNARD_JONES_14_ACOEF" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        lennard_jones_14_acoef.push(val);
                    }
                }
            }
            "LENNARD_JONES_14_BCOEF" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        lennard_jones_14_bcoef.push(val);
                    }
                }
            }
            "HBOND_ACOEF" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        hbond_acoef.push(val);
                    }
                }
            }
            "HBOND_BCOEF" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        hbond_bcoef.push(val);
                    }
                }
            }
            "HBCUT" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        hbcut.push(val);
                    }
                }
            }
            "TREE_CHAIN_CLASSIFICATION" => {
                for tok in tokens {
                    tree_chain_classification.push(tok.to_string());
                }
            }
            "JOIN_ARRAY" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        join_array.push(val);
                    }
                }
            }
            "IROTAT" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        irotat.push(val);
                    }
                }
            }
            "SOLVENT_POINTERS" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        solvent_pointers.push(val);
                    }
                }
            }
            "ATOMS_PER_MOLECULE" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        atoms_per_molecule.push(val);
                    }
                }
            }
            "BOX_DIMENSIONS" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<f32>() {
                        box_dimensions.push(val);
                    }
                }
            }
            "RADIUS_SET" => {
                let joined = tokens.collect::<Vec<_>>().join(" ");
                if !joined.is_empty() {
                    radius_set = Some(joined);
                }
            }
            "IPOL" => {
                for tok in tokens {
                    if let Ok(val) = tok.parse::<usize>() {
                        ipol = val;
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
    if masses.is_empty() {
        masses = vec![12.0; n_atoms];
    }
    if charges.is_empty() {
        charges = vec![0.0; n_atoms];
    }
    if atom_type_indices.is_empty() {
        atom_type_indices = (1..=n_atoms.max(atom_names.len())).collect();
    }
    if amber_atom_types.is_empty() {
        amber_atom_types = atom_names
            .iter()
            .map(|value| truncate_label(value, 4))
            .collect();
    }
    if radii.is_empty() {
        radii = vec![1.5; n_atoms];
    }
    if screen.is_empty() {
        screen = vec![0.8; n_atoms];
    }

    if atom_names.len() < n_atoms {
        atom_names.resize(n_atoms, "X".into());
    }
    if atomic_numbers.len() < n_atoms {
        atomic_numbers.resize(n_atoms, 0);
    }
    if masses.len() < n_atoms {
        masses.resize(n_atoms, 12.0);
    }
    if charges.len() < n_atoms {
        charges.resize(n_atoms, 0.0);
    }
    if atom_type_indices.len() < n_atoms {
        let start = atom_type_indices.len() + 1;
        atom_type_indices.extend(start..=n_atoms);
    }
    if amber_atom_types.len() < n_atoms {
        amber_atom_types.resize(n_atoms, "X".into());
    }
    if radii.len() < n_atoms {
        radii.resize(n_atoms, 1.5);
    }
    if screen.len() < n_atoms {
        screen.resize(n_atoms, 0.8);
    }
    let check_term_alignment = |name: &str, terms_len: usize, type_len: usize| -> PackResult<()> {
        if terms_len != type_len {
            return Err(PackError::Parse(format!(
                "amber prmtop {name} term/type length mismatch: {terms_len} terms, {type_len} type indices"
            )));
        }
        Ok(())
    };
    // Preserve section order exactly: per-term type indices are positional.
    check_term_alignment("bond", bonds.len(), bond_type_indices.len())?;
    check_term_alignment("angle", angles.len(), angle_type_indices.len())?;
    check_term_alignment("dihedral", dihedrals.len(), dihedral_type_indices.len())?;
    check_term_alignment("improper", impropers.len(), improper_type_indices.len())?;

    let mut excluded_atoms = Vec::new();
    let mut cursor = 0usize;
    for count in number_excluded_atoms {
        let end = (cursor + count).min(excluded_atoms_list.len());
        excluded_atoms.push(excluded_atoms_list[cursor..end].to_vec());
        cursor = end;
    }
    while excluded_atoms.len() < atom_names.len() {
        excluded_atoms.push(Vec::new());
    }
    if !pointers.is_empty() {
        let expect = |idx: usize| pointers.get(idx).copied().unwrap_or(0);
        let check = |name: &str, got: usize, want: usize| -> PackResult<()> {
            if want != 0 && got != want {
                return Err(PackError::Parse(format!(
                    "amber prmtop {name} length mismatch: got {got}, expected {want}"
                )));
            }
            Ok(())
        };
        check("NATOM", atom_names.len(), expect(0))?;
        check(
            "NTYPES",
            atom_type_indices.iter().copied().max().unwrap_or(0),
            expect(1),
        )?;
        check(
            "NBONH",
            bond_type_indices
                .iter()
                .enumerate()
                .filter(|(idx, _)| {
                    let (a, b) = bonds[*idx];
                    atomic_numbers.get(a).copied().unwrap_or_default() == 1
                        || atomic_numbers.get(b).copied().unwrap_or_default() == 1
                })
                .count(),
            expect(2),
        )?;
        check(
            "MBONA",
            bond_type_indices
                .iter()
                .enumerate()
                .filter(|(idx, _)| {
                    let (a, b) = bonds[*idx];
                    atomic_numbers.get(a).copied().unwrap_or_default() != 1
                        && atomic_numbers.get(b).copied().unwrap_or_default() != 1
                })
                .count(),
            expect(3),
        )?;
        check("NRES", residue_labels.len(), expect(11))?;
        check("NUMBND", bond_force_constants.len(), expect(15))?;
        check("NUMANG", angle_force_constants.len(), expect(16))?;
        check("NPTRA", dihedral_force_constants.len(), expect(17))?;
        check("NATYP", solty.len().max(1), expect(18))?;
        check("NPHB", hbond_acoef.len(), expect(19))?;
        check(
            "NEXT",
            excluded_atoms.iter().map(|v| v.len()).sum(),
            expect(10),
        )?;
    }

    Ok(AmberTopology {
        atom_names,
        residue_labels,
        residue_pointers,
        atomic_numbers,
        masses,
        charges,
        atom_type_indices,
        amber_atom_types,
        radii,
        screen,
        bonds,
        bond_type_indices,
        bond_force_constants,
        bond_equil_values,
        angles,
        angle_type_indices,
        angle_force_constants,
        angle_equil_values,
        dihedrals,
        dihedral_type_indices,
        dihedral_force_constants,
        dihedral_periodicities,
        dihedral_phases,
        scee_scale_factors,
        scnb_scale_factors,
        solty,
        impropers,
        improper_type_indices,
        excluded_atoms,
        nonbonded_parm_index,
        lennard_jones_acoef,
        lennard_jones_bcoef,
        lennard_jones_14_acoef,
        lennard_jones_14_bcoef,
        hbond_acoef,
        hbond_bcoef,
        hbcut,
        tree_chain_classification,
        join_array,
        irotat,
        solvent_pointers,
        atoms_per_molecule,
        box_dimensions,
        radius_set,
        ipol,
    })
}

pub fn read_prmtop_total_charge(path: &Path) -> PackResult<f32> {
    let topo = parse_prmtop(path, 0)?;
    Ok(topo.charges.iter().sum::<f32>())
}

pub fn read_prmtop_topology(path: &Path) -> PackResult<AmberTopology> {
    parse_prmtop(path, 0)
}

pub fn read_prmtop_atom_charges(path: &Path) -> PackResult<Vec<f32>> {
    let topo = parse_prmtop(path, 0)?;
    Ok(topo.charges)
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
    let charge = topo.charges.get(atom_idx).copied().unwrap_or(0.0);
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

fn bond_adjacency(atom_count: usize, bonds: &[(usize, usize)]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); atom_count];
    for &(a, b) in bonds {
        if let Some(items) = adjacency.get_mut(a) {
            items.push(b);
        }
        if let Some(items) = adjacency.get_mut(b) {
            items.push(a);
        }
    }
    for items in &mut adjacency {
        items.sort_unstable();
        items.dedup();
    }
    adjacency
}

fn inferred_angles(adjacency: &[Vec<usize>]) -> Vec<[usize; 3]> {
    let mut angles = Vec::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        for left_idx in 0..neighbors.len() {
            for right_idx in (left_idx + 1)..neighbors.len() {
                angles.push([neighbors[left_idx], center, neighbors[right_idx]]);
            }
        }
    }
    angles
}

fn inferred_dihedrals(adjacency: &[Vec<usize>], bonds: &[(usize, usize)]) -> Vec<[usize; 4]> {
    let mut items = Vec::new();
    for &(b, c) in bonds {
        for &a in adjacency.get(b).unwrap_or(&Vec::new()) {
            if a == c {
                continue;
            }
            for &d in adjacency.get(c).unwrap_or(&Vec::new()) {
                if d == b || d == a {
                    continue;
                }
                items.push([a, b, c, d]);
            }
        }
    }
    items.sort_unstable();
    items.dedup();
    items
}

fn inferred_impropers(adjacency: &[Vec<usize>]) -> Vec<[usize; 4]> {
    let mut items = Vec::new();
    for (center, neighbors) in adjacency.iter().enumerate() {
        if neighbors.len() < 3 {
            continue;
        }
        for left in 0..neighbors.len() {
            for mid in (left + 1)..neighbors.len() {
                for right in (mid + 1)..neighbors.len() {
                    items.push([neighbors[left], center, neighbors[mid], neighbors[right]]);
                }
            }
        }
    }
    items.sort_unstable();
    items.dedup();
    items
}

fn inferred_exclusions(adjacency: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut all = Vec::with_capacity(adjacency.len());
    for atom_idx in 0..adjacency.len() {
        let mut excluded = std::collections::BTreeSet::new();
        for bonded in adjacency.get(atom_idx).cloned().unwrap_or_default() {
            excluded.insert(bonded + 1);
            for angle in adjacency.get(bonded).cloned().unwrap_or_default() {
                if angle != atom_idx {
                    excluded.insert(angle + 1);
                }
            }
        }
        all.push(excluded.into_iter().collect());
    }
    all
}

fn type_indices_or_default(values: &[usize], len: usize) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }
    if values.len() == len {
        values
            .iter()
            .map(|value| if *value == 0 { 1 } else { *value })
            .collect()
    } else {
        vec![1; len]
    }
}

fn float_params_or_default(values: &[f32], len: usize, default: f32) -> Vec<f32> {
    if len == 0 {
        Vec::new()
    } else if values.len() == len {
        values.to_vec()
    } else {
        vec![default; len]
    }
}

fn write_int_section(file: &mut File, flag: &str, values: &[usize]) -> PackResult<()> {
    writeln!(file, "%FLAG {flag}")?;
    writeln!(file, "%FORMAT(10I8)")?;
    for chunk in values.chunks(10) {
        for value in chunk {
            write!(file, "{value:>8}")?;
        }
        writeln!(file)?;
    }
    Ok(())
}

fn write_isize_section(file: &mut File, flag: &str, values: &[isize]) -> PackResult<()> {
    writeln!(file, "%FLAG {flag}")?;
    writeln!(file, "%FORMAT(10I8)")?;
    for chunk in values.chunks(10) {
        for value in chunk {
            write!(file, "{value:>8}")?;
        }
        writeln!(file)?;
    }
    Ok(())
}

fn write_float_section(file: &mut File, flag: &str, values: &[f32]) -> PackResult<()> {
    writeln!(file, "%FLAG {flag}")?;
    writeln!(file, "%FORMAT(5E16.8)")?;
    for chunk in values.chunks(5) {
        for value in chunk {
            write!(file, "{value:16.8E}")?;
        }
        writeln!(file)?;
    }
    Ok(())
}

fn write_label_section(file: &mut File, flag: &str, values: &[String]) -> PackResult<()> {
    writeln!(file, "%FLAG {flag}")?;
    writeln!(file, "%FORMAT(20a4)")?;
    for chunk in values.chunks(20) {
        let line = chunk
            .iter()
            .map(|name| format!("{:<4}", truncate_label(name, 4)))
            .collect::<String>();
        writeln!(file, "{line}")?;
    }
    Ok(())
}

pub fn write_amber_inpcrd(out: &PackOutput, path: &str, scale: f32) -> PackResult<()> {
    let mut file = File::create(path)?;
    writeln!(file, "TITLE warp-pack")?;
    writeln!(file, "{:>6}", out.atoms.len())?;

    let mut values = Vec::with_capacity(out.atoms.len() * 3);
    for atom in &out.atoms {
        let pos = atom.position.scale(scale);
        values.push(pos.x);
        values.push(pos.y);
        values.push(pos.z);
    }

    for chunk in values.chunks(6) {
        for value in chunk {
            write!(file, "{value:12.7}")?;
        }
        writeln!(file)?;
    }
    Ok(())
}

pub fn write_minimal_prmtop(path: &str, topology: &AmberTopology) -> PackResult<()> {
    if topology.atom_names.is_empty() {
        return Err(PackError::Invalid(
            "prmtop writer requires at least one atom".into(),
        ));
    }
    if topology.residue_labels.is_empty() || topology.residue_pointers.is_empty() {
        return Err(PackError::Invalid(
            "prmtop writer requires residue labels and pointers".into(),
        ));
    }
    if topology.atom_names.len() != topology.atomic_numbers.len()
        || topology.atom_names.len() != topology.charges.len()
    {
        return Err(PackError::Invalid(
            "prmtop writer requires atom_names, atomic_numbers, and charges lengths to match"
                .into(),
        ));
    }
    let atom_count = topology.atom_names.len();
    let adjacency = bond_adjacency(atom_count, &topology.bonds);
    let masses = if topology.masses.len() == atom_count {
        topology.masses.clone()
    } else {
        vec![12.0; atom_count]
    };
    let atom_type_indices = if topology.atom_type_indices.len() == atom_count {
        topology.atom_type_indices.clone()
    } else {
        (1..=atom_count).collect()
    };
    let amber_atom_types = if topology.amber_atom_types.len() == atom_count {
        topology.amber_atom_types.clone()
    } else {
        topology
            .atom_names
            .iter()
            .map(|value| truncate_label(value, 4))
            .collect::<Vec<_>>()
    };
    let radii = if topology.radii.len() == atom_count {
        topology.radii.clone()
    } else {
        vec![1.5; atom_count]
    };
    let screen = if topology.screen.len() == atom_count {
        topology.screen.clone()
    } else {
        vec![0.8; atom_count]
    };
    let angles = if topology.angles.is_empty() {
        inferred_angles(&adjacency)
    } else {
        topology.angles.clone()
    };
    let dihedrals = if topology.dihedrals.is_empty() {
        inferred_dihedrals(&adjacency, &topology.bonds)
    } else {
        topology.dihedrals.clone()
    };
    let impropers = if topology.impropers.is_empty() {
        inferred_impropers(&adjacency)
    } else {
        topology.impropers.clone()
    };
    let excluded_atoms = if topology.excluded_atoms.len() == atom_count {
        topology.excluded_atoms.clone()
    } else {
        inferred_exclusions(&adjacency)
    };
    let bond_type_indices =
        type_indices_or_default(&topology.bond_type_indices, topology.bonds.len());
    let angle_type_indices = type_indices_or_default(&topology.angle_type_indices, angles.len());
    let dihedral_type_indices =
        type_indices_or_default(&topology.dihedral_type_indices, dihedrals.len());
    let improper_type_indices =
        type_indices_or_default(&topology.improper_type_indices, impropers.len());
    let bond_param_count = bond_type_indices.iter().copied().max().unwrap_or(
        topology
            .bond_force_constants
            .len()
            .max(topology.bond_equil_values.len()),
    );
    let angle_param_count = angle_type_indices.iter().copied().max().unwrap_or(
        topology
            .angle_force_constants
            .len()
            .max(topology.angle_equil_values.len()),
    );
    let dihedral_param_count = dihedral_type_indices
        .iter()
        .chain(improper_type_indices.iter())
        .copied()
        .max()
        .unwrap_or(
            topology
                .dihedral_force_constants
                .len()
                .max(topology.dihedral_periodicities.len())
                .max(topology.dihedral_phases.len())
                .max(topology.scee_scale_factors.len())
                .max(topology.scnb_scale_factors.len()),
        );
    let bond_force_constants =
        float_params_or_default(&topology.bond_force_constants, bond_param_count, 300.0);
    let bond_equil_values =
        float_params_or_default(&topology.bond_equil_values, bond_param_count, 1.5);
    let angle_force_constants =
        float_params_or_default(&topology.angle_force_constants, angle_param_count, 40.0);
    let angle_equil_values =
        float_params_or_default(&topology.angle_equil_values, angle_param_count, 109.5);
    let dihedral_force_constants = float_params_or_default(
        &topology.dihedral_force_constants,
        dihedral_param_count,
        0.5,
    );
    let dihedral_periodicities =
        float_params_or_default(&topology.dihedral_periodicities, dihedral_param_count, 3.0);
    let dihedral_phases =
        float_params_or_default(&topology.dihedral_phases, dihedral_param_count, 0.0);
    let scee_scale_factors =
        float_params_or_default(&topology.scee_scale_factors, dihedral_param_count, 1.2);
    let scnb_scale_factors =
        float_params_or_default(&topology.scnb_scale_factors, dihedral_param_count, 2.0);
    let n_types = atom_type_indices.iter().copied().max().unwrap_or(1);
    let solty = if topology.solty.is_empty() {
        vec![0.0; n_types.max(1)]
    } else {
        topology.solty.clone()
    };
    let nonbonded_parm_index = if topology.nonbonded_parm_index.is_empty() {
        (1..=(n_types * n_types)).collect::<Vec<_>>()
    } else {
        topology.nonbonded_parm_index.clone()
    };
    let lennard_jones_acoef = if topology.lennard_jones_acoef.is_empty() {
        vec![1.0; nonbonded_parm_index.len()]
    } else {
        topology.lennard_jones_acoef.clone()
    };
    let lennard_jones_bcoef = if topology.lennard_jones_bcoef.is_empty() {
        vec![1.0; nonbonded_parm_index.len()]
    } else {
        topology.lennard_jones_bcoef.clone()
    };
    let lennard_jones_14_acoef = if topology.lennard_jones_14_acoef.is_empty() {
        lennard_jones_acoef.clone()
    } else {
        topology.lennard_jones_14_acoef.clone()
    };
    let lennard_jones_14_bcoef = if topology.lennard_jones_14_bcoef.is_empty() {
        lennard_jones_bcoef.clone()
    } else {
        topology.lennard_jones_14_bcoef.clone()
    };
    let hbond_acoef = topology.hbond_acoef.clone();
    let hbond_bcoef = topology.hbond_bcoef.clone();
    let hbcut = topology.hbcut.clone();
    let tree_chain_classification = if topology.tree_chain_classification.len() == atom_count {
        topology.tree_chain_classification.clone()
    } else {
        vec!["M".into(); atom_count]
    };
    let join_array = if topology.join_array.len() == atom_count {
        topology.join_array.clone()
    } else {
        vec![0; atom_count]
    };
    let irotat = if topology.irotat.len() == atom_count {
        topology.irotat.clone()
    } else {
        vec![0; atom_count]
    };
    let atoms_per_molecule = if topology.atoms_per_molecule.is_empty() {
        vec![atom_count]
    } else {
        topology.atoms_per_molecule.clone()
    };
    let solvent_pointers = if topology.solvent_pointers.is_empty() {
        vec![
            topology.residue_labels.len(),
            atoms_per_molecule.len(),
            atoms_per_molecule.len() + 1,
        ]
    } else {
        topology.solvent_pointers.clone()
    };
    let box_dimensions = topology.box_dimensions.clone();
    let radius_set = topology
        .radius_set
        .clone()
        .unwrap_or_else(|| "modified Bondi radii".to_string());
    let ipol = topology.ipol;

    let mut file = File::create(path)?;
    writeln!(file, "%VERSION  VERSION_STAMP = V0001.000")?;
    let ifbox = usize::from(!box_dimensions.is_empty());

    let mut bonds_inc_h = Vec::new();
    let mut bonds_without_h = Vec::new();
    let mut bond_count_inc_h = 0usize;
    let mut bond_count_without_h = 0usize;
    for (idx, &(a, b)) in topology.bonds.iter().enumerate() {
        let triple = [a * 3, b * 3, *bond_type_indices.get(idx).unwrap_or(&1)];
        let has_hydrogen = topology.atomic_numbers.get(a).copied().unwrap_or_default() == 1
            || topology.atomic_numbers.get(b).copied().unwrap_or_default() == 1;
        if has_hydrogen {
            bonds_inc_h.extend_from_slice(&triple);
            bond_count_inc_h += 1;
        } else {
            bonds_without_h.extend_from_slice(&triple);
            bond_count_without_h += 1;
        }
    }

    let mut angles_inc_h = Vec::new();
    let mut angles_without_h = Vec::new();
    let mut angle_count_inc_h = 0usize;
    let mut angle_count_without_h = 0usize;
    for (idx, angle) in angles.iter().enumerate() {
        let values = [
            angle[0] * 3,
            angle[1] * 3,
            angle[2] * 3,
            *angle_type_indices.get(idx).unwrap_or(&1),
        ];
        let has_hydrogen = angle.iter().any(|idx| {
            topology
                .atomic_numbers
                .get(*idx)
                .copied()
                .unwrap_or_default()
                == 1
        });
        if has_hydrogen {
            angles_inc_h.extend_from_slice(&values);
            angle_count_inc_h += 1;
        } else {
            angles_without_h.extend_from_slice(&values);
            angle_count_without_h += 1;
        }
    }

    let mut dihedrals_inc_h = Vec::new();
    let mut dihedrals_without_h = Vec::new();
    let mut dihedral_count_inc_h = 0usize;
    let mut dihedral_count_without_h = 0usize;
    for (idx, entry) in dihedrals.iter().enumerate() {
        let values = [
            (entry[0] * 3) as isize,
            (entry[1] * 3) as isize,
            (entry[2] * 3) as isize,
            (entry[3] * 3) as isize,
            *dihedral_type_indices.get(idx).unwrap_or(&1) as isize,
        ];
        let has_hydrogen = entry.iter().any(|idx| {
            topology
                .atomic_numbers
                .get(*idx)
                .copied()
                .unwrap_or_default()
                == 1
        });
        if has_hydrogen {
            dihedrals_inc_h.extend_from_slice(&values);
            dihedral_count_inc_h += 1;
        } else {
            dihedrals_without_h.extend_from_slice(&values);
            dihedral_count_without_h += 1;
        }
    }
    for (idx, entry) in impropers.iter().enumerate() {
        let values = [
            (entry[0] * 3) as isize,
            (entry[1] * 3) as isize,
            -((entry[2] * 3) as isize),
            -((entry[3] * 3) as isize),
            *improper_type_indices.get(idx).unwrap_or(&1) as isize,
        ];
        let has_hydrogen = entry.iter().any(|idx| {
            topology
                .atomic_numbers
                .get(*idx)
                .copied()
                .unwrap_or_default()
                == 1
        });
        if has_hydrogen {
            dihedrals_inc_h.extend_from_slice(&values);
            dihedral_count_inc_h += 1;
        } else {
            dihedrals_without_h.extend_from_slice(&values);
            dihedral_count_without_h += 1;
        }
    }
    let number_excluded_atoms = excluded_atoms
        .iter()
        .map(|items| items.len())
        .collect::<Vec<_>>();
    let flat_exclusions = excluded_atoms.iter().flatten().copied().collect::<Vec<_>>();
    let nmxrs = topology
        .residue_pointers
        .iter()
        .enumerate()
        .map(|(idx, start)| {
            let end = topology
                .residue_pointers
                .get(idx + 1)
                .copied()
                .unwrap_or(atom_count + 1);
            end.saturating_sub(*start)
        })
        .max()
        .unwrap_or(atom_count);
    let pointers = vec![
        atom_count,
        n_types,
        bond_count_inc_h,
        bond_count_without_h,
        angle_count_inc_h,
        angle_count_without_h,
        dihedral_count_inc_h,
        dihedral_count_without_h,
        0,
        0,
        flat_exclusions.len(),
        topology.residue_labels.len(),
        bond_count_without_h,
        angle_count_without_h,
        dihedral_count_without_h,
        bond_force_constants.len(),
        angle_force_constants.len(),
        dihedral_force_constants.len(),
        solty.len(),
        hbond_acoef.len(),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ifbox,
        nmxrs,
        0,
        0,
    ];

    write_int_section(&mut file, "POINTERS", &pointers)?;
    write_label_section(&mut file, "ATOM_NAME", &topology.atom_names)?;
    write_label_section(&mut file, "RESIDUE_LABEL", &topology.residue_labels)?;
    write_int_section(&mut file, "RESIDUE_POINTER", &topology.residue_pointers)?;
    write_int_section(
        &mut file,
        "ATOMIC_NUMBER",
        &topology
            .atomic_numbers
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>(),
    )?;
    write_float_section(&mut file, "MASS", &masses)?;
    write_float_section(
        &mut file,
        "CHARGE",
        &topology
            .charges
            .iter()
            .map(|value| *value * AMBER_CHARGE_SCALE)
            .collect::<Vec<_>>(),
    )?;
    write_int_section(&mut file, "ATOM_TYPE_INDEX", &atom_type_indices)?;
    write_label_section(&mut file, "AMBER_ATOM_TYPE", &amber_atom_types)?;
    write_float_section(&mut file, "BOND_FORCE_CONSTANT", &bond_force_constants)?;
    write_float_section(&mut file, "BOND_EQUIL_VALUE", &bond_equil_values)?;
    write_int_section(&mut file, "BONDS_INC_HYDROGEN", &bonds_inc_h)?;
    write_int_section(&mut file, "BONDS_WITHOUT_HYDROGEN", &bonds_without_h)?;
    write_float_section(&mut file, "ANGLE_FORCE_CONSTANT", &angle_force_constants)?;
    write_float_section(&mut file, "ANGLE_EQUIL_VALUE", &angle_equil_values)?;
    write_int_section(&mut file, "ANGLES_INC_HYDROGEN", &angles_inc_h)?;
    write_int_section(&mut file, "ANGLES_WITHOUT_HYDROGEN", &angles_without_h)?;
    write_float_section(
        &mut file,
        "DIHEDRAL_FORCE_CONSTANT",
        &dihedral_force_constants,
    )?;
    write_float_section(&mut file, "DIHEDRAL_PERIODICITY", &dihedral_periodicities)?;
    write_float_section(&mut file, "DIHEDRAL_PHASE", &dihedral_phases)?;
    write_float_section(&mut file, "SCEE_SCALE_FACTOR", &scee_scale_factors)?;
    write_float_section(&mut file, "SCNB_SCALE_FACTOR", &scnb_scale_factors)?;
    write_float_section(&mut file, "SOLTY", &solty)?;
    write_isize_section(&mut file, "DIHEDRALS_INC_HYDROGEN", &dihedrals_inc_h)?;
    write_isize_section(
        &mut file,
        "DIHEDRALS_WITHOUT_HYDROGEN",
        &dihedrals_without_h,
    )?;
    write_int_section(&mut file, "NUMBER_EXCLUDED_ATOMS", &number_excluded_atoms)?;
    write_int_section(&mut file, "EXCLUDED_ATOMS_LIST", &flat_exclusions)?;
    write_int_section(&mut file, "NONBONDED_PARM_INDEX", &nonbonded_parm_index)?;
    write_float_section(&mut file, "LENNARD_JONES_ACOEF", &lennard_jones_acoef)?;
    write_float_section(&mut file, "LENNARD_JONES_BCOEF", &lennard_jones_bcoef)?;
    write_float_section(&mut file, "LENNARD_JONES_14_ACOEF", &lennard_jones_14_acoef)?;
    write_float_section(&mut file, "LENNARD_JONES_14_BCOEF", &lennard_jones_14_bcoef)?;
    write_float_section(&mut file, "HBOND_ACOEF", &hbond_acoef)?;
    write_float_section(&mut file, "HBOND_BCOEF", &hbond_bcoef)?;
    write_float_section(&mut file, "HBCUT", &hbcut)?;
    write_label_section(
        &mut file,
        "TREE_CHAIN_CLASSIFICATION",
        &tree_chain_classification,
    )?;
    write_int_section(&mut file, "JOIN_ARRAY", &join_array)?;
    write_int_section(&mut file, "IROTAT", &irotat)?;
    if ifbox == 1 {
        write_int_section(&mut file, "SOLVENT_POINTERS", &solvent_pointers)?;
        write_int_section(&mut file, "ATOMS_PER_MOLECULE", &atoms_per_molecule)?;
        write_float_section(&mut file, "BOX_DIMENSIONS", &box_dimensions)?;
    }
    writeln!(file, "%FLAG RADIUS_SET")?;
    writeln!(file, "%FORMAT(1a80)")?;
    writeln!(file, "{radius_set:<80}")?;
    write_float_section(&mut file, "RADII", &radii)?;
    write_float_section(&mut file, "SCREEN", &screen)?;
    write_int_section(&mut file, "IPOL", &[ipol])?;

    Ok(())
}

fn truncate_label(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
}
