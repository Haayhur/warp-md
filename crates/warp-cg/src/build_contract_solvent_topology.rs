use super::*;

pub(super) fn standard_solvent_bonds(name: &str) -> Vec<MoleculeDefinitionBond> {
    match name {
        "DMSO" => vec![standard_solvent_bond(0, 1, 0.300, 8000.0)],
        "HEX" => vec![standard_solvent_bond(0, 1, 0.405, 5000.0)],
        "OCT" => vec![standard_solvent_bond(0, 1, 0.475, 3800.0)],
        "DOD" => vec![
            standard_solvent_bond(0, 1, 0.475, 3800.0),
            standard_solvent_bond(1, 2, 0.475, 3800.0),
        ],
        "HD" => vec![
            standard_solvent_bond(0, 1, 0.475, 3800.0),
            standard_solvent_bond(1, 2, 0.470, 3800.0),
            standard_solvent_bond(2, 3, 0.475, 3800.0),
        ],
        "HXE" => vec![standard_solvent_bond(0, 1, 0.395, 5000.0)],
        "OCE" => vec![standard_solvent_bond(0, 1, 0.470, 3800.0)],
        "DOE" => vec![
            standard_solvent_bond(0, 1, 0.470, 3800.0),
            standard_solvent_bond(1, 2, 0.475, 3800.0),
        ],
        "HXY" => vec![standard_solvent_bond(0, 1, 0.390, 5000.0)],
        "OCY" => vec![standard_solvent_bond(0, 1, 0.468, 3800.0)],
        "HXD14" => vec![standard_solvent_bond(0, 1, 0.385, 5000.0)],
        "OCD912" => vec![
            standard_solvent_bond(0, 1, 0.490, 3800.0),
            standard_solvent_bond(1, 2, 0.490, 3800.0),
            standard_solvent_bond(2, 3, 0.490, 3800.0),
        ],
        "TFEOL" => vec![standard_solvent_bond(0, 1, 0.300, 5000.0)],
        "BTO" => vec![standard_solvent_bond(0, 1, 0.310, 7000.0)],
        "HXO" => vec![standard_solvent_bond(0, 1, 0.385, 7000.0)],
        "HPO" => vec![standard_solvent_bond(0, 1, 0.460, 7000.0)],
        "OCO" => vec![
            standard_solvent_bond(0, 1, 0.390, 5000.0),
            standard_solvent_bond(1, 2, 0.350, 5000.0),
        ],
        "DISH" => vec![
            standard_solvent_bond(0, 1, 0.355, 5000.0),
            standard_solvent_bond(1, 2, 0.355, 5000.0),
        ],
        "DXE" => vec![standard_solvent_bond(0, 1, 0.330, 7000.0)],
        "TXE" => vec![
            standard_solvent_bond(0, 1, 0.330, 7000.0),
            standard_solvent_bond(1, 2, 0.330, 7000.0),
        ],
        "DISS" => vec![
            standard_solvent_bond(0, 1, 0.360, 5000.0),
            standard_solvent_bond(1, 2, 0.360, 5000.0),
        ],
        "ANN" => vec![standard_solvent_bond(0, 1, 0.350, 7000.0)],
        "HXN" => vec![standard_solvent_bond(0, 1, 0.380, 7000.0)],
        "HPN" => vec![standard_solvent_bond(0, 1, 0.450, 7000.0)],
        "BTA" => vec![standard_solvent_bond(0, 1, 0.310, 7000.0)],
        "HXA" => vec![standard_solvent_bond(0, 1, 0.385, 7000.0)],
        "HPA" => vec![standard_solvent_bond(0, 1, 0.455, 7000.0)],
        "ETA" => vec![standard_solvent_bond(0, 1, 0.310, 7000.0)],
        "IBA" => vec![standard_solvent_bond(0, 1, 0.375, 3500.0)],
        "TBA" => vec![standard_solvent_bond(0, 1, 0.376, 7000.0)],
        "NBA" => vec![standard_solvent_bond(0, 1, 0.405, 7000.0)],
        "BTI" => vec![standard_solvent_bond(0, 1, 0.310, 7000.0)],
        "PTI" => vec![standard_solvent_bond(0, 1, 0.340, 7000.0)],
        "HXI" => vec![standard_solvent_bond(0, 1, 0.385, 7000.0)],
        "HPI" => vec![standard_solvent_bond(0, 1, 0.460, 7000.0)],
        "OCI" => vec![
            standard_solvent_bond(0, 1, 0.390, 5000.0),
            standard_solvent_bond(1, 2, 0.350, 5000.0),
        ],
        _ => Vec::new(),
    }
}

fn standard_solvent_bond(
    left: usize,
    right: usize,
    length_nm: f32,
    force_kj_mol_nm2: f32,
) -> MoleculeDefinitionBond {
    MoleculeDefinitionBond {
        bead_indices: [left, right],
        length_nm: Some(length_nm),
        force_kj_mol_nm2: Some(force_kj_mol_nm2),
    }
}

pub(super) fn standard_solvent_angles(name: &str) -> Vec<MoleculeDefinitionAngle> {
    match name {
        "OCI" => vec![standard_solvent_angle(0, 1, 2, 150.0, 100.0)],
        _ => Vec::new(),
    }
}

fn standard_solvent_angle(
    left: usize,
    center: usize,
    right: usize,
    angle_degrees: f32,
    force_kj_mol_rad2: f32,
) -> MoleculeDefinitionAngle {
    MoleculeDefinitionAngle {
        bead_indices: [left, center, right],
        angle_degrees: Some(angle_degrees),
        force_kj_mol_rad2: Some(force_kj_mol_rad2),
    }
}
