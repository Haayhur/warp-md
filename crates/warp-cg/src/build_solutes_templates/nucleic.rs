use crate::build_solutes::{SoluteTemplateBead, SoluteTemplateBond};

pub(super) const DNA_DA_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB1",
        offset_angstrom: [3.763286, 2.446571, -1.334143],
        charge_e: -1.0,
    },
    SoluteTemplateBead {
        name: "BB2",
        offset_angstrom: [0.090286, 2.472571, -1.380143],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "BB3",
        offset_angstrom: [-0.081714, 2.518571, 0.504857],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [-0.662714, -0.461429, 0.439857],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [-2.681714, -1.596429, 0.689857],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC3",
        offset_angstrom: [-1.106714, -3.858429, 0.689857],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC4",
        offset_angstrom: [0.679286, -1.521429, 0.389857],
        charge_e: 0.0,
    },
];

pub(super) const DNA_DC_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB1",
        offset_angstrom: [2.7885, -2.466333, -1.192167],
        charge_e: -1.0,
    },
    SoluteTemplateBead {
        name: "BB2",
        offset_angstrom: [1.6795, 1.035667, -1.238167],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "BB3",
        offset_angstrom: [1.6675, 1.210667, 0.646833],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [-0.7175, 0.005667, 0.474833],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [-2.3535, 1.495667, 0.713833],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC3",
        offset_angstrom: [-3.0645, -1.281333, 0.594833],
        charge_e: 0.0,
    },
];

pub(super) const DNA_DG_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB1",
        offset_angstrom: [1.150286, -4.445286, -1.345],
        charge_e: -1.0,
    },
    SoluteTemplateBead {
        name: "BB2",
        offset_angstrom: [2.312286, -0.961286, -1.391],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "BB3",
        offset_angstrom: [2.405286, -0.809286, 0.494],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [-0.242714, 0.662714, 0.429],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [-0.700714, 3.580714, 0.748],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC3",
        offset_angstrom: [-3.296714, 2.182714, 0.682],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC4",
        offset_angstrom: [-1.627714, -0.210286, 0.383],
        charge_e: 0.0,
    },
];

pub(super) const DNA_DT_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB1",
        offset_angstrom: [-0.804167, 3.557833, -1.184],
        charge_e: -1.0,
    },
    SoluteTemplateBead {
        name: "BB2",
        offset_angstrom: [-1.966167, 0.073833, -1.23],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "BB3",
        offset_angstrom: [-2.059167, -0.078167, 0.655],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [0.555833, -0.516167, 0.483],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [1.033833, -2.668167, 0.723],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC3",
        offset_angstrom: [3.239833, -0.369167, 0.553],
        charge_e: 0.0,
    },
];

pub(super) const NUCLEOBASE_ADEN_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "N1",
        offset_angstrom: [-2.0, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N2",
        offset_angstrom: [0.68, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N3",
        offset_angstrom: [1.9, 0.8, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N4",
        offset_angstrom: [2.3, -2.35, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N5",
        offset_angstrom: [-0.87, -1.5, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N6",
        offset_angstrom: [0.0, -0.75, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const NUCLEOBASE_CYTO_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "N1",
        offset_angstrom: [-1.7, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N2",
        offset_angstrom: [0.85, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N3",
        offset_angstrom: [1.2, -1.6, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N4",
        offset_angstrom: [1.7, -2.95, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const NUCLEOBASE_GUAN_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "N1",
        offset_angstrom: [-2.7, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N2",
        offset_angstrom: [2.96, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N3",
        offset_angstrom: [1.4, -1.2, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N4",
        offset_angstrom: [2.13, -4.34, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N5",
        offset_angstrom: [0.13, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N6",
        offset_angstrom: [0.6, -1.0, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const NUCLEOBASE_THYM_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "N1",
        offset_angstrom: [-2.0, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N2",
        offset_angstrom: [0.86, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N3",
        offset_angstrom: [1.3, -1.7, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N4",
        offset_angstrom: [1.51, -3.57, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N5",
        offset_angstrom: [-0.08, -2.44, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const NUCLEOBASE_URAC_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "N1",
        offset_angstrom: [-2.0, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N2",
        offset_angstrom: [0.9, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N3",
        offset_angstrom: [1.3, -1.7, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N4",
        offset_angstrom: [1.54, -3.58, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "N5",
        offset_angstrom: [-0.3, -2.0, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const SIRAH_WT4_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "WN1",
        offset_angstrom: [-2.8675, -0.835, 0.0275],
        charge_e: -0.41,
    },
    SoluteTemplateBead {
        name: "WN2",
        offset_angstrom: [0.1525, 1.195, -2.4325],
        charge_e: -0.41,
    },
    SoluteTemplateBead {
        name: "WP1",
        offset_angstrom: [1.9725, -1.925, 0.1775],
        charge_e: 0.41,
    },
    SoluteTemplateBead {
        name: "WP2",
        offset_angstrom: [0.7425, 1.565, 2.2275],
        charge_e: 0.41,
    },
];

pub(super) const SIRAH_NAW_BEADS: &[SoluteTemplateBead] = &[SoluteTemplateBead {
    name: "NaW",
    offset_angstrom: [0.0, 0.0, 0.0],
    charge_e: 1.0,
}];

pub(super) const SIRAH_CLW_BEADS: &[SoluteTemplateBead] = &[SoluteTemplateBead {
    name: "ClW",
    offset_angstrom: [0.0, 0.0, 0.0],
    charge_e: -1.0,
}];

pub(crate) const SIRAH_WT4_BONDS: &[SoluteTemplateBond] = &[
    SoluteTemplateBond {
        bead_indices: [0, 1],
        length_nm: 0.45,
        force_kj_mol_nm2: 4184.0,
    },
    SoluteTemplateBond {
        bead_indices: [0, 2],
        length_nm: 0.45,
        force_kj_mol_nm2: 4184.0,
    },
    SoluteTemplateBond {
        bead_indices: [0, 3],
        length_nm: 0.45,
        force_kj_mol_nm2: 4184.0,
    },
    SoluteTemplateBond {
        bead_indices: [1, 2],
        length_nm: 0.45,
        force_kj_mol_nm2: 4184.0,
    },
    SoluteTemplateBond {
        bead_indices: [1, 3],
        length_nm: 0.45,
        force_kj_mol_nm2: 4184.0,
    },
    SoluteTemplateBond {
        bead_indices: [2, 3],
        length_nm: 0.45,
        force_kj_mol_nm2: 4184.0,
    },
];
