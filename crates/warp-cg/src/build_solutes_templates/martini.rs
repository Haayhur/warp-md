use crate::build_solutes::SoluteTemplateBead;

pub(super) const GLY_BEADS: &[SoluteTemplateBead] = &[SoluteTemplateBead {
    name: "BB",
    offset_angstrom: [0.0, 0.0, 0.0],
    charge_e: 0.0,
}];

pub(super) const ARG_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB",
        offset_angstrom: [2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [0.0, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [-2.5, 1.25, 0.0],
        charge_e: 1.0,
    },
];

pub(super) const AMINO_TWO_NEUTRAL_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB",
        offset_angstrom: [2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [-2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const AMINO_TWO_ANION_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB",
        offset_angstrom: [2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [-2.5, 0.0, 0.0],
        charge_e: -1.0,
    },
];

pub(super) const AMINO_THREE_NEUTRAL_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB",
        offset_angstrom: [2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [0.0, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [-2.5, 1.25, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const AMINO_FOUR_NEUTRAL_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB",
        offset_angstrom: [2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [0.0, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [-2.5, 1.25, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC3",
        offset_angstrom: [-2.5, -1.25, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const AMINO_FOUR_CATION_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB",
        offset_angstrom: [2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [0.0, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [-2.5, 1.25, 0.0],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SC3",
        offset_angstrom: [-2.5, -1.25, 0.0],
        charge_e: 0.5,
    },
];

pub(super) const TYR_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB",
        offset_angstrom: [2.5, 1.25, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [0.0, -1.25, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC3",
        offset_angstrom: [0.0, 1.25, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC4",
        offset_angstrom: [-2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const TRP_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "BB",
        offset_angstrom: [2.5, 1.25, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC1",
        offset_angstrom: [2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC2",
        offset_angstrom: [0.0, -1.25, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC3",
        offset_angstrom: [0.0, 1.25, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC4",
        offset_angstrom: [-2.5, 0.0, 0.0],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SC5",
        offset_angstrom: [-2.5, 1.25, 0.0],
        charge_e: 0.0,
    },
];

pub(super) const BENZ_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "R1",
        offset_angstrom: [1.053333, 0.54, 1.01],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "R2",
        offset_angstrom: [-1.046667, -1.06, 0.46],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "R3",
        offset_angstrom: [-0.006667, 0.52, -1.47],
        charge_e: 0.0,
    },
];

pub(super) const TOLU_BEADS: &[SoluteTemplateBead] = BENZ_BEADS;

pub(super) const ENAPH_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "C1",
        offset_angstrom: [-2.011667, 1.79, 1.791667],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "R2",
        offset_angstrom: [-0.781667, 1.97, -0.458333],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "R3",
        offset_angstrom: [1.288333, 1.4, -1.858333],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "R4",
        offset_angstrom: [0.348333, -0.4, -0.288333],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "R5",
        offset_angstrom: [-0.551667, -2.21, 1.271667],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "R6",
        offset_angstrom: [1.708333, -2.55, -0.458333],
        charge_e: 0.0,
    },
];

pub(super) const SUCR_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "A",
        offset_angstrom: [0.0125, 0.0, -1.5],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "B",
        offset_angstrom: [-2.9875, 1.6, -1.5],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "C",
        offset_angstrom: [-2.9875, -1.6, -1.5],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "VS",
        offset_angstrom: [-2.0875, 0.0, -1.5],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "A",
        offset_angstrom: [0.0125, 0.0, 1.5],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "B",
        offset_angstrom: [3.2125, 0.0, 3.1],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "C",
        offset_angstrom: [3.2125, 0.0, -0.1],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "VS",
        offset_angstrom: [1.6125, 0.0, 1.5],
        charge_e: 0.0,
    },
];

pub(super) const GLYL_BEADS: &[SoluteTemplateBead] = &[SoluteTemplateBead {
    name: "P01",
    offset_angstrom: [0.0, 0.0, 0.0],
    charge_e: 0.0,
}];

pub(super) const UREA_BEADS: &[SoluteTemplateBead] = &[SoluteTemplateBead {
    name: "P01",
    offset_angstrom: [0.0, 0.0, 0.0],
    charge_e: 0.0,
}];

pub(super) const TREH_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "S01",
        offset_angstrom: [1.518889, 1.028889, -3.302222],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "S02",
        offset_angstrom: [-0.601111, 0.118889, -2.552222],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "P01",
        offset_angstrom: [3.128889, 2.118889, -1.632222],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "S03",
        offset_angstrom: [0.768889, 1.548889, -1.012222],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "P02",
        offset_angstrom: [0.148889, -0.501111, -0.152222],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "S04",
        offset_angstrom: [-1.071111, -2.081111, 2.857778],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "P03",
        offset_angstrom: [-3.341111, -1.601111, 2.047778],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "S05",
        offset_angstrom: [-1.441111, -0.041111, 1.497778],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "S06",
        offset_angstrom: [0.888889, -0.591111, 2.247778],
        charge_e: 0.0,
    },
];

pub(super) const PUT_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "P01",
        offset_angstrom: [0.845, 0.265, -1.305],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "P02",
        offset_angstrom: [-0.845, -0.265, 1.305],
        charge_e: 0.0,
    },
];

pub(super) const SPER_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "P01",
        offset_angstrom: [-1.913333, 1.44, 1.24],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "C01",
        offset_angstrom: [0.556667, 0.43, -0.52],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "P02",
        offset_angstrom: [1.356667, -1.87, -0.72],
        charge_e: 0.0,
    },
];

pub(super) const MIM_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "SI1",
        offset_angstrom: [0.583333, 1.193333, -1.246667],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI2",
        offset_angstrom: [1.103333, -1.336667, 0.563333],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI3",
        offset_angstrom: [-1.686667, 0.143333, 0.683333],
        charge_e: 0.0,
    },
];

pub(super) const EIM_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "SI1",
        offset_angstrom: [-0.923333, -1.44, 0.34],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI2",
        offset_angstrom: [0.246667, 1.25, 1.51],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI3",
        offset_angstrom: [0.676667, 0.19, -1.85],
        charge_e: 0.0,
    },
];

pub(super) const BIM_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "SI1",
        offset_angstrom: [0.195, 0.8575, 0.36],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI2",
        offset_angstrom: [0.175, 0.3075, -2.75],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI3",
        offset_angstrom: [-2.015, -1.0225, -0.9],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SI4",
        offset_angstrom: [1.645, -0.1425, 3.29],
        charge_e: 0.0,
    },
];

pub(super) const OIM_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "SI1",
        offset_angstrom: [-0.024, -0.126, -1.246],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI2",
        offset_angstrom: [-0.384, 0.554, -4.306],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI3",
        offset_angstrom: [1.706, 2.164, -2.566],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SI4",
        offset_angstrom: [-0.374, -1.036, 1.974],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SI5",
        offset_angstrom: [-0.924, -1.556, 6.144],
        charge_e: 0.0,
    },
];

pub(super) const DIM_BEADS: &[SoluteTemplateBead] = &[
    SoluteTemplateBead {
        name: "SI1",
        offset_angstrom: [-2.796667, -0.673333, 1.111667],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI2",
        offset_angstrom: [-3.056667, -3.103333, -0.888333],
        charge_e: 0.5,
    },
    SoluteTemplateBead {
        name: "SI3",
        offset_angstrom: [-5.616667, -2.003333, 0.601667],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SI4",
        offset_angstrom: [0.453333, -0.563333, 0.521667],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SI5",
        offset_angstrom: [3.503333, 2.336667, 0.861667],
        charge_e: 0.0,
    },
    SoluteTemplateBead {
        name: "SI6",
        offset_angstrom: [7.513333, 4.006667, -2.208333],
        charge_e: 0.0,
    },
];

pub(super) const BF4_BEADS: &[SoluteTemplateBead] = &[SoluteTemplateBead {
    name: "BF4",
    offset_angstrom: [0.0, 0.0, 0.0],
    charge_e: -1.0,
}];
