use crate::build_solutes::SoluteTemplate;

mod martini;
mod nucleic;

use martini::*;
use nucleic::*;

pub(crate) use nucleic::SIRAH_WT4_BONDS;

pub(crate) static TEMPLATES: &[SoluteTemplate] = &[
    SoluteTemplate {
        name: "ARG",
        source: "martini3_amino_acid_library.ARG",
        beads: ARG_BEADS,
    },
    SoluteTemplate {
        name: "LYS",
        source: "martini3_amino_acid_library.LYS",
        beads: ARG_BEADS,
    },
    SoluteTemplate {
        name: "GLY",
        source: "martini3_amino_acid_library.GLY",
        beads: GLY_BEADS,
    },
    SoluteTemplate {
        name: "ALA",
        source: "martini3_amino_acid_library.ALA",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "CYS",
        source: "martini3_amino_acid_library.CYS",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "VAL",
        source: "martini3_amino_acid_library.VAL",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "LEU",
        source: "martini3_amino_acid_library.LEU",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "ILE",
        source: "martini3_amino_acid_library.ILE",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "MET",
        source: "martini3_amino_acid_library.MET",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "PRO",
        source: "martini3_amino_acid_library.PRO",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "HYP",
        source: "martini3_amino_acid_library.HYP",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "ASN",
        source: "martini3_amino_acid_library.ASN",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "GLN",
        source: "martini3_amino_acid_library.GLN",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "THR",
        source: "martini3_amino_acid_library.THR",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "SER",
        source: "martini3_amino_acid_library.SER",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "ASPP",
        source: "martini3_amino_acid_library.ASPP",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "ASH",
        source: "martini3_amino_acid_library.ASH",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "GLUP",
        source: "martini3_amino_acid_library.GLUP",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "GLH",
        source: "martini3_amino_acid_library.GLH",
        beads: AMINO_TWO_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "ASP",
        source: "martini3_amino_acid_library.ASP",
        beads: AMINO_TWO_ANION_BEADS,
    },
    SoluteTemplate {
        name: "GLU",
        source: "martini3_amino_acid_library.GLU",
        beads: AMINO_TWO_ANION_BEADS,
    },
    SoluteTemplate {
        name: "LSN",
        source: "martini3_amino_acid_library.LSN",
        beads: AMINO_THREE_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "LYN",
        source: "martini3_amino_acid_library.LYN",
        beads: AMINO_THREE_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "PHE",
        source: "martini3_amino_acid_library.PHE",
        beads: AMINO_FOUR_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "HIS",
        source: "martini3_amino_acid_library.HIS",
        beads: AMINO_FOUR_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "HIE",
        source: "martini3_amino_acid_library.HIE",
        beads: AMINO_FOUR_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "HSE",
        source: "martini3_amino_acid_library.HSE",
        beads: AMINO_FOUR_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "HSD",
        source: "martini3_amino_acid_library.HSD",
        beads: AMINO_FOUR_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "HID",
        source: "martini3_amino_acid_library.HID",
        beads: AMINO_FOUR_NEUTRAL_BEADS,
    },
    SoluteTemplate {
        name: "HSP",
        source: "martini3_amino_acid_library.HSP",
        beads: AMINO_FOUR_CATION_BEADS,
    },
    SoluteTemplate {
        name: "HIP",
        source: "martini3_amino_acid_library.HIP",
        beads: AMINO_FOUR_CATION_BEADS,
    },
    SoluteTemplate {
        name: "TYR",
        source: "martini3_amino_acid_library.TYR",
        beads: TYR_BEADS,
    },
    SoluteTemplate {
        name: "TRP",
        source: "martini3_amino_acid_library.TRP",
        beads: TRP_BEADS,
    },
    SoluteTemplate {
        name: "BENZ",
        source: "martini3_small_molecule_library.BENZ",
        beads: BENZ_BEADS,
    },
    SoluteTemplate {
        name: "TOLU",
        source: "martini3_small_molecule_library.TOLU",
        beads: TOLU_BEADS,
    },
    SoluteTemplate {
        name: "ENAPH",
        source: "martini3_small_molecule_library.ENAPH",
        beads: ENAPH_BEADS,
    },
    SoluteTemplate {
        name: "SUCR",
        source: "martini3_sugar_library.SUCR",
        beads: SUCR_BEADS,
    },
    SoluteTemplate {
        name: "SUCROSE",
        source: "martini3_sugar_library.SUCR",
        beads: SUCR_BEADS,
    },
    SoluteTemplate {
        name: "GLYL",
        source: "martini2_bacterial_membrane_osmolyte_library.GLYL",
        beads: GLYL_BEADS,
    },
    SoluteTemplate {
        name: "PUT",
        source: "martini2_bacterial_membrane_osmolyte_library.PUT",
        beads: PUT_BEADS,
    },
    SoluteTemplate {
        name: "SPER",
        source: "martini2_bacterial_membrane_osmolyte_library.SPER",
        beads: SPER_BEADS,
    },
    SoluteTemplate {
        name: "UREA",
        source: "martini2_bacterial_membrane_osmolyte_library.UREA",
        beads: UREA_BEADS,
    },
    SoluteTemplate {
        name: "TREH",
        source: "martini2_bacterial_membrane_osmolyte_library.TREH",
        beads: TREH_BEADS,
    },
    SoluteTemplate {
        name: "C1",
        source: "ionic_liquid_tutorial_library.MIM",
        beads: MIM_BEADS,
    },
    SoluteTemplate {
        name: "MIM",
        source: "ionic_liquid_tutorial_library.MIM",
        beads: MIM_BEADS,
    },
    SoluteTemplate {
        name: "C2",
        source: "ionic_liquid_tutorial_library.EIM",
        beads: EIM_BEADS,
    },
    SoluteTemplate {
        name: "EIM",
        source: "ionic_liquid_tutorial_library.EIM",
        beads: EIM_BEADS,
    },
    SoluteTemplate {
        name: "C4",
        source: "ionic_liquid_tutorial_library.BIM",
        beads: BIM_BEADS,
    },
    SoluteTemplate {
        name: "BIM",
        source: "ionic_liquid_tutorial_library.BIM",
        beads: BIM_BEADS,
    },
    SoluteTemplate {
        name: "C8",
        source: "ionic_liquid_tutorial_library.OIM",
        beads: OIM_BEADS,
    },
    SoluteTemplate {
        name: "OIM",
        source: "ionic_liquid_tutorial_library.OIM",
        beads: OIM_BEADS,
    },
    SoluteTemplate {
        name: "C12",
        source: "ionic_liquid_tutorial_library.DIM",
        beads: DIM_BEADS,
    },
    SoluteTemplate {
        name: "DIM",
        source: "ionic_liquid_tutorial_library.DIM",
        beads: DIM_BEADS,
    },
    SoluteTemplate {
        name: "BF4",
        source: "ionic_liquid_tutorial_library.BF4",
        beads: BF4_BEADS,
    },
    SoluteTemplate {
        name: "DA",
        source: "martini2_dna_tutorial_library.DA",
        beads: DNA_DA_BEADS,
    },
    SoluteTemplate {
        name: "DC",
        source: "martini2_dna_tutorial_library.DC",
        beads: DNA_DC_BEADS,
    },
    SoluteTemplate {
        name: "DG",
        source: "martini2_dna_tutorial_library.DG",
        beads: DNA_DG_BEADS,
    },
    SoluteTemplate {
        name: "DT",
        source: "martini2_dna_tutorial_library.DT",
        beads: DNA_DT_BEADS,
    },
    SoluteTemplate {
        name: "ADEN",
        source: "martini3_nucleobase_library.ADEN",
        beads: NUCLEOBASE_ADEN_BEADS,
    },
    SoluteTemplate {
        name: "CYTO",
        source: "martini3_nucleobase_library.CYTO",
        beads: NUCLEOBASE_CYTO_BEADS,
    },
    SoluteTemplate {
        name: "GUAN",
        source: "martini3_nucleobase_library.GUAN",
        beads: NUCLEOBASE_GUAN_BEADS,
    },
    SoluteTemplate {
        name: "THYM",
        source: "martini3_nucleobase_library.THYM",
        beads: NUCLEOBASE_THYM_BEADS,
    },
    SoluteTemplate {
        name: "URAC",
        source: "martini3_nucleobase_library.URAC",
        beads: NUCLEOBASE_URAC_BEADS,
    },
    SoluteTemplate {
        name: "WT4",
        source: "sirah.WT4",
        beads: SIRAH_WT4_BEADS,
    },
    SoluteTemplate {
        name: "NaW",
        source: "sirah.NaW",
        beads: SIRAH_NAW_BEADS,
    },
    SoluteTemplate {
        name: "ClW",
        source: "sirah.ClW",
        beads: SIRAH_CLW_BEADS,
    },
];
