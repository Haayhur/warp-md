use super::*;

pub(super) fn lookup_amino_acid_solvent_library(name: &str) -> Option<SolventLibraryEntry> {
    let beads = match name {
        "GLY" => amino_acid_glycine_beads(),
        "ALA" | "CYS" | "VAL" | "LEU" | "ILE" | "MET" | "PRO" | "HYP" | "ASN" | "GLN" | "THR"
        | "SER" | "ASPP" | "ASH" | "GLUP" | "GLH" => amino_acid_two_beads(0.0),
        "ASP" | "GLU" => amino_acid_two_beads(-1.0),
        "ARG" | "LYS" => amino_acid_three_beads(1.0),
        "LSN" | "LYN" => amino_acid_three_beads(0.0),
        "PHE" | "HIS" | "HIE" | "HSE" | "HSD" | "HID" => amino_acid_four_beads(0.0),
        "HSP" | "HIP" => amino_acid_four_beads(1.0),
        "TYR" => amino_acid_tyr_beads(),
        "TRP" => amino_acid_trp_beads(),
        _ => return None,
    };
    Some(SolventLibraryEntry {
        name: amino_acid_solvent_name(name)?.to_string(),
        mapping_ratio: 1.0,
        molar_mass_g_mol: 18.01528,
        density_kg_m3: 996.69,
        beads,
    })
}

pub(super) fn known_amino_acid_solvent_names() -> Vec<&'static str> {
    vec![
        "GLY", "ALA", "CYS", "VAL", "LEU", "ILE", "MET", "PRO", "HYP", "ASN", "GLN", "THR", "SER",
        "ASPP", "ASH", "GLUP", "GLH", "ASP", "GLU", "ARG", "LYS", "LSN", "LYN", "PHE", "HIS",
        "HIE", "HSE", "HSD", "HID", "HSP", "HIP", "TYR", "TRP",
    ]
}

fn amino_acid_solvent_name(name: &str) -> Option<&'static str> {
    known_amino_acid_solvent_names()
        .into_iter()
        .find(|known| *known == name)
}

fn amino_acid_glycine_beads() -> Vec<SolventLibraryBead> {
    vec![solvent_library_bead("BB", [0.0, 0.0, 0.0], 0.0)]
}

fn amino_acid_two_beads(sidechain_charge: f32) -> Vec<SolventLibraryBead> {
    vec![
        solvent_library_bead("BB", [2.5, 0.0, 0.0], 0.0),
        solvent_library_bead("SC1", [-2.5, 0.0, 0.0], sidechain_charge),
    ]
}

fn amino_acid_three_beads(sidechain_charge: f32) -> Vec<SolventLibraryBead> {
    vec![
        solvent_library_bead("BB", [2.5, 0.0, 0.0], 0.0),
        solvent_library_bead("SC1", [0.0, 0.0, 0.0], 0.0),
        solvent_library_bead("SC2", [-2.5, 1.25, 0.0], sidechain_charge),
    ]
}

fn amino_acid_four_beads(total_sidechain_charge: f32) -> Vec<SolventLibraryBead> {
    let split_charge = total_sidechain_charge / 2.0;
    vec![
        solvent_library_bead("BB", [2.5, 0.0, 0.0], 0.0),
        solvent_library_bead("SC1", [0.0, 0.0, 0.0], 0.0),
        solvent_library_bead("SC2", [-2.5, 1.25, 0.0], split_charge),
        solvent_library_bead("SC3", [-2.5, -1.25, 0.0], split_charge),
    ]
}

fn amino_acid_tyr_beads() -> Vec<SolventLibraryBead> {
    vec![
        solvent_library_bead("BB", [2.5, 1.25, 0.0], 0.0),
        solvent_library_bead("SC1", [2.5, 0.0, 0.0], 0.0),
        solvent_library_bead("SC2", [0.0, -1.25, 0.0], 0.0),
        solvent_library_bead("SC3", [0.0, 1.25, 0.0], 0.0),
        solvent_library_bead("SC4", [-2.5, 0.0, 0.0], 0.0),
    ]
}

fn amino_acid_trp_beads() -> Vec<SolventLibraryBead> {
    let mut beads = amino_acid_tyr_beads();
    beads.push(solvent_library_bead("SC5", [-2.5, 1.25, 0.0], 0.0));
    beads
}
