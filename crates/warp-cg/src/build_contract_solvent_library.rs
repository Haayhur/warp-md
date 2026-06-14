use super::*;

#[derive(Clone, Debug)]
pub(super) struct ResolvedSolventSpecies {
    pub(super) name: String,
    pub(super) ratio: f32,
    pub(super) mapping_ratio: f32,
    pub(super) molar_mass_g_mol: f32,
    pub(super) density_kg_m3: f32,
    pub(super) charge_e: f32,
    pub(super) beads: Vec<ResolvedSolventBead>,
}

#[derive(Clone, Debug)]
pub(super) struct ResolvedSolventBead {
    pub(super) atom_name: String,
    pub(super) offset_angstrom: [f32; 3],
    pub(super) charge_e: f32,
}

#[derive(Clone, Debug)]
pub(super) struct ResolvedIonSpecies {
    pub(super) name: String,
    pub(super) residue_name: String,
    pub(super) atom_name: String,
    pub(super) ratio: f32,
    pub(super) charge_e: i32,
}

#[derive(Clone, Debug)]
pub(super) struct SolventLibraryEntry {
    pub(super) name: String,
    pub(super) mapping_ratio: f32,
    pub(super) molar_mass_g_mol: f32,
    pub(super) density_kg_m3: f32,
    pub(super) beads: Vec<SolventLibraryBead>,
}

#[derive(Clone, Debug)]
pub(super) struct SolventLibraryBead {
    pub(super) atom_name: String,
    pub(super) offset_angstrom: [f32; 3],
    pub(super) charge_e: f32,
}

#[derive(Clone, Debug)]
pub(super) struct IonLibraryEntry {
    pub(super) name: &'static str,
    pub(super) atom_name: &'static str,
    pub(super) charge_e: i32,
    pub(super) default_charge: i32,
}

pub(super) fn standard_solvent_entry(
    name: &str,
    mapping_ratio: f32,
    molar_mass_g_mol: f32,
    density_kg_m3: f32,
    beads: &[(&str, [f32; 3], f32)],
) -> SolventLibraryEntry {
    SolventLibraryEntry {
        name: name.to_string(),
        mapping_ratio,
        molar_mass_g_mol,
        density_kg_m3,
        beads: beads
            .iter()
            .map(|(atom_name, offset, charge)| solvent_library_bead(*atom_name, *offset, *charge))
            .collect(),
    }
}

pub(super) fn atomistic_tip3_water_entry(name: &str) -> SolventLibraryEntry {
    SolventLibraryEntry {
        name: name.to_string(),
        mapping_ratio: 1.0,
        molar_mass_g_mol: 18.01528,
        density_kg_m3: 996.69,
        beads: vec![
            solvent_library_bead("OW", [0.0, 0.0, 0.0], -0.834),
            solvent_library_bead("HW1", [0.74, 0.64, 0.0], 0.417),
            solvent_library_bead("HW2", [-0.74, 0.64, 0.0], 0.417),
        ],
    }
}

pub(super) fn atomistic_tip4_water_entry(name: &str) -> SolventLibraryEntry {
    SolventLibraryEntry {
        name: name.to_string(),
        mapping_ratio: 1.0,
        molar_mass_g_mol: 18.01528,
        density_kg_m3: 996.69,
        beads: vec![
            solvent_library_bead("OW", [0.0, 0.0, 0.0], 0.0),
            solvent_library_bead("HW1", [0.74, 0.64, 0.0], 0.52),
            solvent_library_bead("HW2", [-0.74, 0.64, 0.0], 0.52),
            solvent_library_bead("MW", [0.0, 0.32, 0.0], -1.04),
        ],
    }
}

pub(super) fn atomistic_tip5_water_entry(name: &str) -> SolventLibraryEntry {
    SolventLibraryEntry {
        name: name.to_string(),
        mapping_ratio: 1.0,
        molar_mass_g_mol: 18.01528,
        density_kg_m3: 996.69,
        beads: vec![
            solvent_library_bead("OW", [0.0, 0.0, 0.0], 0.0),
            solvent_library_bead("HW1", [0.74, 0.64, 0.0], 0.241),
            solvent_library_bead("HW2", [-0.74, 0.64, 0.0], 0.241),
            solvent_library_bead("LP1", [0.2, 0.2, 0.0], -0.241),
            solvent_library_bead("LP2", [-0.2, -0.2, 0.0], -0.241),
        ],
    }
}

pub(super) fn lookup_small_molecule_solvent_library(
    name: &str,
    molar_mass_g_mol: f32,
    density_kg_m3: f32,
) -> Option<SolventLibraryEntry> {
    let template = lookup_solute_template(name)?;
    Some(SolventLibraryEntry {
        name: template.name.to_string(),
        mapping_ratio: 1.0,
        molar_mass_g_mol,
        density_kg_m3,
        beads: template
            .beads
            .iter()
            .map(|bead| solvent_library_bead(bead.name, bead.offset_angstrom, bead.charge_e))
            .collect(),
    })
}

pub(super) fn solvent_library_bead(
    atom_name: impl Into<String>,
    offset_angstrom: [f32; 3],
    charge_e: f32,
) -> SolventLibraryBead {
    SolventLibraryBead {
        atom_name: atom_name.into(),
        offset_angstrom,
        charge_e,
    }
}

pub(super) fn normalize_library_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
}

pub(super) fn resolved_solvent_species(solvent: &SolventPolicy) -> Vec<ResolvedSolventSpecies> {
    if solvent.species.is_empty() {
        let library = lookup_solvent_library(&solvent.name);
        let charge_e = library
            .as_ref()
            .map(|entry| entry.net_charge_e())
            .unwrap_or(0.0);
        return vec![ResolvedSolventSpecies {
            name: library
                .as_ref()
                .map(|entry| entry.name.to_string())
                .unwrap_or_else(|| solvent.name.clone()),
            ratio: 1.0,
            mapping_ratio: library
                .as_ref()
                .filter(|_| {
                    (solvent.mapping_ratio - default_solvent_mapping_ratio()).abs() < 1.0e-6
                })
                .map(|entry| entry.mapping_ratio)
                .unwrap_or(solvent.mapping_ratio),
            molar_mass_g_mol: library
                .as_ref()
                .filter(|_| {
                    (solvent.molar_mass_g_mol - default_solvent_molar_mass()).abs() < 1.0e-6
                })
                .map(|entry| entry.molar_mass_g_mol)
                .unwrap_or(solvent.molar_mass_g_mol),
            density_kg_m3: library
                .as_ref()
                .filter(|_| (solvent.density_kg_m3 - default_solvent_density()).abs() < 1.0e-6)
                .map(|entry| entry.density_kg_m3)
                .unwrap_or(solvent.density_kg_m3),
            charge_e,
            beads: resolved_solvent_beads(&solvent.name, library.as_ref(), charge_e),
        }];
    }
    solvent
        .species
        .iter()
        .map(|species| {
            let library = lookup_solvent_library(&species.name);
            let charge_e = library
                .as_ref()
                .filter(|_| species.charge_e == 0.0)
                .map(|entry| entry.net_charge_e())
                .unwrap_or(species.charge_e);
            ResolvedSolventSpecies {
                name: library
                    .as_ref()
                    .map(|entry| entry.name.to_string())
                    .unwrap_or_else(|| species.name.clone()),
                ratio: species.ratio,
                mapping_ratio: library
                    .as_ref()
                    .filter(|_| {
                        (species.mapping_ratio - default_solvent_mapping_ratio()).abs() < 1.0e-6
                    })
                    .map(|entry| entry.mapping_ratio)
                    .unwrap_or(species.mapping_ratio),
                molar_mass_g_mol: library
                    .as_ref()
                    .filter(|_| {
                        (species.molar_mass_g_mol - default_solvent_molar_mass()).abs() < 1.0e-6
                    })
                    .map(|entry| entry.molar_mass_g_mol)
                    .unwrap_or(species.molar_mass_g_mol),
                density_kg_m3: library
                    .as_ref()
                    .filter(|_| (species.density_kg_m3 - default_solvent_density()).abs() < 1.0e-6)
                    .map(|entry| entry.density_kg_m3)
                    .unwrap_or(species.density_kg_m3),
                charge_e,
                beads: resolved_solvent_beads(&species.name, library.as_ref(), charge_e),
            }
        })
        .collect()
}

impl SolventLibraryEntry {
    pub(super) fn net_charge_e(&self) -> f32 {
        self.beads.iter().map(|bead| bead.charge_e).sum()
    }
}

pub(super) fn resolved_solvent_beads(
    species_name: &str,
    library: Option<&SolventLibraryEntry>,
    charge_e: f32,
) -> Vec<ResolvedSolventBead> {
    if let Some(library) = library {
        return library
            .beads
            .iter()
            .map(|bead| ResolvedSolventBead {
                atom_name: bead.atom_name.to_string(),
                offset_angstrom: bead.offset_angstrom,
                charge_e: bead.charge_e,
            })
            .collect();
    }
    vec![ResolvedSolventBead {
        atom_name: solvent_atom_name(species_name),
        offset_angstrom: [0.0, 0.0, 0.0],
        charge_e,
    }]
}

pub(super) fn ion_residue_name(name: &str) -> String {
    ion_atom_name(name)
}

pub(super) fn solvent_atom_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(5)
        .collect::<String>()
        .to_ascii_uppercase()
        .if_empty_then("SOL")
}

pub(super) fn ion_atom_name(name: &str) -> String {
    name.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .take(5)
        .collect::<String>()
        .to_ascii_uppercase()
        .if_empty_then("ION")
}
