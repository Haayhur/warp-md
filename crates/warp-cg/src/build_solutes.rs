#[derive(Clone, Debug)]
pub struct SoluteTemplate {
    pub name: &'static str,
    pub source: &'static str,
    pub beads: &'static [SoluteTemplateBead],
}

impl SoluteTemplate {
    pub fn net_charge_e(&self) -> f32 {
        self.beads.iter().map(|bead| bead.charge_e).sum()
    }
}

#[derive(Clone, Debug)]
pub struct SoluteTemplateBead {
    pub name: &'static str,
    pub offset_angstrom: [f32; 3],
    pub charge_e: f32,
}

#[derive(Clone, Debug)]
pub struct SoluteTemplateBond {
    pub bead_indices: [usize; 2],
    pub length_nm: f32,
    pub force_kj_mol_nm2: f32,
}

use crate::build_solutes_templates::{SIRAH_WT4_BONDS, TEMPLATES};

pub fn lookup_solute_template(name: &str) -> Option<&'static SoluteTemplate> {
    TEMPLATES
        .iter()
        .find(|template| template.name.eq_ignore_ascii_case(name))
}

pub fn known_solute_names() -> Vec<&'static str> {
    TEMPLATES.iter().map(|template| template.name).collect()
}

pub fn lookup_solute_template_bonds(name: &str) -> &'static [SoluteTemplateBond] {
    match lookup_solute_template(name).map(|template| template.name) {
        Some("WT4") => SIRAH_WT4_BONDS,
        _ => &[],
    }
}

#[cfg(test)]
#[path = "build_solutes_tests.rs"]
mod tests;
