use std::collections::BTreeMap;

use warp_common::charge::sum_bead_charges;

use super::{BuildBeadTemplate, SolventPlacementSummary};

#[derive(Clone, Debug)]
pub(super) struct EmittedBead {
    pub(super) residue_id: i32,
    pub(super) residue_name: String,
    pub(super) atom_name: String,
    pub(super) charge_e: f32,
    pub(super) position_angstrom: [f32; 3],
    pub(super) excluded_volume_factor: f32,
}

#[derive(Clone, Debug)]
pub(super) struct SolventEmission {
    pub(super) counts: BTreeMap<String, usize>,
    pub(super) summary: SolventPlacementSummary,
    pub(super) solvent_charge_e: f32,
    pub(super) baseline_ion_charge_e: f32,
    pub(super) neutralization_input_charge_e: f32,
}

#[derive(Clone, Debug)]
pub(super) struct SolventPlacementPlan {
    pub(super) candidates: Vec<[f32; 3]>,
    pub(super) grid_point_count: usize,
    pub(super) final_candidate_count: usize,
    pub(super) grid_squeeze_pass_count: usize,
    pub(super) squeezed_candidate_count: usize,
    pub(super) min_grid_spacing_angstrom: Option<f32>,
    pub(super) kick_attempt_count: usize,
    pub(super) kicked_inserted_count: usize,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum InsertedKind {
    Protein,
    Solute,
}

impl InsertedKind {
    pub(super) fn excluded_volume_factor(self) -> f32 {
        match self {
            InsertedKind::Protein => 2.0,
            InsertedKind::Solute => 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct ResolvedLipid {
    pub(super) name: String,
    pub(super) count: usize,
    pub(super) charge_e: f32,
    pub(super) radius_angstrom: f32,
    pub(super) beads: Vec<BuildBeadTemplate>,
    pub(super) template_source: String,
    pub(super) charge_source: String,
}

impl ResolvedLipid {
    pub(super) fn bead_charge_sum_e(&self) -> f32 {
        sum_bead_charges(
            &self
                .beads
                .iter()
                .map(|bead| bead.charge_e)
                .collect::<Vec<f32>>(),
        )
    }
}
