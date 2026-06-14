use super::*;

pub(super) fn inserted_solvent_library_charge_estimate(
    component: &InsertedComponent,
    library: &SolventLibraryEntry,
) -> Result<warp_common::charge::NetChargeEstimate> {
    let derived = library.net_charge_e();
    if let Some(explicit) = component.net_charge_e {
        if (derived - explicit).abs() > 1.0e-4 {
            return Err(anyhow!(
                "inserted component {} net_charge_e ({explicit}) does not match solvent-library-derived charge ({derived})",
                component.name
            ));
        }
    }
    Ok(warp_common::charge::NetChargeEstimate {
        net_charge_e: Some(derived),
        source: Some(format!("solvent_library:{}", library.name)),
    })
}

pub(super) fn inserted_solvent_library_beads(
    library: &SolventLibraryEntry,
) -> Vec<(String, f32, [f32; 3])> {
    library
        .beads
        .iter()
        .map(|bead| (bead.atom_name.clone(), bead.charge_e, bead.offset_angstrom))
        .collect()
}
