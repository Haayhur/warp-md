use std::path::Path;

use crate::error::{PackError, PackResult};
use crate::io::read_prmtop_total_charge;
pub use warp_common::charge::{
    charge_manifest_field_kinds, charges_match, compute_gromacs_molecule_net_charge,
    compute_solute_net_charge, neutralizer_count, spread_total_charge, sum_atom_charges,
    sum_bead_charges, sum_component_charges, sum_gromacs_molecule_atom_charges, ChargeAtom,
    ChargeManifest, ComponentCharge, NetChargeEstimate, CHARGE_MANIFEST_VERSION,
    LEGACY_CHARGE_MANIFEST_VERSION,
};

pub fn load_charge_manifest(path: &Path) -> PackResult<ChargeManifest> {
    let payload = std::fs::read_to_string(path)?;
    let manifest: ChargeManifest =
        serde_json::from_str(&payload).map_err(|err| PackError::Parse(err.to_string()))?;
    if manifest.schema_version != CHARGE_MANIFEST_VERSION
        && manifest.schema_version != LEGACY_CHARGE_MANIFEST_VERSION
    {
        return Err(PackError::Invalid(format!(
            "unsupported charge manifest version '{}'",
            manifest.schema_version
        )));
    }
    Ok(manifest)
}

pub fn compute_solute_net_charge_from_prmtop(path: &Path) -> PackResult<NetChargeEstimate> {
    Ok(NetChargeEstimate {
        net_charge_e: Some(read_prmtop_total_charge(path)?),
        source: Some("prmtop.total_charge".into()),
    })
}
