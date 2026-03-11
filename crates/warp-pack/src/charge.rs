use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::error::{PackError, PackResult};
use crate::io::read_prmtop_total_charge;

pub const CHARGE_MANIFEST_VERSION: &str = "warp-build.charge-manifest.v1";
pub const LEGACY_CHARGE_MANIFEST_VERSION: &str = "warp-pack.charge-manifest.v1";

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeAtom {
    pub index: usize,
    pub charge_e: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ChargeManifest {
    pub version: String,
    #[serde(default)]
    pub solute_path: Option<String>,
    #[serde(default)]
    pub topology_ref: Option<String>,
    #[serde(default)]
    pub source_topology_ref: Option<String>,
    #[serde(default)]
    pub target_topology_ref: Option<String>,
    #[serde(default)]
    pub forcefield_ref: Option<String>,
    #[serde(default)]
    pub charge_derivation: Option<String>,
    #[serde(default)]
    pub net_charge_e: Option<f32>,
    #[serde(default)]
    pub atom_count: Option<usize>,
    #[serde(default)]
    pub partial_charges: Option<serde_json::Value>,
    #[serde(default)]
    pub atom_charges: Option<Vec<ChargeAtom>>,
    #[serde(default)]
    pub head_charge_e: Option<f32>,
    #[serde(default)]
    pub repeat_charge_e: Option<f32>,
    #[serde(default)]
    pub tail_charge_e: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct NetChargeEstimate {
    pub net_charge_e: Option<f32>,
    pub source: Option<String>,
}

pub fn load_charge_manifest(path: &Path) -> PackResult<ChargeManifest> {
    let payload = std::fs::read_to_string(path)?;
    let manifest: ChargeManifest =
        serde_json::from_str(&payload).map_err(|err| PackError::Parse(err.to_string()))?;
    if manifest.version != CHARGE_MANIFEST_VERSION
        && manifest.version != LEGACY_CHARGE_MANIFEST_VERSION
    {
        return Err(PackError::Invalid(format!(
            "unsupported charge manifest version '{}'",
            manifest.version
        )));
    }
    Ok(manifest)
}

pub fn charge_manifest_field_kinds(manifest: &ChargeManifest) -> Vec<String> {
    let mut kinds = Vec::new();
    if manifest.net_charge_e.is_some() {
        kinds.push("net_charge_e".to_string());
    }
    if manifest.atom_charges.is_some() {
        kinds.push("atom_charges".to_string());
    }
    if manifest.head_charge_e.is_some()
        || manifest.repeat_charge_e.is_some()
        || manifest.tail_charge_e.is_some()
    {
        kinds.push("repeat_scalars".to_string());
    }
    kinds
}

pub fn compute_solute_net_charge(manifest: &ChargeManifest) -> NetChargeEstimate {
    NetChargeEstimate {
        net_charge_e: manifest.net_charge_e,
        source: manifest
            .net_charge_e
            .map(|_| "charge_manifest.net_charge_e".to_string()),
    }
}

pub fn compute_solute_net_charge_from_prmtop(path: &Path) -> PackResult<NetChargeEstimate> {
    Ok(NetChargeEstimate {
        net_charge_e: Some(read_prmtop_total_charge(path)?),
        source: Some("prmtop.total_charge".into()),
    })
}
