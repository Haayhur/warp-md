pub mod orca;
pub mod psi4;
pub mod xtb;

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct OutputInspection {
    pub schema_version: String,
    pub status: String,
    pub engine: String,
    pub path: String,
    pub convergence_status: Option<String>,
    pub final_energy_hartree: Option<f64>,
    pub warnings: Vec<String>,
    pub fatal_errors: Vec<String>,
    pub optimized_geometry: Option<ParsedGeometry>,
    pub esp: Option<EspMetadata>,
    pub frequencies_cm1: Vec<f64>,
    pub imaginary_frequency_count: Option<usize>,
    pub charge_analysis: Vec<ParsedChargeAnalysis>,
    pub nmr_shielding: Vec<ParsedNmrShielding>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ParsedGeometry {
    pub units: String,
    pub atoms: Vec<ParsedAtom>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ParsedAtom {
    pub element: String,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EspMetadata {
    pub grid_path: Option<String>,
    pub point_count: Option<usize>,
    pub potential_unit: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ParsedChargeAnalysis {
    pub model: String,
    pub atom_charges_e: Vec<f64>,
    pub total_charge_e: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ParsedNmrShielding {
    pub atom_index: usize,
    pub element: String,
    pub isotropic_ppm: f64,
    pub anisotropy_ppm: Option<f64>,
}

pub fn inspect_output(path: &Path, engine: &str) -> Result<OutputInspection, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    match engine {
        "orca" => Ok(orca::inspect(path, &text)),
        "psi4" => Ok(psi4::inspect(path, &text)),
        "xtb" => Ok(xtb::inspect(path, &text)),
        other => Err(format!("unsupported engine for inspect-output: {other}")),
    }
}

pub(crate) fn first_float_after(line: &str, marker: &str) -> Option<f64> {
    let tail = line
        .split_once(marker)
        .map(|(_, tail)| tail)
        .unwrap_or(line);
    tail.split_whitespace().find_map(|token| {
        token
            .trim_matches(|ch: char| ch == ':' || ch == '=')
            .parse()
            .ok()
    })
}

pub(crate) fn finish_status(report: &mut OutputInspection) {
    report.status = if report.fatal_errors.is_empty() {
        "ok".into()
    } else {
        "error".into()
    };
}
