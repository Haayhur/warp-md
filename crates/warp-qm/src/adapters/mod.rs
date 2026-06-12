pub mod multiwfn;
pub mod orca;
pub mod psi4;
pub mod workflow;

use crate::contract::QmArtifact;

#[derive(Clone, Debug)]
pub struct AdapterRun {
    pub status: String,
    pub exit_code: i32,
    pub command: Option<Vec<String>>,
    pub artifacts: Vec<QmArtifact>,
    pub warnings: Vec<String>,
    pub properties: serde_json::Map<String, serde_json::Value>,
    pub summary: AdapterSummary,
}

#[derive(Clone, Debug, Default)]
pub struct AdapterSummary {
    pub energy_hartree: Option<f64>,
    pub converged: Option<bool>,
    pub n_atoms: Option<usize>,
}

impl AdapterRun {
    pub fn error(message: impl Into<String>, exit_code: i32) -> Self {
        Self {
            status: "error".into(),
            exit_code,
            command: None,
            artifacts: Vec::new(),
            warnings: vec![message.into()],
            properties: serde_json::Map::new(),
            summary: AdapterSummary::default(),
        }
    }

    pub fn error_code(code: &str, message: impl Into<String>, exit_code: i32) -> Self {
        let message = message.into();
        let mut properties = serde_json::Map::new();
        properties.insert(
            "errors".into(),
            serde_json::json!([{"code": code, "message": message}]),
        );
        Self {
            status: "error".into(),
            exit_code,
            command: None,
            artifacts: Vec::new(),
            warnings: vec![message],
            properties,
            summary: AdapterSummary::default(),
        }
    }
}
