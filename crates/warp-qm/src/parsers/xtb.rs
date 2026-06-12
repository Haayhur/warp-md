use std::path::Path;

use crate::contract::QM_SCHEMA_VERSION;

use super::{finish_status, first_float_after, OutputInspection};

pub fn inspect(path: &Path, text: &str) -> OutputInspection {
    let mut report = OutputInspection {
        schema_version: QM_SCHEMA_VERSION.into(),
        engine: "xtb".into(),
        path: path.to_string_lossy().into_owned(),
        ..OutputInspection::default()
    };
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains("normal termination") {
            report.convergence_status = Some("terminated_normally".into());
        }
        if trimmed.contains("TOTAL ENERGY") {
            report.final_energy_hartree = first_float_after(trimmed, "TOTAL ENERGY");
        }
        if trimmed.contains("WARNING") {
            report.warnings.push(trimmed.to_string());
        }
        if trimmed.contains("ERROR") || trimmed.contains("abnormal termination") {
            report.fatal_errors.push(trimmed.to_string());
        }
    }
    finish_status(&mut report);
    report
}
