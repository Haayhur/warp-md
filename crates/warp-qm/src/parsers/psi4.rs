use std::path::Path;

use crate::contract::QM_SCHEMA_VERSION;

use super::{finish_status, first_float_after, OutputInspection};

pub fn inspect(path: &Path, text: &str) -> OutputInspection {
    let mut report = OutputInspection {
        schema_version: QM_SCHEMA_VERSION.into(),
        engine: "psi4".into(),
        path: path.to_string_lossy().into_owned(),
        ..OutputInspection::default()
    };
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains("Psi4 exiting successfully") {
            report.convergence_status = Some("terminated_normally".into());
        }
        if trimmed.contains("SCF has converged") {
            report.convergence_status = Some("scf_converged".into());
        }
        if trimmed.contains("Total Energy") || trimmed.contains("Final Energy") {
            if report.final_energy_hartree.is_none() {
                report.final_energy_hartree = first_float_after(trimmed, "=");
            }
        }
        if trimmed.contains("WARNING") {
            report.warnings.push(trimmed.to_string());
        }
        if trimmed.contains("Traceback") || trimmed.contains("Fatal Error") {
            report.fatal_errors.push(trimmed.to_string());
        }
        if trimmed.contains("ESP") || trimmed.contains("RESP") {
            report.esp.get_or_insert_with(|| super::EspMetadata {
                grid_path: None,
                point_count: None,
                potential_unit: None,
            });
        }
    }
    finish_status(&mut report);
    report
}
