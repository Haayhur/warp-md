use std::path::Path;

use crate::contract::QM_SCHEMA_VERSION;

use super::{
    finish_status, first_float_after, OutputInspection, ParsedAtom, ParsedChargeAnalysis,
    ParsedGeometry, ParsedNmrShielding,
};

pub fn inspect(path: &Path, text: &str) -> OutputInspection {
    let mut report = OutputInspection {
        schema_version: QM_SCHEMA_VERSION.into(),
        engine: "orca".into(),
        path: path.to_string_lossy().into_owned(),
        ..OutputInspection::default()
    };
    let lines: Vec<&str> = text.lines().collect();
    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.contains("ORCA TERMINATED NORMALLY") {
            report.convergence_status = Some("terminated_normally".into());
        }
        if trimmed.contains("HURRAY") {
            report.convergence_status = Some("geometry_optimized".into());
        }
        if trimmed.contains("SCF CONVERGED") {
            report.convergence_status = Some("scf_converged".into());
        }
        if trimmed.contains("FINAL SINGLE POINT ENERGY") {
            report.final_energy_hartree = first_float_after(trimmed, "FINAL SINGLE POINT ENERGY");
        }
        let upper = trimmed.to_ascii_uppercase();
        if upper != "WARNINGS" && (upper.starts_with("WARNING") || upper.contains(" WARNING:")) {
            report.warnings.push(trimmed.to_string());
        }
        if trimmed.contains("ERROR") || trimmed.contains("ORCA finished by error termination") {
            report.fatal_errors.push(trimmed.to_string());
        }
        if trimmed.contains("CHELPG") {
            report.esp.get_or_insert_with(|| super::EspMetadata {
                grid_path: None,
                point_count: None,
                potential_unit: None,
            });
        }
        if trimmed == "VIBRATIONAL FREQUENCIES" {
            report.frequencies_cm1 = parse_frequencies(&lines[idx + 1..]);
            report.imaginary_frequency_count = Some(
                report
                    .frequencies_cm1
                    .iter()
                    .filter(|value| **value < 0.0)
                    .count(),
            );
        }
        if trimmed == "MULLIKEN ATOMIC CHARGES" {
            if let Some(charges) = parse_charge_table("mulliken", &lines[idx + 1..]) {
                report.charge_analysis.push(charges);
            }
        }
        if trimmed == "LOEWDIN ATOMIC CHARGES" {
            if let Some(charges) = parse_charge_table("loewdin", &lines[idx + 1..]) {
                report.charge_analysis.push(charges);
            }
        }
        if trimmed == "HIRSHFELD ANALYSIS" {
            if let Some(charges) = parse_hirshfeld(&lines[idx + 1..]) {
                report.charge_analysis.push(charges);
            }
        }
        if trimmed == "CHEMICAL SHIELDING SUMMARY (ppm)" {
            report.nmr_shielding = parse_nmr_summary(&lines[idx + 1..]);
        }
    }
    report.optimized_geometry = adjacent_xyz(path);
    finish_status(&mut report);
    report
}

fn parse_frequencies(lines: &[&str]) -> Vec<f64> {
    let mut values = Vec::new();
    let mut in_rows = false;
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if in_rows {
                break;
            }
            continue;
        }
        if let Some((_, rhs)) = trimmed.split_once(':') {
            if let Some(token) = rhs.split_whitespace().next() {
                if let Ok(value) = token.parse::<f64>() {
                    values.push(value);
                    in_rows = true;
                    continue;
                }
            }
        }
        if in_rows {
            break;
        }
    }
    values
}

fn parse_charge_table(model: &str, lines: &[&str]) -> Option<ParsedChargeAnalysis> {
    let mut charges = Vec::new();
    let mut total = None;
    for line in lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if !charges.is_empty() {
                break;
            }
            continue;
        }
        if let Some((_, rhs)) = trimmed.split_once(':') {
            if trimmed.contains("Sum of atomic charges") {
                total = rhs.trim().parse::<f64>().ok();
            } else if let Some(value) = rhs.split_whitespace().next().and_then(|v| v.parse().ok()) {
                charges.push(value);
            }
        }
    }
    if charges.is_empty() {
        None
    } else {
        Some(ParsedChargeAnalysis {
            model: model.into(),
            atom_charges_e: charges,
            total_charge_e: total,
        })
    }
}

fn parse_hirshfeld(lines: &[&str]) -> Option<ParsedChargeAnalysis> {
    let mut charges = Vec::new();
    let mut seen_header = false;
    for line in lines {
        let trimmed = line.trim();
        if trimmed.starts_with("ATOM") && trimmed.contains("CHARGE") {
            seen_header = true;
            continue;
        }
        if seen_header {
            if trimmed.is_empty() {
                if !charges.is_empty() {
                    break;
                }
                continue;
            }
            let parts: Vec<&str> = trimmed.split_whitespace().collect();
            if parts.len() >= 3 {
                if let Ok(value) = parts[2].parse::<f64>() {
                    charges.push(value);
                    continue;
                }
            }
            if !charges.is_empty() {
                break;
            }
        }
    }
    if charges.is_empty() {
        None
    } else {
        Some(ParsedChargeAnalysis {
            model: "hirshfeld".into(),
            total_charge_e: Some(charges.iter().sum()),
            atom_charges_e: charges,
        })
    }
}

fn parse_nmr_summary(lines: &[&str]) -> Vec<ParsedNmrShielding> {
    let mut shieldings = Vec::new();
    let mut seen_header = false;
    for line in lines {
        let trimmed = line.trim();
        if trimmed.starts_with("Nucleus") && trimmed.contains("Isotropic") {
            seen_header = true;
            continue;
        }
        if !seen_header || trimmed.is_empty() || trimmed.starts_with('-') {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() >= 4 {
            if let (Ok(atom_index), Ok(isotropic)) =
                (parts[0].parse::<usize>(), parts[2].parse::<f64>())
            {
                shieldings.push(ParsedNmrShielding {
                    atom_index,
                    element: parts[1].into(),
                    isotropic_ppm: isotropic,
                    anisotropy_ppm: parts.get(3).and_then(|value| value.parse().ok()),
                });
            }
        } else if !shieldings.is_empty() {
            break;
        }
    }
    shieldings
}

fn adjacent_xyz(path: &Path) -> Option<ParsedGeometry> {
    let xyz = path.with_extension("xyz");
    let text = std::fs::read_to_string(xyz).ok()?;
    parse_xyz(&text)
}

fn parse_xyz(text: &str) -> Option<ParsedGeometry> {
    let mut lines = text.lines();
    let count = lines.next()?.trim().parse::<usize>().ok()?;
    let _comment = lines.next();
    let atoms: Vec<ParsedAtom> = lines
        .take(count)
        .filter_map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            Some(ParsedAtom {
                element: parts.first()?.to_string(),
                x: parts.get(1)?.parse().ok()?,
                y: parts.get(2)?.parse().ok()?,
                z: parts.get(3)?.parse().ok()?,
            })
        })
        .collect();
    if atoms.len() == count {
        Some(ParsedGeometry {
            units: "angstrom".into(),
            atoms,
        })
    } else {
        None
    }
}
