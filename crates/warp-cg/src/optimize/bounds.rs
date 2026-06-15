use crate::parameters::{AngleStats, BondedStats, DihedralStats};
use crate::reference::{ReferenceDistributionTarget, ReferenceTargetSet, ReferenceTermKind};

use super::ParameterBound;

pub(super) fn default_bounds(stats: &BondedStats) -> Vec<ParameterBound> {
    let mut bounds =
        Vec::with_capacity((stats.bonds.len() + stats.angles.len() + stats.dihedrals.len()) * 2);
    for stat in &stats.bonds {
        let spread = stat.std.max(0.05);
        bounds.push(ParameterBound {
            name: format!("bond_{}_{}_length_angstrom", stat.bead_i, stat.bead_j),
            min: (stat.mean - 4.0 * spread).max(0.01),
            max: stat.mean + 4.0 * spread,
        });
        bounds.push(ParameterBound {
            name: format!("bond_{}_{}_force", stat.bead_i, stat.bead_j),
            min: 1.0,
            max: 5000.0,
        });
    }
    for stat in &stats.angles {
        let spread = stat.std_deg.max(5.0);
        let min = (stat.mean_deg - 4.0 * spread).clamp(0.0, 180.0);
        let max = (stat.mean_deg + 4.0 * spread).clamp(0.0, 180.0);
        bounds.push(ParameterBound {
            name: angle_value_name(stat),
            min,
            max: max.max(min + 1.0e-6),
        });
        bounds.push(ParameterBound {
            name: angle_force_name(stat),
            min: 1.0,
            max: 500.0,
        });
    }
    for stat in &stats.dihedrals {
        let spread = stat.std_deg.max(10.0);
        let min = (stat.mean_deg - 4.0 * spread).clamp(-180.0, 180.0);
        let max = (stat.mean_deg + 4.0 * spread).clamp(-180.0, 180.0);
        bounds.push(ParameterBound {
            name: dihedral_value_name(stat),
            min,
            max: max.max(min + 1.0e-6),
        });
        bounds.push(ParameterBound {
            name: dihedral_force_name(stat),
            min: 0.1,
            max: 100.0,
        });
    }
    bounds
}

pub(super) fn reference_target_bounds(targets: &ReferenceTargetSet) -> Vec<ParameterBound> {
    let mut bounds = Vec::new();
    for target in &targets.constraints {
        bounds.push(term_value_bound(target));
    }
    for target in &targets.bonds {
        bounds.push(term_value_bound(target));
        bounds.push(term_force_bound(target));
    }
    for target in &targets.angles {
        bounds.push(term_value_bound(target));
        bounds.push(term_force_bound(target));
    }
    for target in &targets.dihedrals {
        bounds.push(term_value_bound(target));
        bounds.push(term_force_bound(target));
    }
    bounds
}

fn term_value_bound(target: &ReferenceDistributionTarget) -> ParameterBound {
    let spread = target.std.max(match target.kind {
        ReferenceTermKind::Constraint | ReferenceTermKind::Bond => 0.02,
        ReferenceTermKind::Angle => 1.0,
        ReferenceTermKind::Dihedral => 5.0,
    });
    let (range_min, range_max) = match target.kind {
        ReferenceTermKind::Constraint | ReferenceTermKind::Bond => (
            0.01,
            targets_max_domain(target).max(target.mean + 4.0 * spread),
        ),
        ReferenceTermKind::Angle => (0.0, 180.0),
        ReferenceTermKind::Dihedral => (-180.0, 180.0),
    };
    let min = (target.mean - 4.0 * spread).clamp(range_min, range_max);
    let max = (target.mean + 4.0 * spread).clamp(range_min, range_max);
    ParameterBound {
        name: reference_target_value_name(target),
        min,
        max: max.max(min + 1.0e-6),
    }
}

fn term_force_bound(target: &ReferenceDistributionTarget) -> ParameterBound {
    let (min, max) = match target.kind {
        ReferenceTermKind::Constraint => (0.0, 0.0),
        ReferenceTermKind::Bond => (1.0, 5000.0),
        ReferenceTermKind::Angle => (1.0, 500.0),
        ReferenceTermKind::Dihedral => (0.1, 100.0),
    };
    ParameterBound {
        name: reference_target_force_name(target)
            .unwrap_or_else(|| format!("{}_force", reference_target_value_name(target))),
        min,
        max,
    }
}

fn targets_max_domain(target: &ReferenceDistributionTarget) -> f64 {
    target
        .bin_edges
        .last()
        .copied()
        .unwrap_or(target.domain[1])
        .max(target.domain[1])
        .max(0.01)
}

pub(super) fn reference_target_value_name(target: &ReferenceDistributionTarget) -> String {
    if let Some(label) = target.label.as_deref().filter(|label| !label.is_empty()) {
        let label = parameter_safe_label(label);
        return match target.kind {
            ReferenceTermKind::Constraint | ReferenceTermKind::Bond => {
                format!("{label}_{}", reference_target_length_suffix(target))
            }
            ReferenceTermKind::Angle => format!("{label}_angle_deg"),
            ReferenceTermKind::Dihedral => format!("{label}_phase_deg"),
        };
    }
    let beads = target
        .members
        .first()
        .cloned()
        .unwrap_or_else(|| target.beads.clone());
    match target.kind {
        ReferenceTermKind::Constraint => {
            format!(
                "constraint_{}_{}_{}",
                beads[0],
                beads[1],
                reference_target_length_suffix(target)
            )
        }
        ReferenceTermKind::Bond => format!(
            "bond_{}_{}_{}",
            beads[0],
            beads[1],
            reference_target_length_suffix(target)
        ),
        ReferenceTermKind::Angle => {
            format!("angle_{}_{}_{}_angle_deg", beads[0], beads[1], beads[2])
        }
        ReferenceTermKind::Dihedral => format!(
            "dihedral_{}_{}_{}_{}_phase_deg",
            beads[0], beads[1], beads[2], beads[3]
        ),
    }
}

fn reference_target_length_suffix(target: &ReferenceDistributionTarget) -> &'static str {
    match target.units.trim().to_ascii_lowercase().as_str() {
        "nm" | "nanometer" | "nanometers" => "length_nm",
        _ => "length_angstrom",
    }
}

pub(super) fn reference_target_force_name(target: &ReferenceDistributionTarget) -> Option<String> {
    let suffix = match target.kind {
        ReferenceTermKind::Constraint => return None,
        ReferenceTermKind::Bond | ReferenceTermKind::Angle | ReferenceTermKind::Dihedral => "force",
    };
    if let Some(label) = target.label.as_deref().filter(|label| !label.is_empty()) {
        let label = parameter_safe_label(label);
        return Some(format!("{label}_{suffix}"));
    }
    let beads = target
        .members
        .first()
        .cloned()
        .unwrap_or_else(|| target.beads.clone());
    Some(match target.kind {
        ReferenceTermKind::Constraint => return None,
        ReferenceTermKind::Bond => format!("bond_{}_{}_force", beads[0], beads[1]),
        ReferenceTermKind::Angle => format!("angle_{}_{}_{}_force", beads[0], beads[1], beads[2]),
        ReferenceTermKind::Dihedral => format!(
            "dihedral_{}_{}_{}_{}_force",
            beads[0], beads[1], beads[2], beads[3]
        ),
    })
}

fn parameter_safe_label(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    let mut last_was_sep = false;
    for ch in label.chars() {
        let safe = ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-';
        if safe {
            out.push(ch);
            last_was_sep = false;
        } else if !last_was_sep {
            out.push('_');
            last_was_sep = true;
        }
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "bonded_class".to_string()
    } else {
        trimmed
    }
}

fn angle_value_name(stat: &AngleStats) -> String {
    format!(
        "angle_{}_{}_{}_angle_deg",
        stat.bead_i, stat.bead_j, stat.bead_k
    )
}

fn angle_force_name(stat: &AngleStats) -> String {
    format!(
        "angle_{}_{}_{}_force",
        stat.bead_i, stat.bead_j, stat.bead_k
    )
}

fn dihedral_value_name(stat: &DihedralStats) -> String {
    format!(
        "dihedral_{}_{}_{}_{}_phase_deg",
        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
    )
}

fn dihedral_force_name(stat: &DihedralStats) -> String {
    format!(
        "dihedral_{}_{}_{}_{}_force",
        stat.bead_i, stat.bead_j, stat.bead_k, stat.bead_l
    )
}
