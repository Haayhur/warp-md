//! Structure validation: bond geometry, steric clashes, missing atoms.

use crate::coord::Vec3;
use crate::non_standard::NonStdResidue;
use crate::residue::{ResName, Structure};

/// A single validation issue.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub kind: IssueKind,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Warning,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueKind {
    MissingAtom,
    BondLength,
    StericClash,
    ChiralityError,
}

/// Expected backbone atom names.
const BACKBONE_REQUIRED: &[&str] = &["N", "CA", "C", "O"];

/// Expected heavy-atom counts per residue (backbone + side chain, no H, no OXT).
fn expected_heavy_atoms(name: ResName) -> usize {
    match name {
        ResName::GLY => 4,
        ResName::ALA => 5,
        ResName::SER => 6,
        ResName::CYS => 6,
        ResName::VAL => 7,
        ResName::ILE => 8,
        ResName::LEU => 8,
        ResName::THR => 7,
        ResName::ARG => 11,
        ResName::LYS => 9,
        ResName::ASP => 8,
        ResName::GLU => 9,
        ResName::ASN => 8,
        ResName::GLN => 9,
        ResName::MET => 8,
        ResName::HIS => 10,
        ResName::PRO => 7,
        ResName::PHE => 11,
        ResName::TYR => 12,
        ResName::TRP => 14,
        ResName::ACE => 3,
        ResName::NME => 2,
    }
}

fn expected_heavy_atoms_for_residue(name: ResName, non_std: Option<NonStdResidue>) -> usize {
    match non_std {
        Some(NonStdResidue::MSE) => expected_heavy_atoms(ResName::MET),
        Some(NonStdResidue::PCA) => expected_heavy_atoms(ResName::GLU) - 1,
        None => expected_heavy_atoms(name),
    }
}

/// Run all validations on a structure. Returns list of issues.
pub fn validate(struc: &Structure) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    check_missing_backbone(struc, &mut issues);
    check_peptide_bond_lengths(struc, &mut issues);
    check_steric_clashes(struc, &mut issues);
    issues
}

/// Check that all residues have the required backbone atoms.
fn check_missing_backbone(struc: &Structure, issues: &mut Vec<ValidationIssue>) {
    for chain in &struc.chains {
        for res in &chain.residues {
            // Skip terminal caps — they don't have standard backbone atoms
            if res.name.is_cap() {
                continue;
            }
            for &name in BACKBONE_REQUIRED {
                if res.atom_coord(name).is_none() {
                    issues.push(ValidationIssue {
                        severity: Severity::Error,
                        kind: IssueKind::MissingAtom,
                        message: format!(
                            "chain {} res {} {}: missing backbone atom {}",
                            chain.id, res.seq_id, res.name.as_str(), name
                        ),
                    });
                }
            }
        }
    }
}

/// Check peptide bond C–N distances between consecutive residues.
fn check_peptide_bond_lengths(struc: &Structure, issues: &mut Vec<ValidationIssue>) {
    for chain in &struc.chains {
        for i in 0..chain.residues.len().saturating_sub(1) {
            let r1 = &chain.residues[i];
            let r2 = &chain.residues[i + 1];
            if let (Some(c), Some(n)) = (r1.atom_coord("C"), r2.atom_coord("N")) {
                let dist = c.sub(n).length();
                if dist < 1.0 || dist > 1.7 {
                    issues.push(ValidationIssue {
                        severity: Severity::Warning,
                        kind: IssueKind::BondLength,
                        message: format!(
                            "chain {} res {}-{}: peptide bond C-N distance {:.3} Å (expected ~1.33)",
                            chain.id, r1.seq_id, r2.seq_id, dist
                        ),
                    });
                }
            }
        }
    }
}

/// Check for steric clashes (non-bonded atoms closer than 1.5 Å).
fn check_steric_clashes(struc: &Structure, issues: &mut Vec<ValidationIssue>) {
    let clash_cutoff = 1.5_f64;
    // Collect all atom positions with identifiers and residue key for bond skipping
    struct AtomEntry {
        coord: Vec3,
        label: String,
        chain_id: char,
        resid: i32,
    }
    let mut all_atoms: Vec<AtomEntry> = Vec::new();
    for chain in &struc.chains {
        for res in &chain.residues {
            for atom in &res.atoms {
                let label = format!("{}/{}{}/{}", chain.id, res.name.as_str(), res.seq_id, atom.name);
                all_atoms.push(AtomEntry {
                    coord: atom.coord,
                    label,
                    chain_id: chain.id,
                    resid: res.seq_id,
                });
            }
        }
    }

    // O(n²) — fine for small structures; for large proteins, use spatial hashing
    let n = all_atoms.len();
    for i in 0..n {
        for j in (i + 1)..n {
            // Skip atoms within the same residue (covalently bonded or 1-3 connected)
            if all_atoms[i].chain_id == all_atoms[j].chain_id
                && all_atoms[i].resid == all_atoms[j].resid
            {
                continue;
            }
            // Skip atoms in adjacent residues (peptide bond C-N, etc.)
            if all_atoms[i].chain_id == all_atoms[j].chain_id
                && (all_atoms[i].resid - all_atoms[j].resid).abs() == 1
            {
                continue;
            }
            let dist = all_atoms[i].coord.sub(all_atoms[j].coord).length();
            if dist < clash_cutoff {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    kind: IssueKind::StericClash,
                    message: format!(
                        "steric clash ({:.2} Å): {} — {}",
                        dist, all_atoms[i].label, all_atoms[j].label
                    ),
                });
            }
        }
    }
}

/// Check atom count vs expected for each residue. Returns issues for short residues.
pub fn check_atom_counts(struc: &Structure) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();
    for chain in &struc.chains {
        for res in &chain.residues {
            let expected = expected_heavy_atoms_for_residue(res.name, res.non_std);
            // Count non-H atoms (exclude hydrogens)
            let heavy: usize = res.atoms.iter().filter(|a| a.element != "H").count();
            if heavy < expected {
                issues.push(ValidationIssue {
                    severity: Severity::Warning,
                    kind: IssueKind::MissingAtom,
                    message: format!(
                        "chain {} res {} {}: {} heavy atoms, expected {}",
                        chain.id, res.seq_id, res.name.as_str(), heavy, expected
                    ),
                });
            }
        }
    }
    issues
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::{make_extended_structure, make_extended_structure_from_specs, parse_three_letter_sequence};

    #[test]
    fn test_valid_structure_no_errors() {
        let struc = make_extended_structure("AAA").unwrap();
        let issues = validate(&struc);
        let errors: Vec<_> = issues.iter().filter(|i| i.severity == Severity::Error).collect();
        assert!(errors.is_empty(), "no errors expected: {:?}", errors);
    }

    #[test]
    fn test_peptide_bond_ok() {
        let struc = make_extended_structure("AG").unwrap();
        let issues = validate(&struc);
        let bond_issues: Vec<_> = issues.iter().filter(|i| i.kind == IssueKind::BondLength).collect();
        assert!(bond_issues.is_empty(), "peptide bonds should be OK: {:?}", bond_issues);
    }

    #[test]
    fn test_atom_counts_ok() {
        let struc = make_extended_structure("ACDE").unwrap();
        let issues = check_atom_counts(&struc);
        assert!(issues.is_empty(), "atom counts should match: {:?}", issues);
    }

    #[test]
    fn test_missing_backbone_detected() {
        use crate::residue::{Atom, Residue, Structure};
        use crate::coord::Vec3;
        let mut struc = Structure::new();
        // Create a residue missing "O"
        let mut res = Residue::new(ResName::ALA, 1);
        res.atoms.push(Atom::new("N", "N", Vec3::zero()));
        res.atoms.push(Atom::new("CA", "C", Vec3::new(1.0, 0.0, 0.0)));
        res.atoms.push(Atom::new("C", "C", Vec3::new(2.0, 0.0, 0.0)));
        // No "O" atom
        struc.chain_a_mut().residues.push(res);
        let issues = validate(&struc);
        assert!(issues.iter().any(|i| i.kind == IssueKind::MissingAtom && i.message.contains("O")));
    }

    #[test]
    fn test_atom_counts_pca_ok() {
        let specs = parse_three_letter_sequence("PCA-ALA").unwrap();
        let struc = make_extended_structure_from_specs(&specs).unwrap();
        let issues = check_atom_counts(&struc);
        assert!(issues.is_empty(), "PCA atom counts should match: {:?}", issues);
    }
}
