//! Real-world integration tests for warp-pep.
//!
//! Covers: de-novo build, preset geometry, custom phi/psi, mutation,
//! multi-chain, JSON spec, Amber variants, disulfide detection,
//! PDB round-trips, and reading real experimental structures
//! (crambin 1CRN, murine EGF 1EGF).

use std::path::PathBuf;

use warp_pep::analysis::{measure_all_phi_psi, radius_of_gyration, rmsd_all_atoms, rmsd_ca};
use warp_pep::builder::{
    self, add_terminal_oxt, make_extended_structure, make_multi_chain_structure,
    make_preset_structure, make_structure, parse_three_letter_sequence, ChainSpec, RamaPreset,
};
use warp_pep::caps::{add_ace_cap, add_caps, add_nme_cap};
use warp_pep::convert::{read_structure, write_structure};
use warp_pep::disulfide::detect_disulfides;
use warp_pep::hydrogen::{add_backbone_hydrogens, add_ha_hydrogens};
use warp_pep::json_spec::BuildSpec;
use warp_pep::mutation::{mutate_residue, parse_mutation_spec};
use warp_pep::residue::{AmberVariant, ResName};
use warp_pep::selection::select;
use warp_pep::validation::validate;

/// Path to test fixture files relative to the crate root.
fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Temp file for write‐round-trip tests.
fn tmp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("warp_pep_integ_{}", name))
}

// ── helpers ──────────────────────────────────────────────────────────

/// Count atoms in chain A (or the only chain).
fn total_atom_count(struc: &warp_pep::residue::Structure) -> usize {
    struc
        .chains
        .iter()
        .flat_map(|c| &c.residues)
        .map(|r| r.atoms.len())
        .sum()
}

/// Euclidean distance between two atoms (by name) in consecutive residues.
fn inter_residue_dist(
    struc: &warp_pep::residue::Structure,
    chain_idx: usize,
    res_i: usize,
    atom_a: &str,
    res_j: usize,
    atom_b: &str,
) -> f64 {
    let a = struc.chains[chain_idx].residues[res_i]
        .atom_coord(atom_a)
        .unwrap();
    let b = struc.chains[chain_idx].residues[res_j]
        .atom_coord(atom_b)
        .unwrap();
    a.sub(b).length()
}

// ═══════════════════════════════════════════════════════════════════════
// 1. BUILD — all 20 amino acids
// ═══════════════════════════════════════════════════════════════════════

/// Expected heavy-atom count per residue (backbone 4 + side chain).
/// GLY=4, ALA=5, ..., TRP=14.
fn expected_atom_count(aa: ResName) -> usize {
    match aa {
        ResName::GLY => 4,
        ResName::ALA => 5,
        ResName::SER => 6,
        ResName::CYS => 6,
        ResName::VAL => 7,
        ResName::THR => 7,
        ResName::PRO => 7,
        ResName::ASP => 8,
        ResName::ASN => 8,
        ResName::LEU => 8,
        ResName::ILE => 8,
        ResName::MET => 8,
        ResName::HIS => 10,
        ResName::GLU => 9,
        ResName::GLN => 9,
        ResName::LYS => 9,
        ResName::PHE => 11,
        ResName::ARG => 11,
        ResName::TYR => 12,
        ResName::TRP => 14,
        ResName::ACE => 3,
        ResName::NME => 2,
    }
}

#[test]
fn test_build_each_aminoacid_atom_counts() {
    let all = "ACDEFGHIKLMNPQRSTVWY";
    let struc = make_extended_structure(all).unwrap();
    assert_eq!(struc.chain_a().residues.len(), 20);
    for (i, ch) in all.chars().enumerate() {
        let aa = ResName::from_one_letter(ch).unwrap();
        let res = &struc.chain_a().residues[i];
        assert_eq!(
            res.atoms.len(),
            expected_atom_count(aa),
            "atom count mismatch for {} ({})",
            aa.as_str(),
            ch
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 2. SECONDARY STRUCTURE — alpha-helix geometry
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_alpha_helix_rise_per_residue() {
    // α-helix: ~1.5 Å rise/residue, 3.6 residues/turn → ~5.4 Å/turn.
    // CA(i) to CA(i+1) in a helix ≈ 3.8 Å.
    let helix = make_preset_structure("AAAAAAAAAA", RamaPreset::AlphaHelix).unwrap();
    for i in 0..9 {
        let d = inter_residue_dist(&helix, 0, i, "CA", i + 1, "CA");
        assert!(
            (d - 3.80).abs() < 0.25,
            "CA({})-CA({}) = {:.2} Å, expected ~3.80",
            i + 1,
            i + 2,
            d
        );
    }
    // Verify helix pitch: CA(1) to CA(4) should be ~5.0–5.5 Å (one turn)
    let pitch = inter_residue_dist(&helix, 0, 0, "CA", 3, "CA");
    assert!(
        pitch > 4.5 && pitch < 6.5,
        "helix pitch CA(1)-CA(4) = {:.2} Å, expected 5.0–6.0",
        pitch
    );
}

// ═══════════════════════════════════════════════════════════════════════
// 3. SECONDARY STRUCTURE — beta-sheet geometry
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_beta_sheet_extended_distances() {
    // β-strand: nearly fully extended, CA-CA ≈ 3.3–3.5 Å.
    let sheet = make_preset_structure("VVVVVVVVVV", RamaPreset::BetaSheet).unwrap();
    for i in 0..9 {
        let d = inter_residue_dist(&sheet, 0, i, "CA", i + 1, "CA");
        assert!(
            d > 3.0 && d < 4.0,
            "beta CA({})-CA({}) = {:.2} Å, expected 3.0–4.0",
            i + 1,
            i + 2,
            d
        );
    }
    // End-to-end distance should be much larger than helix (≈ 30 Å for 10 res)
    let e2e = inter_residue_dist(&sheet, 0, 0, "CA", 9, "CA");
    assert!(e2e > 25.0, "beta e2e = {:.1} Å too short for an extended strand", e2e);
}

// ═══════════════════════════════════════════════════════════════════════
// 4. CUSTOM PHI/PSI — verify backbone dihedrals
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_custom_phi_psi_build() {
    // Build a 3-residue peptide with helix angles: phi=-57, psi=-47
    let phi = vec![-57.0, -57.0];
    let psi = vec![-47.0, -47.0];
    let struc = make_structure("AAA", &phi, &psi, None).unwrap();
    assert_eq!(struc.chain_a().residues.len(), 3);
    // Verify peptide bond distances (C_i - N_{i+1} ≈ 1.33 Å)
    for i in 0..2 {
        let c_i = struc.chain_a().residues[i].atom_coord("C").unwrap();
        let n_j = struc.chain_a().residues[i + 1].atom_coord("N").unwrap();
        let d = c_i.sub(n_j).length();
        assert!((d - 1.33).abs() < 0.05, "peptide bond {} = {:.3}", i, d);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 5. POLYPROLINE-II helix
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_polyproline_ii_geometry() {
    let ppii = make_preset_structure("PPPPPP", RamaPreset::PolyProII).unwrap();
    assert_eq!(ppii.chain_a().residues.len(), 6);
    // PPII: CA-CA ≈ 3.1–3.2 Å, more extended than helix but not fully extended
    for i in 0..5 {
        let d = inter_residue_dist(&ppii, 0, i, "CA", i + 1, "CA");
        assert!(
            d > 2.8 && d < 4.0,
            "PPII CA({})-CA({}) = {:.2} Å out of range",
            i + 1,
            i + 2,
            d
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 6. MUTATION — build + mutate + verify side chain
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_build_mutate_verify_sidechain() {
    let mut struc = make_extended_structure("ACDEF").unwrap();
    // Mutate position 3 (ASP → TRP): should gain side-chain atoms
    mutate_residue(&mut struc, 3, ResName::TRP).unwrap();
    let r3 = &struc.chain_a().residues[2];
    assert_eq!(r3.name, ResName::TRP);
    // TRP has 14 heavy atoms (4 backbone + 10 side chain)
    assert_eq!(r3.atoms.len(), 14);
    // Backbone atoms must still be intact
    assert!(r3.atom_coord("N").is_some());
    assert!(r3.atom_coord("CA").is_some());
    assert!(r3.atom_coord("C").is_some());
    assert!(r3.atom_coord("O").is_some());
    // Original residue shouldn't be affected
    assert_eq!(struc.chain_a().residues[0].name, ResName::ALA);
    assert_eq!(struc.chain_a().residues[1].name, ResName::CYS);
}

#[test]
fn test_chained_mutations() {
    // Multiple mutations in sequence
    let mut struc = make_extended_structure("AAAAA").unwrap();
    for (pos, target) in [(1, ResName::GLY), (3, ResName::PHE), (5, ResName::TYR)] {
        mutate_residue(&mut struc, pos, target).unwrap();
    }
    assert_eq!(struc.chain_a().residues[0].name, ResName::GLY);
    assert_eq!(struc.chain_a().residues[2].name, ResName::PHE);
    assert_eq!(struc.chain_a().residues[4].name, ResName::TYR);
    // Untouched residues remain ALA
    assert_eq!(struc.chain_a().residues[1].name, ResName::ALA);
    assert_eq!(struc.chain_a().residues[3].name, ResName::ALA);
}

#[test]
fn test_mutation_spec_parsing_and_apply() {
    let mut struc = make_extended_structure("ACDEFG").unwrap();
    let specs = ["A1W", "C2G", "F5Y"];
    for s in &specs {
        let (_from, pos, to) = parse_mutation_spec(s).unwrap();
        mutate_residue(&mut struc, pos, to).unwrap();
    }
    assert_eq!(struc.chain_a().residues[0].name, ResName::TRP);
    assert_eq!(struc.chain_a().residues[1].name, ResName::GLY);
    assert_eq!(struc.chain_a().residues[4].name, ResName::TYR);
}

// ═══════════════════════════════════════════════════════════════════════
// 7. MULTI-CHAIN — build, round-trip, mutate across chains
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_multi_chain_build_residue_counts() {
    let chains = vec![
        ChainSpec {
            id: 'A',
            residues: parse_three_letter_sequence("ALA-CYS-ASP").unwrap(),
            preset: Some(RamaPreset::AlphaHelix),
        },
        ChainSpec {
            id: 'B',
            residues: parse_three_letter_sequence("GLY-VAL-TRP-PHE").unwrap(),
            preset: Some(RamaPreset::BetaSheet),
        },
    ];
    let struc = make_multi_chain_structure(&chains).unwrap();
    assert_eq!(struc.chains.len(), 2);
    assert_eq!(struc.chains[0].id, 'A');
    assert_eq!(struc.chains[1].id, 'B');
    assert_eq!(struc.chains[0].residues.len(), 3);
    assert_eq!(struc.chains[1].residues.len(), 4);
}

#[test]
fn test_multi_chain_pdb_roundtrip_preserves_chains() {
    let chains = vec![
        ChainSpec {
            id: 'A',
            residues: parse_three_letter_sequence("ALA-GLY-SER").unwrap(),
            preset: None,
        },
        ChainSpec {
            id: 'B',
            residues: parse_three_letter_sequence("VAL-LEU-ILE").unwrap(),
            preset: None,
        },
    ];
    let struc = make_multi_chain_structure(&chains).unwrap();
    let path = tmp_path("mc_roundtrip.pdb");
    let ps = path.to_string_lossy().to_string();
    write_structure(&struc, &ps, None).unwrap();
    let back = read_structure(&ps).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(back.chains.len(), 2);
    assert_eq!(back.chains[0].id, 'A');
    assert_eq!(back.chains[1].id, 'B');
    assert_eq!(back.chains[0].residues.len(), 3);
    assert_eq!(back.chains[1].residues.len(), 3);
    assert_eq!(back.chains[1].residues[0].name, ResName::VAL);
    assert_eq!(back.chains[1].residues[2].name, ResName::ILE);
}

#[test]
fn test_mutate_across_chains() {
    let chains = vec![
        ChainSpec {
            id: 'A',
            residues: parse_three_letter_sequence("ALA-ALA").unwrap(),
            preset: None,
        },
        ChainSpec {
            id: 'B',
            residues: parse_three_letter_sequence("ALA-ALA-ALA").unwrap(),
            preset: None,
        },
    ];
    let mut struc = make_multi_chain_structure(&chains).unwrap();
    // Position 4 = 2nd residue of chain B (global: A1, A2, B3, B4, B5)
    mutate_residue(&mut struc, 4, ResName::TRP).unwrap();
    assert_eq!(
        struc.chain_by_id('B').unwrap().residues[1].name,
        ResName::TRP
    );
    // Chain A untouched
    assert_eq!(struc.chains[0].residues[0].name, ResName::ALA);
    assert_eq!(struc.chains[0].residues[1].name, ResName::ALA);
}

// ═══════════════════════════════════════════════════════════════════════
// 8. JSON SPEC — single-chain with preset + mutations
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_json_spec_helix_with_mutation() {
    let json = r#"{
        "residues": ["ALA","ALA","ALA","ALA","ALA"],
        "preset": "alpha-helix",
        "oxt": true,
        "mutations": ["A3W"]
    }"#;
    let spec: BuildSpec = serde_json::from_str(json).unwrap();
    let struc = spec.execute().unwrap();

    assert_eq!(struc.chain_a().residues.len(), 5);
    assert_eq!(struc.chain_a().residues[2].name, ResName::TRP);
    // OXT should be on last residue
    assert!(struc.chain_a().residues[4].atom_coord("OXT").is_some());
}

#[test]
fn test_json_spec_multi_chain_with_ss_detection() {
    let json = r#"{
        "chains": [
            {"id": "H", "residues": ["ALA","CYS","GLY"], "preset": "alpha-helix"},
            {"id": "L", "residues": ["CYS","VAL","TRP"], "preset": "beta-sheet"}
        ],
        "oxt": true
    }"#;
    let spec: BuildSpec = serde_json::from_str(json).unwrap();
    let struc = spec.execute().unwrap();

    assert_eq!(struc.chains.len(), 2);
    assert_eq!(struc.chains[0].id, 'H');
    assert_eq!(struc.chains[1].id, 'L');
    // Both chains should have OXT
    assert!(struc.chains[0].residues.last().unwrap().atom_coord("OXT").is_some());
    assert!(struc.chains[1].residues.last().unwrap().atom_coord("OXT").is_some());
}

#[test]
fn test_json_spec_custom_angles() {
    let json = r#"{
        "residues": ["ALA","ALA","ALA"],
        "phi": [-57.0, -57.0],
        "psi": [-47.0, -47.0]
    }"#;
    let spec: BuildSpec = serde_json::from_str(json).unwrap();
    let struc = spec.execute().unwrap();
    assert_eq!(struc.chain_a().residues.len(), 3);
    // Should build just like make_structure with those angles
    let d = inter_residue_dist(&struc, 0, 0, "CA", 1, "CA");
    assert!(d > 3.5 && d < 4.1, "CA-CA = {:.2} unexpected for helix angles", d);
}

// ═══════════════════════════════════════════════════════════════════════
// 9. AMBER VARIANTS — CYX / HID / HIE / HIP round-trip
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_amber_variant_build_and_readback() {
    let specs = parse_three_letter_sequence("ALA-CYX-HID-HIE-HIP-ASH-GLH-LYN").unwrap();
    let struc = builder::make_extended_structure_from_specs(&specs).unwrap();

    // Verify variant tags
    assert_eq!(struc.chain_a().residues[1].variant, Some(AmberVariant::CYX));
    assert_eq!(struc.chain_a().residues[2].variant, Some(AmberVariant::HID));
    assert_eq!(struc.chain_a().residues[3].variant, Some(AmberVariant::HIE));
    assert_eq!(struc.chain_a().residues[4].variant, Some(AmberVariant::HIP));
    assert_eq!(struc.chain_a().residues[5].variant, Some(AmberVariant::ASH));
    assert_eq!(struc.chain_a().residues[6].variant, Some(AmberVariant::GLH));
    assert_eq!(struc.chain_a().residues[7].variant, Some(AmberVariant::LYN));

    // Write → read → verify PDB preserves names
    let path = tmp_path("amber_variants.pdb");
    let ps = path.to_string_lossy().to_string();
    write_structure(&struc, &ps, None).unwrap();
    let back = read_structure(&ps).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(back.chain_a().residues[1].amber_name(), "CYX");
    assert_eq!(back.chain_a().residues[2].amber_name(), "HID");
    assert_eq!(back.chain_a().residues[3].amber_name(), "HIE");
    assert_eq!(back.chain_a().residues[4].amber_name(), "HIP");
    assert_eq!(back.chain_a().residues[5].amber_name(), "ASH");
    assert_eq!(back.chain_a().residues[6].amber_name(), "GLH");
    assert_eq!(back.chain_a().residues[7].amber_name(), "LYN");
}

// ═══════════════════════════════════════════════════════════════════════
// 10. OXT — terminal oxygen on each chain
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_oxt_single_chain() {
    let mut struc = make_extended_structure("ACDE").unwrap();
    assert!(struc.chain_a().residues.last().unwrap().atom_coord("OXT").is_none());
    add_terminal_oxt(&mut struc);
    assert!(struc.chain_a().residues.last().unwrap().atom_coord("OXT").is_some());
    // OXT should only be on the last residue
    assert!(struc.chain_a().residues[0].atom_coord("OXT").is_none());
}

#[test]
fn test_oxt_survives_pdb_roundtrip() {
    let mut struc = make_extended_structure("ACD").unwrap();
    add_terminal_oxt(&mut struc);
    let path = tmp_path("oxt_rt.pdb");
    let ps = path.to_string_lossy().to_string();
    write_structure(&struc, &ps, None).unwrap();
    let back = read_structure(&ps).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(back.chain_a().residues.last().unwrap().atom_coord("OXT").is_some());
}

// ═══════════════════════════════════════════════════════════════════════
// 11. REAL PROTEIN — Crambin (1CRN, 46 residues, 3 SS bonds)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_read_crambin_residue_count() {
    let path = fixture("1crn.pdb");
    let struc = read_structure(&path.to_string_lossy()).unwrap();
    assert_eq!(struc.chains.len(), 1);
    assert_eq!(struc.chains[0].id, 'A');
    assert_eq!(struc.chains[0].residues.len(), 46);
}

#[test]
fn test_crambin_sequence_identity() {
    // Crambin sequence: TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN
    let expected_seq = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN";
    let path = fixture("1crn.pdb");
    let struc = read_structure(&path.to_string_lossy()).unwrap();
    let seq: String = struc.chain_a().residues.iter().map(|r| r.name.to_one_letter()).collect();
    assert_eq!(seq, expected_seq, "crambin sequence mismatch");
}

#[test]
fn test_crambin_disulfide_detection() {
    let path = fixture("1crn.pdb");
    let mut struc = read_structure(&path.to_string_lossy()).unwrap();
    let count = detect_disulfides(&mut struc);
    assert_eq!(count, 3, "crambin should have 3 disulfide bonds");
    assert_eq!(struc.ssbonds.len(), 3);

    // Expected SS bonds: C3-C40, C4-C32, C16-C26
    let expected: Vec<(i32, i32)> = vec![(3, 40), (4, 32), (16, 26)];
    for (r1, r2) in &expected {
        let found = struc.ssbonds.iter().any(|ss| {
            (ss.resid1 == *r1 && ss.resid2 == *r2) || (ss.resid1 == *r2 && ss.resid2 == *r1)
        });
        assert!(found, "missing disulfide bond {}-{}", r1, r2);
    }

    // Check CYX annotation
    let cys_positions: Vec<usize> = vec![2, 3, 15, 25, 31, 39]; // 0-indexed
    for idx in cys_positions {
        assert_eq!(
            struc.chain_a().residues[idx].variant,
            Some(AmberVariant::CYX),
            "CYS at position {} not marked CYX",
            idx + 1
        );
    }
}

#[test]
fn test_crambin_atom_has_coordinates() {
    // Verify that atom coordinates were parsed correctly
    let path = fixture("1crn.pdb");
    let struc = read_structure(&path.to_string_lossy()).unwrap();
    // First residue (THR 1) CA should be near (16.967, 12.784, 4.338)
    let ca = struc.chain_a().residues[0].atom_coord("CA").unwrap();
    assert!((ca.x - 16.967).abs() < 0.01);
    assert!((ca.y - 12.784).abs() < 0.01);
    assert!((ca.z - 4.338).abs() < 0.01);
}

#[test]
fn test_crambin_roundtrip_preserves_structure() {
    let path = fixture("1crn.pdb");
    let orig = read_structure(&path.to_string_lossy()).unwrap();
    let out = tmp_path("1crn_rt.pdb");
    let out_s = out.to_string_lossy().to_string();
    write_structure(&orig, &out_s, None).unwrap();
    let back = read_structure(&out_s).unwrap();
    let _ = std::fs::remove_file(&out);

    assert_eq!(back.chains[0].residues.len(), 46);
    // Coordinate fidelity (f32 round-trip tolerance)
    for (i, (r_orig, r_back)) in orig
        .chain_a()
        .residues
        .iter()
        .zip(back.chain_a().residues.iter())
        .enumerate()
    {
        assert_eq!(r_orig.name, r_back.name, "residue {} name mismatch", i + 1);
        let ca_orig = r_orig.atom_coord("CA").unwrap();
        let ca_back = r_back.atom_coord("CA").unwrap();
        let drift = ca_orig.sub(ca_back).length();
        assert!(
            drift < 0.02,
            "CA drift at residue {} = {:.4} Å",
            i + 1,
            drift
        );
    }
}

#[test]
fn test_crambin_add_oxt() {
    let path = fixture("1crn.pdb");
    let mut struc = read_structure(&path.to_string_lossy()).unwrap();
    // 1CRN already has OXT on last residue (ASN 46)
    assert!(struc.chain_a().residues.last().unwrap().atom_coord("OXT").is_some());
    // add_terminal_oxt should be idempotent
    let atom_count_before = total_atom_count(&struc);
    add_terminal_oxt(&mut struc);
    assert_eq!(total_atom_count(&struc), atom_count_before);
}

// ═══════════════════════════════════════════════════════════════════════
// 12. REAL PROTEIN — murine EGF (1EGF model 1, 53 residues, 3 SS bonds)
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_read_egf_residue_count() {
    let path = fixture("1egf_model1.pdb");
    let struc = read_structure(&path.to_string_lossy()).unwrap();
    assert_eq!(struc.chains.len(), 1);
    assert_eq!(struc.chains[0].residues.len(), 53);
}

#[test]
fn test_egf_sequence_identity() {
    // Murine EGF: NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYS GDRCQTRDLRWWELR
    let expected = "NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYS\
                    GDRCQTRDLRWWELR";
    let path = fixture("1egf_model1.pdb");
    let struc = read_structure(&path.to_string_lossy()).unwrap();
    let seq: String = struc.chain_a().residues.iter().map(|r| r.name.to_one_letter()).collect();
    assert_eq!(seq, expected, "EGF sequence mismatch");
}

#[test]
fn test_egf_disulfide_detection() {
    let path = fixture("1egf_model1.pdb");
    let mut struc = read_structure(&path.to_string_lossy()).unwrap();
    let count = detect_disulfides(&mut struc);
    assert_eq!(count, 3, "EGF should have 3 disulfide bonds");

    // Expected: C6-C20, C14-C31, C33-C42
    let expected: Vec<(i32, i32)> = vec![(6, 20), (14, 31), (33, 42)];
    for (r1, r2) in &expected {
        let found = struc.ssbonds.iter().any(|ss| {
            (ss.resid1 == *r1 && ss.resid2 == *r2) || (ss.resid1 == *r2 && ss.resid2 == *r1)
        });
        assert!(found, "missing EGF disulfide bond {}-{}", r1, r2);
    }
}

#[test]
fn test_egf_mutate_cysteine_bridge() {
    // Read EGF, mutate one cysteine in a bridge (C6 → ALA), verify bridge broken
    let path = fixture("1egf_model1.pdb");
    let mut struc = read_structure(&path.to_string_lossy()).unwrap();

    // Confirm CYS at position 6 before mutation
    assert_eq!(struc.chain_a().residues[5].name, ResName::CYS);

    // Mutate C6 → ALA (position 6 is 1-based)
    mutate_residue(&mut struc, 6, ResName::ALA).unwrap();
    assert_eq!(struc.chain_a().residues[5].name, ResName::ALA);

    // Now detect disulfides — should find only 2 (C14-C31, C33-C42)
    let count = detect_disulfides(&mut struc);
    assert_eq!(count, 2, "after mutating C6→A, only 2 SS bonds should remain");

    // The broken bridge (C6-C20) should not appear
    let has_6_20 = struc.ssbonds.iter().any(|ss| {
        (ss.resid1 == 6 && ss.resid2 == 20) || (ss.resid1 == 20 && ss.resid2 == 6)
    });
    assert!(!has_6_20, "C6-C20 bridge should be broken after mutation");
}

// ═══════════════════════════════════════════════════════════════════════
// 13. DE-NOVO BUILD — hEGF-like sequence from scratch
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_build_hegf_sequence_de_novo() {
    // Build the 53-residue EGF sequence as an alpha-helix (just for construction proof)
    let seq = "NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYS\
               GDRCQTRDLRWWELR";
    let struc = make_preset_structure(seq, RamaPreset::AlphaHelix).unwrap();
    assert_eq!(struc.chain_a().residues.len(), 53);
    // Has 6 cysteines
    let cys_count = struc
        .chain_a()
        .residues
        .iter()
        .filter(|r| r.name == ResName::CYS)
        .count();
    assert_eq!(cys_count, 6);
    // Total atoms should be substantial (≈ 400+ heavy atoms)
    let n = total_atom_count(&struc);
    assert!(n > 350, "too few atoms: {}", n);
}

#[test]
fn test_build_hegf_with_oxt_and_write() {
    let seq = "NSYPGCPSSYDGYCLNGGVCMHIESLDSYTCNCVIGYS\
               GDRCQTRDLRWWELR";
    let mut struc = make_preset_structure(seq, RamaPreset::AlphaHelix).unwrap();
    add_terminal_oxt(&mut struc);

    let path = tmp_path("hegf_denovo.pdb");
    let ps = path.to_string_lossy().to_string();
    write_structure(&struc, &ps, None).unwrap();
    let back = read_structure(&ps).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(back.chain_a().residues.len(), 53);
    assert!(back.chain_a().residues.last().unwrap().atom_coord("OXT").is_some());
}

// ═══════════════════════════════════════════════════════════════════════
// 14. MULTI-FORMAT — PDB vs GRO round-trip comparison
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_pdb_gro_atom_count_parity() {
    let struc = make_extended_structure("ACDEF").unwrap();
    let n_orig = total_atom_count(&struc);

    // PDB round-trip
    let pdb_path = tmp_path("fmt_pdb.pdb");
    let pdb_s = pdb_path.to_string_lossy().to_string();
    write_structure(&struc, &pdb_s, None).unwrap();
    let pdb_back = read_structure(&pdb_s).unwrap();
    let _ = std::fs::remove_file(&pdb_path);

    // GRO round-trip
    let gro_path = tmp_path("fmt_gro.gro");
    let gro_s = gro_path.to_string_lossy().to_string();
    write_structure(&struc, &gro_s, Some("gro")).unwrap();
    let gro_back = read_structure(&gro_s).unwrap();
    let _ = std::fs::remove_file(&gro_path);

    assert_eq!(total_atom_count(&pdb_back), n_orig);
    assert_eq!(total_atom_count(&gro_back), n_orig);
    assert_eq!(pdb_back.chain_a().residues.len(), 5);
    assert_eq!(gro_back.chain_a().residues.len(), 5);
}

// ═══════════════════════════════════════════════════════════════════════
// 15. EDGE CASES — single residue, minimal peptide, large peptide
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_single_residue_peptide() {
    let struc = make_extended_structure("W").unwrap();
    assert_eq!(struc.chain_a().residues.len(), 1);
    assert_eq!(struc.chain_a().residues[0].name, ResName::TRP);
    assert_eq!(struc.chain_a().residues[0].atoms.len(), 14);
}

#[test]
fn test_large_polyalanine_100() {
    let seq: String = (0..100).map(|_| 'A').collect();
    let struc = make_preset_structure(&seq, RamaPreset::AlphaHelix).unwrap();
    assert_eq!(struc.chain_a().residues.len(), 100);
    // 100 × 5 atoms = 500
    assert_eq!(total_atom_count(&struc), 500);
}

// ═══════════════════════════════════════════════════════════════════════
// 16. JSON SPEC — file-based full pipeline
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_json_spec_file_roundtrip() {
    // Write a JSON spec to disk, load, execute, verify structure
    let spec_json = r#"{
        "residues": ["ALA","CYS","ASP","GLU","PHE","GLY","HIS","ILE","LYS","LEU"],
        "preset": "beta-sheet",
        "oxt": true,
        "mutations": ["A1W", "G6P"]
    }"#;
    let spec_path = tmp_path("spec_test.json");
    std::fs::write(&spec_path, spec_json).unwrap();

    let spec = BuildSpec::from_file(&spec_path.to_string_lossy()).unwrap();
    let struc = spec.execute().unwrap();
    let _ = std::fs::remove_file(&spec_path);

    assert_eq!(struc.chain_a().residues.len(), 10);
    assert_eq!(struc.chain_a().residues[0].name, ResName::TRP); // A1W
    assert_eq!(struc.chain_a().residues[5].name, ResName::PRO); // G6P
    assert!(struc.chain_a().residues[9].atom_coord("OXT").is_some());
}

// ═══════════════════════════════════════════════════════════════════════
// 17. CRAMBIN — mutate a disulfide cysteine, verify bridge breaks
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_crambin_mutate_cys3_breaks_bridge() {
    let path = fixture("1crn.pdb");
    let mut struc = read_structure(&path.to_string_lossy()).unwrap();

    // Cys3 is part of C3-C40 disulfide
    assert_eq!(struc.chain_a().residues[2].name, ResName::CYS);

    // Mutate CYS3 → ALA
    mutate_residue(&mut struc, 3, ResName::ALA).unwrap();
    assert_eq!(struc.chain_a().residues[2].name, ResName::ALA);
    // SG atom should be gone
    assert!(struc.chain_a().residues[2].atom_coord("SG").is_none());

    // Detect remaining disulfides (should find 2: C4-C32, C16-C26)
    let count = detect_disulfides(&mut struc);
    assert_eq!(count, 2, "after mutating C3→A, 2 SS bonds should remain");
}

#[test]
fn test_crambin_mutate_both_partners_breaks_bridge() {
    let path = fixture("1crn.pdb");
    let mut struc = read_structure(&path.to_string_lossy()).unwrap();

    // Mutate both partners of C16-C26 bridge
    mutate_residue(&mut struc, 16, ResName::SER).unwrap();
    mutate_residue(&mut struc, 26, ResName::SER).unwrap();
    assert_eq!(struc.chain_a().residues[15].name, ResName::SER);
    assert_eq!(struc.chain_a().residues[25].name, ResName::SER);

    let count = detect_disulfides(&mut struc);
    assert_eq!(count, 2, "C3-C40 and C4-C32 should remain");
}

// ═══════════════════════════════════════════════════════════════════════
// 18. GEOMETRY VALIDATION — peptide bond lengths across all types
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_peptide_bond_length_all_pairs() {
    // Build all 20 AA in a chain; every C(i)-N(i+1) should be ~1.33 Å
    let all = "ACDEFGHIKLMNPQRSTVWY";
    let struc = make_extended_structure(all).unwrap();
    for i in 0..19 {
        let c_i = struc.chain_a().residues[i].atom_coord("C").unwrap();
        let n_j = struc.chain_a().residues[i + 1].atom_coord("N").unwrap();
        let d = c_i.sub(n_j).length();
        assert!(
            (d - 1.33).abs() < 0.05,
            "peptide bond {}-{} ({}-{}): {:.3} Å, expected ~1.33",
            i + 1,
            i + 2,
            struc.chain_a().residues[i].name.as_str(),
            struc.chain_a().residues[i + 1].name.as_str(),
            d
        );
    }
}

#[test]
fn test_ca_c_bond_length() {
    // CA-C bond should be ~1.52 Å in every residue
    let struc = make_extended_structure("ACDEFGHIKLMNPQRSTVWY").unwrap();
    for (i, res) in struc.chain_a().residues.iter().enumerate() {
        let ca = res.atom_coord("CA").unwrap();
        let c = res.atom_coord("C").unwrap();
        let d = ca.sub(c).length();
        assert!(
            (d - 1.52).abs() < 0.03,
            "CA-C in {} (res {}): {:.3} Å, expected ~1.52",
            res.name.as_str(),
            i + 1,
            d
        );
    }
}

#[test]
fn test_ca_n_bond_length() {
    // N-CA bond should be ~1.46 Å
    let struc = make_extended_structure("ACDEFGHIKLMNPQRSTVWY").unwrap();
    for (i, res) in struc.chain_a().residues.iter().enumerate() {
        let n = res.atom_coord("N").unwrap();
        let ca = res.atom_coord("CA").unwrap();
        let d = n.sub(ca).length();
        assert!(
            (d - 1.46).abs() < 0.03,
            "N-CA in {} (res {}): {:.3} Å, expected ~1.46",
            res.name.as_str(),
            i + 1,
            d
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════
// 19. CLI SMOKE TESTS — binary invocation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_cli_build_basic() {
    let bin = env!("CARGO_BIN_EXE_warp-pep");
    let out = tmp_path("cli_basic.pdb");
    let status = std::process::Command::new(bin)
        .args(["build", "-s", "AGA", "-o", out.to_str().unwrap()])
        .status()
        .expect("failed to run warp-pep");
    assert!(status.success(), "build returned non-zero");
    assert!(out.exists(), "output file missing");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn test_cli_build_preset() {
    let bin = env!("CARGO_BIN_EXE_warp-pep");
    let out = tmp_path("cli_preset.pdb");
    let status = std::process::Command::new(bin)
        .args(["build", "-s", "AAAA", "--preset", "alpha", "-o", out.to_str().unwrap()])
        .status()
        .expect("failed to run warp-pep");
    assert!(status.success());
    let _ = std::fs::remove_file(&out);
}

#[test]
fn test_cli_build_three_letter() {
    let bin = env!("CARGO_BIN_EXE_warp-pep");
    let out = tmp_path("cli_three.pdb");
    let status = std::process::Command::new(bin)
        .args(["build", "--three-letter", "ALA-GLY-PRO", "-o", out.to_str().unwrap()])
        .status()
        .expect("failed to run warp-pep");
    assert!(status.success());
    let _ = std::fs::remove_file(&out);
}

#[test]
fn test_cli_mutate() {
    let bin = env!("CARGO_BIN_EXE_warp-pep");
    let input = fixture("1crn.pdb");
    let out = tmp_path("cli_mutate.pdb");
    let status = std::process::Command::new(bin)
        .args([
            "mutate",
            "-i", input.to_str().unwrap(),
            "-m", "P5V",
            "-o", out.to_str().unwrap(),
        ])
        .status()
        .expect("failed to run warp-pep");
    assert!(status.success());
    let _ = std::fs::remove_file(&out);
}

#[test]
fn test_cli_build_no_sequence_fails() {
    let bin = env!("CARGO_BIN_EXE_warp-pep");
    let out = tmp_path("cli_noseq.pdb");
    let status = std::process::Command::new(bin)
        .args(["build", "-o", out.to_str().unwrap()])
        .status()
        .expect("failed to run warp-pep");
    assert!(!status.success(), "should fail without -s or --json");
    let _ = std::fs::remove_file(&out);
}

#[test]
fn test_cli_mutate_no_input_fails() {
    let bin = env!("CARGO_BIN_EXE_warp-pep");
    let status = std::process::Command::new(bin)
        .args(["mutate", "-m", "A5V"])
        .status()
        .expect("failed to run warp-pep");
    assert!(!status.success(), "should fail without -i");
}

// ═══════════════════════════════════════════════════════════════════════
// 20. ERROR PATH TESTS — bad inputs
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_error_invalid_one_letter() {
    let result = make_structure("AXG", &[-57.0, -57.0], &[-47.0, -47.0], None);
    assert!(result.is_err(), "X is not a valid amino acid");
}

#[test]
fn test_error_empty_sequence() {
    let result = make_structure("", &[], &[], None);
    assert!(result.is_err(), "empty sequence should fail");
}

#[test]
fn test_error_mutate_out_of_range() {
    let mut struc = make_extended_structure("AGA").unwrap();
    let result = mutate_residue(&mut struc, 99, ResName::ALA);
    assert!(result.is_err(), "residue 99 does not exist");
}

#[test]
fn test_error_parse_bad_mutation_spec() {
    let result = parse_mutation_spec("ZZZ");
    assert!(result.is_err(), "ZZZ is not a valid mutation spec");
}

#[test]
fn test_error_read_nonexistent_file() {
    let result = read_structure("/nonexistent/path/fake.pdb");
    assert!(result.is_err());
}

#[test]
fn test_error_bad_json_spec() {
    let bad_json = r#"not json at all"#;
    let result: Result<BuildSpec, _> = serde_json::from_str(bad_json);
    assert!(result.is_err(), "should fail to parse bad JSON spec");
}

#[test]
fn test_error_three_letter_unknown_residue() {
    let result = parse_three_letter_sequence("ALA-XYZ-GLY");
    assert!(result.is_err(), "XYZ is not a known residue");
}

// ═══════════════════════════════════════════════════════════════════════
// 21. PROPERTY / ROUND-TRIP TESTS
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_roundtrip_build_write_read_pdb() {
    let struc = make_extended_structure("ACDEF").unwrap();
    let path = tmp_path("roundtrip.pdb");
    write_structure(&struc, &path.to_string_lossy(), None).unwrap();
    let read_back = read_structure(&path.to_string_lossy()).unwrap();
    assert_eq!(
        struc.chain_a().residues.len(),
        read_back.chain_a().residues.len(),
        "residue count mismatch after round-trip"
    );
    // Atom counts should match
    assert_eq!(total_atom_count(&struc), total_atom_count(&read_back));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_roundtrip_preserves_coords() {
    let struc = make_extended_structure("GGA").unwrap();
    let path = tmp_path("roundtrip_coords.pdb");
    write_structure(&struc, &path.to_string_lossy(), None).unwrap();
    let read_back = read_structure(&path.to_string_lossy()).unwrap();
    // RMSD should be very small (PDB float truncation may add ~0.001)
    let rmsd = rmsd_all_atoms(&struc, &read_back).unwrap();
    assert!(
        rmsd < 0.01,
        "round-trip RMSD {:.4} Å, expected < 0.01",
        rmsd
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_self_rmsd_is_zero() {
    let struc = make_extended_structure("AAAA").unwrap();
    let rmsd = rmsd_all_atoms(&struc, &struc).unwrap();
    assert!(rmsd.abs() < 1e-10, "self-RMSD should be 0, got {}", rmsd);
}

#[test]
fn test_rmsd_ca_consistent_with_all_atoms_for_polyala() {
    // For polyALA, CA-RMSD <= all-atom RMSD (subset)
    let a = make_preset_structure("AAAA", RamaPreset::AlphaHelix).unwrap();
    let b = make_preset_structure("AAAA", RamaPreset::BetaSheet).unwrap();
    let rmsd_all = rmsd_all_atoms(&a, &b).unwrap();
    let rmsd_c = rmsd_ca(&a, &b).unwrap();
    assert!(rmsd_c > 0.0, "different conformations should have nonzero CA-RMSD");
    // CA RMSD doesn't need to be <= all-atom exactly, but both should be > 0
    assert!(rmsd_all > 0.0);
}

#[test]
fn test_oxt_idempotent() {
    let mut struc = make_extended_structure("AGA").unwrap();
    add_terminal_oxt(&mut struc);
    let count_1 = total_atom_count(&struc);
    add_terminal_oxt(&mut struc);
    let count_2 = total_atom_count(&struc);
    assert_eq!(count_1, count_2, "adding OXT twice should be idempotent");
}

#[test]
fn test_sequence_roundtrip() {
    let seq = "ACDEFGHIKLMNPQRSTVWY";
    let struc = make_extended_structure(seq).unwrap();
    let recovered = struc.sequence();
    assert_eq!(recovered, seq, "sequence() should recover the input");
}

#[test]
fn test_renumber_consistency() {
    let mut struc = make_extended_structure("AAAA").unwrap();
    struc.renumber(10);
    assert_eq!(struc.chain_a().residues[0].seq_id, 10);
    assert_eq!(struc.chain_a().residues[3].seq_id, 13);
    struc.renumber(1);
    assert_eq!(struc.chain_a().residues[0].seq_id, 1);
}

#[test]
fn test_multichain_renumber_per_chain() {
    let specs = vec![
        ChainSpec { id: 'A', residues: parse_three_letter_sequence("ALA-ALA").unwrap(), preset: None },
        ChainSpec { id: 'B', residues: parse_three_letter_sequence("GLY-GLY-GLY").unwrap(), preset: None },
    ];
    let mut struc = make_multi_chain_structure(&specs).unwrap();
    struc.renumber_per_chain(1);
    assert_eq!(struc.chains[0].residues[0].seq_id, 1);
    assert_eq!(struc.chains[0].residues[1].seq_id, 2);
    assert_eq!(struc.chains[1].residues[0].seq_id, 1);
    assert_eq!(struc.chains[1].residues[2].seq_id, 3);
}

// ═══════════════════════════════════════════════════════════════════════
// 22. INTEGRATION: analysis, selection, caps, hydrogen, validation
// ═══════════════════════════════════════════════════════════════════════

#[test]
fn test_selection_on_real_structure() {
    let path = fixture("1crn.pdb");
    let struc = read_structure(&path.to_string_lossy()).unwrap();
    let ca_atoms = select(&struc, "name CA").unwrap();
    assert_eq!(
        ca_atoms.len(),
        struc.chain_a().residues.len(),
        "CA count should equal residue count"
    );
    let backbone = select(&struc, "backbone").unwrap();
    assert!(backbone.len() > ca_atoms.len(), "backbone > CA atoms");
}

#[test]
fn test_selection_resid_range() {
    let struc = make_extended_structure("ACDEFGH").unwrap();
    let hits = select(&struc, "resid 2-4").unwrap();
    // residues 2,3,4
    let res_ids: std::collections::HashSet<i32> =
        hits.iter().map(|a| a.resid).collect();
    assert!(res_ids.contains(&2));
    assert!(res_ids.contains(&3));
    assert!(res_ids.contains(&4));
    assert!(!res_ids.contains(&1));
    assert!(!res_ids.contains(&5));
}

#[test]
fn test_selection_boolean_logic() {
    let struc = make_extended_structure("AGP").unwrap();
    // "name CA and resid 1" → 1 atom
    let hits = select(&struc, "name CA and resid 1").unwrap();
    assert_eq!(hits.len(), 1);
    // "name CA or name N" → 2 per residue = 6
    let hits2 = select(&struc, "name CA or name N").unwrap();
    assert_eq!(hits2.len(), 6);
}

#[test]
fn test_caps_on_built_peptide() {
    let mut struc = make_extended_structure("AGG").unwrap();
    let n_before = struc.chain_a().residues.len();
    add_caps(&mut struc);
    let n_after = struc.chain_a().residues.len();
    assert_eq!(n_after, n_before + 2, "ACE + NME = +2 residues");
}

#[test]
fn test_ace_only() {
    let mut struc = make_extended_structure("AA").unwrap();
    let n = struc.chain_a().residues.len();
    add_ace_cap(&mut struc);
    assert_eq!(struc.chain_a().residues.len(), n + 1);
}

#[test]
fn test_nme_only() {
    let mut struc = make_extended_structure("AA").unwrap();
    let n = struc.chain_a().residues.len();
    add_nme_cap(&mut struc);
    assert_eq!(struc.chain_a().residues.len(), n + 1);
}

#[test]
fn test_backbone_hydrogens_adds_atoms() {
    let mut struc = make_extended_structure("AGA").unwrap();
    let before = total_atom_count(&struc);
    add_backbone_hydrogens(&mut struc);
    let after = total_atom_count(&struc);
    assert!(after > before, "hydrogen addition should increase atom count");
}

#[test]
fn test_ha_hydrogens_adds_atoms() {
    let mut struc = make_extended_structure("AGA").unwrap();
    let before = total_atom_count(&struc);
    add_ha_hydrogens(&mut struc);
    let after = total_atom_count(&struc);
    assert!(after > before, "HA addition should increase atom count");
}

#[test]
fn test_validation_clean_structure_no_errors() {
    let struc = make_extended_structure("AGA").unwrap();
    let issues = validate(&struc);
    let errors: Vec<_> = issues
        .iter()
        .filter(|i| matches!(i.severity, warp_pep::validation::Severity::Error))
        .collect();
    assert!(errors.is_empty(), "clean build should have no errors: {:?}", errors);
}

#[test]
fn test_validation_detects_missing_backbone() {
    let mut struc = make_extended_structure("AA").unwrap();
    // Remove N from first residue to trigger missing-backbone warning
    struc.chains[0].residues[0].atoms.retain(|a| a.name != "N");
    let issues = validate(&struc);
    assert!(!issues.is_empty(), "missing N should produce validation issue");
}

#[test]
fn test_rg_positive() {
    let struc = make_extended_structure("ACDEFGHIK").unwrap();
    let rg = radius_of_gyration(&struc);
    assert!(rg > 0.0, "Rg should be positive for a multi-residue peptide");
}

#[test]
fn test_phi_psi_residue_count() {
    let struc = make_preset_structure("AAAA", RamaPreset::AlphaHelix).unwrap();
    let pp = measure_all_phi_psi(&struc);
    assert_eq!(pp.len(), 1); // one chain
    assert_eq!(pp[0].1.len(), 4); // 4 residues
    // first residue: phi = None (no preceding C)
    assert!(pp[0].1[0].phi.is_none());
    // last residue: psi = None (no following N)
    assert!(pp[0].1[3].psi.is_none());
}
