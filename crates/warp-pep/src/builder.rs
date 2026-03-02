//! Peptide chain builder: initialize first residue, then extend.
//!
//! Rust port of PeptideBuilder.PeptideBuilder (clauswilke, MIT).

use crate::coord::{calculate_coordinates, Vec3};
use crate::d_amino::{d_geometry, parse_d_amino_acid};
use crate::geometry::{self, Geo};
use crate::non_standard::{self, NonStdResidue};
use crate::residue::{parse_amber_name, AmberVariant, Atom, Chain, ResName, Residue, Structure};

/// Build the side‐chain atoms for a residue, given backbone atoms already placed.
fn build_side_chain(res: &mut Residue, geo: &Geo) {
    for sc in &geo.side_chain {
        let a = find_coord(res, sc.parents.0);
        let b = find_coord(res, sc.parents.1);
        let c = find_coord(res, sc.parents.2);
        let pos = calculate_coordinates(a, b, c, sc.length, sc.angle, sc.dihedral);
        res.add_atom(Atom::new(sc.name, sc.element, pos));
    }
}

/// Look up an atom's coordinate by name within a residue. Panics if missing.
fn find_coord(res: &Residue, name: &str) -> Vec3 {
    res.atom_coord(name)
        .unwrap_or_else(|| panic!("atom '{}' not found in residue {:?}", name, res.name))
}

/// Create a structure with a single first residue placed at the origin.
pub fn initialize_res(geo: &Geo) -> Structure {
    let ca_n_length = geo.ca_n_length;
    let ca_c_length = geo.ca_c_length;
    let n_ca_c_angle = geo.n_ca_c_angle;

    let ca = Vec3::zero();
    let c = Vec3::new(ca_c_length, 0.0, 0.0);
    let rad = n_ca_c_angle * std::f64::consts::PI / 180.0;
    let n = Vec3::new(ca_n_length * rad.cos(), ca_n_length * rad.sin(), 0.0);

    let n_atom = Atom::new("N", "N", n);
    let ca_atom = Atom::new("CA", "C", ca);
    let c_atom = Atom::new("C", "C", c);

    let o_pos = calculate_coordinates(
        n,
        ca,
        c,
        geo.c_o_length,
        geo.ca_c_o_angle,
        geo.n_ca_c_o_diangle,
    );
    let o_atom = Atom::new("O", "O", o_pos);

    let mut res = Residue::new(geo.residue_name, 1);
    res.add_atom(n_atom);
    res.add_atom(ca_atom);
    res.add_atom(c_atom);
    res.add_atom(o_atom);

    build_side_chain(&mut res, geo);

    let mut struc = Structure::new();
    struc.chain_a_mut().residues.push(res);
    struc
}

/// Add the next residue to the chain, using the given geometry and optional
/// backbone overrides.
pub fn add_residue(
    struc: &mut Structure,
    geo: &Geo,
    phi_override: Option<f64>,
    psi_im1_override: Option<f64>,
    omega_override: Option<f64>,
) {
    let chain = struc.chain_a_mut();
    let prev = chain
        .residues
        .last()
        .expect("chain must have at least one residue");
    let prev_n = find_coord(prev, "N");
    let prev_ca = find_coord(prev, "CA");
    let prev_c = find_coord(prev, "C");

    let seg_id = prev.seq_id + 1;

    let phi = phi_override.unwrap_or(geo.phi);
    let psi_im1 = psi_im1_override.unwrap_or(geo.psi_im1);
    let omega = omega_override.unwrap_or(geo.omega);

    // Place new backbone atoms
    let n_coord = calculate_coordinates(
        prev_n,
        prev_ca,
        prev_c,
        geo.peptide_bond,
        geo.ca_c_n_angle,
        psi_im1,
    );
    let ca_coord = calculate_coordinates(
        prev_ca,
        prev_c,
        n_coord,
        geo.ca_n_length,
        geo.c_n_ca_angle,
        omega,
    );
    let c_coord = calculate_coordinates(
        prev_c,
        n_coord,
        ca_coord,
        geo.ca_c_length,
        geo.n_ca_c_angle,
        phi,
    );
    let o_coord = calculate_coordinates(
        n_coord,
        ca_coord,
        c_coord,
        geo.c_o_length,
        geo.ca_c_o_angle,
        geo.n_ca_c_o_diangle,
    );

    let mut res = Residue::new(geo.residue_name, seg_id);
    res.add_atom(Atom::new("N", "N", n_coord));
    res.add_atom(Atom::new("CA", "C", ca_coord));
    res.add_atom(Atom::new("C", "C", c_coord));
    res.add_atom(Atom::new("O", "O", o_coord));

    build_side_chain(&mut res, geo);

    // Fix previous residue's O to face the new N
    let prev_mut = chain.residues.last_mut().unwrap();
    let new_o_for_prev = calculate_coordinates(
        n_coord,
        prev_ca,
        prev_c,
        geo.c_o_length,
        geo.ca_c_o_angle,
        180.0,
    );
    if let Some(o_atom) = prev_mut.atom_mut("O") {
        o_atom.coord = new_o_for_prev;
    }

    // Also fix current residue O: place trans to N across CA-C bond
    // (matching PeptideBuilder reference — ghost N computed but not used for O)
    let corrected_o = calculate_coordinates(
        n_coord,
        ca_coord,
        c_coord,
        geo.c_o_length,
        geo.ca_c_o_angle,
        180.0,
    );
    // Overwrite O in new res before push
    if let Some(o_atom) = res.atom_mut("O") {
        o_atom.coord = corrected_o;
    }

    chain.residues.push(res);
}

/// Add a terminal OXT atom to the last residue of every chain.
pub fn add_terminal_oxt(struc: &mut Structure) {
    for chain in &mut struc.chains {
        if chain.residues.is_empty() {
            continue;
        }
        let last = chain.residues.last().unwrap();
        // C-terminal caps (e.g., NME) intentionally do not carry OXT.
        if last.name.is_cap() {
            continue;
        }
        // Skip if OXT already present
        if last.atom_coord("OXT").is_some() {
            continue;
        }
        let (Some(n), Some(ca), Some(c), Some(o)) = (
            last.atom_coord("N"),
            last.atom_coord("CA"),
            last.atom_coord("C"),
            last.atom_coord("O"),
        ) else {
            continue;
        };

        let ca_c_oxt_angle = bond_angle(ca, c, o);
        let n_ca_c_o_di = crate::coord::calc_dihedral(n, ca, c, o);
        let n_ca_c_oxt_di = if n_ca_c_o_di < 0.0 {
            n_ca_c_o_di + 180.0
        } else {
            n_ca_c_o_di - 180.0
        };

        let oxt_pos = calculate_coordinates(n, ca, c, 1.23, ca_c_oxt_angle, n_ca_c_oxt_di);
        let last_mut = chain.residues.last_mut().unwrap();
        last_mut.add_atom(Atom::new("OXT", "O", oxt_pos));
    }
}

fn bond_angle(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let ba = a.sub(b);
    let bc = c.sub(b);
    let cos_a = ba.dot(bc) / (ba.length() * bc.length());
    cos_a.clamp(-1.0, 1.0).acos() * 180.0 / std::f64::consts::PI
}

// ----- Ramachandran angle presets -----

/// Standard backbone angle presets for common secondary structures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RamaPreset {
    /// Extended conformation (φ=180°, ψ=180°). Default geometry values.
    Extended,
    /// α-helix: φ=−57°, ψ=−47°.
    AlphaHelix,
    /// Anti-parallel β-sheet: φ=−120°, ψ=+130°.
    BetaSheet,
    /// Polyproline-II helix: φ=−75°, ψ=+145°.
    PolyProII,
}

impl RamaPreset {
    /// Parse from string (case-insensitive).
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().replace('-', "").replace('_', "").as_str() {
            "extended" | "ext" => Some(Self::Extended),
            "alphahelix" | "alpha" | "helix" => Some(Self::AlphaHelix),
            "betasheet" | "beta" | "sheet" => Some(Self::BetaSheet),
            "polyproii" | "ppii" | "polyproline" => Some(Self::PolyProII),
            _ => None,
        }
    }

    /// (phi, psi) in degrees.
    pub fn angles(self) -> (f64, f64) {
        match self {
            Self::Extended => (180.0, 180.0),
            Self::AlphaHelix => (-57.0, -47.0),
            Self::BetaSheet => (-120.0, 130.0),
            Self::PolyProII => (-75.0, 145.0),
        }
    }
}

// ----- High-level convenience -----

/// Parse a one-letter code (d-amino: lowercase) and return the appropriate geometry.
fn parse_one_letter_geo(ch: char) -> Result<Geo, String> {
    let (aa, is_d) =
        parse_d_amino_acid(ch).ok_or_else(|| format!("unknown amino acid '{}'", ch))?;
    Ok(if is_d {
        d_geometry(aa)
    } else {
        geometry::geometry(aa)
    })
}

/// Build a peptide in the extended conformation from a one-letter sequence.
/// Supports lowercase for D-amino acids (e.g. 'a' = D-Ala).
pub fn make_extended_structure(sequence: &str) -> Result<Structure, String> {
    let chars: Vec<char> = sequence.chars().collect();
    if chars.is_empty() {
        return Err("empty sequence".into());
    }
    let first = parse_one_letter_geo(chars[0])?;
    let mut struc = initialize_res(&first);

    for &ch in &chars[1..] {
        let geo = parse_one_letter_geo(ch)?;
        add_residue(&mut struc, &geo, None, None, None);
    }
    Ok(struc)
}

/// Build a peptide with explicit phi/psi angles.
/// Supports lowercase for D-amino acids.
pub fn make_structure(
    sequence: &str,
    phi: &[f64],
    psi_im1: &[f64],
    omega: Option<&[f64]>,
) -> Result<Structure, String> {
    let chars: Vec<char> = sequence.chars().collect();
    if chars.is_empty() {
        return Err("empty sequence".into());
    }
    let n = chars.len();
    if phi.len() != n - 1 || psi_im1.len() != n - 1 {
        return Err("phi/psi_im1 arrays must have length sequence_len - 1".into());
    }
    if let Some(om) = omega {
        if om.len() != n - 1 {
            return Err("omega array must have length sequence_len - 1".into());
        }
    }

    let first = parse_one_letter_geo(chars[0])?;
    let mut struc = initialize_res(&first);

    for i in 1..n {
        let geo = parse_one_letter_geo(chars[i])?;
        let om = omega.map(|o| o[i - 1]);
        add_residue(&mut struc, &geo, Some(phi[i - 1]), Some(psi_im1[i - 1]), om);
    }
    Ok(struc)
}

/// Build a peptide with a Ramachandran preset from a one-letter sequence.
/// Supports lowercase for D-amino acids.
pub fn make_preset_structure(sequence: &str, preset: RamaPreset) -> Result<Structure, String> {
    if preset == RamaPreset::Extended {
        return make_extended_structure(sequence);
    }
    let chars: Vec<char> = sequence.chars().collect();
    if chars.is_empty() {
        return Err("empty sequence".into());
    }
    let (phi, psi) = preset.angles();
    let first = parse_one_letter_geo(chars[0])?;
    let mut struc = initialize_res(&first);
    for &ch in &chars[1..] {
        let geo = parse_one_letter_geo(ch)?;
        add_residue(&mut struc, &geo, Some(phi), Some(psi), None);
    }
    Ok(struc)
}

// ----- Amber-aware builders (three-letter codes) -----

/// Residue specification: canonical type + optional Amber variant + non-standard + d-form.
#[derive(Debug, Clone, Copy)]
pub struct ResSpec {
    pub name: ResName,
    pub variant: Option<AmberVariant>,
    pub non_std: Option<NonStdResidue>,
    pub d_form: bool,
}

/// Parse a dash-separated three-letter sequence (e.g. "ALA-CYX-HID-GLU-MSE").
/// Returns a Vec of ResSpec, preserving Amber variant and non-standard info.
pub fn parse_three_letter_sequence(seq: &str) -> Result<Vec<ResSpec>, String> {
    let tokens: Vec<&str> = seq
        .split('-')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return Err("empty sequence".into());
    }
    tokens
        .iter()
        .map(|tok| {
            // Check non-standard first (MSE, PCA)
            if let Some(nsr) = NonStdResidue::from_str(tok) {
                return Ok(ResSpec {
                    name: nsr.canonical(),
                    variant: None,
                    non_std: Some(nsr),
                    d_form: false,
                });
            }
            let (name, variant) = parse_amber_name(tok)
                .ok_or_else(|| format!("unknown residue name '{}'", tok))?;
            if name.is_cap() {
                return Err(format!(
                    "terminal cap residue '{}' is not allowed in three-letter sequence builders; build peptide first, then apply caps",
                    tok.to_uppercase()
                ));
            }
            Ok(ResSpec { name, variant, non_std: None, d_form: false })
        })
        .collect()
}

fn ensure_specs_buildable(specs: &[ResSpec]) -> Result<(), String> {
    if let Some(spec) = specs.iter().find(|s| s.name.is_cap()) {
        return Err(format!(
            "terminal cap residue '{}' cannot be built from backbone geometry; build uncapped peptide and then apply caps",
            spec.name.as_str()
        ));
    }
    Ok(())
}

fn geo_for_spec(spec: &ResSpec) -> Geo {
    let mut geo = if let Some(nsr) = spec.non_std {
        match nsr {
            NonStdResidue::MSE => non_standard::mse_geometry(),
            NonStdResidue::PCA => non_standard::pca_geometry(),
        }
    } else if let Some(v) = spec.variant {
        geometry::variant_geometry(v)
    } else {
        geometry::geometry(spec.name)
    };
    if spec.d_form {
        for sc in &mut geo.side_chain {
            sc.dihedral = -sc.dihedral;
        }
        geo.n_ca_c_o_diangle = -geo.n_ca_c_o_diangle;
    }
    geo
}

fn stamp_variant(struc: &mut Structure, variant: Option<AmberVariant>) {
    if let Some(v) = variant {
        let last = struc.chain_a_mut().residues.last_mut().unwrap();
        last.variant = Some(v);
    }
}

fn stamp_non_std(struc: &mut Structure, non_std: Option<NonStdResidue>) {
    if let Some(nsr) = non_std {
        let last = struc.chain_a_mut().residues.last_mut().unwrap();
        last.non_std = Some(nsr);
    }
}

fn stamp_spec(struc: &mut Structure, spec: &ResSpec) {
    stamp_variant(struc, spec.variant);
    stamp_non_std(struc, spec.non_std);
}

/// Build extended-conformation peptide from three-letter specs.
pub fn make_extended_structure_from_specs(specs: &[ResSpec]) -> Result<Structure, String> {
    if specs.is_empty() {
        return Err("empty sequence".into());
    }
    ensure_specs_buildable(specs)?;
    let mut struc = initialize_res(&geo_for_spec(&specs[0]));
    stamp_spec(&mut struc, &specs[0]);

    for spec in &specs[1..] {
        let geo = geo_for_spec(spec);
        add_residue(&mut struc, &geo, None, None, None);
        stamp_spec(&mut struc, spec);
    }
    Ok(struc)
}

/// Build peptide from three-letter specs using a Ramachandran preset.
pub fn make_preset_structure_from_specs(
    specs: &[ResSpec],
    preset: RamaPreset,
) -> Result<Structure, String> {
    if preset == RamaPreset::Extended {
        return make_extended_structure_from_specs(specs);
    }
    if specs.is_empty() {
        return Err("empty sequence".into());
    }
    ensure_specs_buildable(specs)?;
    let (phi, psi) = preset.angles();
    let mut struc = initialize_res(&geo_for_spec(&specs[0]));
    stamp_spec(&mut struc, &specs[0]);
    for spec in &specs[1..] {
        let geo = geo_for_spec(spec);
        add_residue(&mut struc, &geo, Some(phi), Some(psi), None);
        stamp_spec(&mut struc, spec);
    }
    Ok(struc)
}

/// Build peptide from three-letter specs with explicit phi/psi angles.
pub fn make_structure_from_specs(
    specs: &[ResSpec],
    phi: &[f64],
    psi_im1: &[f64],
    omega: Option<&[f64]>,
) -> Result<Structure, String> {
    if specs.is_empty() {
        return Err("empty sequence".into());
    }
    ensure_specs_buildable(specs)?;
    let n = specs.len();
    if phi.len() != n - 1 || psi_im1.len() != n - 1 {
        return Err("phi/psi_im1 arrays must have length sequence_len - 1".into());
    }
    if let Some(om) = omega {
        if om.len() != n - 1 {
            return Err("omega array must have length sequence_len - 1".into());
        }
    }

    let mut struc = initialize_res(&geo_for_spec(&specs[0]));
    stamp_spec(&mut struc, &specs[0]);

    for i in 1..n {
        let geo = geo_for_spec(&specs[i]);
        let om = omega.map(|o| o[i - 1]);
        add_residue(&mut struc, &geo, Some(phi[i - 1]), Some(psi_im1[i - 1]), om);
        stamp_spec(&mut struc, &specs[i]);
    }
    Ok(struc)
}

// ----- Multi-chain builder -----

/// Chain spec for multi-chain building.
#[derive(Debug, Clone)]
pub struct ChainSpec {
    pub id: char,
    pub residues: Vec<ResSpec>,
    pub preset: Option<RamaPreset>,
}

/// Build a single chain as a Chain object (no wrapping Structure).
fn build_chain_inner(spec: &ChainSpec) -> Result<Chain, String> {
    if spec.residues.is_empty() {
        return Err(format!("chain '{}' has empty sequence", spec.id));
    }
    // Build into a temp Structure, then extract chain A and re-label it.
    let struc = match spec.preset {
        Some(preset) => make_preset_structure_from_specs(&spec.residues, preset)?,
        None => make_extended_structure_from_specs(&spec.residues)?,
    };
    let mut chain = struc.chains.into_iter().next().unwrap();
    chain.id = spec.id;
    Ok(chain)
}

/// Build multi-chain structure from a list of chain specifications.
pub fn make_multi_chain_structure(chain_specs: &[ChainSpec]) -> Result<Structure, String> {
    if chain_specs.is_empty() {
        return Err("no chains specified".into());
    }
    let mut struc = Structure::new_empty();
    for cs in chain_specs {
        let chain = build_chain_inner(cs)?;
        struc.add_chain(chain);
    }
    // Re-number residues globally so seq_ids don't collide across chains
    let mut global_id = 1;
    for chain in &mut struc.chains {
        for res in &mut chain.residues {
            res.seq_id = global_id;
            global_id += 1;
        }
    }
    Ok(struc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_glycine() {
        let struc = make_extended_structure("G").unwrap();
        let chain = struc.chain_a();
        assert_eq!(chain.residues.len(), 1);
        assert_eq!(chain.residues[0].atoms.len(), 4); // N, CA, C, O
    }

    #[test]
    fn test_dipeptide_ala() {
        let struc = make_extended_structure("AA").unwrap();
        let chain = struc.chain_a();
        assert_eq!(chain.residues.len(), 2);
        // Ala has 5 atoms: N, CA, C, O, CB
        assert_eq!(chain.residues[0].atoms.len(), 5);
        assert_eq!(chain.residues[1].atoms.len(), 5);
    }

    #[test]
    fn test_all_twenty() {
        let struc = make_extended_structure("ACDEFGHIKLMNPQRSTVWY").unwrap();
        assert_eq!(struc.chain_a().residues.len(), 20);
    }

    #[test]
    fn test_invalid_code() {
        assert!(make_extended_structure("X").is_err());
    }

    #[test]
    fn test_peptide_bond_distance() {
        let struc = make_extended_structure("AG").unwrap();
        let r0 = &struc.chain_a().residues[0];
        let r1 = &struc.chain_a().residues[1];
        let c0 = r0.atom_coord("C").unwrap();
        let n1 = r1.atom_coord("N").unwrap();
        let dist = c0.sub(n1).length();
        assert!(
            (dist - 1.33).abs() < 0.05,
            "peptide bond distance {dist} not ~1.33"
        );
    }

    #[test]
    fn test_parse_three_letter_sequence() {
        let specs = parse_three_letter_sequence("ALA-CYX-HID-GLU").unwrap();
        assert_eq!(specs.len(), 4);
        assert_eq!(specs[0].name, ResName::ALA);
        assert!(specs[0].variant.is_none());
        assert_eq!(specs[1].name, ResName::CYS);
        assert_eq!(specs[1].variant, Some(AmberVariant::CYX));
        assert_eq!(specs[2].name, ResName::HIS);
        assert_eq!(specs[2].variant, Some(AmberVariant::HID));
        assert_eq!(specs[3].name, ResName::GLU);
        assert!(specs[3].variant.is_none());
    }

    #[test]
    fn test_three_letter_build_cyx() {
        let specs = parse_three_letter_sequence("ALA-CYX-ALA").unwrap();
        let struc = make_extended_structure_from_specs(&specs).unwrap();
        let chain = struc.chain_a();
        assert_eq!(chain.residues.len(), 3);
        // CYX residue should have CYS canonical name + CYX variant
        assert_eq!(chain.residues[1].name, ResName::CYS);
        assert_eq!(chain.residues[1].variant, Some(AmberVariant::CYX));
        assert_eq!(chain.residues[1].amber_name(), "CYX");
    }

    #[test]
    fn test_three_letter_build_hip() {
        let specs = parse_three_letter_sequence("HIP").unwrap();
        let struc = make_extended_structure_from_specs(&specs).unwrap();
        let r = &struc.chain_a().residues[0];
        assert_eq!(r.name, ResName::HIS);
        assert_eq!(r.variant, Some(AmberVariant::HIP));
    }

    #[test]
    fn test_three_letter_invalid() {
        assert!(parse_three_letter_sequence("ALA-ZZZ").is_err());
    }

    #[test]
    fn test_three_letter_rejects_caps() {
        assert!(parse_three_letter_sequence("ACE-ALA").is_err());
        assert!(parse_three_letter_sequence("ALA-NME").is_err());
    }

    #[test]
    fn test_build_from_specs_rejects_caps() {
        let specs = vec![
            ResSpec {
                name: ResName::ACE,
                variant: None,
                non_std: None,
                d_form: false,
            },
            ResSpec {
                name: ResName::ALA,
                variant: None,
                non_std: None,
                d_form: false,
            },
        ];
        let err = make_extended_structure_from_specs(&specs).unwrap_err();
        assert!(err.contains("terminal cap"));
    }

    #[test]
    fn test_preset_alpha_helix() {
        let struc = make_preset_structure("AAAA", RamaPreset::AlphaHelix).unwrap();
        let chain = struc.chain_a();
        assert_eq!(chain.residues.len(), 4);
        // All residues should have backbone atoms
        for r in &chain.residues {
            assert!(r.atom_coord("N").is_some());
            assert!(r.atom_coord("CA").is_some());
            assert!(r.atom_coord("C").is_some());
            assert!(r.atom_coord("O").is_some());
        }
    }

    #[test]
    fn test_preset_beta_sheet() {
        let struc = make_preset_structure("GVL", RamaPreset::BetaSheet).unwrap();
        assert_eq!(struc.chain_a().residues.len(), 3);
    }

    #[test]
    fn test_preset_from_specs() {
        let specs = parse_three_letter_sequence("ALA-CYX-GLU").unwrap();
        let struc = make_preset_structure_from_specs(&specs, RamaPreset::PolyProII).unwrap();
        assert_eq!(struc.chain_a().residues.len(), 3);
        assert_eq!(struc.chain_a().residues[1].amber_name(), "CYX");
    }

    #[test]
    fn test_multi_chain_basic() {
        let chains = vec![
            ChainSpec {
                id: 'A',
                residues: parse_three_letter_sequence("ALA-GLY").unwrap(),
                preset: Some(RamaPreset::AlphaHelix),
            },
            ChainSpec {
                id: 'B',
                residues: parse_three_letter_sequence("VAL-TRP-SER").unwrap(),
                preset: None,
            },
        ];
        let struc = make_multi_chain_structure(&chains).unwrap();
        assert_eq!(struc.chains.len(), 2);
        assert_eq!(struc.chain_by_id('A').unwrap().residues.len(), 2);
        assert_eq!(struc.chain_by_id('B').unwrap().residues.len(), 3);
        // Global numbering: A has 1,2; B has 3,4,5
        assert_eq!(struc.chain_by_id('B').unwrap().residues[0].seq_id, 3);
    }

    #[test]
    fn test_rama_preset_parse() {
        assert_eq!(
            RamaPreset::from_str("alpha-helix"),
            Some(RamaPreset::AlphaHelix)
        );
        assert_eq!(
            RamaPreset::from_str("BETA_SHEET"),
            Some(RamaPreset::BetaSheet)
        );
        assert_eq!(RamaPreset::from_str("ppii"), Some(RamaPreset::PolyProII));
        assert_eq!(RamaPreset::from_str("extended"), Some(RamaPreset::Extended));
        assert!(RamaPreset::from_str("unknown").is_none());
    }

    #[test]
    fn test_oxt_all_chains() {
        let chains = vec![
            ChainSpec {
                id: 'A',
                residues: parse_three_letter_sequence("ALA-GLY").unwrap(),
                preset: None,
            },
            ChainSpec {
                id: 'B',
                residues: parse_three_letter_sequence("VAL-SER").unwrap(),
                preset: None,
            },
        ];
        let mut struc = make_multi_chain_structure(&chains).unwrap();
        add_terminal_oxt(&mut struc);
        // Both chains should have OXT on their last residue
        let a_last = struc.chain_by_id('A').unwrap().residues.last().unwrap();
        let b_last = struc.chain_by_id('B').unwrap().residues.last().unwrap();
        assert!(a_last.atom_coord("OXT").is_some(), "chain A missing OXT");
        assert!(b_last.atom_coord("OXT").is_some(), "chain B missing OXT");
    }

    #[test]
    fn test_oxt_idempotent() {
        let mut struc = make_extended_structure("AA").unwrap();
        add_terminal_oxt(&mut struc);
        add_terminal_oxt(&mut struc); // second call should not duplicate
        let oxt_count = struc
            .chain_a()
            .residues
            .last()
            .unwrap()
            .atoms
            .iter()
            .filter(|a| a.name == "OXT")
            .count();
        assert_eq!(oxt_count, 1);
    }

    #[test]
    fn test_oxt_skips_nme_capped_terminus() {
        let mut struc = make_extended_structure("AA").unwrap();
        crate::caps::add_caps(&mut struc);
        add_terminal_oxt(&mut struc);
        let total_oxt = struc
            .chain_a()
            .residues
            .iter()
            .flat_map(|r| r.atoms.iter())
            .filter(|a| a.name == "OXT")
            .count();
        assert_eq!(
            total_oxt, 0,
            "OXT should not be added when NME cap is terminal"
        );
    }

    // ----- D-amino acid tests -----

    #[test]
    fn test_d_amino_one_letter_build() {
        // lowercase 'a' = D-Ala
        let struc = make_extended_structure("aA").unwrap();
        assert_eq!(struc.chain_a().residues.len(), 2);
        // Both should have ALA-like atom count (5: N, CA, C, O, CB)
        assert_eq!(struc.chain_a().residues[0].atoms.len(), 5);
        assert_eq!(struc.chain_a().residues[1].atoms.len(), 5);
    }

    #[test]
    fn test_d_amino_mixed_case() {
        // "aGlA" = D-Ala, Gly, Leu, Ala — all should build
        let struc = make_extended_structure("aGLA").unwrap();
        assert_eq!(struc.chain_a().residues.len(), 4);
    }

    // ----- Non-standard residue tests -----

    #[test]
    fn test_parse_three_letter_mse() {
        let specs = parse_three_letter_sequence("ALA-MSE-GLY").unwrap();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[1].name, ResName::MET); // canonical
        assert_eq!(specs[1].non_std, Some(NonStdResidue::MSE));
    }

    #[test]
    fn test_parse_three_letter_pca() {
        let specs = parse_three_letter_sequence("PCA-ALA").unwrap();
        assert_eq!(specs[0].name, ResName::GLU); // canonical
        assert_eq!(specs[0].non_std, Some(NonStdResidue::PCA));
    }

    #[test]
    fn test_build_mse_has_selenium() {
        let specs = parse_three_letter_sequence("ALA-MSE-GLY").unwrap();
        let struc = make_extended_structure_from_specs(&specs).unwrap();
        let mse_res = &struc.chain_a().residues[1];
        // MSE should have SE atom instead of SD
        assert!(mse_res.atom_coord("SE").is_some(), "MSE missing SE");
        assert!(mse_res.atom_coord("SD").is_none(), "MSE should not have SD");
        // Residue output name should be MSE
        assert_eq!(mse_res.amber_name(), "MSE");
    }

    #[test]
    fn test_build_pca() {
        let specs = parse_three_letter_sequence("PCA-ALA").unwrap();
        let struc = make_extended_structure_from_specs(&specs).unwrap();
        let pca_res = &struc.chain_a().residues[0];
        // PCA should not have OE2
        assert!(
            pca_res.atom_coord("OE2").is_none(),
            "PCA should not have OE2"
        );
        assert_eq!(pca_res.amber_name(), "PCA");
    }

    // ----- Carbonyl O trans to N test -----

    #[test]
    fn test_carbonyl_o_trans_to_n() {
        // In a dipeptide, the second residue's O should be trans to its N
        // across the CA-C bond (dihedral N-CA-C-O ≈ 180° or close).
        let struc = make_extended_structure("AG").unwrap();
        let r1 = &struc.chain_a().residues[1];
        let n = r1.atom_coord("N").unwrap();
        let ca = r1.atom_coord("CA").unwrap();
        let c = r1.atom_coord("C").unwrap();
        let o = r1.atom_coord("O").unwrap();
        // Compute dihedral N-CA-C-O — should be close to +/-180
        let d = dihedral_angle(n, ca, c, o);
        assert!(
            d.abs() > 150.0,
            "N-CA-C-O dihedral {} should be near ±180°, indicating O trans to N",
            d
        );
    }
}

/// Compute dihedral angle in degrees (used in tests).
#[cfg(test)]
fn dihedral_angle(
    a: crate::coord::Vec3,
    b: crate::coord::Vec3,
    c: crate::coord::Vec3,
    d: crate::coord::Vec3,
) -> f64 {
    let b1 = b.sub(a);
    let b2 = c.sub(b);
    let b3 = d.sub(c);
    let n1 = b1.cross(b2);
    let n2 = b2.cross(b3);
    let m = n1.cross(b2.scale(1.0 / b2.length()));
    let x = n1.dot(n2);
    let y = m.dot(n2);
    -y.atan2(x).to_degrees()
}
