//! Default internal-coordinate geometry for all 20 standard amino acids.
//!
//! Ported from clauswilke/PeptideBuilder (Python, MIT license).
//! Bond lengths in Ångström, angles in degrees.

use crate::residue::{AmberVariant, ResName};

/// Backbone + side-chain internal coordinates for one residue type.
#[derive(Debug, Clone)]
pub struct Geo {
    pub residue_name: ResName,

    // --- peptide bond geometry (to stitch residues) ---
    pub peptide_bond: f64,
    pub ca_c_n_angle: f64,
    pub c_n_ca_angle: f64,

    // --- backbone ---
    pub n_ca_c_angle: f64,
    pub ca_n_length: f64,
    pub ca_c_length: f64,
    pub phi: f64,
    pub psi_im1: f64,
    pub omega: f64,

    // --- carbonyl ---
    pub c_o_length: f64,
    pub ca_c_o_angle: f64,
    pub n_ca_c_o_diangle: f64,

    // --- side chain internal coordinates (variable length) ---
    pub side_chain: Vec<SideChainAtom>,
}

/// One side-chain atom placed relative to three parent atoms.
///
/// The `parent_names` triple names atoms already placed (backbone or earlier
/// side-chain atoms) that serve as reference for the internal coordinate
/// calculation in order (ref_a, ref_b, ref_c) where the new atom is placed at
/// distance `length` from ref_c, with bond angle at ref_c, and dihedral
/// through ref_a-ref_b-ref_c-new.
#[derive(Debug, Clone)]
pub struct SideChainAtom {
    pub name: &'static str,
    pub element: &'static str,
    pub parents: (&'static str, &'static str, &'static str),
    pub length: f64,
    pub angle: f64,
    pub dihedral: f64,
}

impl SideChainAtom {
    pub const fn new(
        name: &'static str,
        element: &'static str,
        parents: (&'static str, &'static str, &'static str),
        length: f64,
        angle: f64,
        dihedral: f64,
    ) -> Self {
        Self { name, element, parents, length, angle, dihedral }
    }
}

fn base(name: ResName, n_ca_c: f64, ca_c_o: f64, n_ca_c_o: f64) -> Geo {
    Geo {
        residue_name: name,
        peptide_bond: 1.33,
        ca_c_n_angle: 116.642992978143,
        c_n_ca_angle: 121.382215820277,
        n_ca_c_angle: n_ca_c,
        ca_n_length: 1.46,
        ca_c_length: 1.52,
        phi: -120.0,
        psi_im1: 140.0,
        omega: 180.0,
        c_o_length: 1.23,
        ca_c_o_angle: ca_c_o,
        n_ca_c_o_diangle: n_ca_c_o,
        side_chain: Vec::new(),
    }
}

/// Return the default geometry for the given amino acid.
pub fn geometry(aa: ResName) -> Geo {
    match aa {
        ResName::GLY => gly(),
        ResName::ALA => ala(),
        ResName::SER => ser(),
        ResName::CYS => cys(),
        ResName::VAL => val(),
        ResName::ILE => ile(),
        ResName::LEU => leu(),
        ResName::THR => thr(),
        ResName::ARG => arg(),
        ResName::LYS => lys(),
        ResName::ASP => asp(),
        ResName::GLU => glu(),
        ResName::ASN => asn(),
        ResName::GLN => gln(),
        ResName::MET => met(),
        ResName::HIS => his(),
        ResName::PRO => pro(),
        ResName::PHE => phe(),
        ResName::TYR => tyr(),
        ResName::TRP => trp(),
        ResName::ACE | ResName::NME => {
            panic!("geometry() is not applicable to terminal caps ACE/NME")
        }
    }
}

fn gly() -> Geo {
    base(ResName::GLY, 110.8914, 120.5117, 180.0)
}

fn ala() -> Geo {
    let mut g = base(ResName::ALA, 111.068, 120.5, -60.5);
    g.side_chain = vec![
        SideChainAtom::new("CB", "C", ("N", "C", "CA"), 1.52, 109.5, 122.6860),
    ];
    g
}

fn ser() -> Geo {
    let mut g = base(ResName::SER, 111.2812, 120.5, -60.0);
    g.side_chain = vec![
        SideChainAtom::new("CB", "C", ("N", "C", "CA"), 1.52, 109.5, 122.6618),
        SideChainAtom::new("OG", "O", ("N", "CA", "CB"), 1.417, 110.773, -63.3),
    ];
    g
}

fn cys() -> Geo {
    let mut g = base(ResName::CYS, 110.8856, 120.5, -60.0);
    g.side_chain = vec![
        SideChainAtom::new("CB", "C", ("N", "C", "CA"), 1.52, 109.5, 122.5037),
        SideChainAtom::new("SG", "S", ("N", "CA", "CB"), 1.808, 113.8169, -62.2),
    ];
    g
}

fn val() -> Geo {
    let mut g = base(ResName::VAL, 109.7698, 120.5686, -60.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),  1.52,  109.5,  123.2347),
        SideChainAtom::new("CG1", "C", ("N", "CA", "CB"), 1.527, 110.7,  177.2),
        SideChainAtom::new("CG2", "C", ("N", "CA", "CB"), 1.527, 110.4,  -63.3),
    ];
    g
}

fn ile() -> Geo {
    let mut g = base(ResName::ILE, 109.7202, 120.5403, -60.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),   1.52,   109.5,  123.2347),
        SideChainAtom::new("CG1", "C", ("N", "CA", "CB"),  1.527,  110.7,  59.7),
        SideChainAtom::new("CG2", "C", ("N", "CA", "CB"),  1.527,  110.4,  -61.6),
        SideChainAtom::new("CD1", "C", ("CA", "CB", "CG1"), 1.52,  113.97, 169.8),
    ];
    g
}

fn leu() -> Geo {
    let mut g = base(ResName::LEU, 110.8652, 120.4647, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),   1.52,  109.5,   122.4948),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),  1.53,  116.10,  -60.1),
        SideChainAtom::new("CD1", "C", ("CA", "CB", "CG"), 1.524, 110.27,  174.9),
        SideChainAtom::new("CD2", "C", ("CA", "CB", "CG"), 1.525, 110.58,  66.7),
    ];
    g
}

fn thr() -> Geo {
    let mut g = base(ResName::THR, 110.7014, 120.5359, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),  1.52, 109.5,  123.0953),
        SideChainAtom::new("OG1", "O", ("N", "CA", "CB"), 1.43, 109.18, 60.0),
        SideChainAtom::new("CG2", "C", ("N", "CA", "CB"), 1.53, 111.13, -60.3),
    ];
    g
}

fn arg() -> Geo {
    let mut g = base(ResName::ARG, 110.98, 120.54, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),   1.52,  109.5,   122.76),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),  1.52,  113.83,  -65.2),
        SideChainAtom::new("CD",  "C", ("CA", "CB", "CG"), 1.52,  111.79,  -179.2),
        SideChainAtom::new("NE",  "N", ("CB", "CG", "CD"), 1.46,  111.68,  -179.3),
        SideChainAtom::new("CZ",  "C", ("CG", "CD", "NE"), 1.33,  124.79,  -178.7),
        SideChainAtom::new("NH1", "N", ("CD", "NE", "CZ"), 1.33,  120.64,  0.0),
        SideChainAtom::new("NH2", "N", ("CD", "NE", "CZ"), 1.33,  119.63,  180.0),
    ];
    g
}

fn lys() -> Geo {
    let mut g = base(ResName::LYS, 111.08, 120.54, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB", "C", ("N", "C", "CA"),   1.52, 109.5,  122.76),
        SideChainAtom::new("CG", "C", ("N", "CA", "CB"),  1.52, 113.83, -64.5),
        SideChainAtom::new("CD", "C", ("CA", "CB", "CG"), 1.52, 111.79, -178.1),
        SideChainAtom::new("CE", "C", ("CB", "CG", "CD"), 1.46, 111.68, -179.6),
        SideChainAtom::new("NZ", "N", ("CG", "CD", "CE"), 1.33, 124.79, 179.6),
    ];
    g
}

fn asp() -> Geo {
    let mut g = base(ResName::ASP, 111.03, 120.51, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),   1.52, 109.5,  122.82),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),  1.52, 113.06, -66.4),
        SideChainAtom::new("OD1", "O", ("CA", "CB", "CG"), 1.25, 119.22, -46.7),
        SideChainAtom::new("OD2", "O", ("CA", "CB", "CG"), 1.25, 118.218, 133.3),
    ];
    g
}

fn glu() -> Geo {
    let mut g = base(ResName::GLU, 111.1703, 120.511, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),   1.52, 109.5,  122.8702),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),  1.52, 113.82, -63.8),
        SideChainAtom::new("CD",  "C", ("CA", "CB", "CG"), 1.52, 113.31, -179.8),
        SideChainAtom::new("OE1", "O", ("CB", "CG", "CD"), 1.25, 119.02, -6.2),
        SideChainAtom::new("OE2", "O", ("CB", "CG", "CD"), 1.25, 118.08, 173.8),
    ];
    g
}

fn asn() -> Geo {
    let mut g = base(ResName::ASN, 111.5, 120.4826, -60.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),   1.52, 109.5,  123.2254),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),  1.52, 112.62, -65.5),
        SideChainAtom::new("OD1", "O", ("CA", "CB", "CG"), 1.23, 120.85, -58.3),
        SideChainAtom::new("ND2", "N", ("CA", "CB", "CG"), 1.33, 116.48, 121.7),
    ];
    g
}

fn gln() -> Geo {
    let mut g = base(ResName::GLN, 111.0849, 120.5029, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),   1.52, 109.5,  122.8134),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),  1.52, 113.75, -60.2),
        SideChainAtom::new("CD",  "C", ("CA", "CB", "CG"), 1.52, 112.78, -69.6),
        SideChainAtom::new("OE1", "O", ("CB", "CG", "CD"), 1.24, 120.86, -50.5),
        SideChainAtom::new("NE2", "N", ("CB", "CG", "CD"), 1.33, 116.50, 129.5),
    ];
    g
}

fn met() -> Geo {
    let mut g = base(ResName::MET, 110.9416, 120.4816, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB", "C", ("N", "C", "CA"),   1.52, 109.5,  122.6733),
        SideChainAtom::new("CG", "C", ("N", "CA", "CB"),  1.52, 113.68, -64.4),
        SideChainAtom::new("SD", "S", ("CA", "CB", "CG"), 1.81, 112.69, -179.6),
        SideChainAtom::new("CE", "C", ("CB", "CG", "SD"), 1.79, 100.61, 70.1),
    ];
    g
}

fn his() -> Geo {
    let mut g = base(ResName::HIS, 111.0859, 120.4732, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),    1.52,  109.5,   122.6711),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),   1.49,  113.74,  -63.2),
        SideChainAtom::new("ND1", "N", ("CA", "CB", "CG"),  1.38,  122.85,  -75.7),
        SideChainAtom::new("CD2", "C", ("CA", "CB", "CG"),  1.35,  130.61,  104.3),
        SideChainAtom::new("CE1", "C", ("CB", "CG", "ND1"), 1.32,  108.5,   180.0),
        SideChainAtom::new("NE2", "N", ("CB", "CG", "CD2"), 1.35,  108.5,   180.0),
    ];
    g
}

fn pro() -> Geo {
    let mut g = base(ResName::PRO, 112.7499, 120.2945, -45.0);
    g.side_chain = vec![
        SideChainAtom::new("CB", "C", ("N", "C", "CA"),   1.52, 109.5,  115.2975),
        SideChainAtom::new("CG", "C", ("N", "CA", "CB"),  1.49, 104.21, 29.6),
        SideChainAtom::new("CD", "C", ("CA", "CB", "CG"), 1.50, 105.03, -34.8),
    ];
    g
}

fn phe() -> Geo {
    let mut g = base(ResName::PHE, 110.7528, 120.5316, 120.0);
    // Atom order matches PeptideBuilder: CB CG CD1 CE1 CD2 CE2 CZ
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),    1.52, 109.5,  122.6054),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),   1.50, 113.85, -64.7),
        SideChainAtom::new("CD1", "C", ("CA", "CB", "CG"),  1.39, 120.0,  93.3),
        SideChainAtom::new("CE1", "C", ("CB", "CG", "CD1"), 1.39, 120.0,  180.0),
        SideChainAtom::new("CD2", "C", ("CA", "CB", "CG"),  1.39, 120.0,  -86.7),
        SideChainAtom::new("CE2", "C", ("CB", "CG", "CD2"), 1.39, 120.0,  180.0),
        SideChainAtom::new("CZ",  "C", ("CG", "CD1", "CE1"), 1.39, 120.0, 0.0),
    ];
    g
}

fn tyr() -> Geo {
    let mut g = base(ResName::TYR, 110.9288, 120.5434, 120.0);
    // Atom order matches PeptideBuilder: CB CG CD1 CE1 CD2 CE2 CZ OH
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),     1.52, 109.5,   122.6023),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),    1.51, 113.8,   -64.3),
        SideChainAtom::new("CD1", "C", ("CA", "CB", "CG"),   1.39, 120.98,  93.1),
        SideChainAtom::new("CE1", "C", ("CB", "CG", "CD1"),  1.39, 120.0,   180.0),
        SideChainAtom::new("CD2", "C", ("CA", "CB", "CG"),   1.39, 120.82,  -86.9),
        SideChainAtom::new("CE2", "C", ("CB", "CG", "CD2"),  1.39, 120.0,   180.0),
        SideChainAtom::new("CZ",  "C", ("CG", "CD1", "CE1"), 1.39, 120.0,   0.0),
        SideChainAtom::new("OH",  "O", ("CD1", "CE1", "CZ"), 1.39, 119.78,  180.0),
    ];
    g
}

fn trp() -> Geo {
    let mut g = base(ResName::TRP, 110.8914, 120.5117, 120.0);
    g.side_chain = vec![
        SideChainAtom::new("CB",  "C", ("N", "C", "CA"),     1.52, 109.5,   122.6112),
        SideChainAtom::new("CG",  "C", ("N", "CA", "CB"),    1.50, 114.10,  -66.4),
        SideChainAtom::new("CD1", "C", ("CA", "CB", "CG"),   1.37, 127.07,  96.3),
        SideChainAtom::new("CD2", "C", ("CA", "CB", "CG"),   1.43, 126.66,  -83.7),
        SideChainAtom::new("NE1", "N", ("CB", "CG", "CD1"),  1.38, 108.5,   180.0),
        SideChainAtom::new("CE2", "C", ("CB", "CG", "CD2"),  1.40, 108.5,   180.0),
        SideChainAtom::new("CE3", "C", ("CB", "CG", "CD2"),  1.40, 133.83,  0.0),
        SideChainAtom::new("CZ2", "C", ("CG", "CD2", "CE2"), 1.40, 120.0,   180.0),
        SideChainAtom::new("CZ3", "C", ("CG", "CD2", "CE3"), 1.40, 120.0,   180.0),
        SideChainAtom::new("CH2", "C", ("CD2", "CE2", "CZ2"), 1.40, 120.0,  0.0),
    ];
    g
}

/// Return geometry for an Amber variant residue.
///
/// For heavy atoms the geometry is identical to the canonical type.
/// The caller is responsible for stamping the variant on the built residue.

impl Geo {
    /// Set side-chain dihedral angles (χ1, χ2, ...) from a rotamer list.
    ///
    /// The first rotamer sets the first side-chain atom's dihedral, the second
    /// sets the next, etc. If the list is shorter than the side chain, remaining
    /// dihedrals keep their default values.
    pub fn set_rotamers(&mut self, rotamers: &[f64]) {
        let n = rotamers.len().min(self.side_chain.len());
        for (sc, &rot) in self.side_chain.iter_mut().zip(rotamers.iter()).take(n) {
            sc.dihedral = rot;
        }
    }

    /// Randomize side-chain dihedrals using common rotamer bins (−60°, 60°, 180°).
    pub fn randomize_rotamers(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let bins = [-60.0_f64, 60.0, 180.0];
        let mut hasher = DefaultHasher::new();
        self.residue_name.as_str().hash(&mut hasher);
        let mut seed = hasher.finish();
        for sc in &mut self.side_chain {
            sc.dihedral = bins[(seed % 3) as usize];
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        }
    }
}

pub fn variant_geometry(variant: AmberVariant) -> Geo {
    geometry(variant.canonical())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_rotamers_leu() {
        let mut geo = geometry(ResName::LEU);
        let orig: Vec<f64> = geo.side_chain.iter().map(|sc| sc.dihedral).collect();
        geo.set_rotamers(&[60.0, -60.0, 180.0]);
        assert!((geo.side_chain[0].dihedral - 60.0).abs() < 1e-10);
        assert!((geo.side_chain[1].dihedral - (-60.0)).abs() < 1e-10);
        assert!((geo.side_chain[2].dihedral - 180.0).abs() < 1e-10);
        // LEU has 4 side-chain atoms; 4th should keep default
        if geo.side_chain.len() > 3 {
            assert!((geo.side_chain[3].dihedral - orig[3]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_set_rotamers_gly_noop() {
        let mut geo = geometry(ResName::GLY);
        geo.set_rotamers(&[60.0]); // GLY has no side chain → no-op
        assert!(geo.side_chain.is_empty());
    }

    #[test]
    fn test_randomize_rotamers_changes_dihedrals() {
        let mut geo = geometry(ResName::TRP);
        let orig: Vec<f64> = geo.side_chain.iter().map(|sc| sc.dihedral).collect();
        geo.randomize_rotamers();
        // At least some dihedrals should change
        let changed = geo.side_chain.iter()
            .zip(orig.iter())
            .any(|(sc, &o)| (sc.dihedral - o).abs() > 1e-10);
        assert!(changed, "randomize_rotamers should change at least one dihedral");
        // All should be from the bins {-60, 60, 180}
        for sc in &geo.side_chain {
            let valid = [-60.0, 60.0, 180.0].iter().any(|b| (sc.dihedral - b).abs() < 1e-10);
            assert!(valid, "dihedral {} not in bins [-60, 60, 180]", sc.dihedral);
        }
    }
}
