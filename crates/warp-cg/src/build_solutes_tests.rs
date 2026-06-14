use super::*;

#[path = "build_solutes_test_parts/amino_acid_registry_coverage.rs"]
mod amino_acid_registry_coverage;

#[test]
fn amino_acid_templates_match_reference_library_charges() {
    assert_eq!(lookup_solute_template("GLY").unwrap().net_charge_e(), 0.0);
    assert_eq!(lookup_solute_template("TYR").unwrap().net_charge_e(), 0.0);
    assert_eq!(lookup_solute_template("ARG").unwrap().net_charge_e(), 1.0);
}

#[test]
fn aromatic_small_molecule_templates_match_reference_imports() {
    let benz = lookup_solute_template("BENZ").unwrap();
    assert_eq!(benz.beads.len(), 3);
    assert_eq!(benz.net_charge_e(), 0.0);
    assert_eq!(benz.beads[0].name, "R1");
    assert!((benz.beads[0].offset_angstrom[0] - 1.053333).abs() < 1.0e-6);

    let tolu = lookup_solute_template("TOLU").unwrap();
    assert_eq!(tolu.beads.len(), 3);
    assert_eq!(tolu.net_charge_e(), 0.0);
    assert_eq!(tolu.beads[1].name, "R2");

    let enaph = lookup_solute_template("ENAPH").unwrap();
    assert_eq!(enaph.beads.len(), 6);
    assert_eq!(enaph.net_charge_e(), 0.0);
    assert_eq!(enaph.beads[0].name, "C1");
    assert!((enaph.beads[5].offset_angstrom[1] + 2.55).abs() < 1.0e-6);
}

#[test]
fn sugar_template_matches_reference_import() {
    let sucrose = lookup_solute_template("SUCR").unwrap();
    assert_eq!(sucrose.beads.len(), 8);
    assert_eq!(sucrose.net_charge_e(), 0.0);
    assert_eq!(sucrose.beads[0].name, "A");
    assert!((sucrose.beads[0].offset_angstrom[0] - 0.0125).abs() < 1.0e-6);
    assert_eq!(sucrose.beads[7].name, "VS");
    assert!((sucrose.beads[7].offset_angstrom[2] - 1.5).abs() < 1.0e-6);

    let alias = lookup_solute_template("SUCROSE").unwrap();
    assert_eq!(alias.beads.len(), sucrose.beads.len());
    assert_eq!(alias.net_charge_e(), sucrose.net_charge_e());
}

#[test]
fn osmolyte_templates_match_reference_imports() {
    let glyl = lookup_solute_template("GLYL").unwrap();
    assert_eq!(glyl.beads.len(), 1);
    assert_eq!(glyl.net_charge_e(), 0.0);
    assert_eq!(glyl.beads[0].name, "P01");

    let put = lookup_solute_template("PUT").unwrap();
    assert_eq!(put.beads.len(), 2);
    assert_eq!(put.net_charge_e(), 0.0);
    assert_eq!(put.beads[0].name, "P01");
    assert!((put.beads[0].offset_angstrom[0] - 0.845).abs() < 1.0e-6);
    assert!((put.beads[1].offset_angstrom[2] - 1.305).abs() < 1.0e-6);

    let sper = lookup_solute_template("SPER").unwrap();
    assert_eq!(sper.beads.len(), 3);
    assert_eq!(sper.net_charge_e(), 0.0);
    assert_eq!(sper.beads[1].name, "C01");
    assert!((sper.beads[2].offset_angstrom[1] + 1.87).abs() < 1.0e-6);

    let urea = lookup_solute_template("UREA").unwrap();
    assert_eq!(urea.beads.len(), 1);
    assert_eq!(urea.net_charge_e(), 0.0);

    let treh = lookup_solute_template("TREH").unwrap();
    assert_eq!(treh.beads.len(), 9);
    assert_eq!(treh.net_charge_e(), 0.0);
    assert_eq!(treh.beads[0].name, "S01");
    assert_eq!(treh.beads[8].name, "S06");
    assert!((treh.beads[0].offset_angstrom[0] - 1.518889).abs() < 1.0e-6);
    assert!((treh.beads[8].offset_angstrom[2] - 2.247778).abs() < 1.0e-6);
}

#[test]
fn ionic_liquid_templates_match_reference_imports() {
    let c1 = lookup_solute_template("C1").unwrap();
    assert_eq!(c1.beads.len(), 3);
    assert_eq!(c1.net_charge_e(), 1.0);
    assert_eq!(c1.beads[0].name, "SI1");
    assert!((c1.beads[0].offset_angstrom[0] - 0.583333).abs() < 1.0e-6);

    let dim = lookup_solute_template("DIM").unwrap();
    assert_eq!(dim.beads.len(), 6);
    assert_eq!(dim.net_charge_e(), 1.0);
    assert_eq!(dim.beads[5].name, "SI6");
    assert!((dim.beads[5].offset_angstrom[2] + 2.208333).abs() < 1.0e-6);

    let bf4 = lookup_solute_template("BF4").unwrap();
    assert_eq!(bf4.beads.len(), 1);
    assert_eq!(bf4.net_charge_e(), -1.0);
}

#[test]
fn dna_nucleotide_templates_match_reference_topology_charges() {
    let da = lookup_solute_template("DA").unwrap();
    assert_eq!(da.beads.len(), 7);
    assert_eq!(da.net_charge_e(), -1.0);
    assert_eq!(da.beads[0].name, "BB1");
    assert!((da.beads[0].offset_angstrom[0] - 3.763286).abs() < 1.0e-6);

    let dc = lookup_solute_template("DC").unwrap();
    assert_eq!(dc.beads.len(), 6);
    assert_eq!(dc.net_charge_e(), -1.0);

    let dg = lookup_solute_template("DG").unwrap();
    assert_eq!(dg.beads.len(), 7);
    assert_eq!(dg.net_charge_e(), -1.0);

    let dt = lookup_solute_template("DT").unwrap();
    assert_eq!(dt.beads.len(), 6);
    assert_eq!(dt.net_charge_e(), -1.0);
}

#[test]
fn nucleobase_templates_match_reference_topology_charges() {
    let cases = [
        ("ADEN", 6_usize),
        ("CYTO", 4_usize),
        ("GUAN", 6_usize),
        ("THYM", 5_usize),
        ("URAC", 5_usize),
    ];
    for (name, bead_count) in cases {
        let template = lookup_solute_template(name).unwrap();
        assert_eq!(template.beads.len(), bead_count);
        assert_eq!(template.net_charge_e(), 0.0);
        assert_eq!(template.beads[0].name, "N1");
    }
    assert_eq!(lookup_solute_template("ADEN").unwrap().beads[5].name, "N6");
    assert_eq!(lookup_solute_template("URAC").unwrap().beads[4].name, "N5");
}

#[test]
fn sirah_templates_match_reference_topology_charges() {
    let wt4 = lookup_solute_template("WT4").unwrap();
    assert_eq!(wt4.beads.len(), 4);
    assert_eq!(wt4.net_charge_e(), 0.0);
    assert_eq!(wt4.beads[0].name, "WN1");
    assert_eq!(wt4.beads[0].charge_e, -0.41);
    assert_eq!(wt4.beads[3].name, "WP2");
    assert_eq!(wt4.beads[3].charge_e, 0.41);
    let wt4_bonds = lookup_solute_template_bonds("WT4");
    assert_eq!(wt4_bonds.len(), 6);
    assert_eq!(wt4_bonds[0].bead_indices, [0, 1]);
    assert_eq!(wt4_bonds[0].length_nm, 0.45);
    assert_eq!(wt4_bonds[0].force_kj_mol_nm2, 4184.0);

    let naw = lookup_solute_template("NaW").unwrap();
    assert_eq!(naw.beads.len(), 1);
    assert_eq!(naw.beads[0].name, "NaW");
    assert_eq!(naw.net_charge_e(), 1.0);

    let clw = lookup_solute_template("ClW").unwrap();
    assert_eq!(clw.beads.len(), 1);
    assert_eq!(clw.beads[0].name, "ClW");
    assert_eq!(clw.net_charge_e(), -1.0);
}
