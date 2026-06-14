use crate::build_solutes::{known_solute_names, lookup_solute_template};

#[test]
fn inserted_solute_templates_cover_coby_martini3_amino_acid_registry() {
    let expected = [
        ("GLY", 1_usize, 0.0_f32),
        ("ALA", 2, 0.0),
        ("CYS", 2, 0.0),
        ("VAL", 2, 0.0),
        ("LEU", 2, 0.0),
        ("ILE", 2, 0.0),
        ("MET", 2, 0.0),
        ("PRO", 2, 0.0),
        ("HYP", 2, 0.0),
        ("ASN", 2, 0.0),
        ("GLN", 2, 0.0),
        ("THR", 2, 0.0),
        ("SER", 2, 0.0),
        ("ASPP", 2, 0.0),
        ("ASH", 2, 0.0),
        ("GLUP", 2, 0.0),
        ("GLH", 2, 0.0),
        ("ASP", 2, -1.0),
        ("GLU", 2, -1.0),
        ("ARG", 3, 1.0),
        ("LYS", 3, 1.0),
        ("LSN", 3, 0.0),
        ("LYN", 3, 0.0),
        ("PHE", 4, 0.0),
        ("HIS", 4, 0.0),
        ("HIE", 4, 0.0),
        ("HSE", 4, 0.0),
        ("HSD", 4, 0.0),
        ("HID", 4, 0.0),
        ("HSP", 4, 1.0),
        ("HIP", 4, 1.0),
        ("TYR", 5, 0.0),
        ("TRP", 6, 0.0),
    ];

    let known = known_solute_names();
    for (name, bead_count, net_charge) in expected {
        assert!(known.contains(&name));
        let template = lookup_solute_template(name).unwrap();
        assert_eq!(
            template.source,
            format!("martini3_amino_acid_library.{name}")
        );
        assert_eq!(template.beads.len(), bead_count);
        assert!((template.net_charge_e() - net_charge).abs() < 1.0e-6);
    }
}

#[test]
fn amino_acid_inserted_templates_preserve_reference_bead_layout_families() {
    let asp = lookup_solute_template("ASP").unwrap();
    assert_eq!(asp.beads[0].name, "BB");
    assert_eq!(asp.beads[1].name, "SC1");
    assert_eq!(asp.beads[0].offset_angstrom, [2.5, 0.0, 0.0]);
    assert_eq!(asp.beads[1].offset_angstrom, [-2.5, 0.0, 0.0]);
    assert_eq!(asp.beads[1].charge_e, -1.0);

    let hip = lookup_solute_template("HIP").unwrap();
    assert_eq!(hip.beads[2].name, "SC2");
    assert_eq!(hip.beads[3].name, "SC3");
    assert_eq!(hip.beads[2].charge_e, 0.5);
    assert_eq!(hip.beads[3].charge_e, 0.5);

    let trp = lookup_solute_template("TRP").unwrap();
    assert_eq!(trp.beads[5].name, "SC5");
    assert_eq!(trp.beads[5].offset_angstrom, [-2.5, 1.25, 0.0]);
}
