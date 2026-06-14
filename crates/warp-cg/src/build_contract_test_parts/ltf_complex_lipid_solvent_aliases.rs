use super::*;

#[test]
fn coby_ltf_named_complex_lipid_solvent_aliases_resolve() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "DO2B".to_string(),
        ..SolventPolicy::default()
    };
    let do2b = resolved_solvent_species(&solvent);
    assert_eq!(do2b[0].name, "DO2B");
    assert_eq!(do2b[0].beads.len(), 13);
    assert_eq!(do2b[0].beads[0].atom_name, "PO4");
    assert_eq!(do2b[0].beads[1].atom_name, "OH1");
    assert_eq!(do2b[0].beads[2].atom_name, "GL1");
    assert_eq!(do2b[0].beads[4].atom_name, "D2A");
    assert_eq!(do2b[0].beads[8].atom_name, "GL2");
    assert_eq!(do2b[0].beads[10].atom_name, "D2B");
    assert_eq!(do2b[0].charge_e, -5.0);

    solvent.name = "DO3B".to_string();
    let do3b = resolved_solvent_species(&solvent);
    assert_eq!(do3b[0].name, "DO3B");
    assert_eq!(do3b[0].beads.len(), 13);
    assert_eq!(do3b[0].beads[0].atom_name, "PO4");
    assert_eq!(do3b[0].beads[1].atom_name, "GL1");
    assert_eq!(do3b[0].beads[2].atom_name, "OH1");
    assert_eq!(do3b[0].beads[4].atom_name, "D2A");
    assert_eq!(do3b[0].beads[7].atom_name, "GL2");
    assert_eq!(do3b[0].beads[10].atom_name, "D2B");
    assert_eq!(do3b[0].charge_e, -5.0);

    solvent.name = "TMCL".to_string();
    let tmcl = resolved_solvent_species(&solvent);
    assert_eq!(tmcl[0].name, "TMCL");
    assert_eq!(tmcl[0].beads.len(), 19);
    assert_eq!(tmcl[0].beads[0].atom_name, "GLC");
    assert_eq!(tmcl[0].beads[1].atom_name, "PO41");
    assert_eq!(tmcl[0].beads[4].atom_name, "C1A1");
    assert_eq!(tmcl[0].beads[10].atom_name, "PO42");
    assert_eq!(tmcl[0].beads[18].atom_name, "C3B2");
    assert_eq!(tmcl[0].charge_e, 0.0);

    solvent.name = "TOCL".to_string();
    let tocl = resolved_solvent_species(&solvent);
    assert_eq!(tocl[0].name, "TOCL");
    assert_eq!(tocl[0].beads.len(), 23);
    assert_eq!(tocl[0].beads[7].atom_name, "C4A1");
    assert_eq!(tocl[0].beads[11].atom_name, "C4B1");
    assert_eq!(tocl[0].beads[18].atom_name, "C4A2");
    assert_eq!(tocl[0].beads[22].atom_name, "C4B2");
    assert_eq!(tocl[0].charge_e, 0.0);

    let known = known_solvent_library_names();
    for name in ["DO2B", "DO3B", "TMCL", "TOCL"] {
        assert!(known.contains(&name), "{name}");
    }
}
