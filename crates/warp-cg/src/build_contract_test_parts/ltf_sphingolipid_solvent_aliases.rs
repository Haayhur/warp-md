use super::*;

#[test]
fn coby_ltf_named_sphingolipid_solvent_aliases_resolve() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "BSM".to_string(),
        ..SolventPolicy::default()
    };
    let bsm = resolved_solvent_species(&solvent);
    assert_eq!(bsm[0].name, "BSM");
    assert_eq!(bsm[0].beads.len(), 13);
    assert_eq!(bsm[0].beads[0].atom_name, "NC3");
    assert_eq!(bsm[0].beads[0].charge_e, 1.0);
    assert_eq!(bsm[0].beads[1].atom_name, "PO4");
    assert_eq!(bsm[0].beads[1].charge_e, -1.0);
    assert_eq!(bsm[0].beads[2].atom_name, "OH1");
    assert_eq!(bsm[0].beads[4].atom_name, "T1A");
    assert_eq!(bsm[0].beads[8].atom_name, "C1B");
    assert_eq!(bsm[0].beads[12].atom_name, "C5B");
    assert_eq!(bsm[0].charge_e, 0.0);

    solvent.name = "NCER".to_string();
    let ncer = resolved_solvent_species(&solvent);
    assert_eq!(ncer[0].name, "NCER");
    assert_eq!(ncer[0].beads.len(), 13);
    assert_eq!(ncer[0].beads[0].atom_name, "COH");
    assert_eq!(ncer[0].beads[1].atom_name, "OH1");
    assert_eq!(ncer[0].beads[3].atom_name, "T1A");
    assert_eq!(ncer[0].beads[10].atom_name, "D4B");
    assert_eq!(ncer[0].beads[12].atom_name, "C6B");
    assert_eq!(ncer[0].charge_e, 0.0);

    solvent.name = "XSM".to_string();
    let xsm = resolved_solvent_species(&solvent);
    assert_eq!(xsm[0].name, "XSM");
    assert_eq!(xsm[0].beads.len(), 14);
    assert_eq!(xsm[0].beads[13].atom_name, "C6B");

    let known = known_solvent_library_names();
    for name in ["BCER", "BSM", "NCER", "NSM", "OCER", "OSM", "XCER", "XSM"] {
        assert!(known.contains(&name), "{name}");
    }
}
