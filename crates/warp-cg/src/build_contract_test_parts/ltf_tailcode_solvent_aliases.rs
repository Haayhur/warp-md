use super::*;

#[test]
fn coby_ltf_named_tailcode_solvent_aliases_resolve() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "IC8".to_string(),
        ..SolventPolicy::default()
    };
    let hydrocarbon = resolved_solvent_species(&solvent);
    assert_eq!(hydrocarbon[0].name, "IC8");
    assert_eq!(hydrocarbon[0].beads.len(), 5);
    assert_eq!(hydrocarbon[0].beads[0].atom_name, "C1A");
    assert_eq!(hydrocarbon[0].beads[1].atom_name, "D2A");
    assert_eq!(hydrocarbon[0].beads[3].atom_name, "D4A");
    assert_eq!(hydrocarbon[0].charge_e, 0.0);

    solvent.name = "LNA".to_string();
    let fatty_acid = resolved_solvent_species(&solvent);
    assert_eq!(fatty_acid[0].name, "LNA");
    assert_eq!(fatty_acid[0].beads.len(), 5);
    assert_eq!(fatty_acid[0].beads[0].atom_name, "COO");
    assert_eq!(fatty_acid[0].beads[0].charge_e, -1.0);
    assert_eq!(fatty_acid[0].beads[4].atom_name, "D4A");
    assert_eq!(fatty_acid[0].charge_e, -1.0);

    solvent.name = "MLN".to_string();
    let monoglyceride = resolved_solvent_species(&solvent);
    assert_eq!(monoglyceride[0].name, "MLN");
    assert_eq!(monoglyceride[0].beads.len(), 6);
    assert_eq!(monoglyceride[0].beads[0].atom_name, "DOH");
    assert_eq!(monoglyceride[0].beads[1].atom_name, "GL1");
    assert_eq!(monoglyceride[0].beads[5].atom_name, "D4A");
    assert_eq!(monoglyceride[0].charge_e, 0.0);

    solvent.name = "TLN".to_string();
    let triglyceride = resolved_solvent_species(&solvent);
    assert_eq!(triglyceride[0].name, "TLN");
    assert_eq!(triglyceride[0].beads.len(), 15);
    assert_eq!(triglyceride[0].beads[0].atom_name, "GL1");
    assert_eq!(triglyceride[0].beads[6].atom_name, "D4A");
    assert_eq!(triglyceride[0].beads[10].atom_name, "D4B");
    assert_eq!(triglyceride[0].beads[14].atom_name, "D4C");
    assert_eq!(triglyceride[0].charge_e, 0.0);

    solvent.name = "DODG".to_string();
    let dioleoyl_diglyceride = resolved_solvent_species(&solvent);
    assert_eq!(dioleoyl_diglyceride[0].name, "DODG");
    assert_eq!(dioleoyl_diglyceride[0].beads.len(), 11);
    assert_eq!(dioleoyl_diglyceride[0].beads[0].atom_name, "COH");
    assert_eq!(dioleoyl_diglyceride[0].beads[1].atom_name, "GL1");
    assert_eq!(dioleoyl_diglyceride[0].beads[4].atom_name, "D2A");
    assert_eq!(dioleoyl_diglyceride[0].beads[8].atom_name, "D2B");
    assert_eq!(dioleoyl_diglyceride[0].charge_e, 0.0);

    solvent.name = "YODG".to_string();
    let mixed_diglyceride = resolved_solvent_species(&solvent);
    assert_eq!(mixed_diglyceride[0].name, "YODG");
    assert_eq!(mixed_diglyceride[0].beads[4].atom_name, "D2A");
    assert_eq!(mixed_diglyceride[0].beads[9].atom_name, "D3B");

    solvent.name = "LFDG".to_string();
    let polyunsaturated_diglyceride = resolved_solvent_species(&solvent);
    assert_eq!(polyunsaturated_diglyceride[0].name, "LFDG");
    assert_eq!(polyunsaturated_diglyceride[0].beads[5].atom_name, "D3A");
    assert_eq!(polyunsaturated_diglyceride[0].beads[8].atom_name, "D2B");
    assert_eq!(polyunsaturated_diglyceride[0].beads[9].atom_name, "D3B");

    let known = known_solvent_library_names();
    for name in [
        "C24", "IC9", "OA", "LNA", "MO", "MLN", "DODG", "YODG", "LFDG", "TO", "TLN",
    ] {
        assert!(known.contains(&name), "{name}");
    }
}
