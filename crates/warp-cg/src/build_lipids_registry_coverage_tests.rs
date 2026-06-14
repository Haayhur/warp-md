use super::*;

#[test]
fn coby_registry_edge_lipid_names_are_generated() {
    let cases = [
        (
            "SAPI",
            "warp-cg.lipid-template.martini-diacyl.v1",
            -1.0,
            &["C1", "PO4", "C1A", "C4B"][..],
        ),
        (
            "YXSM",
            "warp-cg.lipid-template.martini-generated-sphingomyelin.v1",
            0.0,
            &["NC3", "PO4", "C5A", "C3B"][..],
        ),
        (
            "YXDG",
            "warp-cg.lipid-template.martini-diacyl.v1",
            0.0,
            &["OH", "C6A", "C3B"][..],
        ),
        (
            "DAP7.ET",
            "warp-cg.lipid-template.martini-inositol-ether.v1",
            -4.0,
            &["P3", "P5", "ET1", "D4A"][..],
        ),
        (
            "YOPI.GL",
            "warp-cg.lipid-template.martini-diacyl.v1",
            -1.0,
            &["C1", "PO4", "GL1", "D3B"][..],
        ),
        (
            "APLC",
            "warp-cg.lipid-template.martini-plasmalogen.v1",
            0.0,
            &["NC3", "PL2", "D4A"][..],
        ),
    ];

    for (name, source, charge, required_beads) in cases {
        let template = lookup_lipid_template(name, "martini3").unwrap();
        assert_eq!(template.source, source, "{name}");
        assert_eq!(template.net_charge_e, charge, "{name}");
        for bead_name in required_beads {
            assert!(
                template.beads.iter().any(|bead| bead.name == *bead_name),
                "{name}:{bead_name}"
            );
        }
    }
}

#[test]
fn known_lipids_reports_coby_registry_edge_names() {
    let lipids = known_lipids();

    for name in ["SAPI", "YXSM", "YXDG", "DAP7.ET", "YOPI.GL", "APLC"] {
        assert!(lipids.contains(&name.to_string()), "{name}");
    }
}
