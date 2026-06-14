use super::*;

#[test]
fn popg_template_carries_net_negative_charge() {
    let template = lookup_lipid_template("POPG", "martini3").unwrap();
    assert_eq!(template.net_charge_e, -1.0);
    assert_eq!(
        template.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
        -1.0
    );
}

#[test]
fn popi_template_carries_pi_headgroup_charge() {
    let template = lookup_lipid_template("POPI", "martini3").unwrap();
    assert_eq!(template.net_charge_e, -1.0);
    assert!(template.beads.iter().any(|bead| bead.name == "C1"));
    assert_eq!(
        template.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
        -1.0
    );
}

#[test]
fn phosphoinositide_variants_carry_reference_head_charges() {
    let expected: &[(&str, f32, &[&str], &[(&str, f32)])] = &[
        ("POP1", -3.0, &["P3"], &[("P3", -2.0)]),
        ("POP2", -4.0, &["P3", "P4"], &[("P3", -1.5), ("P4", -1.5)]),
        (
            "POP3",
            -5.0,
            &["P3", "P4", "P5"],
            &[("P3", -1.3), ("P4", -1.4), ("P5", -1.3)],
        ),
        ("POP4", -3.0, &["P4"], &[("P4", -2.0)]),
        ("POP5", -3.0, &["P5"], &[("P5", -2.0)]),
        ("POP6", -4.0, &["P4", "P5"], &[("P4", -1.5), ("P5", -1.5)]),
        ("POP7", -4.0, &["P3", "P5"], &[("P3", -1.5), ("P5", -1.5)]),
    ];

    for (name, net_charge, phosphate_beads, bead_charges) in expected {
        let template = lookup_lipid_template(name, "martini3").unwrap();
        assert_eq!(
            template.source,
            "warp-cg.lipid-template.martini-inositol.v1"
        );
        assert_eq!(template.net_charge_e, *net_charge, "{name}");
        assert!(
            template.beads.iter().any(|bead| bead.name == "PO4"),
            "{name}"
        );
        for bead_name in ["GL1", "GL2", "C1A", "C1B"] {
            assert!(
                template.beads.iter().any(|bead| bead.name == bead_name),
                "{name}:{bead_name}"
            );
        }
        for phosphate in *phosphate_beads {
            assert!(
                template.beads.iter().any(|bead| bead.name == *phosphate),
                "{name}"
            );
        }
        for (bead_name, charge) in *bead_charges {
            let bead = template
                .beads
                .iter()
                .find(|bead| bead.name == *bead_name)
                .unwrap();
            assert_eq!(bead.charge_e, *charge, "{name}:{bead_name}");
        }
        assert!(
            (template.beads.iter().map(|bead| bead.charge_e).sum::<f32>() - *net_charge).abs()
                < 1.0e-6,
            "{name}"
        );
    }
}

#[test]
fn linker_suffix_templates_swap_glycerol_and_ether_beads() {
    let default = lookup_lipid_template("POPC", "martini3").unwrap();
    let glycerol = lookup_lipid_template("POPC.GL", "martini3").unwrap();
    let ether = lookup_lipid_template("POPC.ET", "martini3").unwrap();
    let inositol_ether = lookup_lipid_template("POP2.ET", "martini3").unwrap();

    assert_eq!(default.source, "warp-cg.lipid-template.martini-diacyl.v1");
    assert_eq!(glycerol.source, "warp-cg.lipid-template.martini-diacyl.v1");
    assert_eq!(ether.source, "warp-cg.lipid-template.martini-ether.v1");
    assert!(glycerol.beads.iter().any(|bead| bead.name == "GL1"));
    assert!(glycerol.beads.iter().any(|bead| bead.name == "GL2"));
    assert!(ether.beads.iter().any(|bead| bead.name == "ET1"));
    assert!(ether.beads.iter().any(|bead| bead.name == "ET2"));
    assert!(!ether.beads.iter().any(|bead| bead.name == "GL1"));
    assert_eq!(ether.net_charge_e, 0.0);
    assert_eq!(
        inositol_ether.source,
        "warp-cg.lipid-template.martini-inositol-ether.v1"
    );
    assert!(inositol_ether.beads.iter().any(|bead| bead.name == "ET1"));
    assert!(inositol_ether.beads.iter().any(|bead| bead.name == "ET2"));
    assert_eq!(inositol_ether.net_charge_e, -4.0);
    assert!(
        (inositol_ether
            .beads
            .iter()
            .map(|bead| bead.charge_e)
            .sum::<f32>()
            + 4.0)
            .abs()
            < 1.0e-6
    );
    assert!(lookup_lipid_template("PODG.ET", "martini3").is_none());
}

#[test]
fn plasmalogen_templates_use_pl_linker_and_reference_tail_set() {
    let oplc = lookup_lipid_template("OPLC", "martini3").unwrap();
    let aple = lookup_lipid_template("APLE", "martini3").unwrap();
    let dplc = lookup_lipid_template("DPLC", "martini3").unwrap();

    assert_eq!(oplc.source, "warp-cg.lipid-template.martini-plasmalogen.v1");
    assert_eq!(oplc.net_charge_e, 0.0);
    assert!(oplc.beads.iter().any(|bead| bead.name == "NC3"));
    assert!(oplc.beads.iter().any(|bead| bead.name == "PO4"));
    assert!(oplc.beads.iter().any(|bead| bead.name == "GL1"));
    assert!(oplc.beads.iter().any(|bead| bead.name == "PL2"));
    assert!(oplc.beads.iter().any(|bead| bead.name == "D2A"));
    assert!(aple.beads.iter().any(|bead| bead.name == "NH3"));
    assert!(aple.beads.iter().any(|bead| bead.name == "C5A"));
    assert!(dplc.beads.iter().any(|bead| bead.name == "D1A"));
    assert!(dplc.beads.iter().any(|bead| bead.name == "D5A"));
}

#[test]
fn popc_template_is_multibead_and_neutral() {
    let template = lookup_lipid_template("POPC", "martini3").unwrap();
    assert!(template.beads.len() > 1);
    assert_eq!(template.net_charge_e, 0.0);
}

#[test]
fn dlpc_and_dlpe_templates_cover_nanodisc_tutorial_lipids() {
    let dlpc = lookup_lipid_template("DLPC", "martini3").unwrap();
    let dlpe = lookup_lipid_template("DLPE", "martini3").unwrap();
    assert_eq!(dlpc.net_charge_e, 0.0);
    assert_eq!(dlpe.net_charge_e, 0.0);
    assert_eq!(dlpc.beads.len(), 12);
    assert_eq!(dlpe.beads.len(), 12);
    assert!(dlpc.beads.iter().any(|bead| bead.name == "NC3"));
    assert!(dlpe.beads.iter().any(|bead| bead.name == "NH3"));
}

#[test]
fn expanded_diacyl_templates_cover_common_heads_and_tails() {
    let dops = lookup_lipid_template("DOPS", "martini3").unwrap();
    let dppc = lookup_lipid_template("DPPC", "martini3").unwrap();
    let sapienic_pe = lookup_lipid_template("SAPE", "martini3").unwrap();

    assert_eq!(dops.net_charge_e, -1.0);
    assert!(dops.beads.iter().any(|bead| bead.name == "CNO"));
    assert!(dops.beads.iter().any(|bead| bead.name == "D2B"));
    assert_eq!(
        dops.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
        -1.0
    );
    assert_eq!(dppc.net_charge_e, 0.0);
    assert!(dppc.beads.iter().all(|bead| !bead.name.starts_with('D')));
    assert!(sapienic_pe.beads.iter().any(|bead| bead.name == "C5A"));
    assert!(sapienic_pe.beads.iter().any(|bead| bead.name == "NH3"));
}

#[test]
fn coby_ltf_release_diacyl_tail_aliases_use_reference_bead_layouts() {
    let cases = [
        ("DUPC", 10, &["C3A", "C3B"][..], &["D1A", "D5A"][..]),
        ("DPPC", 12, &["C4A", "C4B", "NC3"][..], &["C5A"][..]),
        ("DKPG", 14, &["C5A", "C5B", "GL0"][..], &["D5A"][..]),
        ("DRPG", 10, &["D2A", "D2B", "GL0"][..], &["D6A"][..]),
        ("DYPE", 12, &["D3A", "D3B", "NH3"][..], &["D2A"][..]),
        ("OEPS", 13, &["D3A", "D2B", "CNO"][..], &["C5B"][..]),
        ("DNPI", 19, &["D4A", "D4B", "C6A", "C6B"][..], &["D6A"][..]),
        (
            "DUPC.ET",
            10,
            &["ET1", "ET2", "C3A", "C3B"][..],
            &["GL1"][..],
        ),
    ];

    for (name, bead_count, expected_present, expected_absent) in cases {
        let template = lookup_lipid_template(name, "martini3").unwrap();
        let bead_names = template
            .beads
            .iter()
            .map(|bead| bead.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(template.name, name, "{name}");
        assert_eq!(template.beads.len(), bead_count, "{name}");
        for bead_name in expected_present {
            assert!(bead_names.contains(bead_name), "{name}:{bead_name}");
        }
        for bead_name in expected_absent {
            assert!(!bead_names.contains(bead_name), "{name}:{bead_name}");
        }
    }
}

#[test]
fn coby_tailcode_diacyl_templates_are_generated() {
    let abpc = lookup_lipid_template("ABPC", "martini3").unwrap();
    let rrpg = lookup_lipid_template("RRPG", "martini3").unwrap();

    assert_eq!(abpc.source, "warp-cg.lipid-template.martini-diacyl.v1");
    assert!(abpc.beads.iter().any(|bead| bead.name == "C5A"));
    assert!(abpc.beads.iter().any(|bead| bead.name == "D4B"));
    assert_eq!(rrpg.net_charge_e, -1.0);
    assert!(rrpg.beads.iter().any(|bead| bead.name == "D6A"));
    assert!(rrpg.beads.iter().any(|bead| bead.name == "D6B"));
}

#[test]
fn coby_monoacyl_scaffold_gmo_is_template() {
    let gmo = lookup_lipid_template("GMO", "martini3").unwrap();
    let bead_names = gmo
        .beads
        .iter()
        .map(|bead| bead.name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(gmo.source, "warp-cg.lipid-template.martini-monoacyl.v1");
    assert_eq!(gmo.net_charge_e, 0.0);
    assert_eq!(
        bead_names,
        ["GL1", "GL2", "C1A", "C2A", "D3A", "C4A", "C5A"]
    );
    assert_eq!(gmo.beads.iter().map(|bead| bead.charge_e).sum::<f32>(), 0.0);
}

#[test]
fn coby_ltf_named_monoglyceride_lipid_scaffolds_are_templates() {
    let expected = [
        ("MO", ["DOH", "GL1", "C1A", "D2A", "C3A", "C4A"].as_slice()),
        ("ML", ["DOH", "GL1", "C1A", "D2A", "D3A", "C4A"].as_slice()),
        ("MLN", ["DOH", "GL1", "C1A", "D2A", "D3A", "D4A"].as_slice()),
        ("MS", ["DOH", "GL1", "C1A", "C2A", "C3A", "C4A"].as_slice()),
        ("MP", ["DOH", "GL1", "C1A", "C2A", "C3A", "C4A"].as_slice()),
        ("MM", ["DOH", "GL1", "C1A", "C2A", "C3A"].as_slice()),
        ("MLA", ["DOH", "GL1", "C1A", "C2A", "C3A"].as_slice()),
    ];

    for (name, bead_names) in expected {
        let template = lookup_lipid_template(name, "martini3").unwrap();
        let actual = template
            .beads
            .iter()
            .map(|bead| bead.name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            template.source,
            "warp-cg.lipid-template.martini-ltf-monoglyceride.v1"
        );
        assert_eq!(template.net_charge_e, 0.0);
        assert_eq!(actual, bead_names);
        assert_eq!(
            template.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
            0.0
        );
    }
}

#[test]
fn coby_ltf_named_triglyceride_lipid_scaffolds_are_templates() {
    let expected = [
        (
            "TO",
            &[
                "GL1", "C1A", "D2A", "C3A", "C4A", "GL2", "C1B", "D2B", "C3B", "C4B", "GL3", "C1C",
                "D2C", "C3C", "C4C",
            ][..],
        ),
        (
            "TLN",
            &[
                "GL1", "C1A", "D2A", "D3A", "D4A", "GL2", "C1B", "D2B", "D3B", "D4B", "GL3", "C1C",
                "D2C", "D3C", "D4C",
            ][..],
        ),
        (
            "TM",
            &[
                "GL1", "C1A", "C2A", "C3A", "GL2", "C1B", "C2B", "C3B", "GL3", "C1C", "C2C", "C3C",
            ][..],
        ),
    ];

    for (name, bead_names) in expected {
        let template = lookup_lipid_template(name, "martini3").unwrap();
        let actual = template
            .beads
            .iter()
            .map(|bead| bead.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            template.source,
            "warp-cg.lipid-template.martini-ltf-triglyceride.v1"
        );
        assert_eq!(template.net_charge_e, 0.0);
        assert_eq!(actual, bead_names);
        assert_eq!(
            template.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
            0.0
        );
    }
}

#[test]
fn coby_ltf_single_chain_lipid_scaffolds_are_templates() {
    let c24 = lookup_lipid_template("C24", "martini3").unwrap();
    let ic7 = lookup_lipid_template("IC7", "martini3").unwrap();
    let lna = lookup_lipid_template("LNA", "martini3").unwrap();
    let sa = lookup_lipid_template("SA", "martini3").unwrap();

    assert_eq!(
        c24.source,
        "warp-cg.lipid-template.martini-ltf-hydrocarbon.v1"
    );
    assert_eq!(c24.net_charge_e, 0.0);
    assert_eq!(
        c24.beads
            .iter()
            .map(|bead| bead.name.as_str())
            .collect::<Vec<_>>(),
        ["C1A", "C2A", "C3A", "C4A", "C5A", "C6A"]
    );
    assert!(ic7.beads.iter().any(|bead| bead.name == "D4A"));
    assert!(ic7.beads.iter().any(|bead| bead.name == "D5A"));

    assert_eq!(
        lna.source,
        "warp-cg.lipid-template.martini-ltf-fatty-acid.v1"
    );
    assert_eq!(lna.net_charge_e, -1.0);
    assert_eq!(
        lna.beads
            .iter()
            .map(|bead| bead.name.as_str())
            .collect::<Vec<_>>(),
        ["COO", "C1A", "D2A", "D3A", "D4A"]
    );
    assert_eq!(
        lna.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
        -1.0
    );
    assert_eq!(
        sa.beads
            .iter()
            .map(|bead| bead.name.as_str())
            .collect::<Vec<_>>(),
        ["COO", "C1A", "C2A", "C3A", "C4A"]
    );
}

#[test]
fn coby_ltf_named_diglyceride_lipid_scaffolds_are_templates() {
    let cases = [
        (
            "DTDG",
            &["OH", "GL1", "GL2", "C1A", "C2A", "C1B", "C2B"][..],
        ),
        (
            "DODG",
            &[
                "OH", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A", "C1B", "D2B", "C3B", "C4B",
            ][..],
        ),
        (
            "OEDG",
            &[
                "OH", "GL1", "GL2", "C1A", "C2A", "D3A", "C4A", "C5A", "C1B", "D2B", "C3B", "C4B",
            ][..],
        ),
        (
            "LFDG",
            &[
                "OH", "GL1", "GL2", "C1A", "D2A", "D3A", "D4A", "C1B", "D2B", "D3B", "C4B",
            ][..],
        ),
    ];

    for (name, bead_names) in cases {
        let template = lookup_lipid_template(name, "martini3").unwrap();
        let actual = template
            .beads
            .iter()
            .map(|bead| bead.name.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            template.source,
            "warp-cg.lipid-template.martini-ltf-diglyceride.v1"
        );
        assert_eq!(template.net_charge_e, 0.0);
        assert_eq!(actual, bead_names, "{name}");
        assert_eq!(
            template.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
            0.0
        );
    }
}

#[test]
fn coby_ltf_named_lipid_scaffolds_are_templates() {
    let tmcl = lookup_lipid_template("TMCL", "martini3").unwrap();
    let tocl = lookup_lipid_template("TOCL", "martini3").unwrap();
    let do2b = lookup_lipid_template("DO2B", "martini3").unwrap();
    let do3b = lookup_lipid_template("DO3B", "martini3").unwrap();

    assert_eq!(
        tmcl.source,
        "warp-cg.lipid-template.martini-ltf-cardiolipin.v1"
    );
    assert_eq!(tmcl.net_charge_e, 0.0);
    assert!(tmcl.beads.iter().any(|bead| bead.name == "GLC"));
    assert!(tmcl.beads.iter().any(|bead| bead.name == "PO41"));
    assert!(tmcl.beads.iter().any(|bead| bead.name == "C3B1"));
    assert_eq!(tocl.beads.len(), 23);
    assert!(tocl.beads.iter().any(|bead| bead.name == "C4A2"));

    assert_eq!(do2b.source, "warp-cg.lipid-template.martini-ltf-bmp.v1");
    assert_eq!(do2b.net_charge_e, -1.0);
    assert!(do2b.beads.iter().any(|bead| bead.name == "OH1"));
    assert!(do2b.beads.iter().any(|bead| bead.name == "D2A"));
    assert_eq!(do3b.net_charge_e, -1.0);
    assert!(do3b.beads.iter().any(|bead| bead.name == "GL2"));
    assert!(do3b.beads.iter().any(|bead| bead.name == "C4B"));
}

#[test]
fn coby_ltf_sphingolipid_templates_are_generated() {
    let bsm = lookup_lipid_template("BSM", "martini3").unwrap();
    let ncer = lookup_lipid_template("NCER", "martini3").unwrap();
    let xsm = lookup_lipid_template("XSM", "martini3").unwrap();

    assert_eq!(
        bsm.source,
        "warp-cg.lipid-template.martini-ltf-sphingolipid.v1"
    );
    assert_eq!(bsm.net_charge_e, 0.0);
    assert_eq!(bsm.beads.iter().map(|bead| bead.charge_e).sum::<f32>(), 0.0);
    assert!(bsm
        .beads
        .iter()
        .any(|bead| bead.name == "NC3" && bead.charge_e == 1.0));
    assert!(bsm
        .beads
        .iter()
        .any(|bead| bead.name == "PO4" && bead.charge_e == -1.0));
    assert!(bsm.beads.iter().any(|bead| bead.name == "C5B"));

    assert_eq!(ncer.beads.len(), 13);
    assert!(ncer.beads.iter().any(|bead| bead.name == "D4B"));
    assert_eq!(
        ncer.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
        0.0
    );
    assert_eq!(xsm.beads.len(), 14);
    assert!(xsm.beads.iter().any(|bead| bead.name == "C6B"));
}

#[test]
fn coby_generated_sphingomyelin_scaffolds_are_templates() {
    let cases = [
        (
            "ABSM",
            &["NC3", "PO4", "AM1", "AM2", "T1A", "C4A", "D1B", "C5B"][..],
        ),
        ("APSM", &["T1A", "C3A", "D1B", "C5B"][..]),
        ("DXSM", &["T1A", "C5A", "C1B", "C6B"][..]),
        ("BXSM", &["T1A", "C5A", "C1B", "C5B"][..]),
    ];

    for (name, expected_beads) in cases {
        let template = lookup_lipid_template(name, "martini3").unwrap();
        assert_eq!(
            template.source,
            "warp-cg.lipid-template.martini-generated-sphingomyelin.v1"
        );
        assert_eq!(template.net_charge_e, 0.0, "{name}");
        assert_eq!(
            template.beads.iter().map(|bead| bead.charge_e).sum::<f32>(),
            0.0,
            "{name}"
        );
        for bead_name in expected_beads {
            assert!(
                template.beads.iter().any(|bead| bead.name == *bead_name),
                "{name}:{bead_name}"
            );
        }
    }
}

#[test]
fn saturated_aliases_keep_existing_template_names() {
    let dmpc = lookup_lipid_template("DMPC", "martini3").unwrap();
    let dmpe = lookup_lipid_template("DMPE", "martini3").unwrap();

    assert_eq!(dmpc.name, "DLPC");
    assert_eq!(dmpe.name, "DLPE");
    assert_eq!(dmpc.net_charge_e, 0.0);
    assert_eq!(dmpe.net_charge_e, 0.0);
}

#[test]
fn known_lipids_reports_expanded_builder_surface() {
    let lipids = known_lipids();

    assert!(lipids.contains(&"POPC".to_string()));
    assert!(lipids.contains(&"DOPS".to_string()));
    assert!(lipids.contains(&"DPPC".to_string()));
    assert!(lipids.contains(&"POPC.GL".to_string()));
    assert!(lipids.contains(&"POPC.ET".to_string()));
    assert!(lipids.contains(&"POP1".to_string()));
    assert!(lipids.contains(&"POP1.ET".to_string()));
    assert!(lipids.contains(&"POP7".to_string()));
    assert!(lipids.contains(&"DUPC".to_string()));
    assert!(lipids.contains(&"DUPC.ET".to_string()));
    assert!(lipids.contains(&"DNPI".to_string()));
    assert!(lipids.contains(&"OEPS".to_string()));
    assert!(lipids.contains(&"OPLC".to_string()));
    assert!(lipids.contains(&"DPLE".to_string()));
    assert!(lipids.contains(&"ABPC".to_string()));
    assert!(lipids.contains(&"DRPG".to_string()));
    assert!(lipids.contains(&"TMCL".to_string()));
    assert!(lipids.contains(&"TOCL".to_string()));
    assert!(lipids.contains(&"DO2B".to_string()));
    assert!(lipids.contains(&"DO3B".to_string()));
    assert!(lipids.contains(&"BSM".to_string()));
    assert!(lipids.contains(&"NCER".to_string()));
    assert!(lipids.contains(&"XSM".to_string()));
    assert!(lipids.contains(&"ABSM".to_string()));
    assert!(lipids.contains(&"APSM".to_string()));
    assert!(lipids.contains(&"DXSM".to_string()));
    assert!(lipids.contains(&"GMO".to_string()));
    assert!(lipids.contains(&"MLN".to_string()));
    assert!(lipids.contains(&"MLA".to_string()));
    assert!(lipids.contains(&"TO".to_string()));
    assert!(lipids.contains(&"TLN".to_string()));
    assert!(lipids.contains(&"TLA".to_string()));
    assert!(lipids.contains(&"C24".to_string()));
    assert!(lipids.contains(&"IC7".to_string()));
    assert!(lipids.contains(&"LNA".to_string()));
    assert!(lipids.contains(&"LFDG".to_string()));
    assert!(lipids.contains(&"CHOL".to_string()));
    assert!(lipids.len() > 5_000);
}
