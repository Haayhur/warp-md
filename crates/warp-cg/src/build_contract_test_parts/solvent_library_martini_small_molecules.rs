use super::*;

#[test]
fn atomistic_solvent_library_emits_multisite_waters() {
    let temp = tempfile::tempdir().unwrap();
    let cases = [
        (
            "TIP3",
            3_usize,
            vec![("OW", [0.0, 0.0]), ("HW1", [0.74, 0.64])],
        ),
        (
            "TIP4",
            4_usize,
            vec![("OW", [0.0, 0.0]), ("MW", [0.0, 0.32])],
        ),
        (
            "TIP5",
            5_usize,
            vec![("OW", [0.0, 0.0]), ("LP2", [-0.2, -0.2])],
        ),
        (
            "SOL-TIP4",
            4_usize,
            vec![("OW", [0.0, 0.0]), ("MW", [0.0, 0.32])],
        ),
    ];

    for (name, bead_count, expected_offsets) in cases {
        let gro = temp.path().join(format!("{name}.gro"));
        let request = json!({
            "schema_version": BUILD_SCHEMA_VERSION,
            "mode": "membrane",
            "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
            "membranes": [],
            "environment": {
                "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
                "solvent": {
                    "enabled": true,
                    "name": name,
                    "molarity_mol_l": 0.21,
                    "grid_spacing_angstrom": 20.0
                }
            },
            "outputs": {
                "coordinates": gro,
                "manifest": temp.path().join(format!("{name}_manifest.json"))
            }
        });

        let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
        assert_eq!(code, 0, "{value}");
        let residue_name = if name.starts_with("SOL") { "SOL" } else { name };
        assert_eq!(value["summary"]["solvent_counts"][residue_name], 1);
        assert_eq!(value["summary"]["bead_count"], bead_count);
        assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);

        let atoms = read_gro_residue_atoms(&gro, residue_name);
        assert_eq!(atoms.len(), bead_count);
        let origin = atoms
            .iter()
            .find(|(atom, _)| atom == expected_offsets[0].0)
            .unwrap()
            .1;
        for (atom_name, [dx, dy]) in &expected_offsets {
            let position = atoms.iter().find(|(atom, _)| atom == atom_name).unwrap().1;
            assert!((position[0] - origin[0] - dx).abs() < 0.02);
            assert!((position[1] - origin[1] - dy).abs() < 0.02);
        }
    }
}

#[test]
fn solvent_library_resolves_martini_amino_acid_species() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "ARG".to_string(),
        ..SolventPolicy::default()
    };
    let arg = resolved_solvent_species(&solvent);
    assert_eq!(arg[0].name, "ARG");
    assert_eq!(arg[0].mapping_ratio, 1.0);
    assert_eq!(arg[0].beads.len(), 3);
    assert_eq!(arg[0].beads[2].atom_name, "SC2");
    assert_eq!(arg[0].beads[2].offset_angstrom, [-2.5, 1.25, 0.0]);
    assert_eq!(arg[0].charge_e, 1.0);

    solvent.name = "ASP".to_string();
    let asp = resolved_solvent_species(&solvent);
    assert_eq!(asp[0].beads.len(), 2);
    assert_eq!(asp[0].beads[1].atom_name, "SC1");
    assert_eq!(asp[0].beads[1].charge_e, -1.0);
    assert_eq!(asp[0].charge_e, -1.0);

    solvent.name = "HSP".to_string();
    let hsp = resolved_solvent_species(&solvent);
    assert_eq!(hsp[0].beads.len(), 4);
    assert_eq!(hsp[0].beads[2].charge_e, 0.5);
    assert_eq!(hsp[0].beads[3].charge_e, 0.5);
    assert_eq!(hsp[0].charge_e, 1.0);

    solvent.name = "TRP".to_string();
    let trp = resolved_solvent_species(&solvent);
    assert_eq!(trp[0].beads.len(), 6);
    assert_eq!(trp[0].beads[5].atom_name, "SC5");
    assert_eq!(trp[0].charge_e, 0.0);

    let known = known_solvent_library_names();
    assert!(known.contains(&"ARG"));
    assert!(known.contains(&"ASP"));
    assert!(known.contains(&"TRP"));
}

#[test]
fn solvent_library_resolves_standard_martini3_solvents() {
    let cases = [
        ("DMSO", 2_usize, 0.0_f32, 1.0_f32, "S1"),
        ("DMS", 2, 0.0, 1.0, "S1"),
        ("ACN", 1, 0.0, 1.0, "ACN"),
        ("HEX", 2, 0.0, 1.0, "C1"),
        ("OCT", 2, 0.0, 1.0, "C1"),
        ("DOD", 3, 0.0, 1.0, "C1"),
        ("HD", 4, 0.0, 1.0, "C1"),
        ("HEXADECANE", 4, 0.0, 1.0, "C1"),
        ("HXE", 2, 0.0, 1.0, "D1"),
        ("HEXENE", 2, 0.0, 1.0, "D1"),
        ("OCE", 2, 0.0, 1.0, "D1"),
        ("DOE", 3, 0.0, 1.0, "D1"),
        ("HXY", 2, 0.0, 1.0, "T1"),
        ("OCY", 2, 0.0, 1.0, "T1"),
        ("HXD14", 2, 0.0, 1.0, "D1"),
        ("OCD912", 4, 0.0, 1.0, "C1"),
        ("DCE", 1, 0.0, 1.0, "CX"),
        ("DICHLOROETHANE", 1, 0.0, 1.0, "CX"),
        ("CLF", 1, 0.0, 1.0, "CX"),
        ("CHLOROFORM", 1, 0.0, 1.0, "CX"),
        ("TCM", 1, 0.0, 1.0, "CX"),
        ("TETRACHLOROMETHANE", 1, 0.0, 1.0, "CX"),
        ("TCE", 1, 0.0, 1.0, "CX"),
        ("TRICHLOROETHYLENE", 1, 0.0, 1.0, "CX"),
        ("TFEOL", 2, 0.0, 1.0, "CO"),
        ("TFE", 2, 0.0, 1.0, "CO"),
        ("TRIFLUOROETHANOL", 2, 0.0, 1.0, "CO"),
        ("MEO", 1, 0.0, 2.0, "MEO"),
        ("METHANOL", 1, 0.0, 2.0, "MEO"),
        ("ETO", 1, 0.0, 1.0, "ETO"),
        ("ETHANOL", 1, 0.0, 1.0, "ETO"),
        ("IPO", 1, 0.0, 1.0, "IPO"),
        ("ISOPROPANOL", 1, 0.0, 1.0, "IPO"),
        ("PRO", 1, 0.0, 1.0, "PRO"),
        ("PROPANOL", 1, 0.0, 1.0, "PRO"),
        ("BTO", 2, 0.0, 1.0, "C1"),
        ("BUTANOL", 2, 0.0, 1.0, "C1"),
        ("HXO", 2, 0.0, 1.0, "C1"),
        ("HEXANOL", 2, 0.0, 1.0, "C1"),
        ("HPO", 2, 0.0, 1.0, "C1"),
        ("HEPTANOL", 2, 0.0, 1.0, "C1"),
        ("OCO", 3, 0.0, 1.0, "C1"),
        ("OCTANOL", 3, 0.0, 1.0, "C1"),
        ("ETH", 1, 0.0, 1.0, "CO"),
        ("DIETHYL-ETHER", 1, 0.0, 1.0, "CO"),
        ("DISH", 3, 0.0, 1.0, "C1"),
        ("DIISOPROPYL-ETHER", 3, 0.0, 1.0, "C1"),
        ("DXE", 2, 0.0, 1.0, "R1"),
        ("GLYME", 2, 0.0, 1.0, "R1"),
        ("DIMETHOXYETHANE", 2, 0.0, 1.0, "R1"),
        ("TXE", 3, 0.0, 1.0, "R1"),
        ("TRIGLYME", 3, 0.0, 1.0, "R1"),
        ("DISS", 3, 0.0, 1.0, "C1"),
        ("DIISOPROPYL-SULFIDE", 3, 0.0, 1.0, "C1"),
        ("PPN", 1, 0.0, 1.0, "CO"),
        ("ACETONE", 1, 0.0, 1.0, "CO"),
        ("BTN", 1, 0.0, 1.0, "CO"),
        ("BUTANONE", 1, 0.0, 1.0, "CO"),
        ("ANN", 2, 0.0, 1.0, "R1"),
        ("ACETYLACETONE", 2, 0.0, 1.0, "R1"),
        ("HXN", 2, 0.0, 1.0, "C1"),
        ("HEXA-2-ONE", 2, 0.0, 1.0, "C1"),
        ("HPN", 2, 0.0, 1.0, "C1"),
        ("HEPTA-2-ONE", 2, 0.0, 1.0, "C1"),
        ("BTA", 2, 0.0, 1.0, "C1"),
        ("BUTANAL", 2, 0.0, 1.0, "C1"),
        ("HXA", 2, 0.0, 1.0, "C1"),
        ("HEXANAL", 2, 0.0, 1.0, "C1"),
        ("HPA", 2, 0.0, 1.0, "C1"),
        ("HEPTANAL", 2, 0.0, 1.0, "C1"),
        ("MEA", 1, 0.0, 1.0, "CO"),
        ("METHYL-ACETATE", 1, 0.0, 1.0, "CO"),
        ("ETA", 2, 0.0, 1.0, "R1"),
        ("ETHYL-ACETATE", 2, 0.0, 1.0, "R1"),
        ("IBA", 2, 0.0, 1.0, "R1"),
        ("ISO-BUTYL-ACETATE", 2, 0.0, 1.0, "R1"),
        ("TBA", 2, 0.0, 1.0, "R1"),
        ("T-BUTYL-ACETATE", 2, 0.0, 1.0, "R1"),
        ("NBA", 2, 0.0, 1.0, "R1"),
        ("N-BUTYL-ACETATE", 2, 0.0, 1.0, "R1"),
        ("DEI", 1, 0.0, 1.0, "N1"),
        ("DIETHYLAMINE", 1, 0.0, 1.0, "N1"),
        ("TMI", 1, 0.0, 1.0, "N1"),
        ("TRIMETHYLAMINE", 1, 0.0, 1.0, "N1"),
        ("PPI", 1, 0.0, 1.0, "N1"),
        ("PROPYLAMINE", 1, 0.0, 1.0, "N1"),
        ("NDI", 1, 0.0, 1.0, "N1"),
        ("DIMETHYLETHYLAMINE", 1, 0.0, 1.0, "N1"),
        ("BTI", 2, 0.0, 1.0, "C1"),
        ("BUTYLAMINE", 2, 0.0, 1.0, "C1"),
        ("PTI", 2, 0.0, 1.0, "C1"),
        ("PENTYLAMINE", 2, 0.0, 1.0, "C1"),
        ("HXI", 2, 0.0, 1.0, "C1"),
        ("HEXYLAMINE", 2, 0.0, 1.0, "C1"),
        ("HPI", 2, 0.0, 1.0, "C1"),
        ("HEPTYLAMINE", 2, 0.0, 1.0, "C1"),
        ("OCI", 3, 0.0, 1.0, "C1"),
        ("OCTYLAMINE", 3, 0.0, 1.0, "C1"),
        ("ACAC", 1, 0.0, 1.0, "CO"),
        ("ACETIC-ACID", 1, 0.0, 1.0, "CO"),
        ("PRAC", 1, 0.0, 1.0, "CO"),
        ("PROPANOIC-ACID", 1, 0.0, 1.0, "CO"),
        ("DMFD", 1, 0.0, 1.0, "CNO"),
        ("DIMETHYLFORMAMIDE", 1, 0.0, 1.0, "CNO"),
        ("DMAD", 1, 0.0, 1.0, "CNO"),
        ("DIMETHYLACETAMIDE", 1, 0.0, 1.0, "CNO"),
    ];
    for (name, bead_count, charge, mapping_ratio, first_atom) in cases {
        let solvent = SolventPolicy {
            enabled: true,
            name: name.to_string(),
            ..SolventPolicy::default()
        };
        let species = resolved_solvent_species(&solvent);
        assert_eq!(species[0].mapping_ratio, mapping_ratio, "{name}");
        assert_eq!(species[0].beads.len(), bead_count, "{name}");
        assert_eq!(species[0].beads[0].atom_name, first_atom, "{name}");
        assert!((species[0].charge_e - charge).abs() < 1.0e-6, "{name}");
    }

    let known = known_solvent_library_names();
    for name in [
        "DMSO",
        "DMS",
        "ACN",
        "HEX",
        "OCT",
        "DOD",
        "HD",
        "HEXADECANE",
        "HXE",
        "HEXENE",
        "OCE",
        "OCTENE",
        "DOE",
        "DODECENE",
        "HXY",
        "HEXYNE",
        "OCY",
        "OCTYNE",
        "HXD14",
        "HEXADIENE",
        "OCD912",
        "OCTADECADIENE",
        "DCE",
        "DICHLOROETHANE",
        "CLF",
        "CHLOROFORM",
        "TCM",
        "TETRACHLOROMETHANE",
        "TCE",
        "TRICHLOROETHYLENE",
        "TFEOL",
        "TFE",
        "TRIFLUOROETHANOL",
        "MEO",
        "METHANOL",
        "ETO",
        "ETHANOL",
        "IPO",
        "ISOPROPANOL",
        "PRO",
        "PROPANOL",
        "BTO",
        "BUTANOL",
        "HXO",
        "HEXANOL",
        "HPO",
        "HEPTANOL",
        "OCO",
        "OCTANOL",
        "ETH",
        "DIETHYL-ETHER",
        "DISH",
        "DIISOPROPYL-ETHER",
        "DXE",
        "GLYME",
        "DIMETHOXYETHANE",
        "TXE",
        "TRIGLYME",
        "DISS",
        "DIISOPROPYL-SULFIDE",
        "PPN",
        "ACETONE",
        "BTN",
        "BUTANONE",
        "ANN",
        "ACETYLACETONE",
        "HXN",
        "HEXA-2-ONE",
        "HPN",
        "HEPTA-2-ONE",
        "BTA",
        "BUTANAL",
        "HXA",
        "HEXANAL",
        "HPA",
        "HEPTANAL",
        "MEA",
        "METHYL-ACETATE",
        "ETA",
        "ETHYL-ACETATE",
        "IBA",
        "ISO-BUTYL-ACETATE",
        "TBA",
        "T-BUTYL-ACETATE",
        "NBA",
        "N-BUTYL-ACETATE",
        "DEI",
        "DIETHYLAMINE",
        "TMI",
        "TRIMETHYLAMINE",
        "PPI",
        "PROPYLAMINE",
        "NDI",
        "DIMETHYLETHYLAMINE",
        "BTI",
        "BUTYLAMINE",
        "PTI",
        "PENTYLAMINE",
        "HXI",
        "HEXYLAMINE",
        "HPI",
        "HEPTYLAMINE",
        "OCI",
        "OCTYLAMINE",
        "ACAC",
        "ACETIC-ACID",
        "PRAC",
        "PROPANOIC-ACID",
        "DMFD",
        "DIMETHYLFORMAMIDE",
        "DMAD",
        "DIMETHYLACETAMIDE",
    ] {
        assert!(known.contains(&name), "{name}");
    }
}

#[test]
fn standard_martini3_solvent_library_emits_topology_bonds() {
    let dmso = lookup_solvent_library("DMSO").unwrap();
    let dod = lookup_solvent_library("DOD").unwrap();
    let hxe = lookup_solvent_library("HXE").unwrap();
    let ocd = lookup_solvent_library("OCD912").unwrap();
    let tfeol = lookup_solvent_library("TFEOL").unwrap();
    let oco = lookup_solvent_library("OCTANOL").unwrap();
    let dish = lookup_solvent_library("DISH").unwrap();
    let dxe = lookup_solvent_library("GLYME").unwrap();
    let txe = lookup_solvent_library("TRIGLYME").unwrap();
    let diss = lookup_solvent_library("DISS").unwrap();
    let ann = lookup_solvent_library("ACETYLACETONE").unwrap();
    let hxn = lookup_solvent_library("HEXA-2-ONE").unwrap();
    let hpn = lookup_solvent_library("HEPTA-2-ONE").unwrap();
    let bta = lookup_solvent_library("BUTANAL").unwrap();
    let hxa = lookup_solvent_library("HEXANAL").unwrap();
    let hpa = lookup_solvent_library("HEPTANAL").unwrap();
    let eta = lookup_solvent_library("ETHYL-ACETATE").unwrap();
    let iba = lookup_solvent_library("ISO-BUTYL-ACETATE").unwrap();
    let tba = lookup_solvent_library("T-BUTYL-ACETATE").unwrap();
    let nba = lookup_solvent_library("N-BUTYL-ACETATE").unwrap();
    let bti = lookup_solvent_library("BUTYLAMINE").unwrap();
    let pti = lookup_solvent_library("PENTYLAMINE").unwrap();
    let hxi = lookup_solvent_library("HEXYLAMINE").unwrap();
    let hpi = lookup_solvent_library("HEPTYLAMINE").unwrap();
    let oci = lookup_solvent_library("OCTYLAMINE").unwrap();
    let dmso_topology = solvent_library_topology_block(&dmso);
    let dod_topology = solvent_library_topology_block(&dod);
    let hxe_topology = solvent_library_topology_block(&hxe);
    let ocd_topology = solvent_library_topology_block(&ocd);
    let tfeol_topology = solvent_library_topology_block(&tfeol);
    let oco_topology = solvent_library_topology_block(&oco);
    let dish_topology = solvent_library_topology_block(&dish);
    let dxe_topology = solvent_library_topology_block(&dxe);
    let txe_topology = solvent_library_topology_block(&txe);
    let diss_topology = solvent_library_topology_block(&diss);
    let ann_topology = solvent_library_topology_block(&ann);
    let hxn_topology = solvent_library_topology_block(&hxn);
    let hpn_topology = solvent_library_topology_block(&hpn);
    let bta_topology = solvent_library_topology_block(&bta);
    let hxa_topology = solvent_library_topology_block(&hxa);
    let hpa_topology = solvent_library_topology_block(&hpa);
    let eta_topology = solvent_library_topology_block(&eta);
    let iba_topology = solvent_library_topology_block(&iba);
    let tba_topology = solvent_library_topology_block(&tba);
    let nba_topology = solvent_library_topology_block(&nba);
    let bti_topology = solvent_library_topology_block(&bti);
    let pti_topology = solvent_library_topology_block(&pti);
    let hxi_topology = solvent_library_topology_block(&hxi);
    let hpi_topology = solvent_library_topology_block(&hpi);
    let oci_topology = solvent_library_topology_block(&oci);

    assert!(dmso_topology.contains("DMSO"));
    assert!(dmso_topology.contains("[ bonds ]"));
    assert!(dmso_topology.contains("    1     2     1    0.30000   8000.000"));
    assert!(dod_topology.contains("DOD"));
    assert!(dod_topology.contains("[ bonds ]"));
    assert!(dod_topology.contains("    1     2     1    0.47500   3800.000"));
    assert!(dod_topology.contains("    2     3     1    0.47500   3800.000"));
    assert!(hxe_topology.contains("HXE"));
    assert!(hxe_topology.contains("    1     2     1    0.39500   5000.000"));
    assert!(ocd_topology.contains("OCD912"));
    assert!(ocd_topology.contains("    1     2     1    0.49000   3800.000"));
    assert!(ocd_topology.contains("    3     4     1    0.49000   3800.000"));
    assert!(tfeol_topology.contains("TFEOL"));
    assert!(tfeol_topology.contains("CO"));
    assert!(tfeol_topology.contains("CX"));
    assert!(tfeol_topology.contains("    1     2     1    0.30000   5000.000"));
    assert!(oco_topology.contains("OCO"));
    assert!(oco_topology.contains("PC"));
    assert!(oco_topology.contains("    1     2     1    0.39000   5000.000"));
    assert!(oco_topology.contains("    2     3     1    0.35000   5000.000"));
    assert!(dish_topology.contains("DISH"));
    assert!(dish_topology.contains("    1     2     1    0.35500   5000.000"));
    assert!(dish_topology.contains("    2     3     1    0.35500   5000.000"));
    assert!(dxe_topology.contains("DXE"));
    assert!(dxe_topology.contains("    1     2     1    0.33000   7000.000"));
    assert!(txe_topology.contains("TXE"));
    assert!(txe_topology.contains("    1     2     1    0.33000   7000.000"));
    assert!(txe_topology.contains("    2     3     1    0.33000   7000.000"));
    assert!(diss_topology.contains("DISS"));
    assert!(diss_topology.contains("    1     2     1    0.36000   5000.000"));
    assert!(diss_topology.contains("    2     3     1    0.36000   5000.000"));
    assert!(ann_topology.contains("ANN"));
    assert!(ann_topology.contains("    1     2     1    0.35000   7000.000"));
    assert!(hxn_topology.contains("HXN"));
    assert!(hxn_topology.contains("    1     2     1    0.38000   7000.000"));
    assert!(hpn_topology.contains("HPN"));
    assert!(hpn_topology.contains("    1     2     1    0.45000   7000.000"));
    assert!(bta_topology.contains("BTA"));
    assert!(bta_topology.contains("    1     2     1    0.31000   7000.000"));
    assert!(hxa_topology.contains("HXA"));
    assert!(hxa_topology.contains("    1     2     1    0.38500   7000.000"));
    assert!(hpa_topology.contains("HPA"));
    assert!(hpa_topology.contains("    1     2     1    0.45500   7000.000"));
    assert!(eta_topology.contains("ETA"));
    assert!(eta_topology.contains("    1     2     1    0.31000   7000.000"));
    assert!(iba_topology.contains("IBA"));
    assert!(iba_topology.contains("    1     2     1    0.37500   3500.000"));
    assert!(tba_topology.contains("TBA"));
    assert!(tba_topology.contains("    1     2     1    0.37600   7000.000"));
    assert!(nba_topology.contains("NBA"));
    assert!(nba_topology.contains("    1     2     1    0.40500   7000.000"));
    assert!(bti_topology.contains("BTI"));
    assert!(bti_topology.contains("    1     2     1    0.31000   7000.000"));
    assert!(pti_topology.contains("PTI"));
    assert!(pti_topology.contains("    1     2     1    0.34000   7000.000"));
    assert!(hxi_topology.contains("HXI"));
    assert!(hxi_topology.contains("    1     2     1    0.38500   7000.000"));
    assert!(hpi_topology.contains("HPI"));
    assert!(hpi_topology.contains("    1     2     1    0.46000   7000.000"));
    assert!(oci_topology.contains("OCI"));
    assert!(oci_topology.contains("    1     2     1    0.39000   5000.000"));
    assert!(oci_topology.contains("    2     3     1    0.35000   5000.000"));
    assert!(oci_topology.contains("[ angles ]"));
    assert!(oci_topology.contains("    1     2     3     2    150.000    100.000"));
}

#[test]
fn solvent_library_resolves_martini_aromatic_small_molecules() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "BENZ".to_string(),
        ..SolventPolicy::default()
    };
    let benz = resolved_solvent_species(&solvent);
    assert_eq!(benz[0].name, "BENZ");
    assert_eq!(benz[0].mapping_ratio, 1.0);
    assert_eq!(benz[0].beads.len(), 3);
    assert_eq!(benz[0].beads[0].atom_name, "R1");
    assert_eq!(benz[0].charge_e, 0.0);

    solvent.name = "TOLU".to_string();
    let tolu = resolved_solvent_species(&solvent);
    assert_eq!(tolu[0].name, "TOLU");
    assert_eq!(tolu[0].beads.len(), 3);

    solvent.name = "ENAPH".to_string();
    let enaph = resolved_solvent_species(&solvent);
    assert_eq!(enaph[0].name, "ENAPH");
    assert_eq!(enaph[0].beads.len(), 6);
    assert_eq!(enaph[0].beads[0].atom_name, "C1");

    let known = known_solvent_library_names();
    assert!(known.contains(&"BENZ"));
    assert!(known.contains(&"TOLU"));
    assert!(known.contains(&"ENAPH"));
}

#[test]
fn solvent_library_resolves_martini_sugar_species() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "SUCR".to_string(),
        ..SolventPolicy::default()
    };
    let sucr = resolved_solvent_species(&solvent);
    assert_eq!(sucr[0].name, "SUCR");
    assert_eq!(sucr[0].mapping_ratio, 1.0);
    assert_eq!(sucr[0].beads.len(), 8);
    assert_eq!(sucr[0].beads[0].atom_name, "A");
    assert_eq!(sucr[0].beads[7].atom_name, "VS");
    assert_eq!(sucr[0].charge_e, 0.0);

    solvent.name = "SUCROSE".to_string();
    let sucrose = resolved_solvent_species(&solvent);
    assert_eq!(sucrose[0].name, "SUCR");
    assert_eq!(sucrose[0].beads.len(), sucr[0].beads.len());

    let known = known_solvent_library_names();
    assert!(known.contains(&"SUCR"));
    assert!(known.contains(&"SUCROSE"));
}

#[test]
fn solvent_library_resolves_imported_osmolyte_species() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "GLYL".to_string(),
        ..SolventPolicy::default()
    };
    let glyl = resolved_solvent_species(&solvent);
    assert_eq!(glyl[0].name, "GLYL");
    assert_eq!(glyl[0].mapping_ratio, 1.0);
    assert_eq!(glyl[0].beads.len(), 1);
    assert_eq!(glyl[0].beads[0].atom_name, "P01");
    assert_eq!(glyl[0].charge_e, 0.0);

    solvent.name = "PUT".to_string();
    let put = resolved_solvent_species(&solvent);
    assert_eq!(put[0].name, "PUT");
    assert_eq!(put[0].beads.len(), 2);
    assert_eq!(put[0].beads[1].atom_name, "P02");
    assert!((put[0].beads[1].offset_angstrom[2] - 1.305).abs() < 1.0e-6);

    solvent.name = "SPER".to_string();
    let sper = resolved_solvent_species(&solvent);
    assert_eq!(sper[0].name, "SPER");
    assert_eq!(sper[0].beads.len(), 3);
    assert_eq!(sper[0].beads[1].atom_name, "C01");

    solvent.name = "UREA".to_string();
    let urea = resolved_solvent_species(&solvent);
    assert_eq!(urea[0].name, "UREA");
    assert_eq!(urea[0].beads.len(), 1);

    solvent.name = "TREH".to_string();
    let treh = resolved_solvent_species(&solvent);
    assert_eq!(treh[0].name, "TREH");
    assert_eq!(treh[0].beads.len(), 9);
    assert_eq!(treh[0].beads[0].atom_name, "S01");
    assert_eq!(treh[0].beads[8].atom_name, "S06");
    assert_eq!(treh[0].charge_e, 0.0);

    let known = known_solvent_library_names();
    assert!(known.contains(&"GLYL"));
    assert!(known.contains(&"PUT"));
    assert!(known.contains(&"SPER"));
    assert!(known.contains(&"UREA"));
    assert!(known.contains(&"TREH"));
}
