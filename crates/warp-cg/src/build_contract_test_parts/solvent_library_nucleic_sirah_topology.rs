use super::*;

#[test]
fn solvent_library_resolves_ionic_liquid_tutorial_species() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "C12".to_string(),
        ..SolventPolicy::default()
    };
    let c12 = resolved_solvent_species(&solvent);
    assert_eq!(c12[0].name, "C12");
    assert_eq!(c12[0].mapping_ratio, 1.0);
    assert_eq!(c12[0].beads.len(), 6);
    assert_eq!(c12[0].beads[0].atom_name, "SI1");
    assert_eq!(c12[0].charge_e, 1.0);

    solvent.name = "DIM".to_string();
    let dim_alias = resolved_solvent_species(&solvent);
    assert_eq!(dim_alias[0].name, "C12");
    assert_eq!(dim_alias[0].charge_e, 1.0);

    solvent.name = "BF4".to_string();
    let bf4 = resolved_solvent_species(&solvent);
    assert_eq!(bf4[0].name, "BF4");
    assert_eq!(bf4[0].beads.len(), 1);
    assert_eq!(bf4[0].beads[0].atom_name, "BF4");
    assert_eq!(bf4[0].charge_e, -1.0);

    let known = known_solvent_library_names();
    assert!(known.contains(&"C1"));
    assert!(known.contains(&"C12"));
    assert!(known.contains(&"BF4"));
}

#[test]
fn solvent_library_resolves_martini_dna_nucleotide_species() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "DA".to_string(),
        ..SolventPolicy::default()
    };
    let da = resolved_solvent_species(&solvent);
    assert_eq!(da[0].name, "DA");
    assert_eq!(da[0].mapping_ratio, 1.0);
    assert_eq!(da[0].beads.len(), 7);
    assert_eq!(da[0].beads[0].atom_name, "BB1");
    assert_eq!(da[0].charge_e, -1.0);

    solvent.name = "DC".to_string();
    let dc = resolved_solvent_species(&solvent);
    assert_eq!(dc[0].name, "DC");
    assert_eq!(dc[0].beads.len(), 6);
    assert_eq!(dc[0].charge_e, -1.0);

    solvent.name = "DG".to_string();
    let dg = resolved_solvent_species(&solvent);
    assert_eq!(dg[0].name, "DG");
    assert_eq!(dg[0].beads.len(), 7);
    assert_eq!(dg[0].charge_e, -1.0);

    solvent.name = "DT".to_string();
    let dt = resolved_solvent_species(&solvent);
    assert_eq!(dt[0].name, "DT");
    assert_eq!(dt[0].beads.len(), 6);
    assert_eq!(dt[0].charge_e, -1.0);

    let known = known_solvent_library_names();
    assert!(known.contains(&"DA"));
    assert!(known.contains(&"DC"));
    assert!(known.contains(&"DG"));
    assert!(known.contains(&"DT"));
}

#[test]
fn solvent_library_resolves_martini_nucleobase_species() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "ADEN".to_string(),
        ..SolventPolicy::default()
    };
    let aden = resolved_solvent_species(&solvent);
    assert_eq!(aden[0].name, "ADEN");
    assert_eq!(aden[0].mapping_ratio, 1.0);
    assert_eq!(aden[0].beads.len(), 6);
    assert_eq!(aden[0].beads[0].atom_name, "N1");
    assert_eq!(aden[0].charge_e, 0.0);

    solvent.name = "CYTO".to_string();
    let cyto = resolved_solvent_species(&solvent);
    assert_eq!(cyto[0].beads.len(), 4);
    assert_eq!(cyto[0].charge_e, 0.0);

    solvent.name = "GUAN".to_string();
    let guan = resolved_solvent_species(&solvent);
    assert_eq!(guan[0].beads.len(), 6);
    assert_eq!(guan[0].charge_e, 0.0);

    solvent.name = "THYM".to_string();
    let thym = resolved_solvent_species(&solvent);
    assert_eq!(thym[0].beads.len(), 5);
    assert_eq!(thym[0].charge_e, 0.0);

    solvent.name = "URAC".to_string();
    let urac = resolved_solvent_species(&solvent);
    assert_eq!(urac[0].beads.len(), 5);
    assert_eq!(urac[0].charge_e, 0.0);

    let known = known_solvent_library_names();
    for name in ["ADEN", "CYTO", "GUAN", "THYM", "URAC"] {
        assert!(known.contains(&name));
    }
}

#[test]
fn solvent_library_resolves_sirah_water_species() {
    let solvent = SolventPolicy {
        enabled: true,
        name: "WT4".to_string(),
        ..SolventPolicy::default()
    };
    let wt4 = resolved_solvent_species(&solvent);
    assert_eq!(wt4[0].name, "WT4");
    assert_eq!(wt4[0].mapping_ratio, 1.0);
    assert_eq!(wt4[0].beads.len(), 4);
    assert_eq!(wt4[0].beads[0].atom_name, "WN1");
    assert_eq!(wt4[0].beads[0].charge_e, -0.41);
    assert_eq!(wt4[0].beads[3].atom_name, "WP2");
    assert_eq!(wt4[0].beads[3].charge_e, 0.41);
    assert_eq!(wt4[0].charge_e, 0.0);

    let known = known_solvent_library_names();
    assert!(known.contains(&"WT4"));
}

#[test]
fn generated_solvent_and_ions_emit_library_topology_blocks() {
    let temp = tempfile::tempdir().unwrap();
    let top = temp.path().join("generated_species.top");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [30.0, 30.0, 30.0]},
        "membranes": [],
        "solutes": [{"name": "DA", "count": 1}],
        "environment": {
            "ions": {
                "neutralize": true,
                "salt_molarity_mol_l": 0.0,
                "cation": "NaW",
                "anion": "ClW"
            },
            "solvent": {
                "enabled": true,
                "name": "WT4",
                "molarity_mol_l": 0.07,
                "grid_spacing_angstrom": 30.0
            }
        },
        "outputs": {
            "topology": top,
            "manifest": temp.path().join("manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["WT4"], 1);
    assert_eq!(value["summary"]["solvent_counts"]["NaW"], 1);

    let topology = std::fs::read_to_string(&top).unwrap();
    assert!(topology.contains("WT4"));
    assert!(topology.contains("WN1"));
    assert!(topology.contains("WP2"));
    assert!(topology.contains("[ bonds ]"));
    assert!(topology.contains("1     2     1    0.45000   4184.000"));
    assert!(topology.contains("NaW"));
    assert!(topology.contains("1 NaW"));
    assert!(topology.contains("WT4              1"));
    assert!(topology.contains("NaW              1"));
}

#[test]
fn generated_lipids_emit_library_topology_blocks() {
    let temp = tempfile::tempdir().unwrap();
    let top = temp.path().join("lipids.top");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [60.0, 60.0, 60.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "POPG", "count": 1}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "topology": top,
            "manifest": temp.path().join("manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPG"], 1);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], -1.0);

    let topology = std::fs::read_to_string(&top).unwrap();
    assert!(topology.contains("[ moleculetype ]"));
    assert!(topology.contains("POPG"));
    assert!(topology.contains("PO4"));
    assert!(topology.contains("GL0"));
    assert!(topology.contains("   -1.000"));
    assert!(topology.contains("POPG             1"));
}

#[test]
fn aromatic_small_molecule_solvent_library_emits_multibead_species() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("benz_solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "name": "BENZ",
                "molarity_mol_l": 0.21,
                "grid_spacing_angstrom": 20.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("benz_manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["BENZ"], 1);
    assert_eq!(value["summary"]["bead_count"], 3);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);

    let atoms = read_gro_residue_atoms(&gro, "BENZ");
    assert_eq!(atoms.len(), 3);
    assert!(atoms.iter().any(|(atom, _)| atom == "R1"));
    assert!(atoms.iter().any(|(atom, _)| atom == "R2"));
    assert!(atoms.iter().any(|(atom, _)| atom == "R3"));
}

#[test]
fn ionic_liquid_solvent_library_emits_charged_multibead_species() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("c4_solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "name": "C4",
                "molarity_mol_l": 0.21,
                "grid_spacing_angstrom": 20.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("c4_manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["C4"], 1);
    assert_eq!(value["summary"]["bead_count"], 4);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);
    assert_eq!(value["charge"]["solvent_charge_e"], 1.0);
    assert_eq!(value["charge"]["baseline_ion_charge_e"], 0.0);
    assert_eq!(value["charge"]["neutralization_input_charge_e"], 1.0);

    let atoms = read_gro_residue_atoms(&gro, "C4");
    assert_eq!(atoms.len(), 4);
    assert!(atoms.iter().any(|(atom, _)| atom == "SI4"));
}

#[test]
fn amino_acid_solvent_library_emits_multibead_charged_species() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("arg_solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "name": "ARG",
                "molarity_mol_l": 0.21,
                "grid_spacing_angstrom": 20.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("arg_solvent_manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["ARG"], 1);
    assert_eq!(value["summary"]["bead_count"], 3);

    let atoms = read_gro_residue_atoms(&gro, "ARG");
    assert_eq!(atoms.len(), 3);
    let sc1 = atoms.iter().find(|(atom, _)| atom == "SC1").unwrap().1;
    let sc2 = atoms.iter().find(|(atom, _)| atom == "SC2").unwrap().1;
    assert!((sc2[0] - sc1[0] + 2.5).abs() < 0.02);
    assert!((sc2[1] - sc1[1] - 1.25).abs() < 0.02);
}

#[test]
fn tailcode_solvent_library_builds_hydrocarbon_and_fatty_acid_fragments() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "hydrocarbon:CDtF".to_string(),
        ..SolventPolicy::default()
    };
    let hydrocarbon = resolved_solvent_species(&solvent);
    assert_eq!(hydrocarbon[0].name, "HCCDtF");
    assert_eq!(hydrocarbon[0].mapping_ratio, 1.0);
    assert_eq!(hydrocarbon[0].beads.len(), 4);
    assert_eq!(hydrocarbon[0].beads[0].atom_name, "C1A");
    assert_eq!(hydrocarbon[0].beads[1].atom_name, "D2A");
    assert_eq!(hydrocarbon[0].beads[2].atom_name, "t3A");
    assert_eq!(hydrocarbon[0].beads[3].atom_name, "F4A");
    assert_eq!(hydrocarbon[0].beads[3].offset_angstrom, [0.0, 0.0, -9.0]);
    assert_eq!(hydrocarbon[0].charge_e, 0.0);

    solvent.name = "fattyacid:CD".to_string();
    let fatty_acid = resolved_solvent_species(&solvent);
    assert_eq!(fatty_acid[0].name, "FACD");
    assert_eq!(fatty_acid[0].beads.len(), 3);
    assert_eq!(fatty_acid[0].beads[0].atom_name, "COO");
    assert_eq!(fatty_acid[0].beads[0].charge_e, -1.0);
    assert_eq!(fatty_acid[0].beads[1].atom_name, "C1A");
    assert_eq!(fatty_acid[0].beads[1].offset_angstrom, [0.0, 0.0, -3.0]);
    assert_eq!(fatty_acid[0].beads[2].atom_name, "D2A");
    assert_eq!(fatty_acid[0].charge_e, -1.0);
}

#[test]
fn tailcode_solvent_library_builds_glyceride_fragments() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "monoglyceride:CD".to_string(),
        ..SolventPolicy::default()
    };
    let mono = resolved_solvent_species(&solvent);
    assert_eq!(mono[0].name, "MGCD");
    assert_eq!(mono[0].beads.len(), 4);
    assert_eq!(mono[0].beads[0].atom_name, "DOH");
    assert_eq!(mono[0].beads[0].offset_angstrom, [0.0, 0.0, 3.0]);
    assert_eq!(mono[0].beads[1].atom_name, "GL1");
    assert_eq!(mono[0].beads[2].atom_name, "C1A");
    assert_eq!(mono[0].beads[2].offset_angstrom, [0.0, 0.0, -3.0]);

    solvent.name = "diglyceride:CD,TF".to_string();
    let di = resolved_solvent_species(&solvent);
    assert_eq!(di[0].name, "DGCDTF");
    assert_eq!(di[0].beads.len(), 7);
    assert_eq!(di[0].beads[0].atom_name, "COH");
    assert_eq!(di[0].beads[2].atom_name, "GL2");
    assert_eq!(di[0].beads[2].offset_angstrom, [2.5, 0.0, 0.0]);
    assert_eq!(di[0].beads[3].atom_name, "C1A");
    assert_eq!(di[0].beads[3].offset_angstrom, [0.0, 0.0, -3.0]);
    assert_eq!(di[0].beads[5].atom_name, "T1B");
    assert_eq!(di[0].beads[5].offset_angstrom, [2.5, 0.0, -3.0]);

    solvent.name = "triglyceride:C,D,F".to_string();
    let tri = resolved_solvent_species(&solvent);
    assert_eq!(tri[0].name, "TGCDF");
    assert_eq!(tri[0].beads.len(), 6);
    assert_eq!(tri[0].beads[0].atom_name, "GL1");
    assert_eq!(tri[0].beads[1].atom_name, "GL2");
    assert_eq!(tri[0].beads[2].atom_name, "GL3");
    assert_eq!(tri[0].beads[2].offset_angstrom, [5.0, 0.0, 0.0]);
    assert_eq!(tri[0].beads[5].atom_name, "F1C");
    assert_eq!(tri[0].beads[5].offset_angstrom, [5.0, 0.0, -3.0]);
}

#[test]
fn tailcode_solvent_library_builds_bmp_fragments() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "bmp2:C,D".to_string(),
        ..SolventPolicy::default()
    };
    let bmp2 = resolved_solvent_species(&solvent);
    assert_eq!(bmp2[0].name, "B2CD");
    assert_eq!(bmp2[0].beads.len(), 7);
    assert_eq!(bmp2[0].charge_e, -5.0);
    assert_eq!(bmp2[0].beads[0].atom_name, "PO4");
    assert_eq!(bmp2[0].beads[1].atom_name, "OH1");
    assert_eq!(bmp2[0].beads[1].offset_angstrom, [2.0, 0.0, -2.5]);
    assert_eq!(bmp2[0].beads[2].atom_name, "GL1");
    assert_eq!(bmp2[0].beads[2].offset_angstrom, [2.0, 0.0, -5.5]);
    assert_eq!(bmp2[0].beads[3].atom_name, "C1A");
    assert_eq!(bmp2[0].beads[3].offset_angstrom, [2.0, 0.0, -8.5]);
    assert_eq!(bmp2[0].beads[6].atom_name, "D1B");
    assert_eq!(bmp2[0].beads[6].offset_angstrom, [-2.0, 0.0, -8.5]);

    solvent.name = "bmp3:C,D".to_string();
    let bmp3 = resolved_solvent_species(&solvent);
    assert_eq!(bmp3[0].name, "B3CD");
    assert_eq!(bmp3[0].beads.len(), 7);
    assert_eq!(bmp3[0].charge_e, -5.0);
    assert_eq!(bmp3[0].beads[1].atom_name, "GL1");
    assert_eq!(bmp3[0].beads[1].offset_angstrom, [2.0, 0.0, -2.5]);
    assert_eq!(bmp3[0].beads[2].atom_name, "OH1");
    assert_eq!(bmp3[0].beads[2].offset_angstrom, [4.5, 0.0, -2.5]);
    assert_eq!(bmp3[0].beads[3].atom_name, "C1A");
    assert_eq!(bmp3[0].beads[3].offset_angstrom, [2.0, 0.0, -5.5]);
    assert_eq!(bmp3[0].beads[6].atom_name, "D1B");
    assert_eq!(bmp3[0].beads[6].offset_angstrom, [-2.0, 0.0, -5.5]);
}

#[test]
fn tailcode_solvent_library_builds_cardiolipin_fragments() {
    let solvent = SolventPolicy {
        enabled: true,
        name: "cardiolipin:C,D,T,F".to_string(),
        ..SolventPolicy::default()
    };
    let cardiolipin = resolved_solvent_species(&solvent);
    assert_eq!(cardiolipin[0].name, "LCDTF");
    assert_eq!(cardiolipin[0].beads.len(), 11);
    assert_eq!(cardiolipin[0].charge_e, 0.0);
    assert_eq!(cardiolipin[0].beads[0].atom_name, "GLC");
    assert_eq!(cardiolipin[0].beads[1].atom_name, "PO41");
    assert_eq!(cardiolipin[0].beads[1].offset_angstrom, [-1.5, 0.0, -3.0]);
    assert_eq!(cardiolipin[0].beads[2].atom_name, "GL11");
    assert_eq!(cardiolipin[0].beads[2].offset_angstrom, [-1.5, 0.0, -6.0]);
    assert_eq!(cardiolipin[0].beads[3].atom_name, "GL21");
    assert_eq!(cardiolipin[0].beads[3].offset_angstrom, [-4.0, 0.0, -6.0]);
    assert_eq!(cardiolipin[0].beads[4].atom_name, "C1A1");
    assert_eq!(cardiolipin[0].beads[4].offset_angstrom, [-1.5, 0.0, -9.0]);
    assert_eq!(cardiolipin[0].beads[5].atom_name, "D1B1");
    assert_eq!(cardiolipin[0].beads[5].offset_angstrom, [-4.0, 0.0, -9.0]);
    assert_eq!(cardiolipin[0].beads[6].atom_name, "PO42");
    assert_eq!(cardiolipin[0].beads[6].offset_angstrom, [1.5, 0.0, -3.0]);
    assert_eq!(cardiolipin[0].beads[9].atom_name, "T1A2");
    assert_eq!(cardiolipin[0].beads[9].offset_angstrom, [1.5, 0.0, -9.0]);
    assert_eq!(cardiolipin[0].beads[10].atom_name, "F1B2");
    assert_eq!(cardiolipin[0].beads[10].offset_angstrom, [4.0, 0.0, -9.0]);
}

#[test]
fn tailcode_solvent_library_builds_sphingolipid_fragments() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "sphingolipid:PC,C,D".to_string(),
        ..SolventPolicy::default()
    };
    let smpc = resolved_solvent_species(&solvent);
    assert_eq!(smpc[0].name, "SPCCD");
    assert_eq!(smpc[0].beads.len(), 6);
    assert_eq!(smpc[0].charge_e, 0.0);
    assert_eq!(smpc[0].beads[0].atom_name, "NC3");
    assert_eq!(smpc[0].beads[0].offset_angstrom, [0.0, 0.0, 6.0]);
    assert_eq!(smpc[0].beads[0].charge_e, 1.0);
    assert_eq!(smpc[0].beads[1].atom_name, "PO4");
    assert_eq!(smpc[0].beads[1].offset_angstrom, [0.0, 0.0, 3.0]);
    assert_eq!(smpc[0].beads[1].charge_e, -1.0);
    assert_eq!(smpc[0].beads[2].atom_name, "OH1");
    assert_eq!(smpc[0].beads[3].atom_name, "AM2");
    assert_eq!(smpc[0].beads[3].offset_angstrom, [2.5, 0.0, 0.0]);
    assert_eq!(smpc[0].beads[4].atom_name, "C1A");
    assert_eq!(smpc[0].beads[4].offset_angstrom, [0.0, 0.0, -3.0]);
    assert_eq!(smpc[0].beads[5].atom_name, "D1B");
    assert_eq!(smpc[0].beads[5].offset_angstrom, [2.5, 0.0, -3.0]);

    solvent.name = "sphingolipid:P7,C,D".to_string();
    let smp7 = resolved_solvent_species(&solvent);
    assert_eq!(smp7[0].name, "SP7CD");
    assert_eq!(smp7[0].charge_e, -4.0);
    assert_eq!(smp7[0].beads[4].atom_name, "PO4");
    assert_eq!(smp7[0].beads[4].charge_e, -1.0);
    assert_eq!(smp7[0].beads[5].atom_name, "P3");
    assert_eq!(smp7[0].beads[5].charge_e, -1.5);
    assert_eq!(smp7[0].beads[6].atom_name, "P5");
    assert_eq!(smp7[0].beads[6].charge_e, -1.5);
}

#[test]
fn tailcode_solvent_library_emits_fragment_built_molecules() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("tailcode_solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "name": "fattyacid:CD",
                "molarity_mol_l": 0.21,
                "grid_spacing_angstrom": 20.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("tailcode_solvent_manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["FACD"], 1);
    assert_eq!(value["summary"]["bead_count"], 3);

    let atoms = read_gro_residue_atoms(&gro, "FACD");
    assert_eq!(atoms.len(), 3);
    let coo = atoms.iter().find(|(atom, _)| atom == "COO").unwrap().1;
    let c1a = atoms.iter().find(|(atom, _)| atom == "C1A").unwrap().1;
    let d2a = atoms.iter().find(|(atom, _)| atom == "D2A").unwrap().1;
    assert!((c1a[2] - coo[2] + 3.0).abs() < 0.02);
    assert!((d2a[2] - c1a[2] + 3.0).abs() < 0.02);
}

#[test]
fn glyceride_tailcode_solvent_library_emits_multi_tail_fragments() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("glyceride_tailcode_solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "name": "diglyceride:C,D",
                "molarity_mol_l": 0.21,
                "grid_spacing_angstrom": 20.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("glyceride_tailcode_solvent_manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["DGCD"], 1);
    assert_eq!(value["summary"]["bead_count"], 5);

    let atoms = read_gro_residue_atoms(&gro, "DGCD");
    assert_eq!(atoms.len(), 5);
    let gl1 = atoms.iter().find(|(atom, _)| atom == "GL1").unwrap().1;
    let gl2 = atoms.iter().find(|(atom, _)| atom == "GL2").unwrap().1;
    let c1a = atoms.iter().find(|(atom, _)| atom == "C1A").unwrap().1;
    let d1b = atoms.iter().find(|(atom, _)| atom == "D1B").unwrap().1;
    assert!((gl2[0] - gl1[0] - 2.5).abs() < 0.02);
    assert!((c1a[0] - gl1[0]).abs() < 0.02);
    assert!((c1a[2] - gl1[2] + 3.0).abs() < 0.02);
    assert!((d1b[0] - gl2[0]).abs() < 0.02);
    assert!((d1b[2] - gl2[2] + 3.0).abs() < 0.02);
}
