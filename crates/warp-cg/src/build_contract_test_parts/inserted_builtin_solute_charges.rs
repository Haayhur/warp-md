use super::*;

#[test]
fn inserted_component_charge_topology_rejects_mismatched_explicit_total() {
    let temp = tempfile::tempdir().unwrap();
    let topology = temp.path().join("solute.itp");
    std::fs::write(
        &topology,
        r#"
[ moleculetype ]
  SOL 1

[ atoms ]
  1 P1 1 SOL A 1 -1.0
"#,
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "POPC"}]
            }]
        }],
        "solutes": [{
            "name": "SOL",
            "net_charge_e": 0.0,
            "charge_topology": topology
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does not match charge_topology-derived charge"));
}

#[test]
fn coordinate_less_known_solutes_use_builtin_template_charge_and_beads() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "POPC", "count": 1}]
            }]
        }],
        "solutes": [
            {"name": "ARG", "count": 2},
            {"name": "GLY", "count": 3},
            {"name": "TYR", "count": 1}
        ],
        "environment": {
            "ions": {"neutralize": true, "salt_molarity_mol_l": 0.0},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["ARG"], 2);
    assert_eq!(value["summary"]["inserted_counts"]["GLY"], 3);
    assert_eq!(value["summary"]["inserted_counts"]["TYR"], 1);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 2.0);
    assert_eq!(value["charge"]["neutralization"]["counterion"], "Cl-");
    assert_eq!(value["charge"]["neutralization"]["counterion_count"], 2);
}

#[test]
fn coordinate_less_aromatic_solutes_use_builtin_reference_templates() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [],
        "solutes": [
            {"name": "BENZ", "count": 1},
            {"name": "TOLU", "count": 1},
            {"name": "ENAPH", "count": 1}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["BENZ"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["TOLU"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["ENAPH"], 1);
    assert_eq!(value["summary"]["bead_count"], 12);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);
}

#[test]
fn coordinate_less_sugar_solutes_use_builtin_reference_template() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("sugars.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [80.0, 80.0, 80.0],
            "placement": {"mode": "seeded", "random_seed": 1234}
        },
        "membranes": [],
        "solutes": [
            {"name": "SUCR", "count": 2},
            {"name": "SUCROSE", "count": 1}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("manifest.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["SUCR"], 2);
    assert_eq!(value["summary"]["inserted_counts"]["SUCROSE"], 1);
    assert_eq!(value["summary"]["bead_count"], 24);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);
    assert!(
        value["placement"]["inserted_flood"]["candidate_count"]
            .as_u64()
            .unwrap()
            >= 3
    );

    let atoms = read_gro_residue_atoms(&gro, "SUCR");
    let alias_atoms = read_gro_residue_atoms(&gro, "SUCRO");
    assert_eq!(atoms.len(), 16);
    assert_eq!(alias_atoms.len(), 8);
    assert_eq!(atoms.iter().filter(|(atom, _)| atom == "A").count(), 4);
    assert_eq!(
        alias_atoms.iter().filter(|(atom, _)| atom == "VS").count(),
        2
    );
}

#[test]
fn coordinate_less_osmolyte_solutes_use_builtin_reference_templates() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("osmolytes.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [80.0, 80.0, 80.0],
            "placement": {"mode": "seeded", "random_seed": 1235}
        },
        "membranes": [],
        "solutes": [
            {"name": "GLYL", "count": 1},
            {"name": "PUT", "count": 2},
            {"name": "SPER", "count": 1},
            {"name": "UREA", "count": 1},
            {"name": "TREH", "count": 1}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("manifest.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["GLYL"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["PUT"], 2);
    assert_eq!(value["summary"]["inserted_counts"]["SPER"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["UREA"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["TREH"], 1);
    assert_eq!(value["summary"]["bead_count"], 18);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);

    let put_atoms = read_gro_residue_atoms(&gro, "PUT");
    assert_eq!(put_atoms.len(), 4);
    assert_eq!(
        put_atoms.iter().filter(|(atom, _)| atom == "P01").count(),
        2
    );
    assert_eq!(
        put_atoms.iter().filter(|(atom, _)| atom == "P02").count(),
        2
    );
    let sper_atoms = read_gro_residue_atoms(&gro, "SPER");
    assert_eq!(sper_atoms.len(), 3);
    assert_eq!(
        sper_atoms.iter().filter(|(atom, _)| atom == "C01").count(),
        1
    );
    let treh_atoms = read_gro_residue_atoms(&gro, "TREH");
    assert_eq!(treh_atoms.len(), 9);
    assert_eq!(
        treh_atoms.iter().filter(|(atom, _)| atom == "S01").count(),
        1
    );
    assert_eq!(
        treh_atoms.iter().filter(|(atom, _)| atom == "S06").count(),
        1
    );
}

#[test]
fn coordinate_less_ionic_liquid_solutes_use_builtin_reference_templates() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [],
        "solutes": [
            {"name": "C1", "count": 1},
            {"name": "C2", "count": 1},
            {"name": "C4", "count": 1},
            {"name": "C8", "count": 1},
            {"name": "C12", "count": 1},
            {"name": "BF4", "count": 5}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["C1"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["C12"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["BF4"], 5);
    assert_eq!(value["summary"]["bead_count"], 26);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);
}

#[test]
fn coordinate_less_dna_nucleotide_solutes_use_builtin_reference_templates() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("dna_nucleotides.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [90.0, 90.0, 90.0]},
        "membranes": [],
        "solutes": [
            {"name": "DA", "count": 1},
            {"name": "DC", "count": 1},
            {"name": "DG", "count": 1},
            {"name": "DT", "count": 1}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["DA"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["DC"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["DG"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["DT"], 1);
    assert_eq!(value["summary"]["bead_count"], 26);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], -4.0);

    let da_atoms = read_gro_residue_atoms(&gro, "DA");
    let dc_atoms = read_gro_residue_atoms(&gro, "DC");
    let dg_atoms = read_gro_residue_atoms(&gro, "DG");
    let dt_atoms = read_gro_residue_atoms(&gro, "DT");
    assert_eq!(da_atoms.len(), 7);
    assert_eq!(dc_atoms.len(), 6);
    assert_eq!(dg_atoms.len(), 7);
    assert_eq!(dt_atoms.len(), 6);
    assert_eq!(da_atoms.iter().filter(|(atom, _)| atom == "BB1").count(), 1);
    assert_eq!(dt_atoms.iter().filter(|(atom, _)| atom == "SC3").count(), 1);
}

#[test]
fn coordinate_less_nucleobase_solutes_use_builtin_reference_templates() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("nucleobases.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [90.0, 90.0, 90.0]},
        "membranes": [],
        "solutes": [
            {"name": "ADEN", "count": 1},
            {"name": "CYTO", "count": 1},
            {"name": "GUAN", "count": 1},
            {"name": "THYM", "count": 1},
            {"name": "URAC", "count": 1}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["ADEN"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["CYTO"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["GUAN"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["THYM"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["URAC"], 1);
    assert_eq!(value["summary"]["bead_count"], 26);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);

    let aden_atoms = read_gro_residue_atoms(&gro, "ADEN");
    let cyto_atoms = read_gro_residue_atoms(&gro, "CYTO");
    let guan_atoms = read_gro_residue_atoms(&gro, "GUAN");
    let thym_atoms = read_gro_residue_atoms(&gro, "THYM");
    let urac_atoms = read_gro_residue_atoms(&gro, "URAC");
    assert_eq!(aden_atoms.len(), 6);
    assert_eq!(cyto_atoms.len(), 4);
    assert_eq!(guan_atoms.len(), 6);
    assert_eq!(thym_atoms.len(), 5);
    assert_eq!(urac_atoms.len(), 5);
    assert_eq!(
        aden_atoms.iter().filter(|(atom, _)| atom == "N6").count(),
        1
    );
    assert_eq!(
        cyto_atoms.iter().filter(|(atom, _)| atom == "N4").count(),
        1
    );
    assert_eq!(
        guan_atoms.iter().filter(|(atom, _)| atom == "N6").count(),
        1
    );
    assert_eq!(
        thym_atoms.iter().filter(|(atom, _)| atom == "N5").count(),
        1
    );
    assert_eq!(
        urac_atoms.iter().filter(|(atom, _)| atom == "N5").count(),
        1
    );
}

#[test]
fn coordinate_less_sirah_solutes_use_builtin_reference_templates() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("sirah_solutes.gro");
    let top = temp.path().join("sirah_solutes.top");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [60.0, 60.0, 60.0]},
        "membranes": [],
        "solutes": [
            {"name": "WT4", "count": 1},
            {"name": "NaW", "count": 1},
            {"name": "ClW", "count": 1}
        ],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro,
            "topology": top,
            "manifest": temp.path().join("manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["WT4"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["NaW"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["ClW"], 1);
    assert_eq!(value["summary"]["bead_count"], 6);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 0.0);

    let wt4_atoms = read_gro_residue_atoms(&gro, "WT4");
    let naw_atoms = read_gro_residue_atoms(&gro, "NaW");
    let clw_atoms = read_gro_residue_atoms(&gro, "ClW");
    assert_eq!(wt4_atoms.len(), 4);
    assert_eq!(naw_atoms.len(), 1);
    assert_eq!(clw_atoms.len(), 1);
    assert_eq!(
        wt4_atoms.iter().filter(|(atom, _)| atom == "WN1").count(),
        1
    );
    assert_eq!(
        wt4_atoms.iter().filter(|(atom, _)| atom == "WP2").count(),
        1
    );
    assert_eq!(
        naw_atoms.iter().filter(|(atom, _)| atom == "NaW").count(),
        1
    );
    assert_eq!(
        clw_atoms.iter().filter(|(atom, _)| atom == "ClW").count(),
        1
    );

    let topology = std::fs::read_to_string(&top).unwrap();
    assert!(topology.contains("[ moleculetype ]"));
    assert!(topology.contains("WT4"));
    assert!(topology.contains("WN1"));
    assert!(topology.contains("WP2"));
    assert!(topology.contains("NaW"));
    assert!(topology.contains("ClW"));
    assert!(topology.contains("[ bonds ]"));
    assert!(topology.contains("1     2     1    0.45000   4184.000"));
    assert!(topology.contains("WT4              1"));
    assert!(topology.contains("NaW              1"));
    assert!(topology.contains("ClW              1"));
}

#[test]
fn inserted_component_inline_beads_define_reusable_coordinate_less_molecule() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("inline_solute.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [60.0, 60.0, 60.0]},
        "membranes": [],
        "solutes": [{
            "name": "CUST",
            "count": 2,
            "beads": [
                {"name": "A", "offset_angstrom": [0.0, 0.0, 0.0], "charge_e": 1.0},
                {"name": "B", "offset_angstrom": [2.0, 0.0, 0.0], "charge_e": -0.5}
            ]
        }],
        "environment": {
            "ions": {"neutralize": true, "salt_molarity_mol_l": 0.0},
            "solvent": {"enabled": false}
        },
        "outputs": {"coordinates": gro, "manifest": temp.path().join("manifest.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["CUST"], 2);
    assert_eq!(value["summary"]["bead_count"], 4);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 1.0);
    assert_eq!(
        value["charge"]["component_charges"][0]["source"],
        "inserted_component.beads.charge_e"
    );

    let atoms = read_gro_residue_atoms(&gro, "CUST");
    assert_eq!(atoms.len(), 4);
    assert_eq!(atoms.iter().filter(|(atom, _)| atom == "A").count(), 2);
    assert_eq!(atoms.iter().filter(|(atom, _)| atom == "B").count(), 2);
    let anchors = atoms
        .iter()
        .filter_map(|(atom, position)| (atom == "A").then_some(*position))
        .collect::<Vec<_>>();
    assert_eq!(anchors.len(), 2);
    assert!(squared_distance3(anchors[0], anchors[1]) >= 4.0);
}

#[test]
fn inserted_component_definition_file_defines_reusable_molecule() {
    let temp = tempfile::tempdir().unwrap();
    let definition = temp.path().join("molecule.json");
    let gro = temp.path().join("definition_solute.gro");
    let top = temp.path().join("definition_solute.top");
    std::fs::write(
        &definition,
        serde_json::to_string_pretty(&json!({
            "schema_version": MOLECULE_DEFINITION_SCHEMA_VERSION,
            "name": "DEFN",
            "net_charge_e": 0.5,
            "residues": [{
                "name": "R0",
                "beads": [
                    {"name": "A", "offset_angstrom": [0.0, 0.0, 0.0], "charge_e": 1.0},
                    {"name": "B", "offset_angstrom": [2.0, 0.0, 0.0], "charge_e": -0.5},
                    {"name": "C", "offset_angstrom": [4.0, 0.0, 0.0], "charge_e": 0.0},
                    {"name": "D", "offset_angstrom": [6.0, 0.0, 0.0], "charge_e": 0.0}
                ]
            }],
            "bonds": [{"bead_indices": [0, 1], "length_nm": 0.2, "force_kj_mol_nm2": 500.0}],
            "angles": [{"bead_indices": [0, 1, 2], "angle_degrees": 120.0, "force_kj_mol_rad2": 75.0}],
            "dihedrals": [{"bead_indices": [0, 1, 2, 3], "phase_degrees": 180.0, "force_kj_mol": 5.0, "multiplicity": 2}]
        }))
        .unwrap(),
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [60.0, 60.0, 60.0]},
        "membranes": [],
        "solutes": [{
            "name": "REQ",
            "count": 2,
            "definition": definition
        }],
        "environment": {
            "ions": {"neutralize": true, "salt_molarity_mol_l": 0.0},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro,
            "topology": top,
            "manifest": temp.path().join("manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["REQ"], 2);
    assert_eq!(value["summary"]["bead_count"], 8);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 1.0);
    assert_eq!(
        value["charge"]["component_charges"][0]["source"],
        format!("molecule_definition:{}", definition.display())
    );

    let atoms = read_gro_residue_atoms(&gro, "DEFN");
    assert_eq!(atoms.len(), 8);
    assert_eq!(atoms.iter().filter(|(atom, _)| atom == "A").count(), 2);
    assert_eq!(atoms.iter().filter(|(atom, _)| atom == "B").count(), 2);
    assert_eq!(atoms.iter().filter(|(atom, _)| atom == "C").count(), 2);
    assert_eq!(atoms.iter().filter(|(atom, _)| atom == "D").count(), 2);
    let topology = std::fs::read_to_string(&top).unwrap();
    assert!(topology.contains("[ moleculetype ]"));
    assert!(topology.contains("REQ"));
    assert!(topology.contains("[ bonds ]"));
    assert!(topology.contains("1     2     1    0.20000    500.000"));
    assert!(topology.contains("[ angles ]"));
    assert!(topology.contains("1     2     3     2    120.000     75.000"));
    assert!(topology.contains("[ dihedrals ]"));
    assert!(topology.contains("1     2     3     4     1    180.000      5.000     2"));
}

#[test]
fn inserted_component_definition_file_charge_mismatch_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let definition = temp.path().join("bad_molecule.json");
    std::fs::write(
        &definition,
        serde_json::to_string_pretty(&json!({
            "schema_version": MOLECULE_DEFINITION_SCHEMA_VERSION,
            "name": "BAD",
            "net_charge_e": 0.0,
            "beads": [{"name": "A", "charge_e": 1.0}]
        }))
        .unwrap(),
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [],
        "solutes": [{"name": "BAD", "definition": definition}],
        "environment": {"ions": {"neutralize": false}, "solvent": {"enabled": false}},
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("definition net_charge_e"));
}
