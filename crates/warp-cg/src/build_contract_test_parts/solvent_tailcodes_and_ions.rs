use super::*;

#[test]
fn cardiolipin_tailcode_solvent_library_emits_four_tail_fragments() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("cardiolipin_tailcode_solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "name": "cardiolipin:C,D,T,F",
                "molarity_mol_l": 0.21,
                "grid_spacing_angstrom": 20.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("cardiolipin_tailcode_solvent_manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["LCDTF"], 1);
    assert_eq!(value["summary"]["bead_count"], 11);

    let atoms = read_gro_residue_atoms(&gro, "LCDTF");
    assert_eq!(atoms.len(), 11);
    let glc = atoms.iter().find(|(atom, _)| atom == "GLC").unwrap().1;
    let po41 = atoms.iter().find(|(atom, _)| atom == "PO41").unwrap().1;
    let po42 = atoms.iter().find(|(atom, _)| atom == "PO42").unwrap().1;
    let c1a1 = atoms.iter().find(|(atom, _)| atom == "C1A1").unwrap().1;
    let d1b1 = atoms.iter().find(|(atom, _)| atom == "D1B1").unwrap().1;
    let t1a2 = atoms.iter().find(|(atom, _)| atom == "T1A2").unwrap().1;
    let f1b2 = atoms.iter().find(|(atom, _)| atom == "F1B2").unwrap().1;
    assert!((po41[0] - glc[0] + 1.5).abs() < 0.02);
    assert!((po42[0] - glc[0] - 1.5).abs() < 0.02);
    assert!((c1a1[0] - glc[0] + 1.5).abs() < 0.02);
    assert!((c1a1[2] - glc[2] + 9.0).abs() < 0.02);
    assert!((d1b1[0] - glc[0] + 4.0).abs() < 0.02);
    assert!((t1a2[0] - glc[0] - 1.5).abs() < 0.02);
    assert!((f1b2[0] - glc[0] - 4.0).abs() < 0.02);
}

#[test]
fn sphingolipid_tailcode_solvent_library_emits_two_tail_fragments() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("sphingolipid_tailcode_solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "name": "sphingolipid:PC,C,D",
                "molarity_mol_l": 0.21,
                "grid_spacing_angstrom": 20.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("sphingolipid_tailcode_solvent_manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["SPCCD"], 1);
    assert_eq!(value["summary"]["bead_count"], 6);

    let atoms = read_gro_residue_atoms(&gro, "SPCCD");
    assert_eq!(atoms.len(), 6);
    let po4 = atoms.iter().find(|(atom, _)| atom == "PO4").unwrap().1;
    let oh1 = atoms.iter().find(|(atom, _)| atom == "OH1").unwrap().1;
    let am2 = atoms.iter().find(|(atom, _)| atom == "AM2").unwrap().1;
    let c1a = atoms.iter().find(|(atom, _)| atom == "C1A").unwrap().1;
    let d1b = atoms.iter().find(|(atom, _)| atom == "D1B").unwrap().1;
    assert!((po4[2] - oh1[2] - 3.0).abs() < 0.02);
    assert!((am2[0] - oh1[0] - 2.5).abs() < 0.02);
    assert!((c1a[0] - oh1[0]).abs() < 0.02);
    assert!((c1a[2] - oh1[2] + 3.0).abs() < 0.02);
    assert!((d1b[0] - am2[0]).abs() < 0.02);
    assert!((d1b[2] - am2[2] + 3.0).abs() < 0.02);
}

#[test]
fn bmp_tailcode_solvent_library_emits_two_tail_fragments() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("bmp_tailcode_solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "name": "bmp2:C,D",
                "molarity_mol_l": 0.21,
                "grid_spacing_angstrom": 20.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("bmp_tailcode_solvent_manifest.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["solvent_counts"]["B2CD"], 1);
    assert_eq!(value["summary"]["bead_count"], 7);

    let atoms = read_gro_residue_atoms(&gro, "B2CD");
    assert_eq!(atoms.len(), 7);
    let po4 = atoms.iter().find(|(atom, _)| atom == "PO4").unwrap().1;
    let gl1 = atoms.iter().find(|(atom, _)| atom == "GL1").unwrap().1;
    let gl2 = atoms.iter().find(|(atom, _)| atom == "GL2").unwrap().1;
    let c1a = atoms.iter().find(|(atom, _)| atom == "C1A").unwrap().1;
    let d1b = atoms.iter().find(|(atom, _)| atom == "D1B").unwrap().1;
    assert!((gl1[0] - po4[0] - 2.0).abs() < 0.02);
    assert!((gl1[2] - po4[2] + 5.5).abs() < 0.02);
    assert!((gl2[0] - po4[0] + 2.0).abs() < 0.02);
    assert!((gl2[2] - po4[2] + 5.5).abs() < 0.02);
    assert!((c1a[2] - gl1[2] + 3.0).abs() < 0.02);
    assert!((d1b[2] - gl2[2] + 3.0).abs() < 0.02);
}

#[test]
fn invalid_tailcode_solvent_name_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let invalid_code = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {
                "enabled": true,
                "name": "hydrocarbon:CX"
            }
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });

    let (code, value) = validate_request_json(&serde_json::to_string(&invalid_code).unwrap());
    assert_eq!(code, 2);
    assert!(value["errors"][0]["message"]
        .as_str()
        .unwrap()
        .contains("invalid generated solvent tail code"));

    let invalid_arity = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {
                "enabled": true,
                "name": "diglyceride:CD"
            }
        },
        "outputs": {"manifest": temp.path().join("manifest2.json")}
    });

    let (code, value) = validate_request_json(&serde_json::to_string(&invalid_arity).unwrap());
    assert_eq!(code, 2);
    assert!(value["errors"][0]["message"]
        .as_str()
        .unwrap()
        .contains("invalid generated solvent tail code"));

    let invalid_cardiolipin_arity = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {
                "enabled": true,
                "name": "cardiolipin:C,D,T"
            }
        },
        "outputs": {"manifest": temp.path().join("manifest3.json")}
    });

    let (code, value) =
        validate_request_json(&serde_json::to_string(&invalid_cardiolipin_arity).unwrap());
    assert_eq!(code, 2);
    assert!(value["errors"][0]["message"]
        .as_str()
        .unwrap()
        .contains("invalid generated solvent tail code"));

    let invalid_sphingolipid_head = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {
                "enabled": true,
                "name": "sphingolipid:BAD,C,D"
            }
        },
        "outputs": {"manifest": temp.path().join("manifest4.json")}
    });

    let (code, value) =
        validate_request_json(&serde_json::to_string(&invalid_sphingolipid_head).unwrap());
    assert_eq!(code, 2);
    assert!(value["errors"][0]["message"]
        .as_str()
        .unwrap()
        .contains("invalid generated solvent tail code"));

    let invalid_sphingolipid_arity = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {
                "enabled": true,
                "name": "sphingolipid:PC,C"
            }
        },
        "outputs": {"manifest": temp.path().join("manifest5.json")}
    });

    let (code, value) =
        validate_request_json(&serde_json::to_string(&invalid_sphingolipid_arity).unwrap());
    assert_eq!(code, 2);
    assert!(value["errors"][0]["message"]
        .as_str()
        .unwrap()
        .contains("invalid generated solvent tail code"));
}

#[test]
fn ion_library_resolves_single_bead_names_and_defaulted_charges() {
    let sod = resolved_ion_species(&[], "SOD", default_cation_charge());
    assert_eq!(sod[0].residue_name, "SOD");
    assert_eq!(sod[0].atom_name, "SOD");
    assert_eq!(sod[0].charge_e, 1);

    let ca = resolved_ion_species(&[], "Ca", default_cation_charge());
    assert_eq!(ca[0].name, "Ca");
    assert_eq!(ca[0].residue_name, "CA");
    assert_eq!(ca[0].atom_name, "CA");
    assert_eq!(ca[0].charge_e, 2);

    let m2_ca = resolved_ion_species(&[], "CA+", default_cation_charge());
    assert_eq!(m2_ca[0].name, "CA+");
    assert_eq!(m2_ca[0].residue_name, "CA+");
    assert_eq!(m2_ca[0].atom_name, "CA+");
    assert_eq!(m2_ca[0].charge_e, 1);

    let nc3 = resolved_ion_species(&[], "NC3+", default_cation_charge());
    assert_eq!(nc3[0].name, "NC3+");
    assert_eq!(nc3[0].residue_name, "NC3+");
    assert_eq!(nc3[0].atom_name, "NC3");
    assert_eq!(nc3[0].charge_e, 1);

    let cla = resolved_ion_species(&[], "CLA", default_anion_charge());
    assert_eq!(cla[0].residue_name, "CLA");
    assert_eq!(cla[0].atom_name, "CLA");
    assert_eq!(cla[0].charge_e, -1);

    let anions = resolved_ion_species(
        &[
            IonComponent {
                name: "CLO4".to_string(),
                ratio: 1.0,
                charge_e: 0,
            },
            IonComponent {
                name: "IOD".to_string(),
                ratio: 1.0,
                charge_e: 0,
            },
        ],
        "Cl-",
        default_anion_charge(),
    );
    assert_eq!(anions[0].name, "CLO4");
    assert_eq!(anions[0].residue_name, "CLO4");
    assert_eq!(anions[0].atom_name, "CLO");
    assert_eq!(anions[0].charge_e, -1);
    assert_eq!(anions[1].name, "IOD");
    assert_eq!(anions[1].residue_name, "IOD");
    assert_eq!(anions[1].atom_name, "ID");
}

#[test]
fn ion_library_resolves_sirah_water_ion_aliases() {
    let naw = lookup_ion_library("NaW").unwrap();
    assert_eq!(naw.name, "NaW");
    assert_eq!(naw.atom_name, "NaW");
    assert_eq!(naw.charge_e, 1);

    let clw = lookup_ion_library("ClW").unwrap();
    assert_eq!(clw.name, "ClW");
    assert_eq!(clw.atom_name, "ClW");
    assert_eq!(clw.charge_e, -1);

    let resolved_naw = resolved_ion_species(&[], "NaW", default_cation_charge());
    assert_eq!(resolved_naw[0].residue_name, "NaW");
    assert_eq!(resolved_naw[0].atom_name, "NaW");
    assert_eq!(resolved_naw[0].charge_e, 1);

    let resolved_clw = resolved_ion_species(&[], "ClW", default_anion_charge());
    assert_eq!(resolved_clw[0].residue_name, "ClW");
    assert_eq!(resolved_clw[0].atom_name, "ClW");
    assert_eq!(resolved_clw[0].charge_e, -1);

    let known_cations = known_cation_library_names();
    assert!(known_cations.contains(&"NaW"));
    let known_anions = known_anion_library_names();
    assert!(known_anions.contains(&"ClW"));
}

#[test]
fn ion_library_resolves_m2_cation_aliases_without_colliding_with_divalent_ca() {
    let known = known_cation_library_names();
    assert!(known.contains(&"NC3+"));
    assert!(known.contains(&"CA+"));

    let divalent_ca = lookup_ion_library("CA").unwrap();
    assert_eq!(divalent_ca.name, "CA");
    assert_eq!(divalent_ca.atom_name, "CA");
    assert_eq!(divalent_ca.charge_e, 2);

    let monovalent_ca = lookup_ion_library("CA+").unwrap();
    assert_eq!(monovalent_ca.name, "CA+");
    assert_eq!(monovalent_ca.atom_name, "CA+");
    assert_eq!(monovalent_ca.charge_e, 1);

    let nc3 = lookup_ion_library("NC3+").unwrap();
    assert_eq!(nc3.name, "NC3+");
    assert_eq!(nc3.atom_name, "NC3");
    assert_eq!(nc3.charge_e, 1);
}

#[test]
fn ion_library_preserves_atomistic_signed_alias_atom_names() {
    let known_cations = known_cation_library_names();
    assert!(known_cations.contains(&"Na"));
    assert!(known_cations.contains(&"NA+"));
    let known_anions = known_anion_library_names();
    assert!(known_anions.contains(&"Cl"));
    assert!(known_anions.contains(&"CL-"));

    let mixed_case_sodium = lookup_ion_library("Na").unwrap();
    assert_eq!(mixed_case_sodium.name, "NA");
    assert_eq!(mixed_case_sodium.atom_name, "Na+");
    assert_eq!(mixed_case_sodium.charge_e, 1);

    let sodium = lookup_ion_library("Na+").unwrap();
    assert_eq!(sodium.name, "NA");
    assert_eq!(sodium.atom_name, "Na+");
    assert_eq!(sodium.charge_e, 1);

    let mixed_case_chloride = lookup_ion_library("Cl").unwrap();
    assert_eq!(mixed_case_chloride.name, "CL");
    assert_eq!(mixed_case_chloride.atom_name, "Cl-");
    assert_eq!(mixed_case_chloride.charge_e, -1);

    let chloride = lookup_ion_library("Cl-").unwrap();
    assert_eq!(chloride.name, "CL");
    assert_eq!(chloride.atom_name, "Cl-");
    assert_eq!(chloride.charge_e, -1);

    let resolved_mixed_case_sodium = resolved_ion_species(&[], "Na", default_cation_charge());
    assert_eq!(resolved_mixed_case_sodium[0].residue_name, "NA");
    assert_eq!(resolved_mixed_case_sodium[0].atom_name, "Na+");

    let resolved_sodium = resolved_ion_species(&[], "Na+", default_cation_charge());
    assert_eq!(resolved_sodium[0].residue_name, "NA");
    assert_eq!(resolved_sodium[0].atom_name, "Na+");

    let resolved_mixed_case_chloride = resolved_ion_species(&[], "Cl", default_anion_charge());
    assert_eq!(resolved_mixed_case_chloride[0].residue_name, "CL");
    assert_eq!(resolved_mixed_case_chloride[0].atom_name, "Cl-");

    let resolved_chloride = resolved_ion_species(&[], "Cl-", default_anion_charge());
    assert_eq!(resolved_chloride[0].residue_name, "CL");
    assert_eq!(resolved_chloride[0].atom_name, "Cl-");

    let martini_sodium = lookup_ion_library("NA").unwrap();
    assert_eq!(martini_sodium.atom_name, "NA");
    let martini_chloride = lookup_ion_library("CL").unwrap();
    assert_eq!(martini_chloride.atom_name, "CL");
}

#[test]
fn ion_library_resolves_reference_extra_m3_magnesium() {
    let known = known_cation_library_names();
    assert!(known.contains(&"MG"));

    let magnesium = lookup_ion_library("MG").unwrap();
    assert_eq!(magnesium.name, "MG");
    assert_eq!(magnesium.atom_name, "MG");
    assert_eq!(magnesium.charge_e, 2);

    let fallback = resolved_ion_species(&[], "MG", default_cation_charge());
    assert_eq!(fallback[0].residue_name, "MG");
    assert_eq!(fallback[0].atom_name, "MG");
    assert_eq!(fallback[0].charge_e, 2);

    let component = resolved_ion_species(
        &[IonComponent {
            name: "MG".to_string(),
            ratio: 1.0,
            charge_e: 0,
        }],
        "NA",
        default_cation_charge(),
    );
    assert_eq!(component[0].name, "MG");
    assert_eq!(component[0].residue_name, "MG");
    assert_eq!(component[0].atom_name, "MG");
    assert_eq!(component[0].charge_e, 2);
}

#[test]
fn ion_library_defaulted_component_charges_validate_from_json() {
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [],
        "solutes": [{"name": "GLY", "count": 1}],
        "environment": {
            "ions": {
                "neutralize": false,
                "cations": [{"name": "CA"}, {"name": "MG"}],
                "anions": [{"name": "CLO4"}]
            },
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": "manifest.json"}
    });

    let (code, value) = validate_request_json(&serde_json::to_string(&request).unwrap());

    assert_eq!(code, 0, "{value}");
}

#[test]
fn protein_only_build_does_not_require_membranes() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("protein.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  BB  PRO A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [],
        "proteins": [{
            "name": "PRO",
            "count": 2,
            "coordinates": protein_path,
            "format": "pdb",
            "net_charge_e": 0.0
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": temp.path().join("m.gro"),
            "topology": temp.path().join("t.top"),
            "manifest": temp.path().join("m.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["membrane_count"], 0);
    assert_eq!(value["summary"]["protein_count"], 2);
    assert_eq!(value["summary"]["inserted_counts"]["PRO"], 2);
    assert_eq!(value["summary"]["bead_count"], 2);
    let topology = std::fs::read_to_string(temp.path().join("t.top")).unwrap();
    assert!(topology.contains("PRO"));
}

#[test]
fn protein_only_build_applies_reference_order_rotation_before_translation() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("protein.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  A   PRO A   1       1.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  B   PRO A   1      -1.000   0.000   0.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let gro = temp.path().join("m.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [],
        "proteins": [{
            "name": "PRO",
            "coordinates": protein_path,
            "format": "pdb",
            "net_charge_e": 0.0,
            "placement": {
                "center_method": "cog",
                "center_angstrom": [10.0, 20.0, 0.0],
                "rotate_degrees_xyz": [0.0, 0.0, 90.0]
            }
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("m.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    let text = std::fs::read_to_string(gro).unwrap();
    let coords = text
        .lines()
        .skip(2)
        .take(2)
        .map(|line| {
            [
                line[20..28].trim().parse::<f32>().unwrap() * 10.0,
                line[28..36].trim().parse::<f32>().unwrap() * 10.0,
                line[36..44].trim().parse::<f32>().unwrap() * 10.0,
            ]
        })
        .collect::<Vec<_>>();
    assert!((coords[0][0] - 10.0).abs() < 1.0e-3, "{coords:?}");
    assert!((coords[0][1] - 21.0).abs() < 1.0e-3, "{coords:?}");
    assert!((coords[1][0] - 10.0).abs() < 1.0e-3, "{coords:?}");
    assert!((coords[1][1] - 19.0).abs() < 1.0e-3, "{coords:?}");
}

#[test]
fn inserted_component_rotation_must_be_finite() {
    let mut request: BuildRequest = serde_json::from_value(json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [],
        "proteins": [{
            "name": "PRO",
            "net_charge_e": 0.0,
            "placement": {"rotate_degrees_xyz": [0.0, 0.0, 0.0]}
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": "unused.json"}
    }))
    .unwrap();
    request.proteins[0].placement.rotate_degrees_xyz[2] = f32::NAN;
    let err = validate_request(request).unwrap_err();
    assert!(err
        .to_string()
        .contains("rotate_degrees_xyz values must be finite"));
}
