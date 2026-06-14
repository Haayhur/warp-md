use super::*;

#[test]
fn nearest_allowed_leaflet_point_projects_into_polygon_patch() {
    let membrane = MembraneRequest {
        name: "geom".to_string(),
        center_xy_angstrom: None,
        size_xy_angstrom: None,
        center_z_angstrom: 0.0,
        solvate_voids: true,
        solvent_exclusion_half_thickness_angstrom:
            default_membrane_solvent_exclusion_half_thickness(),
        protein_boundary: None,
        leaflets: Vec::new(),
    };
    let leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: None,
        exclusions: Vec::new(),
        regions: vec![LeafletRegion {
            name: Some("triangle".to_string()),
            role: "patch".to_string(),
            geometry: RegionGeometry::Polygon {
                points_angstrom: vec![[-10.0, -10.0], [10.0, -10.0], [0.0, 10.0]],
                scale_xy: None,
                rotate_degrees: 0.0,
            },
        }],
        composition: Vec::new(),
    };
    let bounds = LayoutBounds {
        xmin: -30.0,
        xmax: 30.0,
        ymin: -30.0,
        ymax: 30.0,
    };
    let point = LayoutPoint {
        x: 20.0,
        y: 0.0,
        radius: 0.5,
    };
    let projected = nearest_allowed_leaflet_point(
        point,
        &membrane,
        &leaflet,
        &[],
        bounds,
        LayoutPeriodicity::default(),
        None,
    )
    .unwrap();
    let projected = projected.unwrap();
    assert!(region_contains_point(
        &leaflet.regions[0],
        [projected.x, projected.y]
    ));
    assert!(projected.x < point.x);
}

#[test]
fn charge_summary_neutralizes_lipid_charge() {
    let temp = tempfile::tempdir().unwrap();
    let mut request: BuildRequest = serde_json::from_value(example_request()).unwrap();
    request.outputs.manifest = temp
        .path()
        .join("manifest.json")
        .to_string_lossy()
        .to_string();
    request.outputs.coordinates = Some(
        temp.path()
            .join("membrane.gro")
            .to_string_lossy()
            .to_string(),
    );
    request.outputs.topology = Some(temp.path().join("topol.top").to_string_lossy().to_string());
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], -16.0);
    assert_eq!(value["charge"]["neutralization"]["counterion"], "Na+");
    assert_eq!(value["charge"]["neutralization"]["counterion_count"], 16);
    assert_eq!(value["charge"]["neutralization"]["salt_method"], "add");
    assert_eq!(value["charge"]["neutralization"]["cation_delta"], 16);
    assert_eq!(value["charge"]["neutralization"]["anion_delta"], 0);
    assert_eq!(value["charge"]["neutralization"]["residual_charge_e"], 0.0);
    assert_eq!(value["summary"]["bead_count"], 17900);
    assert_eq!(value["summary"]["solvent_counts"]["W"], 15626);
    assert_eq!(value["summary"]["solvent_counts"]["NA"], 185);
    assert_eq!(value["summary"]["solvent_counts"]["CL"], 169);
    assert_eq!(value["placement"]["solvent"]["inserted_count"], 15980);
    assert!(temp.path().join("manifest.json").exists());
    assert!(temp.path().join("membrane.gro").exists());
    assert!(temp.path().join("topol.top").exists());
}

#[test]
fn build_outputs_support_gro_pdb_cif_and_log_artifacts() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [{
            "name": "tiny",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "DUM", "count": 1}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": temp.path().join("main.pdb"),
            "gro": temp.path().join("main.gro"),
            "pdb": temp.path().join("copy.pdb"),
            "cif": temp.path().join("copy.cif"),
            "log": temp.path().join("build.log"),
            "snapshot": temp.path().join("snapshot.json"),
            "manifest": temp.path().join("manifest.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    for name in [
        "main.pdb",
        "main.gro",
        "copy.pdb",
        "copy.cif",
        "build.log",
        "snapshot.json",
    ] {
        assert!(temp.path().join(name).exists(), "missing {name}");
    }
    let log = std::fs::read_to_string(temp.path().join("build.log")).unwrap();
    assert!(log.contains("warp-cg build"));
    assert!(log.contains("[ box ]"));
    assert!(log.contains("[ charge ]"));
    assert!(log.contains("[ placement ]"));
    assert!(log.contains("diagnostics: bead_count="));
    assert!(log.contains("exclusion_violation_count="));
    assert!(log.contains("[ artifacts ]"));
    let snapshot: Value =
        serde_json::from_str(&std::fs::read_to_string(temp.path().join("snapshot.json")).unwrap())
            .unwrap();
    assert_eq!(snapshot["schema_version"], "warp-cg.build.snapshot.v1");
    assert_eq!(snapshot["request"]["mode"], "membrane");
    assert_eq!(
        snapshot["result"]["placement"]["diagnostics"]["bead_count"],
        value["placement"]["diagnostics"]["bead_count"]
    );
    assert_eq!(
        value["placement"]["diagnostics"]["exclusion_violation_count"],
        0
    );
    assert!(value["placement"]["diagnostics"]["bounds_min_angstrom"].is_array());
    let snapshot_path = temp
        .path()
        .join("snapshot.json")
        .to_string_lossy()
        .to_string();
    assert_eq!(snapshot["result"]["artifacts"]["snapshot"], snapshot_path);
    assert_eq!(value["artifacts"]["snapshot"], snapshot_path);
}

#[test]
fn build_outputs_respect_overwrite_false() {
    let temp = tempfile::tempdir().unwrap();
    let coordinate_path = temp.path().join("main.gro");
    std::fs::write(&coordinate_path, "sentinel").unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [{
            "name": "tiny",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "DUM", "count": 1}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": coordinate_path,
            "manifest": temp.path().join("manifest.json"),
            "overwrite": false
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_ne!(code, 0, "{value}");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("outputs.overwrite is false"),
        "{value}"
    );
    assert_eq!(
        std::fs::read_to_string(&coordinate_path).unwrap(),
        "sentinel"
    );
}

#[test]
fn build_outputs_backup_existing_files_before_overwrite() {
    let temp = tempfile::tempdir().unwrap();
    let coordinate_path = temp.path().join("main.gro");
    let manifest_path = temp.path().join("manifest.json");
    std::fs::write(&coordinate_path, "coordinate sentinel").unwrap();
    std::fs::write(&manifest_path, "manifest sentinel").unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [{
            "name": "tiny",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "DUM", "count": 1}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": coordinate_path,
            "manifest": manifest_path,
            "overwrite": true,
            "backup_existing": true
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["artifacts"]["output_policy"]["overwrite"], true);
    assert_eq!(value["artifacts"]["output_policy"]["backup_existing"], true);
    assert_eq!(
        std::fs::read_to_string(temp.path().join("main.gro.bak1")).unwrap(),
        "coordinate sentinel"
    );
    assert_eq!(
        std::fs::read_to_string(temp.path().join("manifest.json.bak1")).unwrap(),
        "manifest sentinel"
    );
}

#[test]
fn build_result_reports_orthorhombic_unit_cell_metadata() {
    let temp = tempfile::tempdir().unwrap();
    let gro_path = temp.path().join("orthorhombic.gro");
    let pdb_path = temp.path().join("orthorhombic.pdb");
    let cif_path = temp.path().join("orthorhombic.cif");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [12.0, 13.0, 14.0]},
        "membranes": [{
            "name": "tiny",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "DUM", "count": 1}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "gro": gro_path,
            "pdb": pdb_path,
            "cif": cif_path,
            "manifest": temp.path().join("manifest.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["box_meta"]["box_type"], "orthorhombic");
    assert_eq!(value["box_meta"]["pbc"], "xyz");
    assert_eq!(
        value["box_meta"]["box_size_angstrom"],
        json!([12.0, 13.0, 14.0])
    );
    assert_eq!(
        value["box_meta"]["unit_cell_angstrom"],
        json!([12.0, 13.0, 14.0, 90.0, 90.0, 90.0])
    );
    assert_eq!(
        value["box_meta"]["box_vectors_angstrom"],
        json!([[12.0, 0.0, 0.0], [0.0, 13.0, 0.0], [0.0, 0.0, 14.0]])
    );

    let gro_text = std::fs::read_to_string(&gro_path).unwrap();
    let box_fields = gro_text
        .lines()
        .last()
        .unwrap()
        .split_whitespace()
        .collect::<Vec<_>>();
    assert_eq!(box_fields, ["1.20000", "1.30000", "1.40000"]);

    let pdb_text = std::fs::read_to_string(&pdb_path).unwrap();
    assert!(pdb_text.contains("CRYST1   12.000   13.000   14.000  90.00  90.00  90.00"));

    let cif_text = std::fs::read_to_string(&cif_path).unwrap();
    assert!(cif_text.contains("_cell.length_a 12.000"));
    assert!(cif_text.contains("_cell.length_b 13.000"));
    assert!(cif_text.contains("_cell.length_c 14.000"));
    assert!(cif_text.contains("_cell.angle_alpha 90.00"));
    assert!(cif_text.contains("_cell.angle_beta 90.00"));
    assert!(cif_text.contains("_cell.angle_gamma 90.00"));
}

#[test]
fn build_result_and_gro_output_use_manual_triclinic_unit_cell() {
    let temp = tempfile::tempdir().unwrap();
    let gro_path = temp.path().join("triclinic.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_type": "triclinic",
            "pbc": "xy",
            "box_size_angstrom": [20.0, 20.0, 30.0],
            "unit_cell_angstrom": [10.0, 20.0, 30.0, 90.0, 90.0, 60.0]
        },
        "membranes": [{
            "name": "tiny",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "DUM", "count": 1}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro_path,
            "manifest": temp.path().join("manifest.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["box_meta"]["box_type"], "triclinic");
    assert_eq!(value["box_meta"]["pbc"], "xy");
    assert_eq!(
        value["box_meta"]["unit_cell_angstrom"],
        json!([10.0, 20.0, 30.0, 90.0, 90.0, 60.0])
    );
    let vectors = value["box_meta"]["box_vectors_angstrom"]
        .as_array()
        .unwrap();
    assert!((vectors[1][0].as_f64().unwrap() - 10.0).abs() < 1.0e-4);
    assert!((vectors[1][1].as_f64().unwrap() - 17.320_508).abs() < 1.0e-4);
    let gro_text = std::fs::read_to_string(gro_path).unwrap();
    let box_fields = gro_text
        .lines()
        .last()
        .unwrap()
        .split_whitespace()
        .collect::<Vec<_>>();
    assert_eq!(box_fields.len(), 9);
    assert_eq!(box_fields[0], "1.00000");
    assert_eq!(box_fields[1], "1.73205");
    assert_eq!(box_fields[2], "3.00000");
    assert_eq!(box_fields[5], "1.00000");
}

#[test]
fn orthorhombic_box_rejects_non_orthogonal_manual_unit_cell() {
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_type": "orthorhombic",
            "box_size_angstrom": [20.0, 20.0, 30.0],
            "unit_cell_angstrom": [10.0, 20.0, 30.0, 90.0, 90.0, 60.0]
        },
        "membranes": [{
            "name": "tiny",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "DUM", "count": 1}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": "unused.json"}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_ne!(code, 0, "{value}");
    assert!(
        value["error"]["message"]
            .as_str()
            .unwrap()
            .contains("angles must be 90"),
        "{value}"
    );
}

#[test]
fn salt_method_mean_splits_neutralization_between_cation_add_and_anion_remove() {
    let policy = IonPolicy {
        neutralize: true,
        salt_method: "mean".to_string(),
        ..IonPolicy::default()
    };
    let summary = resolve_neutralization(&policy, Some(-32.0));
    assert_eq!(summary.counterion.as_deref(), Some("Na+"));
    assert_eq!(summary.counterion_count, 32);
    assert_eq!(summary.cation_delta, 16);
    assert_eq!(summary.anion_delta, -16);
    assert_eq!(summary.residual_charge_e, Some(0.0));
}

#[test]
fn invalid_salt_method_is_rejected() {
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
        "environment": {
            "ions": {"neutralize": true, "salt_method": "median"}
        },
        "outputs": {"manifest": "unused.json"}
    });
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = validate_request_json(&text);
    assert_eq!(code, 2);
    assert!(value["errors"][0]["message"]
        .as_str()
        .unwrap()
        .contains("salt_method must be add, remove, or mean"));
}

#[test]
fn solvent_library_resolves_martini_water_defaults() {
    let mut solvent = SolventPolicy {
        enabled: true,
        name: "SW".to_string(),
        ..SolventPolicy::default()
    };
    let species = resolved_solvent_species(&solvent);
    assert_eq!(species[0].name, "SW");
    assert_eq!(species[0].mapping_ratio, 3.0);
    assert_eq!(species[0].molar_mass_g_mol, default_solvent_molar_mass());
    assert_eq!(species[0].density_kg_m3, default_solvent_density());

    solvent.species = vec![
        SolventComponent {
            name: "TW".to_string(),
            ratio: 2.0,
            mapping_ratio: default_solvent_mapping_ratio(),
            molar_mass_g_mol: default_solvent_molar_mass(),
            density_kg_m3: default_solvent_density(),
            charge_e: 0.0,
        },
        SolventComponent {
            name: "custom".to_string(),
            ratio: 1.0,
            mapping_ratio: 7.0,
            molar_mass_g_mol: 44.0,
            density_kg_m3: 700.0,
            charge_e: -0.5,
        },
    ];
    let mixed = resolved_solvent_species(&solvent);
    assert_eq!(mixed[0].name, "TW");
    assert_eq!(mixed[0].mapping_ratio, 2.0);
    assert_eq!(mixed[1].name, "custom");
    assert_eq!(mixed[1].mapping_ratio, 7.0);
    assert_eq!(mixed[1].charge_e, -0.5);

    solvent.species.clear();
    solvent.name = "TIP3".to_string();
    let tip3 = resolved_solvent_species(&solvent);
    assert_eq!(tip3[0].name, "TIP3");
    assert_eq!(tip3[0].mapping_ratio, 1.0);
    assert!((tip3[0].charge_e).abs() < 1.0e-6);
    assert_eq!(tip3[0].beads.len(), 3);
    assert_eq!(tip3[0].beads[0].atom_name, "OW");
    assert_eq!(tip3[0].beads[1].offset_angstrom, [0.74, 0.64, 0.0]);

    solvent.name = "TIP4".to_string();
    let tip4 = resolved_solvent_species(&solvent);
    assert_eq!(tip4[0].name, "TIP4");
    assert_eq!(tip4[0].beads.len(), 4);
    assert_eq!(tip4[0].beads[3].atom_name, "MW");
    assert_eq!(tip4[0].beads[3].offset_angstrom, [0.0, 0.32, 0.0]);
    assert!((tip4[0].charge_e).abs() < 1.0e-6);

    solvent.name = "TIP5".to_string();
    let tip5 = resolved_solvent_species(&solvent);
    assert_eq!(tip5[0].name, "TIP5");
    assert_eq!(tip5[0].beads.len(), 5);
    assert_eq!(tip5[0].beads[4].atom_name, "LP2");
    assert_eq!(tip5[0].beads[4].offset_angstrom, [-0.2, -0.2, 0.0]);
    assert!((tip5[0].charge_e).abs() < 1.0e-6);

    solvent.name = "OPC".to_string();
    let opc = resolved_solvent_species(&solvent);
    assert_eq!(opc[0].name, "OPC");
    assert_eq!(opc[0].beads.len(), 4);
    assert_eq!(opc[0].beads[3].atom_name, "EPW");
    assert_eq!(opc[0].beads[3].offset_angstrom, [0.0, 0.16, 0.0]);
    assert!((opc[0].charge_e).abs() < 1.0e-5);

    solvent.name = "CHARMM-TIP3".to_string();
    let charmm = resolved_solvent_species(&solvent);
    assert_eq!(charmm[0].name, "TIP3");
    assert_eq!(charmm[0].beads[0].atom_name, "OH2");

    solvent.name = "SOL".to_string();
    let sol_tip3 = resolved_solvent_species(&solvent);
    assert_eq!(sol_tip3[0].name, "SOL");
    assert_eq!(sol_tip3[0].beads.len(), 3);
    assert_eq!(sol_tip3[0].beads[0].atom_name, "OW");
    assert!((sol_tip3[0].charge_e).abs() < 1.0e-6);

    solvent.name = "SOL-TIP4".to_string();
    let sol_tip4 = resolved_solvent_species(&solvent);
    assert_eq!(sol_tip4[0].name, "SOL");
    assert_eq!(sol_tip4[0].beads.len(), 4);
    assert_eq!(sol_tip4[0].beads[3].atom_name, "MW");

    solvent.name = "SOL-TIP5".to_string();
    let sol_tip5 = resolved_solvent_species(&solvent);
    assert_eq!(sol_tip5[0].name, "SOL");
    assert_eq!(sol_tip5[0].beads.len(), 5);
    assert_eq!(sol_tip5[0].beads[4].atom_name, "LP2");
}
