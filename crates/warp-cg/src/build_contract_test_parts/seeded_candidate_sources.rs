use super::*;

#[test]
fn seeded_random_leaflet_candidates_use_triclinic_xy_basis() {
    let mut system = test_triclinic_system_with_pbc("xy");
    system.unit_cell_angstrom = Some([10.0, 20.0, 30.0, 90.0, 90.0, 60.0]);
    let membrane = MembraneRequest {
        name: "triclinic-random".to_string(),
        center_xy_angstrom: Some([2.0, -1.0]),
        size_xy_angstrom: Some([10.0, 8.660_254]),
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
        apl_angstrom2: Some(60.0),
        exclusions: Vec::new(),
        regions: Vec::new(),
        composition: Vec::new(),
    };
    let bounds = membrane_layout_bounds(&system, &membrane);
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let basis = membrane_layout_basis(&system, &membrane, bounds, periodicity).unwrap();
    let points = random_leaflet_grid(
        701,
        &[0.5; 24],
        bounds,
        Some(basis),
        periodicity,
        &membrane,
        &leaflet,
        &[],
    )
    .unwrap()
    .unwrap();

    assert_eq!(points.len(), 24);
    let repeat = random_leaflet_grid(
        701,
        &[0.5; 24],
        bounds,
        Some(basis),
        periodicity,
        &membrane,
        &leaflet,
        &[],
    )
    .unwrap()
    .unwrap();
    assert_eq!(points, repeat);
    for point in points {
        let fractional = basis.fractional([point.x, point.y]);
        assert!(fractional[0] > 0.0 && fractional[0] < 1.0);
        assert!(fractional[1] > 0.0 && fractional[1] < 1.0);
    }
}

#[test]
fn seeded_random_solvent_candidate_source_is_reproducible_and_seed_sensitive() {
    let temp = tempfile::tempdir().unwrap();
    let run = |seed: u64, name: &str| {
        let gro = temp.path().join(format!("{name}.gro"));
        let request = json!({
            "schema_version": BUILD_SCHEMA_VERSION,
            "mode": "membrane",
            "system": {
                "box_size_angstrom": [60.0, 60.0, 60.0],
                "placement": {
                    "mode": "seeded",
                    "random_seed": seed,
                    "candidate_source": "random"
                }
            },
            "membranes": [],
            "environment": {
                "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
                "solvent": {
                    "enabled": true,
                    "molarity_mol_l": 0.5,
                    "grid_spacing_angstrom": 6.0
                }
            },
            "outputs": {
                "coordinates": gro,
                "manifest": temp.path().join(format!("{name}.json"))
            }
        });
        let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
        assert_eq!(code, 0, "{value}");
        assert_eq!(
            value["placement"]["solvent"]["algorithm"],
            "free_volume_count_seeded_random"
        );
        (value, read_gro_residue_positions(&gro, "W"))
    };

    let (first_value, first_positions) = run(101, "first");
    let (_second_value, second_positions) = run(101, "second");
    let (_third_value, third_positions) = run(102, "third");
    assert_eq!(first_positions, second_positions);
    assert_ne!(first_positions, third_positions);
    assert_eq!(
        first_value["placement"]["solvent"]["grid_squeeze_pass_count"],
        0
    );
    assert_eq!(first_value["placement"]["solvent"]["kick_attempt_count"], 0);
}

#[test]
fn seeded_solvent_grid_squeeze_expands_dense_candidate_queue() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("dense.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [20.0, 20.0, 20.0],
            "placement": {"mode": "seeded", "random_seed": 44}
        },
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "molarity_mol_l": 8.0,
                "grid_spacing_angstrom": 10.0,
                "excluded_bead_radius_angstrom": 1.0,
                "exclusion_buffer_angstrom": 0.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("manifest.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    let solvent = &value["placement"]["solvent"];
    assert_eq!(solvent["grid_point_count"], 8);
    assert!(solvent["inserted_count"].as_u64().unwrap() > 8);
    assert!(solvent["grid_squeeze_pass_count"].as_u64().unwrap() > 0);
    assert!(solvent["squeezed_candidate_count"].as_u64().unwrap() > 0);
    assert!(solvent["min_grid_spacing_angstrom"].as_f64().unwrap() < 10.0);
    assert!(
        solvent["kicked_inserted_count"].as_u64().unwrap()
            <= solvent["inserted_count"].as_u64().unwrap() - 8
    );
    assert_eq!(
        read_gro_residue_positions(&gro, "W").len() as u64,
        solvent["inserted_count"].as_u64().unwrap()
    );
}

#[test]
fn deterministic_solvent_grid_squeeze_expands_dense_candidate_queue() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("solvent.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [20.0, 20.0, 20.0]},
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "molarity_mol_l": 8.0,
                "grid_spacing_angstrom": 10.0,
                "excluded_bead_radius_angstrom": 1.0,
                "exclusion_buffer_angstrom": 1.0
            }
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("manifest.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    let solvent = &value["placement"]["solvent"];
    assert_eq!(solvent["grid_point_count"], 8);
    assert!(solvent["inserted_count"].as_u64().unwrap() > 8);
    assert!(solvent["grid_squeeze_pass_count"].as_u64().unwrap() > 0);
    assert!(solvent["squeezed_candidate_count"].as_u64().unwrap() > 0);
    assert!(solvent["min_grid_spacing_angstrom"].as_f64().unwrap() < 10.0);
    assert_eq!(solvent["kick_attempt_count"], 0);
    assert_eq!(
        solvent["density"]["target_count"],
        solvent["inserted_count"]
    );
    assert_eq!(
        solvent["density"]["placed_count"],
        solvent["inserted_count"]
    );
    assert_eq!(solvent["density"]["initial_candidate_count"], 8);
    assert_eq!(solvent["density"]["grid_squeeze_required"], true);
    assert!(
        solvent["density"]["candidate_to_target_ratio"]
            .as_f64()
            .unwrap()
            >= 1.0
    );
    assert_eq!(
        read_gro_residue_positions(&gro, "W").len() as u64,
        solvent["inserted_count"].as_u64().unwrap()
    );
}

#[test]
fn seeded_inserted_flood_placement_is_reproducible_and_seed_sensitive() {
    let temp = tempfile::tempdir().unwrap();
    let solute_path = temp.path().join("solute.pdb");
    std::fs::write(
        &solute_path,
        "ATOM      1  BB  SOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
    )
    .unwrap();
    let run = |seed: u64, name: &str| {
        let gro = temp.path().join(format!("{name}.gro"));
        let request = json!({
            "schema_version": BUILD_SCHEMA_VERSION,
            "mode": "membrane",
            "system": {
                "box_size_angstrom": [80.0, 80.0, 80.0],
                "placement": {"mode": "seeded", "random_seed": seed}
            },
            "membranes": [],
            "solutes": [{
                "name": "SOL",
                "count": 5,
                "coordinates": solute_path,
                "format": "pdb",
                "net_charge_e": 0.0,
                "placement": {"center_angstrom": [0.0, 0.0, 0.0], "center_method": "cog"}
            }],
            "environment": {
                "ions": {"neutralize": false},
                "solvent": {"enabled": false}
            },
            "outputs": {
                "coordinates": gro,
                "manifest": temp.path().join(format!("{name}.json"))
            }
        });
        let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
        assert_eq!(code, 0, "{value}");
        assert_eq!(
            value["placement"]["inserted_flood"]["kick_attempt_count"],
            0
        );
        read_gro_residue_positions(&gro, "SOL")
    };

    let first_positions = run(21, "first");
    let second_positions = run(21, "second");
    let third_positions = run(22, "third");
    assert_eq!(first_positions, second_positions);
    assert_ne!(first_positions, third_positions);
}

#[test]
fn seeded_random_inserted_candidate_source_is_reproducible_and_seed_sensitive() {
    let temp = tempfile::tempdir().unwrap();
    let solute_path = temp.path().join("solute.pdb");
    std::fs::write(
        &solute_path,
        "ATOM      1  BB  SOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
    )
    .unwrap();
    let run = |seed: u64, name: &str| {
        let gro = temp.path().join(format!("{name}.gro"));
        let request = json!({
            "schema_version": BUILD_SCHEMA_VERSION,
            "mode": "membrane",
            "system": {
                "box_size_angstrom": [80.0, 80.0, 80.0],
                "placement": {
                    "mode": "seeded",
                    "random_seed": seed,
                    "candidate_source": "random"
                }
            },
            "membranes": [],
            "solutes": [{
                "name": "SOL",
                "count": 6,
                "coordinates": solute_path,
                "format": "pdb",
                "net_charge_e": 0.0,
                "placement": {"center_angstrom": [0.0, 0.0, 0.0], "center_method": "cog"}
            }],
            "environment": {
                "ions": {"neutralize": false},
                "solvent": {"enabled": false}
            },
            "outputs": {
                "coordinates": gro,
                "manifest": temp.path().join(format!("{name}.json"))
            }
        });
        let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
        assert_eq!(code, 0, "{value}");
        assert_eq!(
            value["placement"]["inserted_flood"]["grid_squeeze_pass_count"],
            0
        );
        read_gro_residue_positions(&gro, "SOL")
    };

    let first_positions = run(201, "first");
    let second_positions = run(201, "second");
    let third_positions = run(202, "third");
    assert_eq!(first_positions, second_positions);
    assert_ne!(first_positions, third_positions);
    for i in 0..first_positions.len() {
        for j in (i + 1)..first_positions.len() {
            assert!(squared_distance3(first_positions[i], first_positions[j]) >= 4.0);
        }
    }
}

#[test]
fn deterministic_inserted_flood_reports_grid_squeeze_for_dense_counts() {
    let temp = tempfile::tempdir().unwrap();
    let solute_path = temp.path().join("solute.pdb");
    let gro_path = temp.path().join("dense.gro");
    std::fs::write(
        &solute_path,
        "ATOM      1  BB  SOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [10.0, 10.0, 10.0]},
        "membranes": [],
        "solutes": [{
            "name": "SOL",
            "count": 9,
            "coordinates": solute_path,
            "format": "pdb",
            "net_charge_e": 0.0,
            "placement": {"center_angstrom": [0.0, 0.0, 0.0], "center_method": "cog"}
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
    let flood = &value["placement"]["inserted_flood"];
    assert!(flood["grid_squeeze_pass_count"].as_u64().unwrap() > 0);
    assert!(flood["squeezed_candidate_count"].as_u64().unwrap() > 0);
    assert!(
        flood["min_spacing_angstrom"].as_f64().unwrap()
            < (2.0 + default_solvation_bead_radius() * 2.0) as f64
    );
    assert_eq!(flood["kick_attempt_count"], 0);
    assert_eq!(flood["density"]["target_count"], 9);
    assert_eq!(flood["density"]["placed_count"], 9);
    assert_eq!(flood["density"]["grid_squeeze_required"], true);
    assert!(
        flood["density"]["candidate_to_target_ratio"]
            .as_f64()
            .unwrap()
            >= 1.0
    );
    let positions = read_gro_residue_positions(&gro_path, "SOL");
    assert_eq!(positions.len(), 9);
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            assert!(
                squared_distance3(positions[i], positions[j]) >= 4.0 - 1.0e-3,
                "overlapping squeezed copies: {positions:?}"
            );
        }
    }
}

#[test]
fn seeded_inserted_kicks_expand_center_set() {
    let system = BuildSystem {
        force_field: default_force_field(),
        box_type: default_box_type(),
        pbc: default_pbc(),
        box_size_angstrom: [40.0, 40.0, 40.0],
        unit_cell_angstrom: None,
        box_vectors_angstrom: None,
        placement: PlacementOptions {
            mode: "seeded".to_string(),
            random_seed: Some(77),
            ..PlacementOptions::default()
        },
    };
    let component = InsertedComponent {
        name: "SOL".to_string(),
        count: 3,
        ..InsertedComponent::default()
    };
    let occupied = Vec::new();
    let candidates = vec![[0.0, 0.0, 0.0]];
    let mut centers = candidates.clone();
    let (attempts, inserted) = extend_seeded_kicked_inserted_centers(
        &system,
        &component,
        1.0,
        &occupied,
        default_solvation_bead_radius(),
        &candidates,
        &mut centers,
        77,
        0,
    );
    assert!(attempts > 0);
    assert_eq!(inserted, 2);
    assert_eq!(centers.len(), 3);
    for i in 0..centers.len() {
        for j in (i + 1)..centers.len() {
            assert!(squared_distance3(centers[i], centers[j]) >= 4.0);
        }
    }
}

#[test]
fn seeded_random_orientation_requires_seeded_placement_mode() {
    let temp = tempfile::tempdir().unwrap();
    let solute_path = temp.path().join("solute.pdb");
    std::fs::write(
        &solute_path,
        "ATOM      1  A   SOL A   1       1.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  B   SOL A   1      -1.000   0.000   0.000  1.00  0.00           C\nEND\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [],
        "solutes": [{
            "name": "SOL",
            "count": 2,
            "coordinates": solute_path,
            "format": "pdb",
            "net_charge_e": 0.0,
            "placement": {"orientation": "seeded_random"}
        }],
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_ne!(code, 0);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("orientation seeded_random requires system.placement.mode seeded"));
}

#[test]
fn seeded_random_inserted_orientation_is_reproducible_and_seed_sensitive() {
    let temp = tempfile::tempdir().unwrap();
    let solute_path = temp.path().join("solute.pdb");
    std::fs::write(
        &solute_path,
        "ATOM      1  A   SOL A   1       1.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  B   SOL A   1      -1.000   0.000   0.000  1.00  0.00           C\nEND\n",
    )
    .unwrap();
    let run = |seed: u64, name: &str| {
        let gro = temp.path().join(format!("{name}.gro"));
        let request = json!({
            "schema_version": BUILD_SCHEMA_VERSION,
            "mode": "membrane",
            "system": {
                "box_size_angstrom": [80.0, 80.0, 80.0],
                "placement": {"mode": "seeded", "random_seed": seed}
            },
            "membranes": [],
            "solutes": [{
                "name": "SOL",
                "count": 3,
                "coordinates": solute_path,
                "format": "pdb",
                "net_charge_e": 0.0,
                "placement": {
                    "center_angstrom": [0.0, 0.0, 0.0],
                    "center_method": "cog",
                    "orientation": "seeded_random"
                }
            }],
            "environment": {
                "ions": {"neutralize": false},
                "solvent": {"enabled": false}
            },
            "outputs": {
                "coordinates": gro,
                "manifest": temp.path().join(format!("{name}.json"))
            }
        });
        let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
        assert_eq!(code, 0, "{value}");
        read_gro_residue_positions(&gro, "SOL")
    };

    let first_positions = run(31, "first");
    let second_positions = run(31, "second");
    let third_positions = run(32, "third");
    assert_eq!(first_positions, second_positions);
    assert_ne!(first_positions, third_positions);
    let vectors = first_positions
        .chunks_exact(2)
        .map(|chunk| {
            [
                chunk[1][0] - chunk[0][0],
                chunk[1][1] - chunk[0][1],
                chunk[1][2] - chunk[0][2],
            ]
        })
        .collect::<Vec<_>>();
    assert!(vectors.windows(2).any(|pair| pair[0] != pair[1]));
}

#[test]
fn apl_resolves_lipid_count_when_count_is_omitted() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 100.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [
                {"name": "upper", "side": "upper", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]},
                {"name": "lower", "side": "lower", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]}
            ]
        }],
        "outputs": {
            "coordinates": temp.path().join("m.gro"),
            "topology": temp.path().join("t.top"),
            "manifest": temp.path().join("m.json")
        }
    });
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 400);
    assert_eq!(value["summary"]["bead_count"], 4800);
}

#[test]
fn lipid_ratios_use_largest_remainder_fill() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 100.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 50.0,
                "composition": [
                    {"lipid": "POPC", "fraction": 8.0},
                    {"lipid": "POPE", "fraction": 4.0},
                    {"lipid": "CHOL", "fraction": 3.0}
                ]
            }]
        }],
        "outputs": {
            "coordinates": temp.path().join("m.gro"),
            "topology": temp.path().join("t.top"),
            "manifest": temp.path().join("m.json")
        }
    });
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 107);
    assert_eq!(value["summary"]["lipid_counts"]["POPE"], 53);
    assert_eq!(value["summary"]["lipid_counts"]["CHOL"], 40);
}
