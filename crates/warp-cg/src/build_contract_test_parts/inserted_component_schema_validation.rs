use super::*;

#[test]
fn inserted_component_definition_file_schema_is_validated() {
    let temp = tempfile::tempdir().unwrap();
    let definition = temp.path().join("wrong_schema.json");
    std::fs::write(
        &definition,
        serde_json::to_string_pretty(&json!({
            "schema_version": "warp-cg.molecule_definition.v0",
            "beads": [{"name": "A"}]
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
        .contains(MOLECULE_DEFINITION_SCHEMA_VERSION));
}

#[test]
fn inserted_component_definition_file_bonds_are_validated() {
    let temp = tempfile::tempdir().unwrap();
    let definition = temp.path().join("bad_bond.json");
    std::fs::write(
        &definition,
        serde_json::to_string_pretty(&json!({
            "schema_version": MOLECULE_DEFINITION_SCHEMA_VERSION,
            "beads": [{"name": "A"}],
            "bonds": [{"bead_indices": [0, 1]}]
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
        .contains("bead_indices out of range"));
}

#[test]
fn inserted_component_definition_file_angles_are_validated() {
    let temp = tempfile::tempdir().unwrap();
    let definition = temp.path().join("bad_angle.json");
    std::fs::write(
        &definition,
        serde_json::to_string_pretty(&json!({
            "schema_version": MOLECULE_DEFINITION_SCHEMA_VERSION,
            "beads": [{"name": "A"}, {"name": "B"}],
            "angles": [{"bead_indices": [0, 1, 2]}]
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
        .contains("angles[0].bead_indices out of range"));
}

#[test]
fn inserted_component_definition_file_dihedrals_are_validated() {
    let temp = tempfile::tempdir().unwrap();
    let definition = temp.path().join("bad_dihedral.json");
    std::fs::write(
        &definition,
        serde_json::to_string_pretty(&json!({
            "schema_version": MOLECULE_DEFINITION_SCHEMA_VERSION,
            "beads": [{"name": "A"}, {"name": "B"}, {"name": "C"}],
            "dihedrals": [{"bead_indices": [0, 1, 2, 3]}]
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
        .contains("dihedrals[0].bead_indices out of range"));
}

#[test]
fn inserted_component_inline_bead_charge_mismatch_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [],
        "solutes": [{
            "name": "CUST",
            "net_charge_e": 0.0,
            "beads": [
                {"name": "A", "charge_e": 1.0}
            ]
        }],
        "environment": {"ions": {"neutralize": false}, "solvent": {"enabled": false}},
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does not match explicit bead charge sum"));
}

#[test]
fn inserted_component_inline_beads_validate_names_and_numbers() {
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [],
        "solutes": [{
            "name": "CUST",
            "beads": [{"name": ""}]
        }],
        "environment": {"ions": {"neutralize": false}, "solvent": {"enabled": false}},
        "outputs": {"manifest": "unused.json"}
    });

    let (code, value) = validate_request_json(&serde_json::to_string(&request).unwrap());
    assert_eq!(code, 2);
    assert!(value["errors"][0]["message"]
        .as_str()
        .unwrap()
        .contains("beads[0].name must not be empty"));
}

#[test]
fn flooded_solutes_do_not_reduce_solvent_free_volume() {
    let temp = tempfile::tempdir().unwrap();
    let solute_path = temp.path().join("solute.pdb");
    std::fs::write(
        &solute_path,
        "ATOM      1  BB  SOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
    )
    .unwrap();
    let base = json!({
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
        "environment": {
            "ions": {"neutralize": true, "salt_molarity_mol_l": 0.0},
            "solvent": {"enabled": true}
        },
        "outputs": {"manifest": temp.path().join("base.json")}
    });
    let with_solute = json!({
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
        "solutes": [{
            "name": "SOL",
            "count": 4,
            "coordinates": solute_path,
            "format": "pdb",
            "net_charge_e": 0.0
        }],
        "environment": {
            "ions": {"neutralize": true, "salt_molarity_mol_l": 0.0},
            "solvent": {"enabled": true}
        },
        "outputs": {"manifest": temp.path().join("with_solute.json")}
    });
    let (base_code, base_value) = run_request_json(&serde_json::to_string(&base).unwrap(), false);
    let (solute_code, solute_value) =
        run_request_json(&serde_json::to_string(&with_solute).unwrap(), false);
    assert_eq!(base_code, 0, "{base_value}");
    assert_eq!(solute_code, 0, "{solute_value}");
    assert_eq!(
        base_value["placement"]["solvent"]["free_volume_nm3"],
        solute_value["placement"]["solvent"]["free_volume_nm3"]
    );
    assert_eq!(solute_value["summary"]["inserted_counts"]["SOL"], 4);
    assert_eq!(
        solute_value["summary"]["solvent_counts"]["W"],
        base_value["summary"]["solvent_counts"]["W"]
    );
}

#[test]
fn repeated_coordinate_solutes_are_flood_placed_without_exact_overlap() {
    let temp = tempfile::tempdir().unwrap();
    let solute_path = temp.path().join("solute.pdb");
    let gro = temp.path().join("solute.gro");
    std::fs::write(
        &solute_path,
        "ATOM      1  BB  SOL A   1       0.000   0.000   0.000  1.00  0.00           C\nEND\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [],
        "solutes": [{
            "name": "SOL",
            "count": 4,
            "coordinates": solute_path,
            "format": "pdb",
            "net_charge_e": 0.0,
            "placement": {"center_angstrom": [0.0, 0.0, 0.0], "center_method": "cog"}
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"coordinates": gro, "manifest": temp.path().join("manifest.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    let positions = read_gro_residue_positions(&gro, "SOL");
    assert_eq!(positions.len(), 4);
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let distance_sq = squared_distance3(positions[i], positions[j]);
            assert!(
                distance_sq > 1.0,
                "overlapping solute copies: {positions:?}"
            );
        }
    }
}

#[test]
fn seeded_placement_requires_random_seed() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [40.0, 40.0, 40.0],
            "placement": {"mode": "seeded"}
        },
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": true, "molarity_mol_l": 0.01}
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_ne!(code, 0);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("system.placement.random_seed is required"));
}

#[test]
fn random_candidate_source_requires_seeded_mode() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [40.0, 40.0, 40.0],
            "placement": {"candidate_source": "random"}
        },
        "membranes": [],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_ne!(code, 0);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("candidate_source random requires system.placement.mode seeded"));
}

#[test]
fn seeded_solvent_placement_is_reproducible_and_seed_sensitive() {
    let temp = tempfile::tempdir().unwrap();
    let run = |seed: u64, name: &str| {
        let gro = temp.path().join(format!("{name}.gro"));
        let request = json!({
            "schema_version": BUILD_SCHEMA_VERSION,
            "mode": "membrane",
            "system": {
                "box_size_angstrom": [50.0, 50.0, 50.0],
                "placement": {"mode": "seeded", "random_seed": seed}
            },
            "membranes": [],
            "environment": {
                "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
                "solvent": {
                    "enabled": true,
                    "molarity_mol_l": 1.0,
                    "grid_spacing_angstrom": 5.0
                }
            },
            "outputs": {
                "coordinates": gro,
                "manifest": temp.path().join(format!("{name}.json"))
            }
        });
        let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
        assert_eq!(code, 0, "{value}");
        assert_eq!(value["placement"]["mode"], "seeded");
        assert_eq!(value["placement"]["random_seed"], seed);
        assert_eq!(
            value["placement"]["solvent"]["algorithm"],
            "free_volume_count_seeded_grid"
        );
        (value, read_gro_residue_positions(&gro, "W"))
    };

    let (_first_value, first_positions) = run(11, "first");
    let (_second_value, second_positions) = run(11, "second");
    let (_third_value, third_positions) = run(12, "third");
    assert_eq!(first_positions, second_positions);
    assert_ne!(first_positions, third_positions);
}

#[test]
fn seeded_random_leaflet_candidate_source_is_reproducible_and_seed_sensitive() {
    let temp = tempfile::tempdir().unwrap();
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
                    "candidate_source": "random",
                    "relaxation": true,
                    "max_steps": 80
                }
            },
            "membranes": [{
                "name": "bilayer",
                "center_z_angstrom": 0.0,
                "leaflets": [{
                    "name": "upper",
                    "side": "upper",
                    "apl_angstrom2": 80.0,
                    "composition": [{"lipid": "POPC", "count": 18}]
                }]
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
        assert_eq!(value["placement"]["mode"], "seeded");
        assert_eq!(value["placement"]["candidate_source"], "random");
        assert_eq!(
            value["placement"]["algorithm"],
            "seeded_random_leaflet_candidates_pair_edge_exclusion_relaxation"
        );
        let metrics = &value["placement"]["leaflet_metrics"][0]["metrics"];
        assert_eq!(metrics["relaxation_enabled"], true);
        read_gro_residue_positions(&gro, "POPC")
    };

    let first_positions = run(301, "first");
    let second_positions = run(301, "second");
    let third_positions = run(302, "third");
    assert_eq!(first_positions, second_positions);
    assert_ne!(first_positions, third_positions);
}

#[test]
fn seeded_random_leaflet_lipid_sequence_interleaves_composition() {
    let system = BuildSystem {
        force_field: default_force_field(),
        box_type: default_box_type(),
        pbc: default_pbc(),
        box_size_angstrom: [80.0, 80.0, 80.0],
        unit_cell_angstrom: None,
        box_vectors_angstrom: None,
        placement: PlacementOptions {
            mode: "seeded".to_string(),
            candidate_source: "random".to_string(),
            random_seed: Some(401),
            ..PlacementOptions::default()
        },
    };
    let membrane = MembraneRequest {
        name: "mix".to_string(),
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
        apl_angstrom2: Some(60.0),
        exclusions: Vec::new(),
        regions: Vec::new(),
        composition: Vec::new(),
    };
    let lipids = vec![
        ResolvedLipid {
            name: "A".to_string(),
            count: 8,
            charge_e: 0.0,
            radius_angstrom: 3.0,
            beads: Vec::new(),
            template_source: "test".to_string(),
            charge_source: "test".to_string(),
        },
        ResolvedLipid {
            name: "B".to_string(),
            count: 8,
            charge_e: 0.0,
            radius_angstrom: 3.0,
            beads: Vec::new(),
            template_source: "test".to_string(),
            charge_source: "test".to_string(),
        },
    ];

    let deterministic = leaflet_lipid_sequence(
        &BuildSystem {
            placement: PlacementOptions::default(),
            ..system.clone()
        },
        &membrane,
        &leaflet,
        &lipids,
    );
    assert_eq!(
        deterministic,
        vec![0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    );

    let mixed = leaflet_lipid_sequence(&system, &membrane, &leaflet, &lipids);
    assert_eq!(mixed.iter().filter(|&&idx| idx == 0).count(), 8);
    assert_eq!(mixed.iter().filter(|&&idx| idx == 1).count(), 8);
    assert_ne!(mixed, deterministic);
    assert!(mixed.windows(2).filter(|pair| pair[0] != pair[1]).count() > 2);
}

#[test]
fn seeded_random_leaflet_candidates_are_spatially_partitioned() {
    let membrane = MembraneRequest {
        name: "partitioned".to_string(),
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
        apl_angstrom2: Some(60.0),
        exclusions: Vec::new(),
        regions: Vec::new(),
        composition: Vec::new(),
    };
    let bounds = LayoutBounds {
        xmin: -50.0,
        xmax: 50.0,
        ymin: -25.0,
        ymax: 25.0,
    };
    let points = random_leaflet_grid(
        501,
        &[1.0; 20],
        bounds,
        None,
        LayoutPeriodicity::default(),
        &membrane,
        &leaflet,
        &[],
    )
    .unwrap();
    let points = points.unwrap();
    assert_eq!(points.len(), 20);
    let left_count = points.iter().filter(|point| point.x < 0.0).count();
    let right_count = points.iter().filter(|point| point.x >= 0.0).count();
    assert_eq!(left_count, 10);
    assert_eq!(right_count, 10);
}
