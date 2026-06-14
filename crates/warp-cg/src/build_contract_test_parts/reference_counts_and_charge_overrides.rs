use super::*;

#[test]
fn asymmetric_leaflet_apl_and_ratios_match_reference_counts() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 150.0]},
        "membranes": [{
            "name": "bilayer",
            "center_z_angstrom": 20.0,
            "leaflets": [
                {
                    "name": "upper",
                    "side": "upper",
                    "apl_angstrom2": 55.0,
                    "composition": [
                        {"lipid": "POPC", "fraction": 8.0},
                        {"lipid": "POPE", "fraction": 6.0},
                        {"lipid": "CHOL", "fraction": 1.0}
                    ]
                },
                {
                    "name": "lower",
                    "side": "lower",
                    "apl_angstrom2": 45.0,
                    "composition": [
                        {"lipid": "POPC", "fraction": 9.0},
                        {"lipid": "POPE", "fraction": 4.0},
                        {"lipid": "CHOL", "fraction": 5.0}
                    ]
                }
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
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 208);
    assert_eq!(value["summary"]["lipid_counts"]["POPE"], 122);
    assert_eq!(value["summary"]["lipid_counts"]["CHOL"], 74);
}

#[test]
fn monolayer_solvent_slab_matches_reference_counts() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 160.0]},
        "membranes": [
            {
                "name": "mono_lower",
                "center_z_angstrom": 50.0,
                "leaflets": [{
                    "name": "lower",
                    "side": "lower",
                    "apl_angstrom2": 60.0,
                    "composition": [{"lipid": "POPC"}]
                }]
            },
            {
                "name": "mono_upper",
                "center_z_angstrom": -50.0,
                "leaflets": [{
                    "name": "upper",
                    "side": "upper",
                    "apl_angstrom2": 60.0,
                    "composition": [{"lipid": "POPC"}]
                }]
            }
        ],
        "environment": {
            "ions": {"salt_molarity_mol_l": 0.15, "cation": "Na+", "anion": "Cl-"},
            "solvent": {"enabled": true, "box_size_angstrom": [100.0, 100.0, 100.0]}
        },
        "outputs": {
            "coordinates": temp.path().join("m.gro"),
            "topology": temp.path().join("t.top"),
            "manifest": temp.path().join("m.json")
        }
    });
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 334);
    assert_eq!(value["summary"]["solvent_counts"]["W"], 5781);
    assert_eq!(value["summary"]["solvent_counts"]["NA"], 63);
    assert_eq!(value["summary"]["solvent_counts"]["CL"], 63);
    assert_eq!(value["summary"]["bead_count"], 9915);
}

#[test]
fn solvent_zones_support_phase_separated_salt_counts() {
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
        "environment": {
            "ions": {"neutralize": true, "salt_molarity_mol_l": 0.0, "cation": "Na+", "anion": "Cl-"},
            "solvent": {
                "enabled": true,
                "zones": [
                    {
                        "name": "right",
                        "center_angstrom": [20.0, 0.0, 0.0],
                        "box_size_angstrom": [40.0, 80.0, 80.0],
                        "salt_molarity_mol_l": 0.0
                    },
                    {
                        "name": "left",
                        "center_angstrom": [-20.0, 0.0, 0.0],
                        "box_size_angstrom": [40.0, 80.0, 80.0],
                        "salt_molarity_mol_l": 0.6
                    }
                ]
            }
        },
        "outputs": {
            "coordinates": temp.path().join("m.gro"),
            "topology": temp.path().join("t.top"),
            "manifest": temp.path().join("m.json")
        }
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(
        value["placement"]["solvent"]["algorithm"],
        "multi_zone_free_volume_count_deterministic_grid"
    );
    assert!(value["summary"]["solvent_counts"]["W"].as_u64().unwrap() > 0);
    assert!(value["summary"]["solvent_counts"]["NA"].as_u64().unwrap() > 0);
    assert_eq!(
        value["summary"]["solvent_counts"]["NA"],
        value["summary"]["solvent_counts"]["CL"]
    );
}

#[test]
fn request_beads_override_template_registry() {
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
                "composition": [{
                    "lipid": "POPC",
                    "count": 2,
                    "charge_e": -2.0,
                    "beads": [
                        {"name": "A", "offset_angstrom": [0.0, 0.0, 0.0], "charge_e": -1.0},
                        {"name": "B", "offset_angstrom": [1.0, 0.0, 0.0], "charge_e": -1.0}
                    ]
                }]
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
    assert_eq!(value["summary"]["bead_count"], 4);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], -4.0);
    assert_eq!(
        value["charge"]["component_charges"][0]["source"],
        "request_lipid.charge_e"
    );
    assert_eq!(
        value["charge"]["component_charges"][0]["per_instance_bead_charge_sum_e"],
        -2.0
    );
    let delta = value["charge"]["component_charges"][0]["charge_balance_delta_e"]
        .as_f64()
        .unwrap();
    assert!(delta.abs() < 1.0e-5, "{value}");
    let topology = std::fs::read_to_string(temp.path().join("t.top")).unwrap();
    assert!(topology.contains("POPC"));
    assert!(topology.contains("A"));
    assert!(topology.contains("B"));
    assert!(
        (topology_molecule_charge_sum(&topology, "POPC") + 2.0).abs() < 1.0e-5,
        "{topology}"
    );
}

#[test]
fn template_charge_override_is_spread_to_emitted_beads() {
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
                "composition": [{
                    "lipid": "POPC",
                    "count": 1,
                    "charge_e": -2.0
                }]
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
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], -2.0);
    let bead_charge_sum = value["charge"]["component_charges"][0]["per_instance_bead_charge_sum_e"]
        .as_f64()
        .unwrap();
    assert!((bead_charge_sum + 2.0).abs() < 1.0e-5, "{value}");
    let delta = value["charge"]["component_charges"][0]["charge_balance_delta_e"]
        .as_f64()
        .unwrap();
    assert!(delta.abs() < 1.0e-5, "{value}");
    let topology = std::fs::read_to_string(temp.path().join("t.top")).unwrap();
    assert!(
        (topology_molecule_charge_sum(&topology, "POPC") + 2.0).abs() < 1.0e-2,
        "{topology}"
    );
}

#[test]
fn explicit_bead_charge_sum_can_define_lipid_charge() {
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
                "composition": [{
                    "lipid": "USER",
                    "count": 3,
                    "beads": [
                        {"name": "A", "charge_e": 1.0},
                        {"name": "B", "charge_e": -0.5}
                    ]
                }]
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
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], 1.5);
    assert_eq!(
        value["charge"]["component_charges"][0]["source"],
        "request_lipid.beads.charge_e"
    );
}

#[test]
fn explicit_bead_charge_mismatch_is_rejected() {
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{
                    "lipid": "USER",
                    "count": 1,
                    "charge_e": -1.0,
                    "beads": [
                        {"name": "A", "charge_e": 0.0},
                        {"name": "B", "charge_e": 0.0}
                    ]
                }]
            }]
        }],
        "outputs": {"manifest": "unused.json"}
    });
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("does not match explicit bead charge sum"));
}

#[test]
fn reference_basic_tutorial_solvent_counts_match_free_volume_rule() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [100.0, 100.0, 100.0],
            "placement": {"relaxation": true, "max_steps": 100}
        },
        "membranes": [{
            "name": "bilayer",
            "leaflets": [
                {"name": "upper", "side": "upper", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]},
                {"name": "lower", "side": "lower", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]}
            ]
        }],
        "environment": {
            "ions": {"salt_molarity_mol_l": 0.15, "cation": "Na+", "anion": "Cl-"},
            "solvent": {"enabled": true}
        },
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
    assert_eq!(value["summary"]["solvent_counts"]["W"], 5270);
    assert_eq!(value["summary"]["solvent_counts"]["NA"], 57);
    assert_eq!(value["summary"]["solvent_counts"]["CL"], 57);
    assert_eq!(value["summary"]["bead_count"], 10184);
    assert_eq!(value["placement"]["solvent"]["inserted_count"], 5384);
}

#[test]
fn circular_exclusion_masks_are_honored_by_layout() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [120.0, 120.0, 100.0],
            "placement": {"relaxation": true, "max_steps": 120}
        },
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 60.0,
                "exclusions": [{"name": "protein", "center_angstrom": [0.0, 0.0], "radius_angstrom": 12.0}],
                "composition": [{"lipid": "POPC", "count": 24}]
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
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["exclusion_count"],
        1
    );
    let violation = value["placement"]["leaflet_metrics"][0]["metrics"]
        ["max_exclusion_violation_angstrom"]
        .as_f64()
        .unwrap();
    assert!(violation <= 0.05, "{value}");
}

#[test]
fn protein_coordinates_generate_exclusion_masks() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("protein.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  BB  PRO A   1      -4.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  BB  PRO A   1       4.000   0.000   0.000  1.00  0.00           C\n\
ATOM      3  BB  PRO A   1       0.000   3.000   0.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [120.0, 120.0, 100.0],
            "placement": {"relaxation": true, "max_steps": 120}
        },
        "proteins": [{
            "name": "protein",
            "coordinates": protein_path,
            "format": "pdb",
            "footprint": {}
        }],
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 60.0,
                "composition": [{"lipid": "POPC", "count": 24}]
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
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["exclusion_count"],
        1
    );
    assert_eq!(value["summary"]["protein_count"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["protein"], 1);
    assert_eq!(value["summary"]["bead_count"], 291);
    let violation = value["placement"]["leaflet_metrics"][0]["metrics"]
        ["max_exclusion_violation_angstrom"]
        .as_f64()
        .unwrap();
    assert!(violation <= 0.05, "{value}");
    let topology = std::fs::read_to_string(temp.path().join("t.top")).unwrap();
    assert!(topology.contains("protein"));
}

#[test]
fn protein_coordinates_reduce_apl_derived_lipid_counts() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("protein.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  BB  PRO A   1     -10.000 -10.000   5.000  1.00  0.00           C\n\
ATOM      2  BB  PRO A   1      10.000 -10.000   5.000  1.00  0.00           C\n\
ATOM      3  BB  PRO A   1      10.000  10.000   5.000  1.00  0.00           C\n\
ATOM      4  BB  PRO A   1     -10.000  10.000   5.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 100.0]},
        "proteins": [{
            "name": "protein",
            "coordinates": protein_path,
            "format": "pdb",
            "placement": {"center_method": "none", "center_angstrom": [0.0, 0.0, 0.0]}
        }],
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 60.0,
                "composition": [{"lipid": "POPC"}]
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
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 160);
    assert_eq!(value["summary"]["inserted_counts"]["protein"], 1);
    assert_eq!(value["summary"]["bead_count"], 1924);
}

#[test]
fn protein_footprint_inside_region_hole_does_not_reduce_lipid_counts() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("protein.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  BB  PRO A   1     -10.000 -10.000   5.000  1.00  0.00           C\n\
ATOM      2  BB  PRO A   1      10.000 -10.000   5.000  1.00  0.00           C\n\
ATOM      3  BB  PRO A   1      10.000  10.000   5.000  1.00  0.00           C\n\
ATOM      4  BB  PRO A   1     -10.000  10.000   5.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 100.0]},
        "proteins": [{
            "name": "protein",
            "coordinates": protein_path,
            "format": "pdb",
            "placement": {"center_method": "none", "center_angstrom": [0.0, 0.0, 0.0]}
        }],
        "membranes": [{
            "name": "holed",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 50.0,
                "regions": [{
                    "name": "protein-hole",
                    "role": "hole",
                    "geometry": {"shape": "circle", "center_angstrom": [0.0, 0.0], "radius_angstrom": 30.0}
                }],
                "composition": [{"lipid": "POPC"}]
            }]
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
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 144);
}
