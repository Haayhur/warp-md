use super::*;

#[test]
fn request_requires_one_identity_mode() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "missing_identity",
        "output": {"out_dir": "."}
    });
    let (exit_code, value) = validate_request_json(&request.to_string());
    assert_eq!(exit_code, 2);
    assert_eq!(value["valid"], false);
}

#[test]
fn legacy_v1_fields_are_rejected() {
    for request in [
        json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "legacy_tuning",
            "smiles": "CCO",
            "parameter_tuning": {"enabled": false}
        }),
        json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "legacy_trajectory",
            "smiles": "CCO",
            "trajectory": "traj.xtc"
        }),
        json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "legacy_template",
            "source": {"kind": "coordinates_topology", "coordinates": "a.pdb", "topology": "a.pdb"},
            "mapping_template": "template.json"
        }),
    ] {
        let (exit_code, value) = validate_request_json(&request.to_string());
        assert_eq!(exit_code, 2, "{value}");
        assert_eq!(value["valid"], false);
    }
}

#[test]
fn old_agent_schema_version_is_rejected() {
    let request = json!({
        "schema_version": "warp-cg.agent.v0",
        "name": "old_schema",
        "smiles": "CCO"
    });
    let (exit_code, value) = validate_request_json(&request.to_string());
    assert_eq!(exit_code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("warp-cg.agent.v1"));
}

#[test]
fn external_trajectory_optimization_requires_trajectory_source() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "optimization": {
            "enabled": true,
            "source": "external_trajectory",
            "method": "bayesian_optimization"
        }
    });
    let (exit_code, value) = validate_request_json(&request.to_string());
    assert_eq!(exit_code, 2);
    assert_eq!(value["valid"], false);
}

#[test]
fn topology_top_requires_itp_output() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "output": {
            "write_topology_itp": false,
            "write_topology_top": true
        }
    });
    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("write_topology_top requires"));
}

#[test]
fn forcefield_path_source_reports_missing_bundle() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "forcefield": {
            "kind": "martini3",
            "source": "path",
            "path": "missing_forcefield"
        }
    });
    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2, "{value}");
    assert_eq!(value["valid"], false);
    assert_eq!(value["errors"][0]["code"], "warp_cg.forcefield_missing");
}

#[test]
fn bundled_martini3_forcefield_is_materialized_into_topology_bundle() {
    let tmp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene_ff",
        "smiles": "c1ccccc1",
        "forcefield": {
            "kind": "martini3",
            "source": "bundled"
        },
        "output": {
            "out_dir": tmp.path().to_string_lossy().to_string(),
            "write_mapping_json": false,
            "write_cg_pdb": false,
            "write_topology_itp": true,
            "write_topology_top": true,
            "write_bonded_parameter_map": false
        }
    });

    let (exit_code, result) = run_request_json(&request.to_string(), false);

    assert_eq!(exit_code, 0, "{result}");
    let artifact_kinds = result["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|artifact| artifact["kind"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert!(artifact_kinds.contains(&"forcefield_manifest_json"));
    assert!(artifact_kinds.contains(&"forcefield_directory"));
    let top = std::fs::read_to_string(tmp.path().join("benzene_ff_martini.top")).unwrap();
    assert!(top.contains("#include \"forcefields/martini3/martini_v3.0.0.itp\""));
    assert!(tmp
        .path()
        .join("forcefields/martini3/warp_cg_forcefield_manifest.json")
        .exists());
}

#[test]
fn disabling_itp_without_top_field_remains_valid() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "output": {
            "write_topology_itp": false
        }
    });
    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 0);
    assert_eq!(value["valid"], true);
}

#[test]
fn xtb_gfn_must_not_be_empty() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "ethanol",
        "smiles": "CCO",
        "reference_source": {
            "kind": "xtb",
            "xtb": {
                "gfn": "  "
            }
        }
    });
    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("reference_source.xtb.gfn"));
}

#[test]
fn xtb_optimization_requires_xtb_reference_source() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "ethanol",
        "smiles": "CCO",
        "optimization": {
            "enabled": true,
            "source": "xtb",
            "method": "pso"
        }
    });
    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("xtb parameter tuning requires"));
}

#[test]
fn bonded_term_source_fields_are_validated() {
    let tmp = tempfile::tempdir().unwrap();
    let topology_path = tmp.path().join("model.itp");
    std::fs::write(
        &topology_path,
        concat!(
            "[ moleculetype ]\n",
            "BENZ 1\n",
            "[ atoms ]\n",
            "1 TC5 1 BENZ B1 1 0.0\n",
            "2 TC5 1 BENZ B2 1 0.0\n",
        ),
    )
    .unwrap();
    let base = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "reference_source": {
            "kind": "external",
            "bonded_terms": {
                "kind": "gromacs_itp",
                "path": topology_path.to_string_lossy(),
                "molecule_type": "BENZ"
            }
        }
    });
    let mut accepted = base.clone();
    accepted["optimization"] = json!({
        "enabled": false,
        "target_terms": ["constraints", "bonds", "angles", "dihedrals"]
    });
    let (exit_code, value) = validate_request_json(&accepted.to_string());
    assert_eq!(exit_code, 0, "{value}");
    assert_eq!(value["valid"], true);

    for (field, value) in [
        ("kind", json!("other")),
        ("path", json!("  ")),
        ("molecule_type", json!("  ")),
    ] {
        let mut request = base.clone();
        request["reference_source"]["bonded_terms"][field] = value;
        let (exit_code, result) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(result["error"]["message"]
            .as_str()
            .unwrap()
            .contains("reference_source.bonded_terms"));
    }
}

#[test]
fn reference_metric_sources_are_validated() {
    let tmp = tempfile::tempdir().unwrap();
    let metric_path = tmp.path().join("metrics.json");
    std::fs::write(&metric_path, r#"{"metrics":{"gromacs_sasa_mean_nm2":1.0}}"#).unwrap();
    let base = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "reference_source": {
            "kind": "external",
            "metrics": [{
                "kind": "json",
                "path": metric_path.to_string_lossy(),
                "namespace": "gromacs"
            }]
        }
    });

    let (exit_code, value) = validate_request_json(&base.to_string());
    assert_eq!(exit_code, 0, "{value}");
    assert_eq!(value["valid"], true);

    for (field, value) in [("kind", json!("xvg")), ("path", json!("  "))] {
        let mut request = base.clone();
        request["reference_source"]["metrics"][0][field] = value;
        let (exit_code, result) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(result["error"]["message"]
            .as_str()
            .unwrap()
            .contains("reference_source.metrics"));
    }
}

#[test]
fn reference_transform_fields_are_validated() {
    let base = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "reference_source": {
            "kind": "external",
            "transform": {
                "bond_scaling": 1.1,
                "min_bond_length_nm": 0.25,
                "rg_offset_nm": -0.05,
                "specific_bond_lengths_nm": {"B1": 0.47}
            }
        }
    });

    let (exit_code, value) = validate_request_json(&base.to_string());
    assert_eq!(exit_code, 0, "{value}");
    assert_eq!(value["valid"], true);

    for (field, value) in [
        ("bond_scaling", json!(0.0)),
        ("min_bond_length_nm", json!(-1.0)),
    ] {
        let mut request = base.clone();
        request["reference_source"]["transform"][field] = value;
        let (exit_code, result) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(result["error"]["message"]
            .as_str()
            .unwrap()
            .contains("reference_source.transform"));
    }

    let mut request = base;
    request["reference_source"]["transform"]["specific_bond_lengths_nm"] = json!({"B1": 0.0});
    let (exit_code, result) = validate_request_json(&request.to_string());
    assert_eq!(exit_code, 2);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("specific_bond_lengths_nm"));
}

#[test]
fn json_file_objective_evaluator_fields_are_validated() {
    let tmp = tempfile::tempdir().unwrap();
    let target_path = tmp.path().join("targets.json");
    std::fs::write(
        &target_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "bin_config": {
                "bond_bin_width_nm": 0.01,
                "angle_bin_width_deg": 1.0,
                "dihedral_bin_width_deg": 1.0,
                "bonded_max_range_nm": 3.0
            },
            "constraints": [],
            "bonds": [],
            "angles": [],
            "dihedrals": []
        }))
        .unwrap(),
    )
    .unwrap();
    let base = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "runner_reference",
        "smiles": "CC",
        "reference_source": {
            "kind": "precomputed",
            "precomputed": {
                "target_set": target_path.to_string_lossy()
            }
        },
        "optimization": {
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "evaluator": {
                "kind": "json_file",
                "json_file": {
                    "work_dir": "runner_evaluations",
                    "command": {
                        "program": "/bin/true",
                        "args": []
                    }
                }
            }
        }
    });

    let (exit_code, value) = validate_request_json(&base.to_string());
    assert_eq!(exit_code, 0, "{value}");
    assert_eq!(value["valid"], true);

    for (path, value) in [
        (vec!["kind"], json!("other")),
        (vec!["json_file", "work_dir"], json!("  ")),
        (vec!["json_file", "request_filename"], json!("")),
        (vec!["json_file", "result_filename"], json!("")),
        (vec!["json_file", "command", "program"], json!("")),
    ] {
        let mut request = base.clone();
        let mut field = &mut request["optimization"]["evaluator"];
        for segment in &path[..path.len() - 1] {
            field = &mut field[*segment];
        }
        field[path[path.len() - 1]] = value;
        let (exit_code, result) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(result["error"]["message"]
            .as_str()
            .unwrap()
            .contains("optimization.evaluator"));
    }

    let mut missing_config = base;
    missing_config["optimization"]["evaluator"]["json_file"] = Value::Null;
    let (exit_code, result) = validate_request_json(&missing_config.to_string());
    assert_eq!(exit_code, 2);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("optimization.evaluator.json_file"));

    let mut simulation_without_extraction = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "runner_reference",
        "smiles": "CC",
        "reference_source": {
            "kind": "precomputed",
            "precomputed": {
                "target_set": target_path.to_string_lossy()
            }
        },
        "optimization": {
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "fitting_mode": "simulation_fit",
            "evaluator": {
                "kind": "json_file",
                "json_file": {
                    "work_dir": "runner_evaluations",
                    "command": {"program": "/bin/true"}
                }
            }
        }
    });
    let (exit_code, result) = validate_request_json(&simulation_without_extraction.to_string());
    assert_eq!(exit_code, 2);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("candidate_extraction"));

    simulation_without_extraction["optimization"]["metric_scoring"] = json!({"rg_weight": -1.0});
    simulation_without_extraction["optimization"]["fitting_mode"] = json!("external_evaluator");
    let (exit_code, result) = validate_request_json(&simulation_without_extraction.to_string());
    assert_eq!(exit_code, 2);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("metric_scoring.rg_weight"));
}

#[test]
fn martini_openmm_runner_fields_are_validated() {
    let tmp = tempfile::tempdir().unwrap();
    let target_path = tmp.path().join("targets.json");
    std::fs::write(
        &target_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "bin_config": {
                "bond_bin_width_nm": 0.01,
                "angle_bin_width_deg": 1.0,
                "dihedral_bin_width_deg": 1.0,
                "bonded_max_range_nm": 3.0
            },
            "constraints": [],
            "bonds": [],
            "angles": [],
            "dihedrals": []
        }))
        .unwrap(),
    )
    .unwrap();
    let base = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "martini_runner_reference",
        "smiles": "CC",
        "reference_source": {
            "kind": "precomputed",
            "precomputed": {
                "target_set": target_path.to_string_lossy()
            }
        },
        "optimization": {
            "enabled": true,
            "source": "external_trajectory",
            "method": "bo",
            "fitting_mode": "external_evaluator",
            "runner": {
                "kind": "martini_openmm",
                "work_dir": "martini_evaluations",
                "gro": "system.gro",
                "top": "system.top",
                "template_dir": "candidate_template",
                "replacements": [
                    {
                        "path": "molecule.itp",
                        "parameter": "bond.group_1_length_nm",
                        "format": ".5f"
                    }
                ],
                "protocol": {
                    "dry_run": true,
                    "eq_ns": 0.0,
                    "prod_ns": 0.0,
                    "precision": "mixed",
                    "report_interval_steps": 1000,
                    "trajectory_format": "xtc"
                }
            }
        }
    });

    let (exit_code, value) = validate_request_json(&base.to_string());
    assert_eq!(exit_code, 0, "{value}");
    assert_eq!(value["valid"], true);

    for (pointer, value) in [
        ("/kind", json!("gromacs")),
        ("/gro", json!("")),
        ("/work_dir", json!(" ")),
        ("/protocol/trajectory_format", json!("trr")),
        ("/protocol/precision", json!("quad")),
        ("/protocol/report_interval_steps", json!(0)),
        ("/replacements/0/parameter", json!("")),
    ] {
        let mut request = base.clone();
        *request["optimization"]["runner"]
            .pointer_mut(pointer)
            .unwrap() = value;
        let (exit_code, result) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2, "{result}");
        assert!(result["error"]["message"]
            .as_str()
            .unwrap()
            .contains("optimization.runner"));
    }

    let mut ambiguous = base.clone();
    ambiguous["optimization"]["evaluator"] = json!({
        "kind": "json_file",
        "json_file": {
            "work_dir": "runner_evaluations",
            "command": {"program": "/bin/true"}
        }
    });
    let (exit_code, result) = validate_request_json(&ambiguous.to_string());
    assert_eq!(exit_code, 2);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("either evaluator or runner"));

    let mut simulation_without_extraction = base.clone();
    simulation_without_extraction["optimization"]["fitting_mode"] = json!("simulation_fit");
    let (exit_code, result) = validate_request_json(&simulation_without_extraction.to_string());
    assert_eq!(exit_code, 2);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("runner.candidate_extraction"));

    let mut simulation_with_extraction = simulation_without_extraction;
    simulation_with_extraction["optimization"]["runner"]["candidate_extraction"] = json!({
        "mapping": {
            "bead_names": ["B0", "B1"],
            "atom_indices": [[0], [1]]
        },
        "connections": [[0, 1]],
        "format": "xtc"
    });
    let (exit_code, result) = validate_request_json(&simulation_with_extraction.to_string());
    assert_eq!(exit_code, 0, "{result}");

    let mut simulation_without_trajectory = simulation_with_extraction;
    simulation_without_trajectory["optimization"]["runner"]["protocol"]["trajectory_format"] =
        json!("none");
    let (exit_code, result) = validate_request_json(&simulation_without_trajectory.to_string());
    assert_eq!(exit_code, 2);
    assert!(result["error"]["message"]
        .as_str()
        .unwrap()
        .contains("trajectory_format"));
}

#[test]
fn target_selection_and_atom_indices_are_mutually_exclusive() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "trajectory_source": {
            "path": "traj.xtc",
            "topology": "topology.pdb",
            "target_selection": "resname BENZ",
            "atom_indices": [0, 1, 2, 3, 4, 5]
        }
    });
    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("target_selection or atom_indices"));
}

#[test]
fn tuning_counts_must_be_positive() {
    let base = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "trajectory_source": {
            "kind": "external",
            "path": "traj.xtc"
        }
    });
    for tuning in [
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "bayesian_optimization",
            "max_evaluations": 0
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "swarm_size": 0
        }),
    ] {
        let mut request = base.clone();
        request["optimization"] = tuning;
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert_eq!(value["valid"], false);
    }
}

#[test]
fn pso_advanced_options_validate_method_and_positive_thresholds() {
    let base = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "trajectory_source": {
            "kind": "external",
            "path": "traj.xtc"
        }
    });
    for tuning in [
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "bayesian_optimization",
            "pso": {"fuzzy_self_tuning": true}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "pso": {"reboot_after_local_stall_iterations": 0}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "pso": {"restart_strategy": "elite"}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "pso": {"max_iterations_without_global_best": 0}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "pso": {"checkpoint_interval_evaluations": 0}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "pso": {"checkpoint_path": "  "}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "pso": {"resume_from_checkpoint": true}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "pso": {"discrete_probability_dilation_alpha": 0.0}
        }),
    ] {
        let mut request = base.clone();
        request["optimization"] = tuning;
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert_eq!(value["valid"], false);
    }
}

#[test]
fn bo_alias_and_options_validate_method_and_positive_thresholds() {
    let tmp = tempfile::tempdir().unwrap();
    let target_path = tmp.path().join("targets.json");
    std::fs::write(
        &target_path,
        serde_json::to_vec_pretty(&json!({
            "version": 1,
            "bin_config": {
                "bond_bin_width_nm": 0.01,
                "angle_bin_width_deg": 1.0,
                "dihedral_bin_width_deg": 1.0,
                "bonded_max_range_nm": 3.0
            },
            "constraints": [],
            "bonds": [],
            "angles": [],
            "dihedrals": []
        }))
        .unwrap(),
    )
    .unwrap();
    let base = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "reference_source": {
            "kind": "precomputed",
            "precomputed": {
                "target_set": target_path.to_string_lossy()
            }
        }
    });
    let mut valid = base.clone();
    valid["optimization"] = json!({
        "enabled": true,
        "source": "external_trajectory",
        "method": "bo",
        "evaluator": {
            "kind": "json_file",
            "json_file": {
                "work_dir": "bo_evaluations",
                "command": {
                    "program": "/bin/true",
                    "args": []
                }
            }
        },
        "bo": {
            "n_startup_trials": 2,
            "n_candidates": 16,
            "noise_variance": 1.0e-6,
            "checkpoint_path": "bo_checkpoint.json",
            "checkpoint_interval_evaluations": 1
        }
    });
    let (exit_code, value) = validate_request_json(&valid.to_string());
    assert_eq!(exit_code, 0, "{value}");

    for tuning in [
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "pso",
            "bo": {"n_startup_trials": 2}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "bayesian_optimization",
            "bo": {"n_startup_trials": 0}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "bo",
            "bo": {"checkpoint_interval_evaluations": 0}
        }),
        json!({
            "enabled": true,
            "source": "external_trajectory",
            "method": "bo",
            "bo": {"resume_from_checkpoint": true}
        }),
    ] {
        let mut request = base.clone();
        request["optimization"] = tuning;
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2, "{value}");
        assert_eq!(value["valid"], false);
    }
}

#[test]
fn trajectory_source_kind_must_be_external() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "trajectory_source": {
            "kind": "xtb",
            "path": "traj.xtc"
        }
    });
    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("trajectory_source.kind"));
}

#[test]
fn trajectory_source_string_fields_must_not_be_empty() {
    for field in ["topology", "target_selection", "environment_selection"] {
        let mut request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "kind": "external",
                "path": "traj.xtc"
            }
        });
        request["trajectory_source"][field] = json!("  ");
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"].as_str().unwrap().contains(field));
    }
}

#[test]
fn trajectory_source_sasa_fields_are_validated() {
    for (field, value) in [
        ("probe_radius_nm", json!(-0.1)),
        ("n_sphere_points", json!(7)),
        ("fallback_radius_nm", json!(0.0)),
        ("radii_nm", json!([0.2, -0.1])),
    ] {
        let mut request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "trajectory_source": {
                "kind": "external",
                "path": "traj.xtc",
                "sasa": {}
            }
        });
        request["trajectory_source"]["sasa"][field] = value;
        let (exit_code, value) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(value["error"]["message"].as_str().unwrap().contains(field));
    }
}

#[test]
fn top_level_topology_must_not_be_empty() {
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "benzene",
        "smiles": "c1ccccc1",
        "topology": "  "
    });
    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2);
    assert!(value["error"]["message"]
        .as_str()
        .unwrap()
        .contains("topology must not be empty"));
}

#[test]
fn output_paths_must_not_be_empty() {
    for (field, value) in [("out_dir", json!("  ")), ("mapped_trajectory", json!("  "))] {
        let mut request = json!({
            "schema_version": AGENT_SCHEMA_VERSION,
            "name": "benzene",
            "smiles": "c1ccccc1",
            "output": {
                "out_dir": "."
            }
        });
        request["output"][field] = value;
        let (exit_code, result) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(result["error"]["message"].as_str().unwrap().contains(field));
    }
}

#[test]
fn martini_itp_contains_atoms_and_bonds() {
    let mol = Molecule::from_smiles("c1ccccc1").unwrap();
    let mapping = map_molecule(&mol);
    let itp = render_martini_itp("benzene", &mapping, &[], &[], &[], None, None);

    assert!(itp.contains("[ moleculetype ]"));
    assert!(itp.contains("[ atoms ]"));
    assert!(itp.contains("[ bonds ]"));
    assert!(itp.contains("BENZENE"));
}

#[test]
fn martini_itp_uses_mapping_formal_charges() {
    let mapping = MappingResult {
        bead_names: vec!["Qd".to_string(), "Qa".to_string()],
        atom_groups: vec![vec![0], vec![1]],
        connections: vec![(0, 1)],
        bead_features: vec![vec!["charged".to_string()], vec!["charged".to_string()]],
        bead_formal_charges: vec![1, -1],
    };
    let itp = render_martini_itp("salt", &mapping, &[], &[], &[], None, None);
    let atom_rows = itp
        .lines()
        .filter(|line| {
            line.split_whitespace()
                .next()
                .is_some_and(|col| col.parse::<usize>().is_ok())
        })
        .take(2)
        .map(|line| {
            line.split_whitespace()
                .map(str::to_string)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    assert_eq!(atom_rows[0][1], "Qd");
    assert_eq!(atom_rows[0][6], "1.000");
    assert_eq!(atom_rows[1][1], "Qa");
    assert_eq!(atom_rows[1][6], "-1.000");
}

#[test]
fn martini_itp_contains_angle_and_dihedral_sections() {
    let mapping = MappingResult {
        bead_names: vec![
            "C1".to_string(),
            "C1".to_string(),
            "C1".to_string(),
            "C1".to_string(),
        ],
        atom_groups: vec![vec![0], vec![1], vec![2], vec![3]],
        connections: vec![(0, 1), (1, 2), (2, 3)],
        bead_features: vec![
            vec!["hydrocarbon".to_string()],
            vec!["hydrocarbon".to_string()],
            vec!["hydrocarbon".to_string()],
            vec!["hydrocarbon".to_string()],
        ],
        bead_formal_charges: vec![0, 0, 0, 0],
    };
    let itp = render_martini_itp(
        "chain",
        &mapping,
        &[],
        &[AngleStats {
            bead_i: 0,
            bead_j: 1,
            bead_k: 2,
            mean_deg: 120.0,
            std_deg: 5.0,
            samples: 4,
        }],
        &[DihedralStats {
            bead_i: 0,
            bead_j: 1,
            bead_k: 2,
            bead_l: 3,
            mean_deg: 180.0,
            std_deg: 10.0,
            samples: 4,
        }],
        None,
        None,
    );

    assert!(itp.contains("[ angles ]"));
    assert!(itp.contains("[ dihedrals ]"));
}

#[test]
fn martini_itp_applies_grouped_reference_target_parameters_to_all_members() {
    let mapping = MappingResult {
        bead_names: vec![
            "C1".to_string(),
            "C1".to_string(),
            "C1".to_string(),
            "C1".to_string(),
        ],
        atom_groups: vec![vec![0], vec![1], vec![2], vec![3]],
        connections: vec![(0, 1), (2, 3)],
        bead_features: vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        bead_formal_charges: vec![0, 0, 0, 0],
    };
    let targets = crate::reference::ReferenceTargetSet {
        version: 1,
        bin_config: crate::reference::ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![crate::reference::ReferenceDistributionTarget::from_samples(
            crate::reference::ReferenceTermKind::Bond,
            Some("bond.middle.M0_AR1__M0_SO2".to_string()),
            vec![0, 1],
            vec![vec![0, 1], vec![2, 3]],
            &[4.7, 4.9],
            "angstrom",
            false,
            0.0,
            30.0,
            0.1,
        )],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };
    let report = OptimizationReport {
        status: "ok".to_string(),
        method: "direct_statistics".to_string(),
        objective: "bonded_parameter_parity".to_string(),
        objective_value: 0.0,
        converged: true,
        bounds: Vec::new(),
        best_parameters: vec![
            (
                "bond.middle.M0_AR1__M0_SO2_length_angstrom".to_string(),
                5.0,
            ),
            ("bond.middle.M0_AR1__M0_SO2_force".to_string(), 777.0),
        ],
        evaluations: Vec::new(),
        message: String::new(),
    };

    let itp = render_martini_itp(
        "chain",
        &mapping,
        &[],
        &[],
        &[],
        Some(&targets),
        Some(&report),
    );

    assert!(itp.contains("; class: bond.middle.M0_AR1__M0_SO2"));
    assert!(itp.contains("    1     2     1    0.50000    777.000"));
    assert!(itp.contains("    3     4     1    0.50000    777.000"));
    assert!(!itp.contains("bond_0_1_length_angstrom"));
}

#[test]
fn martini_itp_preserves_grouped_reference_target_nm_lengths() {
    let mapping = MappingResult {
        bead_names: vec![
            "C1".to_string(),
            "C1".to_string(),
            "C1".to_string(),
            "C1".to_string(),
        ],
        atom_groups: vec![vec![0], vec![1], vec![2], vec![3]],
        connections: vec![(0, 1), (2, 3)],
        bead_features: vec![Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        bead_formal_charges: vec![0, 0, 0, 0],
    };
    let targets = crate::reference::ReferenceTargetSet {
        version: 1,
        bin_config: crate::reference::ReferenceBinConfig::default(),
        constraints: vec![crate::reference::ReferenceDistributionTarget::from_samples(
            crate::reference::ReferenceTermKind::Constraint,
            Some("constraint group 1".to_string()),
            vec![0, 1],
            vec![vec![0, 1]],
            &[0.5],
            "nm",
            false,
            0.0,
            3.0,
            0.01,
        )],
        bonds: vec![crate::reference::ReferenceDistributionTarget::from_samples(
            crate::reference::ReferenceTermKind::Bond,
            Some("bond group 1".to_string()),
            vec![2, 3],
            vec![vec![2, 3]],
            &[0.5],
            "nm",
            false,
            0.0,
            3.0,
            0.01,
        )],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };

    let itp = render_martini_itp("chain", &mapping, &[], &[], &[], Some(&targets), None);

    assert!(itp.contains("    1     2     1    0.50000"));
    assert!(itp.contains("    3     4     1    0.50000   2500.000"));
}

#[test]
fn martini_itp_converts_grouped_trajectory_bond_lengths_to_nm() {
    let mapping = MappingResult {
        bead_names: vec!["C1".to_string(), "C1".to_string()],
        atom_groups: vec![vec![0], vec![1]],
        connections: vec![(0, 1)],
        bead_features: vec![Vec::new(), Vec::new()],
        bead_formal_charges: vec![0, 0],
    };
    let values = crate::parameters::BondedValueSeries {
        bonds: vec![crate::parameters::BondValueSeries {
            label: Some("bond.middle.M0_AR1__M0_SO2".to_string()),
            members: vec![[0, 1]],
            bead_i: 0,
            bead_j: 1,
            values: vec![1.22287],
        }],
        ..crate::parameters::BondedValueSeries::default()
    };
    let targets = crate::reference::ReferenceTargetSet::from_values(
        &values,
        crate::reference::ReferenceBinConfig::default(),
    );

    let itp = render_martini_itp("chain", &mapping, &[], &[], &[], Some(&targets), None);

    assert!(itp.contains("; class: bond.middle.M0_AR1__M0_SO2"));
    assert!(itp.contains("    1     2     1    0.12229"));
    assert!(!itp.contains("    1     2     1    1.22287"));
}

#[test]
fn martini_top_includes_generated_itp_and_molecule_count() {
    let top = render_martini_top("benzene", "benzene_martini.itp", &[]);

    assert!(top.contains("#include \"benzene_martini.itp\""));
    assert!(top.contains("[ system ]"));
    assert!(top.contains("[ molecules ]"));
    assert!(top.contains("BENZENE"));
}
