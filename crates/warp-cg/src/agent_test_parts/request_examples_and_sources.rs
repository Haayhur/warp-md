use super::*;

#[test]
fn example_request_validates() {
    let text = serde_json::to_string(&example_request()).unwrap();
    let (exit_code, value) = validate_request_json(&text);
    assert_eq!(exit_code, 0);
    assert_eq!(value["valid"], true);
}

#[test]
fn benzene_mapping_has_three_beads() {
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "benzene".to_string(),
        smiles: Some("c1ccccc1".to_string()),
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: None,
        topology: None,
        trajectory_source: None,
        reference_source: None,
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: tempfile::tempdir()
                .unwrap()
                .path()
                .to_string_lossy()
                .to_string(),
            mapped_trajectory: None,
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };
    let result = run_request(&request, Instant::now()).unwrap();
    assert_eq!(result.bead_count, 3);
    assert!(result.beads.iter().all(|bead| bead.name == "TC5"));
}

#[test]
fn smiles_mapping_honors_target_bead_size() {
    let tmp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "hexane_two_atom_beads",
        "smiles": "CCCCCC",
        "mapping": {
            "mode": "auto",
            "target_bead_size": 2
        },
        "output": {
            "out_dir": tmp.path().to_string_lossy().to_string(),
            "write_mapping_json": false,
            "write_topology_itp": false,
            "write_topology_top": false,
            "write_cg_pdb": false,
            "write_bonded_parameter_map": false
        }
    });

    let (exit_code, result) = run_request_json(&request.to_string(), false);

    assert_eq!(exit_code, 0, "{result}");
    assert_eq!(result["bead_count"], 3);
}

#[test]
fn downstream_setup_artifacts_include_pdb_itp_top_and_parameter_map() {
    let tmp = tempfile::tempdir().unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "benzene".to_string(),
        smiles: Some("c1ccccc1".to_string()),
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: None,
        topology: None,
        trajectory_source: None,
        reference_source: None,
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: true,
            write_topology_itp: true,
            write_topology_top: true,
            write_cg_pdb: true,
            cg_pdb: None,
            write_bonded_parameter_map: true,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    let artifact_kinds: Vec<&str> = result
        .artifacts
        .iter()
        .map(|artifact| artifact.kind.as_str())
        .collect();

    assert!(artifact_kinds.contains(&"coarse_grained_pdb"));
    assert!(artifact_kinds.contains(&"martini_topology_itp"));
    assert!(artifact_kinds.contains(&"martini_topology_top"));
    assert!(artifact_kinds.contains(&"bonded_parameter_map_json"));
    assert!(tmp.path().join("benzene_cg.pdb").exists());
    assert!(tmp.path().join("benzene_martini.itp").exists());
    assert!(tmp.path().join("benzene_martini.top").exists());
    assert!(tmp
        .path()
        .join("benzene_bonded_parameter_map.json")
        .exists());
}

#[test]
fn external_trajectory_reference_uses_gromacs_bonded_terms() {
    let tmp = tempfile::tempdir().unwrap();
    let traj_path = tmp.path().join("benzene_aa.xyz");
    std::fs::write(
        &traj_path,
        concat!(
            "6\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 1.0 0.0 0.0\n",
            "C 2.0 0.0 0.0\n",
            "C 3.0 0.0 0.0\n",
            "C 4.0 0.0 0.0\n",
            "C 5.0 0.0 0.0\n",
            "6\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 1.2 0.0 0.0\n",
            "C 2.2 0.1 0.0\n",
            "C 3.4 0.1 0.0\n",
            "C 4.4 0.0 0.0\n",
            "C 5.6 0.0 0.0\n",
        ),
    )
    .unwrap();
    let topology_path = tmp.path().join("benzene_cg.itp");
    std::fs::write(
        &topology_path,
        concat!(
            "[ moleculetype ]\n",
            "BENZ 1\n",
            "[ atoms ]\n",
            "1 TC5 1 BENZ B1 1 0.0\n",
            "2 TC5 1 BENZ B2 1 0.0\n",
            "3 TC5 1 BENZ B3 1 0.0\n",
            "[ constraints ]\n",
            "1 2 1 0.47\n",
            "[ bonds ]\n",
            "2 3 1 0.47 1250\n",
            "[ angles ]\n",
            "1 2 3 2 180.0 25\n",
        ),
    )
    .unwrap();

    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "benzene".to_string(),
        smiles: Some("c1ccccc1".to_string()),
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: None,
        topology: None,
        trajectory_source: Some(TrajectorySource {
            path: traj_path.to_string_lossy().to_string(),
            topology: None,
            format: Some("xyz".to_string()),
            topology_format: None,
            kind: "external".to_string(),
            stride: None,
            start: None,
            stop: None,
            length_scale: None,
            target_selection: None,
            environment_selection: None,
            atom_indices: None,
            mass_weighted: None,
            make_whole: None,
            sasa: None,
        }),
        reference_source: Some(ReferenceSource {
            kind: "external".to_string(),
            xtb: None,
            precomputed: None,
            bonded_terms: Some(BondedTermSource {
                kind: "gromacs_itp".to_string(),
                path: topology_path.to_string_lossy().to_string(),
                molecule_type: "BENZ".to_string(),
            }),
            metrics: Vec::new(),
            transform: None,
        }),
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: Some("benzene_mapped.gro".to_string()),
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    let artifact_kinds: Vec<&str> = result
        .artifacts
        .iter()
        .map(|artifact| artifact.kind.as_str())
        .collect();
    assert!(artifact_kinds.contains(&"reference_targets_json"));
    let reference = result.reference.as_ref().unwrap();
    assert_eq!(reference.source_kind, "external_trajectory");
    assert!(reference.target_set_available);
    assert_eq!(reference.metadata.mapped_by, "trajectory");
    assert_eq!(reference.metadata.frames_read, 2);
    assert_eq!(reference.metadata.frames_written, 2);
    assert!(reference.metrics.contains_key("rg_mean_nm"));
    assert!(reference
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == "reference_targets_json"));

    let targets: Value = serde_json::from_slice(
        &std::fs::read(tmp.path().join("benzene_reference_targets.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(targets["constraints"].as_array().unwrap().len(), 1);
    assert_eq!(targets["bonds"].as_array().unwrap().len(), 1);
    assert_eq!(targets["angles"].as_array().unwrap().len(), 1);
    assert_eq!(targets["dihedrals"].as_array().unwrap().len(), 0);
}

#[test]
fn source_manifest_request_validates_without_smiles() {
    let tmp = tempfile::tempdir().unwrap();
    let manifest_path = tmp.path().join("polymer_pack_manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "warp-pack.manifest.v1",
            "artifacts": {}
        }))
        .unwrap(),
    )
    .unwrap();
    let request = json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "paa_50mer",
        "source": {
            "kind": "polymer_pack_manifest",
            "path": manifest_path.to_string_lossy()
        },
        "mapping": {
            "mode": "auto",
            "repeat_unit_hint": "PAA",
            "terminal_aware": true
        },
        "output": {"out_dir": tmp.path().to_string_lossy()}
    });
    let (exit_code, value) = validate_request_json(&request.to_string());
    assert_eq!(exit_code, 0, "{value}");
    assert_eq!(value["valid"], true);
    assert_eq!(value["summary"]["input_mode"], "polymer_pack_manifest");
}

#[test]
fn coordinates_topology_source_runs_residue_mapping_without_smiles() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("source.pdb");
    std::fs::write(&source_path, source_polymer_pdb()).unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "paa_source".to_string(),
        smiles: None,
        repeat_smiles: None,
        source: Some(CgSource {
            kind: "coordinates_topology".to_string(),
            path: None,
            coordinates: Some(source_path.to_string_lossy().to_string()),
            topology: Some(source_path.to_string_lossy().to_string()),
            charge_manifest: None,
            trajectory: None,
            target_selection: None,
            selection: None,
            format: Some("pdb".to_string()),
            topology_format: Some("pdb".to_string()),
        }),
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: Some(CgMappingRequest {
            mode: "auto".to_string(),
            strategy: Some("polymer_residue_graph".to_string()),
            target_bead_size: Some(4),
            preserve_functional_groups: Some(true),
            template: None,
            template_policy: None,
            expected_beads_per_role: std::collections::BTreeMap::new(),
            on_bead_count_mismatch: None,
            ndx: None,
            repeat_unit_hint: Some("PAA".to_string()),
            terminal_aware: Some(true),
            bonded_classing: None,
        }),
        topology: None,
        trajectory_source: None,
        reference_source: None,
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: true,
            write_topology_itp: true,
            write_topology_top: true,
            write_cg_pdb: true,
            cg_pdb: None,
            write_bonded_parameter_map: true,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    let artifact_kinds: Vec<&str> = result
        .artifacts
        .iter()
        .map(|artifact| artifact.kind.as_str())
        .collect();

    assert_eq!(result.summary.input_mode, "coordinates_topology");
    assert_eq!(result.summary.aa_atom_count, Some(6));
    assert_eq!(result.summary.mapped_residue_count, Some(3));
    assert_eq!(result.bead_count, 3);
    assert_eq!(result.beads[0].name, "H");
    assert_eq!(result.beads[1].name, "M");
    assert_eq!(result.beads[2].name, "T");
    assert!(artifact_kinds.contains(&"martini_mapping_json"));
    assert!(artifact_kinds.contains(&"aa_to_cg_mapping_provenance"));
    assert!(artifact_kinds.contains(&"coarse_grained_pdb"));
    assert!(artifact_kinds.contains(&"martini_topology_itp"));
    assert!(artifact_kinds.contains(&"martini_topology_top"));
    assert!(artifact_kinds.contains(&"bonded_parameter_map_json"));
    assert!(artifact_kinds.contains(&"mapping_template_json"));

    let pdb = std::fs::read_to_string(tmp.path().join("paa_source_cg.pdb")).unwrap();
    assert_eq!(pdb.matches("\nTER").count(), 0);
    assert!(pdb.contains(" STA A   1"));
    assert!(pdb.contains(" MID A   2"));
    assert!(pdb.contains(" END A   3"));

    let itp = std::fs::read_to_string(tmp.path().join("paa_source_martini.itp")).unwrap();
    let first_atom = itp
        .lines()
        .find(|line| {
            line.split_whitespace()
                .next()
                .is_some_and(|col| col.parse::<usize>().is_ok())
        })
        .unwrap()
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    assert_eq!(first_atom[1], "SC2");
    assert_eq!(first_atom[4], "H1");
    assert_eq!(first_atom[5], "1");
    assert_eq!(first_atom[6].parse::<f64>().unwrap(), 0.0);

    let mapping: Value = serde_json::from_slice(
        &std::fs::read(tmp.path().join("paa_source_martini_mapping.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(mapping["kind"], "martini_source_residue_mapping");
    assert_eq!(mapping["beads"][0]["bead_type"], "SC2");
    assert_eq!(
        mapping["generated_mapping_template"]["residue_role_templates"]["middle"]["beads"][0]
            ["bead_type"],
        "SC2"
    );
    assert_eq!(
        mapping["generated_mapping_template"]["residue_role_templates"]["middle"]["beads"][0]
            ["features"][0],
        "hydrocarbon"
    );
    assert_eq!(
        mapping["generated_mapping_template"]["residue_role_templates"]["middle"]["beads"][0]
            ["local_bonds"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        mapping["generated_mapping_template"]["validation"]["match_by"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    assert_eq!(mapping["residue_to_bead_map"].as_array().unwrap().len(), 3);
    assert_eq!(
        mapping["provenance"]["aa_atom_to_cg_bead"]
            .as_array()
            .unwrap()
            .len(),
        6
    );
    assert_eq!(
        mapping["provenance"]["selection"]["policy"],
        "default_all_source_coordinate_atoms_and_residues"
    );
    assert_eq!(mapping["provenance"]["selection"]["selected_atom_count"], 6);
    assert_eq!(
        mapping["provenance"]["selection"]["selected_residue_count"],
        3
    );
    assert_eq!(
        mapping["provenance"]["residue_interpretation"]["repeat_unit_hint"],
        "PAA"
    );
    assert_eq!(
        mapping["provenance"]["residue_interpretation"]["terminal_aware"],
        true
    );
    assert_eq!(
        mapping["provenance"]["residue_interpretation"]["residue_name_counts"]["MID"],
        1
    );
    assert_eq!(
        mapping["provenance"]["residue_interpretation"]["residues"][0]["role"],
        "head"
    );
    assert_eq!(
        mapping["provenance"]["residue_interpretation"]["residues"][2]["role"],
        "tail"
    );

    let replay_tmp = tempfile::tempdir().unwrap();
    let replay_request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "paa_source_replay".to_string(),
        smiles: None,
        repeat_smiles: None,
        source: request.source.clone(),
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: Some(CgMappingRequest {
            mode: "template".to_string(),
            strategy: None,
            target_bead_size: None,
            preserve_functional_groups: None,
            template: Some(
                tmp.path()
                    .join("paa_source_mapping_template.json")
                    .to_string_lossy()
                    .to_string(),
            ),
            template_policy: None,
            expected_beads_per_role: std::collections::BTreeMap::new(),
            on_bead_count_mismatch: None,
            ndx: None,
            repeat_unit_hint: Some("PAA".to_string()),
            terminal_aware: Some(true),
            bonded_classing: None,
        }),
        topology: None,
        trajectory_source: None,
        reference_source: None,
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: replay_tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: true,
            write_topology_itp: true,
            write_topology_top: false,
            write_cg_pdb: true,
            cg_pdb: None,
            write_bonded_parameter_map: true,
        },
    };
    let replay = run_request(&replay_request, Instant::now()).unwrap();
    assert_eq!(replay.summary.mapping_mode, "template");
    assert_eq!(replay.bead_count, result.bead_count);
    assert_eq!(replay.connections, result.connections);
}

#[test]
fn paa_like_source_auto_template_replay_preserves_carboxylate_features() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("paa_like.pdb");
    std::fs::write(&source_path, paa_like_polymer_pdb()).unwrap();
    let request = source_request("paa_like", &source_path, tmp.path(), "auto", None);

    let result = run_request(&request, Instant::now()).unwrap();
    assert_eq!(result.summary.mapped_residue_count, Some(3));
    assert!(result.beads.iter().any(|bead| bead
        .features
        .iter()
        .any(|feature| feature == "carboxylate_or_carboxylic_acid")));

    let mapping: Value = serde_json::from_slice(
        &std::fs::read(tmp.path().join("paa_like_martini_mapping.json")).unwrap(),
    )
    .unwrap();
    let middle_beads = mapping["generated_mapping_template"]["residue_role_templates"]["middle"]
        ["beads"]
        .as_array()
        .unwrap();
    assert!(middle_beads.iter().any(|bead| bead["features"]
        .as_array()
        .unwrap()
        .iter()
        .any(|feature| feature == "carboxylate_or_carboxylic_acid")));

    let replay_tmp = tempfile::tempdir().unwrap();
    let replay = run_request(
        &source_request(
            "paa_like_replay",
            &source_path,
            replay_tmp.path(),
            "template",
            Some(
                tmp.path()
                    .join("paa_like_mapping_template.json")
                    .to_string_lossy()
                    .to_string(),
            ),
        ),
        Instant::now(),
    )
    .unwrap();
    assert_eq!(replay.bead_count, result.bead_count);
    assert_eq!(replay.connections, result.connections);
}

#[test]
fn template_replay_reference_uses_grouped_bonded_classes() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("paa_like.pdb");
    std::fs::write(&source_path, paa_like_polymer_pdb()).unwrap();
    let auto = run_request(
        &source_request("paa_like_classes", &source_path, tmp.path(), "auto", None),
        Instant::now(),
    )
    .unwrap();
    assert_eq!(auto.summary.mapping_mode, "auto");

    let replay_tmp = tempfile::tempdir().unwrap();
    let mut replay_request = source_request(
        "paa_like_classes_replay",
        &source_path,
        replay_tmp.path(),
        "template",
        Some(
            tmp.path()
                .join("paa_like_classes_mapping_template.json")
                .to_string_lossy()
                .to_string(),
        ),
    );
    replay_request.source.as_mut().unwrap().trajectory =
        Some(source_path.to_string_lossy().to_string());
    replay_request.optimization = Some(ParameterTuningRequest {
        enabled: true,
        source: "aa_trajectory".to_string(),
        method: "bo".to_string(),
        fitting_mode: Some("direct_statistics".to_string()),
        allow_single_frame: Some(true),
        min_samples_per_term: None,
        on_insufficient_samples: None,
        max_evaluations: Some(1),
        seed: Some(1),
        initial_parameters: std::collections::BTreeMap::new(),
        swarm_size: None,
        pso: None,
        bo: None,
        objective: "bonded_parameter_parity".to_string(),
        target_terms: Some(vec!["bonds".to_string()]),
        xtb: None,
        metric_scoring: None,
        evaluator: None,
        runner: None,
    });

    let replay = run_request(&replay_request, Instant::now()).unwrap();
    let report = replay.optimization.unwrap().report.unwrap();
    let names = report
        .best_parameters
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();
    let classing = &replay.mapping_summary.as_ref().unwrap()["bonded_parameter_classing"];

    assert_eq!(classing["class_source"], "template");
    assert!(classing["raw_instance_counts"]["bonds"].as_u64().unwrap() >= 2);
    assert!(
        names.iter().any(|name| name.starts_with("bond.middle.")),
        "{names:?}"
    );
    assert!(!names.iter().any(|name| name.starts_with("bond_")));
}

#[test]
fn grouped_bonded_fitting_rejects_single_frame_reference_when_strict() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("paa_like.pdb");
    std::fs::write(&source_path, paa_like_polymer_pdb()).unwrap();
    run_request(
        &source_request(
            "paa_like_strict_frames",
            &source_path,
            tmp.path(),
            "auto",
            None,
        ),
        Instant::now(),
    )
    .unwrap();

    let replay_tmp = tempfile::tempdir().unwrap();
    let mut replay_request = source_request(
        "paa_like_strict_frames_replay",
        &source_path,
        replay_tmp.path(),
        "template",
        Some(
            tmp.path()
                .join("paa_like_strict_frames_mapping_template.json")
                .to_string_lossy()
                .to_string(),
        ),
    );
    replay_request.source.as_mut().unwrap().trajectory =
        Some(source_path.to_string_lossy().to_string());
    replay_request.optimization = Some(ParameterTuningRequest {
        enabled: true,
        source: "aa_trajectory".to_string(),
        method: "bo".to_string(),
        fitting_mode: Some("distribution_fit".to_string()),
        allow_single_frame: Some(false),
        min_samples_per_term: None,
        on_insufficient_samples: Some("error".to_string()),
        max_evaluations: Some(1),
        seed: Some(1),
        initial_parameters: std::collections::BTreeMap::new(),
        swarm_size: None,
        pso: None,
        bo: None,
        objective: "bonded_parameter_parity".to_string(),
        target_terms: Some(vec!["bonds".to_string()]),
        xtb: None,
        metric_scoring: None,
        evaluator: None,
        runner: None,
    });

    let err = run_request(&replay_request, Instant::now()).unwrap_err();
    assert!(err.to_string().contains("fewer than 2 frames"));
}

#[test]
fn template_replay_applies_source_selection_before_mapping() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("multi_chain_paa_like.pdb");
    let chain_a = paa_like_polymer_pdb();
    let chain_b = paa_like_polymer_pdb()
        .replace(" A   1", " B  11")
        .replace(" A   2", " B  12")
        .replace(" A   3", " B  13");
    std::fs::write(&source_path, format!("{chain_a}{chain_b}")).unwrap();

    let mut auto_request = source_request(
        "paa_like_chain_a_template",
        &source_path,
        tmp.path(),
        "auto",
        None,
    );
    auto_request.source.as_mut().unwrap().selection = Some("chain A".to_string());
    auto_request.bonding = Some(BondingPolicyRequest {
        source: Some("infer_from_coordinates".to_string()),
        infer_bonds: Some(true),
        on_ambiguous: Some("warn".to_string()),
    });
    run_request(&auto_request, Instant::now()).unwrap();

    let replay_tmp = tempfile::tempdir().unwrap();
    let mut replay_request = source_request(
        "paa_like_chain_a_replay",
        &source_path,
        replay_tmp.path(),
        "template",
        Some(
            tmp.path()
                .join("paa_like_chain_a_template_mapping_template.json")
                .to_string_lossy()
                .to_string(),
        ),
    );
    replay_request.source.as_mut().unwrap().selection = Some("chain A".to_string());
    replay_request.bonding = auto_request.bonding.clone();

    let replay = run_request(&replay_request, Instant::now()).unwrap();
    assert_eq!(replay.summary.mapped_residue_count, Some(3));
    let mapping_json: Value = serde_json::from_slice(
        &std::fs::read(
            replay_tmp
                .path()
                .join("paa_like_chain_a_replay_martini_mapping.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(mapping_json["beads"]
        .as_array()
        .unwrap()
        .iter()
        .all(|bead| bead["chain"] == "A"));
    assert!(replay
        .warnings
        .iter()
        .any(|warning| warning["code"] == "warp_cg.source_selection_applied"));
}

#[test]
fn template_replay_source_only_render_uses_grouped_bonded_classes() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("paa_like.pdb");
    std::fs::write(&source_path, paa_like_polymer_pdb()).unwrap();
    run_request(
        &source_request(
            "paa_like_grouped_render",
            &source_path,
            tmp.path(),
            "auto",
            None,
        ),
        Instant::now(),
    )
    .unwrap();

    let replay_tmp = tempfile::tempdir().unwrap();
    let replay_request = source_request(
        "paa_like_grouped_render_replay",
        &source_path,
        replay_tmp.path(),
        "template",
        Some(
            tmp.path()
                .join("paa_like_grouped_render_mapping_template.json")
                .to_string_lossy()
                .to_string(),
        ),
    );
    let replay = run_request(&replay_request, Instant::now()).unwrap();
    assert_eq!(
        replay.mapping_summary.as_ref().unwrap()["bonded_parameter_classing"]["class_source"],
        "template"
    );

    let itp = std::fs::read_to_string(
        replay_tmp
            .path()
            .join("paa_like_grouped_render_replay_martini.itp"),
    )
    .unwrap();
    assert!(itp.contains("; class: bond.middle."), "{itp}");

    let parameter_map: Value = serde_json::from_slice(
        &std::fs::read(
            replay_tmp
                .path()
                .join("paa_like_grouped_render_replay_bonded_parameter_map.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        parameter_map["bonded_parameter_classing"]["class_source"],
        "source_mapping.bonded_terms"
    );
    assert!(parameter_map["bonds"]
        .as_array()
        .unwrap()
        .iter()
        .any(|entry| entry["class_label"]
            .as_str()
            .unwrap_or_default()
            .starts_with("bond.middle.")));
}

#[test]
fn pes_like_source_auto_mapping_preserves_aromatic_ring_semantics() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("pes_like.pdb");
    std::fs::write(&source_path, pes_like_source_pdb()).unwrap();
    let mut request = source_request("pes_like", &source_path, tmp.path(), "auto", None);
    if let Some(mapping) = request.mapping.as_mut() {
        mapping.repeat_unit_hint = Some("PES".to_string());
        mapping.terminal_aware = Some(false);
    }

    let result = run_request(&request, Instant::now()).unwrap();
    assert_eq!(result.summary.mapped_residue_count, Some(1));
    assert_eq!(result.bead_count, 8);
    assert_eq!(
        result
            .beads
            .iter()
            .filter(|bead| bead
                .features
                .iter()
                .any(|feature| feature == "aromatic_ring"))
            .count(),
        6
    );
    assert!(result.beads.iter().any(|bead| bead
        .features
        .iter()
        .any(|feature| feature == "sulfone_or_sulfonate")));
    assert!(result
        .beads
        .iter()
        .any(|bead| bead.features.iter().any(|feature| feature == "ether")));

    let mapping: Value = serde_json::from_slice(
        &std::fs::read(tmp.path().join("pes_like_martini_mapping.json")).unwrap(),
    )
    .unwrap();
    let middle_beads = mapping["generated_mapping_template"]["residue_role_templates"]["middle"]
        ["beads"]
        .as_array()
        .unwrap();
    assert_eq!(middle_beads.len(), 8);
    assert_eq!(
        middle_beads
            .iter()
            .filter(|bead| bead["features"]
                .as_array()
                .unwrap()
                .iter()
                .any(|feature| feature == "aromatic_ring"))
            .count(),
        6
    );
}

#[test]
fn structure_source_infers_bonds_and_reports_mapping_summary() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("benzene_no_conect.pdb");
    std::fs::write(&source_path, benzene_structure_pdb_without_conect()).unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "benzene_structure".to_string(),
        smiles: None,
        repeat_smiles: None,
        source: Some(CgSource {
            kind: "structure".to_string(),
            path: None,
            coordinates: Some(source_path.to_string_lossy().to_string()),
            topology: None,
            charge_manifest: None,
            trajectory: None,
            target_selection: None,
            selection: Some("chain A".to_string()),
            format: Some("pdb".to_string()),
            topology_format: None,
        }),
        bonding: Some(BondingPolicyRequest {
            source: Some("infer_from_coordinates".to_string()),
            infer_bonds: Some(true),
            on_ambiguous: Some("warn".to_string()),
        }),
        chemistry_hints: vec![ChemistryHintRequest {
            kind: "smiles".to_string(),
            scope: "molecule".to_string(),
            value: Some("c1ccccc1".to_string()),
            path: None,
        }],
        chemistry_policy: Some(ChemistryPolicyRequest {
            hint_mode: Some("validate".to_string()),
            on_conflict: Some("warn".to_string()),
        }),
        polymer: Some(PolymerPolicyRequest {
            enabled: Some(false),
            role_mode: Some("infer".to_string()),
            terminal_aware: Some(false),
            end_group_policy: Some("preserve".to_string()),
        }),
        mapping: Some(CgMappingRequest {
            mode: "auto".to_string(),
            strategy: None,
            target_bead_size: Some(4),
            preserve_functional_groups: Some(true),
            template: None,
            template_policy: None,
            expected_beads_per_role: std::collections::BTreeMap::new(),
            on_bead_count_mismatch: None,
            ndx: None,
            repeat_unit_hint: None,
            terminal_aware: None,
            bonded_classing: None,
        }),
        topology: None,
        trajectory_source: None,
        reference_source: None,
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: true,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    assert_eq!(result.summary.input_mode, "structure");
    assert_eq!(result.bead_count, 3);
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning["code"] == "warp_cg.bonds_inferred_from_coordinates"));
    let summary = result.mapping_summary.as_ref().unwrap();
    assert_eq!(summary["bond_source"], "inferred_distance");
    assert_eq!(summary["aromaticity_source"], "geometry");
    assert_eq!(summary["polymer_enabled"], false);
    assert_eq!(summary["chemistry_hint_count"], 1);
    assert_eq!(summary["residue_bead_counts"][0]["role"], "standalone");
    assert_eq!(summary["residue_bead_counts"][0]["bead_count"], 3);

    let provenance: Value = serde_json::from_slice(
        &std::fs::read(
            tmp.path()
                .join("benzene_structure_aa_to_cg_provenance.json"),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        provenance["residue_interpretation"]["polymer_enabled"],
        false
    );
    assert_eq!(
        provenance["residue_interpretation"]["residues"][0]["role"],
        "standalone"
    );
    assert_eq!(provenance["chemistry_hints"][0]["kind"], "smiles");
}

#[test]
fn smiles_hint_reports_aromatic_geometry_conflict() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("distorted_benzene_no_conect.pdb");
    std::fs::write(
        &source_path,
        distorted_benzene_structure_pdb_without_conect(),
    )
    .unwrap();
    let mut request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "distorted_benzene_structure".to_string(),
        smiles: None,
        repeat_smiles: None,
        source: Some(CgSource {
            kind: "structure".to_string(),
            path: None,
            coordinates: Some(source_path.to_string_lossy().to_string()),
            topology: None,
            charge_manifest: None,
            trajectory: None,
            target_selection: None,
            selection: None,
            format: Some("pdb".to_string()),
            topology_format: None,
        }),
        bonding: Some(BondingPolicyRequest {
            source: Some("infer_from_coordinates".to_string()),
            infer_bonds: Some(true),
            on_ambiguous: Some("warn".to_string()),
        }),
        chemistry_hints: vec![ChemistryHintRequest {
            kind: "smiles".to_string(),
            scope: "molecule".to_string(),
            value: Some("c1ccccc1".to_string()),
            path: None,
        }],
        chemistry_policy: Some(ChemistryPolicyRequest {
            hint_mode: Some("validate".to_string()),
            on_conflict: Some("warn".to_string()),
        }),
        polymer: Some(PolymerPolicyRequest {
            enabled: Some(false),
            role_mode: Some("infer".to_string()),
            terminal_aware: Some(false),
            end_group_policy: Some("preserve".to_string()),
        }),
        mapping: Some(CgMappingRequest {
            mode: "auto".to_string(),
            strategy: None,
            target_bead_size: Some(4),
            preserve_functional_groups: Some(true),
            template: None,
            template_policy: None,
            expected_beads_per_role: std::collections::BTreeMap::new(),
            on_bead_count_mismatch: None,
            ndx: None,
            repeat_unit_hint: None,
            terminal_aware: None,
            bonded_classing: None,
        }),
        topology: None,
        trajectory_source: None,
        reference_source: None,
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    let conflict = result
        .warnings
        .iter()
        .find(|warning| warning["code"] == "warp_cg.chemistry_hint_geometry_conflict")
        .unwrap();
    assert_eq!(conflict["hint_aromatic_six_ring_count"], 1);
    assert_eq!(conflict["geometry_aromatic_six_ring_count"], 0);

    request.chemistry_policy.as_mut().unwrap().on_conflict = Some("error".to_string());
    let err = run_request(&request, Instant::now()).unwrap_err();
    assert!(err
        .to_string()
        .contains("warp_cg.chemistry_hint_geometry_conflict"));
}

#[test]
fn template_replay_rejects_wrong_local_bond_signature() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("paa_like.pdb");
    std::fs::write(&source_path, paa_like_polymer_pdb()).unwrap();
    let request = source_request(
        "paa_like_bad_template",
        &source_path,
        tmp.path(),
        "auto",
        None,
    );
    run_request(&request, Instant::now()).unwrap();

    let template_path = tmp
        .path()
        .join("paa_like_bad_template_mapping_template.json");
    let mut template: Value =
        serde_json::from_slice(&std::fs::read(&template_path).unwrap()).unwrap();
    template["residue_role_templates"]["middle"]["beads"][0]["local_bonds"] = json!([["C1", "O2"]]);
    let bad_template_path = tmp.path().join("bad_mapping_template.json");
    std::fs::write(
        &bad_template_path,
        serde_json::to_vec_pretty(&template).unwrap(),
    )
    .unwrap();

    let err = run_request(
        &source_request(
            "paa_like_bad_replay",
            &source_path,
            tmp.path(),
            "template",
            Some(bad_template_path.to_string_lossy().to_string()),
        ),
        Instant::now(),
    )
    .unwrap_err();
    assert!(err.to_string().contains("local bond mismatch"));

    let mut assignment_request = source_request(
        "paa_like_assignment_only_replay",
        &source_path,
        tmp.path(),
        "template",
        Some(bad_template_path.to_string_lossy().to_string()),
    );
    assignment_request.mapping.as_mut().unwrap().template_policy =
        Some("assignment_only".to_string());
    let result = run_request(&assignment_request, Instant::now()).unwrap();
    assert!(result
        .warnings
        .iter()
        .any(|warning| warning["code"] == "warp_cg.template_assignment_only"));
    assert_eq!(
        result.mapping_summary.as_ref().unwrap()["template_policy"],
        "assignment_only"
    );
}

#[test]
fn mapping_expected_beads_per_role_rejects_mismatch() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("paa_like.pdb");
    std::fs::write(&source_path, paa_like_polymer_pdb()).unwrap();
    let mut request = source_request("paa_like_bad_count", &source_path, tmp.path(), "auto", None);
    let mapping = request.mapping.as_mut().unwrap();
    mapping
        .expected_beads_per_role
        .insert("middle".to_string(), 99);
    mapping.on_bead_count_mismatch = Some("error".to_string());
    let err = run_request(&request, Instant::now()).unwrap_err();
    assert!(err.to_string().contains("warp_cg.bead_count_mismatch"));
}

#[test]
fn polymer_manifest_source_run_resolves_relative_artifacts() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("source.pdb");
    let manifest_path = tmp.path().join("polymer_build_manifest.json");
    std::fs::write(&source_path, source_polymer_pdb()).unwrap();
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "warp-build.manifest.v1",
            "artifacts": {
                "coordinates": "source.pdb",
                "topology": {"path": "source.pdb"}
            }
        }))
        .unwrap(),
    )
    .unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "manifest_source".to_string(),
        smiles: None,
        repeat_smiles: None,
        source: Some(CgSource {
            kind: "polymer_build_manifest".to_string(),
            path: Some(manifest_path.to_string_lossy().to_string()),
            coordinates: None,
            topology: None,
            charge_manifest: None,
            trajectory: None,
            target_selection: None,
            selection: None,
            format: Some("pdb".to_string()),
            topology_format: Some("pdb".to_string()),
        }),
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: Some(CgMappingRequest {
            mode: "auto".to_string(),
            strategy: Some("polymer_residue_graph".to_string()),
            target_bead_size: Some(4),
            preserve_functional_groups: Some(true),
            template: None,
            template_policy: None,
            expected_beads_per_role: std::collections::BTreeMap::new(),
            on_bead_count_mismatch: None,
            ndx: None,
            repeat_unit_hint: Some("PAA".to_string()),
            terminal_aware: Some(true),
            bonded_classing: None,
        }),
        topology: None,
        trajectory_source: None,
        reference_source: None,
        forcefield: None,
        optimization: None,
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: None,
            write_mapping_json: true,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    assert_eq!(result.summary.input_mode, "polymer_build_manifest");
    assert_eq!(result.summary.mapped_residue_count, Some(3));
    assert!(tmp
        .path()
        .join("manifest_source_martini_mapping.json")
        .exists());
    assert!(tmp
        .path()
        .join("manifest_source_aa_to_cg_provenance.json")
        .exists());
}
