use super::*;

#[test]
fn source_request_accepts_gromacs_ndx_mapping() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("source.pdb");
    std::fs::write(&source_path, source_polymer_pdb()).unwrap();
    let ndx_path = tmp.path().join("cg_map.ndx");
    std::fs::write(
        &ndx_path,
        concat!("[ B1 ]\n", "1 2 3\n", "[ B2 ]\n", "3 4 5\n", "[ B3 ]\n", "5 6\n",),
    )
    .unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "ndx_source".to_string(),
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
            mode: "ndx".to_string(),
            strategy: None,
            target_bead_size: None,
            preserve_functional_groups: None,
            template: None,
            ndx: Some(ndx_path.to_string_lossy().to_string()),
            repeat_unit_hint: None,
            terminal_aware: None,
        }),
        topology: None,
        trajectory_source: None,
        reference_source: None,
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

    assert_eq!(result.summary.mapping_mode, "ndx");
    assert_eq!(result.bead_count, 3);
    assert_eq!(result.beads[0].name, "B1");
    assert_eq!(result.beads[0].atom_indices, vec![0, 1, 2]);
    assert_eq!(result.beads[1].atom_indices, vec![2, 3, 4]);
    assert!(result
        .connections
        .iter()
        .any(|connection| *connection == [0, 1]));

    let mapping_path = result
        .artifact_paths
        .get("martini_mapping_json")
        .expect("mapping artifact path");
    let mapping: serde_json::Value =
        serde_json::from_slice(&std::fs::read(mapping_path).unwrap()).unwrap();
    assert_eq!(mapping["provenance"]["mapping_mode"], "ndx");
}

#[test]
fn ndx_mapping_preflight_reports_missing_file() {
    let tmp = tempfile::tempdir().unwrap();
    let source_path = tmp.path().join("source.pdb");
    std::fs::write(&source_path, source_polymer_pdb()).unwrap();
    let request = serde_json::json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "missing_ndx",
        "source": {
            "kind": "coordinates_topology",
            "coordinates": source_path,
            "topology": source_path,
            "format": "pdb",
            "topology_format": "pdb"
        },
        "mapping": {
            "mode": "ndx",
            "ndx": tmp.path().join("missing.ndx")
        },
        "output": {"out_dir": tmp.path()}
    });

    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2);
    assert_eq!(value["valid"], false);
    assert!(value["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|error| error["field"] == "mapping.ndx"));
}

#[test]
fn trajectory_source_preflight_reports_missing_file() {
    let tmp = tempfile::tempdir().unwrap();
    let ndx_path = tmp.path().join("cg_map.ndx");
    std::fs::write(&ndx_path, "[ B1 ]\n1 2\n").unwrap();
    let itp_path = tmp.path().join("cg_model.itp");
    std::fs::write(
        &itp_path,
        concat!(
            "[ moleculetype ]\n",
            "MOL 1\n",
            "[ atoms ]\n",
            "1 C1 1 MOL B1 1 0.0\n",
        ),
    )
    .unwrap();
    let request = serde_json::json!({
        "schema_version": AGENT_SCHEMA_VERSION,
        "name": "missing_trajectory",
        "trajectory_source": {
            "kind": "external",
            "path": tmp.path().join("missing.xyz"),
            "format": "xyz"
        },
        "mapping": {
            "mode": "ndx",
            "ndx": ndx_path
        },
        "reference_source": {
            "kind": "external",
            "bonded_terms": {
                "kind": "gromacs_itp",
                "path": itp_path,
                "molecule_type": "MOL"
            }
        },
        "output": {"out_dir": tmp.path()}
    });

    let (exit_code, value) = validate_request_json(&request.to_string());

    assert_eq!(exit_code, 2);
    assert_eq!(value["valid"], false);
    assert!(value["errors"]
        .as_array()
        .unwrap()
        .iter()
        .any(|error| error["field"] == "trajectory_source.path"));
}

#[test]
fn trajectory_ndx_reference_runs_without_smiles_or_source() {
    let tmp = tempfile::tempdir().unwrap();
    let traj_path = tmp.path().join("aa.xyz");
    std::fs::write(
        &traj_path,
        concat!(
            "4\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 1.0 0.0 0.0\n",
            "C 2.0 0.0 0.0\n",
            "C 3.0 0.0 0.0\n",
            "4\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 1.1 0.0 0.0\n",
            "C 2.4 0.0 0.0\n",
            "C 3.7 0.0 0.0\n",
        ),
    )
    .unwrap();
    let ndx_path = tmp.path().join("cg_map.ndx");
    std::fs::write(&ndx_path, "[ B1 ]\n1 2\n[ B2 ]\n3 4\n").unwrap();
    let itp_path = tmp.path().join("cg_model.itp");
    std::fs::write(
        &itp_path,
        concat!(
            "[ moleculetype ]\n",
            "MOL 1\n",
            "[ atoms ]\n",
            "1 C1 1 MOL B1 1 0.0\n",
            "2 C1 1 MOL B2 1 0.0\n",
            "[ bonds ]\n",
            "; bond group 1\n",
            "1 2 1 0.47 1250\n",
        ),
    )
    .unwrap();
    let metrics_path = tmp.path().join("reference_metrics.json");
    std::fs::write(
        &metrics_path,
        r#"{"metrics":{"gromacs_sasa_mean_nm2":12.5}}"#,
    )
    .unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "swarm_reference".to_string(),
        smiles: None,
        repeat_smiles: None,
        source: None,
        bonding: None,
        chemistry_hints: Vec::new(),
        chemistry_policy: None,
        polymer: None,
        mapping: Some(CgMappingRequest {
            mode: "ndx".to_string(),
            strategy: None,
            target_bead_size: None,
            preserve_functional_groups: None,
            template: None,
            ndx: Some(ndx_path.to_string_lossy().to_string()),
            repeat_unit_hint: None,
            terminal_aware: None,
        }),
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
        }),
        reference_source: Some(ReferenceSource {
            kind: "external".to_string(),
            xtb: None,
            precomputed: None,
            bonded_terms: Some(BondedTermSource {
                kind: "gromacs_itp".to_string(),
                path: itp_path.to_string_lossy().to_string(),
                molecule_type: "MOL".to_string(),
            }),
            metrics: vec![ReferenceMetricSourceRequest {
                kind: "json".to_string(),
                path: metrics_path.to_string_lossy().to_string(),
                namespace: Some("gromacs".to_string()),
                artifact_kind: None,
            }],
            transform: Some(ReferenceTransformRequest {
                bond_scaling: Some(2.0),
                min_bond_length_nm: None,
                specific_bond_lengths_nm: std::collections::BTreeMap::new(),
                rg_offset_nm: None,
            }),
        }),
        optimization: None,
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: Some("mapped.gro".to_string()),
            write_mapping_json: true,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();

    assert_eq!(result.summary.input_mode, "trajectory_ndx_reference");
    assert_eq!(result.summary.mapping_mode, "ndx");
    assert_eq!(result.bead_count, 2);
    assert_eq!(result.beads[0].atom_indices, vec![0, 1]);
    assert_eq!(result.connections, vec![[0, 1]]);
    assert!(result.artifact_paths.contains_key("reference_targets_json"));
    assert!(result
        .artifact_paths
        .contains_key("coarse_grained_trajectory"));
    assert!(result.artifact_paths.contains_key("reference_metrics_json"));
    assert!(result.artifact_paths.contains_key("bond_stats_json"));
    let reference = result.reference.as_ref().unwrap();
    assert_eq!(reference.source_kind, "aa_trajectory_ndx");
    assert!(reference.target_set_available);
    assert_eq!(reference.metadata.mapped_by, "trajectory");
    assert_eq!(reference.metadata.frames_read, 2);
    assert_eq!(reference.metadata.frames_written, 2);
    assert_eq!(reference.metrics["gromacs.gromacs_sasa_mean_nm2"], 12.5);
    assert!(reference
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == "reference_metrics_json"));
    let bond_stats_path = result.artifact_paths.get("bond_stats_json").unwrap();
    let bond_stats: Value =
        serde_json::from_slice(&std::fs::read(bond_stats_path).unwrap()).unwrap();
    let transformed_mean = bond_stats[0]["mean"].as_f64().unwrap();
    assert!(
        (transformed_mean - 4.5).abs() < 1.0e-6,
        "transformed_mean={transformed_mean}"
    );
}
