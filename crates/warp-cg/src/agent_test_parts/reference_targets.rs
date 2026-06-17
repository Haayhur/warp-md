use super::*;

#[test]
fn reference_target_optimization_honors_requested_target_terms() {
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
            "C 2.3 0.2 0.0\n",
            "C 3.4 0.2 0.0\n",
            "C 4.5 0.0 0.0\n",
            "C 5.7 0.0 0.0\n",
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
            "[ bonds ]\n",
            "1 2 1 0.47 1250\n",
            "[ angles ]\n",
            "1 2 3 2 180.0 25\n",
        ),
    )
    .unwrap();
    let request = CgRequest {
        schema_version: AGENT_SCHEMA_VERSION.to_string(),
        name: "benzene_terms".to_string(),
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
        optimization: Some(ParameterTuningRequest {
            enabled: true,
            source: "aa_trajectory".to_string(),
            method: "pso".to_string(),
            fitting_mode: None,
            allow_single_frame: None,
            min_samples_per_term: None,
            on_insufficient_samples: None,
            max_evaluations: Some(6),
            seed: Some(4),
            initial_parameters: std::collections::BTreeMap::new(),
            swarm_size: Some(4),
            pso: None,
            bo: None,
            objective: "reference_target_emd".to_string(),
            target_terms: Some(vec!["angles".to_string()]),
            xtb: None,
            metric_scoring: None,
            evaluator: None,
            runner: None,
        }),
        output: CgOutputRequest {
            out_dir: tmp.path().to_string_lossy().to_string(),
            mapped_trajectory: Some("benzene_mapped.gro".to_string()),
            write_mapping_json: false,
            write_topology_itp: false,
            write_topology_top: false,
            write_cg_pdb: false,
            cg_pdb: None,
            write_bonded_parameter_map: false,
            exclusions: None,
            dihedrals: None,
            coordinates: None,
        },
    };

    let result = run_request(&request, Instant::now()).unwrap();
    let report = result.optimization.unwrap().report.unwrap();
    let names = report
        .best_parameters
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>();

    assert_eq!(result.summary.optimized_terms, vec!["angles"]);
    assert_eq!(
        names,
        vec!["angle_group_1_angle_deg", "angle_group_1_force"]
    );
    assert!(report
        .evaluations
        .iter()
        .all(|record| record.metrics["bonds_emd"] == 0.0));
}
