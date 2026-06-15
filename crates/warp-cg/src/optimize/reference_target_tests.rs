use super::*;
use crate::reference::{
    ReferenceBinConfig, ReferenceDistributionTarget, ReferenceTargetSet, ReferenceTermKind,
};

#[test]
fn reference_target_optimization_uses_grouped_emd_objective() {
    let target = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond group 1".to_string()),
            vec![0, 1],
            vec![vec![0, 1], vec![2, 3]],
            &[0.40, 0.41, 0.42, 0.43],
            "nm",
            false,
            0.0,
            3.0,
            0.01,
        )],
        angles: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Angle,
            Some("angle group 1".to_string()),
            vec![0, 1, 2],
            vec![vec![0, 1, 2]],
            &[119.0, 120.0, 121.0],
            "deg",
            false,
            0.0,
            180.0,
            1.0,
        )],
        dihedrals: Vec::new(),
    };

    let report = optimize_reference_targets(
        &target,
        &OptimizationConfig {
            method: "pso".to_string(),
            objective: "reference_target_emd".to_string(),
            max_evaluations: 24,
            seed: 23,
            swarm_size: Some(8),
            pso: None,
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
    );
    let names: Vec<&str> = report
        .best_parameters
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    assert_eq!(report.status, "ok");
    assert_eq!(report.objective, "reference_target_emd");
    assert_eq!(report.best_parameters.len(), 4);
    assert!(names.contains(&"bond_group_1_length_nm"));
    assert!(names.contains(&"bond_group_1_force"));
    assert!(names.contains(&"angle_group_1_angle_deg"));
    assert!(names.contains(&"angle_group_1_force"));
    assert!(report
        .evaluations
        .iter()
        .any(|record| record.metrics.contains_key("bonds_emd")));
    assert!(report.objective_value.is_finite());
}

#[test]
fn reference_target_optimization_evaluates_named_initial_parameters_first() {
    let target = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond.middle.M0_AR1__M0_SO2".to_string()),
            vec![0, 1],
            vec![vec![0, 1], vec![8, 9]],
            &[0.40, 0.42],
            "nm",
            false,
            0.0,
            3.0,
            0.01,
        )],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };
    let mut initial_parameters = std::collections::BTreeMap::new();
    initial_parameters.insert("bond.middle.M0_AR1__M0_SO2_length_nm".to_string(), 0.41);
    initial_parameters.insert("bond.middle.M0_AR1__M0_SO2_force".to_string(), 2500.0);

    let report = optimize_reference_targets(
        &target,
        &OptimizationConfig {
            method: "pso".to_string(),
            objective: "reference_target_emd".to_string(),
            max_evaluations: 4,
            seed: 23,
            swarm_size: Some(4),
            pso: None,
            bo: None,
            initial_parameters,
        },
    );

    assert_eq!(report.status, "ok");
    assert_eq!(report.evaluations[0].parameters, vec![0.41, 2500.0]);
}

#[test]
fn reference_target_optimization_rejects_unknown_initial_parameter_names() {
    let target = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond.group".to_string()),
            vec![0, 1],
            vec![vec![0, 1]],
            &[0.40, 0.42],
            "nm",
            false,
            0.0,
            3.0,
            0.01,
        )],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };
    let mut initial_parameters = std::collections::BTreeMap::new();
    initial_parameters.insert("bond.typo_length_nm".to_string(), 0.41);

    let report = optimize_reference_targets(
        &target,
        &OptimizationConfig {
            method: "pso".to_string(),
            objective: "reference_target_emd".to_string(),
            max_evaluations: 4,
            seed: 23,
            swarm_size: Some(4),
            pso: None,
            bo: None,
            initial_parameters,
        },
    );

    assert_eq!(report.status, "error");
    assert!(report.message.contains("unknown parameter name"));
    assert!(report.evaluations.is_empty());
}

#[test]
fn direct_statistics_from_targets_uses_class_labels_and_force_terms() {
    let target = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond.middle.M0_AR1__M0_SO2".to_string()),
            vec![0, 1],
            vec![vec![0, 1], vec![8, 9], vec![16, 17]],
            &[4.0, 4.1, 4.2],
            "angstrom",
            false,
            0.0,
            30.0,
            0.1,
        )],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };

    let report = direct_statistics_report_from_targets(
        &target,
        &OptimizationConfig {
            method: "bo".to_string(),
            objective: "bonded_parameter_parity".to_string(),
            max_evaluations: 1,
            seed: 1,
            swarm_size: None,
            pso: None,
            bo: None,
            initial_parameters: std::collections::BTreeMap::new(),
        },
    );
    let names: Vec<&str> = report
        .best_parameters
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    assert_eq!(report.status, "ok");
    assert!(names.contains(&"bond.middle.M0_AR1__M0_SO2_length_angstrom"));
    assert!(names.contains(&"bond.middle.M0_AR1__M0_SO2_force"));
    assert_eq!(report.best_parameters.len(), 2);
}
