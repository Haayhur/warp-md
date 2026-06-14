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
        },
    );
    let names: Vec<&str> = report
        .best_parameters
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();

    assert_eq!(report.status, "ok");
    assert_eq!(report.objective, "reference_target_emd");
    assert_eq!(report.best_parameters.len(), 2);
    assert!(names.contains(&"bond_0_1_length_angstrom"));
    assert!(names.contains(&"angle_0_1_2_angle_deg"));
    assert!(report
        .evaluations
        .iter()
        .any(|record| record.metrics.contains_key("bonds_emd")));
    assert!(report.objective_value.is_finite());
}
