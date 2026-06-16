use std::fs;

use super::*;
use crate::bonded_terms::{BondTermGroup, BondedTermSet};

#[test]
fn trajectory_provider_maps_reference_and_records_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("aa.xyz");
    fs::write(
        &traj,
        concat!(
            "2\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 1.0 0.0 0.0\n",
            "2\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 2.0 0.0 0.0\n",
        ),
    )
    .unwrap();
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory",
        &traj,
        NativeTrajectoryOptions::default(),
    );

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "ethanol",
            out_dir: dir.path(),
            mapped_trajectory_name: Some("mapped.gro"),
            mapping: &mapping,
            connections: &[(0, 1)],
            term_set: None,
            metric_sources: &[],
            transform: None,
        })
        .unwrap();

    assert_eq!(data.source_kind, "aa_trajectory");
    assert_eq!(data.bonded_stats.bonds.len(), 1);
    assert_eq!(data.bonded_stats.bonds[0].samples, 2);
    assert_eq!(data.metadata.frames_read, 2);
    assert_eq!(data.metadata.frames_written, 2);
    assert_eq!(data.metrics["rg_samples"], 2.0);
    assert!((data.metrics["rg_mean_nm"] - 0.75).abs() < 1.0e-12);
    assert!((data.metrics["rg_std_nm"] - 0.25).abs() < 1.0e-12);
    assert_eq!(data.metrics["sasa_samples"], 2.0);
    assert_eq!(data.metrics["sasa_approx_samples"], 2.0);
    assert!(data.metrics["sasa_mean_nm2"].is_finite());
    assert!(data.metrics["sasa_mean_nm2"] > 0.0);
    assert_eq!(
        data.metrics["sasa_approx_mean_nm2"],
        data.metrics["sasa_mean_nm2"]
    );
    assert_eq!(data.first_cg_coords.unwrap().len(), 2);
    assert!(data.mapped_trajectory.as_ref().unwrap().is_file());
    let target_set = data.target_set.as_ref().unwrap();
    assert_eq!(target_set.bonds.len(), 1);
    assert_eq!(target_set.bonds[0].samples, 2);
    assert_eq!(target_set.bonds[0].units, "nm");
    assert!((target_set.bonds[0].mean - 1.5).abs() < 1.0e-6);
    assert_eq!(target_set.bonds[0].domain, [1.0, 2.0]);
    assert!((target_set.bonds[0].probabilities.iter().sum::<f64>() - 1.0).abs() < 1.0e-12);
    assert_eq!(data.artifacts.len(), 2);
    assert_eq!(data.artifacts[0].kind, "coarse_grained_trajectory");
    assert_eq!(data.artifacts[1].kind, "reference_targets_json");
    assert!(data.artifacts[1].path.is_file());
}

#[test]
fn trajectory_provider_merges_consumer_metric_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("aa.xyz");
    fs::write(
        &traj,
        concat!(
            "1\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "1\n",
            "frame 1\n",
            "C 1.0 0.0 0.0\n",
        ),
    )
    .unwrap();
    let xvg = dir.path().join("external_sasa.xvg");
    fs::write(&xvg, "# user-owned gmx sasa output\n").unwrap();
    let metrics = dir.path().join("gromacs_metrics.json");
    fs::write(
        &metrics,
        serde_json::json!({
            "metrics": {
                "sasa_mean_nm2": 12.5,
                "sasa_std_nm2": 0.25
            },
            "artifacts": [
                {"path": "external_sasa.xvg", "kind": "gromacs_sasa_xvg"}
            ]
        })
        .to_string(),
    )
    .unwrap();
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string()],
        atom_indices: vec![vec![0]],
    };
    let metric_sources = vec![ReferenceMetricSource {
        path: metrics.clone(),
        kind: "json".to_string(),
        namespace: Some("gromacs".to_string()),
        artifact_kind: None,
    }];
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory",
        &traj,
        NativeTrajectoryOptions::default(),
    );

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "methane",
            out_dir: dir.path(),
            mapped_trajectory_name: None,
            mapping: &mapping,
            connections: &[],
            term_set: None,
            metric_sources: &metric_sources,
            transform: None,
        })
        .unwrap();

    assert_eq!(data.metrics["gromacs.sasa_mean_nm2"], 12.5);
    assert_eq!(data.metrics["gromacs.sasa_std_nm2"], 0.25);
    assert!(data
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == "reference_metrics_json" && artifact.path == metrics));
    assert!(data
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == "gromacs_sasa_xvg" && artifact.path == xvg));
}

#[test]
fn trajectory_provider_applies_min_bond_transform() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("short_bonds.xyz");
    fs::write(
        &traj,
        concat!(
            "2\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 0.1 0.0 0.0\n",
            "2\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 0.2 0.0 0.0\n",
        ),
    )
    .unwrap();
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };
    let transform = ReferenceTransformConfig {
        min_bond_length_nm: Some(0.5),
        ..ReferenceTransformConfig::default()
    };
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory",
        &traj,
        NativeTrajectoryOptions::default(),
    );

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "short_bonds",
            out_dir: dir.path(),
            mapped_trajectory_name: None,
            mapping: &mapping,
            connections: &[(0, 1)],
            term_set: None,
            metric_sources: &[],
            transform: Some(&transform),
        })
        .unwrap();

    assert!((data.bonded_stats.bonds[0].mean - 0.5).abs() < 1.0e-12);
    let target = &data.target_set.as_ref().unwrap().bonds[0];
    assert_eq!(target.units, "nm");
    assert!((target.mean - 0.5).abs() < 1.0e-12);
    assert_eq!(target.samples, 2);
    assert!((target.domain[0] - 1.0 / 3.0).abs() < 1.0e-12);
    assert!((target.domain[1] - 2.0 / 3.0).abs() < 1.0e-12);
}

#[test]
fn trajectory_provider_applies_rg_offset() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("rg_offset.xyz");
    fs::write(
        &traj,
        concat!(
            "2\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 2.0 0.0 0.0\n",
            "2\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 4.0 0.0 0.0\n",
        ),
    )
    .unwrap();
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };
    let transform = ReferenceTransformConfig {
        rg_offset_nm: Some(0.25),
        ..ReferenceTransformConfig::default()
    };
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory",
        &traj,
        NativeTrajectoryOptions::default(),
    );

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "rg_offset",
            out_dir: dir.path(),
            mapped_trajectory_name: None,
            mapping: &mapping,
            connections: &[(0, 1)],
            term_set: None,
            metric_sources: &[],
            transform: Some(&transform),
        })
        .unwrap();

    assert!((data.metrics["rg_mean_nm"] - 1.75).abs() < 1.0e-12);
    assert!((data.metrics["rg_std_nm"] - 0.5).abs() < 1.0e-12);
    assert_eq!(data.metrics["rg_samples"], 2.0);
}

#[test]
fn reference_target_compare_scores_matching_distributions_as_zero() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("aa.xyz");
    fs::write(
        &traj,
        concat!(
            "2\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 1.0 0.0 0.0\n",
            "2\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 2.0 0.0 0.0\n",
        ),
    )
    .unwrap();
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory",
        &traj,
        NativeTrajectoryOptions::default(),
    );

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "ethanol",
            out_dir: dir.path(),
            mapped_trajectory_name: None,
            mapping: &mapping,
            connections: &[(0, 1)],
            term_set: None,
            metric_sources: &[],
            transform: None,
        })
        .unwrap();
    let target_set = data.target_set.as_ref().unwrap();
    let score = target_set.compare(target_set);

    assert_eq!(score.total, 0.0);
    assert_eq!(score.terms.len(), 1);
}

#[test]
fn reference_target_bonded_emd_scales_bonded_distance_terms() {
    let reference = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond group 1".to_string()),
            vec![0, 1],
            vec![vec![0, 1]],
            &[0.40],
            "nm",
            false,
            0.0,
            1.0,
            0.01,
        )],
        angles: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Angle,
            Some("angle group 1".to_string()),
            vec![0, 1, 2],
            vec![vec![0, 1, 2]],
            &[120.0],
            "deg",
            false,
            0.0,
            180.0,
            1.0,
        )],
        dihedrals: Vec::new(),
    };
    let candidate = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond group 1".to_string()),
            vec![0, 1],
            vec![vec![0, 1]],
            &[0.42],
            "nm",
            false,
            0.0,
            1.0,
            0.01,
        )],
        angles: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Angle,
            Some("angle group 1".to_string()),
            vec![0, 1, 2],
            vec![vec![0, 1, 2]],
            &[122.0],
            "deg",
            false,
            0.0,
            180.0,
            1.0,
        )],
        dihedrals: Vec::new(),
    };

    let raw = reference.compare(&candidate);
    let weighted = reference.bonded_emd(&candidate);

    assert!(raw.raw_bonds > 0.0);
    assert!((raw.total - raw.raw_total).abs() < 1.0e-12);
    assert!((weighted.raw_bonds - raw.raw_bonds).abs() < 1.0e-12);
    assert!((weighted.bonds - raw.raw_bonds * 500.0).abs() < 1.0e-9);
    assert!((weighted.angles - raw.raw_angles).abs() < 1.0e-12);
}

#[test]
fn reference_target_bonded_emd_combines_constraints_and_bonds_bucket() {
    let reference = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Constraint,
            Some("constraint group 1".to_string()),
            vec![0, 1],
            vec![vec![0, 1]],
            &[0.10],
            "nm",
            false,
            0.0,
            1.0,
            0.1,
        )],
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond group 1".to_string()),
            vec![1, 2],
            vec![vec![1, 2]],
            &[0.20],
            "nm",
            false,
            0.0,
            1.0,
            0.1,
        )],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };
    let candidate = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Constraint,
            Some("constraint group 1".to_string()),
            vec![0, 1],
            vec![vec![0, 1]],
            &[0.40],
            "nm",
            false,
            0.0,
            1.0,
            0.1,
        )],
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond group 1".to_string()),
            vec![1, 2],
            vec![vec![1, 2]],
            &[0.50],
            "nm",
            false,
            0.0,
            1.0,
            0.1,
        )],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };

    let score = reference.bonded_emd(&candidate);
    let expected_bucket = score.constraints.hypot(score.bonds);

    assert!(score.constraints > 0.0);
    assert!(score.bonds > 0.0);
    assert!((score.constraints_bonds - expected_bucket).abs() < 1.0e-12);
    assert!((score.total - score.constraints_bonds).abs() < 1.0e-12);
    assert!(score.total < score.constraints + score.bonds);
}

#[test]
fn reference_target_filter_terms_keeps_only_requested_groups() {
    let targets = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Constraint,
            Some("constraint group 1".to_string()),
            vec![0, 1],
            vec![vec![0, 1]],
            &[0.40],
            "nm",
            false,
            0.0,
            1.0,
            0.01,
        )],
        bonds: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Bond,
            Some("bond group 1".to_string()),
            vec![1, 2],
            vec![vec![1, 2]],
            &[0.41],
            "nm",
            false,
            0.0,
            1.0,
            0.01,
        )],
        angles: vec![ReferenceDistributionTarget::from_samples(
            ReferenceTermKind::Angle,
            Some("angle group 1".to_string()),
            vec![0, 1, 2],
            vec![vec![0, 1, 2]],
            &[120.0],
            "deg",
            false,
            0.0,
            180.0,
            1.0,
        )],
        dihedrals: Vec::new(),
    };

    let filtered = targets.filter_terms(&["constraints".to_string(), "angles".to_string()]);

    assert_eq!(filtered.constraints.len(), 1);
    assert!(filtered.bonds.is_empty());
    assert_eq!(filtered.angles.len(), 1);
    assert!(filtered.dihedrals.is_empty());
}

#[test]
fn trajectory_provider_uses_explicit_grouped_constraints() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("aa.xyz");
    fs::write(
        &traj,
        concat!(
            "3\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 1.0 0.0 0.0\n",
            "C 2.0 0.0 0.0\n",
            "3\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 1.5 0.0 0.0\n",
            "C 3.0 0.0 0.0\n",
        ),
    )
    .unwrap();
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string(), "B2".to_string()],
        atom_indices: vec![vec![0], vec![1], vec![2]],
    };
    let terms = BondedTermSet {
        constraints: vec![BondTermGroup {
            label: Some("short_constraint_group".to_string()),
            members: vec![[0, 1], [1, 2]],
        }],
        ..BondedTermSet::default()
    };
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory",
        &traj,
        NativeTrajectoryOptions::default(),
    );

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "linear",
            out_dir: dir.path(),
            mapped_trajectory_name: None,
            mapping: &mapping,
            connections: &[],
            term_set: Some(&terms),
            metric_sources: &[],
            transform: None,
        })
        .unwrap();
    let target_set = data.target_set.as_ref().unwrap();

    assert_eq!(target_set.constraints.len(), 1);
    assert_eq!(
        target_set.constraints[0].kind,
        ReferenceTermKind::Constraint
    );
    assert_eq!(
        target_set.constraints[0].label.as_deref(),
        Some("short_constraint_group")
    );
    assert_eq!(
        target_set.constraints[0].members,
        vec![vec![0, 1], vec![1, 2]]
    );
    assert_eq!(target_set.constraints[0].units, "nm");
    assert!((target_set.constraints[0].mean - 1.25).abs() < 1.0e-6);
    assert_eq!(target_set.constraints[0].samples, 4);
    assert!(target_set.bonds.is_empty());
}

#[test]
fn trajectory_provider_accepts_gromacs_topology_term_set() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("aa.xyz");
    fs::write(
        &traj,
        concat!(
            "4\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 1.0 0.0 0.0\n",
            "C 2.0 0.0 0.0\n",
            "C 3.0 1.0 0.0\n",
            "4\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 1.2 0.0 0.0\n",
            "C 2.2 0.2 0.0\n",
            "C 3.0 1.2 0.1\n",
        ),
    )
    .unwrap();
    let topology = r#"
[ moleculetype ]
  MOL 1

[ atoms ]
  1 P1 1 MOL A 1 0
  2 P2 1 MOL B 2 0
  3 P3 1 MOL C 3 0
  4 P4 1 MOL D 4 0

[ constraints ]
  1 2 1 0.47

[ bonds ]
  2 3 1 0.48 1000

[ angles ]
  1 2 3 2 150.0 100.0

[ dihedrals ]
  1 2 3 4 1 180.0 5.0 2
"#;
    let terms = BondedTermSet::from_gromacs_topology_str(topology, "MOL").unwrap();
    let mapping = BeadMapping {
        bead_names: vec![
            "B0".to_string(),
            "B1".to_string(),
            "B2".to_string(),
            "B3".to_string(),
        ],
        atom_indices: vec![vec![0], vec![1], vec![2], vec![3]],
    };
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory",
        &traj,
        NativeTrajectoryOptions::default(),
    );

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "topology_terms",
            out_dir: dir.path(),
            mapped_trajectory_name: None,
            mapping: &mapping,
            connections: &[],
            term_set: Some(&terms),
            metric_sources: &[],
            transform: None,
        })
        .unwrap();
    let target_set = data.target_set.as_ref().unwrap();

    assert_eq!(target_set.constraints.len(), 1);
    assert_eq!(target_set.bonds.len(), 1);
    assert_eq!(target_set.angles.len(), 1);
    assert_eq!(target_set.dihedrals.len(), 1);
    assert_eq!(target_set.constraints[0].members, vec![vec![0, 1]]);
    assert_eq!(target_set.bonds[0].members, vec![vec![1, 2]]);
    assert_eq!(target_set.angles[0].members, vec![vec![0, 1, 2]]);
    assert_eq!(target_set.dihedrals[0].members, vec![vec![0, 1, 2, 3]]);
    assert_eq!(target_set.constraints[0].units, "nm");
    assert_eq!(target_set.bonds[0].units, "nm");
}

#[test]
fn trajectory_provider_preserves_gromacs_groups_as_targets() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("aa.xyz");
    fs::write(
        &traj,
        concat!(
            "5\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 1.0 0.0 0.0\n",
            "C 2.0 0.0 0.0\n",
            "C 3.0 0.0 0.0\n",
            "C 4.0 0.0 0.0\n",
            "5\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 1.2 0.0 0.0\n",
            "C 2.4 0.0 0.0\n",
            "C 3.6 0.0 0.0\n",
            "C 4.8 0.0 0.0\n",
        ),
    )
    .unwrap();
    let topology = r#"
[ moleculetype ]
  MOL 1

[ atoms ]
  1 P1 1 MOL A 1 0
  2 P2 1 MOL B 2 0
  3 P3 1 MOL C 3 0
  4 P4 1 MOL D 4 0
  5 P5 1 MOL E 5 0

[ bonds ]
; bond group 1
  1 2 1 0.48 1000
  3 4 1 0.48 1000

; bond group 2
  4 5 1 0.50 1000
"#;
    let terms = BondedTermSet::from_gromacs_topology_str(topology, "MOL").unwrap();
    let mapping = BeadMapping {
        bead_names: vec![
            "B0".to_string(),
            "B1".to_string(),
            "B2".to_string(),
            "B3".to_string(),
            "B4".to_string(),
        ],
        atom_indices: vec![vec![0], vec![1], vec![2], vec![3], vec![4]],
    };
    let mut provider = TrajectoryReferenceProvider::new(
        "aa_trajectory",
        &traj,
        NativeTrajectoryOptions::default(),
    );

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "grouped_topology_terms",
            out_dir: dir.path(),
            mapped_trajectory_name: None,
            mapping: &mapping,
            connections: &[],
            term_set: Some(&terms),
            metric_sources: &[],
            transform: None,
        })
        .unwrap();
    let target_set = data.target_set.as_ref().unwrap();

    assert_eq!(target_set.bonds.len(), 2);
    assert_eq!(target_set.bonds[0].label.as_deref(), Some("bond group 1"));
    assert_eq!(target_set.bonds[0].members, vec![vec![0, 1], vec![2, 3]]);
    assert_eq!(target_set.bonds[0].units, "nm");
    assert!((target_set.bonds[0].mean - 1.1).abs() < 1.0e-6);
    assert_eq!(target_set.bonds[0].samples, 4);
    assert_eq!(target_set.bonds[1].members, vec![vec![3, 4]]);
    assert_eq!(target_set.bonds[1].units, "nm");
    assert_eq!(target_set.bonds[1].samples, 2);
}
