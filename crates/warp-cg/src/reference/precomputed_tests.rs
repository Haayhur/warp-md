use std::fs;

use super::*;
use crate::parameters::BondStats;

#[test]
fn precomputed_provider_returns_reference_without_side_effects() {
    let mut provider = PrecomputedStatsReferenceProvider::new(
        "aa_statistics",
        BondedStats {
            bonds: vec![BondStats {
                bead_i: 0,
                bead_j: 1,
                mean: 0.42,
                std: 0.01,
                samples: 3,
            }],
            angles: Vec::new(),
            dihedrals: Vec::new(),
        },
    )
    .with_first_cg_coords(Some(vec![[0.0, 0.0, 0.0], [0.42, 0.0, 0.0]]));
    let dir = tempfile::tempdir().unwrap();
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "ethanol",
            out_dir: dir.path(),
            mapped_trajectory_name: Some("ignored.gro"),
            mapping: &mapping,
            connections: &[(0, 1)],
            term_set: None,
            metric_sources: &[],
            transform: None,
        })
        .unwrap();

    assert_eq!(data.source_kind, "aa_statistics");
    assert_eq!(data.bonded_stats.bonds.len(), 1);
    assert_eq!(data.bonded_stats.bonds[0].samples, 3);
    assert!(data.target_set.is_none());
    assert!(data.mapped_trajectory.is_none());
    assert!(data.artifacts.is_empty());
    assert_eq!(data.first_cg_coords.unwrap().len(), 2);
    assert_eq!(data.metadata.mapped_by, "precomputed_stats");
}

#[test]
fn precomputed_provider_loads_target_set_and_metric_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    let targets_path = dir.path().join("targets.json");
    let metrics_path = dir.path().join("metrics.json");
    let target_set = ReferenceTargetSet {
        version: 1,
        bin_config: ReferenceBinConfig::default(),
        constraints: Vec::new(),
        bonds: vec![ReferenceDistributionTarget {
            kind: ReferenceTermKind::Bond,
            label: Some("bond group 1".to_string()),
            beads: vec![0, 1],
            members: vec![vec![0, 1]],
            units: "nm".to_string(),
            periodic: false,
            mean: 0.47,
            std: 0.02,
            samples: 8,
            domain: [0.0, 3.0],
            bin_edges: vec![0.0, 1.0],
            probabilities: vec![1.0],
        }],
        angles: Vec::new(),
        dihedrals: Vec::new(),
    };
    fs::write(
        &targets_path,
        serde_json::to_vec_pretty(&target_set).unwrap(),
    )
    .unwrap();
    fs::write(
        &metrics_path,
        serde_json::json!({"metrics": {"rg_mean_nm": 1.2}}).to_string(),
    )
    .unwrap();
    let metric_sources = vec![ReferenceMetricSource {
        path: metrics_path.clone(),
        kind: "json".to_string(),
        namespace: None,
        artifact_kind: None,
    }];
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string()],
        atom_indices: vec![vec![0], vec![1]],
    };
    let mut provider =
        PrecomputedStatsReferenceProvider::new("precomputed_swarm_cg", BondedStats::default())
            .with_target_set_path(&targets_path);

    let data = provider
        .load_reference(&ReferenceRequest {
            name: "cached",
            out_dir: dir.path(),
            mapped_trajectory_name: None,
            mapping: &mapping,
            connections: &[(0, 1)],
            term_set: None,
            metric_sources: &metric_sources,
            transform: None,
        })
        .unwrap();

    assert_eq!(data.source_kind, "precomputed_swarm_cg");
    assert_eq!(data.target_set.as_ref().unwrap().bonds[0].mean, 0.47);
    assert_eq!(data.bonded_stats.bonds[0].mean, 0.47);
    assert_eq!(data.metrics["rg_mean_nm"], 1.2);
    assert!(
        data.artifacts
            .iter()
            .any(|artifact| artifact.kind == "reference_targets_json"
                && artifact.path == targets_path)
    );
    assert!(
        data.artifacts
            .iter()
            .any(|artifact| artifact.kind == "reference_metrics_json"
                && artifact.path == metrics_path)
    );
}
