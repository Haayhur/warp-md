use std::fs;

use super::*;

#[test]
fn trajectory_target_extractor_builds_reusable_target_contract() {
    let dir = tempfile::tempdir().unwrap();
    let traj = dir.path().join("aa.xyz");
    fs::write(
        &traj,
        concat!(
            "3\n",
            "frame 0\n",
            "C 0.0 0.0 0.0\n",
            "C 1.0 0.0 0.0\n",
            "C 1.0 1.0 0.0\n",
            "3\n",
            "frame 1\n",
            "C 0.0 0.0 0.0\n",
            "C 1.5 0.0 0.0\n",
            "C 1.5 1.5 0.0\n",
        ),
    )
    .unwrap();
    let mapping = BeadMapping {
        bead_names: vec!["B0".to_string(), "B1".to_string(), "B2".to_string()],
        atom_indices: vec![vec![0], vec![1], vec![2]],
    };
    let terms = BondedTermSet::from_connections(3, &[(0, 1), (1, 2)]);
    let mut extractor = TrajectoryTargetExtractor::new(&traj, NativeTrajectoryOptions::default());

    let extraction = extractor
        .extract_targets(&TargetExtractionRequest {
            name: "candidate_or_reference",
            out_dir: dir.path(),
            mapped_trajectory_name: Some("mapped.gro"),
            mapping: &mapping,
            connections: &[],
            term_set: Some(&terms),
            transform: None,
        })
        .unwrap();

    assert_eq!(extraction.bonded_stats.bonds.len(), 2);
    assert_eq!(extraction.target_set.bonds.len(), 2);
    assert_eq!(extraction.target_set.bonds[0].members, vec![vec![0, 1]]);
    assert_eq!(extraction.target_set.bonds[1].members, vec![vec![1, 2]]);
    assert_eq!(extraction.metadata.frames_read, 2);
    assert_eq!(extraction.metadata.frames_written, 2);
    assert!(extraction.mapped_trajectory.as_ref().unwrap().is_file());
    assert!(extraction
        .artifacts
        .iter()
        .any(|artifact| artifact.kind == "reference_targets_json" && artifact.path.is_file()));
    assert_eq!(extraction.metrics["rg_samples"], 2.0);
}
