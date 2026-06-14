use super::*;

#[test]
fn json_file_candidate_extraction_fields_are_validated() {
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
                    "command": {"program": "/bin/true"},
                    "candidate_extraction": {
                        "mapping": {
                            "bead_names": ["B0", "B1"],
                            "atom_indices": [[0], [1]]
                        },
                        "connections": [[0, 1]]
                    }
                }
            }
        }
    });

    let (exit_code, value) = validate_request_json(&base.to_string());
    assert_eq!(exit_code, 0, "{value}");
    assert_eq!(value["valid"], true);

    for (path, value, expected) in [
        (
            vec!["mapping", "bead_names"],
            json!([]),
            "mapping.bead_names",
        ),
        (
            vec!["mapping", "atom_indices"],
            json!([[0]]),
            "mapping.atom_indices length",
        ),
        (vec!["connections"], json!([[0, 9]]), "connections"),
        (vec!["stride"], json!(0), "stride"),
        (
            vec!["mapped_trajectory_name"],
            json!(""),
            "mapped_trajectory_name",
        ),
    ] {
        let mut request = base.clone();
        let mut field =
            &mut request["optimization"]["evaluator"]["json_file"]["candidate_extraction"];
        for segment in &path[..path.len() - 1] {
            field = &mut field[*segment];
        }
        field[path[path.len() - 1]] = value;
        let (exit_code, result) = validate_request_json(&request.to_string());

        assert_eq!(exit_code, 2);
        assert!(
            result["error"]["message"]
                .as_str()
                .unwrap()
                .contains(expected),
            "{result}"
        );
    }
}
