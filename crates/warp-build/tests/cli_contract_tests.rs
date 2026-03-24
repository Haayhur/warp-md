use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};
use warp_pack::io::{write_minimal_prmtop, AmberTopology};

fn temp_path(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "warp_build_cli_test_{}_{}_{}",
        label,
        std::process::id(),
        nanos
    ));
    path
}

fn write_text(path: &Path, text: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, text).expect("write file");
}

fn write_training_oligomer(path: &Path) {
    write_text(
        path,
        "ATOM      1  C1  HDA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  C2  RPT A   2       3.000   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C3  TLA A   3       6.000   0.000   0.000  1.00  0.00           C\n\
END\n",
    );
}

fn write_prmtop(path: &Path) {
    let topology = AmberTopology {
        atom_names: vec!["C1".into(), "C2".into(), "C3".into()],
        residue_labels: vec!["HDA".into(), "RPT".into(), "TLA".into()],
        residue_pointers: vec![1, 2, 3],
        atomic_numbers: vec![6, 6, 6],
        masses: vec![12.01, 12.01, 12.01],
        charges: vec![0.5, 1.0, 0.5],
        atom_type_indices: vec![1, 1, 1],
        amber_atom_types: vec!["CT".into(), "CT".into(), "CT".into()],
        radii: vec![1.7, 1.7, 1.7],
        screen: vec![0.8, 0.8, 0.8],
        bonds: vec![(0, 1), (1, 2)],
        bond_type_indices: vec![1, 1],
        bond_force_constants: vec![310.0],
        bond_equil_values: vec![1.53],
        angles: vec![[0, 1, 2]],
        angle_type_indices: vec![1],
        angle_force_constants: vec![55.0],
        angle_equil_values: vec![109.5],
        dihedrals: Vec::new(),
        dihedral_type_indices: Vec::new(),
        dihedral_force_constants: Vec::new(),
        dihedral_periodicities: Vec::new(),
        dihedral_phases: Vec::new(),
        scee_scale_factors: Vec::new(),
        scnb_scale_factors: Vec::new(),
        solty: vec![0.0],
        impropers: Vec::new(),
        improper_type_indices: Vec::new(),
        excluded_atoms: vec![vec![2], vec![1, 3], vec![2]],
        nonbonded_parm_index: vec![1],
        lennard_jones_acoef: vec![1.0],
        lennard_jones_bcoef: vec![0.5],
        lennard_jones_14_acoef: vec![0.8],
        lennard_jones_14_bcoef: vec![0.4],
        hbond_acoef: vec![0.0],
        hbond_bcoef: vec![0.0],
        hbcut: vec![0.0],
        tree_chain_classification: vec!["M".into(), "M".into(), "E".into()],
        join_array: vec![0, 0, 0],
        irotat: vec![0, 0, 0],
        solvent_pointers: Vec::new(),
        atoms_per_molecule: Vec::new(),
        box_dimensions: Vec::new(),
        radius_set: Some("modified Bondi radii".into()),
        ipol: 0,
    };
    write_minimal_prmtop(path.to_string_lossy().as_ref(), &topology).expect("write prmtop");
}

fn write_json(path: &Path, value: &Value) {
    write_text(
        path,
        &(serde_json::to_string_pretty(value).expect("json") + "\n"),
    );
}

fn make_bundle_dir(label: &str) -> (PathBuf, PathBuf) {
    let dir = temp_path(label);
    fs::create_dir_all(&dir).expect("create dir");
    let training = dir.join("training.pdb");
    let prmtop = dir.join("training.prmtop");
    let charges = dir.join("training_charge.json");
    let bundle = dir.join("bundle.json");
    write_training_oligomer(&training);
    write_prmtop(&prmtop);
    write_json(
        &charges,
        &json!({
            "version": "warp-build.charge-manifest.v1",
            "head_charge_e": 0.5,
            "repeat_charge_e": 1.0,
            "tail_charge_e": 0.5,
        }),
    );
    write_json(
        &bundle,
        &json!({
            "schema_version": "polymer-param-source.bundle.v1",
            "bundle_id": "pmma_param_bundle_v1",
            "training_context": {
                "mode": "oligomer_training",
                "training_oligomer_n": 3,
                "notes": "test"
            },
            "provenance": {},
            "unit_library": {
                "H": {
                    "display_name": "head_cap",
                    "junctions": {"head": "pmma_head_cap", "tail": "pmma_head_cap"},
                    "template_resname": "HDA"
                },
                "A": {
                    "display_name": "repeat",
                    "junctions": {"head": "pmma_head", "tail": "pmma_tail"},
                    "template_resname": "RPT"
                },
                "B": {
                    "display_name": "alternate",
                    "junctions": {"head": "pmma_head", "tail": "pmma_tail"},
                    "template_resname": "RPT"
                },
                "T": {
                    "display_name": "tail_cap",
                    "junctions": {"head": "pmma_tail_cap", "tail": "pmma_tail_cap"},
                    "template_resname": "TLA"
                }
            },
            "motif_library": {},
            "junction_library": {
                "pmma_head_cap": {
                    "attach_atom": {"scope": "unit", "selector": "name C1"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C1"}]
                },
                "pmma_head": {
                    "attach_atom": {"scope": "unit", "selector": "name C2"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C2"}]
                },
                "pmma_tail": {
                    "attach_atom": {"scope": "unit", "selector": "name C2"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C2"}]
                },
                "pmma_tail_cap": {
                    "attach_atom": {"scope": "unit", "selector": "name C3"},
                    "leaving_atoms": [],
                    "bond_order": 1,
                    "anchor_atoms": [{"scope": "unit", "selector": "name C3"}]
                }
            },
            "capabilities": {
                "supported_target_modes": ["linear_homopolymer", "linear_sequence_polymer"],
                "supported_conformation_modes": ["extended", "random_walk"],
                "supported_tacticity_modes": ["inherit", "isotactic", "syndiotactic", "atactic"],
                "supported_termini_policies": ["default", "source_default"],
                "sequence_token_support": {
                    "tokens": ["H", "A", "B", "T"],
                    "allowed_adjacencies": [["H", "A"], ["A", "A"], ["A", "T"]]
                },
                "charge_transfer_supported": true
            },
            "artifacts": {
                "source_coordinates": "training.pdb",
                "source_topology_ref": "training.prmtop",
                "forcefield_ref": "training.ffxml",
                "source_charge_manifest": "training_charge.json"
            },
            "charge_model": {}
        }),
    );
    (dir, bundle)
}

fn warp_build() -> Command {
    Command::new(env!("CARGO_BIN_EXE_warp-build"))
}

#[test]
fn schema_supports_yaml_and_out_path() {
    let out = temp_path("schema.yaml");
    let output = warp_build()
        .args(["schema", "--kind", "request", "--format", "yaml", "--out"])
        .arg(&out)
        .output()
        .expect("run warp-build schema");

    assert!(output.status.success(), "{output:?}");
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        out.display().to_string()
    );

    let written = fs::read_to_string(&out).expect("read schema output");
    let payload: Value = serde_yaml::from_str(&written).expect("parse yaml schema");
    assert!(payload["properties"].get("schema_version").is_some());
}

#[test]
fn capabilities_support_json_alias_and_json_output() {
    let output = warp_build()
        .args(["capabilities", "--json"])
        .output()
        .expect("run warp-build capabilities");

    assert!(output.status.success(), "{output:?}");
    let payload: Value = serde_json::from_slice(&output.stdout).expect("parse json capabilities");
    assert_eq!(
        payload["schema_versions"]["build_request"],
        "warp-build.agent.v1"
    );
}

#[test]
fn inspect_source_supports_yaml_output() {
    let (_dir, bundle) = make_bundle_dir("inspect_source_cli");
    let output = warp_build()
        .args(["inspect-source"])
        .arg(&bundle)
        .args(["--format", "yaml"])
        .output()
        .expect("run warp-build inspect-source");

    assert!(output.status.success(), "{output:?}");
    let payload: Value = serde_yaml::from_slice(&output.stdout).expect("parse yaml inspect-source");
    assert_eq!(payload["bundle_id"], "pmma_param_bundle_v1");
}

#[test]
fn validate_supports_yaml_and_preserves_invalid_exit_code() {
    let (_dir, bundle) = make_bundle_dir("validate_cli");
    let valid_request = temp_path("request_valid.json");
    write_json(
        &valid_request,
        &json!({
            "schema_version": "warp-build.agent.v1",
            "request_id": "cli-validate-001",
            "source_ref": {
                "bundle_id": "pmma_param_bundle_v1",
                "bundle_path": bundle.to_string_lossy(),
            },
            "target": {
                "mode": "linear_homopolymer",
                "repeat_unit": "A",
                "n_repeat": 4,
                "termini": {"head": "default", "tail": "default"},
                "stereochemistry": {"mode": "inherit"},
            },
            "realization": {
                "conformation_mode": "extended"
            },
            "artifacts": {
                "coordinates": temp_path("coords.pdb").to_string_lossy(),
                "build_manifest": temp_path("manifest.json").to_string_lossy(),
                "charge_manifest": temp_path("charge.json").to_string_lossy(),
            }
        }),
    );
    let valid_output = warp_build()
        .args(["validate"])
        .arg(&valid_request)
        .args(["--format", "yaml"])
        .output()
        .expect("run warp-build validate");

    assert!(valid_output.status.success(), "{valid_output:?}");
    let valid_payload: Value =
        serde_yaml::from_slice(&valid_output.stdout).expect("parse yaml validate");
    assert_eq!(valid_payload["schema_version"], "warp-build.agent.v1");

    let invalid_request = temp_path("request_invalid.json");
    write_json(
        &invalid_request,
        &json!({
            "schema_version": "warp-build.agent.v1"
        }),
    );
    let invalid_output = warp_build()
        .args(["validate"])
        .arg(&invalid_request)
        .args(["--json"])
        .output()
        .expect("run warp-build invalid validate");

    assert_eq!(invalid_output.status.code(), Some(2), "{invalid_output:?}");
    assert!(invalid_output.stderr.is_empty(), "{invalid_output:?}");
    let invalid_payload: Value =
        serde_json::from_slice(&invalid_output.stdout).expect("parse json invalid validate");
    assert!(invalid_payload["errors"]
        .as_array()
        .map(|items| !items.is_empty())
        .unwrap_or(false));
}

#[test]
fn example_bundle_out_materializes_runnable_example_flow() {
    let dir = temp_path("example_bundle_flow");
    fs::create_dir_all(&dir).expect("create example dir");
    let bundle = dir.join("source.bundle.json");
    let request = dir.join("request.json");

    let bundle_output = warp_build()
        .args(["example-bundle", "--out"])
        .arg(&bundle)
        .output()
        .expect("run warp-build example-bundle --out");
    assert!(bundle_output.status.success(), "{bundle_output:?}");
    assert_eq!(
        String::from_utf8_lossy(&bundle_output.stdout).trim(),
        bundle.display().to_string()
    );
    assert!(bundle.exists());
    assert!(dir.join("training.pdb").exists());
    assert!(dir.join("training.prmtop").exists());
    assert!(dir.join("training_charge.json").exists());

    let request_output = warp_build()
        .args(["example", "--mode", "random_walk", "--bundle-path"])
        .arg(&bundle)
        .output()
        .expect("run warp-build example");
    assert!(request_output.status.success(), "{request_output:?}");
    fs::write(&request, &request_output.stdout).expect("write request");

    let validate_output = warp_build()
        .args(["validate"])
        .arg(&request)
        .args(["--json"])
        .output()
        .expect("run warp-build validate example flow");
    assert!(validate_output.status.success(), "{validate_output:?}");
    let validate_payload: Value =
        serde_json::from_slice(&validate_output.stdout).expect("parse validate payload");
    assert_eq!(validate_payload["status"], "ok");

    let run_output = warp_build()
        .args(["run"])
        .arg(&request)
        .output()
        .expect("run warp-build example flow");
    assert!(run_output.status.success(), "{run_output:?}");
    let run_payload: Value = serde_json::from_slice(&run_output.stdout).expect("parse run payload");
    assert_eq!(run_payload["status"], "ok");
    assert_eq!(run_payload["schema_version"], "warp-build.agent.v1");
}
