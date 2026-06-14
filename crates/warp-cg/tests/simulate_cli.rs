use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

fn temp_dir(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "warp_cg_simulate_cli_{label}_{}_{}",
        std::process::id(),
        nanos
    ));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

fn warp_cg() -> Command {
    Command::new(env!("CARGO_BIN_EXE_warp-cg"))
}

#[test]
fn simulate_example_validates_and_plans_gromacs_commands() {
    let example_output = warp_cg()
        .args(["simulate", "example", "--engine", "gromacs"])
        .output()
        .expect("example");
    assert!(example_output.status.success(), "{example_output:?}");
    let example: Value = serde_json::from_slice(&example_output.stdout).expect("example json");
    assert_eq!(example["engine"], "gromacs");

    let dir = temp_dir("plan");
    let request_path = dir.join("request.json");
    fs::write(
        &request_path,
        serde_json::to_string_pretty(&example).unwrap(),
    )
    .unwrap();

    let validate_output = warp_cg()
        .args(["simulate", "validate"])
        .arg(&request_path)
        .output()
        .expect("validate");
    assert!(validate_output.status.success(), "{validate_output:?}");
    let validation: Value =
        serde_json::from_slice(&validate_output.stdout).expect("validation json");
    assert_eq!(validation["valid"], true);

    let plan_output = warp_cg()
        .args(["simulate", "plan"])
        .arg(&request_path)
        .output()
        .expect("plan");
    assert!(plan_output.status.success(), "{plan_output:?}");
    let plan: Value = serde_json::from_slice(&plan_output.stdout).expect("plan json");
    assert_eq!(plan["engine"], "gromacs");
    assert_eq!(plan["commands"][0]["program"], "gmx");
    assert_eq!(plan["commands"][0]["args"][0], "grompp");
    assert!(plan["warnings"][0]["message"]
        .as_str()
        .unwrap()
        .contains("plan only"));
}

#[test]
fn simulate_validation_rejects_missing_explicit_protocol_parameters() {
    let dir = temp_dir("invalid");
    let request_path = dir.join("request.json");
    let request = json!({
        "schema_version": "warp-cg.simulate.v1",
        "engine": "gromacs",
        "system": {"coordinates": "system.gro", "topology": "topol.top"},
        "protocol": {
            "stages": [{
                "name": "nvt",
                "type": "md",
                "files": {"mdp": "nvt.mdp"}
            }]
        }
    });
    fs::write(
        &request_path,
        serde_json::to_string_pretty(&request).unwrap(),
    )
    .unwrap();

    let output = warp_cg()
        .args(["simulate", "validate"])
        .arg(&request_path)
        .output()
        .expect("validate");
    assert!(!output.status.success(), "{output:?}");
    let value: Value = serde_json::from_slice(&output.stdout).expect("validation json");
    assert_eq!(value["valid"], false);
    assert!(value["errors"][0]["message"]
        .as_str()
        .unwrap()
        .contains("does not provide scientific protocol defaults"));
}

#[test]
fn simulate_status_detects_checkpoints_and_failures() {
    let dir = temp_dir("status");
    fs::write(dir.join("nvt.cpt"), "checkpoint").unwrap();
    fs::write(dir.join("nvt.log"), "Finished mdrun").unwrap();
    fs::write(dir.join("prod.log"), "Fatal error: bad input").unwrap();

    let output = warp_cg()
        .args(["simulate", "status"])
        .arg(&dir)
        .output()
        .expect("status");
    assert!(!output.status.success(), "{output:?}");
    let value: Value = serde_json::from_slice(&output.stdout).expect("status json");
    assert_eq!(value["status"], "error");
    assert_eq!(value["restart_capable"], true);
    assert_eq!(value["last_checkpoint"], "nvt.cpt");
    assert_eq!(value["completed_stages"][0], "nvt");
    assert_eq!(value["failed_stages"][0], "prod");
}
