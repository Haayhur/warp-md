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
        "warp_cg_build_cli_{label}_{}_{}",
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
fn build_run_emits_membrane_artifacts_and_charge_summary() {
    let dir = temp_dir("membrane");
    let request_path = dir.join("request.json");
    let gro_path = dir.join("membrane.gro");
    let top_path = dir.join("topol.top");
    let manifest_path = dir.join("manifest.json");
    let request = json!({
        "schema_version": "warp-cg.build.v1",
        "run_id": "cli-membrane",
        "mode": "membrane",
        "system": {"force_field": "martini3", "box_size_angstrom": [100.0, 100.0, 100.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [
                {"name": "upper", "side": "upper", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]},
                {"name": "lower", "side": "lower", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPG"}]}
            ]
        }],
        "environment": {"ions": {"neutralize": true, "cation": "Na+", "anion": "Cl-"}},
        "outputs": {
            "coordinates": gro_path,
            "topology": top_path,
            "manifest": manifest_path
        }
    });
    fs::write(
        &request_path,
        serde_json::to_string_pretty(&request).unwrap(),
    )
    .expect("request");

    let output = warp_cg()
        .args(["build", "run"])
        .arg(&request_path)
        .output()
        .expect("run warp-cg build");

    assert!(output.status.success(), "{output:?}");
    let value: Value = serde_json::from_slice(&output.stdout).expect("result json");
    assert_eq!(value["status"], "ok");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 200);
    assert_eq!(value["summary"]["lipid_counts"]["POPG"], 200);
    assert_eq!(value["summary"]["bead_count"], 4800);
    assert_eq!(
        value["charge"]["net_charge_before_neutralization_e"],
        -200.0
    );
    assert_eq!(value["charge"]["neutralization"]["counterion_count"], 200);
    assert_eq!(
        value["placement"]["algorithm"],
        "rectangular_grid_pair_edge_exclusion_relaxation"
    );
    assert_eq!(
        value["placement"]["leaflet_metrics"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert!(gro_path.exists());
    assert!(top_path.exists());
    assert!(manifest_path.exists());

    let gro = fs::read_to_string(&gro_path).expect("gro");
    assert!(gro.contains("POPC"));
    assert!(gro.contains("POPG"));
}
