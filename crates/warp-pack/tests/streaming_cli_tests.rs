use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

mod common;
use common::{temp_path, write_text};

fn unique_missing_parent(label: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    path.push(format!(
        "warp_pack_missing_parent_{label}_{}_{}",
        std::process::id(),
        nanos
    ));
    path
}

fn write_minimal_pdb(path: &PathBuf) {
    write_text(
        path,
        "ATOM      1  H   MOL A   1       0.000   0.000   0.000           H\nEND\n",
    );
}

fn write_minimal_config(config_path: &PathBuf, structure_path: &PathBuf) {
    let json = format!(
        "{{\"box\":{{\"size\":[10.0,10.0,10.0]}},\"structures\":[{{\"path\":\"{}\",\"count\":1}}]}}",
        structure_path.display()
    );
    write_text(config_path, &json);
}

#[test]
fn stream_does_not_emit_pack_complete_on_output_write_error() {
    let bin = env!("CARGO_BIN_EXE_warp-pack");
    let pdb_path = temp_path("stream_fail_input.pdb");
    let cfg_path = temp_path("stream_fail_config.json");
    write_minimal_pdb(&pdb_path);
    write_minimal_config(&cfg_path, &pdb_path);

    let missing_parent = unique_missing_parent("write_fail");
    let out_path = missing_parent.join("out.pdb");
    let _ = fs::remove_dir_all(&missing_parent);

    let output = Command::new(bin)
        .args([
            "--config",
            cfg_path.to_str().unwrap(),
            "--output",
            out_path.to_str().unwrap(),
            "--format",
            "pdb",
            "--stream",
        ])
        .output()
        .expect("failed to execute warp-pack");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !output.status.success(),
        "command should fail on write error"
    );
    assert!(
        stderr.contains("\"event\":\"pack_started\""),
        "expected stream start event in stderr: {stderr}"
    );
    assert!(
        !stderr.contains("\"event\":\"pack_complete\""),
        "pack_complete must not emit before output write succeeds: {stderr}"
    );

    let _ = fs::remove_file(&pdb_path);
    let _ = fs::remove_file(&cfg_path);
}

#[test]
fn stream_emits_pack_complete_after_successful_output_write() {
    let bin = env!("CARGO_BIN_EXE_warp-pack");
    let pdb_path = temp_path("stream_ok_input.pdb");
    let cfg_path = temp_path("stream_ok_config.json");
    let out_path = temp_path("stream_ok_output.pdb");
    write_minimal_pdb(&pdb_path);
    write_minimal_config(&cfg_path, &pdb_path);

    let output = Command::new(bin)
        .args([
            "--config",
            cfg_path.to_str().unwrap(),
            "--output",
            out_path.to_str().unwrap(),
            "--format",
            "pdb",
            "--stream",
        ])
        .output()
        .expect("failed to execute warp-pack");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "command should succeed: {stderr}");
    assert!(
        stderr.contains("\"event\":\"pack_complete\""),
        "expected pack_complete event on success: {stderr}"
    );
    assert!(
        out_path.exists(),
        "expected output file at {}",
        out_path.display()
    );

    let _ = fs::remove_file(&pdb_path);
    let _ = fs::remove_file(&cfg_path);
    let _ = fs::remove_file(&out_path);
}
