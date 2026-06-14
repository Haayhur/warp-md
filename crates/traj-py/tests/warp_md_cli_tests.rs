use std::fs;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

fn warp_md() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_warp-md"));
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("repo root");
    let python_src = repo_root.join("python");
    let mut paths = vec![python_src];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    let python_path = std::env::join_paths(paths).expect("join pythonpath");
    command.env("PYTHONPATH", python_path);
    command
}

fn parse_stdout(output: std::process::Output) -> Value {
    assert!(
        output.status.success(),
        "status={:?}\nstderr={}\nstdout={}",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout)
    );
    serde_json::from_slice(&output.stdout).expect("parse stdout json")
}

#[test]
fn schema_command_prints_request_schema() {
    let output = warp_md()
        .args(["schema", "--kind", "request"])
        .output()
        .expect("run warp-md schema");
    let payload = parse_stdout(output);
    assert_eq!(payload["title"], "RunRequest");
    assert!(payload["properties"].get("analyses").is_some());
    assert!(payload["properties"].get("inputs").is_some());
}

#[test]
fn validate_command_accepts_minimal_request() {
    let mut child = warp_md()
        .args(["validate", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn warp-md validate");
    let request = json!({
        "version": "warp-md.agent.v1",
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rg", "selection": "protein"}]
    });
    serde_json::to_writer(child.stdin.take().expect("stdin"), &request).expect("write stdin");
    let payload = parse_stdout(child.wait_with_output().expect("wait"));
    assert_eq!(payload["valid"], true);
    assert_eq!(
        payload["normalized_request"]["system"]["path"],
        "topology.pdb"
    );
    assert_eq!(payload["normalized_request"]["analyses"][0]["name"], "rg");
}

#[test]
fn validate_command_rejects_invalid_request() {
    let mut child = warp_md()
        .args(["validate", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn warp-md validate");
    let request = json!({
        "version": "warp-md.agent.v0",
        "system": "topology.pdb",
        "trajectory": "traj.xtc",
        "analyses": [{"name": "rg"}]
    });
    serde_json::to_writer(child.stdin.take().expect("stdin"), &request).expect("write stdin");
    let output = child.wait_with_output().expect("wait");
    assert_eq!(output.status.code(), Some(2));
    let payload: Value = serde_json::from_slice(&output.stdout).expect("parse stdout json");
    assert_eq!(payload["valid"], false);
    assert!(payload["errors"].as_array().expect("errors").len() >= 1);
}

#[test]
fn validate_command_reports_malformed_json_as_contract_error() {
    let mut child = warp_md()
        .args(["validate", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn warp-md validate");
    use std::io::Write;
    child
        .stdin
        .take()
        .expect("stdin")
        .write_all(br#"{"system":"topology.pdb""#)
        .expect("write stdin");
    let output = child.wait_with_output().expect("wait");
    assert_eq!(output.status.code(), Some(2));
    let payload: Value = serde_json::from_slice(&output.stdout).expect("parse stdout json");
    assert_eq!(payload["valid"], false);
    assert_eq!(payload["errors"][0]["code"], "E_SCHEMA_VALIDATION");
    assert!(payload.get("normalized_request").is_some());
}

#[test]
fn plan_schema_and_template_resolve_aliases() {
    let schema = parse_stdout(
        warp_md()
            .args(["plan-schema", "radius-of-gyration"])
            .output()
            .expect("run plan-schema"),
    );
    assert_eq!(schema["name"], "rg");
    assert!(schema["required_fields"]
        .as_array()
        .expect("required fields")
        .iter()
        .any(|value| value == "selection"));

    let template = parse_stdout(
        warp_md()
            .args(["template", "radius-of-gyration"])
            .output()
            .expect("run template"),
    );
    assert_eq!(template["analyses"][0]["name"], "rg");
    assert!(template["analyses"][0].get("selection").is_some());
}

#[test]
fn schema_command_writes_yaml_out_path() {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "warp_md_schema_{}_{}.yaml",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ));
    let output = warp_md()
        .args([
            "schema",
            "--kind",
            "plot-manifest",
            "--format",
            "yaml",
            "--out",
            path.to_str().expect("utf8 path"),
        ])
        .output()
        .expect("run schema yaml");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = fs::read_to_string(&path).expect("read schema");
    let payload: Value = serde_yaml::from_str(&text).expect("parse yaml");
    assert_eq!(payload["title"], "PlotManifest");
    let _ = fs::remove_file(path);
}

#[test]
fn run_command_delegates_to_python_runtime() {
    let output = warp_md()
        .args(["run", "--help"])
        .output()
        .expect("run delegated help");
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("--stream"));
    assert!(stdout.contains("--debug-errors"));
}
