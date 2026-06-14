use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use crate::adapters::{multiwfn, orca, AdapterRun, AdapterSummary};
use crate::contract::{
    MoleculeSource, QmArtifact, QmEngineSpec, QmMoleculeSpec, QmOutputSpec, QmRequest,
    QmRuntimeSpec, QmTaskSpec, QmValidationSpec, QM_SCHEMA_VERSION,
};

pub fn run(request: &QmRequest) -> AdapterRun {
    if request.task.kind != "resp2_workflow" {
        return AdapterRun::error_code(
            "E_TASK_UNKNOWN",
            "workflow engine currently supports task.kind=resp2_workflow",
            2,
        );
    }
    let settings = &request.engine.settings;
    let qm_engine = string_setting(settings, "qm_engine").unwrap_or_else(|| "orca".into());
    let fit_engine = string_setting(settings, "fit_engine").unwrap_or_else(|| "multiwfn".into());
    if qm_engine != "orca" || fit_engine != "multiwfn" {
        return AdapterRun::error_code(
            "E_RESP2_WORKFLOW_ENGINE",
            "resp2_workflow currently supports qm_engine=orca and fit_engine=multiwfn",
            2,
        );
    }

    let root = PathBuf::from(&request.runtime.work_dir);
    if let Err(err) = fs::create_dir_all(&root) {
        return AdapterRun::error(
            format!("failed to create RESP2 workflow work dir: {err}"),
            2,
        );
    }

    let gas = run_orca_leg(request, "gas", settings.get("gas"));
    let mut artifacts = gas.artifacts.clone();
    let mut warnings = gas.warnings.clone();
    if gas.status != "ok" {
        return workflow_error("gas", gas, artifacts, warnings);
    }
    let solution = run_orca_leg(request, "solution", settings.get("solution"));
    artifacts.extend(solution.artifacts.clone());
    warnings.extend(solution.warnings.clone());
    if solution.status != "ok" {
        return workflow_error("solution", solution, artifacts, warnings);
    }

    let gas_molden = match find_artifact_path(&gas.artifacts, "molden_input")
        .or_else(|| find_file(&root.join("gas"), "job.molden.input"))
    {
        Some(path) => path,
        None => {
            return AdapterRun::error_code(
                "E_ORCA_MOLDEN_MISSING",
                "RESP2 workflow gas ORCA job did not produce job.molden.input",
                4,
            )
        }
    };
    let solution_molden = match find_artifact_path(&solution.artifacts, "molden_input")
        .or_else(|| find_file(&root.join("solution"), "job.molden.input"))
    {
        Some(path) => path,
        None => {
            return AdapterRun::error_code(
                "E_ORCA_MOLDEN_MISSING",
                "RESP2 workflow solution ORCA job did not produce job.molden.input",
                4,
            )
        }
    };

    let fit = run_multiwfn_fit(request, &root, &gas_molden, &solution_molden);
    artifacts.extend(fit.artifacts.clone());
    warnings.extend(fit.warnings.clone());
    if fit.status != "ok" {
        return workflow_error("fit", fit, artifacts, warnings);
    }

    let manifest_artifact = write_workflow_manifest(
        request,
        &root,
        &gas,
        &solution,
        &fit,
        &gas_molden,
        &solution_molden,
    );
    if let Some(artifact) = manifest_artifact {
        artifacts.push(artifact);
    }

    let mut properties = serde_json::Map::new();
    properties.insert(
        "resp2_workflow".into(),
        json!({
            "qm_engine": "orca",
            "fit_engine": "multiwfn",
            "formula": "q_resp2 = (1 - delta) * q_gas_resp + delta * q_solvent_resp",
            "delta": resp2_delta(settings),
            "gas_molden_input": gas_molden,
            "solution_molden_input": solution_molden,
        }),
    );

    AdapterRun {
        status: "ok".into(),
        exit_code: 0,
        command: None,
        artifacts,
        warnings,
        properties,
        summary: AdapterSummary::default(),
    }
}

fn run_orca_leg(request: &QmRequest, label: &str, leg_settings: Option<&Value>) -> AdapterRun {
    let mut sub = request.clone();
    sub.engine = QmEngineSpec {
        name: "orca".into(),
        mode: "cli".into(),
        executable: string_setting(&request.engine.settings, "orca_executable")
            .or_else(|| request.engine.executable.clone()),
        version_policy: request.engine.version_policy.clone(),
        settings: object_settings(leg_settings),
    };
    sub.engine.settings.insert("basename".into(), json!("job"));
    sub.engine
        .settings
        .insert("export_molden".into(), json!(true));
    for key in ["orca_2mkl_executable", "orca_directory"] {
        if let Some(value) = request.engine.settings.get(key) {
            sub.engine
                .settings
                .entry(key.into())
                .or_insert_with(|| value.clone());
        }
    }
    sub.task = QmTaskSpec {
        kind: "orca_molden_export".into(),
        method: string_from_value(leg_settings, "method")
            .unwrap_or_else(|| request.task.method.clone()),
        basis: string_from_value(leg_settings, "basis").or_else(|| request.task.basis.clone()),
        charge_model: None,
        properties: vec!["energy".into()],
        convergence: request.task.convergence.clone(),
    };
    sub.runtime.work_dir = PathBuf::from(&request.runtime.work_dir)
        .join(label)
        .to_string_lossy()
        .into_owned();
    sub.output.out_dir = sub.runtime.work_dir.clone();
    orca::run(&sub)
}

fn run_multiwfn_fit(
    request: &QmRequest,
    root: &Path,
    gas_molden: &str,
    solution_molden: &str,
) -> AdapterRun {
    let mut settings = BTreeMap::new();
    settings.insert("gas_input_file".into(), json!(gas_molden));
    settings.insert("solvent_input_file".into(), json!(solution_molden));
    settings.insert("delta".into(), json!(resp2_delta(&request.engine.settings)));
    for key in ["charge_projection", "lib_dir", "multiwfn_path"] {
        if let Some(value) = request.engine.settings.get(key) {
            settings.insert(key.into(), value.clone());
        }
    }
    let molecule_path = request
        .molecule
        .source
        .path
        .clone()
        .unwrap_or_else(|| gas_molden.into());
    let sub = QmRequest {
        schema_version: QM_SCHEMA_VERSION.into(),
        request_id: request.request_id.clone(),
        engine: QmEngineSpec {
            name: "multiwfn".into(),
            mode: "cli".into(),
            executable: string_setting(&request.engine.settings, "multiwfn_executable"),
            version_policy: request.engine.version_policy.clone(),
            settings,
        },
        molecule: QmMoleculeSpec {
            source: MoleculeSource {
                kind: "file".into(),
                path: Some(molecule_path),
                format: Some("molden".into()),
                topology: None,
                trajectory: None,
                selection: None,
                environment_selection: None,
                frames: Vec::new(),
            },
            charge: request.molecule.charge,
            multiplicity: request.molecule.multiplicity,
            units: request.molecule.units.clone(),
        },
        task: QmTaskSpec {
            kind: "resp_fit".into(),
            method: "multiwfn".into(),
            basis: None,
            charge_model: Some("resp2".into()),
            properties: vec!["charges".into()],
            convergence: request.task.convergence.clone(),
        },
        runtime: QmRuntimeSpec {
            work_dir: root.join("fit").to_string_lossy().into_owned(),
            threads: request.runtime.threads,
            memory_mb: request.runtime.memory_mb,
            scratch_dir: request.runtime.scratch_dir.clone(),
            keep_raw: request.runtime.keep_raw,
        },
        output: QmOutputSpec {
            out_dir: root.join("fit").to_string_lossy().into_owned(),
            ..request.output.clone()
        },
        validation: QmValidationSpec::default(),
    };
    multiwfn::run(&sub)
}

fn write_workflow_manifest(
    request: &QmRequest,
    root: &Path,
    gas: &AdapterRun,
    solution: &AdapterRun,
    fit: &AdapterRun,
    gas_molden: &str,
    solution_molden: &str,
) -> Option<QmArtifact> {
    let path = root.join("workflow_manifest.json");
    let manifest = json!({
        "schema_version": "warp-qm.resp2-workflow-manifest.v1",
        "request_id": request.request_id,
        "schema_contract": QM_SCHEMA_VERSION,
        "status": "ok",
        "task": "resp2_workflow",
        "qm_engine": "orca",
        "fit_engine": "multiwfn",
        "delta": resp2_delta(&request.engine.settings),
        "formula": "q_resp2 = (1 - delta) * q_gas_resp + delta * q_solvent_resp",
        "gas": {
            "work_dir": root.join("gas").to_string_lossy(),
            "molden_input": gas_molden,
            "artifacts": gas.artifacts.clone()
        },
        "solution": {
            "work_dir": root.join("solution").to_string_lossy(),
            "molden_input": solution_molden,
            "artifacts": solution.artifacts.clone()
        },
        "fit": {
            "work_dir": root.join("fit").to_string_lossy(),
            "charge_manifest": root.join("fit").join("charge_manifest.json").to_string_lossy(),
            "artifacts": fit.artifacts.clone()
        }
    });
    fs::write(&path, serde_json::to_string_pretty(&manifest).ok()?).ok()?;
    Some(QmArtifact {
        path: path.to_string_lossy().into_owned(),
        format: "json".into(),
        kind: "workflow_manifest".into(),
    })
}

fn workflow_error(
    stage: &str,
    run: AdapterRun,
    mut artifacts: Vec<QmArtifact>,
    mut warnings: Vec<String>,
) -> AdapterRun {
    warnings.push(format!("RESP2 workflow {stage} stage failed"));
    artifacts.extend(run.artifacts);
    AdapterRun {
        status: "error".into(),
        exit_code: run.exit_code,
        command: run.command,
        artifacts,
        warnings,
        properties: run.properties,
        summary: run.summary,
    }
}

fn object_settings(value: Option<&Value>) -> BTreeMap<String, Value> {
    value
        .and_then(Value::as_object)
        .map(|map| map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
        .unwrap_or_default()
}

fn string_setting(settings: &BTreeMap<String, Value>, key: &str) -> Option<String> {
    settings
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn string_from_value(value: Option<&Value>, key: &str) -> Option<String> {
    value
        .and_then(|value| value.get(key))
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn resp2_delta(settings: &BTreeMap<String, Value>) -> f64 {
    settings
        .get("resp2")
        .and_then(|value| value.get("delta"))
        .and_then(Value::as_f64)
        .or_else(|| settings.get("delta").and_then(Value::as_f64))
        .unwrap_or(0.5)
}

fn find_artifact_path(artifacts: &[QmArtifact], kind: &str) -> Option<String> {
    artifacts
        .iter()
        .find(|artifact| artifact.kind == kind)
        .map(|artifact| artifact.path.clone())
}

fn find_file(dir: &Path, name: &str) -> Option<String> {
    let path = dir.join(name);
    path.exists().then(|| path.to_string_lossy().into_owned())
}
