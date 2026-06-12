use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

use crate::adapters::{AdapterRun, AdapterSummary};
use crate::contract::{
    ChargeManifest, JobManifest, QmArtifact, QmRequest, CHARGE_MANIFEST_VERSION,
    JOB_MANIFEST_VERSION, QM_SCHEMA_VERSION,
};
use crate::engines::find_executable;
use crate::molecule::QmMolecule;
use crate::parsers;

pub fn run(request: &QmRequest) -> AdapterRun {
    match request.task.kind.as_str() {
        "generic_run" => return run_generic(request),
        "binding_energy" => return run_energy_difference(request, "binding_energy", "fragments"),
        "solvation_energy" => {
            return run_pair_difference(request, "solvation_energy", "gas", "solution")
        }
        "proton_affinity" => {
            return run_pair_difference(request, "proton_affinity", "deprotonated", "protonated")
        }
        _ => {}
    }
    run_standard(request)
}

fn run_generic(request: &QmRequest) -> AdapterRun {
    let work_dir = PathBuf::from(&request.runtime.work_dir);
    if let Err(err) = fs::create_dir_all(&work_dir) {
        return AdapterRun::error(format!("failed to create ORCA work dir: {err}"), 2);
    }

    let (input_path, basename) = match prepare_generic_input(request, &work_dir) {
        Ok(value) => value,
        Err(err) => return AdapterRun::error(err, 2),
    };

    let mut artifacts = vec![artifact(&input_path, "text", "engine_input")];
    if let Ok(manifest_artifact) = write_generic_job_manifest(request, &work_dir, &artifacts) {
        artifacts.push(manifest_artifact);
    }

    let executable = match resolve_orca_executable(request) {
        Some(path) => path,
        None => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: None,
                artifacts,
                warnings: vec![
                    "ORCA executable not found; set engine.executable, WARP_QM_ORCA, ORCA_BINARY, or PATH"
                        .into(),
                ],
                properties: serde_json::Map::new(),
                summary: AdapterSummary::default(),
            }
        }
    };

    let input_name = match input_path.file_name() {
        Some(name) => name.to_string_lossy().into_owned(),
        None => return AdapterRun::error("generic ORCA input path has no file name", 2),
    };
    let command_vec = vec![executable.clone(), input_name.clone()];
    let output = Command::new(&executable)
        .arg(&input_name)
        .current_dir(&work_dir)
        .output();
    let output = match output {
        Ok(output) => output,
        Err(err) => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: Some(command_vec),
                artifacts,
                warnings: vec![format!("failed to execute ORCA: {err}")],
                properties: serde_json::Map::new(),
                summary: AdapterSummary::default(),
            }
        }
    };

    let out_path = work_dir.join(format!("{basename}.out"));
    let err_path = work_dir.join(format!("{basename}.err"));
    let _ = fs::write(&out_path, &output.stdout);
    let _ = fs::write(&err_path, &output.stderr);
    if out_path.exists() {
        artifacts.push(artifact(&out_path, "text", "engine_output"));
    }
    if err_path.exists() && err_path.metadata().map(|m| m.len() > 0).unwrap_or(false) {
        artifacts.push(artifact(&err_path, "text", "engine_stderr"));
    }
    collect_orca_sidecars(&work_dir, &basename, &mut artifacts);
    if request
        .engine
        .settings
        .get("export_molden")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        match run_orca_2mkl(&executable, &work_dir, &basename) {
            Ok(Some(molden)) => artifacts.push(artifact(&molden, "molden", "molden_input")),
            Ok(None) => {}
            Err(err) => {
                return AdapterRun {
                    status: "error".into(),
                    exit_code: 4,
                    command: Some(command_vec),
                    artifacts,
                    warnings: vec![err],
                    properties: serde_json::Map::new(),
                    summary: AdapterSummary::default(),
                }
            }
        }
    }

    let inspection = parsers::inspect_output(&out_path, "orca").ok();
    let mut properties = serde_json::Map::new();
    if let Some(report) = inspection.as_ref() {
        properties.insert("inspection".into(), json!(report));
    }
    let converged = inspection
        .as_ref()
        .map(|report| report.fatal_errors.is_empty() && report.convergence_status.is_some());
    let status = if output.status.success() && converged.unwrap_or(false) {
        "ok"
    } else {
        "error"
    };

    AdapterRun {
        status: status.into(),
        exit_code: if status == "ok" { 0 } else { 4 },
        command: Some(command_vec),
        artifacts,
        warnings: Vec::new(),
        properties,
        summary: AdapterSummary {
            energy_hartree: inspection.and_then(|report| report.final_energy_hartree),
            converged,
            n_atoms: None,
        },
    }
}

fn run_standard(request: &QmRequest) -> AdapterRun {
    let work_dir = PathBuf::from(&request.runtime.work_dir);
    if let Err(err) = fs::create_dir_all(&work_dir) {
        return AdapterRun::error(format!("failed to create ORCA work dir: {err}"), 2);
    }

    let molecule = match QmMolecule::from_request(request) {
        Ok(molecule) => molecule,
        Err(err) => return AdapterRun::error(err, 2),
    };

    let basename = request
        .engine
        .settings
        .get("basename")
        .and_then(Value::as_str)
        .unwrap_or("job");
    let input_path = work_dir.join(format!("{basename}.inp"));
    let input_text = render_input(request, &molecule);
    if let Err(err) = fs::write(&input_path, input_text) {
        return AdapterRun::error(format!("failed to write ORCA input: {err}"), 2);
    }

    let mut artifacts = vec![artifact(&input_path, "text", "engine_input")];
    if let Ok(manifest_artifact) = write_job_manifest(request, &molecule, &work_dir, &artifacts) {
        artifacts.push(manifest_artifact);
    }

    let executable = match resolve_orca_executable(request) {
        Some(path) => path,
        None => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: None,
                artifacts,
                warnings: vec![
                    "ORCA executable not found; set engine.executable, WARP_QM_ORCA, ORCA_BINARY, or PATH"
                        .into(),
                ],
                properties: serde_json::Map::new(),
                summary: AdapterSummary {
                    n_atoms: Some(molecule.atom_count()),
                    ..AdapterSummary::default()
                },
            }
        }
    };

    let command_vec = vec![
        executable.clone(),
        input_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned(),
    ];
    let output = Command::new(&executable)
        .arg(input_path.file_name().unwrap())
        .current_dir(&work_dir)
        .output();
    let output = match output {
        Ok(output) => output,
        Err(err) => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: Some(command_vec),
                artifacts,
                warnings: vec![format!("failed to execute ORCA: {err}")],
                properties: serde_json::Map::new(),
                summary: AdapterSummary {
                    n_atoms: Some(molecule.atom_count()),
                    ..AdapterSummary::default()
                },
            }
        }
    };

    let out_path = work_dir.join(format!("{basename}.out"));
    let err_path = work_dir.join(format!("{basename}.err"));
    let _ = fs::write(&out_path, &output.stdout);
    let _ = fs::write(&err_path, &output.stderr);
    if out_path.exists() {
        artifacts.push(artifact(&out_path, "text", "engine_output"));
    }
    if err_path.exists() && err_path.metadata().map(|m| m.len() > 0).unwrap_or(false) {
        artifacts.push(artifact(&err_path, "text", "engine_stderr"));
    }
    collect_orca_sidecars(&work_dir, basename, &mut artifacts);
    if request.task.kind == "orca_molden_export"
        || request
            .engine
            .settings
            .get("export_molden")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    {
        match run_orca_2mkl(&executable, &work_dir, basename) {
            Ok(Some(molden)) => artifacts.push(artifact(&molden, "molden", "molden_input")),
            Ok(None) => {}
            Err(err) => {
                return AdapterRun {
                    status: "error".into(),
                    exit_code: 4,
                    command: Some(command_vec),
                    artifacts,
                    warnings: vec![err],
                    properties: serde_json::Map::new(),
                    summary: AdapterSummary {
                        n_atoms: Some(molecule.atom_count()),
                        ..AdapterSummary::default()
                    },
                }
            }
        }
    }

    let inspection = parsers::inspect_output(&out_path, "orca").ok();
    let mut properties = serde_json::Map::new();
    if let Some(report) = inspection.as_ref() {
        properties.insert("inspection".into(), json!(report));
    }
    if request.task.kind == "charges" {
        if let Some(report) = inspection.as_ref() {
            if let Some(charge_manifest) =
                write_charge_manifest(request, &work_dir, basename, report)
            {
                artifacts.push(charge_manifest);
            }
        }
    }
    let converged = inspection
        .as_ref()
        .map(|report| report.fatal_errors.is_empty() && report.convergence_status.is_some());
    let status = if output.status.success() && converged.unwrap_or(false) {
        "ok"
    } else {
        "error"
    };

    AdapterRun {
        status: status.into(),
        exit_code: if status == "ok" { 0 } else { 4 },
        command: Some(command_vec),
        artifacts,
        warnings: Vec::new(),
        properties,
        summary: AdapterSummary {
            energy_hartree: inspection.and_then(|report| report.final_energy_hartree),
            converged,
            n_atoms: Some(molecule.atom_count()),
        },
    }
}

fn prepare_generic_input(
    request: &QmRequest,
    work_dir: &Path,
) -> Result<(PathBuf, String), String> {
    let basename = request
        .engine
        .settings
        .get("basename")
        .and_then(Value::as_str)
        .unwrap_or("job");
    let input_name = request
        .engine
        .settings
        .get("input_name")
        .and_then(Value::as_str)
        .and_then(|name| Path::new(name).file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| format!("{basename}.inp"));
    let input_path = work_dir.join(&input_name);
    if let Some(text) = request
        .engine
        .settings
        .get("input_text")
        .and_then(Value::as_str)
    {
        fs::write(&input_path, text)
            .map_err(|err| format!("failed to write generic ORCA input: {err}"))?;
    } else if let Some(path) = request
        .engine
        .settings
        .get("input_file")
        .and_then(Value::as_str)
    {
        let source = Path::new(path);
        if !source.exists() {
            return Err(format!("generic ORCA input file not found: {path}"));
        }
        if source != input_path {
            fs::copy(source, &input_path)
                .map_err(|err| format!("failed to copy generic ORCA input: {err}"))?;
        }
    } else {
        return Err(
            "generic_run requires engine.settings.input_file or engine.settings.input_text".into(),
        );
    }
    let stem = input_path
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| basename.into());
    Ok((input_path, stem))
}

pub fn render_input(request: &QmRequest, molecule: &QmMolecule) -> String {
    let mut out = String::new();
    let keywords = orca_keywords(request);
    if !keywords.is_empty() {
        out.push('!');
        out.push(' ');
        out.push_str(&keywords.join(" "));
        out.push('\n');
    }
    out.push_str("! NormalPrint\n");
    if request.output.write_json {
        out.push_str("%output\n    jsonpropfile true\n    jsongbwfile true\nend\n");
    }
    if let Some(memory_mb) = request.runtime.memory_mb {
        let per_core = request
            .runtime
            .threads
            .map(|threads| (memory_mb / u64::from(threads.max(1))).max(1))
            .unwrap_or(memory_mb);
        out.push_str(&format!("%maxcore {per_core}\n"));
    }
    if let Some(threads) = request.runtime.threads {
        out.push_str(&format!("%pal\n    nprocs {threads}\nend\n"));
    }
    if let Some(moinp) = request.engine.settings.get("moinp").and_then(Value::as_str) {
        out.push_str(&format!("%moinp \"{moinp}\"\n"));
    }
    if let Some(blocks) = request
        .engine
        .settings
        .get("blocks")
        .and_then(Value::as_object)
    {
        for (name, value) in blocks {
            out.push('\n');
            out.push_str(&render_block(name, value));
            out.push('\n');
        }
    }
    out.push('\n');
    out.push_str(&molecule.to_orca_xyz_block());
    out.push('\n');
    out
}

fn orca_keywords(request: &QmRequest) -> Vec<String> {
    let mut keywords = Vec::new();
    keywords.push(request.task.method.clone());
    if let Some(basis) = request.task.basis.as_ref() {
        keywords.push(basis.clone());
    }
    match request.task.kind.as_str() {
        "optimize" => keywords.push(
            request
                .engine
                .settings
                .get("opt")
                .and_then(Value::as_str)
                .unwrap_or("Opt")
                .into(),
        ),
        "frequency" => keywords.push("Freq".into()),
        "nmr_shielding" => keywords.push("NMR".into()),
        "charges" => {
            if request
                .task
                .charge_model
                .as_deref()
                .unwrap_or("hirshfeld")
                .eq_ignore_ascii_case("hirshfeld")
            {
                keywords.push("Hirshfeld".into());
            }
        }
        _ => {}
    }
    // ORCA 6.1 rejects several legacy population-analysis simple keywords used by older wrappers.
    // Keep charge extraction parser-side unless the caller explicitly provides engine.settings.keywords.
    if let Some(extra) = request
        .engine
        .settings
        .get("keywords")
        .and_then(Value::as_array)
    {
        for item in extra {
            if let Some(value) = item.as_str() {
                keywords.push(value.into());
            }
        }
    }
    keywords
}

fn run_energy_difference(request: &QmRequest, workflow: &str, fragments_key: &str) -> AdapterRun {
    let fragments = match request
        .engine
        .settings
        .get(fragments_key)
        .and_then(Value::as_array)
    {
        Some(values) if !values.is_empty() => values,
        _ => {
            return AdapterRun::error(
                format!("{workflow} requires engine.settings.{fragments_key} as a non-empty array"),
                2,
            )
        }
    };
    let complex = run_named_single_point(request, "complex", &request.molecule);
    let mut artifacts = complex.artifacts.clone();
    let mut warnings = complex.warnings.clone();
    if complex.status != "ok" {
        return derived_error(workflow, complex, artifacts, warnings);
    }
    let complex_energy = match complex.summary.energy_hartree {
        Some(value) => value,
        None => return AdapterRun::error(format!("{workflow} complex energy was not parsed"), 4),
    };

    let mut fragment_energies = Vec::new();
    for (idx, fragment) in fragments.iter().enumerate() {
        let label = fragment
            .get("label")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("fragment_{}", idx + 1));
        let molecule = match molecule_from_settings(fragment, &request.molecule) {
            Ok(molecule) => molecule,
            Err(err) => return AdapterRun::error(err, 2),
        };
        let run = run_named_single_point(request, &label, &molecule);
        artifacts.extend(run.artifacts.clone());
        warnings.extend(run.warnings.clone());
        if run.status != "ok" {
            return derived_error(workflow, run, artifacts, warnings);
        }
        match run.summary.energy_hartree {
            Some(energy) => {
                fragment_energies.push(json!({"label": label, "energy_hartree": energy}))
            }
            None => {
                return AdapterRun::error(
                    format!("{workflow} fragment '{label}' energy was not parsed"),
                    4,
                )
            }
        }
    }
    let fragment_sum: f64 = fragment_energies
        .iter()
        .filter_map(|item| item.get("energy_hartree").and_then(Value::as_f64))
        .sum();
    let delta = complex_energy - fragment_sum;
    let mut properties = serde_json::Map::new();
    properties.insert(
        workflow.into(),
        json!({
            "formula": "E_complex - sum(E_fragments)",
            "complex_energy_hartree": complex_energy,
            "fragment_energies": fragment_energies,
            "delta_hartree": delta,
            "delta_kcal_mol": delta * 627.509474,
        }),
    );
    AdapterRun {
        status: "ok".into(),
        exit_code: 0,
        command: None,
        artifacts,
        warnings,
        properties,
        summary: AdapterSummary {
            energy_hartree: Some(delta),
            converged: Some(true),
            n_atoms: complex.summary.n_atoms,
        },
    }
}

fn run_pair_difference(
    request: &QmRequest,
    workflow: &str,
    reference_key: &str,
    target_key: &str,
) -> AdapterRun {
    let reference = match molecule_setting_or_request(request, reference_key) {
        Ok(molecule) => molecule,
        Err(err) => return AdapterRun::error(err, 2),
    };
    let target = match molecule_setting(request, target_key) {
        Ok(molecule) => molecule,
        Err(err) => return AdapterRun::error(err, 2),
    };
    let reference_run = run_named_single_point(request, reference_key, &reference);
    let mut artifacts = reference_run.artifacts.clone();
    let mut warnings = reference_run.warnings.clone();
    if reference_run.status != "ok" {
        return derived_error(workflow, reference_run, artifacts, warnings);
    }
    let target_run = run_named_single_point(request, target_key, &target);
    artifacts.extend(target_run.artifacts.clone());
    warnings.extend(target_run.warnings.clone());
    if target_run.status != "ok" {
        return derived_error(workflow, target_run, artifacts, warnings);
    }
    let reference_energy = match reference_run.summary.energy_hartree {
        Some(value) => value,
        None => {
            return AdapterRun::error(
                format!("{workflow} {reference_key} energy was not parsed"),
                4,
            )
        }
    };
    let target_energy = match target_run.summary.energy_hartree {
        Some(value) => value,
        None => {
            return AdapterRun::error(format!("{workflow} {target_key} energy was not parsed"), 4)
        }
    };
    let delta = target_energy - reference_energy;
    let mut properties = serde_json::Map::new();
    properties.insert(
        workflow.into(),
        json!({
            "formula": format!("E_{target_key} - E_{reference_key}"),
            "reference": reference_key,
            "target": target_key,
            "reference_energy_hartree": reference_energy,
            "target_energy_hartree": target_energy,
            "delta_hartree": delta,
            "delta_kcal_mol": delta * 627.509474,
        }),
    );
    AdapterRun {
        status: "ok".into(),
        exit_code: 0,
        command: None,
        artifacts,
        warnings,
        properties,
        summary: AdapterSummary {
            energy_hartree: Some(delta),
            converged: Some(true),
            n_atoms: reference_run.summary.n_atoms,
        },
    }
}

fn run_named_single_point(
    request: &QmRequest,
    label: &str,
    molecule: &crate::contract::QmMoleculeSpec,
) -> AdapterRun {
    let mut sub = request.clone();
    sub.task.kind = "single_point".into();
    sub.molecule = molecule.clone();
    sub.runtime.work_dir = PathBuf::from(&request.runtime.work_dir)
        .join(label)
        .to_string_lossy()
        .into_owned();
    sub.output.out_dir = sub.runtime.work_dir.clone();
    sub.engine
        .settings
        .insert("basename".into(), json!(sanitize_label(label)));
    run_standard(&sub)
}

fn molecule_setting_or_request(
    request: &QmRequest,
    key: &str,
) -> Result<crate::contract::QmMoleculeSpec, String> {
    match request.engine.settings.get(key) {
        Some(value) => molecule_from_settings(value, &request.molecule),
        None => Ok(request.molecule.clone()),
    }
}

fn molecule_setting(
    request: &QmRequest,
    key: &str,
) -> Result<crate::contract::QmMoleculeSpec, String> {
    let value = request
        .engine
        .settings
        .get(key)
        .ok_or_else(|| format!("{} requires engine.settings.{key}", request.task.kind))?;
    molecule_from_settings(value, &request.molecule)
}

fn molecule_from_settings(
    value: &Value,
    defaults: &crate::contract::QmMoleculeSpec,
) -> Result<crate::contract::QmMoleculeSpec, String> {
    let path = value
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| "derived ORCA molecule settings require path".to_string())?;
    let charge = value
        .get("charge")
        .and_then(Value::as_i64)
        .map(|v| v as i32)
        .unwrap_or(defaults.charge);
    let multiplicity = value
        .get("multiplicity")
        .and_then(Value::as_u64)
        .map(|v| v as u32)
        .unwrap_or(defaults.multiplicity);
    Ok(crate::contract::QmMoleculeSpec {
        source: crate::contract::MoleculeSource {
            kind: "file".into(),
            path: Some(path.into()),
            format: value
                .get("format")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| defaults.source.format.clone()),
            topology: None,
            trajectory: None,
            selection: None,
            environment_selection: None,
            frames: Vec::new(),
        },
        charge,
        multiplicity,
        units: value
            .get("units")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| defaults.units.clone()),
    })
}

fn derived_error(
    workflow: &str,
    run: AdapterRun,
    artifacts: Vec<QmArtifact>,
    mut warnings: Vec<String>,
) -> AdapterRun {
    warnings.push(format!("{workflow} sub-job failed"));
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

fn sanitize_label(label: &str) -> String {
    label
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn render_block(name: &str, value: &Value) -> String {
    let mut out = format!("%{name}\n");
    match value {
        Value::Array(lines) => {
            for line in lines {
                if let Some(line) = line.as_str() {
                    out.push_str("    ");
                    out.push_str(line);
                    out.push('\n');
                }
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                out.push_str("    ");
                out.push_str(key);
                out.push(' ');
                out.push_str(&block_value(value));
                out.push('\n');
            }
        }
        Value::String(raw) => {
            for line in raw.lines() {
                out.push_str("    ");
                out.push_str(line);
                out.push('\n');
            }
        }
        _ => {}
    }
    out.push_str("end");
    out
}

fn block_value(value: &Value) -> String {
    match value {
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}

fn resolve_orca_executable(request: &QmRequest) -> Option<String> {
    request
        .engine
        .executable
        .clone()
        .or_else(|| std::env::var("WARP_QM_ORCA").ok())
        .or_else(|| std::env::var("ORCA_BINARY").ok())
        .or_else(|| find_executable("orca"))
}

fn write_job_manifest(
    request: &QmRequest,
    molecule: &QmMolecule,
    work_dir: &Path,
    artifacts: &[QmArtifact],
) -> Result<QmArtifact, String> {
    let manifest = JobManifest {
        schema_version: JOB_MANIFEST_VERSION.into(),
        request_id: request.request_id.clone(),
        engine: request.engine.clone(),
        task: request.task.clone(),
        molecule: request.molecule.clone(),
        artifacts: artifacts.to_vec(),
        provenance: BTreeMap::from([
            ("tool".into(), json!("warp-qm")),
            ("schema_version".into(), json!(QM_SCHEMA_VERSION)),
            ("molecule".into(), molecule.provenance()),
            (
                "orca_executable".into(),
                json!(resolve_orca_executable(request)),
            ),
        ]),
    };
    let path = work_dir.join("job_manifest.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?,
    )
    .map_err(|err| err.to_string())?;
    Ok(artifact(&path, "json", "job_manifest"))
}

fn write_generic_job_manifest(
    request: &QmRequest,
    work_dir: &Path,
    artifacts: &[QmArtifact],
) -> Result<QmArtifact, String> {
    let manifest = JobManifest {
        schema_version: JOB_MANIFEST_VERSION.into(),
        request_id: request.request_id.clone(),
        engine: request.engine.clone(),
        task: request.task.clone(),
        molecule: request.molecule.clone(),
        artifacts: artifacts.to_vec(),
        provenance: BTreeMap::from([
            ("tool".into(), json!("warp-qm")),
            ("schema_version".into(), json!(QM_SCHEMA_VERSION)),
            ("input_mode".into(), json!("generic_engine_input")),
            (
                "orca_executable".into(),
                json!(resolve_orca_executable(request)),
            ),
        ]),
    };
    let path = work_dir.join("job_manifest.json");
    fs::write(
        &path,
        serde_json::to_string_pretty(&manifest).map_err(|err| err.to_string())?,
    )
    .map_err(|err| err.to_string())?;
    Ok(artifact(&path, "json", "job_manifest"))
}

fn collect_orca_sidecars(work_dir: &Path, basename: &str, artifacts: &mut Vec<QmArtifact>) {
    for (suffix, format, kind) in [
        (".property.json", "json", "orca_property_json"),
        (".json", "json", "orca_gbw_json"),
        (".gbw", "binary", "orca_gbw"),
        (".engrad", "text", "orca_gradient"),
        (".hess", "text", "orca_hessian"),
        (".xyz", "xyz", "optimized_structure"),
    ] {
        let path = work_dir.join(format!("{basename}{suffix}"));
        if path.exists() {
            artifacts.push(artifact(&path, format, kind));
        }
    }
}

fn write_charge_manifest(
    request: &QmRequest,
    work_dir: &Path,
    basename: &str,
    inspection: &parsers::OutputInspection,
) -> Option<QmArtifact> {
    let requested = request
        .task
        .charge_model
        .as_deref()
        .unwrap_or("hirshfeld")
        .to_ascii_lowercase();
    let analysis = inspection
        .charge_analysis
        .iter()
        .find(|analysis| analysis.model.eq_ignore_ascii_case(&requested))
        .or_else(|| inspection.charge_analysis.first())?;
    let manifest = ChargeManifest {
        schema_version: CHARGE_MANIFEST_VERSION.into(),
        model: analysis.model.clone(),
        charge_unit: "elementary_charge".into(),
        total_charge_e: analysis
            .total_charge_e
            .or_else(|| Some(analysis.atom_charges_e.iter().sum())),
        atom_charges_e: analysis.atom_charges_e.clone(),
        atom_labels: None,
        projection: None,
        provenance: BTreeMap::from([
            ("tool".into(), json!("ORCA")),
            (
                "source".into(),
                json!(work_dir.join(format!("{basename}.out")).to_string_lossy()),
            ),
        ]),
    };
    let path = work_dir.join("charge_manifest.json");
    let text = serde_json::to_string_pretty(&manifest).ok()?;
    fs::write(&path, text).ok()?;
    Some(artifact(&path, "json", "charge_manifest"))
}

fn run_orca_2mkl(
    orca_executable: &str,
    work_dir: &Path,
    basename: &str,
) -> Result<Option<PathBuf>, String> {
    let orca_dir = Path::new(orca_executable)
        .parent()
        .ok_or_else(|| "failed to resolve ORCA executable parent directory".to_string())?;
    let orca_2mkl = orca_dir.join("orca_2mkl");
    if !orca_2mkl.exists() {
        return Err(format!(
            "orca_2mkl not found next to ORCA executable: {}",
            orca_2mkl.display()
        ));
    }
    let gbw = work_dir.join(format!("{basename}.gbw"));
    if !gbw.exists() {
        return Ok(None);
    }
    let output = Command::new(&orca_2mkl)
        .arg(basename)
        .arg("-molden")
        .current_dir(work_dir)
        .output()
        .map_err(|err| format!("failed to execute orca_2mkl: {err}"))?;
    if !output.status.success() {
        return Err(format!(
            "orca_2mkl failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let molden = work_dir.join(format!("{basename}.molden.input"));
    if molden.exists() {
        Ok(Some(molden))
    } else {
        Err("orca_2mkl completed but Molden output was not produced".into())
    }
}

fn artifact(path: &Path, format: &str, kind: &str) -> QmArtifact {
    QmArtifact {
        path: path.to_string_lossy().into_owned(),
        format: format.into(),
        kind: kind.into(),
    }
}
