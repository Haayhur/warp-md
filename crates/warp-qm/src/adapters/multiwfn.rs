use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::{collections::BTreeMap, collections::BTreeSet, ffi::OsString};

use serde_json::{json, Value};

use crate::adapters::{AdapterRun, AdapterSummary};
use crate::contract::{
    ChargeDeployableSet, ChargeManifest, ChargeProjectionManifest, ChargeRedistributionRule,
    CubeManifest, EspManifest, QmArtifact, QmRequest, CHARGE_MANIFEST_VERSION,
    CUBE_MANIFEST_VERSION, ESP_MANIFEST_VERSION,
};
use crate::engines::find_executable;

pub fn run(request: &QmRequest) -> AdapterRun {
    let work_dir = PathBuf::from(&request.runtime.work_dir);
    if let Err(err) = fs::create_dir_all(&work_dir) {
        return AdapterRun::error(format!("failed to create Multiwfn work dir: {err}"), 2);
    }
    if is_resp2_request(request) {
        return run_resp2(request, &work_dir);
    }
    let input_file = match multiwfn_input_file(request) {
        Some(path) => path,
        None => {
            return AdapterRun::error_code(
                "E_MULTIWFN_INPUT_MISSING",
                "Multiwfn requires molecule.source.path or engine.settings.input_file",
                2,
            )
        }
    };
    if !Path::new(&input_file).exists() {
        return AdapterRun::error_code(
            "E_MULTIWFN_INPUT_MISSING",
            format!("Multiwfn input file not found: {input_file}"),
            2,
        );
    }
    let recipe = match recipe_for_request(request, &input_file) {
        Ok(recipe) => recipe,
        Err(err) => return AdapterRun::error(err, 2),
    };
    let script_path = work_dir.join("multiwfn_input.txt");
    if let Err(err) = fs::write(&script_path, recipe.script.as_bytes()) {
        return AdapterRun::error(format!("failed to write Multiwfn script: {err}"), 2);
    }
    let mut artifacts = vec![artifact(&script_path, "text", "multiwfn_script")];

    let executable = match resolve_multiwfn_executable(request) {
        Some(path) => path,
        None => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: None,
                artifacts,
                warnings: vec![
                    "Multiwfn executable not found; set engine.executable, WARP_QM_MULTIWFN, MULTIWFN_PATH, or PATH"
                        .into(),
                ],
                properties: merge_error(
                    recipe_properties(&recipe),
                    "E_MULTIWFN_EXECUTABLE_MISSING",
                    "Multiwfn executable not found; set engine.executable, WARP_QM_MULTIWFN, MULTIWFN_PATH, or PATH",
                ),
                summary: AdapterSummary::default(),
            }
        }
    };

    let command_vec = vec![executable.clone(), input_file.clone()];
    let runtime_env = multiwfn_runtime_env(request, &executable);
    let output = run_scripted_command(
        &executable,
        &input_file,
        &work_dir,
        &recipe.script,
        &runtime_env,
    );
    let output = match output {
        Ok(output) => output,
        Err(err) => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: Some(command_vec),
                artifacts,
                warnings: vec![format!("failed to execute Multiwfn: {err}")],
                properties: merge_error(
                    recipe_properties(&recipe),
                    "E_MULTIWFN_PROCESS",
                    format!("failed to execute Multiwfn: {err}"),
                ),
                summary: AdapterSummary::default(),
            }
        }
    };

    let stdout_path = work_dir.join("multiwfn.out");
    let stderr_path = work_dir.join("multiwfn.err");
    let _ = fs::write(&stdout_path, &output.stdout);
    let _ = fs::write(&stderr_path, &output.stderr);
    artifacts.push(artifact(&stdout_path, "text", "multiwfn_stdout"));
    if stderr_path
        .metadata()
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
    {
        artifacts.push(artifact(&stderr_path, "text", "multiwfn_stderr"));
    }
    collect_outputs(&work_dir, &recipe, &mut artifacts);
    if let Some(charge_manifest) = write_charge_manifest(request, &work_dir, &recipe) {
        artifacts.push(charge_manifest);
    }
    if let Some(cube_manifest) = write_cube_manifest(&work_dir, &recipe) {
        artifacts.push(cube_manifest);
    }
    if let Some(esp_manifest) = write_esp_manifest(&work_dir, &recipe) {
        artifacts.push(esp_manifest);
    }
    let missing_outputs = missing_expected_outputs(&work_dir, &recipe);
    let mut warnings = Vec::new();
    if !missing_outputs.is_empty() {
        warnings.push(format!(
            "Multiwfn completed but expected output artifacts are missing: {}",
            missing_outputs.join(", ")
        ));
    }
    let success = output.status.success() && missing_outputs.is_empty();

    AdapterRun {
        status: if success { "ok" } else { "error" }.into(),
        exit_code: if success { 0 } else { 4 },
        command: Some(command_vec),
        artifacts,
        warnings,
        properties: recipe_properties(&recipe),
        summary: AdapterSummary::default(),
    }
}

fn run_resp2(request: &QmRequest, work_dir: &Path) -> AdapterRun {
    let gas_input = match setting_string(request, "gas_input_file")
        .or_else(|| setting_string(request, "gas_file"))
        .or_else(|| setting_string(request, "gas"))
    {
        Some(path) => path,
        None => return AdapterRun::error_code(
            "E_RESP2_INPUT_MISSING",
            "RESP2 requires engine.settings.gas_input_file and engine.settings.solvent_input_file",
            2,
        ),
    };
    let solvent_input = match setting_string(request, "solvent_input_file")
        .or_else(|| setting_string(request, "solv_input_file"))
        .or_else(|| setting_string(request, "solvent_file"))
        .or_else(|| setting_string(request, "solv"))
    {
        Some(path) => path,
        None => return AdapterRun::error_code(
            "E_RESP2_INPUT_MISSING",
            "RESP2 requires engine.settings.gas_input_file and engine.settings.solvent_input_file",
            2,
        ),
    };
    if !Path::new(&gas_input).exists() {
        return AdapterRun::error_code(
            "E_RESP2_INPUT_MISSING",
            format!("RESP2 gas input file not found: {gas_input}"),
            2,
        );
    }
    if !Path::new(&solvent_input).exists() {
        return AdapterRun::error_code(
            "E_RESP2_INPUT_MISSING",
            format!("RESP2 solvent input file not found: {solvent_input}"),
            2,
        );
    }
    let executable = match resolve_multiwfn_executable(request) {
        Some(path) => path,
        None => {
            return AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: None,
                artifacts: Vec::new(),
                warnings: vec![
                    "Multiwfn executable not found; set engine.executable, WARP_QM_MULTIWFN, MULTIWFN_PATH, or PATH"
                        .into(),
                ],
                properties: merge_error(
                    resp2_properties(request, None),
                    "E_MULTIWFN_EXECUTABLE_MISSING",
                    "Multiwfn executable not found; set engine.executable, WARP_QM_MULTIWFN, MULTIWFN_PATH, or PATH",
                ),
                summary: AdapterSummary::default(),
            }
        }
    };

    let mut artifacts = Vec::new();
    let mut warnings = Vec::new();
    let gas_run = match run_resp_component(request, &executable, work_dir, "gas", &gas_input) {
        Ok(run) => run,
        Err(run) => return run,
    };
    artifacts.extend(gas_run.artifacts.clone());
    warnings.extend(gas_run.warnings.clone());
    if gas_run.status != "ok" {
        return resp2_component_error(request, gas_run, artifacts, warnings);
    }
    let solvent_run =
        match run_resp_component(request, &executable, work_dir, "solvent", &solvent_input) {
            Ok(run) => run,
            Err(run) => return run,
        };
    artifacts.extend(solvent_run.artifacts.clone());
    warnings.extend(solvent_run.warnings.clone());
    if solvent_run.status != "ok" {
        return resp2_component_error(request, solvent_run, artifacts, warnings);
    }

    let gas_path = work_dir.join("gas").join(charge_output_name(&gas_input));
    let solvent_path = work_dir
        .join("solvent")
        .join(charge_output_name(&solvent_input));
    let gas_rows = match parse_chg_rows(&gas_path) {
        Some(rows) => rows,
        None => {
            return AdapterRun::error_code(
                "E_RESP2_CHARGE_PARSE",
                "failed to parse RESP2 gas charge table",
                4,
            )
        }
    };
    let solvent_rows = match parse_chg_rows(&solvent_path) {
        Some(rows) => rows,
        None => {
            return AdapterRun::error_code(
                "E_RESP2_CHARGE_PARSE",
                "failed to parse RESP2 solvent charge table",
                4,
            )
        }
    };
    if gas_rows.len() != solvent_rows.len() {
        return AdapterRun::error_code(
            "E_RESP2_ATOM_COUNT_MISMATCH",
            format!(
                "RESP2 gas/solvent atom count mismatch: {} vs {}",
                gas_rows.len(),
                solvent_rows.len()
            ),
            4,
        );
    }
    let delta = resp2_delta(request);
    let combined_rows: Vec<ChgRow> = gas_rows
        .iter()
        .zip(solvent_rows.iter())
        .map(|(gas, solvent)| ChgRow {
            atom: gas.atom.clone(),
            x: gas.x,
            y: gas.y,
            z: gas.z,
            charge: gas.charge * (1.0 - delta) + solvent.charge * delta,
        })
        .collect();
    let resp2_path = work_dir.join("RESP2.chg");
    if let Err(err) = write_chg_rows(&resp2_path, &combined_rows) {
        return AdapterRun::error_code(
            "E_RESP2_WRITE",
            format!("failed to write RESP2 charge table: {err}"),
            4,
        );
    }
    artifacts.push(artifact(&resp2_path, "text", "charge_table"));
    if let Some(manifest) = write_resp2_charge_manifest(
        request,
        work_dir,
        &combined_rows,
        &gas_input,
        &solvent_input,
        &gas_path,
        &solvent_path,
        delta,
    ) {
        artifacts.push(manifest);
    }

    AdapterRun {
        status: "ok".into(),
        exit_code: 0,
        command: Some(vec![
            executable,
            gas_input,
            solvent_input,
            format!("delta={delta}"),
        ]),
        artifacts,
        warnings,
        properties: resp2_properties(request, Some(delta)),
        summary: AdapterSummary::default(),
    }
}

#[derive(Clone, Debug)]
pub struct MultiwfnRecipe {
    pub name: String,
    pub script: String,
    pub expected_outputs: Vec<String>,
    pub charge_model: Option<String>,
}

pub fn recipe_for_request(request: &QmRequest, input_file: &str) -> Result<MultiwfnRecipe, String> {
    if let Some(script) = request
        .engine
        .settings
        .get("menu_script")
        .and_then(Value::as_str)
    {
        return Ok(MultiwfnRecipe {
            name: "custom_menu_script".into(),
            script: ensure_trailing_newline(script),
            expected_outputs: configured_expected_outputs(request),
            charge_model: request.task.charge_model.clone(),
        });
    }
    if let Some(path) = request
        .engine
        .settings
        .get("menu_script_file")
        .and_then(Value::as_str)
    {
        let script = fs::read_to_string(path)
            .map_err(|err| format!("failed to read Multiwfn menu_script_file '{path}': {err}"))?;
        return Ok(MultiwfnRecipe {
            name: "custom_menu_script_file".into(),
            script: ensure_trailing_newline(&script),
            expected_outputs: configured_expected_outputs(request),
            charge_model: request.task.charge_model.clone(),
        });
    }
    match request.task.kind.as_str() {
        "generic_run" => Err(
            "Multiwfn generic_run requires engine.settings.menu_script or engine.settings.menu_script_file"
                .into(),
        ),
        "resp_fit" | "resp_prepare" | "resp_postprocess" => Ok(resp_recipe(input_file)),
        "population" => population_recipe(request, input_file),
        "esp" => Ok(cube_recipe(
            "ESP",
            "5\n12\n",
            grid_quality(request),
            "totesp.cub",
        )),
        "orbital_cube" => orbital_cube_recipe(request),
        "electron_density_cube" => Ok(cube_recipe(
            "electron_density",
            "5\n1\n",
            grid_quality(request),
            "density.cub",
        )),
        "elf_cube" => Ok(cube_recipe(
            "ELF",
            "5\n9\n",
            grid_quality(request),
            "ELF.cub",
        )),
        "lol_cube" => Ok(cube_recipe(
            "LOL",
            "5\n10\n",
            grid_quality(request),
            "LOL.cub",
        )),
        "laplacian_cube" => Ok(cube_recipe(
            "laplacian",
            "5\n3\n",
            grid_quality(request),
            "laplacian.cub",
        )),
        other => Err(format!(
            "Multiwfn adapter does not support task kind '{other}'"
        )),
    }
}

fn resp_recipe(input_file: &str) -> MultiwfnRecipe {
    MultiwfnRecipe {
        name: "resp_standard_two_stage".into(),
        script: "7\n18\n1\ny\n0\n0\nq\n".into(),
        expected_outputs: vec![charge_output_name(input_file)],
        charge_model: Some("resp".into()),
    }
}

fn population_recipe(request: &QmRequest, input_file: &str) -> Result<MultiwfnRecipe, String> {
    match request
        .task
        .charge_model
        .as_deref()
        .unwrap_or("hirshfeld")
        .to_ascii_lowercase()
        .as_str()
    {
        "resp" => Ok(resp_recipe(input_file)),
        "hirshfeld" => Ok(MultiwfnRecipe {
            name: "population_hirshfeld".into(),
            script: "7\n1\n1\ny\n0\nq\n".into(),
            expected_outputs: vec![charge_output_name(input_file)],
            charge_model: Some("hirshfeld".into()),
        }),
        other => Err(format!(
            "Multiwfn population charge model '{other}' is not verified for this adapter; supported verified models are resp and hirshfeld"
        )),
    }
}

fn orbital_cube_recipe(request: &QmRequest) -> Result<MultiwfnRecipe, String> {
    let orbital = request
        .engine
        .settings
        .get("orbital")
        .and_then(Value::as_i64)
        .or_else(|| {
            request
                .engine
                .settings
                .get("orbital_number")
                .and_then(Value::as_i64)
        })
        .ok_or_else(|| {
            "orbital_cube requires engine.settings.orbital or engine.settings.orbital_number"
                .to_string()
        })?;
    if orbital <= 0 {
        return Err("orbital_cube orbital number must be positive".into());
    }
    Ok(MultiwfnRecipe {
        name: format!("cube_orbital_{orbital}"),
        script: exit_script(&format!("5\n4\n{orbital}\n{}\n2\n", grid_quality(request))),
        expected_outputs: vec!["MOvalue.cub".into()],
        charge_model: None,
    })
}

fn cube_recipe(name: &str, menu: &str, quality: &str, output_name: &str) -> MultiwfnRecipe {
    MultiwfnRecipe {
        name: format!("cube_{name}"),
        script: exit_script(&format!("{menu}{quality}\n2\n")),
        expected_outputs: vec![output_name.into()],
        charge_model: None,
    }
}

fn grid_quality(request: &QmRequest) -> &'static str {
    match request
        .engine
        .settings
        .get("grid_quality")
        .and_then(Value::as_str)
        .unwrap_or("medium")
    {
        "low" => "1",
        "high" => "3",
        "ultra" => "4",
        _ => "2",
    }
}

fn exit_script(commands: &str) -> String {
    let mut script = commands.to_string();
    if !script.ends_with('\n') {
        script.push('\n');
    }
    script.push_str("0\nq\n");
    script
}

fn ensure_trailing_newline(script: &str) -> String {
    let mut script = script.to_string();
    if !script.ends_with('\n') {
        script.push('\n');
    }
    script
}

fn is_resp2_request(request: &QmRequest) -> bool {
    matches!(
        request.task.kind.as_str(),
        "resp_fit" | "resp_prepare" | "resp_postprocess"
    ) && request
        .task
        .charge_model
        .as_deref()
        .unwrap_or("")
        .eq_ignore_ascii_case("resp2")
}

fn setting_string(request: &QmRequest, key: &str) -> Option<String> {
    request
        .engine
        .settings
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
}

fn resp2_delta(request: &QmRequest) -> f64 {
    request
        .engine
        .settings
        .get("delta")
        .and_then(Value::as_f64)
        .unwrap_or(0.5)
}

fn multiwfn_input_file(request: &QmRequest) -> Option<String> {
    request
        .engine
        .settings
        .get("input_file")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| request.molecule.source.path.clone())
}

fn run_resp_component(
    request: &QmRequest,
    executable: &str,
    parent_dir: &Path,
    label: &str,
    input_file: &str,
) -> Result<AdapterRun, AdapterRun> {
    let work_dir = parent_dir.join(label);
    if let Err(err) = fs::create_dir_all(&work_dir) {
        return Err(AdapterRun::error(
            format!("failed to create RESP2 {label} work dir: {err}"),
            2,
        ));
    }
    let recipe = resp_recipe(input_file);
    let script_path = work_dir.join("multiwfn_input.txt");
    if let Err(err) = fs::write(&script_path, recipe.script.as_bytes()) {
        return Err(AdapterRun::error(
            format!("failed to write RESP2 {label} Multiwfn script: {err}"),
            2,
        ));
    }
    let mut artifacts = vec![artifact(&script_path, "text", "multiwfn_script")];
    let runtime_env = multiwfn_runtime_env(request, executable);
    let output = run_scripted_command(
        executable,
        input_file,
        &work_dir,
        &recipe.script,
        &runtime_env,
    );
    let output = match output {
        Ok(output) => output,
        Err(err) => {
            return Err(AdapterRun {
                status: "error".into(),
                exit_code: 4,
                command: Some(vec![executable.into(), input_file.into()]),
                artifacts,
                warnings: vec![format!("failed to execute RESP2 {label} Multiwfn: {err}")],
                properties: merge_error(
                    recipe_properties(&recipe),
                    "E_MULTIWFN_PROCESS",
                    format!("failed to execute RESP2 {label} Multiwfn: {err}"),
                ),
                summary: AdapterSummary::default(),
            })
        }
    };
    let stdout_path = work_dir.join("multiwfn.out");
    let stderr_path = work_dir.join("multiwfn.err");
    let _ = fs::write(&stdout_path, &output.stdout);
    let _ = fs::write(&stderr_path, &output.stderr);
    artifacts.push(artifact(&stdout_path, "text", "multiwfn_stdout"));
    if stderr_path
        .metadata()
        .map(|metadata| metadata.len() > 0)
        .unwrap_or(false)
    {
        artifacts.push(artifact(&stderr_path, "text", "multiwfn_stderr"));
    }
    collect_outputs(&work_dir, &recipe, &mut artifacts);
    let missing_outputs = missing_expected_outputs(&work_dir, &recipe);
    let mut warnings = Vec::new();
    if !missing_outputs.is_empty() {
        warnings.push(format!(
            "RESP2 {label} expected output artifacts are missing: {}",
            missing_outputs.join(", ")
        ));
    }
    let success = output.status.success() && missing_outputs.is_empty();
    Ok(AdapterRun {
        status: if success { "ok" } else { "error" }.into(),
        exit_code: if success { 0 } else { 4 },
        command: Some(vec![executable.into(), input_file.into()]),
        artifacts,
        warnings,
        properties: recipe_properties(&recipe),
        summary: AdapterSummary::default(),
    })
}

fn resolve_multiwfn_executable(request: &QmRequest) -> Option<String> {
    request
        .engine
        .executable
        .clone()
        .or_else(|| std::env::var("WARP_QM_MULTIWFN").ok())
        .or_else(|| std::env::var("MULTIWFN_PATH").ok())
        .and_then(|path| {
            let candidate = Path::new(&path);
            if candidate.is_dir() {
                Some(candidate.join("Multiwfn").to_string_lossy().into_owned())
            } else {
                Some(path)
            }
        })
        .or_else(|| find_executable("Multiwfn"))
        .or_else(|| find_executable("Multiwfn_noGUI"))
        .or_else(|| find_executable("multiwfn"))
}

fn multiwfn_runtime_env(request: &QmRequest, executable: &str) -> Vec<(String, OsString)> {
    let mut envs = Vec::new();
    let lib_dir = request
        .engine
        .settings
        .get("lib_dir")
        .and_then(Value::as_str)
        .map(str::to_string)
        .or_else(|| std::env::var("WARP_QM_MULTIWFN_LIB_DIR").ok());
    if let Some(lib_dir) = lib_dir {
        let mut value = OsString::from(lib_dir);
        if let Some(existing) = std::env::var_os("LD_LIBRARY_PATH") {
            value.push(":");
            value.push(existing);
        }
        envs.push(("LD_LIBRARY_PATH".into(), value));
    }
    let multiwfn_path = request
        .engine
        .settings
        .get("multiwfn_path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .or_else(|| Path::new(executable).parent().map(Path::to_path_buf));
    if let Some(path) = multiwfn_path {
        envs.push(("Multiwfnpath".into(), path.into_os_string()));
    }
    if std::env::var_os("OMP_STACKSIZE").is_none() {
        envs.push(("OMP_STACKSIZE".into(), OsString::from("2G")));
    }
    envs
}

fn collect_outputs(work_dir: &Path, recipe: &MultiwfnRecipe, artifacts: &mut Vec<QmArtifact>) {
    for output in &recipe.expected_outputs {
        let path = work_dir.join(output);
        if path.exists() {
            artifacts.push(artifact(&path, output_format(output), output_kind(output)));
        }
    }
}

fn missing_expected_outputs(work_dir: &Path, recipe: &MultiwfnRecipe) -> Vec<String> {
    recipe
        .expected_outputs
        .iter()
        .filter(|output| !work_dir.join(output).exists())
        .cloned()
        .collect()
}

fn configured_expected_outputs(request: &QmRequest) -> Vec<String> {
    request
        .engine
        .settings
        .get("expected_outputs")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

fn charge_output_name(input_file: &str) -> String {
    let name = Path::new(input_file)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("charges");
    let stem = name
        .strip_suffix(".molden.input")
        .or_else(|| name.strip_suffix(".molden"))
        .or_else(|| name.rsplit_once('.').map(|(stem, _)| stem))
        .unwrap_or(name);
    format!("{stem}.chg")
}

fn output_format(path: &str) -> &'static str {
    if path.ends_with(".cub") {
        "cube"
    } else {
        "text"
    }
}

fn output_kind(path: &str) -> &'static str {
    if path.ends_with(".cub") {
        "cube_grid"
    } else if path.ends_with(".chg") {
        "charge_table"
    } else {
        "multiwfn_output"
    }
}

fn recipe_properties(recipe: &MultiwfnRecipe) -> serde_json::Map<String, Value> {
    let mut properties = serde_json::Map::new();
    properties.insert(
        "multiwfn_recipe".into(),
        json!({
            "name": recipe.name,
            "expected_outputs": recipe.expected_outputs,
            "charge_model": recipe.charge_model,
        }),
    );
    properties
}

fn merge_error(
    mut properties: serde_json::Map<String, Value>,
    code: &str,
    message: impl Into<String>,
) -> serde_json::Map<String, Value> {
    properties.insert(
        "errors".into(),
        json!([{"code": code, "message": message.into()}]),
    );
    properties
}

fn resp2_properties(request: &QmRequest, delta: Option<f64>) -> serde_json::Map<String, Value> {
    let mut properties = serde_json::Map::new();
    properties.insert(
        "resp2".into(),
        json!({
            "charge_model": "resp2",
            "formula": "q_resp2 = (1 - delta) * q_gas_resp + delta * q_solvent_resp",
            "delta": delta.unwrap_or_else(|| resp2_delta(request)),
            "gas_input_file": setting_string(request, "gas_input_file")
                .or_else(|| setting_string(request, "gas_file"))
                .or_else(|| setting_string(request, "gas")),
            "solvent_input_file": setting_string(request, "solvent_input_file")
                .or_else(|| setting_string(request, "solv_input_file"))
                .or_else(|| setting_string(request, "solvent_file"))
                .or_else(|| setting_string(request, "solv")),
            "component_recipe": "resp_standard_two_stage",
        }),
    );
    properties
}

fn resp2_component_error(
    request: &QmRequest,
    failed: AdapterRun,
    artifacts: Vec<QmArtifact>,
    warnings: Vec<String>,
) -> AdapterRun {
    let mut properties = resp2_properties(request, Some(resp2_delta(request)));
    properties.insert(
        "component_error".into(),
        json!({
            "status": failed.status,
            "exit_code": failed.exit_code,
            "warnings": failed.warnings,
        }),
    );
    AdapterRun {
        status: "error".into(),
        exit_code: 4,
        command: failed.command,
        artifacts,
        warnings,
        properties,
        summary: AdapterSummary::default(),
    }
}

fn write_charge_manifest(
    request: &QmRequest,
    work_dir: &Path,
    recipe: &MultiwfnRecipe,
) -> Option<QmArtifact> {
    let charge_path = recipe
        .expected_outputs
        .iter()
        .map(|name| work_dir.join(name))
        .find(|path| {
            path.extension().and_then(|ext| ext.to_str()) == Some("chg") && path.exists()
        })?;
    let charges = parse_chg_charges(&charge_path)?;
    let manifest = ChargeManifest {
        schema_version: CHARGE_MANIFEST_VERSION.into(),
        model: recipe
            .charge_model
            .clone()
            .unwrap_or_else(|| recipe.name.clone()),
        charge_unit: "elementary_charge".into(),
        total_charge_e: Some(charges.iter().sum()),
        atom_charges_e: charges,
        atom_labels: parse_chg_labels(&charge_path),
        projection: charge_projection_from_request(request, parse_chg_charges(&charge_path)?),
        provenance: BTreeMap::from([
            ("tool".into(), json!("Multiwfn")),
            ("recipe".into(), json!(recipe.name)),
            ("source".into(), json!(charge_path.to_string_lossy())),
        ]),
    };
    let path = work_dir.join("charge_manifest.json");
    let text = serde_json::to_string_pretty(&manifest).ok()?;
    fs::write(&path, text).ok()?;
    Some(artifact(&path, "json", "charge_manifest"))
}

fn write_resp2_charge_manifest(
    request: &QmRequest,
    work_dir: &Path,
    rows: &[ChgRow],
    gas_input: &str,
    solvent_input: &str,
    gas_charge_path: &Path,
    solvent_charge_path: &Path,
    delta: f64,
) -> Option<QmArtifact> {
    let charges: Vec<f64> = rows.iter().map(|row| row.charge).collect();
    let manifest = ChargeManifest {
        schema_version: CHARGE_MANIFEST_VERSION.into(),
        model: "resp2".into(),
        charge_unit: "elementary_charge".into(),
        total_charge_e: Some(charges.iter().sum()),
        atom_charges_e: charges,
        atom_labels: Some(rows.iter().map(|row| row.atom.clone()).collect()),
        projection: charge_projection_from_request(
            request,
            rows.iter().map(|row| row.charge).collect(),
        ),
        provenance: BTreeMap::from([
            ("tool".into(), json!("Multiwfn")),
            ("recipe".into(), json!("resp2_gas_solvent_mixing")),
            (
                "formula".into(),
                json!("q_resp2 = (1 - delta) * q_gas_resp + delta * q_solvent_resp"),
            ),
            ("delta".into(), json!(delta)),
            ("gas_input_file".into(), json!(gas_input)),
            ("solvent_input_file".into(), json!(solvent_input)),
            (
                "gas_charge_table".into(),
                json!(gas_charge_path.to_string_lossy()),
            ),
            (
                "solvent_charge_table".into(),
                json!(solvent_charge_path.to_string_lossy()),
            ),
        ]),
    };
    let path = work_dir.join("charge_manifest.json");
    let text = serde_json::to_string_pretty(&manifest).ok()?;
    fs::write(&path, text).ok()?;
    Some(artifact(&path, "json", "charge_manifest"))
}

fn parse_chg_charges(path: &Path) -> Option<Vec<f64>> {
    let text = fs::read_to_string(path).ok()?;
    let charges: Vec<f64> = text
        .lines()
        .filter_map(|line| line.split_whitespace().last()?.parse::<f64>().ok())
        .collect();
    if charges.is_empty() {
        None
    } else {
        Some(charges)
    }
}

fn parse_chg_labels(path: &Path) -> Option<Vec<String>> {
    let text = fs::read_to_string(path).ok()?;
    let labels: Vec<String> = text
        .lines()
        .filter_map(|line| line.split_whitespace().next().map(str::to_string))
        .collect();
    if labels.is_empty() {
        None
    } else {
        Some(labels)
    }
}

fn charge_projection_from_request(
    request: &QmRequest,
    charges: Vec<f64>,
) -> Option<ChargeProjectionManifest> {
    let spec = request.engine.settings.get("charge_projection")?;
    let policy = spec
        .get("policy")
        .and_then(Value::as_str)
        .unwrap_or("explicit")
        .to_string();
    let retained_source_policy = spec
        .get("retained_source_policy")
        .and_then(Value::as_str)
        .unwrap_or("preserve");
    if retained_source_policy == "error" && retained_source_overlap(spec) {
        return None;
    }
    let redistribution = redistribution_rules(&charges, spec);
    let projected = projected_charges(&charges, spec)?;
    let deployable_sets = deployable_sets(&charges, &redistribution, spec);
    Some(ChargeProjectionManifest {
        policy,
        projected_charges_e: projected,
        deployable_sets,
        redistribution,
        provenance: BTreeMap::from([
            ("source".into(), json!("engine.settings.charge_projection")),
            (
                "retained_source_policy".into(),
                json!(retained_source_policy),
            ),
            (
                "deployable_projection".into(),
                json!("set_aware_redistribute_only_omitted_sources"),
            ),
            (
                "indexing".into(),
                json!("zero_based atom indices into atom_charges_e"),
            ),
        ]),
    })
}

fn projected_charges(charges: &[f64], spec: &Value) -> Option<Vec<f64>> {
    let mut projected = charges.to_vec();
    for rule in redistribution_rules(charges, spec) {
        if rule.source_atom >= projected.len() {
            return None;
        }
        projected[rule.source_atom] = 0.0;
        for (target, weight) in rule.target_atoms.iter().zip(rule.weights.iter()) {
            if *target >= projected.len() {
                return None;
            }
            projected[*target] += rule.source_charge_e * weight;
        }
    }
    Some(projected)
}

fn redistribution_rules(charges: &[f64], spec: &Value) -> Vec<ChargeRedistributionRule> {
    spec.get("redistribution")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|rule| {
            let source_atom = rule.get("source_atom")?.as_u64()? as usize;
            let target_atoms: Vec<usize> = rule
                .get("target_atoms")?
                .as_array()?
                .iter()
                .filter_map(|value| value.as_u64().map(|idx| idx as usize))
                .collect();
            if target_atoms.is_empty() || source_atom >= charges.len() {
                return None;
            }
            let mut weights: Vec<f64> = rule
                .get("weights")
                .and_then(Value::as_array)
                .map(|items| items.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_else(|| vec![1.0 / target_atoms.len() as f64; target_atoms.len()]);
            if weights.len() != target_atoms.len() {
                return None;
            }
            let sum: f64 = weights.iter().sum();
            if sum != 0.0 && (sum - 1.0).abs() > 1e-12 {
                for weight in &mut weights {
                    *weight /= sum;
                }
            }
            Some(ChargeRedistributionRule {
                source_atom,
                target_atoms,
                weights,
                source_charge_e: charges[source_atom],
            })
        })
        .collect()
}

fn deployable_sets(
    charges: &[f64],
    rules: &[ChargeRedistributionRule],
    spec: &Value,
) -> Vec<ChargeDeployableSet> {
    let Some(sets) = spec.get("deployable_sets").and_then(Value::as_array) else {
        return Vec::new();
    };
    sets.iter()
        .filter_map(|set| {
            let name = set.get("name")?.as_str()?.to_string();
            let atom_indices: Vec<usize> = set
                .get("atom_indices")?
                .as_array()?
                .iter()
                .filter_map(|value| value.as_u64().map(|idx| idx as usize))
                .collect();
            if atom_indices.iter().any(|idx| *idx >= charges.len()) {
                return None;
            }
            let set_members: BTreeSet<usize> = atom_indices.iter().copied().collect();
            let mut set_charges: BTreeMap<usize, f64> = atom_indices
                .iter()
                .map(|idx| (*idx, charges[*idx]))
                .collect();
            for rule in rules {
                if set_members.contains(&rule.source_atom) {
                    continue;
                }
                let retained_targets: Vec<(usize, f64)> = rule
                    .target_atoms
                    .iter()
                    .copied()
                    .zip(rule.weights.iter().copied())
                    .filter(|(target, _)| set_members.contains(target))
                    .collect();
                if retained_targets.is_empty() {
                    continue;
                }
                let retained_weight_sum: f64 =
                    retained_targets.iter().map(|(_, weight)| weight).sum();
                if retained_weight_sum == 0.0 {
                    continue;
                }
                for (target, weight) in retained_targets {
                    if let Some(charge) = set_charges.get_mut(&target) {
                        *charge += rule.source_charge_e * weight / retained_weight_sum;
                    }
                }
            }
            Some(ChargeDeployableSet {
                name,
                charges_e: atom_indices
                    .iter()
                    .map(|idx| set_charges.get(idx).copied().unwrap_or(charges[*idx]))
                    .collect(),
                atom_indices,
                role: set.get("role").and_then(Value::as_str).map(str::to_string),
            })
        })
        .collect()
}

fn retained_source_overlap(spec: &Value) -> bool {
    let source_atoms: BTreeSet<usize> = spec
        .get("redistribution")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|rule| {
            rule.get("source_atom")
                .and_then(Value::as_u64)
                .map(|idx| idx as usize)
        })
        .collect();
    spec.get("deployable_sets")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|set| {
            set.get("atom_indices")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_u64)
                .any(|idx| source_atoms.contains(&(idx as usize)))
        })
}

#[derive(Clone, Debug)]
struct ChgRow {
    atom: String,
    x: f64,
    y: f64,
    z: f64,
    charge: f64,
}

fn parse_chg_rows(path: &Path) -> Option<Vec<ChgRow>> {
    let text = fs::read_to_string(path).ok()?;
    let rows: Vec<ChgRow> = text
        .lines()
        .filter_map(|line| {
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 5 {
                return None;
            }
            Some(ChgRow {
                atom: fields[0].into(),
                x: fields[1].parse().ok()?,
                y: fields[2].parse().ok()?,
                z: fields[3].parse().ok()?,
                charge: fields[4].parse().ok()?,
            })
        })
        .collect();
    if rows.is_empty() {
        None
    } else {
        Some(rows)
    }
}

fn write_chg_rows(path: &Path, rows: &[ChgRow]) -> std::io::Result<()> {
    let mut text = String::new();
    for row in rows {
        text.push_str(&format!(
            "{:<2} {:>12.6} {:>12.6} {:>12.6} {:>16.10}\n",
            row.atom, row.x, row.y, row.z, row.charge
        ));
    }
    fs::write(path, text)
}

fn write_esp_manifest(work_dir: &Path, recipe: &MultiwfnRecipe) -> Option<QmArtifact> {
    if recipe.name != "cube_ESP" {
        return None;
    }
    let grid_path = recipe
        .expected_outputs
        .iter()
        .map(|name| work_dir.join(name))
        .find(|path| {
            path.extension().and_then(|ext| ext.to_str()) == Some("cub") && path.exists()
        })?;
    let manifest = EspManifest {
        schema_version: ESP_MANIFEST_VERSION.into(),
        grid_path: Some(grid_path.to_string_lossy().into_owned()),
        grid_format: Some("cube".into()),
        potential_unit: "atomic_unit".into(),
        coordinate_unit: "bohr".into(),
        point_count: cube_point_count(&grid_path),
        provenance: BTreeMap::from([
            ("tool".into(), json!("Multiwfn")),
            ("recipe".into(), json!(recipe.name)),
        ]),
    };
    let path = work_dir.join("esp_manifest.json");
    let text = serde_json::to_string_pretty(&manifest).ok()?;
    fs::write(&path, text).ok()?;
    Some(artifact(&path, "json", "esp_manifest"))
}

fn write_cube_manifest(work_dir: &Path, recipe: &MultiwfnRecipe) -> Option<QmArtifact> {
    let grid_path = recipe
        .expected_outputs
        .iter()
        .map(|name| work_dir.join(name))
        .find(|path| {
            path.extension().and_then(|ext| ext.to_str()) == Some("cub") && path.exists()
        })?;
    let manifest = CubeManifest {
        schema_version: CUBE_MANIFEST_VERSION.into(),
        property: recipe.name.clone(),
        grid_path: grid_path.to_string_lossy().into_owned(),
        grid_format: "cube".into(),
        coordinate_unit: "bohr".into(),
        value_unit: None,
        point_count: cube_point_count(&grid_path),
        provenance: BTreeMap::from([
            ("tool".into(), json!("Multiwfn")),
            ("recipe".into(), json!(recipe.name)),
        ]),
    };
    let path = work_dir.join("cube_manifest.json");
    let text = serde_json::to_string_pretty(&manifest).ok()?;
    fs::write(&path, text).ok()?;
    Some(artifact(&path, "json", "cube_manifest"))
}

fn cube_point_count(path: &Path) -> Option<usize> {
    let text = fs::read_to_string(path).ok()?;
    let mut lines = text.lines().skip(3).take(3);
    let nx = lines
        .next()?
        .split_whitespace()
        .next()?
        .parse::<isize>()
        .ok()?
        .unsigned_abs();
    let ny = lines
        .next()?
        .split_whitespace()
        .next()?
        .parse::<isize>()
        .ok()?
        .unsigned_abs();
    let nz = lines
        .next()?
        .split_whitespace()
        .next()?
        .parse::<isize>()
        .ok()?
        .unsigned_abs();
    Some(nx * ny * nz)
}

fn artifact(path: &Path, format: &str, kind: &str) -> QmArtifact {
    QmArtifact {
        path: path.to_string_lossy().into_owned(),
        format: format.into(),
        kind: kind.into(),
    }
}

fn run_scripted_command(
    executable: &str,
    input_file: &str,
    work_dir: &Path,
    script: &str,
    runtime_env: &[(String, OsString)],
) -> std::io::Result<std::process::Output> {
    let mut command = Command::new(executable);
    command
        .arg(input_file)
        .current_dir(work_dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (key, value) in runtime_env {
        command.env(key, value);
    }
    let mut child = command.spawn()?;
    if let Some(stdin) = child.stdin.as_mut() {
        stdin.write_all(script.as_bytes())?;
    }
    child.wait_with_output()
}
