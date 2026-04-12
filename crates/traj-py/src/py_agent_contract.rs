const WARP_MD_AGENT_SCHEMA_VERSION: &str = "warp-md.agent.v1";
const WARP_MD_RUN_REQUEST_TOP_LEVEL_FIELDS: &[&str] = &[
    "version",
    "run_id",
    "system",
    "topology",
    "trajectory",
    "traj",
    "device",
    "stream",
    "chunk_frames",
    "output_dir",
    "checkpoint",
    "fail_fast",
    "analyses",
];

#[derive(Clone, Debug, serde::Serialize)]
struct WarpMdFieldSpec {
    #[serde(rename = "type")]
    field_type: String,
    semantic_type: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    default: Option<serde_json::Value>,
    #[serde(default)]
    minimum: Option<f64>,
    #[serde(default)]
    maximum: Option<f64>,
    #[serde(default)]
    unit: Option<String>,
    #[serde(default)]
    choices: Option<Vec<String>>,
}

#[derive(Clone, Debug, serde::Serialize)]
struct WarpMdArtifactSpec {
    kind: String,
    format: String,
    #[serde(default)]
    fields: Vec<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Clone, Debug, serde::Serialize)]
struct WarpMdAnalysisContract {
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    description: String,
    #[serde(default)]
    required_fields: Vec<String>,
    #[serde(default)]
    optional_fields: Vec<String>,
    #[serde(default)]
    fields: std::collections::BTreeMap<String, WarpMdFieldSpec>,
    #[serde(default)]
    outputs: Vec<WarpMdArtifactSpec>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    examples: Vec<serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize)]
struct WarpMdContractCatalog {
    schema_version: String,
    #[serde(default)]
    analysis_shared_fields: Vec<String>,
    #[serde(default)]
    cli_to_analysis: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    analyses: Vec<WarpMdAnalysisContract>,
}

static WARP_MD_AGENT_CONTRACT_CATALOG: std::sync::OnceLock<WarpMdContractCatalog> =
    std::sync::OnceLock::new();
static WARP_MD_AGENT_NAME_LOOKUP: std::sync::OnceLock<std::collections::BTreeMap<String, String>> =
    std::sync::OnceLock::new();

include!("py_agent_contract_catalog.rs");

fn warp_md_agent_contract_catalog_ref() -> &'static WarpMdContractCatalog {
    WARP_MD_AGENT_CONTRACT_CATALOG.get_or_init(warp_md_agent_contract_catalog_native)
}

fn warp_md_agent_name_lookup() -> &'static std::collections::BTreeMap<String, String> {
    WARP_MD_AGENT_NAME_LOOKUP.get_or_init(|| {
        let catalog = warp_md_agent_contract_catalog_ref();
        let mut lookup = std::collections::BTreeMap::new();
        for contract in &catalog.analyses {
            for alias in contract
                .aliases
                .iter()
                .cloned()
                .chain(std::iter::once(contract.name.clone()))
                .chain(std::iter::once(contract.name.replace('_', "-")))
            {
                lookup.insert(alias.clone(), contract.name.clone());
                lookup.insert(alias.replace('-', "_"), contract.name.clone());
            }
        }
        for (alias, canonical) in &catalog.cli_to_analysis {
            lookup.insert(alias.clone(), canonical.clone());
            lookup.insert(alias.replace('-', "_"), canonical.clone());
        }
        lookup
    })
}

fn warp_md_resolve_analysis_name(name: &str) -> Option<String> {
    warp_md_agent_name_lookup().get(name.trim()).cloned()
}

fn warp_md_contract_for_name(name: &str) -> Option<WarpMdAnalysisContract> {
    let canonical = warp_md_resolve_analysis_name(name)?;
    warp_md_agent_contract_catalog_ref()
        .analyses
        .iter()
        .find(|contract| contract.name == canonical)
        .cloned()
}

fn warp_md_catalog_hash() -> String {
    let serialized = serde_json::to_vec(warp_md_agent_contract_catalog_ref())
        .expect("serialize warp-md analysis contract catalog");
    let digest = <sha2::Sha256 as sha2::Digest>::digest(&serialized);
    let hex = format!("{digest:x}");
    hex[..16].to_string()
}

fn warp_md_template_placeholder(
    field_name: &str,
    field_spec: &WarpMdFieldSpec,
) -> serde_json::Value {
    match field_spec.semantic_type.as_str() {
        "selection" | "mask" => serde_json::Value::String(format!("<{field_name}_expression>")),
        "charges" => serde_json::Value::String("by_resname".into()),
        _ => match field_spec.field_type.as_str() {
            "array" => serde_json::json!([]),
            "integer" => field_spec
                .default
                .clone()
                .unwrap_or_else(|| serde_json::json!(0)),
            "float" => field_spec
                .default
                .clone()
                .unwrap_or_else(|| serde_json::json!(0.0)),
            "boolean" => field_spec
                .default
                .clone()
                .unwrap_or_else(|| serde_json::json!(false)),
            _ => serde_json::Value::String(format!("<{field_name}>")),
        },
    }
}

fn warp_md_normalize_request_value(
    mut payload: serde_json::Value,
    strip_unknown: bool,
) -> serde_json::Value {
    let Some(root) = payload.as_object_mut() else {
        return payload;
    };

    if root.contains_key("topology") && !root.contains_key("system") {
        if let Some(value) = root.remove("topology") {
            root.insert("system".into(), value);
        }
    } else {
        root.remove("topology");
    }

    if root.contains_key("traj") && !root.contains_key("trajectory") {
        if let Some(value) = root.remove("traj") {
            root.insert("trajectory".into(), value);
        }
    } else {
        root.remove("traj");
    }

    if strip_unknown {
        let allowed: std::collections::BTreeSet<&str> = WARP_MD_RUN_REQUEST_TOP_LEVEL_FIELDS
            .iter()
            .copied()
            .collect();
        root.retain(|key, _| allowed.contains(key.as_str()));
    }

    let shared_fields: std::collections::BTreeSet<&str> = warp_md_agent_contract_catalog_ref()
        .analysis_shared_fields
        .iter()
        .map(String::as_str)
        .collect();

    if let Some(analyses) = root
        .get_mut("analyses")
        .and_then(serde_json::Value::as_array_mut)
    {
        for analysis in analyses {
            let Some(item) = analysis.as_object_mut() else {
                continue;
            };
            let Some(name_value) = item.get("name").and_then(serde_json::Value::as_str) else {
                continue;
            };
            let Some(contract) = warp_md_contract_for_name(name_value) else {
                continue;
            };
            item.insert(
                "name".into(),
                serde_json::Value::String(contract.name.clone()),
            );

            if strip_unknown {
                let allowed_fields: std::collections::BTreeSet<&str> = contract
                    .fields
                    .keys()
                    .map(String::as_str)
                    .chain(shared_fields.iter().copied())
                    .chain(std::iter::once("name"))
                    .collect();
                item.retain(|key, _| allowed_fields.contains(key.as_str()));
            }

            for (field_name, field_spec) in &contract.fields {
                if !item.contains_key(field_name) {
                    if let Some(default) = &field_spec.default {
                        item.insert(field_name.clone(), default.clone());
                    }
                }
            }
        }
    }

    payload
}

fn warp_md_validation_error(
    code: &str,
    path: &str,
    message: impl Into<String>,
) -> serde_json::Value {
    serde_json::json!({
        "code": code,
        "path": path,
        "message": message.into(),
        "context": {},
    })
}

fn warp_md_validation_result(
    errors: Vec<serde_json::Value>,
    warnings: Vec<String>,
    normalized_request: Option<serde_json::Value>,
) -> serde_json::Value {
    serde_json::json!({
        "schema_version": WARP_MD_AGENT_SCHEMA_VERSION,
        "status": if errors.is_empty() { "ok" } else { "error" },
        "valid": errors.is_empty(),
        "normalized_request": normalized_request,
        "errors": errors,
        "warnings": warnings,
    })
}

fn warp_md_coerce_io_spec(
    value: &serde_json::Value,
    label: &str,
) -> Result<serde_json::Value, String> {
    match value {
        serde_json::Value::String(path) => {
            let trimmed = path.trim();
            if trimmed.is_empty() {
                Err(format!("{label} path cannot be empty"))
            } else {
                Ok(serde_json::json!({"path": trimmed}))
            }
        }
        serde_json::Value::Object(map) => {
            if map.is_empty() {
                Err(format!("{label} spec cannot be empty"))
            } else {
                Ok(value.clone())
            }
        }
        _ => Err(format!("{label} must be a path string or object")),
    }
}

fn warp_md_validate_request_value(payload: serde_json::Value, strict: bool) -> serde_json::Value {
    let mut errors: Vec<serde_json::Value> = Vec::new();
    let warnings: Vec<String> = Vec::new();

    let Some(root) = payload.as_object() else {
        return warp_md_validation_result(
            vec![warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "request payload must be an object",
            )],
            warnings,
            None,
        );
    };

    let allowed_top_level: std::collections::BTreeSet<&str> = WARP_MD_RUN_REQUEST_TOP_LEVEL_FIELDS
        .iter()
        .copied()
        .collect();
    for key in root.keys() {
        if !allowed_top_level.contains(key.as_str()) {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                key,
                "unknown top-level field",
            ));
        }
    }

    let version = match root.get("version") {
        None => serde_json::Value::String(WARP_MD_AGENT_SCHEMA_VERSION.into()),
        Some(serde_json::Value::String(value)) if value == WARP_MD_AGENT_SCHEMA_VERSION => {
            serde_json::Value::String(value.clone())
        }
        Some(serde_json::Value::String(value)) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "version",
                format!(
                    "unsupported run config version: {value}; expected {WARP_MD_AGENT_SCHEMA_VERSION}"
                ),
            ));
            serde_json::Value::String(value.clone())
        }
        Some(_) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "version",
                "version must be a string",
            ));
            serde_json::Value::Null
        }
    };

    let system_spec = match (root.get("system"), root.get("topology")) {
        (Some(_), Some(_)) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "specify only one of `system` or `topology`",
            ));
            None
        }
        (None, None) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "one of `system` or `topology` is required",
            ));
            None
        }
        (Some(value), None) => match warp_md_coerce_io_spec(value, "system") {
            Ok(spec) => Some(spec),
            Err(message) => {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    "system",
                    message,
                ));
                None
            }
        },
        (None, Some(value)) => match warp_md_coerce_io_spec(value, "topology") {
            Ok(spec) => Some(spec),
            Err(message) => {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    "topology",
                    message,
                ));
                None
            }
        },
    };

    let trajectory_spec = match (root.get("trajectory"), root.get("traj")) {
        (Some(_), Some(_)) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "specify only one of `trajectory` or `traj`",
            ));
            None
        }
        (None, None) => {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                "root",
                "one of `trajectory` or `traj` is required",
            ));
            None
        }
        (Some(value), None) => match warp_md_coerce_io_spec(value, "trajectory") {
            Ok(spec) => Some(spec),
            Err(message) => {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    "trajectory",
                    message,
                ));
                None
            }
        },
        (None, Some(value)) => match warp_md_coerce_io_spec(value, "traj") {
            Ok(spec) => Some(spec),
            Err(message) => {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    "traj",
                    message,
                ));
                None
            }
        },
    };

    let mut normalized_root = root.clone();
    normalized_root.insert("version".into(), version);
    if let Some(spec) = system_spec {
        normalized_root.insert("system".into(), spec);
        normalized_root.remove("topology");
    }
    if let Some(spec) = trajectory_spec {
        normalized_root.insert("trajectory".into(), spec);
        normalized_root.remove("traj");
    }

    let analyses_value = root.get("analyses");
    let Some(analyses) = analyses_value.and_then(serde_json::Value::as_array) else {
        errors.push(warp_md_validation_error(
            "E_SCHEMA_VALIDATION",
            "analyses",
            "analyses must be a non-empty array",
        ));
        return warp_md_validation_result(errors, warnings, None);
    };
    if analyses.is_empty() {
        errors.push(warp_md_validation_error(
            "E_SCHEMA_VALIDATION",
            "analyses",
            "analyses must contain at least one item",
        ));
    }

    let shared_fields: std::collections::BTreeSet<&str> = warp_md_agent_contract_catalog_ref()
        .analysis_shared_fields
        .iter()
        .map(String::as_str)
        .collect();
    let mut normalized_analyses = Vec::with_capacity(analyses.len());

    for (idx, analysis) in analyses.iter().enumerate() {
        let path_prefix = format!("analyses[{idx}]");
        let Some(item) = analysis.as_object() else {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &path_prefix,
                "analysis entry must be an object",
            ));
            normalized_analyses.push(analysis.clone());
            continue;
        };

        let mut normalized_analysis = item.clone();
        let Some(name_value) = item.get("name") else {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &format!("{path_prefix}.name"),
                "analysis name is required",
            ));
            normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
            continue;
        };
        let Some(name_str) = name_value.as_str() else {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &format!("{path_prefix}.name"),
                "analysis name must be a string",
            ));
            normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
            continue;
        };
        let trimmed_name = name_str.trim();
        if trimmed_name.is_empty() {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &format!("{path_prefix}.name"),
                "analysis name cannot be empty",
            ));
            normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
            continue;
        }

        let Some(contract) = warp_md_contract_for_name(trimmed_name) else {
            errors.push(warp_md_validation_error(
                "E_SCHEMA_VALIDATION",
                &format!("{path_prefix}.name"),
                format!("unknown analysis: {trimmed_name}"),
            ));
            normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
            continue;
        };
        normalized_analysis.insert(
            "name".into(),
            serde_json::Value::String(contract.name.clone()),
        );

        if let Some(value) = item.get("out") {
            if !value.is_string() && !value.is_null() {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    &format!("{path_prefix}.out"),
                    "out must be a string",
                ));
            }
        }
        if let Some(value) = item.get("device") {
            if !value.is_string() && !value.is_null() {
                errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    &format!("{path_prefix}.device"),
                    "device must be a string",
                ));
            }
        }
        if let Some(value) = item.get("chunk_frames") {
            match value.as_u64() {
                Some(v) if v >= 1 => {}
                _ => errors.push(warp_md_validation_error(
                    "E_SCHEMA_VALIDATION",
                    &format!("{path_prefix}.chunk_frames"),
                    "chunk_frames must be a positive integer",
                )),
            }
        }

        for field_name in &contract.required_fields {
            match item.get(field_name) {
                None | Some(serde_json::Value::Null) => errors.push(warp_md_validation_error(
                    "E_REQUIRED_FIELD",
                    &format!("{path_prefix}.{field_name}"),
                    format!("{} requires field '{field_name}'", contract.name),
                )),
                Some(serde_json::Value::String(value)) if value.trim().is_empty() => {
                    errors.push(warp_md_validation_error(
                        "E_REQUIRED_FIELD",
                        &format!("{path_prefix}.{field_name}"),
                        format!("{} requires field '{field_name}'", contract.name),
                    ))
                }
                _ => {}
            }
        }

        for (field_name, field_value) in item {
            if field_name == "name" {
                continue;
            }
            if shared_fields.contains(field_name.as_str()) {
                continue;
            }
            let Some(field_spec) = contract.fields.get(field_name) else {
                if strict {
                    errors.push(warp_md_validation_error(
                        "E_UNKNOWN_FIELD",
                        &format!("{path_prefix}.{field_name}"),
                        format!("Unknown field for {}: {field_name}", contract.name),
                    ));
                }
                continue;
            };

            let type_ok = match field_spec.field_type.as_str() {
                "boolean" => field_value.is_boolean(),
                "integer" => field_value.as_i64().is_some() || field_value.as_u64().is_some(),
                "float" => field_value.is_number(),
                "array" => field_value.is_array(),
                _ => true,
            };
            if !type_ok {
                errors.push(warp_md_validation_error(
                    "E_FIELD_TYPE",
                    &format!("{path_prefix}.{field_name}"),
                    format!(
                        "Expected {}, got {}",
                        field_spec.field_type,
                        warp_md_json_type_name(field_value)
                    ),
                ));
                continue;
            }

            if let Some(number) = field_value.as_f64() {
                if let Some(minimum) = field_spec.minimum {
                    if number < minimum {
                        errors.push(warp_md_validation_error(
                            "E_VALUE_RANGE",
                            &format!("{path_prefix}.{field_name}"),
                            format!("Value {number} below minimum {minimum}"),
                        ));
                    }
                }
                if let Some(maximum) = field_spec.maximum {
                    if number > maximum {
                        errors.push(warp_md_validation_error(
                            "E_VALUE_RANGE",
                            &format!("{path_prefix}.{field_name}"),
                            format!("Value {number} above maximum {maximum}"),
                        ));
                    }
                }
            }

            if let Some(choices) = &field_spec.choices {
                if !choices
                    .iter()
                    .any(|choice| Some(choice.as_str()) == field_value.as_str())
                {
                    errors.push(warp_md_validation_error(
                        "E_INVALID_CHOICE",
                        &format!("{path_prefix}.{field_name}"),
                        format!("Invalid choice: {field_value}. Must be one of {choices:?}"),
                    ));
                }
            }
        }

        normalized_analyses.push(serde_json::Value::Object(normalized_analysis));
    }

    normalized_root.insert(
        "analyses".into(),
        serde_json::Value::Array(normalized_analyses),
    );
    let normalized_request = if errors.is_empty() {
        Some(warp_md_normalize_request_value(
            serde_json::Value::Object(normalized_root),
            false,
        ))
    } else {
        None
    };

    warp_md_validation_result(errors, warnings, normalized_request)
}

fn warp_md_json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "bool",
        serde_json::Value::Number(number)
            if number.as_i64().is_some() || number.as_u64().is_some() =>
        {
            "int"
        }
        serde_json::Value::Number(_) => "float",
        serde_json::Value::String(_) => "str",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

fn warp_md_lint_result(
    valid: bool,
    expression: &str,
    field_type: &str,
    matched_atoms: Option<usize>,
    total_atoms: Option<usize>,
    error: Option<String>,
    warnings: Vec<String>,
) -> serde_json::Value {
    serde_json::json!({
        "valid": valid,
        "expression": expression,
        "field_type": field_type,
        "matched_atoms": matched_atoms,
        "total_atoms": total_atoms,
        "error": error,
        "warnings": warnings,
    })
}

fn warp_md_load_system_from_path(path: &str) -> Result<System, String> {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match ext.as_str() {
        "pdb" => {
            let mut reader = PdbReader::new(path);
            match reader.read_system() {
                Ok(system) => Ok(system),
                Err(err) => {
                    let message = err.to_string();
                    if message.to_ascii_lowercase().contains("invalid resid") {
                        let reader = PdbReader::new(path);
                        reader.read_system_permissive().map_err(|e| e.to_string())
                    } else {
                        Err(message)
                    }
                }
            }
        }
        "gro" => {
            let mut reader = GroReader::new(path);
            reader.read_system().map_err(|e| e.to_string())
        }
        _ => Err("system.format must be pdb or gro".into()),
    }
}

fn warp_md_lint_selection_value(
    expr: &str,
    field_type: &str,
    system_path: Option<&str>,
) -> serde_json::Value {
    if expr.trim().is_empty() {
        return warp_md_lint_result(
            false,
            expr,
            field_type,
            None,
            None,
            Some("Selection expression cannot be empty".into()),
            vec![],
        );
    }

    let expr_stripped = expr.trim();
    if expr_stripped.matches('\'').count() % 2 != 0 {
        return warp_md_lint_result(
            false,
            expr,
            field_type,
            None,
            None,
            Some("Unbalanced single quotes in selection".into()),
            vec![],
        );
    }
    if expr_stripped.matches('"').count() % 2 != 0 {
        return warp_md_lint_result(
            false,
            expr,
            field_type,
            None,
            None,
            Some("Unbalanced double quotes in selection".into()),
            vec![],
        );
    }

    let mut paren_depth = 0i64;
    for (idx, ch) in expr_stripped.chars().enumerate() {
        if ch == '(' {
            paren_depth += 1;
        } else if ch == ')' {
            paren_depth -= 1;
        }
        if paren_depth < 0 {
            return warp_md_lint_result(
                false,
                expr,
                field_type,
                None,
                None,
                Some(format!("Unbalanced parentheses at position {idx}")),
                vec![],
            );
        }
    }
    if paren_depth != 0 {
        return warp_md_lint_result(
            false,
            expr,
            field_type,
            None,
            None,
            Some("Unbalanced parentheses in selection".into()),
            vec![],
        );
    }

    let mut matched_atoms = None;
    let mut total_atoms = None;
    let mut warnings = Vec::new();

    if let Some(path) = system_path {
        match warp_md_load_system_from_path(path) {
            Ok(mut system) => {
                total_atoms = Some(system.n_atoms());
                match system.select(expr_stripped) {
                    Ok(selection) => {
                        matched_atoms = Some(selection.indices.len());
                        if matched_atoms == Some(0) {
                            warnings.push("Selection matched zero atoms".into());
                        }
                    }
                    Err(err) => {
                        return warp_md_lint_result(
                            false,
                            expr,
                            field_type,
                            None,
                            total_atoms,
                            Some(format!("Selection syntax error: {err}")),
                            vec![],
                        );
                    }
                }
            }
            Err(err) => warnings.push(format!("Could not load topology for atom count: {err}")),
        }
    }

    warp_md_lint_result(
        true,
        expr,
        field_type,
        matched_atoms,
        total_atoms,
        None,
        warnings,
    )
}

const WARP_MD_GOAL_PHRASES: &[&str] = &[
    "radius of gyration",
    "secondary structure",
    "mean square displacement",
    "diffusion coefficient",
    "pair distribution",
    "radial distribution",
    "hydrogen bond",
    "hydrogen bonds",
    "free volume",
    "fractional free volume",
    "water shell",
    "solvation shell",
    "count water",
    "count waters",
    "water count",
    "molecular docking",
    "binding pose",
    "ligand contacts",
];

fn warp_md_goal_phrase_candidates(phrase: &str) -> &'static [(&'static str, f64)] {
    match phrase {
        "radius of gyration" => &[("rg", 12.0)],
        "secondary structure" => &[("dssp", 12.0)],
        "mean square displacement" => &[("msd", 12.0)],
        "diffusion coefficient" => &[("diffusion", 11.0), ("msd", 8.0)],
        "pair distribution" | "radial distribution" => &[("rdf", 11.0)],
        "hydrogen bond" | "hydrogen bonds" => &[("hbond", 12.0)],
        "free volume" => &[("free_volume", 12.0), ("bondi_ffv", 9.0)],
        "fractional free volume" => &[("bondi_ffv", 13.0)],
        "water shell" | "solvation shell" => &[("watershell", 12.0)],
        "count water" | "count waters" | "water count" => &[("water_count", 12.0)],
        "molecular docking" | "binding pose" | "ligand contacts" => &[("docking", 12.0)],
        _ => &[],
    }
}

fn warp_md_is_generic_goal_word(word: &str) -> bool {
    matches!(
        word,
        "analysis"
            | "analyze"
            | "around"
            | "calculate"
            | "compute"
            | "count"
            | "determine"
            | "find"
            | "function"
            | "interface"
            | "measure"
            | "over"
            | "series"
            | "show"
            | "time"
            | "track"
            | "want"
    )
}

fn warp_md_is_generic_tag(tag: &str) -> bool {
    matches!(
        tag,
        "dynamic" | "polymer" | "protein" | "solvent" | "spatial" | "structural" | "transport"
    )
}

fn warp_md_is_docking_goal(goal_lower: &str, words: &std::collections::BTreeSet<String>) -> bool {
    ["binding", "dock", "docking", "ligand", "pose", "receptor"]
        .iter()
        .any(|trigger| goal_lower.contains(trigger) || words.contains(*trigger))
}

fn warp_md_expand_goal_words(text: &str) -> std::collections::BTreeSet<String> {
    let mut words = std::collections::BTreeSet::new();
    for word in text
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
        .filter(|word| !word.is_empty())
    {
        words.insert(word.to_string());
        if word.len() > 4 && word.ends_with("ies") {
            words.insert(format!("{}y", &word[..word.len() - 3]));
        } else if word.len() > 4 && word.ends_with('s') {
            words.insert(word[..word.len() - 1].to_string());
        }
    }
    words
}

fn warp_md_tokenize_contract_text(text: &str) -> std::collections::BTreeSet<String> {
    warp_md_expand_goal_words(&text.to_ascii_lowercase())
}

fn warp_md_goal_keywords(word: &str) -> &'static [&'static str] {
    match word {
        "radius" => &["rg"],
        "gyration" => &["rg"],
        "size" => &["rg"],
        "compactness" => &["rg"],
        "rmsd" => &["rmsd"],
        "alignment" => &["rmsd", "dipole_alignment"],
        "structure" => &["rmsd", "dssp", "native_contacts"],
        "secondary" => &["dssp"],
        "backbone" => &["dssp", "rmsd"],
        "dihedral" => &["jcoupling", "tordiff"],
        "torsion" => &["tordiff", "jcoupling"],
        "motion" => &["msd", "diffusion"],
        "displacement" => &["msd"],
        "diffusion" => &["diffusion", "msd"],
        "transport" => &["diffusion", "msd", "conductivity"],
        "charge" => &["dipole_alignment", "conductivity"],
        "dipole" => &["dipole_alignment"],
        "polarization" => &["dipole_alignment"],
        "conductivity" => &["conductivity"],
        "permittivity" => &["dielectric"],
        "dielectric" => &["dielectric"],
        "electrostatic" => &["dielectric", "dipole_alignment", "conductivity"],
        "distribution" => &["rdf", "bond_length_distribution", "bond_angle_distribution"],
        "pair" => &["rdf"],
        "radial" => &["rdf"],
        "correlation" => &["rdf", "ion_pair_correlation", "rotacf"],
        "neighbor" => &["rdf", "native_contacts"],
        "contact" => &["native_contacts", "hbond"],
        "hydrogen" => &["hbond"],
        "water" => &["water_count", "watershell", "rdf"],
        "waters" => &["water_count", "watershell", "rdf"],
        "shell" => &["watershell"],
        "solvation" => &["watershell", "free_volume", "gist"],
        "hydration" => &["watershell"],
        "density" => &["density", "volmap"],
        "profile" => &["density", "rdf"],
        "spatial" => &["water_count", "free_volume", "volmap"],
        "map" => &["volmap", "free_volume", "gist"],
        "grid" => &["volmap", "free_volume", "gist"],
        "free" => &["free_volume", "bondi_ffv"],
        "volume" => &["free_volume", "bondi_ffv", "volmap"],
        "void" => &["free_volume", "bondi_ffv"],
        "ffv" => &["bondi_ffv"],
        "bondi" => &["bondi_ffv"],
        "energy" => &["equipartition", "gist"],
        "kinetic" => &["equipartition"],
        "temperature" => &["conductivity", "dielectric", "equipartition"],
        "docking" => &["docking"],
        "dock" => &["docking"],
        "binding" => &["docking"],
        "pose" => &["docking"],
        "ligand" => &["docking"],
        "receptor" => &["docking"],
        "cluster" => &["ion_pair_correlation"],
        "protein" => &["dssp", "rmsd", "rg", "native_contacts"],
        "polymer" => &[
            "free_volume",
            "bondi_ffv",
            "rg",
            "chain_rg",
            "contour_length",
            "end_to_end",
            "persistence_length",
        ],
        _ => &[],
    }
}

fn warp_md_suggest_analyses_value(
    goal: &str,
    provided_fields: &[String],
    top_n: usize,
) -> serde_json::Value {
    let provided: std::collections::BTreeSet<&str> =
        provided_fields.iter().map(String::as_str).collect();
    let goal_lower = goal.to_ascii_lowercase();
    let words = warp_md_expand_goal_words(&goal_lower);

    let mut scored: Vec<(String, f64, String)> = Vec::new();

    for contract in &warp_md_agent_contract_catalog_ref().analyses {
        if contract.name == "docking" && !warp_md_is_docking_goal(&goal_lower, &words) {
            continue;
        }

        let mut score = 0.0;
        let mut reasons: Vec<String> = Vec::new();

        for phrase in WARP_MD_GOAL_PHRASES {
            if goal_lower.contains(phrase) {
                for (candidate, weight) in warp_md_goal_phrase_candidates(phrase) {
                    if *candidate == contract.name
                        || contract.aliases.iter().any(|alias| alias == candidate)
                    {
                        score += *weight;
                        reasons.push(format!("phrase match: {phrase}"));
                    }
                }
            }
        }

        if goal_lower.contains(&contract.name) {
            score += 10.0;
            reasons.push(format!("name match: {}", contract.name));
        }
        for alias in &contract.aliases {
            if goal_lower.contains(alias) || goal_lower.contains(&alias.replace('-', "_")) {
                score += 8.0;
                reasons.push(format!("alias match: {alias}"));
            }
        }
        for tag in &contract.tags {
            if goal_lower.contains(tag) {
                score += if warp_md_is_generic_tag(tag) {
                    1.0
                } else {
                    3.0
                };
                reasons.push(format!("tag match: {tag}"));
            }
        }

        let desc_terms = warp_md_tokenize_contract_text(&contract.description);
        let mut matched_desc_terms = 0usize;
        for word in &words {
            if word.len() > 3 && !warp_md_is_generic_goal_word(word) && desc_terms.contains(word) {
                score += 1.5;
                matched_desc_terms += 1;
            }
        }
        if matched_desc_terms > 0 {
            reasons.push("description match".into());
        }

        for word in &words {
            for candidate in warp_md_goal_keywords(word) {
                if *candidate == contract.name
                    || contract.aliases.iter().any(|alias| alias == candidate)
                {
                    score += 4.0;
                    reasons.push(format!("keyword match: {word}"));
                }
            }
        }

        for field_name in contract.fields.keys() {
            if goal_lower.contains(field_name) {
                score += 2.0;
                reasons.push(format!("field match: {field_name}"));
            }
        }

        for output in &contract.outputs {
            if goal_lower.contains(&output.kind) {
                score += 2.0;
                reasons.push(format!("output kind: {}", output.kind));
            }
        }

        if score > 0.0 {
            let mut dedup = Vec::new();
            for reason in reasons {
                if !dedup.contains(&reason) {
                    dedup.push(reason);
                }
            }
            scored.push((contract.name.clone(), score, dedup.join(", ")));
        }
    }

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let candidates: Vec<serde_json::Value> = scored
        .into_iter()
        .take(top_n)
        .filter_map(|(name, score, reason)| {
            let contract = warp_md_contract_for_name(&name)?;
            let missing_fields: Vec<String> = contract
                .required_fields
                .iter()
                .filter(|field| !provided.contains(field.as_str()))
                .cloned()
                .collect();
            Some(serde_json::json!({
                "name": name,
                "reason": reason,
                "missing_fields": missing_fields,
                "score": score,
            }))
        })
        .collect();

    serde_json::json!({
        "goal": goal,
        "total_analyses": warp_md_agent_contract_catalog_ref().analyses.len(),
        "candidates": candidates,
    })
}

#[pyfunction]
fn warp_md_agent_contract_catalog<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = serde_json::to_value(warp_md_agent_contract_catalog_ref())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn warp_md_agent_plan_schema<'py>(py: Python<'py>, name: &str) -> PyResult<PyObject> {
    let contract = warp_md_contract_for_name(name)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown plan: {name}")))?;
    let value =
        serde_json::to_value(contract).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn warp_md_agent_capabilities<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let catalog = warp_md_agent_contract_catalog_ref();
    let available_plans: Vec<String> = catalog
        .analyses
        .iter()
        .map(|contract| contract.name.clone())
        .collect();
    let value = serde_json::json!({
        "schema_version": WARP_MD_AGENT_SCHEMA_VERSION,
        "available_plans": available_plans,
        "plan_catalog_hash": warp_md_catalog_hash(),
        "supports_streaming": true,
        "supports_selection_linting": true,
    });
    json_value_to_py(py, &value)
}

#[pyfunction]
#[pyo3(signature = (analysis_name, fill_defaults=false))]
fn warp_md_agent_generate_template<'py>(
    py: Python<'py>,
    analysis_name: &str,
    fill_defaults: bool,
) -> PyResult<PyObject> {
    let contract = warp_md_contract_for_name(analysis_name)
        .ok_or_else(|| PyRuntimeError::new_err(format!("unknown analysis: {analysis_name}")))?;
    let mut analysis = serde_json::Map::new();
    analysis.insert(
        "name".into(),
        serde_json::Value::String(contract.name.clone()),
    );

    for field_name in &contract.required_fields {
        if let Some(field_spec) = contract.fields.get(field_name) {
            analysis.insert(
                field_name.clone(),
                warp_md_template_placeholder(field_name, field_spec),
            );
        }
    }

    if fill_defaults {
        for field_name in &contract.optional_fields {
            if let Some(field_spec) = contract.fields.get(field_name) {
                if let Some(default) = &field_spec.default {
                    analysis.insert(field_name.clone(), default.clone());
                }
            }
        }
    }

    let value = serde_json::json!({
        "version": WARP_MD_AGENT_SCHEMA_VERSION,
        "system": {"path": "<topology-path>"},
        "trajectory": {"path": "<trajectory-path>"},
        "analyses": [serde_json::Value::Object(analysis)],
    });
    json_value_to_py(py, &value)
}

#[pyfunction]
#[pyo3(signature = (json, strip_unknown=false))]
fn warp_md_agent_normalize_request<'py>(
    py: Python<'py>,
    json: &str,
    strip_unknown: bool,
) -> PyResult<PyObject> {
    let payload: serde_json::Value =
        serde_json::from_str(json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let normalized = warp_md_normalize_request_value(payload, strip_unknown);
    json_value_to_py(py, &normalized)
}

#[pyfunction]
#[pyo3(signature = (json, strict=false))]
fn warp_md_agent_validate_request<'py>(
    py: Python<'py>,
    json: &str,
    strict: bool,
) -> PyResult<PyObject> {
    let payload: serde_json::Value =
        serde_json::from_str(json).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let result = warp_md_validate_request_value(payload, strict);
    json_value_to_py(py, &result)
}

#[pyfunction]
#[pyo3(signature = (expr, field_type="selection", system_path=None))]
fn warp_md_agent_lint_selection<'py>(
    py: Python<'py>,
    expr: &str,
    field_type: &str,
    system_path: Option<&str>,
) -> PyResult<PyObject> {
    let result = warp_md_lint_selection_value(expr, field_type, system_path);
    json_value_to_py(py, &result)
}

#[pyfunction]
#[pyo3(signature = (goal, provided_fields=None, top_n=5))]
fn warp_md_agent_suggest_analyses<'py>(
    py: Python<'py>,
    goal: &str,
    provided_fields: Option<Vec<String>>,
    top_n: usize,
) -> PyResult<PyObject> {
    let result = warp_md_suggest_analyses_value(goal, &provided_fields.unwrap_or_default(), top_n);
    json_value_to_py(py, &result)
}

fn register_warp_md_agent_contract(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(warp_md_agent_contract_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_plan_schema, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_generate_template, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_normalize_request, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_validate_request, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_lint_selection, m)?)?;
    m.add_function(wrap_pyfunction!(warp_md_agent_suggest_analyses, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_alias_to_canonical_analysis_name() {
        assert_eq!(
            warp_md_resolve_analysis_name("dipole-alignment").as_deref(),
            Some("dipole_alignment")
        );
        assert_eq!(
            warp_md_resolve_analysis_name("free-volume-grid").as_deref(),
            Some("free_volume")
        );
    }

    #[test]
    fn normalizes_request_aliases_and_defaults() {
        let payload = serde_json::json!({
            "topology": "top.pdb",
            "traj": "traj.xtc",
            "analyses": [
                {
                    "name": "rdf",
                    "sel_a": "resname SOL",
                    "sel_b": "resname SOL",
                    "r_max": 1.0,
                    "mystery": true
                }
            ]
        });

        let normalized = warp_md_normalize_request_value(payload, true);
        assert_eq!(normalized["system"], "top.pdb");
        assert_eq!(normalized["trajectory"], "traj.xtc");
        assert_eq!(normalized["analyses"][0]["name"], "rdf");
        assert_eq!(normalized["analyses"][0]["bins"], 200);
        assert!(normalized["analyses"][0].get("mystery").is_none());
    }

    #[test]
    fn catalog_embeds_expected_analysis() {
        let contract = warp_md_contract_for_name("bondi-ffv").expect("bondi_ffv contract");
        assert_eq!(contract.name, "bondi_ffv");
        assert!(contract.fields.contains_key("bondi_scale"));
    }

    #[test]
    fn validates_request_and_reports_range_errors() {
        let payload = serde_json::json!({
            "system": "top.pdb",
            "trajectory": "traj.xtc",
            "analyses": [
                {
                    "name": "rdf",
                    "sel_a": "resname SOL",
                    "sel_b": "resname SOL",
                    "bins": -1,
                    "r_max": 1.0,
                }
            ]
        });
        let result = warp_md_validate_request_value(payload, false);
        assert_eq!(result["valid"], false);
        let errors = result["errors"].as_array().expect("errors");
        assert!(errors.iter().any(|error| error["code"] == "E_VALUE_RANGE"));
    }

    #[test]
    fn suggests_expected_analysis_for_radius_goal() {
        let result = warp_md_suggest_analyses_value("radius of gyration", &[], 5);
        assert_eq!(result["candidates"][0]["name"], "rg");
    }

    #[test]
    fn suggests_water_count_without_docking_false_positive() {
        let result = warp_md_suggest_analyses_value("count waters around a protein", &[], 5);
        assert_eq!(result["candidates"][0]["name"], "water_count");
    }

    #[test]
    fn suggests_hbond_for_hydrogen_bond_goal() {
        let result =
            warp_md_suggest_analyses_value("hydrogen bonds in protein water interface", &[], 5);
        assert_eq!(result["candidates"][0]["name"], "hbond");
    }

    #[test]
    fn suggests_free_volume_for_generic_free_volume_goal() {
        let result = warp_md_suggest_analyses_value("measure free volume in polymer", &[], 5);
        assert_eq!(result["candidates"][0]["name"], "free_volume");
    }
}
