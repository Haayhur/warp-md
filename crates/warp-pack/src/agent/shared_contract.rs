use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ErrorDetail {
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub message: String,
    pub severity: String,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
#[serde(untagged)]
pub enum OneOrManyStrings {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ArtifactEnvelope {
    pub coordinates: String,
    pub manifest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub md_package: Option<String>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct RunSummary {
    pub component_count: usize,
    pub total_atoms: usize,
    pub water_count: usize,
    pub ion_counts: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct RunSuccessEnvelope {
    pub schema_version: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub output_dir: String,
    pub artifacts: ArtifactEnvelope,
    pub summary: RunSummary,
    pub manifest_path: String,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct RunErrorEnvelope {
    pub schema_version: String,
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub exit_code: i32,
    pub error: ErrorDetail,
    pub errors: Vec<ErrorDetail>,
    pub warnings: Vec<ErrorDetail>,
}

#[derive(Clone, Debug, Serialize, JsonSchema)]
pub struct ValidateSuccessEnvelope {
    pub schema_version: String,
    pub status: String,
    pub valid: bool,
    pub normalized_request: super::BuildRequest,
    pub resolved_inputs: super::ResolvedInputsSummary,
    pub warnings: Vec<ErrorDetail>,
}

pub fn to_error(
    code: impl Into<String>,
    path: impl Into<Option<String>>,
    message: impl Into<String>,
) -> ErrorDetail {
    ErrorDetail {
        code: code.into(),
        path: path.into().and_then(|value| json_pointer(&value)),
        message: message.into(),
        severity: String::from("error"),
    }
}

pub fn to_warning(
    code: impl Into<String>,
    path: impl Into<Option<String>>,
    message: impl Into<String>,
) -> ErrorDetail {
    ErrorDetail {
        code: code.into(),
        path: path.into().and_then(|value| json_pointer(&value)),
        message: message.into(),
        severity: String::from("warning"),
    }
}

fn json_pointer_token(token: &str) -> String {
    token.replace('~', "~0").replace('/', "~1")
}

pub fn json_pointer(path: &str) -> Option<String> {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('/') {
        return Some(trimmed.to_string());
    }

    let mut segments = Vec::new();
    let mut current = String::new();
    let mut chars = trimmed.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '.' => {
                if !current.is_empty() {
                    segments.push(std::mem::take(&mut current));
                }
            }
            '[' => {
                if !current.is_empty() {
                    segments.push(std::mem::take(&mut current));
                }
                let mut index = String::new();
                while let Some(next) = chars.next() {
                    if next == ']' {
                        break;
                    }
                    index.push(next);
                }
                if !index.is_empty() {
                    segments.push(index);
                }
            }
            _ => current.push(ch),
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    if segments.is_empty() {
        None
    } else {
        Some(format!(
            "/{}",
            segments
                .iter()
                .map(|segment| json_pointer_token(segment))
                .collect::<Vec<_>>()
                .join("/")
        ))
    }
}
