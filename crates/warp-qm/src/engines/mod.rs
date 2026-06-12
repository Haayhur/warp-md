pub mod multiwfn;
pub mod orca;
pub mod psi4;
pub mod xtb;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EngineProbe {
    pub engine: String,
    pub installed: bool,
    pub executable_path: Option<String>,
    pub version: Option<String>,
    pub license_status: String,
    pub supported_tasks: Vec<String>,
    pub warnings: Vec<String>,
}

pub fn probe_all() -> Vec<EngineProbe> {
    vec![
        psi4::probe(),
        orca::probe(),
        multiwfn::probe(),
        xtb::probe(),
    ]
}

pub(crate) fn find_executable(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().into_owned());
        }
    }
    None
}

pub(crate) fn command_version(executable: &str, args: &[&str]) -> Option<String> {
    let output = std::process::Command::new(executable)
        .args(args)
        .output()
        .ok()?;
    let raw = if output.stdout.is_empty() {
        String::from_utf8_lossy(&output.stderr).into_owned()
    } else {
        String::from_utf8_lossy(&output.stdout).into_owned()
    };
    let lines: Vec<&str> = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    lines
        .iter()
        .copied()
        .find(|line| line.to_ascii_lowercase().contains("version"))
        .or_else(|| {
            lines
                .iter()
                .copied()
                .find(|line| line.chars().any(char::is_alphanumeric))
        })
        .map(str::to_string)
}

pub(crate) fn common_tasks() -> Vec<String> {
    [
        "single_point",
        "optimize",
        "frequency",
        "charges",
        "esp",
        "resp_prepare",
        "resp_postprocess",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}
