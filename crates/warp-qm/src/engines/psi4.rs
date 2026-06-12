use super::{command_version, find_executable, EngineProbe};

pub fn probe() -> EngineProbe {
    let executable_path = find_executable("psi4");
    let version = executable_path
        .as_deref()
        .and_then(|path| command_version(path, &["--version"]));
    EngineProbe {
        engine: "psi4".into(),
        installed: executable_path.is_some(),
        executable_path,
        version,
        license_status: "not_required_or_unchecked".into(),
        supported_tasks: vec!["generic_run".into()],
        warnings: vec![
            "Psi4 structured task rendering is not implemented yet; use generic_run with an agent-supplied Psi4 input file".into(),
        ],
    }
}
