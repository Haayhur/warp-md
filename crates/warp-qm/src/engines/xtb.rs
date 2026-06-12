use super::{command_version, find_executable, EngineProbe};

pub fn probe() -> EngineProbe {
    let executable_path = find_executable("xtb");
    let version = executable_path
        .as_deref()
        .and_then(|path| command_version(path, &["--version"]));
    EngineProbe {
        engine: "xtb".into(),
        installed: executable_path.is_some(),
        executable_path,
        version,
        license_status: "not_required_or_unchecked".into(),
        supported_tasks: Vec::new(),
        warnings: vec!["xTB execution adapter is not implemented yet".into()],
    }
}
