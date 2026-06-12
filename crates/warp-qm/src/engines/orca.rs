use super::{common_tasks, find_executable, EngineProbe};

pub fn probe() -> EngineProbe {
    let executable_path = std::env::var("ORCA_BINARY")
        .ok()
        .or_else(|| std::env::var("WARP_QM_ORCA").ok())
        .or_else(|| find_executable("orca"));
    EngineProbe {
        engine: "orca".into(),
        installed: executable_path.is_some(),
        executable_path,
        version: None,
        license_status: "external_license_required_unchecked".into(),
        supported_tasks: orca_tasks(),
        warnings: vec![
            "ORCA must be installed separately; warp-qm never bundles it".into(),
            "ORCA does not expose a stable --version probe; version is recorded from outputs when available".into(),
        ],
    }
}

fn orca_tasks() -> Vec<String> {
    let mut tasks = common_tasks();
    tasks.extend(
        [
            "generic_run",
            "orca_molden_export",
            "nmr_shielding",
            "binding_energy",
            "solvation_energy",
            "proton_affinity",
        ]
        .into_iter()
        .map(str::to_string),
    );
    tasks
}
