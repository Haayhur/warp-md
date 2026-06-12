use std::path::Path;

use super::{find_executable, EngineProbe};

pub fn probe() -> EngineProbe {
    let executable_path = std::env::var("WARP_QM_MULTIWFN")
        .ok()
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
        .or_else(|| find_executable("multiwfn"));
    let mut warnings = vec![
        "Multiwfn must be installed separately; warp-qm drives it through deterministic stdin scripts"
            .into(),
        "Multiwfn has no noninteractive --version probe; set WARP_QM_MULTIWFN_LIB_DIR when shared libraries are outside the system loader path".into(),
    ];
    if std::env::var_os("WARP_QM_MULTIWFN_LIB_DIR").is_none()
        && std::env::var_os("LD_LIBRARY_PATH").is_none()
    {
        warnings.push("Multiwfn shared-library path not configured in environment".into());
    }
    EngineProbe {
        engine: "multiwfn".into(),
        installed: executable_path.is_some(),
        executable_path,
        version: None,
        license_status: "external_license_required_unchecked".into(),
        supported_tasks: vec![
            "generic_run",
            "esp",
            "resp_fit",
            "resp_prepare",
            "resp_postprocess",
            "population",
            "orbital_cube",
            "electron_density_cube",
            "elf_cube",
            "lol_cube",
            "laplacian_cube",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
        warnings,
    }
}
