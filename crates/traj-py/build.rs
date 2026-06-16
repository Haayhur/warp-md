fn main() {
    copy_warp_cg_forcefield_assets();
    if cfg!(target_os = "linux") && std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_none() {
        let config = pyo3_build_config::get();
        if config.shared {
            if let Some(lib_dir) = &config.lib_dir {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
            }
        }
    }
}

fn copy_warp_cg_forcefield_assets() {
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let source = manifest_dir
        .join("../warp-cg/assets/forcefields/martini3")
        .canonicalize()
        .expect("warp-cg Martini3 forcefield assets must exist");
    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dest = out_dir.join("forcefields/martini3");

    std::fs::create_dir_all(&dest).expect("failed to create package asset directory");
    for entry in std::fs::read_dir(&source).expect("failed to read Martini3 asset directory") {
        let entry = entry.expect("failed to read Martini3 asset entry");
        let path = entry.path();
        if path.is_file() {
            let target = dest.join(entry.file_name());
            std::fs::copy(&path, target).expect("failed to copy Martini3 package asset");
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}
