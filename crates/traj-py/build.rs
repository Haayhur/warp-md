fn main() {
    if cfg!(target_os = "linux") && std::env::var_os("CARGO_FEATURE_EXTENSION_MODULE").is_none() {
        let config = pyo3_build_config::get();
        if config.shared {
            if let Some(lib_dir) = &config.lib_dir {
                println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
            }
        }
    }
}
