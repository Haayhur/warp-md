use std::path::Path;

pub fn resolve_relative_path(base_path: impl AsRef<Path>, value: &str) -> String {
    let path = Path::new(value);
    if path.is_absolute() {
        value.to_string()
    } else {
        base_path
            .as_ref()
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(path)
            .to_string_lossy()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn build_style_relative_path_uses_base_parent() {
        let base = PathBuf::from("/tmp/build/config.toml");
        let resolved = resolve_relative_path(&base, "inputs/system.pdb");
        assert_eq!(
            PathBuf::from(resolved),
            PathBuf::from("/tmp/build").join("inputs/system.pdb")
        );
    }

    #[test]
    fn pack_style_relative_path_uses_base_parent() {
        let base = PathBuf::from("configs/pack/input.yaml");
        let resolved = resolve_relative_path(&base, "../assets/template.pdb");
        assert_eq!(
            PathBuf::from(resolved),
            PathBuf::from("configs/pack").join("../assets/template.pdb")
        );
    }

    #[test]
    fn absolute_path_is_preserved() {
        let absolute = if cfg!(windows) {
            r"C:\tmp\input.xtc"
        } else {
            "/tmp/input.xtc"
        };
        assert_eq!(resolve_relative_path("config.toml", absolute), absolute);
    }
}
