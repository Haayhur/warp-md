use super::*;

pub(super) fn write_manifest(result: &BuildResult) -> Result<()> {
    prepare_output_path(
        &result.artifacts.manifest,
        result.artifacts.output_policy.overwrite,
        result.artifacts.output_policy.backup_existing,
    )?;
    let path = Path::new(&result.artifacts.manifest);
    std::fs::write(path, serde_json::to_string_pretty(result)?)?;
    Ok(())
}

pub(super) fn prepare_output_path(
    path: &str,
    overwrite: bool,
    backup_existing: bool,
) -> Result<()> {
    ensure_parent_dir(path)?;
    let path_ref = Path::new(path);
    if !path_ref.exists() {
        return Ok(());
    }
    if !overwrite {
        return Err(anyhow!(
            "output path {path} already exists and outputs.overwrite is false"
        ));
    }
    if backup_existing {
        let backup = backup_path(path_ref);
        std::fs::copy(path_ref, &backup)?;
    }
    Ok(())
}

fn backup_path(path: &Path) -> std::path::PathBuf {
    let mut idx = 1usize;
    loop {
        let candidate = path.with_extension(match path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) if !ext.is_empty() => format!("{ext}.bak{idx}"),
            _ => format!("bak{idx}"),
        });
        if !candidate.exists() {
            return candidate;
        }
        idx += 1;
    }
}

fn ensure_parent_dir(path: &str) -> Result<()> {
    let path = Path::new(path);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}
