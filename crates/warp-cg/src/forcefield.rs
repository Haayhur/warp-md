use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::agent::ForcefieldRequest;

pub const FORCEFIELD_MANIFEST_SCHEMA: &str = "warp-cg.forcefield-manifest.v1";
pub const MARTINI3_KIND: &str = "martini3";
pub const MARTINI3_VERSION: &str = "3.0.0";
pub const MARTINI3_DEFAULT_INCLUDE: &str = "martini_v3.0.0.itp";
pub const MARTINI3_DEFAULT_DEST: &str = "forcefields/martini3";
pub const MANIFEST_FILE: &str = "warp_cg_forcefield_manifest.json";

const MARTINI3_FILES: &[&str] = &[
    "LICENSE",
    "NOTICE.md",
    "martini_v3.0.0.itp",
    "martini_v3.0.0_ions_v1.itp",
    "martini_v3.0.0_nucleobases_v1.itp",
    "martini_v3.0.0_phospholipids_v1.itp",
    "martini_v3.0.0_small_molecules_v1.itp",
    "martini_v3.0.0_solvents_v1.itp",
    "martini_v3.0.0_sugars_v1.itp",
    "martini_v3.0_sterols_v1.0.itp",
];

#[derive(Clone, Debug)]
pub struct MaterializedForcefield {
    pub root: PathBuf,
    pub manifest_path: PathBuf,
    pub include_paths: Vec<String>,
}

pub fn bundled_forcefield_root(kind: &str) -> Result<PathBuf> {
    if kind != MARTINI3_KIND {
        bail!("unsupported bundled forcefield kind '{kind}'");
    }
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("assets")
        .join("forcefields")
        .join("martini3");
    ensure_martini3_files(&root)?;
    Ok(root)
}

pub fn bundled_manifest_json(kind: &str) -> Result<Value> {
    let root = bundled_forcefield_root(kind)?;
    manifest_json(
        kind,
        MARTINI3_VERSION,
        "bundled",
        &root,
        MARTINI3_DEFAULT_INCLUDE,
    )
}

pub fn install_bundled_forcefield(kind: &str, dest: &Path, overwrite: bool) -> Result<Value> {
    if kind != MARTINI3_KIND {
        bail!("unsupported bundled forcefield kind '{kind}'");
    }
    let root = bundled_forcefield_root(kind)?;
    copy_forcefield_files(&root, dest, MARTINI3_FILES, overwrite)?;
    let manifest = manifest_json(
        kind,
        MARTINI3_VERSION,
        "bundled_install",
        dest,
        MARTINI3_DEFAULT_INCLUDE,
    )?;
    std::fs::write(
        dest.join(MANIFEST_FILE),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    Ok(manifest)
}

pub fn materialize_request_forcefield(
    request: &ForcefieldRequest,
    out_dir: &Path,
) -> Result<MaterializedForcefield> {
    validate_request_forcefield(request)?;
    let source_root = source_root(request)?;
    let source_version = request
        .version
        .as_deref()
        .unwrap_or_else(|| default_version(&request.kind));
    let include_files = include_files(request);
    let materialize = request.materialize.as_str();
    let (root, include_paths) = if materialize == "copy" {
        let dest = out_dir.join(
            request
                .dest
                .as_deref()
                .unwrap_or_else(|| default_dest(&request.kind)),
        );
        let files = files_for_kind(&request.kind)?;
        copy_forcefield_files(
            &source_root,
            &dest,
            files,
            request.overwrite.unwrap_or(true),
        )?;
        let includes = include_files
            .iter()
            .map(|file| {
                PathBuf::from(
                    request
                        .dest
                        .as_deref()
                        .unwrap_or_else(|| default_dest(&request.kind)),
                )
                .join(file)
                .to_string_lossy()
                .replace('\\', "/")
            })
            .collect::<Vec<_>>();
        (dest, includes)
    } else {
        let includes = include_files
            .iter()
            .map(|file| source_root.join(file).to_string_lossy().to_string())
            .collect::<Vec<_>>();
        (source_root, includes)
    };
    ensure_include_files(&root, &include_files)?;
    std::fs::create_dir_all(out_dir)?;
    let manifest = manifest_json(
        &request.kind,
        source_version,
        &request.source,
        &root,
        include_files
            .first()
            .map(String::as_str)
            .unwrap_or_else(|| default_include(&request.kind)),
    )?;
    let manifest_path = if materialize == "copy" {
        root.join(MANIFEST_FILE)
    } else {
        out_dir.join(MANIFEST_FILE)
    };
    std::fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(MaterializedForcefield {
        root,
        manifest_path,
        include_paths,
    })
}

pub fn validate_request_forcefield(request: &ForcefieldRequest) -> Result<()> {
    if request.kind != MARTINI3_KIND {
        bail!("forcefield.kind must be martini3");
    }
    if request.source != "bundled" && request.source != "path" {
        bail!("forcefield.source must be bundled or path");
    }
    if request.source == "path" {
        let Some(path) = request.path.as_deref() else {
            bail!("forcefield.path is required when forcefield.source=path");
        };
        if path.trim().is_empty() {
            bail!("forcefield.path must not be empty");
        }
    }
    if request.source == "bundled" && request.path.is_some() {
        bail!("forcefield.path is only accepted when forcefield.source=path");
    }
    if request.materialize != "copy" && request.materialize != "none" {
        bail!("forcefield.materialize must be copy or none");
    }
    if request.source == "bundled" && request.materialize == "none" {
        bail!("forcefield.materialize=none requires forcefield.source=path");
    }
    if let Some(dest) = request.dest.as_deref() {
        if dest.trim().is_empty() {
            bail!("forcefield.dest must not be empty");
        }
        if Path::new(dest).is_absolute() {
            bail!("forcefield.dest must be relative to output.out_dir");
        }
    }
    for include in &request.include_files {
        validate_relative_file(include, "forcefield.include_files")?;
    }
    Ok(())
}

pub fn forcefield_path_exists(request: &ForcefieldRequest) -> bool {
    source_root(request).is_ok()
}

fn source_root(request: &ForcefieldRequest) -> Result<PathBuf> {
    match request.source.as_str() {
        "bundled" => bundled_forcefield_root(&request.kind),
        "path" => {
            let path = PathBuf::from(
                request
                    .path
                    .as_deref()
                    .ok_or_else(|| anyhow!("forcefield.path is required"))?,
            );
            ensure_martini3_files(&path)?;
            Ok(path)
        }
        other => bail!("unsupported forcefield.source '{other}'"),
    }
}

fn copy_forcefield_files(
    source: &Path,
    dest: &Path,
    files: &[&str],
    overwrite: bool,
) -> Result<()> {
    std::fs::create_dir_all(dest)
        .with_context(|| format!("failed to create forcefield directory '{}'", dest.display()))?;
    for file in files {
        let source_path = source.join(file);
        let dest_path = dest.join(file);
        if dest_path.exists() && !overwrite {
            bail!(
                "forcefield destination '{}' already exists; pass overwrite=true to replace it",
                dest_path.display()
            );
        }
        std::fs::copy(&source_path, &dest_path).with_context(|| {
            format!(
                "failed to copy forcefield file '{}' to '{}'",
                source_path.display(),
                dest_path.display()
            )
        })?;
    }
    Ok(())
}

fn manifest_json(
    kind: &str,
    version: &str,
    source: &str,
    root: &Path,
    default_include: &str,
) -> Result<Value> {
    let files = files_for_kind(kind)?
        .iter()
        .map(|file| file_manifest(root, file))
        .collect::<Result<Vec<_>>>()?;
    Ok(json!({
        "schema_version": FORCEFIELD_MANIFEST_SCHEMA,
        "kind": kind,
        "version": version,
        "source": source,
        "root": root.to_string_lossy(),
        "default_include": default_include,
        "license": "Apache-2.0 for bundled Martini force-field parameter files; see NOTICE.md",
        "upstream": {
            "repository": "https://github.com/marrink-lab/martini-forcefields",
            "description": "Martini 3 force-field interaction and molecule parameter files"
        },
        "files": files
    }))
}

fn file_manifest(root: &Path, file: &str) -> Result<Value> {
    let path = root.join(file);
    let bytes = std::fs::read(&path)
        .with_context(|| format!("failed to read forcefield file '{}'", path.display()))?;
    Ok(json!({
        "path": file,
        "bytes": bytes.len(),
        "sha256": sha256_hex(&bytes)
    }))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn ensure_martini3_files(root: &Path) -> Result<()> {
    if !root.is_dir() {
        bail!(
            "Martini3 forcefield directory '{}' does not exist",
            root.display()
        );
    }
    for file in MARTINI3_FILES {
        let path = root.join(file);
        if !path.is_file() {
            bail!(
                "Martini3 forcefield file '{}' is missing from '{}'",
                file,
                root.display()
            );
        }
    }
    Ok(())
}

fn ensure_include_files(root: &Path, includes: &[String]) -> Result<()> {
    for include in includes {
        let path = root.join(include);
        if !path.is_file() {
            bail!(
                "forcefield include '{}' does not exist under '{}'",
                include,
                root.display()
            );
        }
    }
    Ok(())
}

fn include_files(request: &ForcefieldRequest) -> Vec<String> {
    if request.include_files.is_empty() {
        vec![default_include(&request.kind).to_string()]
    } else {
        request.include_files.clone()
    }
}

fn files_for_kind(kind: &str) -> Result<&'static [&'static str]> {
    match kind {
        MARTINI3_KIND => Ok(MARTINI3_FILES),
        other => bail!("unsupported forcefield kind '{other}'"),
    }
}

fn default_include(kind: &str) -> &'static str {
    match kind {
        MARTINI3_KIND => MARTINI3_DEFAULT_INCLUDE,
        _ => MARTINI3_DEFAULT_INCLUDE,
    }
}

fn default_dest(kind: &str) -> &'static str {
    match kind {
        MARTINI3_KIND => MARTINI3_DEFAULT_DEST,
        _ => MARTINI3_DEFAULT_DEST,
    }
}

fn default_version(kind: &str) -> &'static str {
    match kind {
        MARTINI3_KIND => MARTINI3_VERSION,
        _ => MARTINI3_VERSION,
    }
}

fn validate_relative_file(value: &str, field: &str) -> Result<()> {
    if value.trim().is_empty() {
        bail!("{field} entries must not be empty");
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path
            .components()
            .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        bail!("{field} entries must be relative file paths inside the forcefield directory");
    }
    Ok(())
}
