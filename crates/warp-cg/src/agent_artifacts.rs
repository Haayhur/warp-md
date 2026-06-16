use std::path::Path;

use anyhow::Result;
use serde_json::json;

use crate::forcefield::materialize_request_forcefield;
use crate::parameters::{AngleStats, BondStats, DihedralStats};

use super::agent_render::render_martini_top;
use super::{CgArtifact, CgRequest};

pub(super) fn write_bonded_stats(
    name: &str,
    out_dir: &Path,
    artifacts: &mut Vec<CgArtifact>,
    bond_stats: &[BondStats],
    angle_stats: &[AngleStats],
    dihedral_stats: &[DihedralStats],
) -> Result<()> {
    if !bond_stats.is_empty() {
        let stats_path = out_dir.join(format!("{name}_bond_stats.json"));
        std::fs::write(&stats_path, serde_json::to_vec_pretty(bond_stats)?)?;
        artifacts.push(CgArtifact {
            path: stats_path.to_string_lossy().to_string(),
            kind: "bond_stats_json".to_string(),
        });
    }
    if !angle_stats.is_empty() || !dihedral_stats.is_empty() {
        let stats_path = out_dir.join(format!("{name}_bonded_stats.json"));
        std::fs::write(
            &stats_path,
            serde_json::to_vec_pretty(&json!({
                "bonds": bond_stats,
                "angles": angle_stats,
                "dihedrals": dihedral_stats,
            }))?,
        )?;
        artifacts.push(CgArtifact {
            path: stats_path.to_string_lossy().to_string(),
            kind: "bonded_stats_json".to_string(),
        });
    }
    Ok(())
}

pub(super) fn write_topology_top(
    request: &CgRequest,
    out_dir: &Path,
    artifacts: &mut Vec<CgArtifact>,
) -> Result<()> {
    if request.output.write_topology_top {
        let top_path = out_dir.join(format!("{}_martini.top", request.name));
        let itp_file = format!("{}_martini.itp", request.name);
        let forcefield = request
            .forcefield
            .as_ref()
            .map(|forcefield| materialize_request_forcefield(forcefield, out_dir))
            .transpose()?;
        if let Some(forcefield) = &forcefield {
            artifacts.push(CgArtifact {
                path: forcefield.manifest_path.to_string_lossy().to_string(),
                kind: "forcefield_manifest_json".to_string(),
            });
            artifacts.push(CgArtifact {
                path: forcefield.root.to_string_lossy().to_string(),
                kind: "forcefield_directory".to_string(),
            });
        }
        let include_paths = forcefield
            .as_ref()
            .map(|forcefield| forcefield.include_paths.as_slice())
            .unwrap_or(&[]);
        let top = render_martini_top(&request.name, &itp_file, include_paths);
        std::fs::write(&top_path, top)?;
        artifacts.push(CgArtifact {
            path: top_path.to_string_lossy().to_string(),
            kind: "martini_topology_top".to_string(),
        });
    }
    Ok(())
}
