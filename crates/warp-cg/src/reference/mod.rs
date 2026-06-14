use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

use crate::bonded_terms::BondedTermSet;
use crate::parameters::{AngleStats, BondStats, BondedStats, DihedralStats};
use crate::trajectory::{BeadMapping, NativeTrajectoryOptions};
use crate::xtb::{run_xtb_pipeline_with_config, XtbRunConfig};
pub use extractor::{
    TargetExtraction, TargetExtractionRequest, TargetExtractor, TrajectoryTargetExtractor,
};
pub use target::{
    ReferenceBinConfig, ReferenceDistributionTarget, ReferenceScore, ReferenceTargetSet,
    ReferenceTermKind, ReferenceTermScore, ReferenceTransformConfig,
};

pub mod extractor;
#[cfg(test)]
mod extractor_tests;
#[cfg(test)]
mod precomputed_tests;
pub mod target;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone)]
pub struct ReferenceData {
    pub source_kind: String,
    pub bonded_stats: BondedStats,
    pub target_set: Option<ReferenceTargetSet>,
    pub first_cg_coords: Option<Vec<[f32; 3]>>,
    pub mapped_trajectory: Option<PathBuf>,
    pub artifacts: Vec<ReferenceArtifact>,
    pub metrics: BTreeMap<String, f64>,
    pub metadata: ReferenceMetadata,
}

#[derive(Debug, Clone)]
pub struct ReferenceArtifact {
    pub path: PathBuf,
    pub kind: String,
}

#[derive(Debug, Clone)]
pub struct ReferenceMetricSource {
    pub path: PathBuf,
    pub kind: String,
    pub namespace: Option<String>,
    pub artifact_kind: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ReferenceMetadata {
    pub frames_read: usize,
    pub frames_written: usize,
    pub source_path: Option<PathBuf>,
    pub mapped_by: String,
}

pub trait ReferenceProvider {
    fn load_reference(&mut self, request: &ReferenceRequest<'_>) -> Result<ReferenceData>;
}

pub struct ReferenceRequest<'a> {
    pub name: &'a str,
    pub out_dir: &'a Path,
    pub mapped_trajectory_name: Option<&'a str>,
    pub mapping: &'a BeadMapping,
    pub connections: &'a [(usize, usize)],
    pub term_set: Option<&'a BondedTermSet>,
    pub metric_sources: &'a [ReferenceMetricSource],
    pub transform: Option<&'a ReferenceTransformConfig>,
}

pub struct PrecomputedStatsReferenceProvider {
    source_kind: String,
    bonded_stats: BondedStats,
    target_set: Option<ReferenceTargetSet>,
    target_set_path: Option<PathBuf>,
    first_cg_coords: Option<Vec<[f32; 3]>>,
}

impl PrecomputedStatsReferenceProvider {
    pub fn new(source_kind: impl Into<String>, bonded_stats: BondedStats) -> Self {
        Self {
            source_kind: source_kind.into(),
            bonded_stats,
            target_set: None,
            target_set_path: None,
            first_cg_coords: None,
        }
    }

    pub fn with_target_set(mut self, target_set: ReferenceTargetSet) -> Self {
        if self.bonded_stats.bonds.is_empty()
            && self.bonded_stats.angles.is_empty()
            && self.bonded_stats.dihedrals.is_empty()
        {
            self.bonded_stats = bonded_stats_from_targets(&target_set);
        }
        self.target_set = Some(target_set);
        self
    }

    pub fn with_target_set_path(mut self, target_set_path: impl Into<PathBuf>) -> Self {
        self.target_set_path = Some(target_set_path.into());
        self
    }

    pub fn with_first_cg_coords(mut self, first_cg_coords: Option<Vec<[f32; 3]>>) -> Self {
        self.first_cg_coords = first_cg_coords;
        self
    }
}

impl ReferenceProvider for PrecomputedStatsReferenceProvider {
    fn load_reference(&mut self, request: &ReferenceRequest<'_>) -> Result<ReferenceData> {
        let target_set = if let Some(target_set) = self.target_set.clone() {
            Some(target_set)
        } else if let Some(path) = self.target_set_path.as_ref() {
            let data = std::fs::read_to_string(path).with_context(|| {
                format!(
                    "failed to read precomputed reference target set '{}'",
                    path.display()
                )
            })?;
            let target_set: ReferenceTargetSet =
                serde_json::from_str(&data).with_context(|| {
                    format!(
                        "failed to parse precomputed reference target set '{}'",
                        path.display()
                    )
                })?;
            Some(target_set)
        } else {
            None
        };
        let bonded_stats = if self.bonded_stats.bonds.is_empty()
            && self.bonded_stats.angles.is_empty()
            && self.bonded_stats.dihedrals.is_empty()
        {
            target_set
                .as_ref()
                .map(bonded_stats_from_targets)
                .unwrap_or_default()
        } else {
            self.bonded_stats.clone()
        };
        let mut artifacts = Vec::new();
        if let Some(path) = self.target_set_path.as_ref() {
            artifacts.push(ReferenceArtifact {
                path: path.clone(),
                kind: "reference_targets_json".to_string(),
            });
        }
        let mut metrics = BTreeMap::new();
        merge_metric_sources(request.metric_sources, &mut metrics, &mut artifacts)?;
        Ok(ReferenceData {
            source_kind: self.source_kind.clone(),
            bonded_stats,
            target_set,
            first_cg_coords: self.first_cg_coords.clone(),
            mapped_trajectory: None,
            artifacts,
            metrics,
            metadata: ReferenceMetadata {
                mapped_by: "precomputed_stats".to_string(),
                ..ReferenceMetadata::default()
            },
        })
    }
}

fn bonded_stats_from_targets(targets: &ReferenceTargetSet) -> BondedStats {
    BondedStats {
        bonds: targets
            .bonds
            .iter()
            .filter_map(|target| {
                let beads = target.beads.as_slice();
                Some(BondStats {
                    bead_i: *beads.first()?,
                    bead_j: *beads.get(1)?,
                    mean: target.mean,
                    std: target.std,
                    samples: target.samples,
                })
            })
            .collect(),
        angles: targets
            .angles
            .iter()
            .filter_map(|target| {
                let beads = target.beads.as_slice();
                Some(AngleStats {
                    bead_i: *beads.first()?,
                    bead_j: *beads.get(1)?,
                    bead_k: *beads.get(2)?,
                    mean_deg: target.mean,
                    std_deg: target.std,
                    samples: target.samples,
                })
            })
            .collect(),
        dihedrals: targets
            .dihedrals
            .iter()
            .filter_map(|target| {
                let beads = target.beads.as_slice();
                Some(DihedralStats {
                    bead_i: *beads.first()?,
                    bead_j: *beads.get(1)?,
                    bead_k: *beads.get(2)?,
                    bead_l: *beads.get(3)?,
                    mean_deg: target.mean,
                    std_deg: target.std,
                    samples: target.samples,
                })
            })
            .collect(),
    }
}

pub struct TrajectoryReferenceProvider {
    source_kind: String,
    trajectory_path: PathBuf,
    options: NativeTrajectoryOptions,
}

impl TrajectoryReferenceProvider {
    pub fn new(
        source_kind: impl Into<String>,
        trajectory_path: impl Into<PathBuf>,
        options: NativeTrajectoryOptions,
    ) -> Self {
        Self {
            source_kind: source_kind.into(),
            trajectory_path: trajectory_path.into(),
            options,
        }
    }
}

impl ReferenceProvider for TrajectoryReferenceProvider {
    fn load_reference(&mut self, request: &ReferenceRequest<'_>) -> Result<ReferenceData> {
        let mut extractor =
            TrajectoryTargetExtractor::new(&self.trajectory_path, self.options.clone());
        let mut extraction = extractor.extract_targets(&TargetExtractionRequest {
            name: request.name,
            out_dir: request.out_dir,
            mapped_trajectory_name: request.mapped_trajectory_name,
            mapping: request.mapping,
            connections: request.connections,
            term_set: request.term_set,
            transform: request.transform,
        })?;
        merge_metric_sources(
            request.metric_sources,
            &mut extraction.metrics,
            &mut extraction.artifacts,
        )?;
        Ok(ReferenceData {
            source_kind: self.source_kind.clone(),
            bonded_stats: extraction.bonded_stats,
            target_set: Some(extraction.target_set),
            first_cg_coords: extraction.first_cg_coords,
            mapped_trajectory: extraction.mapped_trajectory,
            artifacts: extraction.artifacts,
            metrics: extraction.metrics,
            metadata: extraction.metadata,
        })
    }
}

pub struct XtbReferenceProvider {
    smiles: String,
    config: XtbRunConfig,
    work_dir: Option<PathBuf>,
    trajectory_options: NativeTrajectoryOptions,
}

impl XtbReferenceProvider {
    pub fn new(
        smiles: impl Into<String>,
        config: XtbRunConfig,
        work_dir: Option<PathBuf>,
        trajectory_options: NativeTrajectoryOptions,
    ) -> Self {
        Self {
            smiles: smiles.into(),
            config,
            work_dir,
            trajectory_options,
        }
    }
}

impl ReferenceProvider for XtbReferenceProvider {
    fn load_reference(&mut self, request: &ReferenceRequest<'_>) -> Result<ReferenceData> {
        let work_dir = self
            .work_dir
            .clone()
            .unwrap_or_else(|| request.out_dir.to_path_buf());
        let result =
            run_xtb_pipeline_with_config(request.name, &self.smiles, &work_dir, &self.config)?;
        let reference_path = result
            .trajectory_trj
            .clone()
            .unwrap_or_else(|| result.opt_xyz.clone());
        let mut provider = TrajectoryReferenceProvider::new(
            "xtb",
            reference_path,
            self.trajectory_options.clone(),
        );
        let mut data = provider.load_reference(request)?;
        data.artifacts.push(ReferenceArtifact {
            path: result.opt_xyz,
            kind: "xtb_optimized_xyz".to_string(),
        });
        if let Some(trj) = result.trajectory_trj {
            data.artifacts.push(ReferenceArtifact {
                path: trj,
                kind: "xtb_reference_trajectory".to_string(),
            });
        }
        data.source_kind = "xtb".to_string();
        data.metadata.mapped_by = "xtb_then_trajectory".to_string();
        Ok(data)
    }
}

fn merge_metric_sources(
    sources: &[ReferenceMetricSource],
    metrics: &mut BTreeMap<String, f64>,
    artifacts: &mut Vec<ReferenceArtifact>,
) -> Result<()> {
    for source in sources {
        if source.kind != "json" {
            return Err(anyhow!(
                "reference metric source kind '{}' is unsupported; expected json",
                source.kind
            ));
        }
        let data = std::fs::read_to_string(&source.path).with_context(|| {
            format!(
                "failed to read reference metric source '{}'",
                source.path.display()
            )
        })?;
        let value: Value = serde_json::from_str(&data).with_context(|| {
            format!(
                "failed to parse reference metric source '{}' as JSON",
                source.path.display()
            )
        })?;
        merge_metric_json(&value, source, metrics, artifacts)?;
        artifacts.push(ReferenceArtifact {
            path: source.path.clone(),
            kind: source
                .artifact_kind
                .clone()
                .unwrap_or_else(|| "reference_metrics_json".to_string()),
        });
    }
    Ok(())
}

fn merge_metric_json(
    value: &Value,
    source: &ReferenceMetricSource,
    metrics: &mut BTreeMap<String, f64>,
    artifacts: &mut Vec<ReferenceArtifact>,
) -> Result<()> {
    let metric_object = value
        .get("metrics")
        .and_then(Value::as_object)
        .or_else(|| value.as_object())
        .ok_or_else(|| {
            anyhow!(
                "reference metric source '{}' must be a JSON object or contain a metrics object",
                source.path.display()
            )
        })?;

    for (key, value) in metric_object {
        if key == "metrics" || key == "artifacts" {
            continue;
        }
        if let Some(number) = value.as_f64().filter(|number| number.is_finite()) {
            metrics.insert(metric_key(source.namespace.as_deref(), key), number);
        }
    }

    if let Some(source_artifacts) = value.get("artifacts").and_then(Value::as_array) {
        let base_dir = source.path.parent().unwrap_or_else(|| Path::new("."));
        for artifact in source_artifacts {
            let Some(path) = artifact.get("path").and_then(Value::as_str) else {
                continue;
            };
            let Some(kind) = artifact.get("kind").and_then(Value::as_str) else {
                continue;
            };
            let artifact_path = PathBuf::from(path);
            artifacts.push(ReferenceArtifact {
                path: if artifact_path.is_absolute() {
                    artifact_path
                } else {
                    base_dir.join(artifact_path)
                },
                kind: kind.to_string(),
            });
        }
    }
    Ok(())
}

fn metric_key(namespace: Option<&str>, key: &str) -> String {
    namespace
        .filter(|namespace| !namespace.is_empty())
        .map(|namespace| format!("{namespace}.{key}"))
        .unwrap_or_else(|| key.to_string())
}
