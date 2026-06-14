use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::bonded_terms::BondedTermSet;
use crate::parameters::{BondStats, BondValueSeries, BondedStats};
use crate::trajectory::{
    map_native_trajectory, map_native_trajectory_with_terms, BeadMapping, NativeTrajectoryOptions,
};

use super::{
    ReferenceArtifact, ReferenceBinConfig, ReferenceMetadata, ReferenceTargetSet,
    ReferenceTransformConfig,
};

pub struct TargetExtraction {
    pub bonded_stats: BondedStats,
    pub target_set: ReferenceTargetSet,
    pub first_cg_coords: Option<Vec<[f32; 3]>>,
    pub mapped_trajectory: Option<PathBuf>,
    pub artifacts: Vec<ReferenceArtifact>,
    pub metrics: BTreeMap<String, f64>,
    pub metadata: ReferenceMetadata,
}

pub trait TargetExtractor {
    fn extract_targets(
        &mut self,
        request: &TargetExtractionRequest<'_>,
    ) -> Result<TargetExtraction>;
}

pub struct TargetExtractionRequest<'a> {
    pub name: &'a str,
    pub out_dir: &'a Path,
    pub mapped_trajectory_name: Option<&'a str>,
    pub mapping: &'a BeadMapping,
    pub connections: &'a [(usize, usize)],
    pub term_set: Option<&'a BondedTermSet>,
    pub transform: Option<&'a ReferenceTransformConfig>,
}

pub struct TrajectoryTargetExtractor {
    source_path: PathBuf,
    options: NativeTrajectoryOptions,
}

impl TrajectoryTargetExtractor {
    pub fn new(source_path: impl Into<PathBuf>, options: NativeTrajectoryOptions) -> Self {
        Self {
            source_path: source_path.into(),
            options,
        }
    }
}

impl TargetExtractor for TrajectoryTargetExtractor {
    fn extract_targets(
        &mut self,
        request: &TargetExtractionRequest<'_>,
    ) -> Result<TargetExtraction> {
        let output_path = request
            .mapped_trajectory_name
            .map(|name| resolve_output_path(request.out_dir, name));
        let report = if let Some(terms) = request.term_set {
            map_native_trajectory_with_terms(
                &self.source_path,
                output_path.as_deref(),
                request.mapping,
                terms,
                &self.options,
            )?
        } else {
            map_native_trajectory(
                &self.source_path,
                output_path.as_deref(),
                request.mapping,
                request.connections,
                &self.options,
            )?
        };
        let bonded_values = request
            .transform
            .map(|transform| transform.apply(&report.bonded_values))
            .unwrap_or_else(|| report.bonded_values.clone());
        let target_set =
            ReferenceTargetSet::from_values(&bonded_values, ReferenceBinConfig::default());

        let mut artifacts = Vec::new();
        if let Some(path) = output_path.as_ref() {
            artifacts.push(ReferenceArtifact {
                path: path.clone(),
                kind: "coarse_grained_trajectory".to_string(),
            });
        }
        let target_path = request
            .out_dir
            .join(format!("{}_reference_targets.json", request.name));
        std::fs::write(&target_path, serde_json::to_vec_pretty(&target_set)?)?;
        artifacts.push(ReferenceArtifact {
            path: target_path,
            kind: "reference_targets_json".to_string(),
        });

        let mut metrics = BTreeMap::new();
        if let Some(rg) = &report.rg_stats {
            metrics.insert(
                "rg_mean_nm".to_string(),
                transformed_rg_mean(rg.mean, request.transform),
            );
            metrics.insert("rg_std_nm".to_string(), rg.std);
            metrics.insert("rg_samples".to_string(), rg.samples as f64);
        }
        if let Some(sasa) = &report.sasa_stats {
            metrics.insert("sasa_approx_mean_nm2".to_string(), sasa.mean);
            metrics.insert("sasa_approx_std_nm2".to_string(), sasa.std);
            metrics.insert("sasa_approx_samples".to_string(), sasa.samples as f64);
        }

        Ok(TargetExtraction {
            bonded_stats: BondedStats {
                bonds: if request.transform.is_some() {
                    bond_stats_from_series(&bonded_values.bonds)
                } else {
                    report.bond_stats
                },
                angles: report.angle_stats,
                dihedrals: report.dihedral_stats,
            },
            target_set,
            first_cg_coords: report.first_cg_coords,
            mapped_trajectory: output_path,
            artifacts,
            metrics,
            metadata: ReferenceMetadata {
                frames_read: report.frames_read,
                frames_written: report.frames_written,
                source_path: Some(self.source_path.clone()),
                mapped_by: "trajectory".to_string(),
            },
        })
    }
}

fn resolve_output_path(out_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        out_dir.join(candidate)
    }
}

fn transformed_rg_mean(mean: f64, transform: Option<&ReferenceTransformConfig>) -> f64 {
    mean + transform
        .and_then(|transform| transform.rg_offset_nm)
        .unwrap_or(0.0)
}

fn bond_stats_from_series(series: &[BondValueSeries]) -> Vec<BondStats> {
    series
        .iter()
        .filter_map(|series| {
            if series.values.is_empty() {
                return None;
            }
            let mean = series.values.iter().sum::<f64>() / series.values.len() as f64;
            let variance = series
                .values
                .iter()
                .map(|value| {
                    let delta = value - mean;
                    delta * delta
                })
                .sum::<f64>()
                / series.values.len() as f64;
            Some(BondStats {
                bead_i: series.bead_i,
                bead_j: series.bead_j,
                mean,
                std: variance.sqrt(),
                samples: series.values.len(),
            })
        })
        .collect()
}
