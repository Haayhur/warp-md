use std::fs;
use std::path::Path;

use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use crate::optimize::{EvaluationRecord, ParameterBound};

use super::settings::{AcquisitionKind, BoAlgorithm};

const CHECKPOINT_SCHEMA_VERSION: &str = "warp-cg.bo-checkpoint.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct BoCheckpoint {
    schema_version: String,
    algorithm: BoAlgorithm,
    acquisition: AcquisitionKind,
    seed: u64,
    bound_signature: Vec<BoundSignature>,
    objective_signature: String,
    evaluator_signature: Option<String>,
    pub(super) evaluations: Vec<EvaluationRecord>,
    pub(super) best_parameters: Vec<f64>,
    pub(super) best_value: f64,
    pub(super) rng: ChaCha12Rng,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BoundSignature {
    name: String,
    min: f64,
    max: f64,
}

impl BoCheckpoint {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        algorithm: BoAlgorithm,
        acquisition: AcquisitionKind,
        seed: u64,
        bounds: &[ParameterBound],
        objective_signature: String,
        evaluator_signature: Option<String>,
        evaluations: Vec<EvaluationRecord>,
        best_parameters: Vec<f64>,
        best_value: f64,
        rng: ChaCha12Rng,
    ) -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_string(),
            algorithm,
            acquisition,
            seed,
            bound_signature: bound_signature(bounds),
            objective_signature,
            evaluator_signature,
            evaluations,
            best_parameters,
            best_value,
            rng,
        }
    }

    fn matches_problem(
        &self,
        algorithm: BoAlgorithm,
        acquisition: AcquisitionKind,
        seed: u64,
        bounds: &[ParameterBound],
        objective_signature: &str,
        evaluator_signature: Option<&str>,
    ) -> bool {
        self.schema_version == CHECKPOINT_SCHEMA_VERSION
            && self.algorithm == algorithm
            && self.acquisition == acquisition
            && self.seed == seed
            && self.bound_signature == bound_signature(bounds)
            && self.objective_signature == objective_signature
            && self.evaluator_signature.as_deref() == evaluator_signature
    }
}

pub(super) fn load_checkpoint(
    path: &Path,
    algorithm: BoAlgorithm,
    acquisition: AcquisitionKind,
    seed: u64,
    bounds: &[ParameterBound],
    objective_signature: &str,
    evaluator_signature: Option<&str>,
) -> Option<BoCheckpoint> {
    let bytes = fs::read(path).ok()?;
    let checkpoint: BoCheckpoint = serde_json::from_slice(&bytes).ok()?;
    checkpoint
        .matches_problem(
            algorithm,
            acquisition,
            seed,
            bounds,
            objective_signature,
            evaluator_signature,
        )
        .then_some(checkpoint)
}

pub(super) fn save_checkpoint(path: &Path, checkpoint: &BoCheckpoint) {
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    if let Ok(bytes) = serde_json::to_vec_pretty(checkpoint) {
        let _ = fs::write(path, bytes);
    }
}

fn bound_signature(bounds: &[ParameterBound]) -> Vec<BoundSignature> {
    bounds
        .iter()
        .map(|bound| BoundSignature {
            name: bound.name.clone(),
            min: bound.min,
            max: bound.max,
        })
        .collect()
}
