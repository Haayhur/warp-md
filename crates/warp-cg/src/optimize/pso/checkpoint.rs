use std::fs;
use std::path::Path;

use rand_chacha::ChaCha12Rng;
use serde::{Deserialize, Serialize};

use super::particle::Particle;
use crate::optimize::{EvaluationRecord, ParameterBound};

const CHECKPOINT_SCHEMA_VERSION: &str = "warp-cg.pso-checkpoint.v1";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct PsoCheckpoint {
    schema_version: String,
    seed: u64,
    bound_signature: Vec<BoundSignature>,
    pub(super) particles: Vec<Particle>,
    pub(super) global_best: Vec<f64>,
    pub(super) global_value: f64,
    pub(super) worst_value: f64,
    pub(super) evaluations: Vec<EvaluationRecord>,
    pub(super) iterations_without_global_best: usize,
    pub(super) rng: ChaCha12Rng,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BoundSignature {
    name: String,
    min: f64,
    max: f64,
}

impl PsoCheckpoint {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        seed: u64,
        bounds: &[ParameterBound],
        particles: Vec<Particle>,
        global_best: Vec<f64>,
        global_value: f64,
        worst_value: f64,
        evaluations: Vec<EvaluationRecord>,
        iterations_without_global_best: usize,
        rng: ChaCha12Rng,
    ) -> Self {
        Self {
            schema_version: CHECKPOINT_SCHEMA_VERSION.to_string(),
            seed,
            bound_signature: bound_signature(bounds),
            particles,
            global_best,
            global_value,
            worst_value,
            evaluations,
            iterations_without_global_best,
            rng,
        }
    }

    fn matches_problem(&self, seed: u64, bounds: &[ParameterBound]) -> bool {
        self.schema_version == CHECKPOINT_SCHEMA_VERSION
            && self.seed == seed
            && self.bound_signature == bound_signature(bounds)
    }
}

pub(super) fn load_checkpoint(
    path: &Path,
    seed: u64,
    bounds: &[ParameterBound],
) -> Option<PsoCheckpoint> {
    let bytes = fs::read(path).ok()?;
    let checkpoint: PsoCheckpoint = serde_json::from_slice(&bytes).ok()?;
    checkpoint
        .matches_problem(seed, bounds)
        .then_some(checkpoint)
}

pub(super) fn save_checkpoint(path: &Path, checkpoint: &PsoCheckpoint) {
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
