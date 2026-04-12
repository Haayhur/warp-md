//! NDJSON streaming events for warp-pep agent integration.

use serde_json::json;
pub use warp_structure::ndjson::duration_ms;
use warp_structure::ndjson::NdjsonEmitter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PepOperation {
    Build,
    Mutate,
    Convert,
}

impl PepOperation {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Mutate => "mutate",
            Self::Convert => "convert",
        }
    }
}

#[derive(Debug, Clone)]
pub struct OperationStartedEvent {
    pub operation: PepOperation,
    pub input_path: Option<String>,
    pub total_chains: usize,
    pub total_residues: usize,
    pub total_mutations: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct ChainCompleteEvent {
    pub chain_index: usize,
    pub chain_id: String,
    pub residue_count: usize,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone)]
pub struct MutationCompleteEvent {
    pub mutation_index: usize,
    pub total_mutations: usize,
    pub mutation_spec: String,
    pub successful: bool,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone)]
pub struct OperationCompleteEvent {
    pub operation: PepOperation,
    pub total_atoms: usize,
    pub total_residues: usize,
    pub total_chains: usize,
    pub output_path: Option<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct StreamEmitter {
    inner: NdjsonEmitter,
}

impl StreamEmitter {
    pub fn new(enabled: bool) -> Self {
        Self {
            inner: NdjsonEmitter::new(enabled),
        }
    }

    pub fn disabled() -> Self {
        Self {
            inner: NdjsonEmitter::disabled(),
        }
    }

    pub fn enabled() -> Self {
        Self {
            inner: NdjsonEmitter::enabled(),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }

    pub fn emit_operation_started(&self, event: &OperationStartedEvent) {
        self.inner.emit(&json!({
            "event": "operation_started",
            "operation": event.operation.as_str(),
            "input_path": event.input_path,
            "total_chains": event.total_chains,
            "total_residues": event.total_residues,
            "total_mutations": event.total_mutations,
        }));
    }

    pub fn emit_chain_complete(&self, event: &ChainCompleteEvent) {
        self.inner.emit(&json!({
            "event": "chain_complete",
            "chain_index": event.chain_index,
            "chain_id": event.chain_id,
            "residue_count": event.residue_count,
            "elapsed_ms": event.elapsed_ms,
        }));
    }

    pub fn emit_mutation_complete(&self, event: &MutationCompleteEvent) {
        let progress_pct = if event.total_mutations > 0 {
            event.mutation_index as f64 / event.total_mutations as f64 * 100.0
        } else {
            0.0
        };
        self.inner.emit(&json!({
            "event": "mutation_complete",
            "mutation_index": event.mutation_index,
            "total_mutations": event.total_mutations,
            "mutation_spec": event.mutation_spec,
            "successful": event.successful,
            "elapsed_ms": event.elapsed_ms,
            "progress_pct": progress_pct,
        }));
    }

    pub fn emit_operation_complete(&self, event: &OperationCompleteEvent) {
        self.inner.emit(&json!({
            "event": "operation_complete",
            "operation": event.operation.as_str(),
            "total_atoms": event.total_atoms,
            "total_residues": event.total_residues,
            "total_chains": event.total_chains,
            "output_path": event.output_path,
            "elapsed_ms": event.elapsed_ms,
        }));
    }

    pub fn emit_error(&self, code: &str, message: &str) {
        self.inner.emit(&json!({
            "event": "error",
            "code": code,
            "message": message,
        }));
    }
}
