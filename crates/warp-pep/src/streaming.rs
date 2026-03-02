//! NDJSON streaming events for warp-pep agent integration.
//!
//! Emits progress events to stderr for agent consumption during
//! peptide building and mutation operations.

use std::time::Duration;

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

/// Streaming emitter for NDJSON events.
///
/// Emits events to stderr when enabled.
#[derive(Debug, Clone, Copy)]
pub struct StreamEmitter {
    enabled: bool,
}

impl StreamEmitter {
    /// Create a new emitter.
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Create a disabled emitter.
    pub fn disabled() -> Self {
        Self { enabled: false }
    }

    /// Create an enabled emitter.
    pub fn enabled() -> Self {
        Self { enabled: true }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn emit_json(&self, json: &str) {
        if self.enabled {
            eprintln!("{}", json);
        }
    }

    pub fn emit_operation_started(&self, event: &OperationStartedEvent) {
        let input = serde_json::to_string(&event.input_path).unwrap_or("null".to_string());
        let mutations = event
            .total_mutations
            .map(|n| n.to_string())
            .unwrap_or("null".to_string());
        let json = format!(
            r#"{{"event":"operation_started","operation":"{}","input_path":{},"total_chains":{},"total_residues":{},"total_mutations":{}}}"#,
            event.operation.as_str(),
            input,
            event.total_chains,
            event.total_residues,
            mutations
        );
        self.emit_json(&json);
    }

    pub fn emit_chain_complete(&self, event: &ChainCompleteEvent) {
        let json = format!(
            r#"{{"event":"chain_complete","chain_index":{},"chain_id":"{}","residue_count":{},"elapsed_ms":{}}}"#,
            event.chain_index, event.chain_id, event.residue_count, event.elapsed_ms
        );
        self.emit_json(&json);
    }

    pub fn emit_mutation_complete(&self, event: &MutationCompleteEvent) {
        let progress_pct = if event.total_mutations > 0 {
            (event.mutation_index as f64 / event.total_mutations as f64 * 100.0)
        } else {
            0.0
        };
        let json = format!(
            r#"{{"event":"mutation_complete","mutation_index":{},"total_mutations":{},"mutation_spec":"{}","successful":{},"elapsed_ms":{},"progress_pct":{:.1}}}"#,
            event.mutation_index,
            event.total_mutations,
            event.mutation_spec,
            event.successful,
            event.elapsed_ms,
            progress_pct
        );
        self.emit_json(&json);
    }

    pub fn emit_operation_complete(&self, event: &OperationCompleteEvent) {
        let output = serde_json::to_string(&event.output_path).unwrap_or("null".to_string());
        let json = format!(
            r#"{{"event":"operation_complete","operation":"{}","total_atoms":{},"total_residues":{},"total_chains":{},"output_path":{},"elapsed_ms":{}}}"#,
            event.operation.as_str(),
            event.total_atoms,
            event.total_residues,
            event.total_chains,
            output,
            event.elapsed_ms
        );
        self.emit_json(&json);
    }

    pub fn emit_error(&self, code: &str, message: &str) {
        let json = format!(
            r#"{{"event":"error","code":"{}","message":"{}"}}"#,
            code, message
        );
        self.emit_json(&json);
    }
}

pub fn duration_ms(d: Duration) -> u64 {
    d.as_millis().try_into().unwrap_or(u64::MAX)
}
