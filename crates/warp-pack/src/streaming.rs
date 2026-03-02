//! NDJSON streaming events for warp-pack agent integration.
//!
//! Emits progress events to stderr for agent consumption.
//!
//! Event types:
//!   - pack_started: Initial configuration
//!   - phase_started: New packing phase (placement, movebad, gencan, relax)
//!   - molecule_placed: Individual molecule placement progress
//!   - gencan_iteration: Per-iteration optimization progress
//!   - phase_complete: Phase finished with timing
//!   - pack_complete: Final result envelope

use std::time::Duration;

#[derive(Debug, Clone)]
pub struct PackStartedEvent {
    pub total_molecules: usize,
    pub box_size: [f32; 3],
    pub box_origin: [f32; 3],
    pub output_path: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingPhase {
    TemplateLoad,
    CorePlacement,
    MoveBad,
    GencanOptimization,
    Relax,
}

impl PackingPhase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::TemplateLoad => "template_load",
            Self::CorePlacement => "core_placement",
            Self::MoveBad => "movebad",
            Self::GencanOptimization => "gencan",
            Self::Relax => "relax",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PhaseStartedEvent {
    pub phase: PackingPhase,
    pub total_molecules: Option<usize>,
    pub max_iterations: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct MoleculePlacedEvent {
    pub molecule_index: usize,
    pub total_molecules: usize,
    pub molecule_name: String,
    pub successful: bool,
}

#[derive(Debug, Clone)]
pub struct GencanIterationEvent {
    pub iteration: usize,
    pub max_iterations: usize,
    pub obj_value: f32,
    pub obj_overlap: f32,
    pub obj_constraint: f32,
    pub pg_sup: f32,
    pub pg_norm: f32,
    pub elapsed_ms: u64,
}

#[derive(Debug, Clone)]
pub struct PhaseCompleteEvent {
    pub phase: PackingPhase,
    pub elapsed_ms: u64,
    pub iterations: Option<usize>,
    pub final_obj_value: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct PackCompleteEvent {
    pub total_atoms: usize,
    pub total_molecules: usize,
    pub final_box_size: [f32; 3],
    pub output_path: Option<String>,
    pub elapsed_ms: u64,
    pub profile: PackProfile,
}

#[derive(Debug, Clone, Default)]
pub struct PackProfile {
    pub templates_ms: u64,
    pub place_core_ms: u64,
    pub movebad_ms: u64,
    pub gencan_ms: u64,
    pub relax_ms: u64,
}

/// Streaming emitter for NDJSON events.
///
/// Emits events to stderr when enabled. This allows agents to monitor
/// progress without interfering with stdout output (which may contain
/// the final PDB/coordinates).
#[derive(Debug, Clone, Copy)]
pub struct StreamEmitter {
    enabled: bool,
}

impl StreamEmitter {
    /// Create a new emitter.
    ///
    /// Pass `true` to enable NDJSON streaming to stderr.
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Create a disabled emitter (no output).
    pub fn disabled() -> Self {
        Self { enabled: false }
    }

    /// Create an enabled emitter.
    pub fn enabled() -> Self {
        Self { enabled: true }
    }

    /// Check if streaming is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn emit_json(&self, json: &str) {
        if self.enabled {
            eprintln!("{}", json);
        }
    }

    pub fn emit_pack_started(&self, event: &PackStartedEvent) {
        let json = format!(
            r#"{{"event":"pack_started","total_molecules":{},"box_size":[{},{},{}],"box_origin":[{},{},{}],"output_path":{}}}"#,
            event.total_molecules,
            event.box_size[0],
            event.box_size[1],
            event.box_size[2],
            event.box_origin[0],
            event.box_origin[1],
            event.box_origin[2],
            serde_json::to_string(&event.output_path).unwrap_or("null".to_string())
        );
        self.emit_json(&json);
    }

    pub fn emit_phase_started(&self, event: &PhaseStartedEvent) {
        let total = event
            .total_molecules
            .map(|n| n.to_string())
            .unwrap_or("null".to_string());
        let maxit = event
            .max_iterations
            .map(|n| n.to_string())
            .unwrap_or("null".to_string());
        let json = format!(
            r#"{{"event":"phase_started","phase":"{}","total_molecules":{},"max_iterations":{}}}"#,
            event.phase.as_str(),
            total,
            maxit
        );
        self.emit_json(&json);
    }

    pub fn emit_molecule_placed(&self, event: &MoleculePlacedEvent) {
        let progress_pct = if event.total_molecules > 0 {
            event.molecule_index as f64 / event.total_molecules as f64 * 100.0
        } else {
            0.0
        };
        let molecule_name =
            serde_json::to_string(&event.molecule_name).unwrap_or("\"\"".to_string());
        let json = format!(
            r#"{{"event":"molecule_placed","molecule_index":{},"total_molecules":{},"molecule_name":{},"successful":{},"progress_pct":{:.1}}}"#,
            event.molecule_index,
            event.total_molecules,
            molecule_name,
            event.successful,
            progress_pct
        );
        self.emit_json(&json);
    }

    pub fn emit_gencan_iteration(&self, event: &GencanIterationEvent) {
        let progress_pct = if event.max_iterations > 0 {
            event.iteration as f64 / event.max_iterations as f64 * 100.0
        } else {
            0.0
        };
        let eta_ms = if event.iteration > 0 && event.elapsed_ms > 0 {
            (event.elapsed_ms * event.max_iterations as u64 / event.iteration as u64)
                .saturating_sub(event.elapsed_ms)
        } else {
            0
        };
        let json = format!(
            r#"{{"event":"gencan_iteration","iteration":{},"max_iterations":{},"obj_value":{:.6e},"obj_overlap":{:.6e},"obj_constraint":{:.6e},"pg_sup":{:.6e},"pg_norm":{:.6e},"elapsed_ms":{},"progress_pct":{:.1},"eta_ms":{}}}"#,
            event.iteration,
            event.max_iterations,
            event.obj_value,
            event.obj_overlap,
            event.obj_constraint,
            event.pg_sup,
            event.pg_norm,
            event.elapsed_ms,
            progress_pct,
            eta_ms
        );
        self.emit_json(&json);
    }

    pub fn emit_phase_complete(&self, event: &PhaseCompleteEvent) {
        let iters = event
            .iterations
            .map(|n| n.to_string())
            .unwrap_or("null".to_string());
        let obj = event
            .final_obj_value
            .map(|v| format!("{:.6e}", v))
            .unwrap_or("null".to_string());
        let json = format!(
            r#"{{"event":"phase_complete","phase":"{}","elapsed_ms":{},"iterations":{},"final_obj_value":{}}}"#,
            event.phase.as_str(),
            event.elapsed_ms,
            iters,
            obj
        );
        self.emit_json(&json);
    }

    pub fn emit_pack_complete(&self, event: &PackCompleteEvent) {
        let output = serde_json::to_string(&event.output_path).unwrap_or("null".to_string());
        let json = format!(
            r#"{{"event":"pack_complete","total_atoms":{},"total_molecules":{},"final_box_size":[{},{},{}],"output_path":{},"elapsed_ms":{},"profile_ms":{{"templates":{},"place_core":{},"movebad":{},"gencan":{},"relax":{}}}}}"#,
            event.total_atoms,
            event.total_molecules,
            event.final_box_size[0],
            event.final_box_size[1],
            event.final_box_size[2],
            output,
            event.elapsed_ms,
            event.profile.templates_ms,
            event.profile.place_core_ms,
            event.profile.movebad_ms,
            event.profile.gencan_ms,
            event.profile.relax_ms,
        );
        self.emit_json(&json);
    }

    /// Emit an error event.
    pub fn emit_error(&self, code: &str, message: &str, context: Option<&str>) {
        let code = serde_json::to_string(code).unwrap_or("\"unknown\"".to_string());
        let message = serde_json::to_string(message).unwrap_or("\"Unknown error\"".to_string());
        let ctx = context
            .map(|s| serde_json::to_string(s).unwrap_or_default())
            .unwrap_or("null".to_string());
        let json = format!(
            r#"{{"event":"error","code":{},"message":{},"context":{}}}"#,
            code, message, ctx
        );
        self.emit_json(&json);
    }
}

pub(crate) fn duration_ms(d: Duration) -> u64 {
    d.as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_started_format() {
        let emitter = StreamEmitter::enabled();
        let event = PackStartedEvent {
            total_molecules: 100,
            box_size: [50.0, 50.0, 50.0],
            box_origin: [0.0, 0.0, 0.0],
            output_path: Some("out.pdb".to_string()),
        };
        emitter.emit_pack_started(&event);
    }

    #[test]
    fn test_gencan_iteration_format() {
        let emitter = StreamEmitter::enabled();
        let event = GencanIterationEvent {
            iteration: 42,
            max_iterations: 1000,
            obj_value: 1.234e-3,
            obj_overlap: 1.0e-3,
            obj_constraint: 2.34e-4,
            pg_sup: 0.05,
            pg_norm: 0.123,
            elapsed_ms: 5000,
        };
        emitter.emit_gencan_iteration(&event);
    }

    #[test]
    fn test_disabled_emitter() {
        let emitter = StreamEmitter::disabled();
        assert!(!emitter.is_enabled());
        // Should not panic
        emitter.emit_pack_started(&PackStartedEvent {
            total_molecules: 100,
            box_size: [50.0, 50.0, 50.0],
            box_origin: [0.0, 0.0, 0.0],
            output_path: None,
        });
    }
}
