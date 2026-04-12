//! NDJSON streaming events for warp-pack agent integration.

use serde_json::json;
pub(crate) use warp_structure::ndjson::duration_ms;
use warp_structure::ndjson::NdjsonEmitter;

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

    pub fn emit_pack_started(&self, event: &PackStartedEvent) {
        self.inner.emit(&json!({
            "event": "pack_started",
            "total_molecules": event.total_molecules,
            "box_size": event.box_size,
            "box_origin": event.box_origin,
            "output_path": event.output_path,
        }));
    }

    pub fn emit_phase_started(&self, event: &PhaseStartedEvent) {
        self.inner.emit(&json!({
            "event": "phase_started",
            "phase": event.phase.as_str(),
            "total_molecules": event.total_molecules,
            "max_iterations": event.max_iterations,
        }));
    }

    pub fn emit_molecule_placed(&self, event: &MoleculePlacedEvent) {
        let progress_pct = if event.total_molecules > 0 {
            event.molecule_index as f64 / event.total_molecules as f64 * 100.0
        } else {
            0.0
        };
        self.inner.emit(&json!({
            "event": "molecule_placed",
            "molecule_index": event.molecule_index,
            "total_molecules": event.total_molecules,
            "molecule_name": event.molecule_name,
            "successful": event.successful,
            "progress_pct": progress_pct,
        }));
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
        self.inner.emit(&json!({
            "event": "gencan_iteration",
            "iteration": event.iteration,
            "max_iterations": event.max_iterations,
            "obj_value": event.obj_value,
            "obj_overlap": event.obj_overlap,
            "obj_constraint": event.obj_constraint,
            "pg_sup": event.pg_sup,
            "pg_norm": event.pg_norm,
            "elapsed_ms": event.elapsed_ms,
            "progress_pct": progress_pct,
            "eta_ms": eta_ms,
        }));
    }

    pub fn emit_phase_complete(&self, event: &PhaseCompleteEvent) {
        self.inner.emit(&json!({
            "event": "phase_complete",
            "phase": event.phase.as_str(),
            "elapsed_ms": event.elapsed_ms,
            "iterations": event.iterations,
            "final_obj_value": event.final_obj_value,
        }));
    }

    pub fn emit_pack_complete(&self, event: &PackCompleteEvent) {
        self.inner.emit(&json!({
            "event": "pack_complete",
            "total_atoms": event.total_atoms,
            "total_molecules": event.total_molecules,
            "final_box_size": event.final_box_size,
            "output_path": event.output_path,
            "elapsed_ms": event.elapsed_ms,
            "profile_ms": {
                "templates": event.profile.templates_ms,
                "place_core": event.profile.place_core_ms,
                "movebad": event.profile.movebad_ms,
                "gencan": event.profile.gencan_ms,
                "relax": event.profile.relax_ms,
            },
        }));
    }

    pub fn emit_error(&self, code: &str, message: &str, context: Option<&str>) {
        self.inner.emit(&json!({
            "event": "error",
            "code": code,
            "message": message,
            "context": context,
        }));
    }
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
        emitter.emit_pack_started(&PackStartedEvent {
            total_molecules: 100,
            box_size: [50.0, 50.0, 50.0],
            box_origin: [0.0, 0.0, 0.0],
            output_path: None,
        });
    }
}
