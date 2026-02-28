use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

use super::ConductivityPlan;

impl Plan for ConductivityPlan {
    fn name(&self) -> &'static str {
        "conductivity"
    }

    fn set_frames_hint(&mut self, n_frames: Option<usize>) {
        self.frames_hint = n_frames;
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        if matches!(
            self.group_by,
            crate::plans::analysis::grouping::GroupBy::Atom
        ) {
            Some(self.selection.indices.as_ref())
        } else {
            None
        }
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        self.io_selection.as_deref()
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.init_state(system, device)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.process_chunk_state(chunk, system, device)
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        if let Some(expected) = self.io_selection.as_deref() {
            if expected != source_selection {
                return Err(TrajError::Mismatch(
                    "conductivity selected chunk does not match expected IO selection".into(),
                ));
            }
        }
        self.process_chunk_state(chunk, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.finalize_state()
    }
}
