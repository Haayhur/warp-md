use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::analysis::secondary_structure::{
    build_backbone_io_selection, build_backbone_model, compute_backbone_frame_into,
    remap_backbone_model, BackboneFrame, BackboneModel,
};

pub struct KabschSanderPlan {
    selection: Selection,
    model: BackboneModel,
    selected_model: BackboneModel,
    io_selection: Vec<u32>,
    frame: BackboneFrame,
    energies: Vec<f32>,
    frames: usize,
}

impl KabschSanderPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            model: BackboneModel::default(),
            selected_model: BackboneModel::default(),
            io_selection: Vec::new(),
            frame: BackboneFrame::default(),
            energies: Vec::new(),
            frames: 0,
        }
    }

    pub fn labels(&self) -> &[String] {
        &self.model.labels
    }
}

impl Plan for KabschSanderPlan {
    fn name(&self) -> &'static str {
        "kabsch_sander"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.energies.clear();
        self.frames = 0;
        self.model = build_backbone_model(system, &self.selection);
        self.io_selection = build_backbone_io_selection(&self.model);
        self.selected_model = remap_backbone_model(&self.model, &self.io_selection);
        Ok(())
    }

    fn preferred_selection_hint(&self, _system: &System) -> Option<&[u32]> {
        if self.model.residues.is_empty() {
            None
        } else {
            Some(self.io_selection.as_slice())
        }
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.model.residues.is_empty() {
            None
        } else {
            Some(self.io_selection.as_slice())
        }
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        process_kabsch_sander_chunk(
            &self.model,
            &mut self.frame,
            chunk,
            &mut self.energies,
            &mut self.frames,
        );
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        if self.model.residues.is_empty() {
            return self.process_chunk(chunk, system, device);
        }
        if source_selection != self.io_selection.as_slice() {
            return Err(TrajError::Mismatch(
                "kabsch_sander selected chunk does not match expected backbone selection".into(),
            ));
        }
        if chunk.n_atoms != self.io_selection.len() {
            return Err(TrajError::Mismatch(
                "kabsch_sander selected chunk atom count does not match backbone selection".into(),
            ));
        }
        process_kabsch_sander_chunk(
            &self.selected_model,
            &mut self.frame,
            chunk,
            &mut self.energies,
            &mut self.frames,
        );
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_res = self.model.residues.len();
        if self.frames == 0 || n_res == 0 {
            return Ok(PlanOutput::Matrix {
                data: Vec::new(),
                rows: 0,
                cols: n_res * n_res,
            });
        }
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.energies),
            rows: self.frames,
            cols: n_res * n_res,
        })
    }
}

fn process_kabsch_sander_chunk(
    model: &BackboneModel,
    frame_scratch: &mut BackboneFrame,
    chunk: &FrameChunk,
    energies: &mut Vec<f32>,
    frames: &mut usize,
) {
    let n_res = model.residues.len();
    if n_res == 0 {
        return;
    }
    for frame in 0..chunk.n_frames {
        compute_backbone_frame_into(model, chunk, frame, frame_scratch);
        energies.extend(frame_scratch.hbond_energy.iter().map(|&value| value as f32));
        *frames += 1;
    }
}
