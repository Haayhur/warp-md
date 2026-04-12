use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::analysis::secondary_structure::{
    build_backbone_model, compute_backbone_frame, BackboneModel,
};

pub struct DsspPlan {
    selection: Selection,
    model: BackboneModel,
    codes: Vec<u8>,
    frames: usize,
}

impl DsspPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            model: BackboneModel::default(),
            codes: Vec::new(),
            frames: 0,
        }
    }

    pub fn labels(&self) -> &[String] {
        &self.model.labels
    }
}

impl Plan for DsspPlan {
    fn name(&self) -> &'static str {
        "dssp"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.codes.clear();
        self.frames = 0;
        self.model = build_backbone_model(system, &self.selection);
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_res = self.model.residues.len();
        if n_res == 0 {
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let frame_data = compute_backbone_frame(&self.model, chunk, frame);
            self.codes.extend(frame_data.states.into_iter());
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_res = self.model.residues.len();
        if self.frames == 0 || n_res == 0 {
            return Ok(PlanOutput::Matrix {
                data: Vec::new(),
                rows: 0,
                cols: n_res,
            });
        }
        let data: Vec<f32> = self.codes.iter().map(|&value| value as f32).collect();
        Ok(PlanOutput::Matrix {
            data,
            rows: self.frames,
            cols: n_res,
        })
    }
}
