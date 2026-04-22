use super::*;

impl Plan for GistDirectPlan {
    fn name(&self) -> &'static str {
        "gist_direct"
    }

    fn init(&mut self, _system: &System, device: &Device) -> TrajResult<()> {
        self.reset_runtime();
        self.init_gpu_state(device)
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.validate_process_inputs(chunk.n_atoms)?;

        for local_frame in 0..chunk.n_frames {
            let abs_frame = self.global_frame + local_frame;
            if !self.keep_frame(abs_frame) {
                continue;
            }
            self.process_frame(chunk, device, local_frame)?;
        }

        self.global_frame += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.finalize_histograms()?;
        self.validate_final_state()?;
        finalize_counts_orientation(
            &self.counts,
            &self.orient_counts,
            self.dims,
            self.orientation_bins,
        )
    }
}
