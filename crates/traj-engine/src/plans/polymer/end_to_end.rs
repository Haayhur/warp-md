use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};
use crate::plans::polymer::common::PolymerChains;

#[cfg(feature = "cuda")]
use traj_gpu::{convert_coords, GpuPolymer};

pub struct EndToEndPlan {
    selection: Selection,
    chains: Option<PolymerChains>,
    endpoint_pairs: Vec<(usize, usize)>,
    endpoint_atoms: Vec<u32>,
    use_selected_read: bool,
    results: Vec<f32>,
    n_chains: usize,
    #[cfg(feature = "cuda")]
    gpu: Option<GpuPolymer>,
}

impl EndToEndPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            chains: None,
            endpoint_pairs: Vec::new(),
            endpoint_atoms: Vec::new(),
            use_selected_read: true,
            results: Vec::new(),
            n_chains: 0,
            #[cfg(feature = "cuda")]
            gpu: None,
        }
    }

    pub fn n_chains(&self) -> usize {
        self.n_chains
    }
}

impl Plan for EndToEndPlan {
    fn name(&self) -> &'static str {
        "polymer_end_to_end"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(false, false)
    }

    fn preferred_n_atoms_hint(&self, system: &System) -> Option<usize> {
        if !self.use_selected_read {
            return None;
        }
        let mut chain_ids = std::collections::BTreeSet::new();
        for &idx in self.selection.indices.iter() {
            let atom_idx = idx as usize;
            if atom_idx < system.atoms.chain_id.len() {
                chain_ids.insert(system.atoms.chain_id[atom_idx]);
            }
        }
        if chain_ids.is_empty() {
            Some(self.selection.indices.len())
        } else {
            Some(chain_ids.len().saturating_mul(2))
        }
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_read {
            if self.endpoint_atoms.is_empty() {
                Some(&self.selection.indices)
            } else {
                Some(&self.endpoint_atoms)
            }
        } else {
            None
        }
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        let _ = device;
        self.results.clear();
        self.endpoint_pairs.clear();
        self.endpoint_atoms.clear();
        let chains = PolymerChains::from_selection(system, &self.selection)?;
        self.n_chains = chains.n_chains();

        self.use_selected_read = true;
        #[cfg(feature = "cuda")]
        if matches!(device, Device::Cuda(_)) {
            // CUDA path expects full-coordinate chunks with original atom indexing.
            self.use_selected_read = false;
        }

        if self.use_selected_read {
            self.endpoint_pairs.reserve(self.n_chains);
            self.endpoint_atoms.reserve(self.n_chains.saturating_mul(2));
            for chain in &chains.chains {
                let first = *chain.first().unwrap() as u32;
                let last = *chain.last().unwrap() as u32;
                let first_local = self.endpoint_atoms.len();
                self.endpoint_atoms.push(first);
                let last_local = self.endpoint_atoms.len();
                self.endpoint_atoms.push(last);
                self.endpoint_pairs.push((first_local, last_local));
            }
        } else {
            self.endpoint_pairs.reserve(self.n_chains);
            for chain in &chains.chains {
                let first = *chain.first().unwrap();
                let last = *chain.last().unwrap();
                self.endpoint_pairs.push((first, last));
            }
        }

        self.chains = Some(chains);
        #[cfg(feature = "cuda")]
        {
            self.gpu = None;
            if let Device::Cuda(ctx) = device {
                let chains = self.chains.as_ref().unwrap();
                let gpu = ctx.polymer_data(&chains.offsets, &chains.indices, None, None)?;
                self.gpu = Some(gpu);
            }
        }
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        let _ = device;
        let _ = self.chains.as_ref().unwrap();
        self.results
            .reserve(chunk.n_frames.saturating_mul(self.n_chains));
        #[cfg(feature = "cuda")]
        if let (Device::Cuda(ctx), Some(gpu)) = (device, &self.gpu) {
            let coords = convert_coords(&chunk.coords);
            let out = ctx.polymer_end_to_end(&coords, chunk.n_atoms, chunk.n_frames, gpu)?;
            self.results.extend(out);
            return Ok(());
        }
        if self.n_chains == 1 {
            let (first, last) = self.endpoint_pairs[0];
            for frame_coords in chunk
                .coords
                .chunks_exact(chunk.n_atoms)
                .take(chunk.n_frames)
            {
                let p = frame_coords[first];
                let q = frame_coords[last];
                let dx = q[0] - p[0];
                let dy = q[1] - p[1];
                let dz = q[2] - p[2];
                self.results.push((dx * dx + dy * dy + dz * dz).sqrt());
            }
            return Ok(());
        }
        for frame_coords in chunk
            .coords
            .chunks_exact(chunk.n_atoms)
            .take(chunk.n_frames)
        {
            for &(first, last) in &self.endpoint_pairs {
                let p = frame_coords[first];
                let q = frame_coords[last];
                let dx = q[0] - p[0];
                let dy = q[1] - p[1];
                let dz = q[2] - p[2];
                self.results.push((dx * dx + dy * dy + dz * dz).sqrt());
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let data = std::mem::take(&mut self.results);
        let rows = if self.n_chains > 0 {
            data.len() / self.n_chains
        } else {
            0
        };
        Ok(PlanOutput::Matrix {
            data,
            rows,
            cols: self.n_chains,
        })
    }
}
