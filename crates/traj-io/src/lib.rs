pub mod dcd;
pub mod gro;
pub mod pdb;
pub mod pdb_traj;
pub mod trr;
pub mod xtc;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};
use traj_core::system::System;

pub trait TopologyReader {
    fn read_system(&mut self) -> TrajResult<System>;
}

pub trait TrajReader {
    fn n_atoms(&self) -> usize;
    fn n_frames_hint(&self) -> Option<usize>;
    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize>;
    fn skip_frames(&mut self, n_frames: usize) -> TrajResult<usize> {
        let chunk_frames = n_frames.min(256).max(1);
        let mut builder = FrameChunkBuilder::new(self.n_atoms(), chunk_frames);
        builder.set_requirements(false, false);
        let mut skipped = 0usize;
        while skipped < n_frames {
            let target = (n_frames - skipped).min(chunk_frames);
            let read = self.read_chunk(target, &mut builder)?;
            if read == 0 {
                break;
            }
            skipped += read;
        }
        Ok(skipped)
    }

    fn read_chunk_selected(
        &mut self,
        max_frames: usize,
        selection: &[u32],
        out: &mut FrameChunkBuilder,
    ) -> TrajResult<usize> {
        let n_atoms = self.n_atoms();
        for &idx in selection {
            if (idx as usize) >= n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "selection index {idx} out of bounds for trajectory with {n_atoms} atoms"
                )));
            }
        }

        let max_frames = max_frames.max(1);
        let mut full = FrameChunkBuilder::new(n_atoms, max_frames);
        full.set_requirements(out.needs_box(), out.needs_time());
        full.set_optional_requirements(
            out.needs_velocities(),
            out.needs_forces(),
            out.needs_lambda(),
        );
        let read = self.read_chunk(max_frames, &mut full)?;
        if read == 0 {
            out.reset(selection.len(), max_frames);
            return Ok(0);
        }
        let chunk = full.finish_take()?;

        out.reset(selection.len(), max_frames);
        for frame in 0..chunk.n_frames {
            let time_ps = chunk
                .time_ps
                .as_ref()
                .and_then(|time| time.get(frame).copied());
            let lambda_value = chunk
                .lambda_values
                .as_ref()
                .and_then(|values| values.get(frame).copied());
            let box_ = chunk.box_.get(frame).copied().unwrap_or(Box3::None);
            let dst = out.start_frame(box_, time_ps);
            let src_base = frame * chunk.n_atoms;
            for (dst_atom, &src_idx) in dst.iter_mut().zip(selection.iter()) {
                *dst_atom = chunk.coords[src_base + src_idx as usize];
            }
            let velocities = chunk.velocities.as_ref().map(|data| {
                selection
                    .iter()
                    .map(|&src_idx| data[src_base + src_idx as usize])
                    .collect::<Vec<_>>()
            });
            let forces = chunk.forces.as_ref().map(|data| {
                selection
                    .iter()
                    .map(|&src_idx| data[src_base + src_idx as usize])
                    .collect::<Vec<_>>()
            });
            let velocities_ref = velocities.as_deref();
            let forces_ref = forces.as_deref();
            out.set_frame_extras(velocities_ref, forces_ref, lambda_value)?;
        }
        Ok(chunk.n_frames)
    }
}
