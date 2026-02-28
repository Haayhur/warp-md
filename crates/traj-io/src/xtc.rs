use std::io::{Seek, SeekFrom};
use std::path::PathBuf;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};
use xdrfile::{Frame, Trajectory, XTCTrajectory};

use crate::TrajReader;

pub struct XtcReader {
    traj: XTCTrajectory,
    n_atoms: usize,
    frame: Frame,
    _path: PathBuf,
    selection_cache_u32: Vec<u32>,
    selection_cache_usize: Vec<usize>,
}

impl XtcReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let mut traj = XTCTrajectory::open_read(&path).map_err(map_xtc_err)?;
        let n_atoms = traj.get_num_atoms().map_err(map_xtc_err)?;
        let frame = Frame::with_len(n_atoms);
        Ok(Self {
            traj,
            n_atoms,
            frame,
            _path: path,
            selection_cache_u32: Vec::new(),
            selection_cache_usize: Vec::new(),
        })
    }

    fn ensure_selection_cache(&mut self, selection: &[u32]) -> TrajResult<()> {
        if self.selection_cache_u32.as_slice() == selection {
            return Ok(());
        }
        self.selection_cache_usize = validate_and_materialize_selection(selection, self.n_atoms)?;
        self.selection_cache_u32.clear();
        self.selection_cache_u32.extend_from_slice(selection);
        Ok(())
    }

    pub fn reset(&mut self) -> TrajResult<()> {
        self.traj.seek(SeekFrom::Start(0))?;
        Ok(())
    }

    pub fn read_chunk_into_coords3(
        &mut self,
        max_frames: usize,
        coords_out: &mut [f32],
    ) -> TrajResult<usize> {
        const NM_TO_ANGSTROM: f32 = 10.0;
        let frames_cap = coords_out.len() / (self.n_atoms * 3);
        if frames_cap == 0 {
            return Ok(0);
        }
        let target = max_frames.min(frames_cap).max(1);
        let mut frames = 0usize;
        while frames < target {
            match self.traj.read(&mut self.frame) {
                Ok(()) => {
                    let base = frames * self.n_atoms * 3;
                    for (i, src) in self.frame.coords.iter().enumerate() {
                        let out = base + i * 3;
                        coords_out[out] = src[0] * NM_TO_ANGSTROM;
                        coords_out[out + 1] = src[1] * NM_TO_ANGSTROM;
                        coords_out[out + 2] = src[2] * NM_TO_ANGSTROM;
                    }
                    frames += 1;
                }
                Err(err) => {
                    if err.is_eof() {
                        break;
                    }
                    return Err(map_xtc_err(err));
                }
            }
        }
        Ok(frames)
    }

    pub fn read_chunk_into_coords3_selected(
        &mut self,
        max_frames: usize,
        selection: &[u32],
        coords_out: &mut [f32],
    ) -> TrajResult<usize> {
        const NM_TO_ANGSTROM: f32 = 10.0;
        self.ensure_selection_cache(selection)?;
        let n_sel = self.selection_cache_usize.len();
        if n_sel == 0 {
            return Ok(0);
        }
        let frames_cap = coords_out.len() / (n_sel * 3);
        if frames_cap == 0 {
            return Ok(0);
        }
        let target = max_frames.min(frames_cap).max(1);
        let mut frames = 0usize;
        while frames < target {
            match self.traj.read(&mut self.frame) {
                Ok(()) => {
                    let base = frames * n_sel * 3;
                    fill_selected_coords3(
                        &mut coords_out[base..base + n_sel * 3],
                        &self.frame.coords,
                        self.selection_cache_usize.as_slice(),
                        NM_TO_ANGSTROM,
                    );
                    frames += 1;
                }
                Err(err) => {
                    if err.is_eof() {
                        break;
                    }
                    return Err(map_xtc_err(err));
                }
            }
        }
        Ok(frames)
    }
}

impl TrajReader for XtcReader {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        None
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        const NM_TO_ANGSTROM: f32 = 10.0;
        let needs_box = out.needs_box();
        let needs_time = out.needs_time();
        out.reset(self.n_atoms, max_frames);
        let mut frames = 0;
        while frames < max_frames {
            match self.traj.read(&mut self.frame) {
                Ok(()) => {
                    let box_ = if needs_box {
                        convert_box(self.frame.box_vector)
                    } else {
                        Box3::None
                    };
                    let time_ps = if needs_time {
                        Some(self.frame.time)
                    } else {
                        None
                    };
                    let coords = out.start_frame(box_, time_ps);
                    for (dst, src) in coords.iter_mut().zip(self.frame.coords.iter()) {
                        dst[0] = src[0] * NM_TO_ANGSTROM;
                        dst[1] = src[1] * NM_TO_ANGSTROM;
                        dst[2] = src[2] * NM_TO_ANGSTROM;
                        dst[3] = 1.0;
                    }
                    frames += 1;
                }
                Err(err) => {
                    if err.is_eof() {
                        break;
                    }
                    return Err(map_xtc_err(err));
                }
            }
        }
        Ok(frames)
    }

    fn read_chunk_selected(
        &mut self,
        max_frames: usize,
        selection: &[u32],
        out: &mut FrameChunkBuilder,
    ) -> TrajResult<usize> {
        const NM_TO_ANGSTROM: f32 = 10.0;
        self.ensure_selection_cache(selection)?;
        let needs_box = out.needs_box();
        let needs_time = out.needs_time();
        out.reset(self.selection_cache_usize.len(), max_frames);
        let mut frames = 0;
        while frames < max_frames {
            match self.traj.read(&mut self.frame) {
                Ok(()) => {
                    let box_ = if needs_box {
                        convert_box(self.frame.box_vector)
                    } else {
                        Box3::None
                    };
                    let time_ps = if needs_time {
                        Some(self.frame.time)
                    } else {
                        None
                    };
                    let coords = out.start_frame(box_, time_ps);
                    fill_selected_coords(
                        coords,
                        &self.frame.coords,
                        self.selection_cache_usize.as_slice(),
                        NM_TO_ANGSTROM,
                    );
                    frames += 1;
                }
                Err(err) => {
                    if err.is_eof() {
                        break;
                    }
                    return Err(map_xtc_err(err));
                }
            }
        }
        Ok(frames)
    }
}

#[inline(always)]
fn fill_selected_coords(dst: &mut [[f32; 4]], src: &[[f32; 3]], selection: &[usize], scale: f32) {
    match selection {
        [a] => {
            copy_scaled(&mut dst[0], src[*a], scale);
        }
        [a, b] => {
            copy_scaled(&mut dst[0], src[*a], scale);
            copy_scaled(&mut dst[1], src[*b], scale);
        }
        [a, b, c] => {
            copy_scaled(&mut dst[0], src[*a], scale);
            copy_scaled(&mut dst[1], src[*b], scale);
            copy_scaled(&mut dst[2], src[*c], scale);
        }
        [a, b, c, d] => {
            copy_scaled(&mut dst[0], src[*a], scale);
            copy_scaled(&mut dst[1], src[*b], scale);
            copy_scaled(&mut dst[2], src[*c], scale);
            copy_scaled(&mut dst[3], src[*d], scale);
        }
        _ => {
            // Bounds are validated once per chunk before entering this hot loop.
            let src_ptr = src.as_ptr();
            for i in 0..selection.len() {
                let src_idx = unsafe { *selection.get_unchecked(i) };
                let src_atom = unsafe { *src_ptr.add(src_idx) };
                let dst_atom = unsafe { dst.get_unchecked_mut(i) };
                copy_scaled(dst_atom, src_atom, scale);
            }
        }
    }
}

#[inline(always)]
fn copy_scaled(dst: &mut [f32; 4], src: [f32; 3], scale: f32) {
    dst[0] = src[0] * scale;
    dst[1] = src[1] * scale;
    dst[2] = src[2] * scale;
    dst[3] = 1.0;
}

fn convert_box(box_vec: [[f32; 3]; 3]) -> Box3 {
    let m = [
        box_vec[0][0] * 10.0,
        box_vec[0][1] * 10.0,
        box_vec[0][2] * 10.0,
        box_vec[1][0] * 10.0,
        box_vec[1][1] * 10.0,
        box_vec[1][2] * 10.0,
        box_vec[2][0] * 10.0,
        box_vec[2][1] * 10.0,
        box_vec[2][2] * 10.0,
    ];
    let tol = 1e-5;
    let is_orth = m[1].abs() < tol
        && m[2].abs() < tol
        && m[3].abs() < tol
        && m[5].abs() < tol
        && m[6].abs() < tol
        && m[7].abs() < tol;
    if is_orth {
        Box3::Orthorhombic {
            lx: m[0],
            ly: m[4],
            lz: m[8],
        }
    } else {
        Box3::Triclinic { m }
    }
}

fn map_xtc_err(err: xdrfile::Error) -> TrajError {
    TrajError::Parse(format!("xtc error: {err}"))
}

fn validate_and_materialize_selection(selection: &[u32], n_atoms: usize) -> TrajResult<Vec<usize>> {
    let mut out = Vec::with_capacity(selection.len());
    for &idx in selection {
        let src = idx as usize;
        if src >= n_atoms {
            return Err(TrajError::Mismatch(format!(
                "selection index {idx} out of bounds for trajectory with {n_atoms} atoms"
            )));
        }
        out.push(src);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use xdrfile::{FileMode, Trajectory};

    #[test]
    fn read_xtc_simple() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.xtc");
        let mut traj = XTCTrajectory::open(path.clone(), FileMode::Write).unwrap();
        let mut frame = Frame::with_len(2);
        frame.step = 0;
        frame.time = 2.0;
        frame.box_vector = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];
        frame.coords[0] = [0.1, 0.2, 0.3];
        frame.coords[1] = [0.4, 0.5, 0.6];
        traj.write(&frame).unwrap();
        traj.flush().unwrap();

        let mut reader = XtcReader::open(&path).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 2);
        let count = reader.read_chunk(2, &mut builder).unwrap();
        assert_eq!(count, 1);
        let chunk = builder.finish().unwrap();
        assert_eq!(chunk.n_frames, 1);
        assert_eq!(
            chunk.box_[0],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 20.0,
                lz: 30.0
            }
        );
        assert!((chunk.coords[0][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn read_xtc_selected_endpoints() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_selected.xtc");
        let mut traj = XTCTrajectory::open(path.clone(), FileMode::Write).unwrap();
        let mut frame = Frame::with_len(3);
        frame.step = 0;
        frame.time = 1.0;
        frame.box_vector = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
        frame.coords[0] = [0.1, 0.2, 0.3];
        frame.coords[1] = [0.4, 0.5, 0.6];
        frame.coords[2] = [0.7, 0.8, 0.9];
        traj.write(&frame).unwrap();
        traj.flush().unwrap();

        let mut reader = XtcReader::open(&path).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 1);
        let count = reader
            .read_chunk_selected(1, &[2, 0], &mut builder)
            .unwrap();
        assert_eq!(count, 1);
        let chunk = builder.finish().unwrap();
        assert_eq!(chunk.n_frames, 1);
        assert_eq!(chunk.n_atoms, 2);
        assert!((chunk.coords[0][0] - 7.0).abs() < 1e-6);
        assert!((chunk.coords[0][1] - 8.0).abs() < 1e-6);
        assert!((chunk.coords[0][2] - 9.0).abs() < 1e-6);
        assert!((chunk.coords[1][0] - 1.0).abs() < 1e-6);
        assert!((chunk.coords[1][1] - 2.0).abs() < 1e-6);
        assert!((chunk.coords[1][2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn reset_rewinds_xtc_reader() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_reset.xtc");
        let mut traj = XTCTrajectory::open(path.clone(), FileMode::Write).unwrap();
        let mut frame = Frame::with_len(1);
        frame.step = 0;
        frame.time = 0.0;
        frame.box_vector = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        frame.coords[0] = [0.1, 0.0, 0.0];
        traj.write(&frame).unwrap();
        frame.step = 1;
        frame.time = 1.0;
        frame.coords[0] = [0.2, 0.0, 0.0];
        traj.write(&frame).unwrap();
        traj.flush().unwrap();

        let mut reader = XtcReader::open(&path).unwrap();
        let mut buf = vec![0.0f32; 3];
        let read = reader.read_chunk_into_coords3(1, &mut buf).unwrap();
        assert_eq!(read, 1);
        assert!((buf[0] - 1.0).abs() < 1e-6);

        reader.reset().unwrap();
        let read = reader.read_chunk_into_coords3(1, &mut buf).unwrap();
        assert_eq!(read, 1);
        assert!((buf[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn read_chunk_into_coords3_returns_xyz_angstrom() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_into.xtc");
        let mut traj = XTCTrajectory::open(path.clone(), FileMode::Write).unwrap();
        let mut frame = Frame::with_len(2);
        frame.step = 0;
        frame.time = 0.0;
        frame.box_vector = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        frame.coords[0] = [0.1, 0.2, 0.3];
        frame.coords[1] = [0.4, 0.5, 0.6];
        traj.write(&frame).unwrap();
        traj.flush().unwrap();

        let mut reader = XtcReader::open(&path).unwrap();
        let mut out = vec![0.0f32; 2 * 3];
        let read = reader.read_chunk_into_coords3(1, &mut out).unwrap();
        assert_eq!(read, 1);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
        assert!((out[3] - 4.0).abs() < 1e-6);
        assert!((out[4] - 5.0).abs() < 1e-6);
        assert!((out[5] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn read_chunk_into_coords3_selected_returns_requested_atoms() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_into_selected.xtc");
        let mut traj = XTCTrajectory::open(path.clone(), FileMode::Write).unwrap();
        let mut frame = Frame::with_len(3);
        frame.step = 0;
        frame.time = 0.0;
        frame.box_vector = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        frame.coords[0] = [0.1, 0.2, 0.3];
        frame.coords[1] = [0.4, 0.5, 0.6];
        frame.coords[2] = [0.7, 0.8, 0.9];
        traj.write(&frame).unwrap();
        traj.flush().unwrap();

        let mut reader = XtcReader::open(&path).unwrap();
        let mut out = vec![0.0f32; 2 * 3];
        let read = reader
            .read_chunk_into_coords3_selected(1, &[2, 0], &mut out)
            .unwrap();
        assert_eq!(read, 1);
        assert!((out[0] - 7.0).abs() < 1e-6);
        assert!((out[1] - 8.0).abs() < 1e-6);
        assert!((out[2] - 9.0).abs() < 1e-6);
        assert!((out[3] - 1.0).abs() < 1e-6);
        assert!((out[4] - 2.0).abs() < 1e-6);
        assert!((out[5] - 3.0).abs() < 1e-6);
    }
}

#[inline(always)]
fn fill_selected_coords3(dst: &mut [f32], src: &[[f32; 3]], selection: &[usize], scale: f32) {
    // Bounds are validated once per chunk before entering this hot loop.
    let src_ptr = src.as_ptr();
    for i in 0..selection.len() {
        let src_idx = unsafe { *selection.get_unchecked(i) };
        let src_atom = unsafe { *src_ptr.add(src_idx) };
        let out = i * 3;
        dst[out] = src_atom[0] * scale;
        dst[out + 1] = src_atom[1] * scale;
        dst[out + 2] = src_atom[2] * scale;
    }
}
