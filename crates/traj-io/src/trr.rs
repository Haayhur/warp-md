use std::ffi::CString;
use std::path::{Path, PathBuf};

use ::xdrfile::c_abi::xdr_seek;
use ::xdrfile::c_abi::xdrfile::{self as xdr_cabi, Matrix, Rvec, XDRFILE};
use ::xdrfile::c_abi::xdrfile_trr;
use ::xdrfile::{Error as XdrError, ErrorCode as XdrErrorCode, ErrorTask as XdrErrorTask};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::TrajReader;

#[derive(Clone, Debug)]
pub struct TrrFrameData {
    pub coords: Vec<[f32; 4]>,
    pub box_: Box3,
    pub time_ps: Option<f32>,
    pub velocities: Option<Vec<[f32; 3]>>,
    pub forces: Option<Vec<[f32; 3]>>,
    pub lambda_value: Option<f32>,
}

pub struct TrrReader {
    xdr: *mut XDRFILE,
    n_atoms: usize,
    n_frames: Option<usize>,
    coords_buf: Vec<[f32; 3]>,
    velocities_buf: Vec<[f32; 3]>,
    forces_buf: Vec<[f32; 3]>,
}

pub struct TrrWriter {
    xdr: *mut XDRFILE,
    n_atoms: usize,
}

impl TrrReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let c_path = path_to_cstring(&path)?;
        let mode = CString::new("r").expect("CString::new('r') should be infallible");
        let xdr = unsafe { xdr_cabi::xdrfile_open(c_path.as_ptr(), mode.as_ptr()) };
        if xdr.is_null() {
            return Err(TrajError::Io(std::io::Error::other(format!(
                "failed to open TRR for read: {}",
                path.display()
            ))));
        }

        let mut natoms = 0i32;
        let natom_code = unsafe { xdrfile_trr::read_trr_natoms(c_path.as_ptr(), &mut natoms) };
        if natom_code != xdr_cabi::exdrOK || natoms <= 0 {
            unsafe {
                xdr_cabi::xdrfile_close(xdr);
            }
            return Err(map_trr_code(natom_code, XdrErrorTask::Read));
        }

        let mut nframes = 0u64;
        let frame_code = unsafe { xdrfile_trr::read_trr_nframes(c_path.as_ptr(), &mut nframes) };
        let n_frames = if frame_code == xdr_cabi::exdrOK {
            usize::try_from(nframes).ok()
        } else {
            None
        };

        Ok(Self {
            xdr,
            n_atoms: natoms as usize,
            n_frames,
            coords_buf: vec![[0.0; 3]; natoms as usize],
            velocities_buf: vec![[0.0; 3]; natoms as usize],
            forces_buf: vec![[0.0; 3]; natoms as usize],
        })
    }

    pub fn reset(&mut self) -> TrajResult<()> {
        let code = unsafe { xdr_seek::xdr_seek(self.xdr, 0, 0) };
        if code == xdr_cabi::exdrOK {
            Ok(())
        } else {
            Err(map_trr_code(code, XdrErrorTask::Seek))
        }
    }

    pub fn collect_selected_frames(
        &mut self,
        source_indices: &[usize],
        include_box: bool,
        include_time: bool,
        include_velocities: bool,
        include_forces: bool,
        include_lambda: bool,
    ) -> TrajResult<Vec<TrrFrameData>> {
        if source_indices.is_empty() {
            return Ok(Vec::new());
        }

        let mut targets: Vec<(usize, usize)> = source_indices
            .iter()
            .copied()
            .enumerate()
            .map(|(pos, frame_idx)| (frame_idx, pos))
            .collect();
        targets.sort_unstable_by_key(|(frame_idx, _)| *frame_idx);

        let mut ordered: Vec<Option<TrrFrameData>> = vec![None; source_indices.len()];
        let mut target_cursor = 0usize;
        let target_len = targets.len();
        let mut global = 0usize;

        while target_cursor < target_len {
            let target_frame = targets[target_cursor].0;
            if target_frame > global {
                let skipped = self.skip_frames(target_frame - global)?;
                global += skipped;
                if global < target_frame {
                    break;
                }
            }

            let Some(frame) = self.read_next_frame(
                include_box,
                include_time,
                include_velocities,
                include_forces,
                include_lambda,
            )?
            else {
                break;
            };

            let mut end = target_cursor + 1;
            while end < target_len && targets[end].0 == target_frame {
                end += 1;
            }
            if end == target_cursor + 1 {
                ordered[targets[target_cursor].1] = Some(frame);
            } else {
                for &(.., out_pos) in &targets[target_cursor..end - 1] {
                    ordered[out_pos] = Some(frame.clone());
                }
                ordered[targets[end - 1].1] = Some(frame);
            }
            target_cursor = end;
            global += 1;
        }

        Ok(ordered.into_iter().flatten().collect())
    }

    pub fn collect_strided_frames(
        &mut self,
        begin: usize,
        end: usize,
        step: usize,
        include_box: bool,
        include_time: bool,
        include_velocities: bool,
        include_forces: bool,
        include_lambda: bool,
    ) -> TrajResult<(Vec<TrrFrameData>, Vec<usize>)> {
        if begin >= end {
            return Ok((Vec::new(), Vec::new()));
        }
        if step == 0 {
            return Err(TrajError::Parse("step must be >= 1".into()));
        }

        let expected = 1 + (end - begin - 1) / step;
        let mut frames = Vec::with_capacity(expected);
        let mut source_indices = Vec::with_capacity(expected);
        let mut global = 0usize;

        if begin > 0 {
            let skipped = self.skip_frames(begin)?;
            global += skipped;
            if global < begin {
                return Ok((frames, source_indices));
            }
        }

        while global < end {
            let Some(frame) = self.read_next_frame(
                include_box,
                include_time,
                include_velocities,
                include_forces,
                include_lambda,
            )?
            else {
                break;
            };
            frames.push(frame);
            source_indices.push(global);
            global += 1;
            if global >= end {
                break;
            }
            let gap = (step - 1).min(end - global);
            let skipped = self.skip_frames(gap)?;
            global += skipped;
            if skipped < gap {
                break;
            }
        }

        Ok((frames, source_indices))
    }

    fn read_next_frame(
        &mut self,
        include_box: bool,
        include_time: bool,
        include_velocities: bool,
        include_forces: bool,
        include_lambda: bool,
    ) -> TrajResult<Option<TrrFrameData>> {
        let mut step = 0i32;
        let mut time_ps = 0.0f32;
        let mut lambda = 0.0f32;
        let mut box_vec: Matrix = [[0.0; 3]; 3];
        let vel_ptr = if include_velocities {
            self.velocities_buf.as_mut_ptr() as *mut Rvec
        } else {
            std::ptr::null_mut()
        };
        let force_ptr = if include_forces {
            self.forces_buf.as_mut_ptr() as *mut Rvec
        } else {
            std::ptr::null_mut()
        };
        let code = unsafe {
            xdrfile_trr::read_trr(
                self.xdr,
                self.n_atoms as i32,
                &mut step,
                &mut time_ps,
                &mut lambda,
                &mut box_vec,
                self.coords_buf.as_mut_ptr() as *mut Rvec,
                vel_ptr,
                force_ptr,
            )
        };
        let err_code = XdrErrorCode::from(code);
        if err_code.is_eof() {
            return Ok(None);
        }
        if err_code != XdrErrorCode::ExdrOk {
            return Err(map_trr_code(code, XdrErrorTask::Read));
        }

        let coords = self
            .coords_buf
            .iter()
            .map(|value| [value[0] * 10.0, value[1] * 10.0, value[2] * 10.0, 1.0])
            .collect::<Vec<_>>();
        let velocities = if include_velocities {
            Some(
                self.velocities_buf
                    .iter()
                    .map(|value| [value[0] * 10.0, value[1] * 10.0, value[2] * 10.0])
                    .collect(),
            )
        } else {
            None
        };
        let forces = if include_forces {
            Some(self.forces_buf.clone())
        } else {
            None
        };

        Ok(Some(TrrFrameData {
            coords,
            box_: if include_box {
                convert_box(box_vec)
            } else {
                Box3::None
            },
            time_ps: if include_time { Some(time_ps) } else { None },
            velocities,
            forces,
            lambda_value: if include_lambda { Some(lambda) } else { None },
        }))
    }

    fn skip_one_frame(&mut self) -> TrajResult<bool> {
        let mut step = 0i32;
        let mut time_ps = 0.0f32;
        let mut lambda = 0.0f32;
        let mut box_vec: Matrix = [[0.0; 3]; 3];
        let code = unsafe {
            xdrfile_trr::read_trr(
                self.xdr,
                self.n_atoms as i32,
                &mut step,
                &mut time_ps,
                &mut lambda,
                &mut box_vec,
                self.coords_buf.as_mut_ptr() as *mut Rvec,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        let err_code = XdrErrorCode::from(code);
        if err_code.is_eof() {
            return Ok(false);
        }
        if err_code != XdrErrorCode::ExdrOk {
            return Err(map_trr_code(code, XdrErrorTask::Read));
        }
        Ok(true)
    }
}

impl Drop for TrrReader {
    fn drop(&mut self) {
        if !self.xdr.is_null() {
            unsafe {
                xdr_cabi::xdrfile_close(self.xdr);
            }
            self.xdr = std::ptr::null_mut();
        }
    }
}

impl TrajReader for TrrReader {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        self.n_frames
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        let needs_box = out.needs_box();
        let needs_time = out.needs_time();
        let needs_velocities = out.needs_velocities();
        let needs_forces = out.needs_forces();
        let needs_lambda = out.needs_lambda();
        out.reset(self.n_atoms, max_frames);
        let mut frames = 0usize;
        while frames < max_frames {
            let Some(frame) = self.read_next_frame(
                needs_box,
                needs_time,
                needs_velocities,
                needs_forces,
                needs_lambda,
            )?
            else {
                break;
            };
            let dst = out.start_frame(frame.box_, frame.time_ps);
            dst.copy_from_slice(&frame.coords);
            out.set_frame_extras(
                frame.velocities.as_deref(),
                frame.forces.as_deref(),
                frame.lambda_value,
            )?;
            frames += 1;
        }
        Ok(frames)
    }

    fn skip_frames(&mut self, n_frames: usize) -> TrajResult<usize> {
        let mut skipped = 0usize;
        while skipped < n_frames {
            if !self.skip_one_frame()? {
                break;
            }
            skipped += 1;
        }
        Ok(skipped)
    }
}

impl TrrWriter {
    pub fn create(path: impl Into<PathBuf>, n_atoms: usize) -> TrajResult<Self> {
        let path = path.into();
        let c_path = path_to_cstring(&path)?;
        let mode = CString::new("w").expect("CString::new('w') should be infallible");
        let xdr = unsafe { xdr_cabi::xdrfile_open(c_path.as_ptr(), mode.as_ptr()) };
        if xdr.is_null() {
            return Err(TrajError::Io(std::io::Error::other(format!(
                "failed to open TRR for write: {}",
                path.display()
            ))));
        }
        Ok(Self { xdr, n_atoms })
    }

    pub fn write_frame(
        &mut self,
        coords: &[[f32; 3]],
        box_: Box3,
        step: usize,
        time_ps: Option<f32>,
        velocities: Option<&[[f32; 3]]>,
        forces: Option<&[[f32; 3]]>,
        lambda: Option<f32>,
    ) -> TrajResult<()> {
        if coords.len() != self.n_atoms {
            return Err(TrajError::Mismatch(format!(
                "frame atom count {} does not match writer atom count {}",
                coords.len(),
                self.n_atoms
            )));
        }
        if let Some(vel) = velocities {
            if vel.len() != self.n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "velocity atom count {} does not match writer atom count {}",
                    vel.len(),
                    self.n_atoms
                )));
            }
        }
        if let Some(force) = forces {
            if force.len() != self.n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "force atom count {} does not match writer atom count {}",
                    force.len(),
                    self.n_atoms
                )));
            }
        }

        let box_vec = box_to_trr(box_);
        let coords_nm = scale_triplets(coords, 0.1);
        let velocities_nm = velocities.map(|vel| scale_triplets(vel, 0.1));
        let forces_native = forces.map(|force| force.to_vec());

        let code = unsafe {
            xdrfile_trr::write_trr(
                self.xdr,
                self.n_atoms as i32,
                step as i32,
                time_ps.unwrap_or(step as f32),
                lambda.unwrap_or(0.0),
                &box_vec,
                coords_nm.as_ptr() as *const Rvec,
                velocities_nm
                    .as_ref()
                    .map(|v| v.as_ptr() as *const Rvec)
                    .unwrap_or(std::ptr::null()),
                forces_native
                    .as_ref()
                    .map(|f| f.as_ptr() as *const Rvec)
                    .unwrap_or(std::ptr::null()),
            )
        };
        if code == xdr_cabi::exdrOK {
            Ok(())
        } else {
            Err(map_trr_code(code, XdrErrorTask::Write))
        }
    }

    pub fn flush(&mut self) -> TrajResult<()> {
        let code = unsafe { xdr_seek::xdr_flush(self.xdr) };
        if code == xdr_cabi::exdrOK {
            Ok(())
        } else {
            Err(map_trr_code(code, XdrErrorTask::Flush))
        }
    }
}

impl Drop for TrrWriter {
    fn drop(&mut self) {
        if !self.xdr.is_null() {
            unsafe {
                xdr_cabi::xdrfile_close(self.xdr);
            }
            self.xdr = std::ptr::null_mut();
        }
    }
}

fn path_to_cstring(path: &Path) -> TrajResult<CString> {
    let Some(s) = path.to_str() else {
        return Err(TrajError::Invalid("TRR path is not valid UTF-8".into()));
    };
    CString::new(s).map_err(|_| TrajError::Invalid("TRR path contains interior NUL".into()))
}

fn scale_triplets(values: &[[f32; 3]], scale: f32) -> Vec<[f32; 3]> {
    values
        .iter()
        .map(|value| [value[0] * scale, value[1] * scale, value[2] * scale])
        .collect()
}

fn convert_box(box_vec: Matrix) -> Box3 {
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

fn box_to_trr(box_: Box3) -> [[f32; 3]; 3] {
    match box_ {
        Box3::None => [[0.0; 3]; 3],
        Box3::Orthorhombic { lx, ly, lz } => [
            [lx * 0.1, 0.0, 0.0],
            [0.0, ly * 0.1, 0.0],
            [0.0, 0.0, lz * 0.1],
        ],
        Box3::Triclinic { m } => [
            [m[0] * 0.1, m[1] * 0.1, m[2] * 0.1],
            [m[3] * 0.1, m[4] * 0.1, m[5] * 0.1],
            [m[6] * 0.1, m[7] * 0.1, m[8] * 0.1],
        ],
    }
}

fn map_trr_code(code: i32, task: XdrErrorTask) -> TrajError {
    let err = XdrError::from((XdrErrorCode::from(code), task));
    TrajError::Parse(format!("trr error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use xdrfile::c_abi::xdrfile_trr::read_trr;
    use xdrfile::{Frame, TRRTrajectory, Trajectory};

    use tempfile::NamedTempFile;

    #[test]
    fn write_trr_simple() {
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();

        let mut writer = TrrWriter::create(path, 2).unwrap();
        writer
            .write_frame(
                &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 20.0,
                    lz: 30.0,
                },
                7,
                Some(2.5),
                Some(&[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
                None,
                Some(1.25),
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);

        let c_path = CString::new(path.to_str().unwrap()).unwrap();
        let mode = CString::new("r").unwrap();
        let mut step = 0;
        let mut time = 0.0f32;
        let mut lambda = 0.0f32;
        let mut box_vec: Matrix = [[0.0; 3]; 3];
        let mut coords = vec![[0.0; 3]; 2];
        let mut velocities = vec![[0.0; 3]; 2];
        let mut forces = vec![[0.0; 3]; 2];
        unsafe {
            let xdr = xdr_cabi::xdrfile_open(c_path.as_ptr(), mode.as_ptr());
            assert!(!xdr.is_null());
            let code = read_trr(
                xdr,
                2,
                &mut step,
                &mut time,
                &mut lambda,
                &mut box_vec,
                coords.as_mut_ptr() as *mut Rvec,
                velocities.as_mut_ptr() as *mut Rvec,
                forces.as_mut_ptr() as *mut Rvec,
            );
            assert_eq!(code, xdr_cabi::exdrOK);
            xdr_cabi::xdrfile_close(xdr);
        }
        assert_eq!(step, 7);
        assert_eq!(time, 2.5);
        assert_eq!(lambda, 1.25);

        let mut traj = TRRTrajectory::open_read(path).unwrap();
        let mut frame = Frame::with_len(2);
        traj.read(&mut frame).unwrap();
        assert_eq!(frame.step, 7);
        assert_eq!(frame.time, 2.5);
        assert_eq!(
            frame.box_vector,
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        );
        assert_eq!(frame.coords[0], [0.1, 0.2, 0.3]);
        assert_eq!(frame.coords[1], [0.4, 0.5, 0.6]);
    }

    #[test]
    fn read_trr_with_extras() {
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();

        let mut writer = TrrWriter::create(path, 1).unwrap();
        writer
            .write_frame(
                &[[1.0, 2.0, 3.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 20.0,
                    lz: 30.0,
                },
                4,
                Some(1.5),
                Some(&[[4.0, 5.0, 6.0]]),
                Some(&[[7.0, 8.0, 9.0]]),
                Some(0.75),
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);

        let mut reader = TrrReader::open(path).unwrap();
        let frames = reader
            .collect_selected_frames(&[0], true, true, true, true, true)
            .unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].coords[0], [1.0, 2.0, 3.0, 1.0]);
        assert_eq!(
            frames[0].box_,
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 20.0,
                lz: 30.0
            }
        );
        assert_eq!(frames[0].time_ps, Some(1.5));
        assert_eq!(frames[0].velocities.as_ref().unwrap()[0], [4.0, 5.0, 6.0]);
        assert_eq!(frames[0].forces.as_ref().unwrap()[0], [7.0, 8.0, 9.0]);
        assert_eq!(frames[0].lambda_value, Some(0.75));
    }

    #[test]
    fn read_chunk_surfaces_trr_extras() {
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();

        let mut writer = TrrWriter::create(path, 2).unwrap();
        writer
            .write_frame(
                &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 20.0,
                    lz: 30.0,
                },
                3,
                Some(1.25),
                Some(&[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
                Some(&[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]),
                Some(0.5),
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);

        let mut reader = TrrReader::open(path).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 1);
        builder.set_requirements(true, true);
        builder.set_optional_requirements(true, true, true);
        let read = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(read, 1);
        let chunk = builder.finish_take().unwrap();
        assert_eq!(chunk.time_ps.as_ref().unwrap(), &[1.25]);
        assert_eq!(chunk.lambda_values.as_ref().unwrap(), &[0.5]);
        assert_eq!(
            chunk.velocities.as_ref().unwrap(),
            &[[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]
        );
        assert_eq!(
            chunk.forces.as_ref().unwrap(),
            &[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]
        );

        reader.reset().unwrap();
        let mut selected = FrameChunkBuilder::new(1, 1);
        selected.set_requirements(true, true);
        selected.set_optional_requirements(true, true, true);
        let read = reader.read_chunk_selected(1, &[1], &mut selected).unwrap();
        assert_eq!(read, 1);
        let chunk = selected.finish_take().unwrap();
        assert_eq!(chunk.coords, vec![[4.0, 5.0, 6.0, 1.0]]);
        assert_eq!(chunk.time_ps.as_ref().unwrap(), &[1.25]);
        assert_eq!(chunk.lambda_values.as_ref().unwrap(), &[0.5]);
        assert_eq!(chunk.velocities.as_ref().unwrap(), &[[10.0, 11.0, 12.0]]);
        assert_eq!(chunk.forces.as_ref().unwrap(), &[[16.0, 17.0, 18.0]]);
    }
}
