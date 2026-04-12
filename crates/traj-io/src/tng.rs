use std::ffi::{c_char, c_int, c_void, CString};
use std::path::{Path, PathBuf};

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::TrajReader;

const ANGSTROM_TO_NM: f32 = 0.1;
const PS_TO_SECONDS: f64 = 1.0e-12;
const SECONDS_TO_PS: f32 = 1.0e12;
const DEFAULT_TIME_PER_FRAME_S: f64 = PS_TO_SECONDS;

const TNG_SUCCESS: c_int = 0;
const TNG_FAILURE: c_int = 1;
const TNG_CRITICAL: c_int = 2;

const TNG_INT_DATA: c_char = 1;
const TNG_FLOAT_DATA: c_char = 2;
const TNG_DOUBLE_DATA: c_char = 3;
const TNG_NON_PARTICLE_BLOCK_DATA: c_char = 0;
const TNG_GZIP_COMPRESSION: c_char = 3;
const TNG_USE_HASH: c_char = 1;

const TNG_TRAJ_BOX_SHAPE: i64 = 0x0000_0000_1000_0000;
const TNG_TRAJ_POSITIONS: i64 = 0x0000_0000_1000_0001;
const TNG_TRAJ_VELOCITIES: i64 = 0x0000_0000_1000_0002;
const TNG_TRAJ_FORCES: i64 = 0x0000_0000_1000_0003;
const TNG_WARP_FRAME_TIME_PS: i64 = 0x5752_5000_0000_0001;
const TNG_WARP_FRAME_TIME_PS_NAME: &[u8] = b"WARP_FRAME_TIME_PS\0";

#[repr(C)]
struct TngTrajectoryOpaque {
    _private: [u8; 0],
}

type TngTrajectory = *mut TngTrajectoryOpaque;

unsafe extern "C" {
    fn free(ptr: *mut c_void);

    fn tng_util_trajectory_open(
        filename: *const c_char,
        mode: c_char,
        tng_data_p: *mut TngTrajectory,
    ) -> c_int;
    fn tng_util_trajectory_close(tng_data_p: *mut TngTrajectory) -> c_int;

    fn tng_num_frames_get(tng_data: TngTrajectory, n: *mut i64) -> c_int;
    fn tng_num_particles_get(tng_data: TngTrajectory, n: *mut i64) -> c_int;
    fn tng_distance_unit_exponential_get(tng_data: TngTrajectory, exp: *mut i64) -> c_int;
    fn tng_distance_unit_exponential_set(tng_data: TngTrajectory, exp: i64) -> c_int;
    fn tng_compression_precision_set(tng_data: TngTrajectory, precision: f64) -> c_int;
    fn tng_implicit_num_particles_set(tng_data: TngTrajectory, n: i64) -> c_int;
    fn tng_num_frames_per_frame_set_set(tng_data: TngTrajectory, n: i64) -> c_int;
    fn tng_time_per_frame_set(tng_data: TngTrajectory, time: f64) -> c_int;
    fn tng_util_time_of_frame_get(tng_data: TngTrajectory, frame_nr: i64, time: *mut f64) -> c_int;
    fn tng_particle_data_vector_interval_get(
        tng_data: TngTrajectory,
        block_id: i64,
        start_frame_nr: i64,
        end_frame_nr: i64,
        hash_mode: c_char,
        values: *mut *mut c_void,
        n_particles: *mut i64,
        stride_length: *mut i64,
        n_values_per_frame: *mut i64,
        data_type: *mut c_char,
    ) -> c_int;
    fn tng_data_vector_interval_get(
        tng_data: TngTrajectory,
        block_id: i64,
        start_frame_nr: i64,
        end_frame_nr: i64,
        hash_mode: c_char,
        values: *mut *mut c_void,
        stride_length: *mut i64,
        n_values_per_frame: *mut i64,
        data_type: *mut c_char,
    ) -> c_int;
    fn tng_util_generic_write_interval_double_set(
        tng_data: TngTrajectory,
        i: i64,
        n_values_per_frame: i64,
        block_id: i64,
        block_name: *const c_char,
        particle_dependency: c_char,
        compression: c_char,
    ) -> c_int;

    fn tng_util_pos_write_interval_double_set(tng_data: TngTrajectory, i: i64) -> c_int;
    fn tng_util_vel_write_interval_double_set(tng_data: TngTrajectory, i: i64) -> c_int;
    fn tng_util_force_write_interval_double_set(tng_data: TngTrajectory, i: i64) -> c_int;
    fn tng_util_box_shape_write_interval_double_set(tng_data: TngTrajectory, i: i64) -> c_int;
    fn tng_util_pos_with_time_double_write(
        tng_data: TngTrajectory,
        frame_nr: i64,
        time: f64,
        positions: *const f64,
    ) -> c_int;
    fn tng_util_vel_with_time_double_write(
        tng_data: TngTrajectory,
        frame_nr: i64,
        time: f64,
        velocities: *const f64,
    ) -> c_int;
    fn tng_util_force_with_time_double_write(
        tng_data: TngTrajectory,
        frame_nr: i64,
        time: f64,
        forces: *const f64,
    ) -> c_int;
    fn tng_util_box_shape_with_time_double_write(
        tng_data: TngTrajectory,
        frame_nr: i64,
        time: f64,
        box_shape: *const f64,
    ) -> c_int;
    fn tng_util_generic_with_time_double_write(
        tng_data: TngTrajectory,
        frame_nr: i64,
        time: f64,
        values: *const f64,
        n_values_per_frame: i64,
        block_id: i64,
        block_name: *const c_char,
        particle_dependency: c_char,
        compression: c_char,
    ) -> c_int;

    fn tng_data_vector_get(
        tng_data: TngTrajectory,
        block_id: i64,
        values: *mut *mut c_void,
        n_frames: *mut i64,
        stride_length: *mut i64,
        n_values_per_frame: *mut i64,
        data_type: *mut c_char,
    ) -> c_int;
}

pub struct TngReader {
    path: PathBuf,
    positions: TngTrajectory,
    n_atoms: usize,
    n_frames: Option<usize>,
    position_frame_index: usize,
    distance_scale_angstrom: f32,
    static_box: Option<Box3>,
    explicit_time_ps: Option<Vec<f32>>,
}

pub struct TngWriter {
    traj: TngTrajectory,
    n_atoms: usize,
    last_frame_nr: Option<i64>,
    last_time_s: Option<f64>,
}

#[derive(Debug)]
struct OwnedTngValues(*mut c_void);

impl TngReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let positions = open_tng_handle(&path, b'r' as c_char)?;
        let n_atoms = read_i64_metadata(positions, tng_num_particles_get, "number of particles")?;
        let n_frames = read_optional_usize_metadata(positions, tng_num_frames_get);
        let distance_exp =
            read_optional_i64_metadata(positions, tng_distance_unit_exponential_get).unwrap_or(-9);
        let distance_scale_angstrom = distance_scale_to_angstrom(distance_exp);
        let static_box = read_static_box(&path, distance_scale_angstrom)?;
        let explicit_time_ps = read_explicit_time_ps(positions)?;

        Ok(Self {
            path,
            positions,
            n_atoms: usize::try_from(n_atoms)
                .map_err(|_| TrajError::Parse("TNG atom count is too large".into()))?,
            n_frames,
            position_frame_index: 0,
            distance_scale_angstrom,
            static_box,
            explicit_time_ps,
        })
    }

    pub fn reset(&mut self) -> TrajResult<()> {
        let reopened = Self::open(self.path.clone())?;
        *self = reopened;
        Ok(())
    }

    fn next_position_frame(&mut self) -> TrajResult<Option<(usize, i64, f64, Vec<f32>)>> {
        if let Some(n_frames) = self.n_frames {
            if self.position_frame_index >= n_frames {
                return Ok(None);
            }
        }
        let frame_nr = i64::try_from(self.position_frame_index)
            .map_err(|_| TrajError::Parse("TNG frame index is too large".into()))?;
        let mut time_s = 0.0f64;
        let Some(values) = read_particle_frame(
            self.positions,
            TNG_TRAJ_POSITIONS,
            frame_nr,
            self.n_atoms,
            3,
            self.distance_scale_angstrom,
            "position",
        )?
        else {
            return Ok(None);
        };
        let _ = unsafe { tng_util_time_of_frame_get(self.positions, frame_nr, &mut time_s) };
        let frame_index = self.position_frame_index;
        self.position_frame_index += 1;
        Ok(Some((frame_index, frame_nr, time_s, values)))
    }

    fn read_box_for_frame(&mut self, frame_nr: i64) -> TrajResult<Box3> {
        if let Some(box_) = self.static_box {
            return Ok(box_);
        }
        match read_non_particle_frame(
            self.positions,
            TNG_TRAJ_BOX_SHAPE,
            frame_nr,
            9,
            self.distance_scale_angstrom,
            "box",
        )? {
            Some(values) => box_from_values(&values),
            None => Ok(Box3::None),
        }
    }

    fn read_time_for_frame(
        &mut self,
        frame_index: usize,
        native_time_s: f64,
    ) -> TrajResult<Option<f32>> {
        if let Some(values) = &self.explicit_time_ps {
            if let Some(time_ps) = values.get(frame_index) {
                return Ok(Some(*time_ps));
            }
        }
        Ok(Some((native_time_s as f32) * SECONDS_TO_PS))
    }

    fn read_velocities_for_frame(&mut self, frame_nr: i64) -> TrajResult<Option<Vec<[f32; 3]>>> {
        read_particle_frame(
            self.positions,
            TNG_TRAJ_VELOCITIES,
            frame_nr,
            self.n_atoms,
            3,
            self.distance_scale_angstrom,
            "velocity",
        )?
        .map(|values| particle_values_to_triplets(&values, self.n_atoms, "velocity"))
        .transpose()
    }

    fn read_forces_for_frame(&mut self, frame_nr: i64) -> TrajResult<Option<Vec<[f32; 3]>>> {
        read_particle_frame(
            self.positions,
            TNG_TRAJ_FORCES,
            frame_nr,
            self.n_atoms,
            3,
            1.0,
            "force",
        )?
        .map(|values| particle_values_to_triplets(&values, self.n_atoms, "force"))
        .transpose()
    }
}

impl Drop for TngReader {
    fn drop(&mut self) {
        close_tng_handle(&mut self.positions);
    }
}

impl TrajReader for TngReader {
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

        out.reset(self.n_atoms, max_frames);
        let mut frames = 0usize;
        while frames < max_frames {
            let Some((frame_index, frame_nr, time_s, values)) = self.next_position_frame()? else {
                break;
            };
            let box_ = if needs_box {
                self.read_box_for_frame(frame_nr)?
            } else {
                Box3::None
            };
            let time_ps = if needs_time {
                self.read_time_for_frame(frame_index, time_s)?
            } else {
                None
            };
            let coords = out.start_frame(box_, time_ps);
            fill_coords(coords, &values, self.n_atoms)?;

            let velocities = if needs_velocities {
                self.read_velocities_for_frame(frame_nr)?
            } else {
                None
            };
            let forces = if needs_forces {
                self.read_forces_for_frame(frame_nr)?
            } else {
                None
            };
            out.set_frame_extras(velocities.as_deref(), forces.as_deref(), None)?;
            frames += 1;
        }
        Ok(frames)
    }

    fn skip_frames(&mut self, n_frames: usize) -> TrajResult<usize> {
        let mut skipped = 0usize;
        while skipped < n_frames {
            if self.next_position_frame()?.is_none() {
                break;
            }
            skipped += 1;
        }
        Ok(skipped)
    }
}

impl TngWriter {
    pub fn create(path: impl Into<PathBuf>, n_atoms: usize) -> TrajResult<Self> {
        if n_atoms == 0 {
            return Err(TrajError::Invalid(
                "TNG writer requires at least one atom".into(),
            ));
        }

        let path = path.into();
        let traj = open_tng_handle(&path, b'w' as c_char)?;
        let n_atoms_i64 = i64::try_from(n_atoms)
            .map_err(|_| TrajError::Invalid("TNG atom count does not fit in i64".into()))?;

        tng_ok(
            unsafe { tng_implicit_num_particles_set(traj, n_atoms_i64) },
            "set implicit TNG particle count",
        )?;
        tng_ok(
            unsafe { tng_distance_unit_exponential_set(traj, -9) },
            "set TNG distance unit",
        )?;
        tng_ok(
            unsafe { tng_compression_precision_set(traj, 1000.0) },
            "set TNG compression precision",
        )?;
        tng_ok(
            unsafe { tng_num_frames_per_frame_set_set(traj, 1) },
            "set TNG frames per frame set",
        )?;
        tng_ok(
            unsafe { tng_util_pos_write_interval_double_set(traj, 1) },
            "set TNG position interval",
        )?;
        tng_ok(
            unsafe { tng_util_vel_write_interval_double_set(traj, 1) },
            "set TNG velocity interval",
        )?;
        tng_ok(
            unsafe { tng_util_force_write_interval_double_set(traj, 1) },
            "set TNG force interval",
        )?;
        tng_ok(
            unsafe { tng_util_box_shape_write_interval_double_set(traj, 1) },
            "set TNG box interval",
        )?;
        tng_ok(
            unsafe {
                tng_util_generic_write_interval_double_set(
                    traj,
                    1,
                    1,
                    TNG_WARP_FRAME_TIME_PS,
                    TNG_WARP_FRAME_TIME_PS_NAME.as_ptr().cast(),
                    TNG_NON_PARTICLE_BLOCK_DATA,
                    TNG_GZIP_COMPRESSION,
                )
            },
            "set TNG frame-time interval",
        )?;

        Ok(Self {
            traj,
            n_atoms,
            last_frame_nr: None,
            last_time_s: None,
        })
    }

    pub fn write_frame(
        &mut self,
        coords: &[[f32; 3]],
        box_: Box3,
        step: usize,
        time_ps: Option<f32>,
        velocities: Option<&[[f32; 3]]>,
        forces: Option<&[[f32; 3]]>,
    ) -> TrajResult<()> {
        if self.traj.is_null() {
            return Err(TrajError::Invalid("TNG writer is closed".into()));
        }
        if coords.len() != self.n_atoms {
            return Err(TrajError::Mismatch(format!(
                "frame atom count {} does not match writer atom count {}",
                coords.len(),
                self.n_atoms
            )));
        }
        if let Some(values) = velocities {
            if values.len() != self.n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "velocity atom count {} does not match writer atom count {}",
                    values.len(),
                    self.n_atoms
                )));
            }
        }
        if let Some(values) = forces {
            if values.len() != self.n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "force atom count {} does not match writer atom count {}",
                    values.len(),
                    self.n_atoms
                )));
            }
        }

        // Keep TNG frame numbers dense and sequential. Our reader surface exposes
        // frame order and time, not simulation step ids, and sparse starting steps
        // trigger empty native frame sets that break time recovery.
        let frame_nr = self.last_frame_nr.map_or(0, |last| last + 1);
        let default_time_ps = step as f32;
        let frame_time_ps = f64::from(time_ps.unwrap_or(default_time_ps));
        let time_s = frame_time_ps * PS_TO_SECONDS;
        let time_per_frame_s = match (self.last_frame_nr, self.last_time_s) {
            (Some(last_frame), Some(last_time_s)) if frame_nr > last_frame => {
                let delta_time = time_s - last_time_s;
                if delta_time > 0.0 {
                    delta_time / (frame_nr - last_frame) as f64
                } else {
                    DEFAULT_TIME_PER_FRAME_S
                }
            }
            _ => DEFAULT_TIME_PER_FRAME_S,
        };
        tng_ok(
            unsafe { tng_time_per_frame_set(self.traj, time_per_frame_s) },
            "set TNG time per frame",
        )?;

        let coords_nm = flatten_triplets_f64(coords, f64::from(ANGSTROM_TO_NM));
        tng_ok(
            unsafe {
                tng_util_pos_with_time_double_write(self.traj, frame_nr, time_s, coords_nm.as_ptr())
            },
            "write TNG positions",
        )?;
        tng_ok(
            unsafe {
                tng_util_generic_with_time_double_write(
                    self.traj,
                    frame_nr,
                    time_s,
                    &frame_time_ps,
                    1,
                    TNG_WARP_FRAME_TIME_PS,
                    TNG_WARP_FRAME_TIME_PS_NAME.as_ptr().cast(),
                    TNG_NON_PARTICLE_BLOCK_DATA,
                    TNG_GZIP_COMPRESSION,
                )
            },
            "write TNG frame time block",
        )?;

        if let Some(box_nm) = box_to_tng_f64(box_) {
            tng_ok(
                unsafe {
                    tng_util_box_shape_with_time_double_write(
                        self.traj,
                        frame_nr,
                        time_s,
                        box_nm.as_ptr(),
                    )
                },
                "write TNG box",
            )?;
        }
        if let Some(values) = velocities {
            let velocities_nm = flatten_triplets_f64(values, f64::from(ANGSTROM_TO_NM));
            tng_ok(
                unsafe {
                    tng_util_vel_with_time_double_write(
                        self.traj,
                        frame_nr,
                        time_s,
                        velocities_nm.as_ptr(),
                    )
                },
                "write TNG velocities",
            )?;
        }
        if let Some(values) = forces {
            let forces_native = flatten_triplets_f64(values, 1.0);
            tng_ok(
                unsafe {
                    tng_util_force_with_time_double_write(
                        self.traj,
                        frame_nr,
                        time_s,
                        forces_native.as_ptr(),
                    )
                },
                "write TNG forces",
            )?;
        }

        self.last_frame_nr = Some(frame_nr);
        self.last_time_s = Some(time_s);
        Ok(())
    }

    pub fn flush(&mut self) -> TrajResult<()> {
        close_tng_handle(&mut self.traj);
        Ok(())
    }
}

impl Drop for TngWriter {
    fn drop(&mut self) {
        close_tng_handle(&mut self.traj);
    }
}

impl OwnedTngValues {
    unsafe fn from_raw(raw: *mut c_void) -> Self {
        Self(raw)
    }

    unsafe fn convert_to_f32(
        self,
        count: usize,
        data_type: c_char,
        scale: f32,
    ) -> TrajResult<Vec<f32>> {
        let ptr = self.0;
        let values = match data_type {
            TNG_INT_DATA => {
                let slice = std::slice::from_raw_parts(ptr.cast::<i32>(), count);
                slice.iter().map(|value| *value as f32 * scale).collect()
            }
            TNG_FLOAT_DATA => {
                let slice = std::slice::from_raw_parts(ptr.cast::<f32>(), count);
                slice.iter().map(|value| *value * scale).collect()
            }
            TNG_DOUBLE_DATA => {
                let slice = std::slice::from_raw_parts(ptr.cast::<f64>(), count);
                slice.iter().map(|value| *value as f32 * scale).collect()
            }
            other => {
                return Err(TrajError::Parse(format!(
                    "unsupported TNG data type: {other}"
                )))
            }
        };
        Ok(values)
    }
}

impl Drop for OwnedTngValues {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { free(self.0) };
            self.0 = std::ptr::null_mut();
        }
    }
}

fn open_tng_handle(path: &Path, mode: c_char) -> TrajResult<TngTrajectory> {
    let c_path = path_to_cstring(path)?;
    let mut traj = std::ptr::null_mut();
    tng_ok(
        unsafe { tng_util_trajectory_open(c_path.as_ptr(), mode, &mut traj) },
        &format!("open TNG trajectory: {}", path.display()),
    )?;
    Ok(traj)
}

fn close_tng_handle(traj: &mut TngTrajectory) {
    if !traj.is_null() {
        let _ = unsafe { tng_util_trajectory_close(traj) };
        *traj = std::ptr::null_mut();
    }
}

fn path_to_cstring(path: &Path) -> TrajResult<CString> {
    let Some(path_str) = path.to_str() else {
        return Err(TrajError::Invalid("TNG path is not valid UTF-8".into()));
    };
    CString::new(path_str).map_err(|_| TrajError::Invalid("TNG path contains interior NUL".into()))
}

fn read_i64_metadata(
    traj: TngTrajectory,
    getter: unsafe extern "C" fn(TngTrajectory, *mut i64) -> c_int,
    what: &str,
) -> TrajResult<i64> {
    let mut value = 0i64;
    tng_ok(
        unsafe { getter(traj, &mut value) },
        &format!("read TNG {what}"),
    )?;
    Ok(value)
}

fn read_optional_i64_metadata(
    traj: TngTrajectory,
    getter: unsafe extern "C" fn(TngTrajectory, *mut i64) -> c_int,
) -> Option<i64> {
    let mut value = 0i64;
    let status = unsafe { getter(traj, &mut value) };
    (status == TNG_SUCCESS).then_some(value)
}

fn read_optional_usize_metadata(
    traj: TngTrajectory,
    getter: unsafe extern "C" fn(TngTrajectory, *mut i64) -> c_int,
) -> Option<usize> {
    read_optional_i64_metadata(traj, getter).and_then(|value| usize::try_from(value).ok())
}

fn read_static_box(path: &Path, scale: f32) -> TrajResult<Option<Box3>> {
    let mut traj = open_tng_handle(path, b'r' as c_char)?;
    let mut raw = std::ptr::null_mut();
    let mut n_frames = 0i64;
    let mut stride_length = 0i64;
    let mut n_values = 0i64;
    let mut data_type = 0;
    let result = match unsafe {
        tng_data_vector_get(
            traj,
            TNG_TRAJ_BOX_SHAPE,
            &mut raw,
            &mut n_frames,
            &mut stride_length,
            &mut n_values,
            &mut data_type,
        )
    } {
        TNG_SUCCESS if n_frames == 1 && n_values == 9 => {
            let values =
                unsafe { OwnedTngValues::from_raw(raw).convert_to_f32(9, data_type, scale)? };
            box_from_values(&values).map(Some)
        }
        TNG_SUCCESS | TNG_FAILURE => {
            if !raw.is_null() {
                unsafe { free(raw) };
            }
            Ok(None)
        }
        status => Err(tng_status_error(status, "read static TNG box")),
    };
    close_tng_handle(&mut traj);
    result
}

fn read_explicit_time_ps(traj: TngTrajectory) -> TrajResult<Option<Vec<f32>>> {
    let mut raw = std::ptr::null_mut();
    let mut n_frames = 0i64;
    let mut stride_length = 0i64;
    let mut n_values = 0i64;
    let mut data_type = 0;
    match unsafe {
        tng_data_vector_get(
            traj,
            TNG_WARP_FRAME_TIME_PS,
            &mut raw,
            &mut n_frames,
            &mut stride_length,
            &mut n_values,
            &mut data_type,
        )
    } {
        TNG_SUCCESS if n_values == 1 && stride_length == 1 && n_frames >= 0 => {
            let count = usize::try_from(n_frames)
                .map_err(|_| TrajError::Parse("TNG frame-time vector is too large".into()))?;
            let values =
                unsafe { OwnedTngValues::from_raw(raw).convert_to_f32(count, data_type, 1.0)? };
            Ok(Some(values))
        }
        TNG_SUCCESS | TNG_FAILURE => {
            if !raw.is_null() {
                unsafe { free(raw) };
            }
            Ok(None)
        }
        status => Err(tng_status_error(status, "read TNG frame-time vector")),
    }
}

fn read_particle_frame(
    traj: TngTrajectory,
    block_id: i64,
    frame_nr: i64,
    n_atoms: usize,
    n_values_per_atom: usize,
    scale: f32,
    label: &str,
) -> TrajResult<Option<Vec<f32>>> {
    let mut raw = std::ptr::null_mut();
    let mut n_particles = 0i64;
    let mut stride_length = 0i64;
    let mut n_values_per_frame = 0i64;
    let mut data_type = 0;
    match unsafe {
        tng_particle_data_vector_interval_get(
            traj,
            block_id,
            frame_nr,
            frame_nr,
            TNG_USE_HASH,
            &mut raw,
            &mut n_particles,
            &mut stride_length,
            &mut n_values_per_frame,
            &mut data_type,
        )
    } {
        TNG_SUCCESS => {
            let expected_count = n_atoms
                .checked_mul(n_values_per_atom)
                .ok_or_else(|| TrajError::Parse(format!("TNG {label} payload is too large")))?;
            let actual_count = usize::try_from(n_particles).ok().and_then(|particles| {
                usize::try_from(n_values_per_frame)
                    .ok()
                    .and_then(|values| particles.checked_mul(values))
            });
            if stride_length <= 0 || actual_count != Some(expected_count) {
                if !raw.is_null() {
                    unsafe { free(raw) };
                }
                return Err(TrajError::Parse(format!(
                    "unexpected TNG {label} payload metadata"
                )));
            }
            let count = n_atoms * n_values_per_atom;
            let values =
                unsafe { OwnedTngValues::from_raw(raw).convert_to_f32(count, data_type, scale)? };
            Ok(Some(values))
        }
        TNG_FAILURE => {
            if !raw.is_null() {
                unsafe { free(raw) };
            }
            Ok(None)
        }
        status => Err(tng_status_error(status, &format!("read TNG {label} frame"))),
    }
}

fn read_non_particle_frame(
    traj: TngTrajectory,
    block_id: i64,
    frame_nr: i64,
    n_values_expected: usize,
    scale: f32,
    label: &str,
) -> TrajResult<Option<Vec<f32>>> {
    let mut raw = std::ptr::null_mut();
    let mut stride_length = 0i64;
    let mut n_values_per_frame = 0i64;
    let mut data_type = 0;
    match unsafe {
        tng_data_vector_interval_get(
            traj,
            block_id,
            frame_nr,
            frame_nr,
            TNG_USE_HASH,
            &mut raw,
            &mut stride_length,
            &mut n_values_per_frame,
            &mut data_type,
        )
    } {
        TNG_SUCCESS => {
            if stride_length <= 0
                || usize::try_from(n_values_per_frame).ok() != Some(n_values_expected)
            {
                if !raw.is_null() {
                    unsafe { free(raw) };
                }
                return Err(TrajError::Parse(format!(
                    "unexpected TNG {label} payload metadata"
                )));
            }
            let values = unsafe {
                OwnedTngValues::from_raw(raw).convert_to_f32(n_values_expected, data_type, scale)?
            };
            Ok(Some(values))
        }
        TNG_FAILURE => {
            if !raw.is_null() {
                unsafe { free(raw) };
            }
            Ok(None)
        }
        status => Err(tng_status_error(status, &format!("read TNG {label} frame"))),
    }
}

fn distance_scale_to_angstrom(exp: i64) -> f32 {
    10f32.powi((exp + 10) as i32)
}

fn fill_coords(dst: &mut [[f32; 4]], src: &[f32], n_atoms: usize) -> TrajResult<()> {
    if src.len() != n_atoms * 3 {
        return Err(TrajError::Parse(
            "unexpected TNG coordinate payload length".into(),
        ));
    }
    for (dst_atom, src_atom) in dst.iter_mut().zip(src.chunks_exact(3)) {
        dst_atom[0] = src_atom[0];
        dst_atom[1] = src_atom[1];
        dst_atom[2] = src_atom[2];
        dst_atom[3] = 1.0;
    }
    Ok(())
}

fn particle_values_to_triplets(
    values: &[f32],
    n_atoms: usize,
    label: &str,
) -> TrajResult<Vec<[f32; 3]>> {
    if values.len() != n_atoms * 3 {
        return Err(TrajError::Parse(format!(
            "unexpected TNG {label} payload length: {}",
            values.len()
        )));
    }
    Ok(values
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect())
}

fn box_from_values(values: &[f32]) -> TrajResult<Box3> {
    if values.len() != 9 {
        return Err(TrajError::Parse("unexpected TNG box payload length".into()));
    }
    let m = [
        values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
        values[8],
    ];
    let tol = 1e-5;
    let is_orth = m[1].abs() < tol
        && m[2].abs() < tol
        && m[3].abs() < tol
        && m[5].abs() < tol
        && m[6].abs() < tol
        && m[7].abs() < tol;
    Ok(if is_orth {
        Box3::Orthorhombic {
            lx: m[0],
            ly: m[4],
            lz: m[8],
        }
    } else {
        Box3::Triclinic { m }
    })
}

fn box_to_tng_f64(box_: Box3) -> Option<[f64; 9]> {
    match box_ {
        Box3::None => None,
        Box3::Orthorhombic { lx, ly, lz } => Some([
            f64::from(lx) * f64::from(ANGSTROM_TO_NM),
            0.0,
            0.0,
            0.0,
            f64::from(ly) * f64::from(ANGSTROM_TO_NM),
            0.0,
            0.0,
            0.0,
            f64::from(lz) * f64::from(ANGSTROM_TO_NM),
        ]),
        Box3::Triclinic { m } => Some([
            f64::from(m[0]) * f64::from(ANGSTROM_TO_NM),
            f64::from(m[1]) * f64::from(ANGSTROM_TO_NM),
            f64::from(m[2]) * f64::from(ANGSTROM_TO_NM),
            f64::from(m[3]) * f64::from(ANGSTROM_TO_NM),
            f64::from(m[4]) * f64::from(ANGSTROM_TO_NM),
            f64::from(m[5]) * f64::from(ANGSTROM_TO_NM),
            f64::from(m[6]) * f64::from(ANGSTROM_TO_NM),
            f64::from(m[7]) * f64::from(ANGSTROM_TO_NM),
            f64::from(m[8]) * f64::from(ANGSTROM_TO_NM),
        ]),
    }
}

fn flatten_triplets_f64(values: &[[f32; 3]], scale: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len() * 3);
    for value in values {
        out.push(f64::from(value[0]) * scale);
        out.push(f64::from(value[1]) * scale);
        out.push(f64::from(value[2]) * scale);
    }
    out
}

fn tng_ok(status: c_int, context: &str) -> TrajResult<()> {
    match status {
        TNG_SUCCESS => Ok(()),
        other => Err(tng_status_error(other, context)),
    }
}

fn tng_status_error(status: c_int, context: &str) -> TrajError {
    let level = match status {
        TNG_FAILURE => "failure",
        TNG_CRITICAL => "critical",
        _ => "unknown",
    };
    TrajError::Parse(format!("TNG {level}: {context}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, OnceLock};
    use tempfile::NamedTempFile;

    fn tng_test_guard() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
    }

    fn assert_f32_slice_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn write_tng_roundtrip_with_extras() {
        let _guard = tng_test_guard();
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();

        let mut writer = TngWriter::create(path, 2).unwrap();
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
                Some(&[[13.0, 14.0, 15.0], [16.0, 17.0, 18.0]]),
            )
            .unwrap();
        writer
            .write_frame(
                &[[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
                Box3::Triclinic {
                    m: [10.0, 1.0, 0.0, 0.0, 20.0, 2.0, 0.0, 0.0, 30.0],
                },
                7,
                Some(3.5),
                None,
                None,
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);

        let mut reader = TngReader::open(path).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 2);
        builder.set_requirements(true, true);
        builder.set_optional_requirements(true, true, false);
        let read = reader.read_chunk(2, &mut builder).unwrap();
        assert_eq!(read, 2);
        let chunk = builder.finish_take().unwrap();
        assert_eq!(chunk.n_frames, 2);
        assert_eq!(chunk.time_ps.as_ref().unwrap(), &[2.5, 3.5]);
        assert_eq!(
            chunk.box_[0],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 20.0,
                lz: 30.0
            }
        );
        assert_eq!(
            chunk.box_[1],
            Box3::Triclinic {
                m: [10.0, 1.0, 0.0, 0.0, 20.0, 2.0, 0.0, 0.0, 30.0]
            }
        );
        assert_f32_slice_close(&chunk.coords[0], &[1.0, 2.0, 3.0, 1.0]);
        assert_f32_slice_close(&chunk.coords[1], &[4.0, 5.0, 6.0, 1.0]);
        assert_f32_slice_close(&chunk.coords[2], &[2.0, 3.0, 4.0, 1.0]);
        assert_f32_slice_close(&chunk.coords[3], &[5.0, 6.0, 7.0, 1.0]);
        let velocities = chunk.velocities.as_ref().unwrap();
        assert_f32_slice_close(&velocities[0], &[7.0, 8.0, 9.0]);
        assert_f32_slice_close(&velocities[1], &[10.0, 11.0, 12.0]);
        assert_eq!(velocities[2], [0.0, 0.0, 0.0]);
        assert_eq!(velocities[3], [0.0, 0.0, 0.0]);
        let forces = chunk.forces.as_ref().unwrap();
        assert_f32_slice_close(&forces[0], &[13.0, 14.0, 15.0]);
        assert_f32_slice_close(&forces[1], &[16.0, 17.0, 18.0]);
        assert_eq!(forces[2], [0.0, 0.0, 0.0]);
        assert_eq!(forces[3], [0.0, 0.0, 0.0]);

        reader.reset().unwrap();
        let mut selected = FrameChunkBuilder::new(1, 1);
        selected.set_requirements(true, true);
        let read = reader.read_chunk_selected(1, &[1], &mut selected).unwrap();
        assert_eq!(read, 1);
        let chunk = selected.finish_take().unwrap();
        assert_eq!(chunk.coords.len(), 1);
        assert_f32_slice_close(&chunk.coords[0], &[4.0, 5.0, 6.0, 1.0]);
    }

    #[test]
    fn reset_rewinds_tng_reader() {
        let _guard = tng_test_guard();
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();

        let mut writer = TngWriter::create(path, 1).unwrap();
        writer
            .write_frame(
                &[[1.0, 0.0, 0.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 10.0,
                    lz: 10.0,
                },
                0,
                Some(0.0),
                None,
                None,
            )
            .unwrap();
        writer
            .write_frame(
                &[[2.0, 0.0, 0.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 10.0,
                    lz: 10.0,
                },
                1,
                Some(1.0),
                None,
                None,
            )
            .unwrap();
        drop(writer);

        let mut reader = TngReader::open(path).unwrap();
        let mut builder = FrameChunkBuilder::new(1, 1);
        let read = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(read, 1);
        let chunk = builder.finish_take().unwrap();
        assert_f32_slice_close(&chunk.coords[0], &[1.0, 0.0, 0.0, 1.0]);

        reader.reset().unwrap();
        let mut builder = FrameChunkBuilder::new(1, 1);
        let read = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(read, 1);
        let chunk = builder.finish_take().unwrap();
        assert_f32_slice_close(&chunk.coords[0], &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn reader_handles_double_precision_tng_blocks() {
        let _guard = tng_test_guard();
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();
        write_double_precision_tng(path).unwrap();

        let mut reader = TngReader::open(path).unwrap();
        let mut builder = FrameChunkBuilder::new(1, 1);
        builder.set_requirements(true, true);
        builder.set_optional_requirements(true, true, false);
        let read = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(read, 1);
        let chunk = builder.finish_take().unwrap();
        assert_f32_slice_close(&chunk.coords[0], &[1.0, 2.0, 3.0, 1.0]);
        assert_eq!(
            chunk.box_[0],
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 20.0,
                lz: 30.0
            }
        );
        assert_eq!(chunk.time_ps.as_ref().unwrap(), &[2.5]);
        assert_f32_slice_close(&chunk.velocities.as_ref().unwrap()[0], &[4.0, 5.0, 6.0]);
        assert_f32_slice_close(&chunk.forces.as_ref().unwrap()[0], &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn reader_handles_gromacs_tng_fixture() {
        let _guard = tng_test_guard();
        let fixture_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/warp_md/tests/fixtures/tng");
        let tng_path = fixture_dir.join("spc2-traj.tng");
        let xtc_path = fixture_dir.join("spc2-traj.xtc");

        let mut tng = TngReader::open(&tng_path).unwrap();
        let mut xtc = crate::xtc::XtcReader::open(&xtc_path).unwrap();
        let mut tng_builder = FrameChunkBuilder::new(6, 2);
        tng_builder.set_requirements(true, true);
        let mut xtc_builder = FrameChunkBuilder::new(6, 2);
        xtc_builder.set_requirements(true, true);

        assert_eq!(tng.read_chunk(2, &mut tng_builder).unwrap(), 2);
        assert_eq!(xtc.read_chunk(2, &mut xtc_builder).unwrap(), 2);

        let tng_chunk = tng_builder.finish_take().unwrap();
        let xtc_chunk = xtc_builder.finish_take().unwrap();
        assert_eq!(tng_chunk.n_frames, 2);
        assert_eq!(tng_chunk.time_ps.as_ref().unwrap(), &[0.0, 1.0]);
        assert_eq!(tng_chunk.box_, xtc_chunk.box_);
        assert_eq!(tng_chunk.coords.len(), xtc_chunk.coords.len());
        for (tng_atom, xtc_atom) in tng_chunk.coords.iter().zip(xtc_chunk.coords.iter()) {
            assert_f32_slice_close(tng_atom, xtc_atom);
        }
    }

    #[test]
    fn flush_finalizes_tng_for_immediate_readback() {
        let _guard = tng_test_guard();
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();

        let mut writer = TngWriter::create(path, 1).unwrap();
        writer
            .write_frame(
                &[[1.0, 2.0, 3.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 10.0,
                    lz: 10.0,
                },
                0,
                Some(0.0),
                None,
                None,
            )
            .unwrap();
        writer
            .write_frame(
                &[[4.0, 5.0, 6.0]],
                Box3::Orthorhombic {
                    lx: 10.0,
                    ly: 10.0,
                    lz: 10.0,
                },
                1,
                Some(1.0),
                None,
                None,
            )
            .unwrap();
        writer.flush().unwrap();

        let mut reader = TngReader::open(path).unwrap();
        let mut builder = FrameChunkBuilder::new(1, 2);
        builder.set_requirements(true, true);
        assert_eq!(reader.read_chunk(2, &mut builder).unwrap(), 2);
        let chunk = builder.finish_take().unwrap();
        assert_eq!(chunk.time_ps.as_ref().unwrap(), &[0.0, 1.0]);
        assert_f32_slice_close(&chunk.coords[0], &[1.0, 2.0, 3.0, 1.0]);
        assert_f32_slice_close(&chunk.coords[1], &[4.0, 5.0, 6.0, 1.0]);

        let err = writer
            .write_frame(&[[7.0, 8.0, 9.0]], Box3::None, 2, Some(2.0), None, None)
            .unwrap_err();
        assert!(matches!(err, TrajError::Invalid(message) if message.contains("closed")));
    }

    fn write_double_precision_tng(path: &Path) -> TrajResult<()> {
        let mut traj = open_tng_handle(path, b'w' as c_char)?;
        tng_ok(
            unsafe { tng_implicit_num_particles_set(traj, 1) },
            "set double-test particles",
        )?;
        tng_ok(
            unsafe { tng_distance_unit_exponential_set(traj, -9) },
            "set double-test distance unit",
        )?;
        tng_ok(
            unsafe { tng_num_frames_per_frame_set_set(traj, 1) },
            "set double-test frame set size",
        )?;
        tng_ok(
            unsafe { tng_time_per_frame_set(traj, DEFAULT_TIME_PER_FRAME_S) },
            "set double-test time per frame",
        )?;
        tng_ok(
            unsafe { tng_util_pos_write_interval_double_set(traj, 1) },
            "set double-test position interval",
        )?;
        tng_ok(
            unsafe { tng_util_vel_write_interval_double_set(traj, 1) },
            "set double-test velocity interval",
        )?;
        tng_ok(
            unsafe { tng_util_force_write_interval_double_set(traj, 1) },
            "set double-test force interval",
        )?;
        tng_ok(
            unsafe { tng_util_box_shape_write_interval_double_set(traj, 1) },
            "set double-test box interval",
        )?;

        let positions = [0.1f64, 0.2, 0.3];
        let velocities = [0.4f64, 0.5, 0.6];
        let forces = [7.0f64, 8.0, 9.0];
        let box_shape = [1.0f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        tng_ok(
            unsafe {
                tng_util_pos_with_time_double_write(
                    traj,
                    0,
                    2.5 * PS_TO_SECONDS,
                    positions.as_ptr(),
                )
            },
            "write double-test positions",
        )?;
        tng_ok(
            unsafe {
                tng_util_vel_with_time_double_write(
                    traj,
                    0,
                    2.5 * PS_TO_SECONDS,
                    velocities.as_ptr(),
                )
            },
            "write double-test velocities",
        )?;
        tng_ok(
            unsafe {
                tng_util_force_with_time_double_write(traj, 0, 2.5 * PS_TO_SECONDS, forces.as_ptr())
            },
            "write double-test forces",
        )?;
        tng_ok(
            unsafe {
                tng_util_box_shape_with_time_double_write(
                    traj,
                    0,
                    2.5 * PS_TO_SECONDS,
                    box_shape.as_ptr(),
                )
            },
            "write double-test box",
        )?;
        close_tng_handle(&mut traj);
        Ok(())
    }
}
