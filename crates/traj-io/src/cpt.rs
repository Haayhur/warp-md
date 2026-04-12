use std::ffi::CString;
use std::path::{Path, PathBuf};

use ::xdrfile::c_abi::xdr_seek;
use ::xdrfile::c_abi::xdrfile::{self as xdr_cabi, XDRFILE};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::TrajReader;

const ANGSTROM_TO_NM: f32 = 0.1;
const NM_TO_ANGSTROM: f32 = 10.0;
const BOX_TOL: f32 = 1.0e-6;
const CPT_MAGIC1: i32 = 171_817;
const CPT_MAGIC2: i32 = 171_819;
const CPT_STRLEN: usize = 1024;
const CHECKPOINT_VERSION_MD_MODULES: i32 = 21;
const CHECKPOINT_VERSION_MODULAR_SIMULATOR: i32 = 22;
const CHECKPOINT_VERSION_ADD_MAGIC_NUMBER: i32 = 2;
const CHECKPOINT_VERSION_HOST_INFORMATION: i32 = 12;
const CHECKPOINT_VERSION_DOUBLE_PRECISION_BUILD: i32 = 13;
const CHECKPOINT_VERSION_NOSE_HOOVER_THERMOSTAT: i32 = 10;
const CHECKPOINT_VERSION_NOSE_HOOVER_BAROSTAT: i32 = 11;
const CHECKPOINT_VERSION_LAMBDA_STATE_AND_HISTORY: i32 = 14;
const CHECKPOINT_VERSION_SAFE_SIMULATION_PART: i32 = 3;
const CHECKPOINT_VERSION_SAFE_STEPS: i32 = 5;
const CHECKPOINT_VERSION_EKIN_DATA_AND_FLAGS: i32 = 4;
const CHECKPOINT_VERSION_ESSENTIAL_DYNAMICS: i32 = 15;
const CHECKPOINT_VERSION_SWAP_STATE: i32 = 16;
const CHECKPOINT_VERSION_AWH_HISTORY_FLAGS: i32 = 17;
const CHECKPOINT_VERSION_REMOVE_BUILD_MACHINE_INFORMATION: i32 = 18;
const CHECKPOINT_VERSION_FILE_CHECKSUM_AND_SIZE: i32 = 8;
const CPT_WRITER_FILE_VERSION: i32 = 20;
const GROMACS_CHECKSUM_LEN: usize = 16;
const FEP_LAMBDA_COUNT: usize = 7;
const FEP_LAMBDA_INDEX: usize = 0;

const STATE_ENTRY_LAMBDA: usize = 0;
const STATE_ENTRY_BOX: usize = 1;
const STATE_ENTRY_BOX_REL: usize = 2;
const STATE_ENTRY_BOX_V: usize = 3;
const STATE_ENTRY_PRESSURE_PREVIOUS: usize = 4;
const STATE_ENTRY_NHXI: usize = 5;
const STATE_ENTRY_THERM_INT: usize = 6;
const STATE_ENTRY_X: usize = 7;
const STATE_ENTRY_V: usize = 8;
const STATE_ENTRY_CGP: usize = 10;
const STATE_ENTRY_DISRE_INIT_F: usize = 13;
const STATE_ENTRY_DISRE_RM3TAV: usize = 14;
const STATE_ENTRY_ORIRE_INIT_F: usize = 15;
const STATE_ENTRY_ORIRE_DTAV: usize = 16;
const STATE_ENTRY_SVIR_PREV: usize = 17;
const STATE_ENTRY_NHVXI: usize = 18;
const STATE_ENTRY_VETA: usize = 19;
const STATE_ENTRY_VOL0: usize = 20;
const STATE_ENTRY_NHPRESXI: usize = 21;
const STATE_ENTRY_NHPRESVXI: usize = 22;
const STATE_ENTRY_FVIR_PREV: usize = 23;
const STATE_ENTRY_FEP_STATE: usize = 24;
const STATE_ENTRY_BAROS_INT: usize = 27;
const STATE_ENTRY_PULL_COM_PREV_STEP: usize = 28;
const STATE_ENTRY_COUNT: usize = 29;

const EKIN_ENTRY_NUMBER: usize = 0;
const EKIN_ENTRY_HALF_STEP: usize = 1;
const EKIN_ENTRY_DEKIN_D_LAMBDA: usize = 2;
const EKIN_ENTRY_MVCOS: usize = 3;
const EKIN_ENTRY_FULL_STEP: usize = 4;
const EKIN_ENTRY_HALF_STEP_OLD: usize = 5;
const EKIN_ENTRY_SCALE_FULL_STEP: usize = 6;
const EKIN_ENTRY_SCALE_HALF_STEP: usize = 7;
const EKIN_ENTRY_VELOCITY_SCALE: usize = 8;
const EKIN_ENTRY_TOTAL: usize = 9;
const EKIN_ENTRY_COUNT: usize = 10;

const ENERGY_ENTRY_N: usize = 0;
const ENERGY_ENTRY_AVER: usize = 1;
const ENERGY_ENTRY_SUM: usize = 2;
const ENERGY_ENTRY_NUM_SUM: usize = 3;
const ENERGY_ENTRY_SUM_SIM: usize = 4;
const ENERGY_ENTRY_NUM_SUM_SIM: usize = 5;
const ENERGY_ENTRY_NUM_STEPS: usize = 6;
const ENERGY_ENTRY_NUM_STEPS_SIM: usize = 7;
const ENERGY_ENTRY_DELTA_H_NN: usize = 8;
const ENERGY_ENTRY_DELTA_H_LIST: usize = 9;
const ENERGY_ENTRY_DELTA_H_START_TIME: usize = 10;
const ENERGY_ENTRY_DELTA_H_START_LAMBDA: usize = 11;
const ENERGY_ENTRY_COUNT: usize = 12;

const PULL_ENTRY_NUM_COORDINATES: usize = 0;
const PULL_ENTRY_NUM_GROUPS: usize = 1;
const PULL_ENTRY_NUM_VALUES_IN_X_SUM: usize = 2;
const PULL_ENTRY_NUM_VALUES_IN_F_SUM: usize = 3;
const PULL_ENTRY_COUNT: usize = 4;

const DF_ENTRY_IS_EQUILIBRATED: usize = 0;
const DF_ENTRY_NUM_AT_LAMBDA_STATS: usize = 1;
const DF_ENTRY_NUM_AT_LAMBDA_EQUIL: usize = 2;
const DF_ENTRY_WANG_LANDAU_HISTOGRAM: usize = 3;
const DF_ENTRY_WANG_LANDAU_DELTA: usize = 4;
const DF_ENTRY_SUM_WEIGHTS: usize = 5;
const DF_ENTRY_SUM_DG: usize = 6;
const DF_ENTRY_SUM_MIN_VAR: usize = 7;
const DF_ENTRY_SUM_VAR: usize = 8;
const DF_ENTRY_ACCUM_P: usize = 9;
const DF_ENTRY_ACCUM_M: usize = 10;
const DF_ENTRY_ACCUM_P2: usize = 11;
const DF_ENTRY_ACCUM_M2: usize = 12;
const DF_ENTRY_TIJ: usize = 13;
const DF_ENTRY_TIJ_EMP: usize = 14;
const DF_ENTRY_COUNT: usize = 15;

const KVT_TAG_OBJECT: u8 = b'O';
const KVT_TAG_ARRAY: u8 = b'A';
const KVT_TAG_STRING: u8 = b's';
const KVT_TAG_BOOL: u8 = b'b';
const KVT_TAG_CHAR: u8 = b'c';
const KVT_TAG_UCHAR: u8 = b'u';
const KVT_TAG_INT: u8 = b'i';
const KVT_TAG_INT64: u8 = b'l';
const KVT_TAG_FLOAT: u8 = b'f';
const KVT_TAG_DOUBLE: u8 = b'd';

#[derive(Clone, Debug)]
pub struct CptFrameData {
    pub step: i64,
    pub simulation_part: i32,
    pub coords: Vec<[f32; 3]>,
    pub box_: Box3,
    pub time_ps: Option<f32>,
    pub velocities: Option<Vec<[f32; 3]>>,
    pub lambda_value: Option<f32>,
    pub fep_state: Option<i32>,
}

pub struct CptReader {
    path: PathBuf,
    frame: CptFrameData,
    consumed: bool,
}

pub struct CptWriter {
    xdr: XdrHandle,
    n_atoms: usize,
    wrote_frame: bool,
}

#[derive(Clone, Debug)]
struct CptHeader {
    file_version: i32,
    double_prec: i32,
    natoms: usize,
    ngtc: i32,
    nhchainlength: i32,
    nnhpres: i32,
    nlambda: i32,
    integrator: i32,
    simulation_part: i32,
    step: i64,
    time_ps: f64,
    nnodes: i32,
    dd_nc: [i32; 3],
    npme: i32,
    flags_state: i32,
    flags_eks: i32,
    flags_enh: i32,
    flags_dfh: i32,
    n_ed: i32,
    e_swap_coords: i32,
    flags_awhh: i32,
    flags_pull_history: i32,
    is_modular_simulator_checkpoint: bool,
}

struct XdrHandle {
    xdr: *mut XDRFILE,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum XdrDataType {
    Int,
    Float,
    Double,
    Int64,
    Char,
    String,
}

impl CptReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let mut xdr = XdrHandle::open(&path, "r", "CPT")?;
        let header = read_cpt_header(&mut xdr)?;
        let frame = read_cpt_frame(&mut xdr, &header)?;
        Ok(Self {
            path,
            frame,
            consumed: false,
        })
    }

    pub fn reset(&mut self) -> TrajResult<()> {
        let reopened = Self::open(self.path.clone())?;
        *self = reopened;
        Ok(())
    }

    pub fn step(&self) -> i64 {
        self.frame.step
    }

    pub fn simulation_part(&self) -> i32 {
        self.frame.simulation_part
    }

    pub fn fep_state(&self) -> Option<i32> {
        self.frame.fep_state
    }
}

impl TrajReader for CptReader {
    fn n_atoms(&self) -> usize {
        self.frame.coords.len()
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(1)
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        out.reset(self.n_atoms(), max_frames.max(1));
        if self.consumed {
            return Ok(0);
        }
        let box_ = if out.needs_box() {
            self.frame.box_
        } else {
            Box3::None
        };
        let time_ps = if out.needs_time() {
            self.frame.time_ps
        } else {
            None
        };
        let dst = out.start_frame(box_, time_ps);
        for (atom, coord) in dst.iter_mut().zip(self.frame.coords.iter()) {
            atom[0] = coord[0];
            atom[1] = coord[1];
            atom[2] = coord[2];
            atom[3] = 1.0;
        }
        let velocities = if out.needs_velocities() {
            self.frame.velocities.as_deref()
        } else {
            None
        };
        let lambda_value = if out.needs_lambda() {
            self.frame.lambda_value
        } else {
            None
        };
        out.set_frame_extras(velocities, None, lambda_value)?;
        self.consumed = true;
        Ok(1)
    }
}

impl CptWriter {
    pub fn create(path: impl Into<PathBuf>, n_atoms: usize) -> TrajResult<Self> {
        if n_atoms == 0 {
            return Err(TrajError::Invalid(
                "CPT writer requires at least one atom".into(),
            ));
        }
        let path = path.into();
        Ok(Self {
            xdr: XdrHandle::open(&path, "w", "CPT")?,
            n_atoms,
            wrote_frame: false,
        })
    }

    pub fn write_frame(
        &mut self,
        coords: &[[f32; 3]],
        box_: Box3,
        step: usize,
        time_ps: Option<f32>,
        velocities: Option<&[[f32; 3]]>,
        lambda_value: Option<f32>,
    ) -> TrajResult<()> {
        if self.wrote_frame {
            return Err(TrajError::Invalid(
                "CPT writer supports exactly one frame".into(),
            ));
        }
        if coords.len() != self.n_atoms {
            return Err(TrajError::Mismatch(format!(
                "frame atom count {} does not match writer atom count {}",
                coords.len(),
                self.n_atoms
            )));
        }
        if let Some(velocities) = velocities {
            if velocities.len() != self.n_atoms {
                return Err(TrajError::Mismatch(format!(
                    "velocity atom count {} does not match writer atom count {}",
                    velocities.len(),
                    self.n_atoms
                )));
            }
        }
        let header = build_writer_header(
            self.n_atoms,
            i64::try_from(step)
                .map_err(|_| TrajError::Invalid("CPT step does not fit in i64".into()))?,
            time_ps.unwrap_or(step as f32) as f64,
            box_,
            velocities.is_some(),
            lambda_value.is_some(),
        );
        write_cpt_header(&mut self.xdr, &header)?;
        write_cpt_state(
            &mut self.xdr,
            &header,
            coords,
            box_,
            velocities,
            lambda_value,
        )?;
        write_empty_ekin_section(&mut self.xdr, header.flags_eks)?;
        write_empty_energy_history(&mut self.xdr, header.flags_enh)?;
        write_empty_pull_history(&mut self.xdr, header.flags_pull_history)?;
        write_empty_df_history(&mut self.xdr, header.flags_dfh)?;
        write_empty_ed_state(&mut self.xdr, header.n_ed)?;
        write_empty_awh(&mut self.xdr, header.flags_awhh)?;
        write_empty_swap_state(&mut self.xdr, header.e_swap_coords)?;
        write_empty_output_files(&mut self.xdr, header.file_version)?;
        write_cpt_footer(&mut self.xdr, header.file_version)?;
        self.wrote_frame = true;
        Ok(())
    }

    pub fn flush(&mut self) -> TrajResult<()> {
        self.xdr.flush("CPT")?;
        Ok(())
    }
}

impl XdrHandle {
    fn open(path: &Path, mode: &str, format_name: &str) -> TrajResult<Self> {
        let c_path = path_to_cstring(path, format_name)?;
        let c_mode = CString::new(mode)
            .map_err(|_| TrajError::Invalid(format!("{format_name} mode contains interior NUL")))?;
        let xdr = unsafe { xdr_cabi::xdrfile_open(c_path.as_ptr(), c_mode.as_ptr()) };
        if xdr.is_null() {
            return Err(TrajError::Io(std::io::Error::other(format!(
                "failed to open {format_name} file: {}",
                path.display()
            ))));
        }
        Ok(Self { xdr })
    }

    fn flush(&mut self, format_name: &str) -> TrajResult<()> {
        let code = unsafe { xdr_seek::xdr_flush(self.xdr) };
        if code == xdr_cabi::exdrOK {
            Ok(())
        } else {
            Err(TrajError::Io(std::io::Error::other(format!(
                "failed to flush {format_name} file"
            ))))
        }
    }

    fn read_i32(&mut self, desc: &str) -> TrajResult<i32> {
        let mut value = 0i32;
        let count = unsafe { xdr_cabi::xdrfile_read_int(&mut value, 1, self.xdr) };
        if count == 1 {
            Ok(value)
        } else {
            Err(TrajError::Parse(format!(
                "failed to read checkpoint {desc} as i32"
            )))
        }
    }

    fn write_i32(&mut self, value: i32, desc: &str) -> TrajResult<()> {
        let mut value = value;
        let count = unsafe { xdr_cabi::xdrfile_write_int(&mut value, 1, self.xdr) };
        if count == 1 {
            Ok(())
        } else {
            Err(TrajError::Io(std::io::Error::other(format!(
                "failed to write checkpoint {desc} as i32"
            ))))
        }
    }

    fn read_u8(&mut self, desc: &str) -> TrajResult<u8> {
        let mut value = 0u8;
        let count = unsafe { xdr_cabi::xdrfile_read_uchar(&mut value, 1, self.xdr) };
        if count == 1 {
            Ok(value)
        } else {
            Err(TrajError::Parse(format!(
                "failed to read checkpoint {desc} as u8"
            )))
        }
    }

    fn read_char(&mut self, desc: &str) -> TrajResult<char> {
        let mut value = 0i8;
        let count = unsafe { xdr_cabi::xdrfile_read_char(&mut value, 1, self.xdr) };
        if count == 1 {
            Ok(value as u8 as char)
        } else {
            Err(TrajError::Parse(format!(
                "failed to read checkpoint {desc} as char"
            )))
        }
    }

    fn read_f32(&mut self, desc: &str) -> TrajResult<f32> {
        let mut value = 0.0f32;
        let count = unsafe { xdr_cabi::xdrfile_read_float(&mut value, 1, self.xdr) };
        if count == 1 {
            Ok(value)
        } else {
            Err(TrajError::Parse(format!(
                "failed to read checkpoint {desc} as f32"
            )))
        }
    }

    fn write_f32(&mut self, value: f32, desc: &str) -> TrajResult<()> {
        let mut value = value;
        let count = unsafe { xdr_cabi::xdrfile_write_float(&mut value, 1, self.xdr) };
        if count == 1 {
            Ok(())
        } else {
            Err(TrajError::Io(std::io::Error::other(format!(
                "failed to write checkpoint {desc} as f32"
            ))))
        }
    }

    fn read_f64(&mut self, desc: &str) -> TrajResult<f64> {
        let mut value = 0.0f64;
        let count = unsafe { xdr_cabi::xdrfile_read_double(&mut value, 1, self.xdr) };
        if count == 1 {
            Ok(value)
        } else {
            Err(TrajError::Parse(format!(
                "failed to read checkpoint {desc} as f64"
            )))
        }
    }

    fn write_f64(&mut self, value: f64, desc: &str) -> TrajResult<()> {
        let mut value = value;
        let count = unsafe { xdr_cabi::xdrfile_write_double(&mut value, 1, self.xdr) };
        if count == 1 {
            Ok(())
        } else {
            Err(TrajError::Io(std::io::Error::other(format!(
                "failed to write checkpoint {desc} as f64"
            ))))
        }
    }

    fn read_i64(&mut self, desc: &str) -> TrajResult<i64> {
        let bytes = self.read_opaque_exact(8, desc)?;
        Ok(i64::from_be_bytes(
            bytes.try_into().expect("8-byte opaque read"),
        ))
    }

    fn write_i64(&mut self, value: i64, desc: &str) -> TrajResult<()> {
        self.write_opaque_exact(&value.to_be_bytes(), desc)
    }

    fn read_header_string(&mut self, max_len: usize, desc: &str) -> TrajResult<String> {
        let mut buffer = vec![0i8; max_len.max(1)];
        let count =
            unsafe { xdr_cabi::xdrfile_read_string(buffer.as_mut_ptr(), max_len as i32, self.xdr) };
        if count <= 0 {
            return Err(TrajError::Parse(format!(
                "failed to read checkpoint {desc} string"
            )));
        }
        let bytes = buffer
            .iter()
            .take_while(|&&value| value != 0)
            .map(|&value| value as u8)
            .collect::<Vec<_>>();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn write_header_string(&mut self, value: &str, desc: &str) -> TrajResult<()> {
        let c_value = CString::new(value)
            .map_err(|_| TrajError::Invalid(format!("checkpoint {desc} contains interior NUL")))?;
        let count =
            unsafe { xdr_cabi::xdrfile_write_string(c_value.as_ptr() as *mut i8, self.xdr) };
        if count > 0 {
            Ok(())
        } else {
            Err(TrajError::Io(std::io::Error::other(format!(
                "failed to write checkpoint {desc} string"
            ))))
        }
    }

    fn read_serializer_string(&mut self, desc: &str) -> TrajResult<String> {
        let declared_len = self.read_i32(&format!("{desc} length"))?;
        if declared_len <= 0 {
            return Err(TrajError::Parse(format!(
                "invalid checkpoint {desc} string length {declared_len}"
            )));
        }
        self.read_header_string(declared_len as usize, desc)
    }

    fn read_opaque_exact(&mut self, size: usize, desc: &str) -> TrajResult<Vec<u8>> {
        let mut buffer = vec![0i8; size];
        let count =
            unsafe { xdr_cabi::xdrfile_read_opaque(buffer.as_mut_ptr(), size as i32, self.xdr) };
        if count == size as i32 {
            Ok(buffer.into_iter().map(|value| value as u8).collect())
        } else {
            Err(TrajError::Parse(format!(
                "failed to read checkpoint {desc} opaque payload"
            )))
        }
    }

    fn write_opaque_exact(&mut self, bytes: &[u8], desc: &str) -> TrajResult<()> {
        let mut buffer = bytes.iter().map(|&value| value as i8).collect::<Vec<_>>();
        let count = unsafe {
            xdr_cabi::xdrfile_write_opaque(buffer.as_mut_ptr(), buffer.len() as i32, self.xdr)
        };
        if count == buffer.len() as i32 {
            Ok(())
        } else {
            Err(TrajError::Io(std::io::Error::other(format!(
                "failed to write checkpoint {desc} opaque payload"
            ))))
        }
    }
}

impl Drop for XdrHandle {
    fn drop(&mut self) {
        if !self.xdr.is_null() {
            unsafe {
                xdr_cabi::xdrfile_close(self.xdr);
            }
            self.xdr = std::ptr::null_mut();
        }
    }
}

impl TryFrom<i32> for XdrDataType {
    type Error = TrajError;

    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Int),
            1 => Ok(Self::Float),
            2 => Ok(Self::Double),
            3 => Ok(Self::Int64),
            4 => Ok(Self::Char),
            5 => Ok(Self::String),
            other => Err(TrajError::Parse(format!(
                "unsupported checkpoint XDR data type {other}"
            ))),
        }
    }
}

fn path_to_cstring(path: &Path, format_name: &str) -> TrajResult<CString> {
    let Some(s) = path.to_str() else {
        return Err(TrajError::Invalid(format!(
            "{format_name} path is not valid UTF-8"
        )));
    };
    CString::new(s)
        .map_err(|_| TrajError::Invalid(format!("{format_name} path contains interior NUL")))
}

fn read_cpt_header(xdr: &mut XdrHandle) -> TrajResult<CptHeader> {
    let magic = xdr.read_i32("header magic")?;
    if magic != CPT_MAGIC1 {
        return Err(TrajError::Parse(format!(
            "invalid checkpoint header magic {magic}; expected {CPT_MAGIC1}"
        )));
    }
    let _version = xdr.read_header_string(CPT_STRLEN, "GROMACS version")?;
    let _btime = xdr.read_header_string(CPT_STRLEN, "GROMACS build time UNUSED")?;
    let _buser = xdr.read_header_string(CPT_STRLEN, "GROMACS build user UNUSED")?;
    let _bhost = xdr.read_header_string(CPT_STRLEN, "GROMACS build host UNUSED")?;
    let _fprog = xdr.read_header_string(CPT_STRLEN, "generating program")?;
    let _ftime = xdr.read_header_string(CPT_STRLEN, "generation time")?;
    let file_version = xdr.read_i32("checkpoint file version")?;
    if file_version < CHECKPOINT_VERSION_ADD_MAGIC_NUMBER {
        return Err(TrajError::Invalid(format!(
            "checkpoint version {file_version} is too old"
        )));
    }
    let double_prec = if file_version >= CHECKPOINT_VERSION_DOUBLE_PRECISION_BUILD {
        xdr.read_i32("GROMACS double precision")?
    } else {
        -1
    };
    if file_version >= CHECKPOINT_VERSION_HOST_INFORMATION {
        let _host = xdr.read_header_string(CPT_STRLEN, "generating host")?;
    }
    let natoms = xdr.read_i32("#atoms")?;
    if natoms <= 0 {
        return Err(TrajError::Parse(format!(
            "checkpoint atom count must be > 0, got {natoms}"
        )));
    }
    let ngtc = xdr.read_i32("#T-coupling groups")?;
    let nhchainlength = if file_version >= CHECKPOINT_VERSION_NOSE_HOOVER_THERMOSTAT {
        xdr.read_i32("#Nose-Hoover T-chains")?
    } else {
        1
    };
    let nnhpres = if file_version >= CHECKPOINT_VERSION_NOSE_HOOVER_BAROSTAT {
        xdr.read_i32("#Nose-Hoover T-chains for barostat")?
    } else {
        0
    };
    let nlambda = if file_version >= CHECKPOINT_VERSION_LAMBDA_STATE_AND_HISTORY {
        xdr.read_i32("# of total lambda states")?
    } else {
        0
    };
    let integrator = xdr.read_i32("integrator")?;
    let simulation_part = if file_version >= CHECKPOINT_VERSION_SAFE_SIMULATION_PART {
        xdr.read_i32("simulation part")?
    } else {
        1
    };
    let step = if file_version >= CHECKPOINT_VERSION_SAFE_STEPS {
        xdr.read_i64("step")?
    } else {
        i64::from(xdr.read_i32("legacy step")?)
    };
    let time_ps = xdr.read_f64("time")?;
    let nnodes = xdr.read_i32("#PP-ranks")?;
    let dd_nc = [
        xdr.read_i32("dd_nc[x]")?,
        xdr.read_i32("dd_nc[y]")?,
        xdr.read_i32("dd_nc[z]")?,
    ];
    let npme = xdr.read_i32("#PME-only ranks")?;
    let flags_state = xdr.read_i32("state flags")?;
    let (flags_eks, flags_enh) = if file_version >= CHECKPOINT_VERSION_EKIN_DATA_AND_FLAGS {
        (
            xdr.read_i32("ekin data flags")?,
            xdr.read_i32("energy history flags")?,
        )
    } else {
        (0, 0)
    };
    let flags_dfh = if file_version >= CHECKPOINT_VERSION_LAMBDA_STATE_AND_HISTORY {
        xdr.read_i32("df history flags")?
    } else {
        0
    };
    let n_ed = if file_version >= CHECKPOINT_VERSION_ESSENTIAL_DYNAMICS {
        xdr.read_i32("ED data sets")?
    } else {
        0
    };
    let e_swap_coords = if file_version >= CHECKPOINT_VERSION_SWAP_STATE {
        xdr.read_i32("swap state")?
    } else {
        0
    };
    let flags_awhh = if file_version >= CHECKPOINT_VERSION_AWH_HISTORY_FLAGS {
        xdr.read_i32("AWH history flags")?
    } else {
        0
    };
    let flags_pull_history = if file_version >= CHECKPOINT_VERSION_REMOVE_BUILD_MACHINE_INFORMATION
    {
        xdr.read_i32("pull history flags")?
    } else {
        0
    };
    let is_modular_simulator_checkpoint = if file_version >= CHECKPOINT_VERSION_MODULAR_SIMULATOR {
        xdr.read_i32("is modular simulator checkpoint")? != 0
    } else {
        false
    };

    Ok(CptHeader {
        file_version,
        double_prec,
        natoms: natoms as usize,
        ngtc,
        nhchainlength,
        nnhpres,
        nlambda,
        integrator,
        simulation_part,
        step,
        time_ps,
        nnodes,
        dd_nc,
        npme,
        flags_state,
        flags_eks,
        flags_enh,
        flags_dfh,
        n_ed,
        e_swap_coords,
        flags_awhh,
        flags_pull_history,
        is_modular_simulator_checkpoint,
    })
}

fn write_cpt_header(xdr: &mut XdrHandle, header: &CptHeader) -> TrajResult<()> {
    xdr.write_i32(CPT_MAGIC1, "header magic")?;
    xdr.write_header_string("warp-md", "GROMACS version")?;
    xdr.write_header_string("", "GROMACS build time UNUSED")?;
    xdr.write_header_string("", "GROMACS build user UNUSED")?;
    xdr.write_header_string("", "GROMACS build host UNUSED")?;
    xdr.write_header_string("warp-md", "generating program")?;
    xdr.write_header_string("", "generation time")?;
    xdr.write_i32(header.file_version, "checkpoint file version")?;
    if header.file_version >= CHECKPOINT_VERSION_DOUBLE_PRECISION_BUILD {
        xdr.write_i32(header.double_prec, "GROMACS double precision")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_HOST_INFORMATION {
        xdr.write_header_string("warp-md", "generating host")?;
    }
    xdr.write_i32(header.natoms as i32, "#atoms")?;
    xdr.write_i32(header.ngtc, "#T-coupling groups")?;
    if header.file_version >= CHECKPOINT_VERSION_NOSE_HOOVER_THERMOSTAT {
        xdr.write_i32(header.nhchainlength, "#Nose-Hoover T-chains")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_NOSE_HOOVER_BAROSTAT {
        xdr.write_i32(header.nnhpres, "#Nose-Hoover T-chains for barostat")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_LAMBDA_STATE_AND_HISTORY {
        xdr.write_i32(header.nlambda, "# of total lambda states")?;
    }
    xdr.write_i32(header.integrator, "integrator")?;
    if header.file_version >= CHECKPOINT_VERSION_SAFE_SIMULATION_PART {
        xdr.write_i32(header.simulation_part, "simulation part")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_SAFE_STEPS {
        xdr.write_i64(header.step, "step")?;
    } else {
        xdr.write_i32(header.step as i32, "legacy step")?;
    }
    xdr.write_f64(header.time_ps, "time")?;
    xdr.write_i32(header.nnodes, "#PP-ranks")?;
    xdr.write_i32(header.dd_nc[0], "dd_nc[x]")?;
    xdr.write_i32(header.dd_nc[1], "dd_nc[y]")?;
    xdr.write_i32(header.dd_nc[2], "dd_nc[z]")?;
    xdr.write_i32(header.npme, "#PME-only ranks")?;
    xdr.write_i32(header.flags_state, "state flags")?;
    if header.file_version >= CHECKPOINT_VERSION_EKIN_DATA_AND_FLAGS {
        xdr.write_i32(header.flags_eks, "ekin data flags")?;
        xdr.write_i32(header.flags_enh, "energy history flags")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_LAMBDA_STATE_AND_HISTORY {
        xdr.write_i32(header.flags_dfh, "df history flags")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_ESSENTIAL_DYNAMICS {
        xdr.write_i32(header.n_ed, "ED data sets")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_SWAP_STATE {
        xdr.write_i32(header.e_swap_coords, "swap state")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_AWH_HISTORY_FLAGS {
        xdr.write_i32(header.flags_awhh, "AWH history flags")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_REMOVE_BUILD_MACHINE_INFORMATION {
        xdr.write_i32(header.flags_pull_history, "pull history flags")?;
    }
    if header.file_version >= CHECKPOINT_VERSION_MODULAR_SIMULATOR {
        xdr.write_i32(
            i32::from(header.is_modular_simulator_checkpoint),
            "is modular simulator checkpoint",
        )?;
    }
    Ok(())
}

fn read_cpt_frame(xdr: &mut XdrHandle, header: &CptHeader) -> TrajResult<CptFrameData> {
    let mut coords = None;
    let mut velocities = None;
    let mut box_ = Box3::None;
    let mut lambda_value = None;
    let mut fep_state = None;

    for entry in 0..STATE_ENTRY_COUNT {
        if !bit_is_set(header.flags_state, entry) {
            continue;
        }
        match entry {
            STATE_ENTRY_LAMBDA => {
                let values =
                    read_checkpoint_real_vector_as_f32(xdr, Some(FEP_LAMBDA_COUNT), "FE-lambda")?;
                lambda_value = values.get(FEP_LAMBDA_INDEX).copied();
            }
            STATE_ENTRY_FEP_STATE => {
                let values = read_checkpoint_int_vector(xdr, Some(1), "fep_state")?;
                fep_state = values.first().copied();
            }
            STATE_ENTRY_BOX => {
                let values = read_checkpoint_real_vector_as_f32(xdr, Some(9), "box")?;
                box_ = matrix_to_box3_nm(&values);
            }
            STATE_ENTRY_BOX_REL
            | STATE_ENTRY_BOX_V
            | STATE_ENTRY_PRESSURE_PREVIOUS
            | STATE_ENTRY_SVIR_PREV
            | STATE_ENTRY_FVIR_PREV => {
                skip_checkpoint_vector(xdr, Some(9), &format!("state entry {entry}"))?;
            }
            STATE_ENTRY_NHXI
            | STATE_ENTRY_THERM_INT
            | STATE_ENTRY_NHVXI
            | STATE_ENTRY_NHPRESXI
            | STATE_ENTRY_NHPRESVXI
            | STATE_ENTRY_PULL_COM_PREV_STEP => {
                skip_checkpoint_vector(xdr, None, &format!("state entry {entry}"))?;
            }
            STATE_ENTRY_BAROS_INT => {
                skip_checkpoint_vector(xdr, Some(1), "barostat integral")?;
            }
            STATE_ENTRY_VETA
            | STATE_ENTRY_VOL0
            | STATE_ENTRY_DISRE_INIT_F
            | STATE_ENTRY_ORIRE_INIT_F => {
                skip_checkpoint_vector(xdr, Some(1), &format!("state entry {entry}"))?;
            }
            STATE_ENTRY_DISRE_RM3TAV | STATE_ENTRY_ORIRE_DTAV | STATE_ENTRY_CGP => {
                skip_checkpoint_vector(xdr, None, &format!("state entry {entry}"))?;
            }
            STATE_ENTRY_X => {
                let values = read_checkpoint_real_vector_as_f32(
                    xdr,
                    Some(header.natoms * 3),
                    "coordinates",
                )?;
                coords = Some(reals_to_triplets_nm(&values, "coordinates")?);
            }
            STATE_ENTRY_V => {
                let values =
                    read_checkpoint_real_vector_as_f32(xdr, Some(header.natoms * 3), "velocities")?;
                velocities = Some(reals_to_triplets_nm(&values, "velocities")?);
            }
            9 | 11 | 12 | 25 | 26 => {
                skip_checkpoint_vector(xdr, None, &format!("legacy state entry {entry}"))?;
            }
            other => {
                return Err(TrajError::Invalid(format!(
                    "unsupported checkpoint state entry {other}"
                )));
            }
        }
    }

    skip_ekin_state(xdr, header.flags_eks)?;
    skip_energy_history(xdr, header.flags_enh)?;
    skip_pull_history(xdr, header.flags_pull_history)?;
    skip_df_history(xdr, header.flags_dfh)?;
    if header.n_ed != 0 {
        skip_ed_state(xdr, header.n_ed, header.double_prec)?;
    }
    if header.flags_awhh != 0 {
        return Err(TrajError::Invalid(
            "AWH checkpoint history is not supported yet".into(),
        ));
    }
    if header.e_swap_coords != 0 {
        return Err(TrajError::Invalid(
            "swap-state checkpoints are not supported yet".into(),
        ));
    }
    skip_output_files(xdr, header.file_version)?;
    if header.file_version >= CHECKPOINT_VERSION_MD_MODULES {
        skip_kvt_object(xdr)?;
    }
    if header.file_version >= CHECKPOINT_VERSION_MODULAR_SIMULATOR {
        skip_kvt_object(xdr)?;
        if header.is_modular_simulator_checkpoint {
            return Err(TrajError::Invalid(
                "modular simulator checkpoints are not supported yet".into(),
            ));
        }
    }
    read_cpt_footer(xdr, header.file_version)?;

    let coords = coords
        .ok_or_else(|| TrajError::Parse("checkpoint did not contain coordinate state".into()))?;
    Ok(CptFrameData {
        step: header.step,
        simulation_part: header.simulation_part,
        coords,
        box_,
        time_ps: Some(header.time_ps as f32),
        velocities,
        lambda_value,
        fep_state,
    })
}

fn write_cpt_state(
    xdr: &mut XdrHandle,
    header: &CptHeader,
    coords: &[[f32; 3]],
    box_: Box3,
    velocities: Option<&[[f32; 3]]>,
    lambda_value: Option<f32>,
) -> TrajResult<()> {
    for entry in 0..STATE_ENTRY_COUNT {
        if !bit_is_set(header.flags_state, entry) {
            continue;
        }
        match entry {
            STATE_ENTRY_LAMBDA => {
                let mut values = [0.0f32; FEP_LAMBDA_COUNT];
                values[FEP_LAMBDA_INDEX] = lambda_value.unwrap_or(0.0);
                write_checkpoint_f32_vector(xdr, &values, "FE-lambda")?;
            }
            STATE_ENTRY_BOX => {
                let values = box_to_nm_matrix(box_);
                write_checkpoint_f32_vector(xdr, &values, "box")?;
            }
            STATE_ENTRY_X => {
                let values = triplets_to_real_nm(coords);
                write_checkpoint_f32_vector(xdr, &values, "coordinates")?;
            }
            STATE_ENTRY_V => {
                let values = triplets_to_real_nm(
                    velocities.expect("writer state flags and velocities must match"),
                );
                write_checkpoint_f32_vector(xdr, &values, "velocities")?;
            }
            STATE_ENTRY_FEP_STATE => {
                write_checkpoint_i32_vector(xdr, &[0], "fep_state")?;
            }
            other => {
                return Err(TrajError::Invalid(format!(
                    "unsupported writer state entry {other}"
                )));
            }
        }
    }
    Ok(())
}

fn build_writer_header(
    n_atoms: usize,
    step: i64,
    time_ps: f64,
    box_: Box3,
    has_velocities: bool,
    has_lambda: bool,
) -> CptHeader {
    let mut flags_state = bit_mask(STATE_ENTRY_X);
    if !matches!(box_, Box3::None) {
        flags_state |= bit_mask(STATE_ENTRY_BOX);
    }
    if has_velocities {
        flags_state |= bit_mask(STATE_ENTRY_V);
    }
    if has_lambda {
        flags_state |= bit_mask(STATE_ENTRY_LAMBDA) | bit_mask(STATE_ENTRY_FEP_STATE);
    }
    CptHeader {
        file_version: CPT_WRITER_FILE_VERSION,
        double_prec: 0,
        natoms: n_atoms,
        ngtc: 0,
        nhchainlength: 1,
        nnhpres: 0,
        nlambda: if has_lambda {
            FEP_LAMBDA_COUNT as i32
        } else {
            0
        },
        integrator: 0,
        simulation_part: 1,
        step,
        time_ps,
        nnodes: 1,
        dd_nc: [1, 1, 1],
        npme: 0,
        flags_state,
        flags_eks: 0,
        flags_enh: 0,
        flags_dfh: 0,
        n_ed: 0,
        e_swap_coords: 0,
        flags_awhh: 0,
        flags_pull_history: 0,
        is_modular_simulator_checkpoint: false,
    }
}

fn write_empty_ekin_section(_xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        return Ok(());
    }
    Err(TrajError::Invalid(
        "non-empty checkpoint kinetic state writing is not supported".into(),
    ))
}

fn write_empty_energy_history(_xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        return Ok(());
    }
    Err(TrajError::Invalid(
        "non-empty checkpoint energy history writing is not supported".into(),
    ))
}

fn write_empty_pull_history(_xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        Ok(())
    } else {
        Err(TrajError::Invalid(
            "pull-history checkpoint writing is not supported".into(),
        ))
    }
}

fn write_empty_df_history(_xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        Ok(())
    } else {
        Err(TrajError::Invalid(
            "df-history checkpoint writing is not supported".into(),
        ))
    }
}

fn write_empty_ed_state(_xdr: &mut XdrHandle, n_ed: i32) -> TrajResult<()> {
    if n_ed == 0 {
        Ok(())
    } else {
        Err(TrajError::Invalid(
            "ED checkpoint writing is not supported".into(),
        ))
    }
}

fn write_empty_awh(_xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        Ok(())
    } else {
        Err(TrajError::Invalid(
            "AWH checkpoint writing is not supported".into(),
        ))
    }
}

fn write_empty_swap_state(_xdr: &mut XdrHandle, swap_state: i32) -> TrajResult<()> {
    if swap_state == 0 {
        Ok(())
    } else {
        Err(TrajError::Invalid(
            "swap-state checkpoint writing is not supported".into(),
        ))
    }
}

fn write_empty_output_files(xdr: &mut XdrHandle, _file_version: i32) -> TrajResult<()> {
    xdr.write_i32(0, "number of output files")
}

fn write_cpt_footer(xdr: &mut XdrHandle, file_version: i32) -> TrajResult<()> {
    if file_version >= CHECKPOINT_VERSION_ADD_MAGIC_NUMBER {
        xdr.write_i32(CPT_MAGIC2, "footer magic")?;
    }
    Ok(())
}

fn read_cpt_footer(xdr: &mut XdrHandle, file_version: i32) -> TrajResult<()> {
    if file_version >= CHECKPOINT_VERSION_ADD_MAGIC_NUMBER {
        let magic = xdr.read_i32("footer magic")?;
        if magic != CPT_MAGIC2 {
            return Err(TrajError::Parse(format!(
                "invalid checkpoint footer magic {magic}; expected {CPT_MAGIC2}"
            )));
        }
    }
    Ok(())
}

fn skip_ekin_state(xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        return Ok(());
    }
    let mut ekin_n = 0usize;
    for entry in 0..EKIN_ENTRY_COUNT {
        if !bit_is_set(flags, entry) {
            continue;
        }
        match entry {
            EKIN_ENTRY_NUMBER => {
                let values = read_checkpoint_int_vector(xdr, Some(1), "ekin_n")?;
                ekin_n = values[0].max(0) as usize;
            }
            EKIN_ENTRY_HALF_STEP | EKIN_ENTRY_FULL_STEP | EKIN_ENTRY_HALF_STEP_OLD => {
                let nf = xdr.read_i32("ekin matrix count")?;
                if nf < 0 {
                    return Err(TrajError::Parse(
                        "negative checkpoint ekin matrix count".into(),
                    ));
                }
                let nf = nf as usize;
                if ekin_n != 0 && nf != ekin_n {
                    return Err(TrajError::Parse(format!(
                        "checkpoint ekin matrix count mismatch: expected {ekin_n}, got {nf}"
                    )));
                }
                let expected = nf * 9;
                skip_checkpoint_vector(xdr, Some(expected), "ekin matrices")?;
            }
            EKIN_ENTRY_DEKIN_D_LAMBDA | EKIN_ENTRY_MVCOS => {
                skip_checkpoint_vector(xdr, Some(1), "ekin scalar real")?;
            }
            EKIN_ENTRY_SCALE_FULL_STEP | EKIN_ENTRY_SCALE_HALF_STEP | EKIN_ENTRY_VELOCITY_SCALE => {
                skip_checkpoint_vector(xdr, None, "ekin scale vector")?;
            }
            EKIN_ENTRY_TOTAL => {
                skip_checkpoint_vector(xdr, Some(9), "ekin total matrix")?;
            }
            other => {
                return Err(TrajError::Invalid(format!(
                    "unsupported checkpoint kinetic entry {other}"
                )));
            }
        }
    }
    Ok(())
}

fn skip_energy_history(xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        return Ok(());
    }
    let mut num_delta_h = 0usize;
    for entry in 0..ENERGY_ENTRY_COUNT {
        if !bit_is_set(flags, entry) {
            continue;
        }
        match entry {
            ENERGY_ENTRY_N => {
                let _ = read_checkpoint_int_vector(xdr, Some(1), "energy history count")?;
            }
            ENERGY_ENTRY_AVER | ENERGY_ENTRY_SUM | ENERGY_ENTRY_SUM_SIM => {
                skip_checkpoint_vector(xdr, None, "energy history vector")?;
            }
            ENERGY_ENTRY_NUM_SUM
            | ENERGY_ENTRY_NUM_SUM_SIM
            | ENERGY_ENTRY_NUM_STEPS
            | ENERGY_ENTRY_NUM_STEPS_SIM => {
                let _ = xdr.read_i64("energy history step counter")?;
            }
            ENERGY_ENTRY_DELTA_H_NN => {
                let values = read_checkpoint_int_vector(xdr, Some(1), "delta_h count")?;
                num_delta_h = values[0].max(0) as usize;
            }
            ENERGY_ENTRY_DELTA_H_LIST => {
                for _ in 0..num_delta_h {
                    skip_checkpoint_vector(xdr, None, "delta_h list")?;
                }
            }
            ENERGY_ENTRY_DELTA_H_START_TIME | ENERGY_ENTRY_DELTA_H_START_LAMBDA => {
                skip_checkpoint_vector(xdr, Some(1), "delta_h scalar")?;
            }
            other => {
                return Err(TrajError::Invalid(format!(
                    "unsupported checkpoint energy entry {other}"
                )));
            }
        }
    }
    Ok(())
}

fn skip_pull_history(xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        return Ok(());
    }
    let mut num_coords = 0usize;
    let mut num_groups = 0usize;
    let mut num_values_x = 0i32;
    let mut num_values_f = 0i32;
    for entry in 0..PULL_ENTRY_COUNT {
        if !bit_is_set(flags, entry) {
            continue;
        }
        let value = xdr.read_i32("pull history scalar")?;
        match entry {
            PULL_ENTRY_NUM_COORDINATES => num_coords = value.max(0) as usize,
            PULL_ENTRY_NUM_GROUPS => num_groups = value.max(0) as usize,
            PULL_ENTRY_NUM_VALUES_IN_X_SUM => num_values_x = value,
            PULL_ENTRY_NUM_VALUES_IN_F_SUM => num_values_f = value,
            _ => {}
        }
    }
    if num_values_x > 0 || num_values_f > 0 {
        for _ in 0..num_coords {
            for _ in 0..14 {
                skip_checkpoint_vector(xdr, Some(1), "pull coordinate scalar")?;
            }
        }
        for _ in 0..num_groups {
            for _ in 0..3 {
                skip_checkpoint_vector(xdr, Some(1), "pull group scalar")?;
            }
        }
    }
    Ok(())
}

fn skip_df_history(xdr: &mut XdrHandle, flags: i32) -> TrajResult<()> {
    if flags == 0 {
        return Ok(());
    }
    for entry in 0..DF_ENTRY_COUNT {
        if !bit_is_set(flags, entry) {
            continue;
        }
        match entry {
            DF_ENTRY_IS_EQUILIBRATED => {
                let _ = read_checkpoint_int_vector(xdr, Some(1), "df history bool")?;
            }
            DF_ENTRY_NUM_AT_LAMBDA_STATS | DF_ENTRY_NUM_AT_LAMBDA_EQUIL => {
                let _ = read_checkpoint_int_vector(xdr, None, "df history int vector")?;
            }
            DF_ENTRY_WANG_LANDAU_HISTOGRAM
            | DF_ENTRY_SUM_WEIGHTS
            | DF_ENTRY_SUM_DG
            | DF_ENTRY_SUM_MIN_VAR
            | DF_ENTRY_SUM_VAR
            | DF_ENTRY_ACCUM_P
            | DF_ENTRY_ACCUM_M
            | DF_ENTRY_ACCUM_P2
            | DF_ENTRY_ACCUM_M2
            | DF_ENTRY_TIJ
            | DF_ENTRY_TIJ_EMP => {
                skip_checkpoint_vector(xdr, None, "df history real vector")?;
            }
            DF_ENTRY_WANG_LANDAU_DELTA => {
                skip_checkpoint_vector(xdr, Some(1), "df history scalar")?;
            }
            other => {
                return Err(TrajError::Invalid(format!(
                    "unsupported checkpoint df-history entry {other}"
                )));
            }
        }
    }
    Ok(())
}

fn skip_ed_state(xdr: &mut XdrHandle, n_ed: i32, double_prec: i32) -> TrajResult<()> {
    for _ in 0..n_ed.max(0) {
        let nref = xdr.read_i32("ED reference atom count")?.max(0) as usize;
        skip_native_real_triplets(xdr, nref, double_prec, "ED reference coords")?;
        let nav = xdr.read_i32("ED average atom count")?.max(0) as usize;
        skip_native_real_triplets(xdr, nav, double_prec, "ED average coords")?;
    }
    Ok(())
}

fn skip_output_files(xdr: &mut XdrHandle, file_version: i32) -> TrajResult<()> {
    let nfiles = xdr.read_i32("number of output files")?;
    if nfiles < 0 {
        return Err(TrajError::Parse(format!(
            "negative checkpoint output-file count {nfiles}"
        )));
    }
    for index in 0..nfiles {
        let _ = xdr
            .read_header_string(CPT_STRLEN, "output filename")
            .map_err(|err| {
                TrajError::Parse(format!(
                    "failed to read checkpoint output filename {index}/{nfiles}: {err}"
                ))
            })?;
        let _ = xdr.read_i32("output offset high")?;
        let _ = xdr.read_i32("output offset low")?;
        if file_version >= CHECKPOINT_VERSION_FILE_CHECKSUM_AND_SIZE {
            let _ = xdr.read_i32("output checksum size")?;
            for _ in 0..GROMACS_CHECKSUM_LEN {
                let _ = xdr.read_u8("output checksum")?;
            }
        }
    }
    Ok(())
}

fn skip_kvt_object(xdr: &mut XdrHandle) -> TrajResult<()> {
    let count = xdr.read_i32("KVT object property count")?;
    if count < 0 {
        return Err(TrajError::Parse(format!(
            "negative key-value-tree property count {count}"
        )));
    }
    for _ in 0..count {
        let _ = xdr.read_serializer_string("KVT key")?;
        skip_kvt_value(xdr)?;
    }
    Ok(())
}

fn skip_kvt_value(xdr: &mut XdrHandle) -> TrajResult<()> {
    match xdr.read_u8("KVT type tag")? {
        KVT_TAG_OBJECT => skip_kvt_object(xdr),
        KVT_TAG_ARRAY => {
            let count = xdr.read_i32("KVT array count")?;
            if count < 0 {
                return Err(TrajError::Parse(format!(
                    "negative key-value-tree array count {count}"
                )));
            }
            for _ in 0..count {
                skip_kvt_value(xdr)?;
            }
            Ok(())
        }
        KVT_TAG_STRING => {
            let _ = xdr.read_serializer_string("KVT string")?;
            Ok(())
        }
        KVT_TAG_BOOL => {
            let _ = xdr.read_i32("KVT bool")?;
            Ok(())
        }
        KVT_TAG_CHAR => {
            let _ = xdr.read_char("KVT char")?;
            Ok(())
        }
        KVT_TAG_UCHAR => {
            let _ = xdr.read_u8("KVT uchar")?;
            Ok(())
        }
        KVT_TAG_INT => {
            let _ = xdr.read_i32("KVT int")?;
            Ok(())
        }
        KVT_TAG_INT64 => {
            let _ = xdr.read_i64("KVT int64")?;
            Ok(())
        }
        KVT_TAG_FLOAT => {
            let _ = xdr.read_f32("KVT float")?;
            Ok(())
        }
        KVT_TAG_DOUBLE => {
            let _ = xdr.read_f64("KVT double")?;
            Ok(())
        }
        other => Err(TrajError::Invalid(format!(
            "unsupported key-value-tree type tag '{}'",
            other as char
        ))),
    }
}

fn read_checkpoint_vector_header(
    xdr: &mut XdrHandle,
    desc: &str,
) -> TrajResult<(usize, XdrDataType)> {
    let count = xdr.read_i32(&format!("{desc} count"))?;
    if count < 0 {
        return Err(TrajError::Parse(format!(
            "negative checkpoint vector count {count} for {desc}"
        )));
    }
    let data_type = XdrDataType::try_from(xdr.read_i32(&format!("{desc} type"))?)?;
    Ok((count as usize, data_type))
}

fn skip_checkpoint_vector(
    xdr: &mut XdrHandle,
    expected_len: Option<usize>,
    desc: &str,
) -> TrajResult<()> {
    let (count, data_type) = read_checkpoint_vector_header(xdr, desc)?;
    if let Some(expected_len) = expected_len {
        if count != expected_len {
            return Err(TrajError::Parse(format!(
                "checkpoint vector length mismatch for {desc}: expected {expected_len}, got {count}"
            )));
        }
    }
    match data_type {
        XdrDataType::Int => {
            for _ in 0..count {
                let _ = xdr.read_i32(desc)?;
            }
        }
        XdrDataType::Float => {
            for _ in 0..count {
                let _ = xdr.read_f32(desc)?;
            }
        }
        XdrDataType::Double => {
            for _ in 0..count {
                let _ = xdr.read_f64(desc)?;
            }
        }
        other => {
            return Err(TrajError::Invalid(format!(
                "unsupported checkpoint vector type {other:?} for {desc}"
            )))
        }
    }
    Ok(())
}

fn read_checkpoint_real_vector_as_f32(
    xdr: &mut XdrHandle,
    expected_len: Option<usize>,
    desc: &str,
) -> TrajResult<Vec<f32>> {
    let (count, data_type) = read_checkpoint_vector_header(xdr, desc)?;
    if let Some(expected_len) = expected_len {
        if count != expected_len {
            return Err(TrajError::Parse(format!(
                "checkpoint real-vector length mismatch for {desc}: expected {expected_len}, got {count}"
            )));
        }
    }
    match data_type {
        XdrDataType::Float => {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(xdr.read_f32(desc)?);
            }
            Ok(values)
        }
        XdrDataType::Double => {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(xdr.read_f64(desc)? as f32);
            }
            Ok(values)
        }
        other => Err(TrajError::Parse(format!(
            "checkpoint {desc} expected float/double vector, got {other:?}"
        ))),
    }
}

fn read_checkpoint_int_vector(
    xdr: &mut XdrHandle,
    expected_len: Option<usize>,
    desc: &str,
) -> TrajResult<Vec<i32>> {
    let (count, data_type) = read_checkpoint_vector_header(xdr, desc)?;
    if let Some(expected_len) = expected_len {
        if count != expected_len {
            return Err(TrajError::Parse(format!(
                "checkpoint int-vector length mismatch for {desc}: expected {expected_len}, got {count}"
            )));
        }
    }
    if data_type != XdrDataType::Int {
        return Err(TrajError::Parse(format!(
            "checkpoint {desc} expected int vector, got {data_type:?}"
        )));
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(xdr.read_i32(desc)?);
    }
    Ok(values)
}

fn write_checkpoint_f32_vector(xdr: &mut XdrHandle, values: &[f32], desc: &str) -> TrajResult<()> {
    xdr.write_i32(values.len() as i32, &format!("{desc} count"))?;
    xdr.write_i32(1, &format!("{desc} type"))?;
    for &value in values {
        xdr.write_f32(value, desc)?;
    }
    Ok(())
}

fn write_checkpoint_i32_vector(xdr: &mut XdrHandle, values: &[i32], desc: &str) -> TrajResult<()> {
    xdr.write_i32(values.len() as i32, &format!("{desc} count"))?;
    xdr.write_i32(0, &format!("{desc} type"))?;
    for &value in values {
        xdr.write_i32(value, desc)?;
    }
    Ok(())
}

fn skip_native_real_triplets(
    xdr: &mut XdrHandle,
    count: usize,
    double_prec: i32,
    desc: &str,
) -> TrajResult<()> {
    for _ in 0..(count * 3) {
        if double_prec == 1 {
            let _ = xdr.read_f64(desc)?;
        } else {
            let _ = xdr.read_f32(desc)?;
        }
    }
    Ok(())
}

fn reals_to_triplets_nm(values: &[f32], desc: &str) -> TrajResult<Vec<[f32; 3]>> {
    if values.len() % 3 != 0 {
        return Err(TrajError::Parse(format!(
            "checkpoint {desc} length {} is not divisible by 3",
            values.len()
        )));
    }
    Ok(values
        .chunks_exact(3)
        .map(|chunk| {
            [
                chunk[0] * NM_TO_ANGSTROM,
                chunk[1] * NM_TO_ANGSTROM,
                chunk[2] * NM_TO_ANGSTROM,
            ]
        })
        .collect())
}

fn triplets_to_real_nm(values: &[[f32; 3]]) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len() * 3);
    for value in values {
        out.push(value[0] * ANGSTROM_TO_NM);
        out.push(value[1] * ANGSTROM_TO_NM);
        out.push(value[2] * ANGSTROM_TO_NM);
    }
    out
}

fn matrix_to_box3_nm(values: &[f32]) -> Box3 {
    let matrix = [
        values[0] * NM_TO_ANGSTROM,
        values[1] * NM_TO_ANGSTROM,
        values[2] * NM_TO_ANGSTROM,
        values[3] * NM_TO_ANGSTROM,
        values[4] * NM_TO_ANGSTROM,
        values[5] * NM_TO_ANGSTROM,
        values[6] * NM_TO_ANGSTROM,
        values[7] * NM_TO_ANGSTROM,
        values[8] * NM_TO_ANGSTROM,
    ];
    if matrix[1].abs() < BOX_TOL
        && matrix[2].abs() < BOX_TOL
        && matrix[3].abs() < BOX_TOL
        && matrix[5].abs() < BOX_TOL
        && matrix[6].abs() < BOX_TOL
        && matrix[7].abs() < BOX_TOL
    {
        Box3::Orthorhombic {
            lx: matrix[0],
            ly: matrix[4],
            lz: matrix[8],
        }
    } else {
        Box3::Triclinic { m: matrix }
    }
}

fn box_to_nm_matrix(box_: Box3) -> [f32; 9] {
    match box_ {
        Box3::None => [0.0; 9],
        Box3::Orthorhombic { lx, ly, lz } => [
            lx * ANGSTROM_TO_NM,
            0.0,
            0.0,
            0.0,
            ly * ANGSTROM_TO_NM,
            0.0,
            0.0,
            0.0,
            lz * ANGSTROM_TO_NM,
        ],
        Box3::Triclinic { m } => [
            m[0] * ANGSTROM_TO_NM,
            m[1] * ANGSTROM_TO_NM,
            m[2] * ANGSTROM_TO_NM,
            m[3] * ANGSTROM_TO_NM,
            m[4] * ANGSTROM_TO_NM,
            m[5] * ANGSTROM_TO_NM,
            m[6] * ANGSTROM_TO_NM,
            m[7] * ANGSTROM_TO_NM,
            m[8] * ANGSTROM_TO_NM,
        ],
    }
}

fn bit_mask(entry: usize) -> i32 {
    1_i32 << entry
}

fn bit_is_set(flags: i32, entry: usize) -> bool {
    flags & bit_mask(entry) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::BufReader;
    use std::process::Command;
    use tempfile::{NamedTempFile, TempDir};
    use traj_core::pdb_gro::parse_gro_reader;

    fn read_single_frame<R: TrajReader>(
        reader: &mut R,
        include_velocities: bool,
        include_lambda: bool,
    ) -> traj_core::frame::FrameChunk {
        let mut builder = FrameChunkBuilder::new(reader.n_atoms(), 1);
        builder.set_requirements(true, true);
        builder.set_optional_requirements(include_velocities, false, include_lambda);
        let frames = reader.read_chunk(1, &mut builder).unwrap();
        assert_eq!(frames, 1);
        builder.finish().unwrap()
    }

    fn gmx_binary() -> Option<String> {
        let candidate = std::env::var("GMX_BINARY").unwrap_or_else(|_| "gmx".into());
        let ok = Command::new(&candidate)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        ok.then_some(candidate)
    }

    fn gmx_tpr_fixture() -> Option<PathBuf> {
        let candidate = std::env::var("GMX_CPT_TPR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                PathBuf::from(
                    "/home/ayodele/tmp/gromacs/src/gromacs/trajectoryanalysis/tests/trpcage.tpr",
                )
            });
        candidate.is_file().then_some(candidate)
    }

    fn run_gromacs_checkpoint_fixture() -> Option<(TempDir, PathBuf, PathBuf, PathBuf)> {
        let gmx = gmx_binary()?;
        let tpr = gmx_tpr_fixture()?;
        let dir = TempDir::new().ok()?;
        let cpt = dir.path().join("state.cpt");
        let trr = dir.path().join("run.trr");
        let gro = dir.path().join("run.gro");
        let status = Command::new(gmx)
            .args([
                "mdrun",
                "-s",
                tpr.to_string_lossy().as_ref(),
                "-nsteps",
                "1",
                "-cpt",
                "0.0001",
                "-cpo",
                cpt.to_string_lossy().as_ref(),
                "-deffnm",
                dir.path().join("run").to_string_lossy().as_ref(),
                "-ntmpi",
                "1",
                "-ntomp",
                "1",
                "-pin",
                "off",
            ])
            .status()
            .ok()?;
        if !status.success() || !cpt.is_file() || !trr.is_file() || !gro.is_file() {
            return None;
        }
        Some((dir, cpt, trr, gro))
    }

    #[test]
    fn cpt_writer_roundtrips_single_frame() {
        let tempfile = tempfile::Builder::new().suffix(".cpt").tempfile().unwrap();
        let path = tempfile.path();

        let mut writer = CptWriter::create(path, 2).unwrap();
        writer
            .write_frame(
                &[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]],
                Box3::Orthorhombic {
                    lx: 15.0,
                    ly: 25.0,
                    lz: 35.0,
                },
                7,
                Some(2.5),
                Some(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                Some(0.25),
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);

        let mut reader = CptReader::open(path).unwrap();
        assert_eq!(reader.step(), 7);
        assert_eq!(reader.fep_state(), Some(0));
        assert_eq!(reader.n_frames_hint(), Some(1));
        let chunk = read_single_frame(&mut reader, true, true);
        assert_eq!(chunk.n_atoms, 2);
        assert_eq!(chunk.n_frames, 1);
        assert_eq!(chunk.coords[0], [10.0, 20.0, 30.0, 1.0]);
        assert_eq!(chunk.coords[1], [40.0, 50.0, 60.0, 1.0]);
        assert_eq!(
            chunk.box_[0],
            Box3::Orthorhombic {
                lx: 15.0,
                ly: 25.0,
                lz: 35.0,
            }
        );
        assert_eq!(chunk.time_ps.as_ref().unwrap()[0], 2.5);
        assert_eq!(chunk.lambda_values.as_ref().unwrap()[0], 0.25);
        assert_eq!(chunk.velocities.as_ref().unwrap()[0], [1.0, 2.0, 3.0]);
        assert_eq!(chunk.velocities.as_ref().unwrap()[1], [4.0, 5.0, 6.0]);
        assert_eq!(
            reader
                .read_chunk(1, &mut FrameChunkBuilder::new(2, 1))
                .unwrap(),
            0
        );
        reader.reset().unwrap();
        let chunk = read_single_frame(&mut reader, true, true);
        assert_eq!(chunk.coords[0], [10.0, 20.0, 30.0, 1.0]);
    }

    #[test]
    fn cpt_writer_rejects_second_frame() {
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();
        let mut writer = CptWriter::create(path, 1).unwrap();
        writer
            .write_frame(&[[1.0, 2.0, 3.0]], Box3::None, 0, Some(0.0), None, None)
            .unwrap();
        let err = writer
            .write_frame(&[[4.0, 5.0, 6.0]], Box3::None, 1, Some(1.0), None, None)
            .unwrap_err();
        assert!(err.to_string().contains("exactly one frame"));
    }

    #[test]
    fn cpt_reader_matches_gromacs_trr_when_available() {
        let Some((_dir, cpt_path, _trr_path, gro_path)) = run_gromacs_checkpoint_fixture() else {
            return;
        };
        let mut cpt = CptReader::open(&cpt_path).unwrap();
        let gro = parse_gro_reader(BufReader::new(File::open(&gro_path).unwrap()), true).unwrap();

        let cpt_chunk = read_single_frame(&mut cpt, true, true);

        assert_eq!(cpt_chunk.n_atoms, gro.atoms.len());
        assert_eq!(cpt_chunk.n_frames, 1);
        assert_eq!(cpt.step(), 1);
        assert!((cpt_chunk.time_ps.as_ref().unwrap()[0] - 0.002).abs() < 1.0e-6);
        if let Some(box_vectors) = gro.box_vectors {
            match cpt_chunk.box_[0] {
                Box3::Orthorhombic { lx, ly, lz } => {
                    assert!((lx - box_vectors[0][0] * NM_TO_ANGSTROM).abs() < 1.0e-3);
                    assert!((ly - box_vectors[1][1] * NM_TO_ANGSTROM).abs() < 1.0e-3);
                    assert!((lz - box_vectors[2][2] * NM_TO_ANGSTROM).abs() < 1.0e-3);
                }
                other => panic!("unexpected checkpoint box shape: {other:?}"),
            }
        }
        for atom in &cpt_chunk.coords {
            assert!(atom[0].is_finite());
            assert!(atom[1].is_finite());
            assert!(atom[2].is_finite());
        }
        assert_eq!(
            cpt_chunk.velocities.as_ref().unwrap().len(),
            cpt_chunk.n_atoms
        );
    }

    #[test]
    fn gmx_dump_reads_writer_output_when_available() {
        let Some(gmx) = gmx_binary() else {
            return;
        };
        let tempfile = tempfile::Builder::new().suffix(".cpt").tempfile().unwrap();
        let path = tempfile.path();

        let mut writer = CptWriter::create(path, 1).unwrap();
        writer
            .write_frame(
                &[[12.0, 24.0, 36.0]],
                Box3::Orthorhombic {
                    lx: 40.0,
                    ly: 50.0,
                    lz: 60.0,
                },
                9,
                Some(3.5),
                Some(&[[1.5, 2.5, 3.5]]),
                Some(0.75),
            )
            .unwrap();
        writer.flush().unwrap();
        drop(writer);

        let output = Command::new(gmx)
            .args(["dump", "-cp", path.to_string_lossy().as_ref()])
            .output()
            .unwrap();
        if !output.status.success() {
            eprintln!("stdout:\n{}", String::from_utf8_lossy(&output.stdout));
            eprintln!("stderr:\n{}", String::from_utf8_lossy(&output.stderr));
        }
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("checkpoint file version = 20"));
        assert!(stdout.contains("#atoms = 1"));
        assert!(stdout.contains("step = 9"));
        assert!(stdout.contains("t = 3.500000"));
        assert!(stdout.contains("FE-lambda (7):"));
        assert!(stdout.contains("x[0]= 1.20000e+00"));
        assert!(stdout.contains("v[0]= 1.50000e-01"));
    }
}
