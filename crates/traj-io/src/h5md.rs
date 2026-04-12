use std::path::PathBuf;
use std::str::FromStr;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Dataset, File, Group};
use ndarray::{s, Array1, Array2, Array3, Ix1, Ix2, Ix3};
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunkBuilder};

use crate::TrajReader;

const ANGSTROM_TO_NM: f32 = 0.1;
const NM_TO_ANGSTROM: f32 = 10.0;
const POSITION_PATH: &str = "/particles/system/position/value";
const POSITION_TIME_PATH: &str = "/particles/system/position/time";
const VELOCITY_PATH: &str = "/particles/system/velocity/value";
const FORCE_PATH: &str = "/particles/system/force/value";
const BOX_PATH: &str = "/particles/system/box/edges/value";
const BOX_TIME_PATH: &str = "/particles/system/box/edges/time";
const BOX_STEP_PATH: &str = "/particles/system/box/edges/step";

pub struct H5mdReader {
    _path: PathBuf,
    _file: File,
    n_atoms: usize,
    n_frames: usize,
    frame_index: usize,
    positions: ParticleFrameDataset,
    times: Option<ScalarFrameDataset>,
    boxes: Option<BoxFrameDataset>,
    velocities: Option<ParticleFrameDataset>,
    forces: Option<ParticleFrameDataset>,
}

pub struct H5mdWriter {
    file: File,
    n_atoms: usize,
    frame_index: usize,
    steps: Dataset,
    times: Dataset,
    positions: Dataset,
    boxes: Dataset,
    velocity_steps: Dataset,
    velocity_times: Dataset,
    velocities: Dataset,
    force_steps: Dataset,
    force_times: Dataset,
    forces: Dataset,
}

struct ScalarFrameDataset {
    values: Dataset,
    scale: f32,
    n_frames: usize,
}

struct ParticleFrameDataset {
    values: Dataset,
    scale: f32,
    n_frames: usize,
}

struct BoxFrameDataset {
    values: Dataset,
    scale: f32,
    layout: BoxFrameLayout,
}

#[derive(Clone, Copy)]
enum BoxFrameLayout {
    StaticVector,
    StaticMatrix,
    DynamicVector { n_frames: usize },
    DynamicMatrix { n_frames: usize },
}

impl H5mdReader {
    pub fn open(path: impl Into<PathBuf>) -> TrajResult<Self> {
        let path = path.into();
        let file = File::open(&path).map_err(map_h5_err)?;
        let positions = open_particle_dataset(&file, POSITION_PATH, "position", NM_TO_ANGSTROM)?;
        let times = open_scalar_dataset(
            &file,
            POSITION_TIME_PATH,
            "position/time",
            1.0,
            parse_time_scale_to_ps,
        )?;
        let boxes = open_box_dataset(&file)?;
        let velocities = open_optional_particle_dataset(
            &file,
            VELOCITY_PATH,
            "velocity",
            positions.n_atoms(),
            Some(positions.n_frames),
            NM_TO_ANGSTROM,
            parse_velocity_scale_to_angstrom_per_ps,
        )?;
        let forces = open_optional_particle_dataset(
            &file,
            FORCE_PATH,
            "force",
            positions.n_atoms(),
            Some(positions.n_frames),
            1.0,
            parse_force_scale_to_native,
        )?;

        Ok(Self {
            _path: path,
            _file: file,
            n_atoms: positions.n_atoms(),
            n_frames: positions.n_frames,
            frame_index: 0,
            positions,
            times,
            boxes,
            velocities,
            forces,
        })
    }

    pub fn reset(&mut self) -> TrajResult<()> {
        self.frame_index = 0;
        Ok(())
    }
}

impl TrajReader for H5mdReader {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.n_frames)
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        out.reset(self.n_atoms, max_frames);
        if self.frame_index >= self.n_frames {
            return Ok(0);
        }

        let start = self.frame_index;
        let frames = (self.n_frames - start).min(max_frames.max(1));
        let end = start + frames;

        let position_frames = self.positions.read_frames(start, end)?;
        let time_values = if out.needs_time() {
            self.times
                .as_ref()
                .map(|dataset| dataset.read_frames(start, end))
                .transpose()?
        } else {
            None
        };
        let box_values = if out.needs_box() {
            self.boxes
                .as_ref()
                .map(|dataset| dataset.read_frames(start, end))
                .transpose()?
        } else {
            None
        };
        let velocity_values = if out.needs_velocities() {
            self.velocities
                .as_ref()
                .map(|dataset| dataset.read_frames(start, end))
                .transpose()?
        } else {
            None
        };
        let force_values = if out.needs_forces() {
            self.forces
                .as_ref()
                .map(|dataset| dataset.read_frames(start, end))
                .transpose()?
        } else {
            None
        };

        for frame in 0..frames {
            let box_ = box_values
                .as_ref()
                .and_then(|values| values.get(frame).copied())
                .unwrap_or(Box3::None);
            let time_ps = time_values
                .as_ref()
                .and_then(|values| values.get(frame).copied());
            let coords = out.start_frame(box_, time_ps);
            let src = &position_frames[frame * self.n_atoms..(frame + 1) * self.n_atoms];
            for (dst, atom) in coords.iter_mut().zip(src.iter()) {
                dst[0] = atom[0];
                dst[1] = atom[1];
                dst[2] = atom[2];
                dst[3] = 1.0;
            }

            let velocities = velocity_values.as_ref().and_then(|values| {
                if values.len() >= (frame + 1) * self.n_atoms {
                    Some(&values[frame * self.n_atoms..(frame + 1) * self.n_atoms])
                } else {
                    None
                }
            });
            let forces = force_values.as_ref().and_then(|values| {
                if values.len() >= (frame + 1) * self.n_atoms {
                    Some(&values[frame * self.n_atoms..(frame + 1) * self.n_atoms])
                } else {
                    None
                }
            });
            out.set_frame_extras(velocities, forces, None)?;
        }

        self.frame_index = end;
        Ok(frames)
    }

    fn skip_frames(&mut self, n_frames: usize) -> TrajResult<usize> {
        let skipped = n_frames.min(self.n_frames.saturating_sub(self.frame_index));
        self.frame_index += skipped;
        Ok(skipped)
    }
}

impl H5mdWriter {
    pub fn create(path: impl Into<PathBuf>, n_atoms: usize) -> TrajResult<Self> {
        if n_atoms == 0 {
            return Err(TrajError::Invalid(
                "H5MD writer requires at least one atom".into(),
            ));
        }

        let path = path.into();
        let file = File::create(&path).map_err(map_h5_err)?;
        setup_h5md_metadata(&file)?;

        let position_group = file
            .create_group("/particles/system/position")
            .map_err(map_h5_err)?;
        let velocity_group = file
            .create_group("/particles/system/velocity")
            .map_err(map_h5_err)?;
        let force_group = file
            .create_group("/particles/system/force")
            .map_err(map_h5_err)?;
        let box_group = file
            .create_group("/particles/system/box")
            .map_err(map_h5_err)?;
        let edges_group = box_group.create_group("edges").map_err(map_h5_err)?;

        let steps = create_scalar_i64_dataset(&edges_group, "step")?;
        let times = create_scalar_f64_dataset(&edges_group, "time", Some("ps"))?;
        let positions = create_particle_dataset(&position_group, "value", n_atoms, "nm")?;
        let boxes = create_box_dataset(&edges_group, "value", "nm")?;

        file.link_hard(BOX_STEP_PATH, "/particles/system/position/step")
            .map_err(map_h5_err)?;
        file.link_hard(BOX_TIME_PATH, "/particles/system/position/time")
            .map_err(map_h5_err)?;

        let velocity_steps = create_scalar_i64_dataset(&velocity_group, "step")?;
        let velocity_times = create_scalar_f64_dataset(&velocity_group, "time", Some("ps"))?;
        let velocities = create_particle_dataset(&velocity_group, "value", n_atoms, "nm ps-1")?;

        let force_steps = create_scalar_i64_dataset(&force_group, "step")?;
        let force_times = create_scalar_f64_dataset(&force_group, "time", Some("ps"))?;
        let forces = create_particle_dataset(&force_group, "value", n_atoms, "kJ mol-1 nm-1")?;

        setup_box_group_attributes(&box_group)?;

        Ok(Self {
            file,
            n_atoms,
            frame_index: 0,
            steps,
            times,
            positions,
            boxes,
            velocity_steps,
            velocity_times,
            velocities,
            force_steps,
            force_times,
            forces,
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
        validate_atom_frame_len(coords.len(), self.n_atoms, "coordinate")?;
        if let Some(values) = velocities {
            validate_atom_frame_len(values.len(), self.n_atoms, "velocity")?;
        }
        if let Some(values) = forces {
            validate_atom_frame_len(values.len(), self.n_atoms, "force")?;
        }

        let frame = self.frame_index;
        let step_i64 = i64::try_from(step)
            .map_err(|_| TrajError::Invalid("H5MD step does not fit in i64".into()))?;
        let time_value = f64::from(time_ps.unwrap_or(step as f32));

        append_scalar_i64(&self.steps, frame, step_i64)?;
        append_scalar_f64(&self.times, frame, time_value)?;
        append_particle_values(&self.positions, frame, coords, ANGSTROM_TO_NM)?;

        if let Some(box_values) = box_to_h5md_values(box_) {
            append_box_values(&self.boxes, frame, &box_values, ANGSTROM_TO_NM)?;
        }
        if let Some(values) = velocities {
            append_scalar_i64(&self.velocity_steps, frame, step_i64)?;
            append_scalar_f64(&self.velocity_times, frame, time_value)?;
            append_particle_values(&self.velocities, frame, values, ANGSTROM_TO_NM)?;
        }
        if let Some(values) = forces {
            append_scalar_i64(&self.force_steps, frame, step_i64)?;
            append_scalar_f64(&self.force_times, frame, time_value)?;
            append_particle_values(&self.forces, frame, values, 1.0)?;
        }

        self.frame_index += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> TrajResult<()> {
        self.file.flush().map_err(map_h5_err)
    }
}

impl ScalarFrameDataset {
    fn read_frames(&self, start: usize, end: usize) -> TrajResult<Vec<f32>> {
        if start >= self.n_frames {
            return Ok(Vec::new());
        }
        let end = end.min(self.n_frames);
        if start >= end {
            return Ok(Vec::new());
        }
        let values = self
            .values
            .read_slice::<f64, _, Ix1>(s![start..end])
            .map_err(map_h5_err)?;
        Ok(values
            .iter()
            .map(|value| *value as f32 * self.scale)
            .collect())
    }
}

impl ParticleFrameDataset {
    fn n_atoms(&self) -> usize {
        self.values.shape()[1]
    }

    fn read_frames(&self, start: usize, end: usize) -> TrajResult<Vec<[f32; 3]>> {
        if start >= self.n_frames {
            return Ok(Vec::new());
        }
        let end = end.min(self.n_frames);
        if start >= end {
            return Ok(Vec::new());
        }
        let frames = self
            .values
            .read_slice::<f32, _, Ix3>(s![start..end, .., ..])
            .map_err(map_h5_err)?;
        let mut out = Vec::with_capacity((end - start) * self.n_atoms());
        for frame in frames.outer_iter() {
            for atom in frame.outer_iter() {
                out.push([
                    atom[0] * self.scale,
                    atom[1] * self.scale,
                    atom[2] * self.scale,
                ]);
            }
        }
        Ok(out)
    }
}

impl BoxFrameDataset {
    fn read_frames(&self, start: usize, end: usize) -> TrajResult<Vec<Box3>> {
        match self.layout {
            BoxFrameLayout::StaticVector => {
                let values = self.values.read::<f32, Ix1>().map_err(map_h5_err)?;
                let box_ = box_from_vector(&values, self.scale)?;
                Ok(vec![box_; end.saturating_sub(start)])
            }
            BoxFrameLayout::StaticMatrix => {
                let values = self.values.read::<f32, Ix2>().map_err(map_h5_err)?;
                let box_ = box_from_matrix(&values, self.scale)?;
                Ok(vec![box_; end.saturating_sub(start)])
            }
            BoxFrameLayout::DynamicVector { n_frames } => {
                if start >= n_frames {
                    return Ok(Vec::new());
                }
                let end = end.min(n_frames);
                let values = self
                    .values
                    .read_slice::<f32, _, Ix2>(s![start..end, ..])
                    .map_err(map_h5_err)?;
                values
                    .outer_iter()
                    .map(|frame| box_from_vector(&frame.to_owned(), self.scale))
                    .collect()
            }
            BoxFrameLayout::DynamicMatrix { n_frames } => {
                if start >= n_frames {
                    return Ok(Vec::new());
                }
                let end = end.min(n_frames);
                let values = self
                    .values
                    .read_slice::<f32, _, Ix3>(s![start..end, .., ..])
                    .map_err(map_h5_err)?;
                values
                    .outer_iter()
                    .map(|frame| box_from_matrix(&frame.to_owned(), self.scale))
                    .collect()
            }
        }
    }
}

fn open_particle_dataset(
    file: &File,
    path: &str,
    label: &str,
    default_scale: f32,
) -> TrajResult<ParticleFrameDataset> {
    open_optional_particle_dataset(
        file,
        path,
        label,
        0,
        None,
        default_scale,
        parse_length_scale_to_angstrom,
    )?
    .ok_or_else(|| TrajError::Parse(format!("missing H5MD {label} dataset: {path}")))
}

fn open_optional_particle_dataset(
    file: &File,
    path: &str,
    label: &str,
    expected_atoms: usize,
    expected_frames: Option<usize>,
    default_scale: f32,
    parse_scale: fn(&str) -> Option<f32>,
) -> TrajResult<Option<ParticleFrameDataset>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let values = file.dataset(path).map_err(map_h5_err)?;
    let shape = values.shape();
    if shape.len() != 3 || shape[2] != 3 {
        return Err(TrajError::Parse(format!(
            "unexpected H5MD {label} dataset shape: {shape:?}"
        )));
    }
    if expected_atoms != 0 && shape[1] != expected_atoms {
        return Err(TrajError::Mismatch(format!(
            "H5MD {label} atom count {} does not match expected {}",
            shape[1], expected_atoms
        )));
    }
    if let Some(expected_frames) = expected_frames {
        if shape[0] != 0 && shape[0] > expected_frames {
            return Err(TrajError::Mismatch(format!(
                "H5MD {label} frame count {} exceeds position frames {}",
                shape[0], expected_frames
            )));
        }
    }
    if shape[0] == 0 {
        return Ok(None);
    }
    let scale = read_dataset_scale(&values, default_scale, parse_scale)?;
    Ok(Some(ParticleFrameDataset {
        values,
        scale,
        n_frames: shape[0],
    }))
}

fn open_scalar_dataset(
    file: &File,
    path: &str,
    label: &str,
    default_scale: f32,
    parse_scale: fn(&str) -> Option<f32>,
) -> TrajResult<Option<ScalarFrameDataset>> {
    if !file.link_exists(path) {
        return Ok(None);
    }
    let values = file.dataset(path).map_err(map_h5_err)?;
    let shape = values.shape();
    if shape.len() != 1 {
        return Err(TrajError::Parse(format!(
            "unexpected H5MD {label} dataset shape: {shape:?}"
        )));
    }
    if shape[0] == 0 {
        return Ok(None);
    }
    let scale = read_dataset_scale(&values, default_scale, parse_scale)?;
    Ok(Some(ScalarFrameDataset {
        values,
        scale,
        n_frames: shape[0],
    }))
}

fn open_box_dataset(file: &File) -> TrajResult<Option<BoxFrameDataset>> {
    if !file.link_exists(BOX_PATH) {
        return Ok(None);
    }
    let values = file.dataset(BOX_PATH).map_err(map_h5_err)?;
    let shape = values.shape();
    let scale = read_dataset_scale(&values, NM_TO_ANGSTROM, parse_length_scale_to_angstrom)?;
    let layout = match shape.as_slice() {
        [3] => BoxFrameLayout::StaticVector,
        [3, 3] => BoxFrameLayout::StaticMatrix,
        [0, 3] | [0, 3, 3] => return Ok(None),
        [frames, 3] => BoxFrameLayout::DynamicVector { n_frames: *frames },
        [frames, 3, 3] => BoxFrameLayout::DynamicMatrix { n_frames: *frames },
        _ => {
            return Err(TrajError::Parse(format!(
                "unexpected H5MD box dataset shape: {shape:?}"
            )))
        }
    };
    Ok(Some(BoxFrameDataset {
        values,
        scale,
        layout,
    }))
}

fn read_dataset_scale(
    dataset: &Dataset,
    default_scale: f32,
    parse_scale: fn(&str) -> Option<f32>,
) -> TrajResult<f32> {
    let Some(unit) = read_dataset_unit(dataset)? else {
        return Ok(default_scale);
    };
    parse_scale(&unit).ok_or_else(|| {
        TrajError::Unsupported(format!(
            "unsupported H5MD unit '{unit}' for dataset {}",
            dataset.name()
        ))
    })
}

fn read_dataset_unit(dataset: &Dataset) -> TrajResult<Option<String>> {
    let Ok(attr) = dataset.attr("unit") else {
        return Ok(None);
    };
    if let Ok(value) = attr.read_scalar::<VarLenUnicode>() {
        return Ok(Some(value.to_string()));
    }
    if let Ok(bytes) = attr.read_raw::<u8>() {
        let trimmed = bytes
            .into_iter()
            .take_while(|byte| *byte != 0)
            .collect::<Vec<_>>();
        if let Ok(value) = String::from_utf8(trimmed) {
            return Ok(Some(value));
        }
    }
    Err(TrajError::Parse(format!(
        "unsupported H5MD unit attribute representation on {}",
        dataset.name()
    )))
}

fn parse_length_scale_to_angstrom(unit: &str) -> Option<f32> {
    match normalize_unit(unit).as_str() {
        "nm" | "nanometer" | "nanometers" => Some(NM_TO_ANGSTROM),
        "a" | "angstrom" | "angstroms" => Some(1.0),
        "pm" => Some(0.01),
        _ => None,
    }
}

fn parse_velocity_scale_to_angstrom_per_ps(unit: &str) -> Option<f32> {
    match normalize_unit(unit).as_str() {
        "nm ps-1" | "nm/ps" => Some(NM_TO_ANGSTROM),
        "a ps-1" | "angstrom ps-1" | "angstroms ps-1" | "a/ps" | "angstrom/ps" => Some(1.0),
        _ => None,
    }
}

fn parse_force_scale_to_native(unit: &str) -> Option<f32> {
    match normalize_unit(unit).as_str() {
        "kj mol-1 nm-1" => Some(1.0),
        "kj mol-1 a-1" | "kj mol-1 angstrom-1" => Some(10.0),
        _ => None,
    }
}

fn parse_time_scale_to_ps(unit: &str) -> Option<f32> {
    match normalize_unit(unit).as_str() {
        "ps" => Some(1.0),
        "fs" => Some(0.001),
        "ns" => Some(1000.0),
        _ => None,
    }
}

fn normalize_unit(unit: &str) -> String {
    unit.to_ascii_lowercase()
        .replace('Å', "a")
        .replace("ångström", "angstrom")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn box_from_vector(values: &Array1<f32>, scale: f32) -> TrajResult<Box3> {
    if values.len() != 3 {
        return Err(TrajError::Parse(format!(
            "unexpected H5MD box vector length: {}",
            values.len()
        )));
    }
    Ok(Box3::Orthorhombic {
        lx: values[0] * scale,
        ly: values[1] * scale,
        lz: values[2] * scale,
    })
}

fn box_from_matrix(values: &Array2<f32>, scale: f32) -> TrajResult<Box3> {
    if values.shape() != [3, 3] {
        return Err(TrajError::Parse(format!(
            "unexpected H5MD box matrix shape: {:?}",
            values.shape()
        )));
    }
    let flat = [
        values[[0, 0]] * scale,
        values[[0, 1]] * scale,
        values[[0, 2]] * scale,
        values[[1, 0]] * scale,
        values[[1, 1]] * scale,
        values[[1, 2]] * scale,
        values[[2, 0]] * scale,
        values[[2, 1]] * scale,
        values[[2, 2]] * scale,
    ];
    let tol = 1e-5;
    let is_orth = flat[1].abs() < tol
        && flat[2].abs() < tol
        && flat[3].abs() < tol
        && flat[5].abs() < tol
        && flat[6].abs() < tol
        && flat[7].abs() < tol;
    Ok(if is_orth {
        Box3::Orthorhombic {
            lx: flat[0],
            ly: flat[4],
            lz: flat[8],
        }
    } else {
        Box3::Triclinic { m: flat }
    })
}

fn box_to_h5md_values(box_: Box3) -> Option<[f32; 9]> {
    match box_ {
        Box3::None => None,
        Box3::Orthorhombic { lx, ly, lz } => Some([lx, 0.0, 0.0, 0.0, ly, 0.0, 0.0, 0.0, lz]),
        Box3::Triclinic { m } => Some(m),
    }
}

fn setup_h5md_metadata(file: &File) -> TrajResult<()> {
    let h5md = file.create_group("/h5md").map_err(map_h5_err)?;
    let author = h5md.create_group("author").map_err(map_h5_err)?;
    let creator = h5md.create_group("creator").map_err(map_h5_err)?;
    let modules = h5md.create_group("modules").map_err(map_h5_err)?;
    let units = modules.create_group("units").map_err(map_h5_err)?;

    h5md.new_attr_builder()
        .with_data(&[1i32, 1i32])
        .create("version")
        .map_err(map_h5_err)?;
    set_group_string_attr(&author, "name", "warp-md")?;
    set_group_string_attr(&creator, "name", "warp-md")?;
    set_group_string_attr(&creator, "version", env!("CARGO_PKG_VERSION"))?;
    units
        .new_attr_builder()
        .with_data(&[1i32, 0i32])
        .create("version")
        .map_err(map_h5_err)?;
    set_group_string_attr(&units, "system", "SI")?;
    Ok(())
}

fn setup_box_group_attributes(box_group: &hdf5_metno::Group) -> TrajResult<()> {
    let periodic = h5_str("periodic")?;
    let boundary = [periodic.clone(), periodic.clone(), periodic];
    box_group
        .new_attr_builder()
        .with_data(&boundary)
        .create("boundary")
        .map_err(map_h5_err)?;
    box_group
        .new_attr::<i32>()
        .create("dimension")
        .and_then(|attr| attr.write_scalar(&3i32).map(|_| attr))
        .map_err(map_h5_err)?;
    Ok(())
}

fn create_scalar_i64_dataset(group: &hdf5_metno::Group, name: &str) -> TrajResult<Dataset> {
    group
        .new_dataset::<i64>()
        .shape((0..,))
        .chunk((1,))
        .create(name)
        .map_err(map_h5_err)
}

fn create_scalar_f64_dataset(
    group: &hdf5_metno::Group,
    name: &str,
    unit: Option<&str>,
) -> TrajResult<Dataset> {
    let dataset = group
        .new_dataset::<f64>()
        .shape((0..,))
        .chunk((1,))
        .create(name)
        .map_err(map_h5_err)?;
    if let Some(unit) = unit {
        set_dataset_string_attr(&dataset, "unit", unit)?;
    }
    Ok(dataset)
}

fn create_particle_dataset(
    group: &hdf5_metno::Group,
    name: &str,
    n_atoms: usize,
    unit: &str,
) -> TrajResult<Dataset> {
    let dataset = group
        .new_dataset::<f32>()
        .shape((0.., n_atoms, 3))
        .chunk((1, n_atoms, 3))
        .create(name)
        .map_err(map_h5_err)?;
    set_dataset_string_attr(&dataset, "unit", unit)?;
    Ok(dataset)
}

fn create_box_dataset(group: &hdf5_metno::Group, name: &str, unit: &str) -> TrajResult<Dataset> {
    let dataset = group
        .new_dataset::<f32>()
        .shape((0.., 3, 3))
        .chunk((1, 3, 3))
        .create(name)
        .map_err(map_h5_err)?;
    set_dataset_string_attr(&dataset, "unit", unit)?;
    Ok(dataset)
}

fn append_scalar_i64(dataset: &Dataset, frame: usize, value: i64) -> TrajResult<()> {
    dataset.resize((frame + 1,)).map_err(map_h5_err)?;
    let values = Array1::from_vec(vec![value]);
    dataset
        .write_slice(values.view(), s![frame..frame + 1])
        .map_err(map_h5_err)
}

fn append_scalar_f64(dataset: &Dataset, frame: usize, value: f64) -> TrajResult<()> {
    dataset.resize((frame + 1,)).map_err(map_h5_err)?;
    let values = Array1::from_vec(vec![value]);
    dataset
        .write_slice(values.view(), s![frame..frame + 1])
        .map_err(map_h5_err)
}

fn append_particle_values(
    dataset: &Dataset,
    frame: usize,
    values: &[[f32; 3]],
    scale: f32,
) -> TrajResult<()> {
    dataset
        .resize((frame + 1, values.len(), 3))
        .map_err(map_h5_err)?;
    let mut flat = Vec::with_capacity(values.len() * 3);
    for value in values {
        flat.push(value[0] * scale);
        flat.push(value[1] * scale);
        flat.push(value[2] * scale);
    }
    let frame_values = Array3::from_shape_vec((1, values.len(), 3), flat).map_err(map_shape_err)?;
    dataset
        .write_slice(frame_values.view(), s![frame..frame + 1, .., ..])
        .map_err(map_h5_err)
}

fn append_box_values(
    dataset: &Dataset,
    frame: usize,
    values: &[f32; 9],
    scale: f32,
) -> TrajResult<()> {
    dataset.resize((frame + 1, 3, 3)).map_err(map_h5_err)?;
    let frame_values = Array3::from_shape_vec(
        (1, 3, 3),
        values.iter().map(|value| *value * scale).collect(),
    )
    .map_err(map_shape_err)?;
    dataset
        .write_slice(frame_values.view(), s![frame..frame + 1, .., ..])
        .map_err(map_h5_err)
}

fn validate_atom_frame_len(actual: usize, expected: usize, label: &str) -> TrajResult<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(TrajError::Mismatch(format!(
            "{label} atom count {actual} does not match writer atom count {expected}"
        )))
    }
}

fn h5_str(value: &str) -> TrajResult<VarLenUnicode> {
    VarLenUnicode::from_str(value).map_err(map_h5_err)
}

fn set_group_string_attr(group: &Group, name: &str, value: &str) -> TrajResult<()> {
    let attr = group
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)
        .map_err(map_h5_err)?;
    let value = h5_str(value)?;
    attr.write_scalar(&value).map_err(map_h5_err)
}

fn set_dataset_string_attr(dataset: &Dataset, name: &str, value: &str) -> TrajResult<()> {
    let attr = dataset
        .new_attr::<VarLenUnicode>()
        .shape(())
        .create(name)
        .map_err(map_h5_err)?;
    let value = h5_str(value)?;
    attr.write_scalar(&value).map_err(map_h5_err)
}

fn map_h5_err(err: impl std::fmt::Display) -> TrajError {
    TrajError::Parse(format!("H5MD error: {err}"))
}

fn map_shape_err(err: impl std::fmt::Display) -> TrajError {
    TrajError::Invalid(format!("invalid H5MD array shape: {err}"))
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tempfile::NamedTempFile;

    use super::*;
    use crate::xtc::XtcReader;

    fn assert_f32_close(left: f32, right: f32) {
        assert!((left - right).abs() < 1e-4, "{left} !~= {right}");
    }

    fn assert_triplet_close(left: [f32; 3], right: [f32; 3]) {
        assert_f32_close(left[0], right[0]);
        assert_f32_close(left[1], right[1]);
        assert_f32_close(left[2], right[2]);
    }

    #[test]
    fn reader_handles_gromacs_h5md_fixture() {
        let fixture_dir =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../python/warp_md/tests/fixtures/h5md");
        let h5md_path = fixture_dir.join("spc2-traj.h5md");
        let xtc_path = fixture_dir.join("spc2-traj.xtc");

        let mut h5md = H5mdReader::open(&h5md_path).unwrap();
        let mut xtc = XtcReader::open(&xtc_path).unwrap();
        let mut h5md_builder = FrameChunkBuilder::new(6, 2);
        h5md_builder.set_requirements(true, true);
        h5md_builder.set_optional_requirements(true, false, false);
        let mut xtc_builder = FrameChunkBuilder::new(6, 2);
        xtc_builder.set_requirements(true, true);

        assert_eq!(h5md.read_chunk(2, &mut h5md_builder).unwrap(), 2);
        assert_eq!(xtc.read_chunk(2, &mut xtc_builder).unwrap(), 2);

        let h5md_chunk = h5md_builder.finish_take().unwrap();
        let xtc_chunk = xtc_builder.finish_take().unwrap();
        assert_eq!(h5md_chunk.n_frames, 2);
        assert_eq!(h5md_chunk.time_ps.as_ref().unwrap(), &[0.0, 1.0]);
        assert_eq!(h5md_chunk.box_, xtc_chunk.box_);
        assert_eq!(h5md_chunk.coords.len(), xtc_chunk.coords.len());
        for (h5md_atom, xtc_atom) in h5md_chunk.coords.iter().zip(xtc_chunk.coords.iter()) {
            assert_f32_close(h5md_atom[0], xtc_atom[0]);
            assert_f32_close(h5md_atom[1], xtc_atom[1]);
            assert_f32_close(h5md_atom[2], xtc_atom[2]);
        }
        assert!(h5md_chunk.velocities.is_some());
        assert_eq!(h5md_chunk.velocities.as_ref().unwrap().len(), 12);
    }

    #[test]
    fn write_h5md_roundtrip_with_extras() {
        let tempfile = NamedTempFile::new().unwrap();
        let path = tempfile.path();
        let mut writer = H5mdWriter::create(path, 2).unwrap();

        writer
            .write_frame(
                &[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                Box3::Orthorhombic {
                    lx: 20.0,
                    ly: 21.0,
                    lz: 22.0,
                },
                0,
                Some(0.25),
                Some(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
                Some(&[[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]]),
            )
            .unwrap();
        writer
            .write_frame(
                &[[5.5, 4.5, 3.5], [2.5, 1.5, 0.5]],
                Box3::Triclinic {
                    m: [23.0, 1.0, 0.0, 0.0, 24.0, 0.5, 0.0, 0.0, 25.0],
                },
                1,
                Some(1.75),
                Some(&[[0.6, 0.5, 0.4], [0.3, 0.2, 0.1]]),
                Some(&[[3.5, 3.0, 2.5], [2.0, 1.5, 1.0]]),
            )
            .unwrap();
        writer.flush().unwrap();

        let mut reader = H5mdReader::open(path).unwrap();
        let mut builder = FrameChunkBuilder::new(2, 2);
        builder.set_requirements(true, true);
        builder.set_optional_requirements(true, true, false);
        assert_eq!(reader.read_chunk(2, &mut builder).unwrap(), 2);
        let chunk = builder.finish_take().unwrap();

        assert_eq!(chunk.time_ps.as_ref().unwrap(), &[0.25, 1.75]);
        match chunk.box_[0] {
            Box3::Orthorhombic { lx, ly, lz } => {
                assert_f32_close(lx, 20.0);
                assert_f32_close(ly, 21.0);
                assert_f32_close(lz, 22.0);
            }
            other => panic!("expected orthorhombic box, got {other:?}"),
        }
        match chunk.box_[1] {
            Box3::Triclinic { m } => {
                assert_f32_close(m[0], 23.0);
                assert_f32_close(m[4], 24.0);
                assert_f32_close(m[5], 0.5);
                assert_f32_close(m[8], 25.0);
            }
            other => panic!("expected triclinic box, got {other:?}"),
        }
        assert_triplet_close(chunk.velocities.as_ref().unwrap()[0], [0.1, 0.2, 0.3]);
        assert_triplet_close(chunk.velocities.as_ref().unwrap()[3], [0.3, 0.2, 0.1]);
        assert_triplet_close(chunk.forces.as_ref().unwrap()[0], [1.0, 1.5, 2.0]);
        assert_triplet_close(chunk.forces.as_ref().unwrap()[3], [2.0, 1.5, 1.0]);
    }
}
