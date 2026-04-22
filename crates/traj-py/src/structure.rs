use super::*;

use crate::io::{
    open_trajectory_auto, resolve_system_format_token, resolve_trajectory_format_token,
};
use std::fs;
use std::path::{Path, PathBuf};
use warp_structure::io::{read_molecule, write_output, OutputWriteResult};
use warp_structure::{BoxVectors, StructureError};

#[derive(Clone)]
struct StructureTemplate {
    atoms: Vec<AtomRecord>,
    bonds: Vec<(usize, usize)>,
    box_vectors: Option<BoxVectors>,
    ter_after: Vec<usize>,
}

#[pyclass(unsendable)]
struct PyStructureWriter {
    template: StructureTemplate,
    n_atoms: usize,
}

#[pymethods]
impl PyStructureWriter {
    #[staticmethod]
    #[pyo3(signature = (topology_path, n_atoms, topology_format=None))]
    fn open(topology_path: &str, n_atoms: usize, topology_format: Option<&str>) -> PyResult<Self> {
        if n_atoms == 0 {
            return Err(PyRuntimeError::new_err("n_atoms must be > 0"));
        }
        let template =
            load_structure_template(topology_path, topology_format, n_atoms).map_err(to_py_err)?;
        Ok(Self { template, n_atoms })
    }

    #[pyo3(
        signature = (path, coords, box_lengths=None, box_matrix=None, frame_index=0, time_ps=None)
    )]
    fn write_structure(
        &self,
        path: &str,
        coords: PyReadonlyArray2<'_, f32>,
        box_lengths: Option<PyReadonlyArray1<'_, f32>>,
        box_matrix: Option<PyReadonlyArray2<'_, f32>>,
        frame_index: usize,
        time_ps: Option<f32>,
    ) -> PyResult<()> {
        let coords = coords_to_vec(coords, self.n_atoms)?;
        let box_ = py_box_to_box3(box_lengths, box_matrix)?;
        match infer_structure_output_format(path).map_err(to_py_err)? {
            StructureOutputFormat::Pdb => {
                write_pdb_structure(path, &self.template, &coords, box_, frame_index, time_ps)
            }
            StructureOutputFormat::Gro => {
                write_gro_structure(path, &self.template, &coords, box_, frame_index, time_ps)
            }
        }
        .map_err(to_py_err)
        .map(|_| ())
    }
}

#[derive(Clone, Copy)]
enum StructureOutputFormat {
    Pdb,
    Gro,
}

fn infer_structure_output_format(path: &str) -> TrajResult<StructureOutputFormat> {
    match Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "pdb" => Ok(StructureOutputFormat::Pdb),
        "gro" => Ok(StructureOutputFormat::Gro),
        other => Err(TrajError::Invalid(format!(
            "unsupported structure output extension: .{other}"
        ))),
    }
}

fn infer_topology_format(path: &str, topology_format: Option<&str>) -> String {
    topology_format
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| {
            Path::new(path)
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
        })
}

fn load_structure_template(
    topology_path: &str,
    topology_format: Option<&str>,
    n_atoms: usize,
) -> TrajResult<StructureTemplate> {
    let format = infer_topology_format(topology_path, topology_format);
    let molecule = read_molecule(
        Path::new(topology_path),
        Some(format.as_str()),
        false,
        false,
        None,
    )
    .map_err(structure_error_to_traj)?;
    if molecule.atoms.len() != n_atoms {
        return Err(TrajError::Mismatch(format!(
            "topology atom count {} does not match trajectory atom count {}",
            molecule.atoms.len(),
            n_atoms
        )));
    }
    Ok(StructureTemplate {
        atoms: molecule.atoms,
        bonds: molecule.bonds,
        box_vectors: molecule.box_vectors,
        ter_after: molecule.ter_after,
    })
}

fn write_pdb_structure(
    path: &str,
    template: &StructureTemplate,
    coords: &[[f32; 3]],
    box_: Box3,
    frame_index: usize,
    time_ps: Option<f32>,
) -> TrajResult<String> {
    write_structure_output(
        path,
        template,
        coords,
        box_,
        StructureOutputFormat::Pdb,
        frame_index,
        time_ps,
    )
}

fn write_gro_structure(
    path: &str,
    template: &StructureTemplate,
    coords: &[[f32; 3]],
    box_: Box3,
    frame_index: usize,
    time_ps: Option<f32>,
) -> TrajResult<String> {
    write_structure_output(
        path,
        template,
        coords,
        box_,
        StructureOutputFormat::Gro,
        frame_index,
        time_ps,
    )
}

fn write_structure_output(
    path: &str,
    template: &StructureTemplate,
    coords: &[[f32; 3]],
    box_: Box3,
    format: StructureOutputFormat,
    frame_index: usize,
    time_ps: Option<f32>,
) -> TrajResult<String> {
    match format {
        StructureOutputFormat::Pdb => validate_pdb_coords(coords)?,
        StructureOutputFormat::Gro => validate_gro_coords(coords)?,
    }
    let (out, used_placeholder_box) = build_structure_output(template, coords, box_, format)?;
    let spec = OutputSpec {
        path: path.into(),
        format: structure_output_format_name(format).into(),
        scale: None,
    };
    let written = write_output(
        &out,
        &spec,
        matches!(format, StructureOutputFormat::Pdb),
        0.0,
        matches!(format, StructureOutputFormat::Pdb) && !out.bonds.is_empty(),
        false,
    )
    .map_err(structure_error_to_traj)?;
    if written.path != path || written.format != structure_output_format_name(format) {
        let _ = fs::remove_file(&written.path);
        return Err(TrajError::Invalid(format!(
            "structure output for '{path}' fell back to '{}'; export directly to .{} for this writer",
            written.path, written.format
        )));
    }
    postprocess_structure_output(&written, used_placeholder_box, frame_index, time_ps)?;
    Ok(written.path)
}

fn postprocess_structure_output(
    written: &OutputWriteResult,
    used_placeholder_box: bool,
    frame_index: usize,
    time_ps: Option<f32>,
) -> TrajResult<()> {
    match written.format.as_str() {
        "pdb" | "pdb-strict" | "brk" | "ent" => {
            let canonical = fs::read_to_string(&written.path)?;
            let mut text = String::from("HEADER    GENERATED BY WARP-MD FRAMES\n");
            text.push_str(&format!(
                "TITLE     {}\n",
                structure_frame_label(frame_index, time_ps)
            ));
            text.push_str("REMARK     Generated by warp-md frames\n");
            if used_placeholder_box {
                text.push_str("REMARK     Unit cell missing; CRYST1 set to unitary values\n");
            }
            text.push_str(&canonical);
            fs::write(&written.path, text)?;
            Ok(())
        }
        "gro" => postprocess_gro_structure(&written.path, frame_index, time_ps),
        _ => Ok(()),
    }
}

fn postprocess_gro_structure(
    path: &str,
    frame_index: usize,
    time_ps: Option<f32>,
) -> TrajResult<()> {
    let mut lines: Vec<String> = fs::read_to_string(path)?
        .lines()
        .map(str::to_string)
        .collect();
    if lines.is_empty() {
        return Err(TrajError::Invalid(
            "canonical gro writer produced empty output".into(),
        ));
    }
    lines[0] = structure_gro_header(frame_index, time_ps);
    let mut text = lines.join("\n");
    text.push('\n');
    fs::write(path, text)?;
    Ok(())
}

fn structure_output_format_name(format: StructureOutputFormat) -> &'static str {
    match format {
        StructureOutputFormat::Pdb => "pdb",
        StructureOutputFormat::Gro => "gro",
    }
}

fn validate_pdb_coords(coords: &[[f32; 3]]) -> TrajResult<()> {
    for coord in coords {
        for value in coord {
            if *value < -999.9995 || *value > 9999.9995 {
                return Err(TrajError::Invalid(
                    "PDB files require coordinates between -999.999 and 9999.999 A".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_gro_coords(coords: &[[f32; 3]]) -> TrajResult<()> {
    for coord in coords {
        for value in coord {
            let value_nm = *value * 0.1;
            if value_nm < -999.9995 || value_nm > 9999.9995 {
                return Err(TrajError::Invalid(
                    "GRO files require coordinates between -999.999 and 9999.999 nm".into(),
                ));
            }
        }
    }
    Ok(())
}

fn structure_frame_label(frame_index: usize, time_ps: Option<f32>) -> String {
    let mut label = format!("warp-md frame {frame_index}");
    if let Some(time_ps) = time_ps {
        label.push_str(&format!(" time_ps={time_ps:.6}"));
        while label.contains('.') && label.ends_with('0') {
            label.pop();
        }
        if label.ends_with('.') {
            label.pop();
        }
    }
    label
}

fn structure_gro_header(frame_index: usize, time_ps: Option<f32>) -> String {
    let header = format!(
        "Generated by {}",
        structure_frame_label(frame_index, time_ps)
    );
    if header.len() > 79 {
        header[..79].to_string()
    } else {
        header
    }
}

fn build_structure_output(
    template: &StructureTemplate,
    coords: &[[f32; 3]],
    box_: Box3,
    format: StructureOutputFormat,
) -> TrajResult<(PackOutput, bool)> {
    if template.atoms.len() != coords.len() {
        return Err(TrajError::Mismatch(format!(
            "structure atom count {} does not match coordinate count {}",
            template.atoms.len(),
            coords.len()
        )));
    }
    let mut out_atoms = template.atoms.clone();
    for (atom, coord) in out_atoms.iter_mut().zip(coords.iter()) {
        atom.position = Vec3::new(coord[0], coord[1], coord[2]);
    }
    let (box_size, box_vectors, used_placeholder_box) =
        structure_output_box(template, box_, format);
    Ok((
        PackOutput {
            atoms: out_atoms,
            bonds: template.bonds.clone(),
            box_size,
            box_vectors,
            ter_after: template.ter_after.clone(),
        },
        used_placeholder_box,
    ))
}

fn structure_output_box(
    template: &StructureTemplate,
    box_: Box3,
    format: StructureOutputFormat,
) -> ([f32; 3], Option<BoxVectors>, bool) {
    match box_ {
        Box3::None => match template.box_vectors {
            Some(box_vectors) => (structure_box_size(box_vectors), Some(box_vectors), false),
            None => match format {
                StructureOutputFormat::Pdb => ([1.0, 1.0, 1.0], None, true),
                StructureOutputFormat::Gro => ([0.0, 0.0, 0.0], None, false),
            },
        },
        Box3::Orthorhombic { lx, ly, lz } => ([lx, ly, lz], None, false),
        Box3::Triclinic { m } => {
            let box_vectors = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
            (structure_box_size(box_vectors), Some(box_vectors), false)
        }
    }
}

fn structure_box_size(box_vectors: BoxVectors) -> [f32; 3] {
    [
        structure_vector_norm(box_vectors[0]),
        structure_vector_norm(box_vectors[1]),
        structure_vector_norm(box_vectors[2]),
    ]
}

fn structure_vector_norm(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn structure_error_to_traj(err: StructureError) -> TrajError {
    match err {
        StructureError::Io(source) => TrajError::Io(source),
        StructureError::Parse(message) => TrajError::Parse(message),
        StructureError::Invalid(message) => TrajError::Invalid(message),
    }
}

#[pyclass]
struct PyFrameEditor;

#[derive(Clone)]
struct FrameEditFrame {
    index: usize,
    coords: Vec<[f32; 4]>,
    box_: Box3,
    time_ps: Option<f32>,
    velocities: Option<Vec<[f32; 3]>>,
    forces: Option<Vec<[f32; 3]>>,
    lambda_value: Option<f32>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum FrameEditOutputFormat {
    Pdb,
    Gro,
    Dcd,
    Xtc,
    H5md,
    Tng,
    Trr,
}

impl FrameEditOutputFormat {
    fn is_structure(self) -> bool {
        matches!(self, Self::Pdb | Self::Gro)
    }
}

enum FrameEditTrajectoryWriter {
    Dcd(DcdWriter),
    Xtc(XtcWriter),
    H5md(H5mdWriter),
    Tng(TngWriter),
    Trr(TrrWriter),
}

#[pymethods]
impl PyFrameEditor {
    #[staticmethod]
    #[pyo3(
        signature = (
            topology_path,
            traj_path,
            out_path,
            begin=0,
            end=None,
            step=1,
            index=None,
            topology_format=None,
            traj_format=None,
            traj_length_scale=None,
            chunk_frames=None
        )
    )]
    fn run<'py>(
        py: Python<'py>,
        topology_path: &str,
        traj_path: &str,
        out_path: &str,
        begin: usize,
        end: Option<usize>,
        step: usize,
        index: Option<usize>,
        topology_format: Option<&str>,
        traj_format: Option<&str>,
        traj_length_scale: Option<f32>,
        chunk_frames: Option<usize>,
    ) -> PyResult<PyObject> {
        let topology_format = resolve_frame_edit_topology_format(topology_path, topology_format)?;
        let traj_format = resolve_frame_edit_traj_format(traj_path, traj_format)?;
        let output_format = infer_frame_edit_output_format(out_path)?;
        let preserve_trr_fields = output_format == FrameEditOutputFormat::Trr;

        let system = load_frame_edit_system(topology_path, &topology_format)?;
        let n_atoms = system.system.borrow().n_atoms();
        let mut traj =
            open_frame_edit_trajectory(traj_path, &traj_format, &system, traj_length_scale)?;
        let chunk_frames = resolve_frame_edit_chunk_frames(&traj, chunk_frames)?;

        let template = if output_format.is_structure() {
            Some(
                load_structure_template(topology_path, Some(topology_format.as_str()), n_atoms)
                    .map_err(to_py_err)?,
            )
        } else {
            None
        };

        let out_path_buf = PathBuf::from(out_path);
        let mut outputs = Vec::new();
        let (selection, total_frames, expected_frames, output_mode, written_frames) =
            if let Some(index) = index {
                let frame = read_frame_edit_single_frame(&mut traj, index, preserve_trr_fields)?;
                write_frame_edit_outputs(
                    &out_path_buf,
                    output_format,
                    template.as_ref(),
                    n_atoms,
                    1,
                    &[frame],
                    &mut outputs,
                )
                .map_err(to_py_err)?;
                (
                    frame_edit_single_selection_dict(py, index)?,
                    None,
                    1usize,
                    if output_format.is_structure() {
                        "single_structure"
                    } else {
                        "trajectory"
                    },
                    1usize,
                )
            } else if let Some(end) = end {
                let begin_frame = begin;
                if step == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err("step must be >= 1"));
                }
                if begin_frame >= end {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "begin frame is greater than or equal to end frame",
                    ));
                }
                let expected_frames = frame_count_for_range(begin_frame, end, step);
                let written_frames = stream_frame_edit_range(
                    &mut traj,
                    begin_frame,
                    end,
                    step,
                    chunk_frames,
                    preserve_trr_fields,
                    &out_path_buf,
                    output_format,
                    template.as_ref(),
                    n_atoms,
                    expected_frames,
                    &mut outputs,
                )
                .map_err(to_py_err)?;
                if written_frames == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "begin frame is greater than or equal to end frame",
                    ));
                }
                (
                    frame_edit_range_selection_dict(py, begin_frame, end, step)?,
                    None,
                    expected_frames,
                    if output_format.is_structure() && expected_frames == 1 {
                        "single_structure"
                    } else if output_format.is_structure() {
                        "structure_series"
                    } else {
                        "trajectory"
                    },
                    written_frames,
                )
            } else {
                let total_frames = traj.count_frames(Some(chunk_frames))?;
                traj.reset()?;
                if total_frames == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "trajectory has no frames",
                    ));
                }
                if step == 0 {
                    return Err(pyo3::exceptions::PyValueError::new_err("step must be >= 1"));
                }
                let begin_frame = begin.min(total_frames);
                if begin_frame >= total_frames {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "begin frame is greater than or equal to end frame",
                    ));
                }
                let expected_frames = frame_count_for_range(begin_frame, total_frames, step);
                let written_frames = stream_frame_edit_range(
                    &mut traj,
                    begin_frame,
                    total_frames,
                    step,
                    chunk_frames,
                    preserve_trr_fields,
                    &out_path_buf,
                    output_format,
                    template.as_ref(),
                    n_atoms,
                    expected_frames,
                    &mut outputs,
                )
                .map_err(to_py_err)?;
                if written_frames != expected_frames {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "expected to write {expected_frames} frames but wrote {written_frames}"
                    )));
                }
                (
                    frame_edit_range_selection_dict(py, begin_frame, total_frames, step)?,
                    Some(total_frames),
                    expected_frames,
                    if output_format.is_structure() && expected_frames == 1 {
                        "single_structure"
                    } else if output_format.is_structure() {
                        "structure_series"
                    } else {
                        "trajectory"
                    },
                    written_frames,
                )
            };

        let payload = PyDict::new_bound(py);
        payload.set_item("status", "ok")?;
        payload.set_item("command", "frames")?;
        let topology = PyDict::new_bound(py);
        topology.set_item("path", topology_path)?;
        topology.set_item("format", topology_format)?;
        payload.set_item("topology", topology)?;
        let trajectory = PyDict::new_bound(py);
        trajectory.set_item("path", traj_path)?;
        trajectory.set_item("format", traj_format)?;
        trajectory.set_item("length_scale", traj_length_scale)?;
        trajectory.set_item("total_frames", total_frames)?;
        payload.set_item("trajectory", trajectory)?;
        payload.set_item("selection", selection)?;
        payload.set_item("written_frames", written_frames)?;
        payload.set_item("output_mode", output_mode)?;
        payload.set_item("outputs", outputs)?;
        let _ = expected_frames;
        Ok(payload.into_py(py))
    }
}

fn resolve_frame_edit_topology_format(
    topology_path: &str,
    topology_format: Option<&str>,
) -> PyResult<String> {
    resolve_system_format_token(topology_path, topology_format).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("topology format must be pdb, pdbqt, or gro")
    })
}

fn resolve_frame_edit_traj_format(traj_path: &str, traj_format: Option<&str>) -> PyResult<String> {
    resolve_trajectory_format_token(traj_path, traj_format)
}

fn infer_frame_edit_output_format(path: &str) -> PyResult<FrameEditOutputFormat> {
    match Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "pdb" => Ok(FrameEditOutputFormat::Pdb),
        "gro" => Ok(FrameEditOutputFormat::Gro),
        "dcd" => Ok(FrameEditOutputFormat::Dcd),
        "xtc" => Ok(FrameEditOutputFormat::Xtc),
        "h5md" => Ok(FrameEditOutputFormat::H5md),
        "tng" => Ok(FrameEditOutputFormat::Tng),
        "trr" => Ok(FrameEditOutputFormat::Trr),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported output extension: .{other}"
        ))),
    }
}

fn load_frame_edit_system(topology_path: &str, topology_format: &str) -> PyResult<PySystem> {
    PySystem::from_file(topology_path, Some(topology_format))
}

fn open_frame_edit_trajectory(
    traj_path: &str,
    traj_format: &str,
    system: &PySystem,
    traj_length_scale: Option<f32>,
) -> PyResult<PyTrajectory> {
    open_trajectory_auto(traj_path, system, Some(traj_format), traj_length_scale)
}

fn resolve_frame_edit_chunk_frames(
    traj: &PyTrajectory,
    chunk_frames: Option<usize>,
) -> PyResult<usize> {
    if let Some(chunk_frames) = chunk_frames {
        return Ok(chunk_frames.max(1));
    }
    let inner = traj.inner.borrow();
    resolve_chunk_frames_for_streaming(&inner)
}

fn frame_count_for_range(begin: usize, end: usize, step: usize) -> usize {
    if begin >= end || step == 0 {
        0
    } else {
        1 + (end - begin - 1) / step
    }
}

fn frame_edit_single_selection_dict<'py>(
    py: Python<'py>,
    index: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let selection = PyDict::new_bound(py);
    selection.set_item("mode", "single")?;
    selection.set_item("index", index)?;
    Ok(selection)
}

fn frame_edit_range_selection_dict<'py>(
    py: Python<'py>,
    begin: usize,
    end: usize,
    step: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let selection = PyDict::new_bound(py);
    selection.set_item("mode", "range")?;
    selection.set_item("begin", begin)?;
    selection.set_item("end", end)?;
    selection.set_item("step", step)?;
    Ok(selection)
}

fn read_frame_edit_single_frame(
    traj: &mut PyTrajectory,
    index: usize,
    preserve_trr_fields: bool,
) -> PyResult<FrameEditFrame> {
    let mut inner = traj.inner.borrow_mut();
    let reader = &mut *inner;
    if index > 0 {
        let skipped = frame_edit_skip_frames(reader, index).map_err(to_py_err)?;
        if skipped < index {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "frame index {index} is out of range"
            )));
        }
    }
    let mut builder = FrameChunkBuilder::new(frame_edit_n_atoms(reader), 1);
    builder.set_requirements(true, true);
    builder.set_optional_requirements(
        preserve_trr_fields,
        preserve_trr_fields,
        preserve_trr_fields,
    );
    let read = frame_edit_read_chunk(reader, 1, &mut builder).map_err(to_py_err)?;
    if read == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "frame index {index} is out of range"
        )));
    }
    let chunk = builder.finish_take().map_err(to_py_err)?;
    Ok(extract_frame_edit_frame(&chunk, 0, index))
}

fn stream_frame_edit_range(
    traj: &mut PyTrajectory,
    begin: usize,
    end: usize,
    step: usize,
    chunk_frames: usize,
    preserve_trr_fields: bool,
    out_path: &Path,
    output_format: FrameEditOutputFormat,
    template: Option<&StructureTemplate>,
    n_atoms: usize,
    expected_frames: usize,
    outputs: &mut Vec<String>,
) -> TrajResult<usize> {
    let mut inner = traj.inner.borrow_mut();
    let reader = &mut *inner;
    if begin > 0 {
        let skipped = frame_edit_skip_frames(reader, begin)?;
        if skipped < begin {
            return Ok(0);
        }
    }

    if output_format.is_structure() {
        let template = template.expect("structure template required for structure outputs");
        let single_output = expected_frames == 1;
        return stream_frame_edit_structure_outputs(
            reader,
            begin,
            end,
            step,
            chunk_frames,
            preserve_trr_fields,
            out_path,
            output_format,
            template,
            single_output,
            outputs,
        );
    }

    let mut writer =
        FrameEditTrajectoryWriter::open(out_path, output_format, n_atoms, expected_frames)?;
    outputs.push(out_path.to_string_lossy().into_owned());
    let written = stream_frame_edit_frames(
        reader,
        begin,
        end,
        step,
        chunk_frames,
        preserve_trr_fields,
        |frame| writer.write_frame(&frame),
    )?;
    writer.flush()?;
    Ok(written)
}

fn stream_frame_edit_structure_outputs(
    reader: &mut TrajKind,
    begin: usize,
    end: usize,
    step: usize,
    chunk_frames: usize,
    preserve_trr_fields: bool,
    out_path: &Path,
    output_format: FrameEditOutputFormat,
    template: &StructureTemplate,
    single_output: bool,
    outputs: &mut Vec<String>,
) -> TrajResult<usize> {
    stream_frame_edit_frames(
        reader,
        begin,
        end,
        step,
        chunk_frames,
        preserve_trr_fields,
        |frame| {
            let target = if single_output {
                out_path.to_path_buf()
            } else {
                frame_edit_series_path(out_path, frame.index)
            };
            let written =
                write_frame_edit_structure_output(&target, output_format, template, &frame)?;
            outputs.push(written);
            Ok(())
        },
    )
}

fn write_frame_edit_outputs(
    out_path: &Path,
    output_format: FrameEditOutputFormat,
    template: Option<&StructureTemplate>,
    n_atoms: usize,
    expected_frames: usize,
    frames: &[FrameEditFrame],
    outputs: &mut Vec<String>,
) -> TrajResult<()> {
    if output_format.is_structure() {
        let template = template.expect("structure template required for structure outputs");
        let single_output = expected_frames == 1;
        for frame in frames {
            let target = if single_output {
                out_path.to_path_buf()
            } else {
                frame_edit_series_path(out_path, frame.index)
            };
            let written =
                write_frame_edit_structure_output(&target, output_format, template, frame)?;
            outputs.push(written);
        }
        return Ok(());
    }

    let mut writer =
        FrameEditTrajectoryWriter::open(out_path, output_format, n_atoms, expected_frames)?;
    outputs.push(out_path.to_string_lossy().into_owned());
    for frame in frames {
        writer.write_frame(frame)?;
    }
    writer.flush()
}

fn stream_frame_edit_frames<F>(
    reader: &mut TrajKind,
    begin: usize,
    end: usize,
    step: usize,
    chunk_frames: usize,
    preserve_trr_fields: bool,
    mut on_frame: F,
) -> TrajResult<usize>
where
    F: FnMut(FrameEditFrame) -> TrajResult<()>,
{
    let chunk_frames = chunk_frames.max(1);
    let mut written = 0usize;
    let mut global = begin;
    if step == 1 {
        let mut builder = FrameChunkBuilder::new(frame_edit_n_atoms(reader), chunk_frames);
        builder.set_requirements(true, true);
        builder.set_optional_requirements(
            preserve_trr_fields,
            preserve_trr_fields,
            preserve_trr_fields,
        );
        while global < end {
            let wanted = (end - global).min(chunk_frames);
            let read = frame_edit_read_chunk(reader, wanted, &mut builder)?;
            if read == 0 {
                break;
            }
            let chunk = builder.finish_take()?;
            for local_index in 0..chunk.n_frames {
                on_frame(extract_frame_edit_frame(&chunk, local_index, global))?;
                global += 1;
                written += 1;
            }
            builder.reclaim(chunk);
            if read < wanted {
                break;
            }
        }
        return Ok(written);
    }

    let mut builder = FrameChunkBuilder::new(frame_edit_n_atoms(reader), 1);
    builder.set_requirements(true, true);
    builder.set_optional_requirements(
        preserve_trr_fields,
        preserve_trr_fields,
        preserve_trr_fields,
    );
    while global < end {
        let read = frame_edit_read_chunk(reader, 1, &mut builder)?;
        if read == 0 {
            break;
        }
        let chunk = builder.finish_take()?;
        on_frame(extract_frame_edit_frame(&chunk, 0, global))?;
        builder.reclaim(chunk);
        written += 1;
        global += 1;
        if global >= end {
            break;
        }
        let gap = (step - 1).min(end - global);
        let skipped = frame_edit_skip_frames(reader, gap)?;
        global += skipped;
        if skipped < gap {
            break;
        }
    }
    Ok(written)
}

fn frame_edit_n_atoms(reader: &TrajKind) -> usize {
    match reader {
        TrajKind::Dcd { reader } => reader.n_atoms(),
        TrajKind::Xtc { reader } => reader.n_atoms(),
        TrajKind::Gro { reader } => reader.n_atoms(),
        TrajKind::G96 { reader } => reader.n_atoms(),
        TrajKind::Cpt { reader } => reader.n_atoms(),
        TrajKind::H5md { reader } => reader.n_atoms(),
        TrajKind::Tng { reader } => reader.n_atoms(),
        TrajKind::Trr { reader } => reader.n_atoms(),
        TrajKind::Pdb { reader } => reader.n_atoms(),
        TrajKind::Memory { reader } => reader.n_atoms(),
    }
}

fn frame_edit_skip_frames(reader: &mut TrajKind, n_frames: usize) -> TrajResult<usize> {
    match reader {
        TrajKind::Dcd { reader } => reader.skip_frames(n_frames),
        TrajKind::Xtc { reader } => reader.skip_frames(n_frames),
        TrajKind::Gro { reader } => reader.skip_frames(n_frames),
        TrajKind::G96 { reader } => reader.skip_frames(n_frames),
        TrajKind::Cpt { reader } => reader.skip_frames(n_frames),
        TrajKind::H5md { reader } => reader.skip_frames(n_frames),
        TrajKind::Tng { reader } => reader.skip_frames(n_frames),
        TrajKind::Trr { reader } => reader.skip_frames(n_frames),
        TrajKind::Pdb { reader } => reader.skip_frames(n_frames),
        TrajKind::Memory { reader } => reader.skip_frames(n_frames),
    }
}

fn frame_edit_read_chunk(
    reader: &mut TrajKind,
    max_frames: usize,
    builder: &mut FrameChunkBuilder,
) -> TrajResult<usize> {
    match reader {
        TrajKind::Dcd { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::Xtc { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::Gro { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::G96 { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::Cpt { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::H5md { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::Tng { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::Trr { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::Pdb { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::Memory { reader } => reader.read_chunk(max_frames, builder),
    }
}

fn extract_frame_edit_frame(
    chunk: &traj_core::frame::FrameChunk,
    frame: usize,
    absolute_index: usize,
) -> FrameEditFrame {
    let start = frame * chunk.n_atoms;
    let end = start + chunk.n_atoms;
    FrameEditFrame {
        index: absolute_index,
        coords: chunk.coords[start..end].to_vec(),
        box_: chunk.box_.get(frame).copied().unwrap_or(Box3::None),
        time_ps: chunk
            .time_ps
            .as_ref()
            .and_then(|values| values.get(frame).copied()),
        velocities: chunk
            .velocities
            .as_ref()
            .map(|values| values[start..end].to_vec()),
        forces: chunk
            .forces
            .as_ref()
            .map(|values| values[start..end].to_vec()),
        lambda_value: chunk
            .lambda_values
            .as_ref()
            .and_then(|values| values.get(frame).copied()),
    }
}

impl FrameEditTrajectoryWriter {
    fn open(
        path: &Path,
        output_format: FrameEditOutputFormat,
        n_atoms: usize,
        expected_frames: usize,
    ) -> TrajResult<Self> {
        match output_format {
            FrameEditOutputFormat::Dcd => Ok(Self::Dcd(DcdWriter::create(
                path,
                n_atoms,
                expected_frames,
            )?)),
            FrameEditOutputFormat::Xtc => Ok(Self::Xtc(XtcWriter::create(path, n_atoms)?)),
            FrameEditOutputFormat::H5md => Ok(Self::H5md(H5mdWriter::create(path, n_atoms)?)),
            FrameEditOutputFormat::Tng => Ok(Self::Tng(TngWriter::create(path, n_atoms)?)),
            FrameEditOutputFormat::Trr => Ok(Self::Trr(TrrWriter::create(path, n_atoms)?)),
            FrameEditOutputFormat::Pdb | FrameEditOutputFormat::Gro => Err(TrajError::Invalid(
                "structure output requested from trajectory writer".into(),
            )),
        }
    }

    fn write_frame(&mut self, frame: &FrameEditFrame) -> TrajResult<()> {
        let coords = frame_edit_coords3(&frame.coords);
        match self {
            Self::Dcd(writer) => writer.write_frame(&coords, frame.box_),
            Self::Xtc(writer) => {
                writer.write_frame(&coords, frame.box_, frame.index, frame.time_ps)
            }
            Self::H5md(writer) => writer.write_frame(
                &coords,
                frame.box_,
                frame.index,
                frame.time_ps,
                frame.velocities.as_deref(),
                frame.forces.as_deref(),
            ),
            Self::Tng(writer) => writer.write_frame(
                &coords,
                frame.box_,
                frame.index,
                frame.time_ps,
                frame.velocities.as_deref(),
                frame.forces.as_deref(),
            ),
            Self::Trr(writer) => writer.write_frame(
                &coords,
                frame.box_,
                frame.index,
                frame.time_ps,
                frame.velocities.as_deref(),
                frame.forces.as_deref(),
                frame.lambda_value,
            ),
        }
    }

    fn flush(&mut self) -> TrajResult<()> {
        match self {
            Self::Dcd(writer) => writer.flush(),
            Self::Xtc(writer) => writer.flush(),
            Self::H5md(writer) => writer.flush(),
            Self::Tng(writer) => writer.flush(),
            Self::Trr(writer) => writer.flush(),
        }
    }
}

fn write_frame_edit_structure_output(
    path: &Path,
    output_format: FrameEditOutputFormat,
    template: &StructureTemplate,
    frame: &FrameEditFrame,
) -> TrajResult<String> {
    let coords = frame_edit_coords3(&frame.coords);
    match output_format {
        FrameEditOutputFormat::Pdb => write_pdb_structure(
            path.to_string_lossy().as_ref(),
            template,
            &coords,
            frame.box_,
            frame.index,
            frame.time_ps,
        ),
        FrameEditOutputFormat::Gro => write_gro_structure(
            path.to_string_lossy().as_ref(),
            template,
            &coords,
            frame.box_,
            frame.index,
            frame.time_ps,
        ),
        FrameEditOutputFormat::Dcd
        | FrameEditOutputFormat::Xtc
        | FrameEditOutputFormat::H5md
        | FrameEditOutputFormat::Tng
        | FrameEditOutputFormat::Trr => Err(TrajError::Invalid(
            "trajectory output requested from structure writer".into(),
        )),
    }
}

fn frame_edit_coords3(coords: &[[f32; 4]]) -> Vec<[f32; 3]> {
    coords
        .iter()
        .map(|value| [value[0], value[1], value[2]])
        .collect()
}

fn frame_edit_series_path(base: &Path, frame_index: usize) -> PathBuf {
    let stem = base
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("frame");
    let ext = base
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    base.with_file_name(format!("{stem}_{frame_index}.{ext}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process;
    use std::time::{SystemTime, UNIX_EPOCH};
    use warp_structure::PdbAtomMetadata;

    fn temp_output_path(ext: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "warp_md_traj_py_structure_{}_{}.{}",
            process::id(),
            nonce,
            ext
        ))
    }

    fn test_atom(record_kind: AtomRecordKind, name: &str) -> AtomRecord {
        AtomRecord {
            record_kind,
            name: name.into(),
            element: name.into(),
            resname: "GLY".into(),
            resid: 7,
            chain: 'A',
            segid: String::new(),
            charge: 0.0,
            position: Vec3::new(0.0, 0.0, 0.0),
            mol_id: 1,
            pdb_metadata: Some(PdbAtomMetadata {
                occupancy: Some(1.0),
                temp_factor: Some(0.0),
                altloc: None,
                insertion_code: None,
                formal_charge: None,
                pqr_radius: None,
            }),
        }
    }

    fn test_template(atoms: Vec<AtomRecord>) -> StructureTemplate {
        StructureTemplate {
            atoms,
            bonds: vec![(0, 1)],
            box_vectors: None,
            ter_after: vec![0],
        }
    }

    fn write_topology_pdb(path: &Path) {
        fs::write(
            path,
            concat!(
                "ATOM      1  CA  GLY A   7      10.000  11.000  12.000  1.00  0.00      TEST C  \n",
                "ATOM      2  CB  GLY A   7      13.000  14.000  15.000  1.00  0.00      TEST C  \n",
                "TER       3      GLY A   7\n",
                "CONECT    1    2\n",
                "END\n"
            ),
        )
        .unwrap();
    }

    #[test]
    fn load_structure_template_preserves_pdb_topology_metadata() {
        let path = temp_output_path("pdb");
        write_topology_pdb(&path);

        let template = load_structure_template(path.to_str().unwrap(), Some("pdb"), 2).unwrap();

        assert_eq!(template.bonds, vec![(0, 1)]);
        assert_eq!(template.ter_after, vec![1]);

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_pdb_structure_preserves_frame_metadata() {
        let path = temp_output_path("pdb");
        let template = test_template(vec![
            test_atom(AtomRecordKind::HetAtom, "CA"),
            test_atom(AtomRecordKind::Atom, "CB"),
        ]);
        let coords = [[12.0, 13.5, 14.25], [15.0, 16.5, 17.25]];
        write_pdb_structure(
            path.to_str().unwrap(),
            &template,
            &coords,
            Box3::None,
            0,
            Some(0.75),
        )
        .unwrap();

        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("HEADER    GENERATED BY WARP-MD FRAMES"));
        assert!(text.contains("TITLE     warp-md frame 0 time_ps=0.75"));
        assert!(text.contains("REMARK     Generated by warp-md frames"));
        assert!(text.contains("HETATM    1   CA GLY A   7"));
        assert!(text.contains("TER       2      GLY A   7"));
        assert!(text.contains("ATOM      2   CB GLY A   7"));
        assert!(text.contains("CONECT    1    2"));
        assert!(text.contains("CONECT    2    1"));
        assert!(text.trim_end().ends_with("END"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn write_gro_structure_preserves_frame_header() {
        let path = temp_output_path("gro");
        let template = test_template(vec![
            test_atom(AtomRecordKind::Atom, "CA"),
            test_atom(AtomRecordKind::Atom, "CB"),
        ]);
        let coords = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        write_gro_structure(
            path.to_str().unwrap(),
            &template,
            &coords,
            Box3::Orthorhombic {
                lx: 10.0,
                ly: 20.0,
                lz: 30.0,
            },
            3,
            Some(1.25),
        )
        .unwrap();

        let text = fs::read_to_string(&path).unwrap();
        assert!(text
            .lines()
            .next()
            .unwrap_or_default()
            .starts_with("Generated by warp-md frame 3 time_ps=1.25"));
        assert!(text.contains("    2"));

        let _ = fs::remove_file(path);
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStructureWriter>()?;
    m.add_class::<PyFrameEditor>()?;
    Ok(())
}
