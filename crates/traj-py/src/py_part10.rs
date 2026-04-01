use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

#[derive(Clone)]
struct StructureAtom {
    record_kind: PdbRecordKind,
    name: String,
    resname: String,
    resid: i32,
    chain: char,
    segid: String,
    element: String,
    occupancy: f32,
    temp_factor: f32,
    altloc: char,
    icode: char,
    charge: String,
}

#[pyclass(unsendable)]
struct PyStructureWriter {
    atoms: Vec<StructureAtom>,
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
        let atoms =
            load_structure_atoms(topology_path, topology_format, n_atoms).map_err(to_py_err)?;
        Ok(Self { atoms, n_atoms })
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
                write_pdb_structure(path, &self.atoms, &coords, box_, frame_index, time_ps)
            }
            StructureOutputFormat::Gro => {
                write_gro_structure(path, &self.atoms, &coords, box_, frame_index, time_ps)
            }
        }
        .map_err(to_py_err)
    }
}

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

fn load_structure_atoms(
    topology_path: &str,
    topology_format: Option<&str>,
    n_atoms: usize,
) -> TrajResult<Vec<StructureAtom>> {
    let file = File::open(topology_path)?;
    let reader = BufReader::new(file);
    let format = infer_topology_format(topology_path, topology_format);
    let atoms: Vec<StructureAtom> = match format.as_str() {
        "pdb" => {
            let parsed = parse_pdb_reader(
                reader,
                &PdbParseOptions {
                    include_conect: false,
                    non_standard_conect: false,
                    include_ter: false,
                    strict: false,
                    only_first_model: true,
                },
            )?;
            parsed
                .atoms
                .into_iter()
                .map(|atom| StructureAtom {
                    record_kind: atom.record_kind,
                    name: atom.name,
                    resname: atom.resname,
                    resid: atom.resid,
                    chain: atom.chain,
                    segid: atom.segid,
                    element: atom.element,
                    occupancy: 1.0,
                    temp_factor: 0.0,
                    altloc: ' ',
                    icode: ' ',
                    charge: String::new(),
                })
                .collect()
        }
        "gro" => {
            let parsed = parse_gro_reader(reader, false)?;
            parsed
                .atoms
                .into_iter()
                .map(|atom| StructureAtom {
                    record_kind: PdbRecordKind::Atom,
                    name: atom.name,
                    resname: atom.resname,
                    resid: atom.resid,
                    chain: 'A',
                    segid: String::new(),
                    element: atom.element,
                    occupancy: 1.0,
                    temp_factor: 0.0,
                    altloc: ' ',
                    icode: ' ',
                    charge: String::new(),
                })
                .collect()
        }
        other => {
            return Err(TrajError::Invalid(format!(
                "unsupported topology format: {other}"
            )))
        }
    };
    if atoms.len() != n_atoms {
        return Err(TrajError::Mismatch(format!(
            "topology atom count {} does not match trajectory atom count {}",
            atoms.len(),
            n_atoms
        )));
    }
    Ok(atoms)
}

fn write_pdb_structure(
    path: &str,
    atoms: &[StructureAtom],
    coords: &[[f32; 3]],
    box_: Box3,
    frame_index: usize,
    time_ps: Option<f32>,
) -> TrajResult<()> {
    validate_pdb_coords(coords)?;
    let file = File::create(path)?;
    let mut handle = BufWriter::new(file);
    let dimensions = pdb_dimensions(box_);

    handle.write_all(b"HEADER    GENERATED BY WARP-MD FRAMES\n")?;
    let mut title = format!("warp-md frame {frame_index}");
    if let Some(time_ps) = time_ps {
        title.push_str(&format!(" time_ps={time_ps:.6}"));
        while title.contains('.') && title.ends_with('0') {
            title.pop();
        }
        if title.ends_with('.') {
            title.pop();
        }
    }
    writeln!(handle, "TITLE     {title}")?;
    handle.write_all(b"REMARK     Generated by warp-md frames\n")?;
    handle.write_all(b"REMARK     Native writer path\n")?;
    if let Some([a, b, c, alpha, beta, gamma]) = dimensions {
        writeln!(
            handle,
            "CRYST1{a:9.3}{b:9.3}{c:9.3}{alpha:7.2}{beta:7.2}{gamma:7.2} P 1           1"
        )?;
    } else {
        handle.write_all(b"CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n")?;
        handle.write_all(b"REMARK     Unit cell missing; CRYST1 set to unitary values\n")?;
    }

    for (idx, (atom, coord)) in atoms.iter().zip(coords.iter()).enumerate() {
        let record = match atom.record_kind {
            PdbRecordKind::Atom => "ATOM",
            PdbRecordKind::HetAtom => "HETATM",
        };
        let altloc = sanitize_pdb_single_char(atom.altloc);
        let icode = sanitize_pdb_single_char(atom.icode);
        writeln!(
            handle,
            "{record:<6}{serial:5} {name:<4}{altloc}{resname:<4}{chain}{resid:4}{icode}   {x:8.3}{y:8.3}{z:8.3}{occ:6.2}{temp:6.2}      {segid:<4}{element:>2}{charge}",
            serial = idx + 1,
            name = truncate(&atom.name, 4),
            resname = truncate(&atom.resname, 4),
            chain = format_pdb_chain(atom.chain),
            resid = atom.resid,
            x = coord[0],
            y = coord[1],
            z = coord[2],
            occ = atom.occupancy,
            temp = atom.temp_factor,
            segid = truncate(&atom.segid, 4),
            element = format_pdb_element(&atom.element, &atom.name),
            charge = format_pdb_charge(&atom.charge),
        )?;
    }
    handle.write_all(b"END\n")?;
    Ok(())
}

fn write_gro_structure(
    path: &str,
    atoms: &[StructureAtom],
    coords: &[[f32; 3]],
    box_: Box3,
    frame_index: usize,
    time_ps: Option<f32>,
) -> TrajResult<()> {
    validate_gro_coords(coords)?;
    let file = File::create(path)?;
    let mut handle = BufWriter::new(file);
    let mut header = format!("Generated by warp-md frame {frame_index}");
    if let Some(time_ps) = time_ps {
        header.push_str(&format!(" time_ps={time_ps:.6}"));
        while header.contains('.') && header.ends_with('0') {
            header.pop();
        }
        if header.ends_with('.') {
            header.pop();
        }
    }
    let header = if header.len() > 79 { &header[..79] } else { &header };
    writeln!(handle, "{header}")?;
    writeln!(handle, "{:5}", atoms.len())?;
    for (idx, (atom, coord)) in atoms.iter().zip(coords.iter()).enumerate() {
        writeln!(
            handle,
            "{resid:5}{resname:<5}{name:>5}{serial:5}{x:8.3}{y:8.3}{z:8.3}",
            resid = atom.resid.rem_euclid(100000),
            resname = truncate(&atom.resname, 5),
            name = truncate(&atom.name, 5),
            serial = (idx + 1) % 100000,
            x = coord[0] * 0.1,
            y = coord[1] * 0.1,
            z = coord[2] * 0.1,
        )?;
    }
    let box_fields = gro_box_fields(box_);
    for (idx, value) in box_fields.iter().enumerate() {
        if idx > 0 {
            handle.write_all(b" ")?;
        }
        write!(handle, "{value:.5}")?;
    }
    handle.write_all(b"\n")?;
    Ok(())
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

fn truncate(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
}

fn sanitize_pdb_single_char(value: char) -> char {
    if value == '\0' { ' ' } else { value }
}

fn format_pdb_chain(chain: char) -> char {
    if chain.is_ascii_alphanumeric() { chain } else { 'X' }
}

fn infer_element(atom_name: &str) -> String {
    let letters = atom_name
        .trim()
        .chars()
        .filter(|value| value.is_ascii_alphabetic())
        .collect::<Vec<_>>();
    if letters.is_empty() {
        return "X".into();
    }
    if letters.len() >= 2 && letters[1].is_ascii_lowercase() {
        format!("{}{}", letters[0].to_ascii_uppercase(), letters[1].to_ascii_lowercase())
    } else {
        letters[0].to_ascii_uppercase().to_string()
    }
}

fn format_pdb_element(element: &str, atom_name: &str) -> String {
    let raw = if element.trim().is_empty() {
        infer_element(atom_name)
    } else {
        element.trim().to_string()
    };
    truncate(&raw, 2)
}

fn format_pdb_charge(charge: &str) -> String {
    let trimmed = charge.trim();
    if trimmed.is_empty() {
        "  ".into()
    } else if trimmed.len() > 2 {
        trimmed[trimmed.len() - 2..].to_string()
    } else {
        format!("{trimmed:>2}")
    }
}

fn pdb_dimensions(box_: Box3) -> Option<[f32; 6]> {
    match box_ {
        Box3::None => None,
        Box3::Orthorhombic { lx, ly, lz } => {
            if lx <= 0.0 || ly <= 0.0 || lz <= 0.0 {
                None
            } else {
                Some([lx, ly, lz, 90.0, 90.0, 90.0])
            }
        }
        Box3::Triclinic { m } => {
            let a = [m[0], m[1], m[2]];
            let b = [m[3], m[4], m[5]];
            let c = [m[6], m[7], m[8]];
            let len_a = vector_norm(a);
            let len_b = vector_norm(b);
            let len_c = vector_norm(c);
            if len_a <= 0.0 || len_b <= 0.0 || len_c <= 0.0 {
                return None;
            }
            Some([
                len_a,
                len_b,
                len_c,
                angle_degrees(b, c),
                angle_degrees(a, c),
                angle_degrees(a, b),
            ])
        }
    }
}

fn gro_box_fields(box_: Box3) -> Vec<f32> {
    match box_ {
        Box3::None => vec![0.0, 0.0, 0.0],
        Box3::Orthorhombic { lx, ly, lz } => vec![lx * 0.1, ly * 0.1, lz * 0.1],
        Box3::Triclinic { m } => {
            let matrix_nm = [
                [m[0] * 0.1, m[1] * 0.1, m[2] * 0.1],
                [m[3] * 0.1, m[4] * 0.1, m[5] * 0.1],
                [m[6] * 0.1, m[7] * 0.1, m[8] * 0.1],
            ];
            if is_diagonal(matrix_nm) {
                vec![matrix_nm[0][0], matrix_nm[1][1], matrix_nm[2][2]]
            } else {
                vec![
                    matrix_nm[0][0],
                    matrix_nm[1][1],
                    matrix_nm[2][2],
                    matrix_nm[0][1],
                    matrix_nm[0][2],
                    matrix_nm[1][0],
                    matrix_nm[1][2],
                    matrix_nm[2][0],
                    matrix_nm[2][1],
                ]
            }
        }
    }
}

fn is_diagonal(matrix: [[f32; 3]; 3]) -> bool {
    const EPS: f32 = 1.0e-6;
    matrix[0][1].abs() <= EPS
        && matrix[0][2].abs() <= EPS
        && matrix[1][0].abs() <= EPS
        && matrix[1][2].abs() <= EPS
        && matrix[2][0].abs() <= EPS
        && matrix[2][1].abs() <= EPS
}

fn vector_norm(value: [f32; 3]) -> f32 {
    (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt()
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn angle_degrees(a: [f32; 3], b: [f32; 3]) -> f32 {
    let denom = vector_norm(a) * vector_norm(b);
    if denom <= 0.0 {
        return 90.0;
    }
    let cosine = (dot(a, b) / denom).clamp(-1.0, 1.0);
    cosine.acos().to_degrees()
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
        let mut traj = open_frame_edit_trajectory(
            traj_path,
            &traj_format,
            &system,
            traj_length_scale,
        )?;
        let chunk_frames = resolve_frame_edit_chunk_frames(&traj, chunk_frames)?;

        let atoms = if output_format.is_structure() {
            Some(load_structure_atoms(
                topology_path,
                Some(topology_format.as_str()),
                n_atoms,
            )
            .map_err(to_py_err)?)
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
                    atoms.as_deref(),
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
                    atoms.as_deref(),
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
                    atoms.as_deref(),
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
    let format = topology_format
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| infer_topology_format(topology_path, None));
    if matches!(format.as_str(), "pdb" | "gro") {
        Ok(format)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(
            "topology format must be pdb or gro",
        ))
    }
}

fn resolve_frame_edit_traj_format(
    traj_path: &str,
    traj_format: Option<&str>,
) -> PyResult<String> {
    let format = traj_format
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_ascii_lowercase)
        .unwrap_or_else(|| {
            Path::new(traj_path)
                .extension()
                .and_then(|value| value.to_str())
                .unwrap_or("")
                .to_ascii_lowercase()
        });
    if matches!(format.as_str(), "dcd" | "xtc" | "trr" | "pdb" | "pdbqt") {
        Ok(format)
    } else {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported trajectory format '{format}'; expected one of: dcd, xtc, trr, pdb, pdbqt"
        )))
    }
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
        "trr" => Ok(FrameEditOutputFormat::Trr),
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported output extension: .{other}"
        ))),
    }
}

fn load_frame_edit_system(topology_path: &str, topology_format: &str) -> PyResult<PySystem> {
    match topology_format {
        "pdb" => match PySystem::from_pdb(topology_path) {
            Ok(system) => Ok(system),
            Err(err) => {
                if err.to_string().to_ascii_lowercase().contains("invalid resid") {
                    PySystem::from_pdb_permissive(topology_path)
                } else {
                    Err(err)
                }
            }
        },
        "gro" => PySystem::from_gro(topology_path),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "topology format must be pdb or gro",
        )),
    }
}

fn open_frame_edit_trajectory(
    traj_path: &str,
    traj_format: &str,
    system: &PySystem,
    traj_length_scale: Option<f32>,
) -> PyResult<PyTrajectory> {
    match traj_format {
        "dcd" => PyTrajectory::open_dcd(traj_path, system, traj_length_scale),
        "xtc" => PyTrajectory::open_xtc(traj_path, system),
        "trr" => PyTrajectory::open_trr(traj_path, system),
        "pdb" => PyTrajectory::open_pdb(traj_path, system),
        "pdbqt" => PyTrajectory::open_pdbqt(traj_path, system),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported trajectory format '{traj_format}'"
        ))),
    }
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
    atoms: Option<&[StructureAtom]>,
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
        let atoms = atoms.expect("structure atoms required for structure outputs");
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
            atoms,
            single_output,
            outputs,
        );
    }

    let mut writer =
        FrameEditTrajectoryWriter::open(out_path, output_format, n_atoms, expected_frames)?;
    outputs.push(out_path.to_string_lossy().into_owned());
    let written = stream_frame_edit_frames(reader, begin, end, step, chunk_frames, preserve_trr_fields, |frame| {
        writer.write_frame(&frame)
    })?;
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
    atoms: &[StructureAtom],
    single_output: bool,
    outputs: &mut Vec<String>,
) -> TrajResult<usize> {
    stream_frame_edit_frames(reader, begin, end, step, chunk_frames, preserve_trr_fields, |frame| {
        let target = if single_output {
            out_path.to_path_buf()
        } else {
            frame_edit_series_path(out_path, frame.index)
        };
        write_frame_edit_structure_output(&target, output_format, atoms, &frame)?;
        outputs.push(target.to_string_lossy().into_owned());
        Ok(())
    })
}

fn write_frame_edit_outputs(
    out_path: &Path,
    output_format: FrameEditOutputFormat,
    atoms: Option<&[StructureAtom]>,
    n_atoms: usize,
    expected_frames: usize,
    frames: &[FrameEditFrame],
    outputs: &mut Vec<String>,
) -> TrajResult<()> {
    if output_format.is_structure() {
        let atoms = atoms.expect("structure atoms required for structure outputs");
        let single_output = expected_frames == 1;
        for frame in frames {
            let target = if single_output {
                out_path.to_path_buf()
            } else {
                frame_edit_series_path(out_path, frame.index)
            };
            write_frame_edit_structure_output(&target, output_format, atoms, frame)?;
            outputs.push(target.to_string_lossy().into_owned());
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
        TrajKind::Trr { reader } => reader.n_atoms(),
        TrajKind::Pdb { reader } => reader.n_atoms(),
    }
}

fn frame_edit_skip_frames(reader: &mut TrajKind, n_frames: usize) -> TrajResult<usize> {
    match reader {
        TrajKind::Dcd { reader } => reader.skip_frames(n_frames),
        TrajKind::Xtc { reader } => reader.skip_frames(n_frames),
        TrajKind::Trr { reader } => reader.skip_frames(n_frames),
        TrajKind::Pdb { reader } => reader.skip_frames(n_frames),
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
        TrajKind::Trr { reader } => reader.read_chunk(max_frames, builder),
        TrajKind::Pdb { reader } => reader.read_chunk(max_frames, builder),
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
        time_ps: chunk.time_ps.as_ref().and_then(|values| values.get(frame).copied()),
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
            Self::Xtc(writer) => writer.write_frame(&coords, frame.box_, frame.index, frame.time_ps),
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
            Self::Trr(writer) => writer.flush(),
        }
    }
}

fn write_frame_edit_structure_output(
    path: &Path,
    output_format: FrameEditOutputFormat,
    atoms: &[StructureAtom],
    frame: &FrameEditFrame,
) -> TrajResult<()> {
    let coords = frame_edit_coords3(&frame.coords);
    match output_format {
        FrameEditOutputFormat::Pdb => {
            write_pdb_structure(
                path.to_string_lossy().as_ref(),
                atoms,
                &coords,
                frame.box_,
                frame.index,
                frame.time_ps,
            )
        }
        FrameEditOutputFormat::Gro => {
            write_gro_structure(
                path.to_string_lossy().as_ref(),
                atoms,
                &coords,
                frame.box_,
                frame.index,
                frame.time_ps,
            )
        }
        FrameEditOutputFormat::Dcd | FrameEditOutputFormat::Xtc | FrameEditOutputFormat::Trr => {
            Err(TrajError::Invalid(
                "trajectory output requested from structure writer".into(),
            ))
        }
    }
}

fn frame_edit_coords3(coords: &[[f32; 4]]) -> Vec<[f32; 3]> {
    coords.iter().map(|value| [value[0], value[1], value[2]]).collect()
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
