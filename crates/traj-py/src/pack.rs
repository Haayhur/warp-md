use super::*;

#[pyclass]
struct PyPersistenceLengthPlan {
    plan: RefCell<PersistenceLengthPlan>,
}

#[pymethods]
impl PyPersistenceLengthPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = PersistenceLengthPlan::new(selection.selection.clone());
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<PyObject> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
        )?;
        match output {
            PlanOutput::Persistence(p) => persistence_to_py(py, p),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

fn pack_to_py<'py>(py: Python<'py>, cfg: &PackConfigInput) -> PyResult<PyObject> {
    let out = warp_pack::pack::run(cfg).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let n = out.atoms.len();
    let mut coords = Vec::with_capacity(n * 3);
    let mut name = Vec::with_capacity(n);
    let mut element = Vec::with_capacity(n);
    let mut resname = Vec::with_capacity(n);
    let mut resid = Vec::with_capacity(n);
    let mut chain = Vec::with_capacity(n);
    let mut segid = Vec::with_capacity(n);
    let mut charge = Vec::with_capacity(n);
    let mut mol_id = Vec::with_capacity(n);
    let mut record_kind = Vec::with_capacity(n);
    for atom in out.atoms.iter() {
        coords.push(atom.position.x);
        coords.push(atom.position.y);
        coords.push(atom.position.z);
        name.push(atom.name.clone());
        element.push(atom.element.clone());
        resname.push(atom.resname.clone());
        resid.push(atom.resid);
        chain.push(atom.chain.to_string());
        segid.push(atom.segid.clone());
        charge.push(atom.charge);
        mol_id.push(atom.mol_id);
        record_kind.push(match atom.record_kind {
            AtomRecordKind::Atom => "ATOM".to_string(),
            AtomRecordKind::HetAtom => "HETATM".to_string(),
        });
    }
    let coords = Array2::from_shape_vec((n, 3), coords)
        .map_err(|_| PyRuntimeError::new_err("failed to build coords array"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("coords", coords.into_pyarray_bound(py))?;
    dict.set_item("box", out.box_size)?;
    dict.set_item("name", name)?;
    dict.set_item("element", element)?;
    dict.set_item("resname", resname)?;
    dict.set_item("resid", resid)?;
    dict.set_item("chain", chain)?;
    dict.set_item("segid", segid)?;
    dict.set_item("charge", charge)?;
    dict.set_item("mol_id", mol_id)?;
    let bonds: Vec<(usize, usize)> = out.bonds.clone();
    dict.set_item("bonds", bonds)?;
    dict.set_item("record_kind", record_kind)?;
    dict.set_item("ter_after", out.ter_after.clone())?;
    Ok(dict.into_py(py))
}

pub(crate) fn get_attr_or_item<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if let Ok(val) = obj.getattr(name) {
        return Ok(val);
    }
    if let Ok(val) = obj.get_item(name) {
        return Ok(val);
    }
    Err(PyRuntimeError::new_err(format!(
        "pack result missing '{name}'"
    )))
}

pub(crate) fn get_attr_or_item_opt<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(val) = obj.getattr(name) {
        if val.is_none() {
            return Ok(None);
        }
        return Ok(Some(val));
    }
    if let Ok(val) = obj.get_item(name) {
        if val.is_none() {
            return Ok(None);
        }
        return Ok(Some(val));
    }
    Ok(None)
}

fn extract_coords(any: &Bound<'_, PyAny>) -> PyResult<Vec<Vec3>> {
    if let Ok(array) = any.extract::<numpy::PyReadonlyArray2<f32>>() {
        let view = array.as_array();
        let shape = view.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyRuntimeError::new_err("coords must be (N, 3) array"));
        }
        let mut coords = Vec::with_capacity(shape[0]);
        for row in view.outer_iter() {
            coords.push(Vec3::new(row[0], row[1], row[2]));
        }
        return Ok(coords);
    }
    if let Ok(vec) = any.extract::<Vec<[f32; 3]>>() {
        return Ok(vec
            .into_iter()
            .map(|v| Vec3::new(v[0], v[1], v[2]))
            .collect());
    }
    if let Ok(vec) = any.extract::<Vec<(f32, f32, f32)>>() {
        return Ok(vec.into_iter().map(|v| Vec3::new(v.0, v.1, v.2)).collect());
    }
    if let Ok(vec) = any.extract::<Vec<Vec<f32>>>() {
        let mut coords = Vec::with_capacity(vec.len());
        for row in vec {
            if row.len() != 3 {
                return Err(PyRuntimeError::new_err("coords rows must have 3 values"));
            }
            coords.push(Vec3::new(row[0], row[1], row[2]));
        }
        return Ok(coords);
    }
    Err(PyRuntimeError::new_err(
        "coords must be a numpy array or list of 3-float rows",
    ))
}

fn extract_box(any: &Bound<'_, PyAny>) -> PyResult<[f32; 3]> {
    if let Ok(arr) = any.extract::<[f32; 3]>() {
        return Ok(arr);
    }
    if let Ok(vec) = any.extract::<Vec<f32>>() {
        if vec.len() == 3 {
            return Ok([vec[0], vec[1], vec[2]]);
        }
    }
    Err(PyRuntimeError::new_err("box must be a 3-length sequence"))
}

fn build_pack_output(result: &Bound<'_, PyAny>) -> PyResult<PackOutput> {
    let coords_any = get_attr_or_item(result, "coords")?;
    let coords = extract_coords(&coords_any)?;
    let n = coords.len();

    let name: Vec<String> = get_attr_or_item(result, "name")?.extract()?;
    let element: Vec<String> = get_attr_or_item(result, "element")?.extract()?;
    let resname: Vec<String> = get_attr_or_item(result, "resname")?.extract()?;
    let resid: Vec<i32> = get_attr_or_item(result, "resid")?.extract()?;
    let chain_raw: Vec<String> = get_attr_or_item(result, "chain")?.extract()?;

    if name.len() != n
        || element.len() != n
        || resname.len() != n
        || resid.len() != n
        || chain_raw.len() != n
    {
        return Err(PyRuntimeError::new_err(
            "pack result arrays must have matching length",
        ));
    }

    let segid: Option<Vec<String>> = match get_attr_or_item_opt(result, "segid")? {
        Some(any) => Some(any.extract()?),
        None => None,
    };
    if let Some(segid_list) = &segid {
        if !segid_list.is_empty() && segid_list.len() != n {
            return Err(PyRuntimeError::new_err("segid length mismatch"));
        }
    }
    let charge: Option<Vec<f32>> = match get_attr_or_item_opt(result, "charge")? {
        Some(any) => Some(any.extract()?),
        None => None,
    };
    if let Some(charges) = &charge {
        if charges.len() != n {
            return Err(PyRuntimeError::new_err("charge length mismatch"));
        }
    }
    let mol_id: Option<Vec<i32>> = match get_attr_or_item_opt(result, "mol_id")? {
        Some(any) => Some(any.extract()?),
        None => None,
    };
    if let Some(mols) = &mol_id {
        if mols.len() != n {
            return Err(PyRuntimeError::new_err("mol_id length mismatch"));
        }
    }
    let record_kind: Option<Vec<String>> = match get_attr_or_item_opt(result, "record_kind")? {
        Some(any) => Some(any.extract()?),
        None => None,
    };
    if let Some(kinds) = &record_kind {
        if !kinds.is_empty() && kinds.len() != n {
            return Err(PyRuntimeError::new_err("record_kind length mismatch"));
        }
    }

    let mut atoms = Vec::with_capacity(n);
    for i in 0..n {
        let chain = chain_raw[i].chars().next().unwrap_or('A');
        let segid_val = segid
            .as_ref()
            .and_then(|s| s.get(i))
            .cloned()
            .unwrap_or_default();
        let charge_val = charge
            .as_ref()
            .and_then(|c| c.get(i))
            .copied()
            .unwrap_or(0.0);
        let mol_val = mol_id.as_ref().and_then(|m| m.get(i)).copied().unwrap_or(1);
        let kind = record_kind
            .as_ref()
            .and_then(|k| k.get(i))
            .map(|s| s.to_ascii_uppercase())
            .map(|s| {
                if s.starts_with("HET") {
                    AtomRecordKind::HetAtom
                } else {
                    AtomRecordKind::Atom
                }
            })
            .unwrap_or(AtomRecordKind::Atom);
        atoms.push(AtomRecord {
            record_kind: kind,
            name: name[i].clone(),
            element: element[i].clone(),
            resname: resname[i].clone(),
            resid: resid[i],
            chain,
            segid: segid_val,
            charge: charge_val,
            position: coords[i],
            mol_id: mol_val,
            pdb_metadata: None,
        });
    }

    let bonds = match get_attr_or_item_opt(result, "bonds")? {
        Some(any) => {
            if let Ok(list) = any.extract::<Vec<(usize, usize)>>() {
                list
            } else if let Ok(list) = any.extract::<Vec<Vec<usize>>>() {
                let mut out = Vec::with_capacity(list.len());
                for pair in list {
                    if pair.len() != 2 {
                        return Err(PyRuntimeError::new_err("bond entries must have 2 indices"));
                    }
                    out.push((pair[0], pair[1]));
                }
                out
            } else {
                Vec::new()
            }
        }
        None => Vec::new(),
    };

    let ter_after = match get_attr_or_item_opt(result, "ter_after")? {
        Some(any) => any.extract::<Vec<usize>>()?,
        None => Vec::new(),
    };

    let box_size = match get_attr_or_item_opt(result, "box")? {
        Some(any) => extract_box(&any)?,
        None => [0.0, 0.0, 0.0],
    };

    Ok(PackOutput {
        atoms,
        bonds,
        box_size,
        ter_after,
        box_vectors: None,
    })
}

#[pyfunction]
fn pack_from_json<'py>(py: Python<'py>, json: &str) -> PyResult<PyObject> {
    let cfg: PackConfigInput = serde_json::from_str(json)
        .map_err(|e| PyRuntimeError::new_err(format!("pack config parse error: {e}")))?;
    pack_to_py(py, &cfg)
}

#[pyfunction]
fn pack_from_inp<'py>(py: Python<'py>, path: &str) -> PyResult<PyObject> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("pack inp read error: {e}")))?;
    let cfg = warp_pack::inp::parse_packmol_inp(&content)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    pack_to_py(py, &cfg)
}

#[pyfunction]
fn pack_config_from_inp<'py>(py: Python<'py>, path: &str) -> PyResult<PyObject> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| PyRuntimeError::new_err(format!("pack inp read error: {e}")))?;
    let cfg = warp_pack::inp::parse_packmol_inp(&content)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let json = serde_json::to_string(&cfg)
        .map_err(|e| PyRuntimeError::new_err(format!("pack config serialize error: {e}")))?;
    let json_mod = PyModule::import_bound(py, "json")?;
    let dict = json_mod.call_method1("loads", (json,))?;
    Ok(dict.into_py(py))
}

#[pyfunction]
fn pack_config_normalize_json<'py>(py: Python<'py>, json: &str) -> PyResult<PyObject> {
    let cfg: PackConfigInput = serde_json::from_str(json)
        .map_err(|e| PyRuntimeError::new_err(format!("pack config parse error: {e}")))?;
    let normalized = cfg
        .normalized()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let text = serde_json::to_string(&normalized)
        .map_err(|e| PyRuntimeError::new_err(format!("pack config serialize error: {e}")))?;
    json_string_to_py(py, text)
}

fn json_string_to_py<'py>(py: Python<'py>, text: String) -> PyResult<PyObject> {
    let json_mod = PyModule::import_bound(py, "json")?;
    let value = json_mod.call_method1("loads", (text,))?;
    Ok(value.into_py(py))
}

pub(crate) fn json_value_to_py<'py>(
    py: Python<'py>,
    value: &serde_json::Value,
) -> PyResult<PyObject> {
    let text = serde_json::to_string(value)
        .map_err(|e| PyRuntimeError::new_err(format!("json serialize error: {e}")))?;
    json_string_to_py(py, text)
}

#[pyfunction]
#[pyo3(signature = (kind="request"))]
fn build_agent_schema<'py>(py: Python<'py>, kind: &str) -> PyResult<PyObject> {
    let text = warp_build::schema_json(kind).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_string_to_py(py, text)
}

#[pyfunction]
#[pyo3(signature = (mode="random_walk", bundle_path="source.bundle.json"))]
fn build_agent_example<'py>(py: Python<'py>, mode: &str, bundle_path: &str) -> PyResult<PyObject> {
    let value = warp_build::example_request_for_bundle(mode, bundle_path);
    json_value_to_py(py, &value)
}

#[pyfunction]
fn build_agent_example_bundle<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = warp_build::example_bundle();
    json_value_to_py(py, &value)
}

#[pyfunction]
fn build_agent_write_example_bundle<'py>(py: Python<'py>, path: &str) -> PyResult<PyObject> {
    let value = warp_build::write_example_bundle(path)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn build_agent_capabilities<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = warp_build::capabilities();
    json_value_to_py(py, &value)
}

#[pyfunction]
fn build_agent_inspect_source<'py>(py: Python<'py>, path: &str) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_build::inspect_source_json(path);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
fn build_agent_validate<'py>(py: Python<'py>, json: &str) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_build::validate_request_json(json);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (json, stream_ndjson=false))]
fn build_agent_run<'py>(
    py: Python<'py>,
    json: &str,
    stream_ndjson: bool,
) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_build::run_request_json(json, stream_ndjson);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (kind="request"))]
fn pack_agent_schema<'py>(py: Python<'py>, kind: &str) -> PyResult<PyObject> {
    let text =
        warp_pack::agent::schema_json(kind).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_string_to_py(py, text)
}

#[pyfunction]
#[pyo3(signature = (mode="solute_solvate"))]
fn pack_agent_example<'py>(py: Python<'py>, mode: &str) -> PyResult<PyObject> {
    let value = warp_pack::agent::example_request(mode)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn pack_agent_capabilities<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = warp_pack::agent::capabilities();
    json_value_to_py(py, &value)
}

#[pyfunction]
fn pack_resolve_chemistry<'py>(py: Python<'py>, json: &str) -> PyResult<PyObject> {
    let value = warp_pack::agent::resolve_chemistry_json(json)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn pack_agent_validate<'py>(py: Python<'py>, json: &str) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_pack::agent::validate_request_json(json);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (json, stream_ndjson=false))]
fn pack_agent_run<'py>(
    py: Python<'py>,
    json: &str,
    stream_ndjson: bool,
) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_pack::agent::run_request_json(json, stream_ndjson);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (kind="request"))]
fn cg_agent_schema<'py>(py: Python<'py>, kind: &str) -> PyResult<PyObject> {
    let text =
        warp_cg::agent::schema_json(kind).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_string_to_py(py, text)
}

#[pyfunction]
fn cg_agent_example<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = warp_cg::agent::example_request();
    json_value_to_py(py, &value)
}

#[pyfunction]
fn cg_agent_capabilities<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = warp_cg::agent::capabilities();
    json_value_to_py(py, &value)
}

#[pyfunction]
fn cg_agent_validate<'py>(py: Python<'py>, json: &str) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_cg::agent::validate_request_json(json);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (json, stream_ndjson=false))]
fn cg_agent_run<'py>(
    py: Python<'py>,
    json: &str,
    stream_ndjson: bool,
) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_cg::agent::run_request_json(json, stream_ndjson);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (kind="request"))]
fn cg_build_schema<'py>(py: Python<'py>, kind: &str) -> PyResult<PyObject> {
    let text = warp_cg::build_contract::schema_json(kind)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_string_to_py(py, text)
}

#[pyfunction]
fn cg_build_example<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = warp_cg::build_contract::example_request();
    json_value_to_py(py, &value)
}

#[pyfunction]
fn cg_build_capabilities<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = warp_cg::build_contract::capabilities();
    json_value_to_py(py, &value)
}

#[pyfunction]
fn cg_build_validate<'py>(py: Python<'py>, json: &str) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_cg::build_contract::validate_request_json(json);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (json, stream_ndjson=false))]
fn cg_build_run<'py>(
    py: Python<'py>,
    json: &str,
    stream_ndjson: bool,
) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_cg::build_contract::run_request_json(json, stream_ndjson);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (kind="request"))]
fn cg_simulate_schema<'py>(py: Python<'py>, kind: &str) -> PyResult<PyObject> {
    let text = warp_cg::simulate_contract::schema_json(kind)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_string_to_py(py, text)
}

#[pyfunction]
#[pyo3(signature = (engine="gromacs"))]
fn cg_simulate_example<'py>(py: Python<'py>, engine: &str) -> PyResult<PyObject> {
    let value = warp_cg::simulate_contract::example_request(engine)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    json_value_to_py(py, &value)
}

#[pyfunction]
fn cg_simulate_capabilities<'py>(py: Python<'py>) -> PyResult<PyObject> {
    let value = warp_cg::simulate_contract::capabilities();
    json_value_to_py(py, &value)
}

#[pyfunction]
fn cg_simulate_validate<'py>(py: Python<'py>, json: &str) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_cg::simulate_contract::validate_request_json(json);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (json, engine=None))]
fn cg_simulate_plan<'py>(
    py: Python<'py>,
    json: &str,
    engine: Option<&str>,
) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_cg::simulate_contract::plan_request_json(json, engine);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
fn cg_simulate_status<'py>(py: Python<'py>, run_dir: &str) -> PyResult<(i32, PyObject)> {
    let (exit_code, value) = warp_cg::simulate_contract::status_json(run_dir);
    Ok((exit_code, json_value_to_py(py, &value)?))
}

#[pyfunction]
#[pyo3(signature = (data0, data1, mode="distance"))]
fn crank_delta<'py>(
    py: Python<'py>,
    data0: PyReadonlyArrayDyn<'py, f64>,
    data1: PyReadonlyArrayDyn<'py, f64>,
    mode: &str,
) -> PyResult<&'py PyArrayDyn<f64>> {
    let x = data0.as_array();
    let y = data1.as_array();
    if x.shape() != y.shape() {
        return Err(PyRuntimeError::new_err(
            "crank requires data0 and data1 with the same shape",
        ));
    }
    let mut out = Vec::with_capacity(x.len());
    let mode = mode.to_ascii_lowercase();
    match mode.trim() {
        "distance" => {
            for (&a, &b) in x.iter().zip(y.iter()) {
                out.push(b - a);
            }
        }
        "angle" => {
            for (&a, &b) in x.iter().zip(y.iter()) {
                let delta = b - a;
                out.push((delta + 180.0).rem_euclid(360.0) - 180.0);
            }
        }
        _ => {
            return Err(PyRuntimeError::new_err(
                "crank mode must be 'distance' or 'angle'",
            ))
        }
    }
    let arr = numpy::ndarray::ArrayD::from_shape_vec(x.raw_dim(), out)
        .map_err(|_| PyRuntimeError::new_err("invalid crank output shape"))?;
    Ok(arr.into_pyarray_bound(py).into_gil_ref())
}

#[pyfunction]
#[pyo3(signature = (time, data, fit_component="p2", fit_window=None))]
fn rotdif_fit<'py>(
    _py: Python<'py>,
    time: PyReadonlyArrayDyn<'py, f64>,
    data: PyReadonlyArrayDyn<'py, f64>,
    fit_component: &str,
    fit_window: Option<(f64, f64)>,
) -> PyResult<(f64, f64, f64, f64, usize)> {
    let t = time
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("time must be a 1D array"))?;
    let y = data
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix2>()
        .map_err(|_| PyRuntimeError::new_err("data must be a 2D array"))?;
    let n_rows = y.nrows();
    let n_cols = y.ncols();
    if t.len() != n_rows {
        return Err(PyRuntimeError::new_err(
            "time and data row counts must match",
        ));
    }

    let col = match fit_component {
        "p1" => 0usize,
        "p2" => 1usize,
        _ => {
            return Err(PyRuntimeError::new_err(
                "fit_component must be 'p1' or 'p2'",
            ))
        }
    };
    if n_cols <= col {
        return Err(PyRuntimeError::new_err(format!(
            "data must have at least {} columns",
            col + 1
        )));
    }

    let window = fit_window.map(|(a, b)| if b < a { (b, a) } else { (a, b) });
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..n_rows {
        let ti = t[i];
        let yi = y[(i, col)];
        if !ti.is_finite() || !yi.is_finite() || ti <= 0.0 || yi <= 0.0 {
            continue;
        }
        if let Some((lo, hi)) = window {
            if ti < lo || ti > hi {
                continue;
            }
        }
        xs.push(ti);
        ys.push(yi.ln());
    }

    if xs.len() < 2 {
        return Ok((0.0, f64::INFINITY, 0.0, 0.0, xs.len()));
    }

    let n = xs.len() as f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    for (&x, &yln) in xs.iter().zip(ys.iter()) {
        sum_x += x;
        sum_y += yln;
        sum_xx += x * x;
        sum_xy += x * yln;
    }
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() <= f64::EPSILON {
        return Ok((0.0, f64::INFINITY, 0.0, 0.0, xs.len()));
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    let d_denom = if col == 0 { 2.0 } else { 6.0 };
    let d_rot = (-slope / d_denom).max(0.0);
    let tau = if slope >= 0.0 {
        f64::INFINITY
    } else {
        -1.0 / slope
    };
    Ok((d_rot, tau, slope, intercept, xs.len()))
}

#[pyfunction]
#[pyo3(signature = (energy_sw_direct, energy_ww_direct, direct_sw_total, direct_ww_total, direct_sw_frame, direct_ww_frame, pme_sw_frame, pme_ww_frame, frame_offsets, frame_cells, frame_sw, frame_ww))]
fn gist_apply_pme_scaling<'py>(
    py: Python<'py>,
    energy_sw_direct: PyReadonlyArrayDyn<'py, f64>,
    energy_ww_direct: PyReadonlyArrayDyn<'py, f64>,
    direct_sw_total: f64,
    direct_ww_total: f64,
    direct_sw_frame: PyReadonlyArrayDyn<'py, f64>,
    direct_ww_frame: PyReadonlyArrayDyn<'py, f64>,
    pme_sw_frame: PyReadonlyArrayDyn<'py, f64>,
    pme_ww_frame: PyReadonlyArrayDyn<'py, f64>,
    frame_offsets: PyReadonlyArrayDyn<'py, u64>,
    frame_cells: PyReadonlyArrayDyn<'py, u32>,
    frame_sw: PyReadonlyArrayDyn<'py, f64>,
    frame_ww: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(&'py PyArrayDyn<f64>, &'py PyArrayDyn<f64>)> {
    let sw_direct = energy_sw_direct.as_array();
    let ww_direct = energy_ww_direct.as_array();
    if sw_direct.shape() != ww_direct.shape() {
        return Err(PyRuntimeError::new_err(
            "energy_sw_direct and energy_ww_direct must have identical shape",
        ));
    }
    let grid_dim = sw_direct.raw_dim();
    let n_cells = sw_direct.len();

    let direct_sw = direct_sw_frame
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("direct_sw_frame must be a 1D array"))?;
    let direct_ww = direct_ww_frame
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("direct_ww_frame must be a 1D array"))?;
    let pme_sw = pme_sw_frame
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("pme_sw_frame must be a 1D array"))?;
    let pme_ww = pme_ww_frame
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("pme_ww_frame must be a 1D array"))?;
    let offsets = frame_offsets
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("frame_offsets must be a 1D array"))?;
    let cells = frame_cells
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("frame_cells must be a 1D array"))?;
    let vals_sw = frame_sw
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("frame_sw must be a 1D array"))?;
    let vals_ww = frame_ww
        .as_array()
        .into_dimensionality::<numpy::ndarray::Ix1>()
        .map_err(|_| PyRuntimeError::new_err("frame_ww must be a 1D array"))?;

    let n_direct_frames = direct_sw.len();
    if direct_ww.len() != n_direct_frames {
        return Err(PyRuntimeError::new_err(
            "direct_sw_frame and direct_ww_frame must have identical lengths",
        ));
    }
    let n_pme_frames = pme_sw.len();
    if pme_ww.len() != n_pme_frames {
        return Err(PyRuntimeError::new_err(
            "pme_sw_frame and pme_ww_frame must have identical lengths",
        ));
    }
    let use_sparse = offsets.len() == n_pme_frames + 1;
    let (out_sw, out_ww) = if use_sparse {
        if n_direct_frames != n_pme_frames {
            return Err(PyRuntimeError::new_err(
                "sparse scaling requires direct and pme frame arrays with identical lengths",
            ));
        }
        let sparse_len = offsets[offsets.len() - 1] as usize;
        if cells.len() != sparse_len || vals_sw.len() != sparse_len || vals_ww.len() != sparse_len {
            return Err(PyRuntimeError::new_err(
                "frame sparse arrays must match frame_offsets end",
            ));
        }

        let mut out_sw = vec![0.0f64; n_cells];
        let mut out_ww = vec![0.0f64; n_cells];
        for frame_idx in 0..n_pme_frames {
            let start = offsets[frame_idx] as usize;
            let end = offsets[frame_idx + 1] as usize;
            if end < start || end > sparse_len {
                return Err(PyRuntimeError::new_err(
                    "frame_offsets must be non-decreasing and within sparse range",
                ));
            }
            let sw_scale = if direct_sw[frame_idx] != 0.0 {
                pme_sw[frame_idx] / direct_sw[frame_idx]
            } else {
                0.0
            };
            let ww_scale = if direct_ww[frame_idx] != 0.0 {
                pme_ww[frame_idx] / direct_ww[frame_idx]
            } else {
                0.0
            };
            for k in start..end {
                let cell = cells[k] as usize;
                if cell >= n_cells {
                    return Err(PyRuntimeError::new_err(
                        "frame cell index exceeds grid size",
                    ));
                }
                out_sw[cell] += vals_sw[k] * sw_scale;
                out_ww[cell] += vals_ww[k] * ww_scale;
            }
        }
        (out_sw, out_ww)
    } else {
        let sw_scale = if direct_sw_total != 0.0 {
            pme_sw.sum() / direct_sw_total
        } else {
            0.0
        };
        let ww_scale = if direct_ww_total != 0.0 {
            pme_ww.sum() / direct_ww_total
        } else {
            0.0
        };
        (
            sw_direct.iter().map(|v| *v * sw_scale).collect(),
            ww_direct.iter().map(|v| *v * ww_scale).collect(),
        )
    };

    let sw = numpy::ndarray::ArrayD::from_shape_vec(grid_dim.clone(), out_sw)
        .map_err(|_| PyRuntimeError::new_err("invalid sw output shape"))?;
    let ww = numpy::ndarray::ArrayD::from_shape_vec(grid_dim, out_ww)
        .map_err(|_| PyRuntimeError::new_err("invalid ww output shape"))?;
    Ok((
        sw.into_pyarray_bound(py).into_gil_ref(),
        ww.into_pyarray_bound(py).into_gil_ref(),
    ))
}

#[pyfunction]
fn pack_write_output(
    py: Python<'_>,
    result: &Bound<'_, PyAny>,
    format: &str,
    path: &str,
    scale: f32,
    add_box_sides: bool,
    box_sides_fix: f32,
    write_conect: bool,
    hexadecimal_indices: bool,
) -> PyResult<PyObject> {
    let out = build_pack_output(result)?;
    let spec = OutputSpec {
        path: path.to_string(),
        format: format.to_string(),
        scale: Some(scale),
    };
    let written = warp_structure::io::write_output(
        &out,
        &spec,
        add_box_sides,
        box_sides_fix,
        write_conect,
        hexadecimal_indices,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("path", written.path)?;
    dict.set_item("format", written.format)?;
    dict.set_item("fallback_applied", written.fallback_applied)?;
    Ok(dict.into_py(py))
}

fn pep_emit(
    struc: &warp_pep::residue::Structure,
    output: Option<&str>,
    format: Option<&str>,
) -> Result<(), String> {
    match output {
        Some(path) => warp_pep::convert::write_structure(struc, path, format),
        None => warp_pep::convert::write_structure_stdout(struc, format.unwrap_or("pdb")),
    }
}

fn pep_parse_preset(preset: Option<&str>) -> Result<Option<warp_pep::builder::RamaPreset>, String> {
    match preset {
        Some(p) => Ok(Some(
            warp_pep::builder::RamaPreset::from_str(p)
                .ok_or_else(|| format!("unknown preset '{p}'"))?,
        )),
        None => Ok(None),
    }
}

#[pyfunction]
#[pyo3(signature = (sequence=None, three_letter=None, json_path=None, output=None, format=None, oxt=false, preset=None, phi=None, psi=None, omega=None, detect_ss=false))]
fn pep_build(
    sequence: Option<String>,
    three_letter: Option<String>,
    json_path: Option<String>,
    output: Option<String>,
    format: Option<String>,
    oxt: bool,
    preset: Option<String>,
    phi: Option<Vec<f64>>,
    psi: Option<Vec<f64>>,
    omega: Option<Vec<f64>>,
    detect_ss: bool,
) -> PyResult<()> {
    if let Some(path) = json_path.as_deref() {
        // JSON mode is self-contained; preserve warp-pep CLI behavior:
        // spec output/format take precedence over explicit overrides.
        let spec = warp_pep::json_spec::BuildSpec::from_file(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let struc = spec
            .execute()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out = spec.output.as_deref().or(output.as_deref());
        let fmt = spec.format.as_deref().or(format.as_deref());
        return pep_emit(&struc, out, fmt).map_err(|e| PyRuntimeError::new_err(e.to_string()));
    }

    let rama =
        pep_parse_preset(preset.as_deref()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    if let Some(tl) = three_letter.as_deref() {
        let specs = warp_pep::builder::parse_three_letter_sequence(&tl.to_uppercase())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let mut struc = if let Some(p) = rama {
            warp_pep::builder::make_preset_structure_from_specs(&specs, p)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
        } else {
            match (&phi, &psi) {
                (Some(phi_v), Some(psi_v)) => warp_pep::builder::make_structure_from_specs(
                    &specs,
                    phi_v,
                    psi_v,
                    omega.as_deref(),
                )
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                (None, None) => warp_pep::builder::make_extended_structure_from_specs(&specs)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                _ => {
                    return Err(PyRuntimeError::new_err(
                        "must provide both phi and psi or neither",
                    ))
                }
            }
        };
        if oxt {
            warp_pep::builder::add_terminal_oxt(&mut struc);
        }
        if detect_ss {
            warp_pep::disulfide::detect_disulfides(&mut struc);
        }
        return pep_emit(&struc, output.as_deref(), format.as_deref())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()));
    }

    let seq = sequence
        .as_deref()
        .ok_or_else(|| PyRuntimeError::new_err("provide sequence, three_letter, or json_path"))?;
    let mut struc = if let Some(p) = rama {
        warp_pep::builder::make_preset_structure(seq, p)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else {
        match (&phi, &psi) {
            (Some(phi_v), Some(psi_v)) => {
                warp_pep::builder::make_structure(seq, phi_v, psi_v, omega.as_deref())
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
            }
            (None, None) => warp_pep::builder::make_extended_structure(seq)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "must provide both phi and psi or neither",
                ))
            }
        }
    };

    if oxt {
        warp_pep::builder::add_terminal_oxt(&mut struc);
    }
    if detect_ss {
        warp_pep::disulfide::detect_disulfides(&mut struc);
    }

    pep_emit(&struc, output.as_deref(), format.as_deref())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (input_path=None, sequence=None, three_letter=None, mutations=None, output=None, format=None, oxt=false, preset=None, detect_ss=false))]
fn pep_mutate(
    input_path: Option<String>,
    sequence: Option<String>,
    three_letter: Option<String>,
    mutations: Option<Vec<String>>,
    output: Option<String>,
    format: Option<String>,
    oxt: bool,
    preset: Option<String>,
    detect_ss: bool,
) -> PyResult<()> {
    let mutation_specs = mutations.unwrap_or_default();
    if mutation_specs.is_empty() {
        return Err(PyRuntimeError::new_err(
            "must provide at least one mutation spec",
        ));
    }

    let rama =
        pep_parse_preset(preset.as_deref()).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    let mut struc = if let Some(path) = input_path.as_deref() {
        warp_pep::convert::read_structure(path)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?
    } else if let Some(tl) = three_letter.as_deref() {
        let specs = warp_pep::builder::parse_three_letter_sequence(&tl.to_uppercase())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        match rama {
            Some(p) => warp_pep::builder::make_preset_structure_from_specs(&specs, p)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            None => warp_pep::builder::make_extended_structure_from_specs(&specs)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        }
    } else if let Some(seq) = sequence.as_deref() {
        match rama {
            Some(p) => warp_pep::builder::make_preset_structure(seq, p)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
            None => warp_pep::builder::make_extended_structure(seq)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
        }
    } else {
        return Err(PyRuntimeError::new_err(
            "provide input_path, sequence, or three_letter",
        ));
    };

    for spec in mutation_specs {
        let (from, pos, to) = warp_pep::mutation::parse_mutation_spec(&spec)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        warp_pep::mutation::mutate_residue_checked(&mut struc, Some(from), pos, to)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }

    if oxt {
        warp_pep::builder::add_terminal_oxt(&mut struc);
    }
    if detect_ss {
        warp_pep::disulfide::detect_disulfides(&mut struc);
    }

    pep_emit(&struc, output.as_deref(), format.as_deref())
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPersistenceLengthPlan>()?;
    m.add_function(wrap_pyfunction!(pack_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(pack_from_inp, m)?)?;
    m.add_function(wrap_pyfunction!(pack_config_from_inp, m)?)?;
    m.add_function(wrap_pyfunction!(pack_config_normalize_json, m)?)?;
    m.add_function(wrap_pyfunction!(build_agent_schema, m)?)?;
    m.add_function(wrap_pyfunction!(build_agent_example, m)?)?;
    m.add_function(wrap_pyfunction!(build_agent_example_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(build_agent_write_example_bundle, m)?)?;
    m.add_function(wrap_pyfunction!(build_agent_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(build_agent_inspect_source, m)?)?;
    m.add_function(wrap_pyfunction!(build_agent_validate, m)?)?;
    m.add_function(wrap_pyfunction!(build_agent_run, m)?)?;
    m.add_function(wrap_pyfunction!(pack_agent_schema, m)?)?;
    m.add_function(wrap_pyfunction!(pack_agent_example, m)?)?;
    m.add_function(wrap_pyfunction!(pack_agent_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(pack_resolve_chemistry, m)?)?;
    m.add_function(wrap_pyfunction!(pack_agent_validate, m)?)?;
    m.add_function(wrap_pyfunction!(pack_agent_run, m)?)?;
    m.add_function(wrap_pyfunction!(cg_agent_schema, m)?)?;
    m.add_function(wrap_pyfunction!(cg_agent_example, m)?)?;
    m.add_function(wrap_pyfunction!(cg_agent_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(cg_agent_validate, m)?)?;
    m.add_function(wrap_pyfunction!(cg_agent_run, m)?)?;
    m.add_function(wrap_pyfunction!(cg_build_schema, m)?)?;
    m.add_function(wrap_pyfunction!(cg_build_example, m)?)?;
    m.add_function(wrap_pyfunction!(cg_build_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(cg_build_validate, m)?)?;
    m.add_function(wrap_pyfunction!(cg_build_run, m)?)?;
    m.add_function(wrap_pyfunction!(cg_simulate_schema, m)?)?;
    m.add_function(wrap_pyfunction!(cg_simulate_example, m)?)?;
    m.add_function(wrap_pyfunction!(cg_simulate_capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(cg_simulate_validate, m)?)?;
    m.add_function(wrap_pyfunction!(cg_simulate_plan, m)?)?;
    m.add_function(wrap_pyfunction!(cg_simulate_status, m)?)?;
    m.add_function(wrap_pyfunction!(crank_delta, m)?)?;
    m.add_function(wrap_pyfunction!(rotdif_fit, m)?)?;
    m.add_function(wrap_pyfunction!(gist_apply_pme_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(pack_write_output, m)?)?;
    m.add_function(wrap_pyfunction!(pep_build, m)?)?;
    m.add_function(wrap_pyfunction!(pep_mutate, m)?)?;
    Ok(())
}

pub(crate) fn to_py_err(err: TrajError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

pub(crate) fn run_plan<P: Plan>(
    plan: &mut P,
    traj: &mut TrajKind,
    system: &System,
    chunk_frames: Option<usize>,
    device: &str,
) -> PyResult<PlanOutput> {
    let is_cuda = device.trim().to_ascii_lowercase().starts_with("cuda");
    let preferred_selection = if is_cuda {
        None
    } else {
        plan.preferred_selection_hint(system)
            .map(|sel| sel.to_vec())
    };
    let preferred_n_atoms = if is_cuda {
        None
    } else {
        plan.preferred_n_atoms_hint(system)
            .or_else(|| preferred_selection.as_ref().map(|sel| sel.len()))
    };
    let chunk_frames = resolve_chunk_frames(
        traj,
        chunk_frames,
        preferred_n_atoms,
        preferred_selection.as_deref(),
    )?;
    let mut exec = Executor::new(system.clone())
        .with_device_spec(device)
        .map_err(to_py_err)?;
    exec = exec.with_chunk_frames(chunk_frames);
    let output = match traj {
        TrajKind::Dcd { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::Xtc { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::Gro { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::G96 { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::Cpt { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::H5md { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::Tng { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::Trr { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::Pdb { reader, .. } => exec.run_plan(plan, reader),
        TrajKind::Memory { reader, .. } => exec.run_plan(plan, reader),
    }
    .map_err(to_py_err)?;
    Ok(output)
}

pub(crate) fn reset_traj(traj: &mut TrajKind) -> TrajResult<()> {
    match traj {
        TrajKind::Dcd { reader, .. } => reader.reset(),
        TrajKind::Xtc { reader, .. } => reader.reset(),
        TrajKind::Gro { reader, .. } => reader.reset(),
        TrajKind::G96 { reader, .. } => reader.reset(),
        TrajKind::Cpt { reader, .. } => reader.reset(),
        TrajKind::H5md { reader, .. } => reader.reset(),
        TrajKind::Tng { reader, .. } => reader.reset(),
        TrajKind::Trr { reader, .. } => reader.reset(),
        TrajKind::Pdb { reader, .. } => {
            reader.reset();
            Ok(())
        }
        TrajKind::Memory { reader, .. } => {
            reader.reset();
            Ok(())
        }
    }
}

pub(crate) fn run_plan_with_frame_subset<P: Plan>(
    plan: &mut P,
    traj: &mut TrajKind,
    system: &System,
    chunk_frames: Option<usize>,
    device: &str,
    frame_indices: Option<Vec<i64>>,
) -> PyResult<PlanOutput> {
    let is_cuda = device.trim().to_ascii_lowercase().starts_with("cuda");
    let preferred_selection = if is_cuda {
        None
    } else {
        plan.preferred_selection_hint(system)
            .map(|sel| sel.to_vec())
    };
    let preferred_n_atoms = if is_cuda {
        None
    } else {
        plan.preferred_n_atoms_hint(system)
            .or_else(|| preferred_selection.as_ref().map(|sel| sel.len()))
    };
    let chunk_frames = resolve_chunk_frames(
        traj,
        chunk_frames,
        preferred_n_atoms,
        preferred_selection.as_deref(),
    )?;
    let Some(raw_indices) = frame_indices else {
        return run_plan(plan, traj, system, Some(chunk_frames), device);
    };
    let max_frames = chunk_frames;
    let has_negative = raw_indices.iter().any(|&idx| idx < 0);

    let source_indices = if has_negative {
        let n_frames = if let Some(hint) = traj_n_frames_hint(traj) {
            hint
        } else {
            let counted = match traj {
                TrajKind::Dcd { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::Xtc { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::Gro { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::G96 { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::Cpt { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::H5md { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::Tng { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::Trr { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::Pdb { reader, .. } => traj_engine::count_frames(reader, max_frames),
                TrajKind::Memory { reader, .. } => traj_engine::count_frames(reader, max_frames),
            }
            .map_err(to_py_err)?;
            reset_traj(traj).map_err(to_py_err)?;
            counted
        };
        traj_engine::normalize_frame_indices(raw_indices, n_frames)
    } else {
        raw_indices
            .into_iter()
            .filter_map(|idx| usize::try_from(idx).ok())
            .collect()
    };

    let mut exec = Executor::new(system.clone())
        .with_device_spec(device)
        .map_err(to_py_err)?;
    exec = exec.with_chunk_frames(chunk_frames);
    let output = match traj {
        TrajKind::Dcd { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::Xtc { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::Gro { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::G96 { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::Cpt { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::H5md { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::Tng { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::Trr { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::Pdb { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
        TrajKind::Memory { reader, .. } => {
            exec.run_plan_on_selected_frames(plan, reader, &source_indices)
        }
    }
    .map_err(to_py_err)?;
    Ok(output)
}

const AUTO_CHUNK_MIN: usize = 16;
const AUTO_CHUNK_MAX: usize = 256;
const AUTO_CHUNK_TARGET_BYTES: usize = 8 * 1024 * 1024;
const AUTO_TUNE_MIN_FRAMES: usize = 2048;
const AUTO_TUNE_SAMPLE_FRAMES: usize = 128;

fn resolve_chunk_frames(
    traj: &mut TrajKind,
    chunk_frames: Option<usize>,
    preferred_n_atoms: Option<usize>,
    preferred_selection: Option<&[u32]>,
) -> PyResult<usize> {
    if let Some(frames) = chunk_frames {
        return Ok(frames.max(1));
    }
    let traj_atoms = traj_n_atoms(traj);
    let n_atoms = preferred_n_atoms
        .filter(|&n| n > 0)
        .map(|n| n.min(traj_atoms))
        .unwrap_or(traj_atoms);
    let heuristic = heuristic_chunk_frames(n_atoms);
    if !matches!(traj, TrajKind::Dcd { .. }) {
        return Ok(heuristic);
    }

    let n_frames_hint = traj_n_frames_hint(traj).unwrap_or(0);
    if n_frames_hint < AUTO_TUNE_MIN_FRAMES {
        return Ok(heuristic);
    }

    let candidates = chunk_candidates(heuristic);
    let sample_frames = AUTO_TUNE_SAMPLE_FRAMES.min(n_frames_hint);

    let mut best_chunk = heuristic;
    let mut best_fps = 0.0f64;
    for chunk in candidates.into_iter() {
        let chunk = chunk.clamp(AUTO_CHUNK_MIN, AUTO_CHUNK_MAX);
        let fps = bench_chunk_frames(traj, chunk, sample_frames, preferred_selection)?;
        if fps > best_fps {
            best_fps = fps;
            best_chunk = chunk;
        }
        reset_traj(traj).map_err(to_py_err)?;
    }
    Ok(best_chunk)
}

fn chunk_candidates(heuristic: usize) -> Vec<usize> {
    let h = heuristic.clamp(AUTO_CHUNK_MIN, AUTO_CHUNK_MAX);
    let lower = (h / 2).max(AUTO_CHUNK_MIN);
    let upper = h.saturating_mul(2).clamp(AUTO_CHUNK_MIN, AUTO_CHUNK_MAX);
    let mut candidates = vec![lower, h, upper];
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

fn traj_n_atoms(traj: &TrajKind) -> usize {
    match traj {
        TrajKind::Dcd { reader, .. } => reader.n_atoms(),
        TrajKind::Xtc { reader, .. } => reader.n_atoms(),
        TrajKind::Gro { reader, .. } => reader.n_atoms(),
        TrajKind::G96 { reader, .. } => reader.n_atoms(),
        TrajKind::Cpt { reader, .. } => reader.n_atoms(),
        TrajKind::H5md { reader, .. } => reader.n_atoms(),
        TrajKind::Tng { reader, .. } => reader.n_atoms(),
        TrajKind::Trr { reader, .. } => reader.n_atoms(),
        TrajKind::Pdb { reader, .. } => reader.n_atoms(),
        TrajKind::Memory { reader, .. } => reader.n_atoms(),
    }
}

pub(crate) fn traj_n_frames_hint(traj: &TrajKind) -> Option<usize> {
    match traj {
        TrajKind::Dcd { reader, .. } => reader.n_frames_hint(),
        TrajKind::Xtc { reader, .. } => reader.n_frames_hint(),
        TrajKind::Gro { reader, .. } => reader.n_frames_hint(),
        TrajKind::G96 { reader, .. } => reader.n_frames_hint(),
        TrajKind::Cpt { reader, .. } => reader.n_frames_hint(),
        TrajKind::H5md { reader, .. } => reader.n_frames_hint(),
        TrajKind::Tng { reader, .. } => reader.n_frames_hint(),
        TrajKind::Trr { reader, .. } => reader.n_frames_hint(),
        TrajKind::Pdb { reader, .. } => reader.n_frames_hint(),
        TrajKind::Memory { reader, .. } => reader.n_frames_hint(),
    }
}

pub(crate) fn heuristic_chunk_frames(n_atoms: usize) -> usize {
    let bytes_per_frame = n_atoms.saturating_mul(std::mem::size_of::<[f32; 4]>());
    if bytes_per_frame == 0 {
        return 64;
    }
    let raw = (AUTO_CHUNK_TARGET_BYTES / bytes_per_frame).clamp(AUTO_CHUNK_MIN, AUTO_CHUNK_MAX);
    let mut p = 1usize;
    while p < raw {
        p <<= 1;
    }
    let lower = p >> 1;
    let upper = p;
    let chunk = if lower >= AUTO_CHUNK_MIN && raw.saturating_sub(lower) <= upper.saturating_sub(raw)
    {
        lower
    } else {
        upper
    };
    chunk.clamp(AUTO_CHUNK_MIN, AUTO_CHUNK_MAX)
}

fn bench_chunk_frames(
    traj: &mut TrajKind,
    chunk_frames: usize,
    sample_frames: usize,
    preferred_selection: Option<&[u32]>,
) -> PyResult<f64> {
    let use_selected = preferred_selection
        .map(|sel| !sel.is_empty())
        .unwrap_or(false);
    let n_atoms = if use_selected {
        preferred_selection.unwrap().len()
    } else {
        traj_n_atoms(traj)
    };
    let mut builder = FrameChunkBuilder::new(n_atoms, chunk_frames);
    let mut total = 0usize;
    let t0 = std::time::Instant::now();
    while total < sample_frames {
        let read = match traj {
            TrajKind::Dcd { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::Xtc { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::Gro { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::G96 { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::Cpt { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::H5md { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::Tng { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::Trr { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::Pdb { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
            TrajKind::Memory { reader, .. } => {
                if let Some(selection) = preferred_selection {
                    if use_selected {
                        reader.read_chunk_selected(chunk_frames, selection, &mut builder)
                    } else {
                        reader.read_chunk(chunk_frames, &mut builder)
                    }
                } else {
                    reader.read_chunk(chunk_frames, &mut builder)
                }
            }
        }
        .map_err(to_py_err)?;
        if read == 0 {
            break;
        }
        total += read;
    }
    let dt = t0.elapsed().as_secs_f64();
    if total == 0 || dt <= 0.0 {
        return Ok(0.0);
    }
    Ok(total as f64 / dt)
}

pub(crate) fn resolve_chunk_frames_for_streaming(traj: &TrajKind) -> PyResult<usize> {
    let n_atoms = traj_n_atoms(traj);
    Ok(heuristic_chunk_frames(n_atoms))
}

pub(crate) fn matrix_to_py<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<&'py PyArray2<f32>> {
    if rows * cols != data.len() {
        return Err(PyRuntimeError::new_err("matrix shape mismatch"));
    }
    let array = Array2::from_shape_vec((rows, cols), data)
        .map_err(|_| PyRuntimeError::new_err("invalid matrix shape"))?;
    Ok(array.into_pyarray_bound(py).into_gil_ref())
}

pub(crate) fn split_rama_phi_psi(
    data: &[f32],
    rows: usize,
    cols: usize,
) -> PyResult<(Vec<f32>, Vec<f32>, usize)> {
    if rows.saturating_mul(cols) != data.len() {
        return Err(PyRuntimeError::new_err("rama output shape mismatch"));
    }
    if cols % 2 != 0 {
        return Err(PyRuntimeError::new_err(
            "rama output must have even column count",
        ));
    }
    let n_res = cols / 2;
    let mut phi = Vec::with_capacity(rows * n_res);
    let mut psi = Vec::with_capacity(rows * n_res);
    for row in 0..rows {
        let row_offset = row * cols;
        for resid in 0..n_res {
            let base = row_offset + resid * 2;
            phi.push(data[base]);
            psi.push(data[base + 1]);
        }
    }
    Ok((phi, psi, n_res))
}

pub(crate) fn saltbr_class_name(code: u8) -> &'static str {
    match code {
        0 => "plus_plus",
        1 => "min_min",
        2 => "plus_min",
        _ => "unknown",
    }
}

pub(crate) fn timeseries_to_py(
    py: Python<'_>,
    time: Vec<f32>,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
) -> PyResult<PyObject> {
    if rows * cols != data.len() {
        return Err(PyRuntimeError::new_err("timeseries shape mismatch"));
    }
    let t = PyArray1::from_vec_bound(py, time);
    let array = Array2::from_shape_vec((rows, cols), data)
        .map_err(|_| PyRuntimeError::new_err("invalid timeseries shape"))?;
    Ok((t, array.into_pyarray_bound(py).into_gil_ref()).into_py(py))
}

pub(crate) fn hist_to_py(py: Python<'_>, centers: Vec<f32>, counts: Vec<u64>) -> PyObject {
    let centers = PyArray1::from_vec_bound(py, centers);
    let counts = PyArray1::from_vec_bound(py, counts);
    (centers, counts).into_py(py)
}

pub(crate) fn pca_to_py(py: Python<'_>, output: traj_engine::PcaOutput) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "eigenvalues",
        PyArray1::from_vec_bound(py, output.eigenvalues),
    )?;
    let array = Array2::from_shape_vec(
        (output.n_components, output.n_features),
        output.eigenvectors,
    )
    .map_err(|_| PyRuntimeError::new_err("invalid pca eigenvector shape"))?;
    dict.set_item("eigenvectors", array.into_pyarray_bound(py))?;
    Ok(dict.into_py(py))
}

pub(crate) fn persistence_to_py(
    py: Python<'_>,
    output: traj_engine::PersistenceOutput,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "bond_autocorrelation",
        PyArray1::from_vec_bound(py, output.bond_autocorrelation),
    )?;
    dict.set_item("lb", output.lb)?;
    dict.set_item("lp", output.lp)?;
    dict.set_item("fit", PyArray1::from_vec_bound(py, output.fit))?;
    dict.set_item("kuhn_length", output.kuhn_length)?;
    Ok(dict.into_py(py))
}

pub(crate) fn dielectric_to_py(
    py: Python<'_>,
    output: traj_engine::DielectricOutput,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("time", PyArray1::from_vec_bound(py, output.time))?;
    dict.set_item("rot_sq", PyArray1::from_vec_bound(py, output.rot_sq))?;
    dict.set_item("trans_sq", PyArray1::from_vec_bound(py, output.trans_sq))?;
    dict.set_item("rot_trans", PyArray1::from_vec_bound(py, output.rot_trans))?;
    dict.set_item("dielectric_rot", output.dielectric_rot)?;
    dict.set_item("dielectric_total", output.dielectric_total)?;
    dict.set_item("mu_avg", output.mu_avg)?;
    Ok(dict.into_py(py))
}

pub(crate) fn water_order_to_py(
    py: Python<'_>,
    output: traj_engine::H2OrderOutput,
) -> PyResult<PyObject> {
    if output.rows.saturating_mul(output.cols) != output.dipole.len() {
        return Err(PyRuntimeError::new_err("h2order dipole shape mismatch"));
    }
    let dipole = Array2::from_shape_vec((output.rows, output.cols), output.dipole)
        .map_err(|_| PyRuntimeError::new_err("invalid h2order dipole shape"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "coordinate",
        PyArray1::from_vec_bound(py, output.coordinate),
    )?;
    dict.set_item("order", PyArray1::from_vec_bound(py, output.order))?;
    dict.set_item("dipole", dipole.into_pyarray_bound(py))?;
    dict.set_item("counts", PyArray1::from_vec_bound(py, output.counts))?;
    dict.set_item("axis", axis_name(output.axis))?;
    dict.set_item(
        "bounds",
        PyArray1::from_vec_bound(py, output.bounds.to_vec()),
    )?;
    dict.set_item("slice_width", output.slice_width)?;
    dict.set_item("n_frames", output.n_frames)?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("length_scale", output.length_scale)?;
    dict.set_item("dipole_unit", output.dipole_unit)?;
    Ok(dict.into_py(py))
}

pub(crate) fn helix_to_py(py: Python<'_>, output: traj_engine::HelixOutput) -> PyResult<PyObject> {
    let scalar_shape = (output.frames, output.residues);
    let fragment_mask = Array2::from_shape_vec(scalar_shape, output.fragment_mask)
        .map_err(|_| PyRuntimeError::new_err("invalid helix fragment mask shape"))?;
    let residue_rmsd = Array2::from_shape_vec(scalar_shape, output.residue_rmsd)
        .map_err(|_| PyRuntimeError::new_err("invalid helix residue rmsd shape"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("labels", output.labels)?;
    dict.set_item("time", PyArray1::from_vec_bound(py, output.time))?;
    dict.set_item(
        "fragment_start",
        PyArray1::from_vec_bound(py, output.fragment_start),
    )?;
    dict.set_item(
        "fragment_end",
        PyArray1::from_vec_bound(py, output.fragment_end),
    )?;
    dict.set_item("radius", PyArray1::from_vec_bound(py, output.radius))?;
    dict.set_item("twist", PyArray1::from_vec_bound(py, output.twist))?;
    dict.set_item("rise", PyArray1::from_vec_bound(py, output.rise))?;
    dict.set_item("length", PyArray1::from_vec_bound(py, output.length))?;
    dict.set_item("dipole", PyArray1::from_vec_bound(py, output.dipole))?;
    dict.set_item("rmsd", PyArray1::from_vec_bound(py, output.rmsd))?;
    dict.set_item("ca_phi", PyArray1::from_vec_bound(py, output.ca_phi))?;
    dict.set_item("phi", PyArray1::from_vec_bound(py, output.phi))?;
    dict.set_item("psi", PyArray1::from_vec_bound(py, output.psi))?;
    dict.set_item("hb3", PyArray1::from_vec_bound(py, output.hb3))?;
    dict.set_item("hb4", PyArray1::from_vec_bound(py, output.hb4))?;
    dict.set_item("hb5", PyArray1::from_vec_bound(py, output.hb5))?;
    dict.set_item(
        "ellipticity",
        PyArray1::from_vec_bound(py, output.ellipticity),
    )?;
    dict.set_item("fragment_mask", fragment_mask.into_pyarray_bound(py))?;
    dict.set_item("residue_rmsd", residue_rmsd.into_pyarray_bound(py))?;
    dict.set_item(
        "helicity_fraction",
        PyArray1::from_vec_bound(py, output.helicity_fraction),
    )?;
    dict.set_item("jca_ha", PyArray1::from_vec_bound(py, output.jca_ha))?;
    dict.set_item("frames", output.frames)?;
    dict.set_item("residues", output.residues)?;
    dict.set_item("fit", output.fit)?;
    dict.set_item("check_each_frame", output.check_each_frame)?;
    dict.set_item("length_scale", output.length_scale)?;
    dict.set_item("used_box", output.used_box)?;
    Ok(dict.into_py(py))
}

pub(crate) fn helix_orientation_to_py(
    py: Python<'_>,
    output: traj_engine::HelixOrientOutput,
) -> PyResult<PyObject> {
    let vec_shape = (output.frames, output.residues, 3);
    let scalar_shape = (output.frames, output.residues);
    let axis = Array3::from_shape_vec(vec_shape, output.axis)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient axis shape"))?;
    let center = Array3::from_shape_vec(vec_shape, output.center)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient center shape"))?;
    let residue_vector = Array3::from_shape_vec(vec_shape, output.residue_vector)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient residue vector shape"))?;
    let normal = Array3::from_shape_vec(vec_shape, output.normal)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient normal shape"))?;
    let rise = Array2::from_shape_vec(scalar_shape, output.rise)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient rise shape"))?;
    let radius = Array2::from_shape_vec(scalar_shape, output.radius)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient radius shape"))?;
    let twist = Array2::from_shape_vec(scalar_shape, output.twist)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient twist shape"))?;
    let bending = Array2::from_shape_vec(scalar_shape, output.bending)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient bending shape"))?;
    let tilt = Array2::from_shape_vec(scalar_shape, output.tilt)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient tilt shape"))?;
    let rotation = Array2::from_shape_vec(scalar_shape, output.rotation)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient rotation shape"))?;
    let theta1 = Array2::from_shape_vec(scalar_shape, output.theta1)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient theta1 shape"))?;
    let theta2 = Array2::from_shape_vec(scalar_shape, output.theta2)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient theta2 shape"))?;
    let theta3 = Array2::from_shape_vec(scalar_shape, output.theta3)
        .map_err(|_| PyRuntimeError::new_err("invalid helixorient theta3 shape"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("labels", output.labels)?;
    dict.set_item("time", PyArray1::from_vec_bound(py, output.time))?;
    dict.set_item("axis", axis.into_pyarray_bound(py))?;
    dict.set_item("center", center.into_pyarray_bound(py))?;
    dict.set_item("residue_vector", residue_vector.into_pyarray_bound(py))?;
    dict.set_item("normal", normal.into_pyarray_bound(py))?;
    dict.set_item("rise", rise.into_pyarray_bound(py))?;
    dict.set_item("radius", radius.into_pyarray_bound(py))?;
    dict.set_item("twist", twist.into_pyarray_bound(py))?;
    dict.set_item("bending", bending.into_pyarray_bound(py))?;
    dict.set_item("tilt", tilt.into_pyarray_bound(py))?;
    dict.set_item("rotation", rotation.into_pyarray_bound(py))?;
    dict.set_item("theta1", theta1.into_pyarray_bound(py))?;
    dict.set_item("theta2", theta2.into_pyarray_bound(py))?;
    dict.set_item("theta3", theta3.into_pyarray_bound(py))?;
    dict.set_item("frames", output.frames)?;
    dict.set_item("residues", output.residues)?;
    dict.set_item("use_sidechain", output.use_sidechain)?;
    dict.set_item("incremental", output.incremental)?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("length_scale", output.length_scale)?;
    Ok(dict.into_py(py))
}

pub(crate) fn bundle_to_py(
    py: Python<'_>,
    output: traj_engine::BundleOutput,
) -> PyResult<PyObject> {
    let vec_shape = (output.frames, output.axes, 3);
    let scalar_shape = (output.frames, output.axes);
    let reference_axis = Array2::from_shape_vec((output.frames, 3), output.reference_axis)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle reference-axis shape"))?;
    let top = Array3::from_shape_vec(vec_shape, output.top)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle top shape"))?;
    let bottom = Array3::from_shape_vec(vec_shape, output.bottom)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle bottom shape"))?;
    let mid = Array3::from_shape_vec(vec_shape, output.mid)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle mid shape"))?;
    let direction = Array3::from_shape_vec(vec_shape, output.direction)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle direction shape"))?;
    let length = Array2::from_shape_vec(scalar_shape, output.length)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle length shape"))?;
    let distance = Array2::from_shape_vec(scalar_shape, output.distance)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle distance shape"))?;
    let z_shift = Array2::from_shape_vec(scalar_shape, output.z_shift)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle z-shift shape"))?;
    let tilt = Array2::from_shape_vec(scalar_shape, output.tilt)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle tilt shape"))?;
    let radial_tilt = Array2::from_shape_vec(scalar_shape, output.radial_tilt)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle radial-tilt shape"))?;
    let lateral_tilt = Array2::from_shape_vec(scalar_shape, output.lateral_tilt)
        .map_err(|_| PyRuntimeError::new_err("invalid bundle lateral-tilt shape"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("labels", output.labels)?;
    dict.set_item("time", PyArray1::from_vec_bound(py, output.time))?;
    dict.set_item("reference_axis", reference_axis.into_pyarray_bound(py))?;
    dict.set_item("top", top.into_pyarray_bound(py))?;
    dict.set_item("bottom", bottom.into_pyarray_bound(py))?;
    dict.set_item("mid", mid.into_pyarray_bound(py))?;
    dict.set_item("direction", direction.into_pyarray_bound(py))?;
    dict.set_item("length", length.into_pyarray_bound(py))?;
    dict.set_item("distance", distance.into_pyarray_bound(py))?;
    dict.set_item("z_shift", z_shift.into_pyarray_bound(py))?;
    dict.set_item("tilt", tilt.into_pyarray_bound(py))?;
    dict.set_item("radial_tilt", radial_tilt.into_pyarray_bound(py))?;
    dict.set_item("lateral_tilt", lateral_tilt.into_pyarray_bound(py))?;
    if output.has_kink {
        let kink = Array3::from_shape_vec(vec_shape, output.kink)
            .map_err(|_| PyRuntimeError::new_err("invalid bundle kink shape"))?;
        let kink_angle = Array2::from_shape_vec(scalar_shape, output.kink_angle)
            .map_err(|_| PyRuntimeError::new_err("invalid bundle kink-angle shape"))?;
        let kink_radial = Array2::from_shape_vec(scalar_shape, output.kink_radial)
            .map_err(|_| PyRuntimeError::new_err("invalid bundle kink-radial shape"))?;
        let kink_lateral = Array2::from_shape_vec(scalar_shape, output.kink_lateral)
            .map_err(|_| PyRuntimeError::new_err("invalid bundle kink-lateral shape"))?;
        dict.set_item("kink", kink.into_pyarray_bound(py))?;
        dict.set_item("kink_angle", kink_angle.into_pyarray_bound(py))?;
        dict.set_item("kink_radial", kink_radial.into_pyarray_bound(py))?;
        dict.set_item("kink_lateral", kink_lateral.into_pyarray_bound(py))?;
    } else {
        dict.set_item("kink", py.None())?;
        dict.set_item("kink_angle", py.None())?;
        dict.set_item("kink_radial", py.None())?;
        dict.set_item("kink_lateral", py.None())?;
    }
    dict.set_item("frames", output.frames)?;
    dict.set_item("axes", output.axes)?;
    dict.set_item("has_kink", output.has_kink)?;
    dict.set_item("use_z_reference", output.use_z_reference)?;
    dict.set_item("mass_weighted", output.mass_weighted)?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("length_scale", output.length_scale)?;
    Ok(dict.into_py(py))
}

pub(crate) fn mdmat_to_py(py: Python<'_>, output: traj_engine::MdmatOutput) -> PyResult<PyObject> {
    let mean_matrix =
        Array2::from_shape_vec((output.residues, output.residues), output.mean_matrix)
            .map_err(|_| PyRuntimeError::new_err("invalid mdmat mean-matrix shape"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("labels", output.labels)?;
    dict.set_item("mean_matrix", mean_matrix.into_pyarray_bound(py))?;
    if output.frame_matrices.is_empty() {
        dict.set_item("time", py.None())?;
        dict.set_item("frame_matrices", py.None())?;
    } else {
        let frame_shape = (output.frames, output.residues, output.residues);
        let frame_matrices = Array3::from_shape_vec(frame_shape, output.frame_matrices)
            .map_err(|_| PyRuntimeError::new_err("invalid mdmat frame-matrix shape"))?;
        dict.set_item("time", PyArray1::from_vec_bound(py, output.time))?;
        dict.set_item("frame_matrices", frame_matrices.into_pyarray_bound(py))?;
    }
    dict.set_item("frames", output.frames)?;
    dict.set_item("residues", output.residues)?;
    dict.set_item("truncate", output.truncate)?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("length_scale", output.length_scale)?;
    if output.distinct_contact_atoms.is_empty() {
        dict.set_item("distinct_contact_atoms", py.None())?;
        dict.set_item("mean_contact_atoms", py.None())?;
        dict.set_item("contact_ratio", py.None())?;
        dict.set_item("residue_atom_counts", py.None())?;
        dict.set_item("mean_contact_atoms_per_residue_atom", py.None())?;
    } else {
        dict.set_item(
            "distinct_contact_atoms",
            PyArray1::from_vec_bound(py, output.distinct_contact_atoms),
        )?;
        dict.set_item(
            "mean_contact_atoms",
            PyArray1::from_vec_bound(py, output.mean_contact_atoms),
        )?;
        dict.set_item(
            "contact_ratio",
            PyArray1::from_vec_bound(py, output.contact_ratio),
        )?;
        dict.set_item(
            "residue_atom_counts",
            PyArray1::from_vec_bound(py, output.residue_atom_counts),
        )?;
        dict.set_item(
            "mean_contact_atoms_per_residue_atom",
            PyArray1::from_vec_bound(py, output.mean_contact_atoms_per_residue_atom),
        )?;
    }
    Ok(dict.into_py(py))
}

pub(crate) fn hydration_order_to_py(
    py: Python<'_>,
    output: traj_engine::HydOrderOutput,
) -> PyResult<PyObject> {
    let grid_shape = (output.dims[0], output.dims[1], output.dims[2]);
    let sg_grid = Array3::from_shape_vec(grid_shape, output.sg_grid)
        .map_err(|_| PyRuntimeError::new_err("invalid hydorder sg grid shape"))?;
    let sk_grid = Array3::from_shape_vec(grid_shape, output.sk_grid)
        .map_err(|_| PyRuntimeError::new_err("invalid hydorder sk grid shape"))?;
    let counts = Array3::from_shape_vec(grid_shape, output.counts)
        .map_err(|_| PyRuntimeError::new_err("invalid hydorder count grid shape"))?;
    let bounds = Array2::from_shape_vec((3, 2), output.bounds.to_vec())
        .map_err(|_| PyRuntimeError::new_err("invalid hydorder bounds shape"))?;
    let interface_shape = (
        output.interface_blocks,
        output.interface_rows,
        output.interface_cols,
    );
    let interface_lower = Array3::from_shape_vec(interface_shape, output.interface_lower)
        .map_err(|_| PyRuntimeError::new_err("invalid hydorder lower interface shape"))?;
    let interface_upper = Array3::from_shape_vec(interface_shape, output.interface_upper)
        .map_err(|_| PyRuntimeError::new_err("invalid hydorder upper interface shape"))?;
    let mut x = Vec::with_capacity(output.dims[0]);
    let mut y = Vec::with_capacity(output.dims[1]);
    let mut z = Vec::with_capacity(output.dims[2]);
    for i in 0..output.dims[0] {
        x.push(output.bounds[0] + (i as f32 + 0.5) * output.bin_width);
    }
    for i in 0..output.dims[1] {
        y.push(output.bounds[2] + (i as f32 + 0.5) * output.bin_width);
    }
    for i in 0..output.dims[2] {
        z.push(output.bounds[4] + (i as f32 + 0.5) * output.bin_width);
    }
    let dict = PyDict::new_bound(py);
    dict.set_item("sg_mean", output.sg_mean)?;
    dict.set_item("sk_mean", output.sk_mean)?;
    dict.set_item("sg_grid", sg_grid.into_pyarray_bound(py))?;
    dict.set_item("sk_grid", sk_grid.into_pyarray_bound(py))?;
    dict.set_item("counts", counts.into_pyarray_bound(py))?;
    dict.set_item("x", PyArray1::from_vec_bound(py, x))?;
    dict.set_item("y", PyArray1::from_vec_bound(py, y))?;
    dict.set_item("z", PyArray1::from_vec_bound(py, z))?;
    dict.set_item("dims", output.dims.to_vec())?;
    dict.set_item("bounds", bounds.into_pyarray_bound(py))?;
    dict.set_item("bin_width", output.bin_width)?;
    dict.set_item("axis", axis_name(output.axis))?;
    dict.set_item(
        "plane_axes",
        vec![
            axis_name(output.plane_axes[0]).to_string(),
            axis_name(output.plane_axes[1]).to_string(),
        ],
    )?;
    dict.set_item("n_frames", output.n_frames)?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("length_scale", output.length_scale)?;
    dict.set_item("interface_lower", interface_lower.into_pyarray_bound(py))?;
    dict.set_item("interface_upper", interface_upper.into_pyarray_bound(py))?;
    dict.set_item("interface_blocks", output.interface_blocks)?;
    dict.set_item("interface_threshold", output.interface_threshold)?;
    dict.set_item("block_size", output.block_size)?;
    Ok(dict.into_py(py))
}

pub(crate) fn solvent_orientation_to_py(
    py: Python<'_>,
    output: traj_engine::SOrientOutput,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "cos_theta1",
        PyArray1::from_vec_bound(py, output.cos_theta1),
    )?;
    dict.set_item(
        "cos_theta1_distribution",
        PyArray1::from_vec_bound(py, output.cos_theta1_distribution),
    )?;
    dict.set_item(
        "abs_cos_theta2",
        PyArray1::from_vec_bound(py, output.abs_cos_theta2),
    )?;
    dict.set_item(
        "abs_cos_theta2_distribution",
        PyArray1::from_vec_bound(py, output.abs_cos_theta2_distribution),
    )?;
    dict.set_item("r", PyArray1::from_vec_bound(py, output.r))?;
    dict.set_item(
        "mean_cos_theta1",
        PyArray1::from_vec_bound(py, output.mean_cos_theta1),
    )?;
    dict.set_item(
        "mean_p2_theta2",
        PyArray1::from_vec_bound(py, output.mean_p2_theta2),
    )?;
    dict.set_item(
        "cumulative_r",
        PyArray1::from_vec_bound(py, output.cumulative_r),
    )?;
    dict.set_item(
        "cumulative_cos_theta1",
        PyArray1::from_vec_bound(py, output.cumulative_cos_theta1),
    )?;
    dict.set_item(
        "cumulative_p2_theta2",
        PyArray1::from_vec_bound(py, output.cumulative_p2_theta2),
    )?;
    dict.set_item(
        "count_density",
        PyArray1::from_vec_bound(py, output.count_density),
    )?;
    dict.set_item("counts", PyArray1::from_vec_bound(py, output.counts))?;
    dict.set_item("window_count", output.window_count)?;
    dict.set_item("average_shell_size", output.average_shell_size)?;
    dict.set_item("window_mean_cos_theta1", output.window_mean_cos_theta1)?;
    dict.set_item("window_mean_p2_theta2", output.window_mean_p2_theta2)?;
    dict.set_item(
        "r_window",
        PyArray1::from_vec_bound(py, output.r_window.to_vec()),
    )?;
    dict.set_item("cbin", output.cbin)?;
    dict.set_item("rbin", output.rbin)?;
    dict.set_item("r_profile_max", output.r_profile_max)?;
    dict.set_item("use_vector23", output.use_vector23)?;
    dict.set_item("use_com", output.use_com)?;
    dict.set_item("n_frames", output.n_frames)?;
    dict.set_item("n_reference_positions", output.n_reference_positions)?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("length_scale", output.length_scale)?;
    Ok(dict.into_py(py))
}

pub(crate) fn solvent_polarization_to_py(
    py: Python<'_>,
    output: traj_engine::SpolOutput,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("r", PyArray1::from_vec_bound(py, output.r))?;
    dict.set_item(
        "cumulative_count",
        PyArray1::from_vec_bound(py, output.cumulative_count),
    )?;
    dict.set_item(
        "shell_count",
        PyArray1::from_vec_bound(py, output.shell_count),
    )?;
    dict.set_item(
        "shell_count_per_frame",
        PyArray1::from_vec_bound(py, output.shell_count_per_frame),
    )?;
    dict.set_item("average_shell_size", output.average_shell_size)?;
    dict.set_item("average_dipole", output.average_dipole)?;
    dict.set_item("dipole_std", output.dipole_std)?;
    dict.set_item("average_radial_dipole", output.average_radial_dipole)?;
    dict.set_item(
        "average_radial_polarization",
        output.average_radial_polarization,
    )?;
    dict.set_item("window_count", output.window_count)?;
    dict.set_item(
        "r_window",
        PyArray1::from_vec_bound(py, output.r_window.to_vec()),
    )?;
    dict.set_item("bin_width", output.bin_width)?;
    dict.set_item("r_hist_max", output.r_hist_max)?;
    dict.set_item("use_com", output.use_com)?;
    dict.set_item("reference_atom", output.reference_atom)?;
    dict.set_item("refdip", output.refdip)?;
    dict.set_item("n_frames", output.n_frames)?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("length_scale", output.length_scale)?;
    dict.set_item("dipole_unit", output.dipole_unit)?;
    Ok(dict.into_py(py))
}

pub(crate) fn current_to_py(
    py: Python<'_>,
    output: traj_engine::CurrentOutput,
) -> PyResult<PyObject> {
    if output
        .conductivity_rows
        .saturating_mul(output.conductivity_cols)
        != output.conductivity.len()
    {
        return Err(PyRuntimeError::new_err(
            "current conductivity shape mismatch",
        ));
    }
    let conductivity = Array2::from_shape_vec(
        (output.conductivity_rows, output.conductivity_cols),
        output.conductivity,
    )
    .map_err(|_| PyRuntimeError::new_err("invalid current conductivity shape"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "conductivity_time",
        PyArray1::from_vec_bound(py, output.conductivity_time),
    )?;
    dict.set_item("conductivity", conductivity.into_pyarray_bound(py))?;
    dict.set_item("time", PyArray1::from_vec_bound(py, output.frame_time))?;
    dict.set_item("md_sq", PyArray1::from_vec_bound(py, output.md_sq))?;
    dict.set_item("mj_sq", PyArray1::from_vec_bound(py, output.mj_sq))?;
    dict.set_item("md_mj", PyArray1::from_vec_bound(py, output.md_mj))?;
    dict.set_item("dielectric_rot", output.dielectric_rot)?;
    dict.set_item("dielectric_total", output.dielectric_total)?;
    dict.set_item("mu_avg", output.mu_avg)?;
    dict.set_item("conductivity_static", output.conductivity_static)?;
    Ok(dict.into_py(py))
}

pub(crate) fn potential_to_py(
    py: Python<'_>,
    output: traj_engine::PotentialOutput,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item(
        "coordinate",
        PyArray1::from_vec_bound(py, output.coordinate),
    )?;
    dict.set_item(
        "charge_density",
        PyArray1::from_vec_bound(py, output.charge_density),
    )?;
    dict.set_item("field", PyArray1::from_vec_bound(py, output.field))?;
    dict.set_item("potential", PyArray1::from_vec_bound(py, output.potential))?;
    dict.set_item("axis", axis_name(output.axis))?;
    dict.set_item(
        "bounds",
        PyArray1::from_vec_bound(py, output.bounds.to_vec()),
    )?;
    dict.set_item("slice_width", output.slice_width)?;
    dict.set_item("n_frames", output.n_frames)?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("centered", output.centered)?;
    dict.set_item("symmetrized", output.symmetrized)?;
    dict.set_item("corrected", output.corrected)?;
    dict.set_item("length_scale", output.length_scale)?;
    dict.set_item("discard_start", output.discard_start)?;
    dict.set_item("discard_end", output.discard_end)?;
    Ok(dict.into_py(py))
}

pub(crate) fn structure_factor_to_py(
    py: Python<'_>,
    output: traj_engine::StructureFactorOutput,
) -> PyResult<PyObject> {
    let r = PyArray1::from_vec_bound(py, output.r);
    let g = PyArray1::from_vec_bound(py, output.g_r);
    let q = PyArray1::from_vec_bound(py, output.q);
    let s = PyArray1::from_vec_bound(py, output.s_q);
    Ok((r, g, q, s).into_py(py))
}

pub(crate) fn validate_vanhove_integral_radius(radius: Option<f64>) -> PyResult<Option<f32>> {
    match radius {
        Some(value) if !value.is_finite() || value < 0.0 => Err(PyValueError::new_err(
            "integral_radius must be finite and >= 0",
        )),
        Some(value) => Ok(Some(value as f32)),
        None => Ok(None),
    }
}

pub(crate) fn validate_vanhove_curve_step(step: Option<i64>) -> PyResult<Option<usize>> {
    match step {
        Some(value) if value <= 0 => Err(PyValueError::new_err("curve_step must be positive")),
        Some(value) => Ok(Some(value as usize)),
        None => Ok(None),
    }
}

fn vanhove_integral_profile(
    matrix: &[f32],
    rows: usize,
    cols: usize,
    r_bin: f32,
    radius: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    if rows == 0 || cols == 0 {
        return out;
    }
    let max_bin = ((radius / r_bin).floor() as usize)
        .saturating_add(1)
        .min(cols);
    if max_bin == 0 {
        return out;
    }
    for row in 0..rows {
        let row_slice = &matrix[row * cols..(row + 1) * cols];
        let mut sum = 0.0f32;
        for (bin, &value) in row_slice[..max_bin].iter().enumerate() {
            let weight = if bin == 0 { 0.5 } else { 1.0 };
            sum += value * weight;
        }
        out[row] = sum * r_bin;
    }
    out
}

fn vanhove_curve_indices(
    rows: usize,
    curve_lags: Option<&[i64]>,
    curve_step: Option<usize>,
) -> Vec<usize> {
    if let Some(lags) = curve_lags {
        let mut indices = Vec::with_capacity(lags.len());
        for &value in lags {
            let idx = if value < 0 {
                rows as i64 + value
            } else {
                value
            };
            if idx >= 0 {
                let idx = idx as usize;
                if idx < rows {
                    indices.push(idx);
                }
            }
        }
        return indices;
    }
    if let Some(step) = curve_step {
        return (0..rows).step_by(step).collect();
    }
    Vec::new()
}

fn vanhove_curve_matrix(matrix: &[f32], rows: usize, cols: usize, indices: &[usize]) -> Vec<f32> {
    let mut out = Vec::with_capacity(indices.len().saturating_mul(cols));
    for &row in indices {
        if row < rows {
            out.extend_from_slice(&matrix[row * cols..(row + 1) * cols]);
        }
    }
    out
}

pub(crate) fn vanhove_to_py(
    py: Python<'_>,
    output: traj_engine::VanHoveOutput,
    integral_radius: Option<f64>,
    curve_lags: Option<Vec<i64>>,
    curve_step: Option<i64>,
) -> PyResult<PyObject> {
    let traj_engine::VanHoveOutput {
        time,
        time_sqrt,
        r,
        matrix,
        rows,
        cols,
        counts,
        r_bin,
        r_max,
    } = output;
    if time.len() != rows || time_sqrt.len() != rows {
        return Err(PyRuntimeError::new_err("invalid vanhove time axis shape"));
    }
    if r.len() != cols || matrix.len() != rows.saturating_mul(cols) {
        return Err(PyRuntimeError::new_err("invalid vanhove matrix shape"));
    }
    let integral_radius = validate_vanhove_integral_radius(integral_radius)?;
    let curve_step = validate_vanhove_curve_step(curve_step)?;
    let integral =
        integral_radius.map(|radius| vanhove_integral_profile(&matrix, rows, cols, r_bin, radius));
    let curve_indices = if curve_lags.is_some() || curve_step.is_some() {
        Some(vanhove_curve_indices(
            rows,
            curve_lags.as_deref(),
            curve_step,
        ))
    } else {
        None
    };
    let curve_time = curve_indices
        .as_ref()
        .map(|indices| indices.iter().map(|&idx| time[idx]).collect::<Vec<f32>>());
    let curve_matrix = curve_indices
        .as_ref()
        .map(|indices| {
            Array2::from_shape_vec(
                (indices.len(), cols),
                vanhove_curve_matrix(&matrix, rows, cols, indices),
            )
            .map_err(|_| PyRuntimeError::new_err("invalid vanhove curve matrix shape"))
        })
        .transpose()?;
    let dict = PyDict::new_bound(py);
    dict.set_item("time", PyArray1::from_vec_bound(py, time))?;
    dict.set_item("time_sqrt", PyArray1::from_vec_bound(py, time_sqrt))?;
    dict.set_item("r", PyArray1::from_vec_bound(py, r))?;
    let matrix = Array2::from_shape_vec((rows, cols), matrix)
        .map_err(|_| PyRuntimeError::new_err("invalid vanhove matrix shape"))?;
    dict.set_item("matrix", matrix.into_pyarray_bound(py))?;
    dict.set_item("counts", PyArray1::from_vec_bound(py, counts))?;
    dict.set_item("r_bin", r_bin)?;
    dict.set_item("r_max", r_max)?;
    if let (Some(radius), Some(integral)) = (integral_radius, integral) {
        dict.set_item("integral_radius", radius)?;
        dict.set_item("integral", PyArray1::from_vec_bound(py, integral))?;
    }
    if let (Some(indices), Some(curve_time), Some(curve_matrix)) =
        (curve_indices, curve_time, curve_matrix)
    {
        let curve_indices: Vec<i64> = indices.iter().map(|&idx| idx as i64).collect();
        dict.set_item("curve_indices", PyArray1::from_vec_bound(py, curve_indices))?;
        dict.set_item("curve_time", PyArray1::from_vec_bound(py, curve_time))?;
        dict.set_item("curve_matrix", curve_matrix.into_pyarray_bound(py))?;
    }
    Ok(dict.into_py(py))
}

#[cfg(test)]
mod vanhove_py_tests {
    use super::*;

    #[test]
    fn vanhove_integral_matches_python_contract() {
        let matrix = vec![
            1.0f32, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.25, 0.5,
        ];
        let integral = vanhove_integral_profile(&matrix, 3, 3, 1.0, 1.0);
        assert_eq!(integral, vec![0.5, 1.0, 0.25]);
    }

    #[test]
    fn vanhove_curve_selection_matches_python_contract() {
        assert_eq!(
            vanhove_curve_indices(3, Some(&[1, -1, 8, -9]), None),
            vec![1, 2]
        );
        assert_eq!(vanhove_curve_indices(5, None, Some(2)), vec![0, 2, 4]);
    }
}

#[cfg(test)]
mod rama_py_tests {
    use super::*;

    #[test]
    fn rama_split_matches_python_contract() {
        let data = vec![f32::NAN, 10.0, -29.0, 1.0, 45.0, f32::NAN];
        let (phi, psi, n_res) = split_rama_phi_psi(&data, 1, 6).unwrap();
        assert_eq!(n_res, 3);
        assert!(phi[0].is_nan());
        assert_eq!(phi[1], -29.0);
        assert_eq!(phi[2], 45.0);
        assert_eq!(psi[0], 10.0);
        assert_eq!(psi[1], 1.0);
        assert!(psi[2].is_nan());
    }

    #[test]
    fn rama_split_rejects_odd_columns() {
        let err = split_rama_phi_psi(&[0.0, 1.0, 2.0], 1, 3).unwrap_err();
        assert!(err.to_string().contains("even column count"));
    }
}

pub(crate) fn grid_to_py(py: Python<'_>, output: traj_engine::GridOutput) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("dims", vec![output.dims[0], output.dims[1], output.dims[2]])?;
    dict.set_item("mean", PyArray1::from_vec_bound(py, output.mean))?;
    dict.set_item("std", PyArray1::from_vec_bound(py, output.std))?;
    dict.set_item("first", PyArray1::from_vec_bound(py, output.first))?;
    dict.set_item("last", PyArray1::from_vec_bound(py, output.last))?;
    dict.set_item("min", PyArray1::from_vec_bound(py, output.min))?;
    dict.set_item("max", PyArray1::from_vec_bound(py, output.max))?;
    Ok(dict.into_py(py))
}

pub(crate) fn density_map_to_py(
    py: Python<'_>,
    output: traj_engine::DensityMapOutput,
) -> PyResult<PyObject> {
    if output.rows.saturating_mul(output.cols) != output.matrix.len() {
        return Err(PyRuntimeError::new_err("density map shape mismatch"));
    }
    let matrix = Array2::from_shape_vec((output.rows, output.cols), output.matrix)
        .map_err(|_| PyRuntimeError::new_err("invalid density map shape"))?;
    let bounds = Array2::from_shape_vec(
        (2, 2),
        vec![
            output.bounds[0],
            output.bounds[1],
            output.bounds[2],
            output.bounds[3],
        ],
    )
    .map_err(|_| PyRuntimeError::new_err("invalid density map bounds"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("axis1", PyArray1::from_vec_bound(py, output.axis1))?;
    dict.set_item("axis2", PyArray1::from_vec_bound(py, output.axis2))?;
    dict.set_item("matrix", matrix.into_pyarray_bound(py))?;
    dict.set_item(
        "plane_axes",
        vec![
            axis_name(output.plane_axes[0]).to_string(),
            axis_name(output.plane_axes[1]).to_string(),
        ],
    )?;
    dict.set_item("average_axis", axis_name(output.average_axis))?;
    dict.set_item("unit", output.unit)?;
    dict.set_item("n_frames", output.n_frames)?;
    dict.set_item("bounds", bounds.into_pyarray_bound(py))?;
    dict.set_item(
        "bin_width",
        PyArray1::from_vec_bound(py, output.bin_width.to_vec()),
    )?;
    dict.set_item("used_box", output.used_box)?;
    dict.set_item("length_scale", output.length_scale)?;
    Ok(dict.into_py(py))
}

pub(crate) fn bondi_ffv_to_py(
    py: Python<'_>,
    time: Vec<f32>,
    data: Vec<f32>,
    rows: usize,
    cols: usize,
    bondi_scale: f64,
    molar_mass_dalton: f64,
    probe_radius: f64,
    ninsert_per_nm3: usize,
    seed: i64,
) -> PyResult<PyObject> {
    if cols != 6 || rows * cols != data.len() {
        return Err(PyRuntimeError::new_err("invalid bondi_ffv shape"));
    }
    let mut total_volume_a3 = Vec::with_capacity(rows);
    let mut vdw_volume_a3 = Vec::with_capacity(rows);
    let mut raw_free_volume_a3 = Vec::with_capacity(rows);
    let mut raw_free_volume_fraction = Vec::with_capacity(rows);
    let mut fractional_free_volume = Vec::with_capacity(rows);
    let mut density_g_cm3 = Vec::with_capacity(rows);
    for row in 0..rows {
        let base = row * cols;
        total_volume_a3.push(data[base]);
        vdw_volume_a3.push(data[base + 1]);
        raw_free_volume_a3.push(data[base + 2]);
        raw_free_volume_fraction.push(data[base + 3]);
        fractional_free_volume.push(data[base + 4]);
        density_g_cm3.push(data[base + 5]);
    }

    let dict = PyDict::new_bound(py);
    dict.set_item("time", PyArray1::from_vec_bound(py, time))?;
    dict.set_item(
        "total_volume_a3",
        PyArray1::from_vec_bound(py, total_volume_a3.clone()),
    )?;
    dict.set_item(
        "total_volume_nm3",
        PyArray1::from_vec_bound(
            py,
            total_volume_a3
                .iter()
                .map(|value| *value / 1000.0)
                .collect::<Vec<f32>>(),
        ),
    )?;
    dict.set_item(
        "vdw_volume_a3",
        PyArray1::from_vec_bound(py, vdw_volume_a3.clone()),
    )?;
    dict.set_item(
        "vdw_volume_nm3",
        PyArray1::from_vec_bound(
            py,
            vdw_volume_a3
                .iter()
                .map(|value| *value / 1000.0)
                .collect::<Vec<f32>>(),
        ),
    )?;
    dict.set_item(
        "raw_free_volume_a3",
        PyArray1::from_vec_bound(py, raw_free_volume_a3),
    )?;
    dict.set_item(
        "raw_free_volume_fraction",
        PyArray1::from_vec_bound(py, raw_free_volume_fraction.clone()),
    )?;
    dict.set_item(
        "raw_free_volume_percent",
        PyArray1::from_vec_bound(
            py,
            raw_free_volume_fraction
                .iter()
                .map(|value| *value * 100.0)
                .collect::<Vec<f32>>(),
        ),
    )?;
    dict.set_item(
        "fractional_free_volume",
        PyArray1::from_vec_bound(py, fractional_free_volume.clone()),
    )?;
    dict.set_item(
        "fractional_free_volume_percent",
        PyArray1::from_vec_bound(
            py,
            fractional_free_volume
                .iter()
                .map(|value| *value * 100.0)
                .collect::<Vec<f32>>(),
        ),
    )?;
    dict.set_item("density_g_cm3", PyArray1::from_vec_bound(py, density_g_cm3))?;
    dict.set_item("bondi_scale", bondi_scale)?;
    dict.set_item("molar_mass_dalton", molar_mass_dalton)?;
    dict.set_item("probe_radius", probe_radius)?;
    dict.set_item("ninsert_per_nm3", ninsert_per_nm3)?;
    dict.set_item("seed", seed)?;
    dict.set_item("reference", "Bondi 1964; Lourenco et al. 2013")?;
    Ok(dict.into_py(py))
}

pub(crate) fn clustering_to_py(
    py: Python<'_>,
    output: traj_engine::ClusteringOutput,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("labels", PyArray1::from_vec_bound(py, output.labels))?;
    dict.set_item("centroids", PyArray1::from_vec_bound(py, output.centroids))?;
    dict.set_item("sizes", PyArray1::from_vec_bound(py, output.sizes))?;
    dict.set_item("method", output.method)?;
    dict.set_item("n_frames", output.n_frames)?;
    Ok(dict.into_py(py))
}

pub(crate) fn parse_reference(value: &str) -> PyResult<ReferenceMode> {
    match value {
        "topology" => Ok(ReferenceMode::Topology),
        "frame0" => Ok(ReferenceMode::Frame0),
        _ => Err(PyRuntimeError::new_err(
            "reference must be 'topology' or 'frame0'",
        )),
    }
}

fn axis_name(axis: usize) -> &'static str {
    match axis {
        0 => "x",
        1 => "y",
        _ => "z",
    }
}

pub(crate) fn parse_axis(value: &str) -> PyResult<usize> {
    match value.trim().to_ascii_lowercase().as_str() {
        "x" => Ok(0),
        "y" => Ok(1),
        "z" => Ok(2),
        _ => Err(PyValueError::new_err("axis must be one of x, y, or z")),
    }
}

pub(crate) fn parse_density_map_unit(value: &str) -> PyResult<DensityMapUnit> {
    match value.trim().to_ascii_lowercase().as_str() {
        "nm-3" | "number_density" | "number" => Ok(DensityMapUnit::NumberDensity),
        "nm-2" | "area_density" | "column_density" => Ok(DensityMapUnit::AreaDensity),
        "count" => Ok(DensityMapUnit::Count),
        _ => Err(PyValueError::new_err(
            "densmap unit must be one of nm-3, nm-2, count",
        )),
    }
}

pub(crate) fn parse_pbc(value: &str) -> PyResult<PbcMode> {
    match value {
        "orthorhombic" => Ok(PbcMode::Orthorhombic),
        "none" => Ok(PbcMode::None),
        _ => Err(PyRuntimeError::new_err(
            "pbc must be 'orthorhombic' or 'none'",
        )),
    }
}

pub(crate) fn parse_pairwise_metric(value: &str) -> PyResult<PairwiseMetric> {
    match value {
        "rms" => Ok(PairwiseMetric::Rms),
        "nofit" => Ok(PairwiseMetric::Nofit),
        "dme" => Ok(PairwiseMetric::Dme),
        _ => Err(PyRuntimeError::new_err(
            "metric must be 'rms', 'nofit', or 'dme'",
        )),
    }
}

pub(crate) fn parse_matrix_mode(value: &str) -> PyResult<MatrixMode> {
    match value {
        "dist" => Ok(MatrixMode::Distance),
        "covar" => Ok(MatrixMode::Covariance),
        "mwcovar" => Ok(MatrixMode::MwCovariance),
        "correl" => Ok(MatrixMode::Correlation),
        _ => Err(PyRuntimeError::new_err(
            "mode must be 'dist', 'covar', 'mwcovar', or 'correl'",
        )),
    }
}

pub(crate) fn parse_group_by(value: &str) -> PyResult<GroupBy> {
    GroupBy::parse(value).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

pub(crate) fn parse_lag_mode(value: &str) -> PyResult<LagMode> {
    match value {
        "auto" => Ok(LagMode::Auto),
        "multi_tau" | "multi-tau" => Ok(LagMode::MultiTau),
        "ring" => Ok(LagMode::Ring),
        "fft" => Ok(LagMode::Fft),
        _ => Err(PyRuntimeError::new_err(
            "lag_mode must be 'auto', 'multi_tau', 'ring', or 'fft'",
        )),
    }
}
