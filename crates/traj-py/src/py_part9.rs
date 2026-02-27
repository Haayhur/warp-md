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
            warp_pack::pack::AtomRecordKind::Atom => "ATOM".to_string(),
            warp_pack::pack::AtomRecordKind::HetAtom => "HETATM".to_string(),
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

fn get_attr_or_item<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
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

fn get_attr_or_item_opt<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if let Ok(val) = obj.getattr(name) {
        return Ok(Some(val));
    }
    if let Ok(val) = obj.get_item(name) {
        return Ok(Some(val));
    }
    Ok(None)
}

fn extract_coords(any: &Bound<'_, PyAny>) -> PyResult<Vec<warp_pack::geom::Vec3>> {
    if let Ok(array) = any.extract::<numpy::PyReadonlyArray2<f32>>() {
        let view = array.as_array();
        let shape = view.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(PyRuntimeError::new_err("coords must be (N, 3) array"));
        }
        let mut coords = Vec::with_capacity(shape[0]);
        for row in view.outer_iter() {
            coords.push(warp_pack::geom::Vec3::new(row[0], row[1], row[2]));
        }
        return Ok(coords);
    }
    if let Ok(vec) = any.extract::<Vec<[f32; 3]>>() {
        return Ok(vec
            .into_iter()
            .map(|v| warp_pack::geom::Vec3::new(v[0], v[1], v[2]))
            .collect());
    }
    if let Ok(vec) = any.extract::<Vec<(f32, f32, f32)>>() {
        return Ok(vec
            .into_iter()
            .map(|v| warp_pack::geom::Vec3::new(v.0, v.1, v.2))
            .collect());
    }
    if let Ok(vec) = any.extract::<Vec<Vec<f32>>>() {
        let mut coords = Vec::with_capacity(vec.len());
        for row in vec {
            if row.len() != 3 {
                return Err(PyRuntimeError::new_err("coords rows must have 3 values"));
            }
            coords.push(warp_pack::geom::Vec3::new(row[0], row[1], row[2]));
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

fn build_pack_output(result: &Bound<'_, PyAny>) -> PyResult<warp_pack::pack::PackOutput> {
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
                    warp_pack::pack::AtomRecordKind::HetAtom
                } else {
                    warp_pack::pack::AtomRecordKind::Atom
                }
            })
            .unwrap_or(warp_pack::pack::AtomRecordKind::Atom);
        atoms.push(warp_pack::pack::AtomRecord {
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

    Ok(warp_pack::pack::PackOutput {
        atoms,
        bonds,
        box_size,
        ter_after,
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
    _py: Python<'_>,
    result: &Bound<'_, PyAny>,
    format: &str,
    path: &str,
    scale: f32,
    add_box_sides: bool,
    box_sides_fix: f32,
    write_conect: bool,
    hexadecimal_indices: bool,
) -> PyResult<()> {
    let out = build_pack_output(result)?;
    let spec = warp_pack::config::OutputSpec {
        path: path.to_string(),
        format: format.to_string(),
        scale: Some(scale),
    };
    warp_pack::io::write_output(
        &out,
        &spec,
        add_box_sides,
        box_sides_fix,
        write_conect,
        hexadecimal_indices,
    )
    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn traj_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "__rust_build_profile__",
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
    )?;
    m.add("__rust_cuda_enabled__", cfg!(feature = "cuda"))?;
    m.add_class::<PySystem>()?;
    m.add_class::<PySelection>()?;
    m.add_class::<PyTrajectory>()?;
    m.add_class::<PyRgPlan>()?;
    m.add_class::<PyRadgyrTensorPlan>()?;
    m.add_class::<PyRmsdPlan>()?;
    m.add_class::<PySymmRmsdPlan>()?;
    m.add_class::<PyDistanceRmsdPlan>()?;
    m.add_class::<PyPairwiseRmsdPlan>()?;
    m.add_class::<PyTrajectoryClusterPlan>()?;
    m.add_class::<PyMatrixPlan>()?;
    m.add_class::<PyPcaPlan>()?;
    m.add_class::<PyAnalyzeModesPlan>()?;
    m.add_class::<PyProjectionPlan>()?;
    m.add_class::<PyRmsdPerResPlan>()?;
    m.add_class::<PyRmsfPlan>()?;
    m.add_class::<PyBfactorsPlan>()?;
    m.add_class::<PyAtomicFluctPlan>()?;
    m.add_class::<PyDistancePlan>()?;
    m.add_class::<PyLowestCurvePlan>()?;
    m.add_class::<PyVectorPlan>()?;
    m.add_class::<PyGetVelocityPlan>()?;
    m.add_class::<PySetVelocityPlan>()?;
    m.add_class::<PyPairwiseDistancePlan>()?;
    m.add_class::<PyAnglePlan>()?;
    m.add_class::<PyDihedralPlan>()?;
    m.add_class::<PyMultiDihedralPlan>()?;
    m.add_class::<PyPermuteDihedralsPlan>()?;
    m.add_class::<PyDihedralRmsPlan>()?;
    m.add_class::<PyPuckerPlan>()?;
    m.add_class::<PyRotateDihedralPlan>()?;
    m.add_class::<PySetDihedralPlan>()?;
    m.add_class::<PyCheckChiralityPlan>()?;
    m.add_class::<PyMindistPlan>()?;
    m.add_class::<PyHausdorffPlan>()?;
    m.add_class::<PyCheckStructurePlan>()?;
    m.add_class::<PyAtomMapPlan>()?;
    m.add_class::<PyFixImageBondsPlan>()?;
    m.add_class::<PyRandomizeIonsPlan>()?;
    m.add_class::<PyClosestAtomPlan>()?;
    m.add_class::<PySearchNeighborsPlan>()?;
    m.add_class::<PyWatershellPlan>()?;
    m.add_class::<PyClosestPlan>()?;
    m.add_class::<PyNativeContactsPlan>()?;
    m.add_class::<PyCenterTrajectoryPlan>()?;
    m.add_class::<PyTranslatePlan>()?;
    m.add_class::<PyTransformPlan>()?;
    m.add_class::<PyRotatePlan>()?;
    m.add_class::<PyScalePlan>()?;
    m.add_class::<PyImagePlan>()?;
    m.add_class::<PyAutoImagePlan>()?;
    m.add_class::<PyReplicateCellPlan>()?;
    m.add_class::<PyXtalSymmPlan>()?;
    m.add_class::<PyVolumePlan>()?;
    m.add_class::<PyStripPlan>()?;
    m.add_class::<PyMeanStructurePlan>()?;
    m.add_class::<PyAverageFramePlan>()?;
    m.add_class::<PyMakeStructurePlan>()?;
    m.add_class::<PyCenterOfMassPlan>()?;
    m.add_class::<PyCenterOfGeometryPlan>()?;
    m.add_class::<PyDistanceToPointPlan>()?;
    m.add_class::<PyDistanceToReferencePlan>()?;
    m.add_class::<PyPrincipalAxesPlan>()?;
    m.add_class::<PyAlignPlan>()?;
    m.add_class::<PySuperposePlan>()?;
    m.add_class::<PyRotationMatrixPlan>()?;
    m.add_class::<PyAlignPrincipalAxisPlan>()?;
    m.add_class::<PyMsdPlan>()?;
    m.add_class::<PyAtomicCorrPlan>()?;
    m.add_class::<PyVelocityAutoCorrPlan>()?;
    m.add_class::<PyXcorrPlan>()?;
    m.add_class::<PyWaveletPlan>()?;
    m.add_class::<PySurfPlan>()?;
    m.add_class::<PyMolSurfPlan>()?;
    m.add_class::<PyTorsionDiffusionPlan>()?;
    m.add_class::<PyToroidalDiffusionPlan>()?;
    m.add_class::<PyMultiPuckerPlan>()?;
    m.add_class::<PyNmrIredPlan>()?;
    m.add_class::<PyRotAcfPlan>()?;
    m.add_class::<PyConductivityPlan>()?;
    m.add_class::<PyDielectricPlan>()?;
    m.add_class::<PyDipoleAlignmentPlan>()?;
    m.add_class::<PyIonPairCorrelationPlan>()?;
    m.add_class::<PyStructureFactorPlan>()?;
    m.add_class::<PyDockingPlan>()?;
    m.add_class::<PyDsspPlan>()?;
    m.add_class::<PyGistGridPlan>()?;
    m.add_class::<PyGistDirectPlan>()?;
    m.add_class::<PyWaterCountPlan>()?;
    m.add_class::<PyCountInVoxelPlan>()?;
    m.add_class::<PyDensityPlan>()?;
    m.add_class::<PyVolmapPlan>()?;
    m.add_class::<PyEquipartitionPlan>()?;
    m.add_class::<PyHbondPlan>()?;
    m.add_class::<PyRdfPlan>()?;
    m.add_class::<PyPairDistPlan>()?;
    m.add_class::<PyEndToEndPlan>()?;
    m.add_class::<PyContourLengthPlan>()?;
    m.add_class::<PyChainRgPlan>()?;
    m.add_class::<PyBondLengthDistributionPlan>()?;
    m.add_class::<PyBondAngleDistributionPlan>()?;
    m.add_class::<PyPersistenceLengthPlan>()?;
    m.add_function(wrap_pyfunction!(pack_from_json, m)?)?;
    m.add_function(wrap_pyfunction!(pack_from_inp, m)?)?;
    m.add_function(wrap_pyfunction!(pack_config_from_inp, m)?)?;
    m.add_function(wrap_pyfunction!(crank_delta, m)?)?;
    m.add_function(wrap_pyfunction!(rotdif_fit, m)?)?;
    m.add_function(wrap_pyfunction!(gist_apply_pme_scaling, m)?)?;
    m.add_function(wrap_pyfunction!(pack_write_output, m)?)?;
    Ok(())
}

fn to_py_err(err: TrajError) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

fn run_plan<P: Plan>(
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
        TrajKind::Pdb { reader, .. } => exec.run_plan(plan, reader),
    }
    .map_err(to_py_err)?;
    Ok(output)
}

fn reset_traj(traj: &mut TrajKind) -> TrajResult<()> {
    match traj {
        TrajKind::Dcd { reader, .. } => reader.reset(),
        TrajKind::Xtc { reader, .. } => reader.reset(),
        TrajKind::Pdb { reader, .. } => {
            reader.reset();
            Ok(())
        }
    }
}

fn run_plan_with_frame_subset<P: Plan>(
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
                TrajKind::Pdb { reader, .. } => traj_engine::count_frames(reader, max_frames),
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
        TrajKind::Pdb { reader, .. } => {
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
        TrajKind::Pdb { reader, .. } => reader.n_atoms(),
    }
}

fn traj_n_frames_hint(traj: &TrajKind) -> Option<usize> {
    match traj {
        TrajKind::Dcd { reader, .. } => reader.n_frames_hint(),
        TrajKind::Xtc { reader, .. } => reader.n_frames_hint(),
        TrajKind::Pdb { reader, .. } => reader.n_frames_hint(),
    }
}

fn heuristic_chunk_frames(n_atoms: usize) -> usize {
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

fn resolve_chunk_frames_for_streaming(traj: &TrajKind) -> PyResult<usize> {
    let n_atoms = traj_n_atoms(traj);
    Ok(heuristic_chunk_frames(n_atoms))
}

fn matrix_to_py<'py>(
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

fn timeseries_to_py(
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

fn hist_to_py(py: Python<'_>, centers: Vec<f32>, counts: Vec<u64>) -> PyObject {
    let centers = PyArray1::from_vec_bound(py, centers);
    let counts = PyArray1::from_vec_bound(py, counts);
    (centers, counts).into_py(py)
}

fn pca_to_py(py: Python<'_>, output: traj_engine::PcaOutput) -> PyResult<PyObject> {
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

fn persistence_to_py(py: Python<'_>, output: traj_engine::PersistenceOutput) -> PyResult<PyObject> {
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

fn dielectric_to_py(py: Python<'_>, output: traj_engine::DielectricOutput) -> PyResult<PyObject> {
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

fn structure_factor_to_py(
    py: Python<'_>,
    output: traj_engine::StructureFactorOutput,
) -> PyResult<PyObject> {
    let r = PyArray1::from_vec_bound(py, output.r);
    let g = PyArray1::from_vec_bound(py, output.g_r);
    let q = PyArray1::from_vec_bound(py, output.q);
    let s = PyArray1::from_vec_bound(py, output.s_q);
    Ok((r, g, q, s).into_py(py))
}

fn grid_to_py(py: Python<'_>, output: traj_engine::GridOutput) -> PyResult<PyObject> {
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

fn clustering_to_py(
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

fn parse_reference(value: &str) -> PyResult<ReferenceMode> {
    match value {
        "topology" => Ok(ReferenceMode::Topology),
        "frame0" => Ok(ReferenceMode::Frame0),
        _ => Err(PyRuntimeError::new_err(
            "reference must be 'topology' or 'frame0'",
        )),
    }
}

fn parse_pbc(value: &str) -> PyResult<PbcMode> {
    match value {
        "orthorhombic" => Ok(PbcMode::Orthorhombic),
        "none" => Ok(PbcMode::None),
        _ => Err(PyRuntimeError::new_err(
            "pbc must be 'orthorhombic' or 'none'",
        )),
    }
}

fn parse_pairwise_metric(value: &str) -> PyResult<PairwiseMetric> {
    match value {
        "rms" => Ok(PairwiseMetric::Rms),
        "nofit" => Ok(PairwiseMetric::Nofit),
        "dme" => Ok(PairwiseMetric::Dme),
        _ => Err(PyRuntimeError::new_err(
            "metric must be 'rms', 'nofit', or 'dme'",
        )),
    }
}

fn parse_matrix_mode(value: &str) -> PyResult<MatrixMode> {
    match value {
        "dist" => Ok(MatrixMode::Distance),
        "covar" => Ok(MatrixMode::Covariance),
        "mwcovar" => Ok(MatrixMode::MwCovariance),
        _ => Err(PyRuntimeError::new_err(
            "mode must be 'dist', 'covar', or 'mwcovar'",
        )),
    }
}

fn parse_group_by(value: &str) -> PyResult<GroupBy> {
    GroupBy::parse(value).map_err(|err| PyRuntimeError::new_err(err.to_string()))
}

fn parse_lag_mode(value: &str) -> PyResult<LagMode> {
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
