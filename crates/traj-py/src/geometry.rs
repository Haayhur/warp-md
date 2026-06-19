use super::*;

fn compute_lowestcurve_array(
    arr: numpy::ndarray::ArrayView2<'_, f64>,
    points: usize,
    step: f64,
) -> PyResult<Array2<f32>> {
    if points == 0 {
        return Err(PyValueError::new_err("points must be positive"));
    }
    if !step.is_finite() || step <= 0.0 {
        return Err(PyValueError::new_err("step must be positive"));
    }
    let shape = arr.shape();
    let (n_points, transposed) = match shape {
        [2, n] => (*n, false),
        [n, 2] => (*n, true),
        _ => {
            return Err(PyValueError::new_err(
                "data must have shape (2, N) or (N, 2)",
            ))
        }
    };
    if n_points == 0 {
        return Ok(Array2::<f32>::zeros((2, 0)));
    }

    let get_xy = |idx: usize| -> (f64, f64) {
        if transposed {
            (arr[[idx, 0]], arr[[idx, 1]])
        } else {
            (arr[[0, idx]], arr[[1, idx]])
        }
    };
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    for idx in 0..n_points {
        let (x, _) = get_xy(idx);
        if x < min_x {
            min_x = x;
        }
        if x > max_x {
            max_x = x;
        }
    }
    if !min_x.is_finite() || !max_x.is_finite() || min_x >= max_x {
        return Ok(Array2::<f32>::zeros((2, 0)));
    }

    let n_bins = ((max_x - min_x) / step).ceil().max(0.0) as usize;
    let mut out = Array2::<f32>::zeros((2, n_bins));
    for bin in 0..n_bins {
        let b0 = min_x + step * (bin as f64);
        let b1 = b0 + step;
        out[[0, bin]] = b0 as f32;
        let mut values = Vec::new();
        for idx in 0..n_points {
            let (x, y) = get_xy(idx);
            if x >= b0 && x < b1 {
                values.push(y);
            }
        }
        if values.is_empty() {
            continue;
        }
        values.sort_by(|a, b| a.total_cmp(b));
        let take = points.min(values.len());
        let sum: f64 = values.iter().take(take).sum();
        out[[1, bin]] = (sum / (take as f64)) as f32;
    }
    Ok(out)
}

#[pyfunction]
fn lowestcurve_array<'py>(
    py: Python<'py>,
    xy: PyReadonlyArray2<'_, f64>,
    points: usize,
    step: f64,
) -> PyResult<&'py PyArray2<f32>> {
    let out = compute_lowestcurve_array(xy.as_array(), points, step)?;
    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

fn compute_apply_orthorhombic_pbc_vectors(
    vectors: numpy::ndarray::ArrayView2<'_, f64>,
    box_lengths: numpy::ndarray::ArrayView2<'_, f64>,
) -> PyResult<Array2<f64>> {
    if vectors.ndim() != 2 || vectors.shape()[1] != 3 {
        return Err(PyValueError::new_err(
            "vectors must have shape (n_frames, 3)",
        ));
    }
    if box_lengths.ndim() != 2 || box_lengths.shape()[1] != 3 {
        return Err(PyValueError::new_err("box must have shape (n_frames, 3)"));
    }
    if vectors.shape()[0] != box_lengths.shape()[0] {
        return Err(PyValueError::new_err(
            "vectors and box must have the same frame count",
        ));
    }
    let n_frames = vectors.shape()[0];
    let mut out = vectors.to_owned();
    for frame in 0..n_frames {
        for axis in 0..3 {
            let length = box_lengths[[frame, axis]];
            if length > 0.0 {
                out[[frame, axis]] -= (out[[frame, axis]] / length).round_ties_even() * length;
            }
        }
    }
    Ok(out)
}

#[pyfunction]
fn apply_orthorhombic_pbc_vectors<'py>(
    py: Python<'py>,
    vectors: PyReadonlyArray2<'_, f64>,
    box_lengths: PyReadonlyArray2<'_, f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let out = compute_apply_orthorhombic_pbc_vectors(vectors.as_array(), box_lengths.as_array())?;
    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

fn compute_fluct_aggregate(
    values: numpy::ndarray::ArrayView1<'_, f32>,
    indices: numpy::ndarray::ArrayView1<'_, i64>,
    resids: Option<numpy::ndarray::ArrayView1<'_, i64>>,
    mode: &str,
) -> PyResult<Array2<f32>> {
    if values.len() != indices.len() {
        return Err(PyValueError::new_err(
            "values and indices must have the same length",
        ));
    }
    match mode {
        "bymask" => {
            let mean = if values.is_empty() {
                0.0
            } else {
                values.iter().map(|&v| v as f64).sum::<f64>() / values.len() as f64
            };
            let mut out = Array2::<f32>::zeros((1, 2));
            out[[0, 1]] = mean as f32;
            Ok(out)
        }
        "byres" => {
            let resids = resids
                .ok_or_else(|| PyValueError::new_err("resids are required for byres mode"))?;
            if resids.len() != values.len() {
                return Err(PyValueError::new_err(
                    "resids and values must have the same length",
                ));
            }
            let mut order = Vec::<i64>::new();
            let mut sums = Vec::<f64>::new();
            let mut counts = Vec::<usize>::new();
            for pos in 0..values.len() {
                let resid = resids[pos];
                let group = order.iter().position(|&existing| existing == resid);
                let idx = match group {
                    Some(idx) => idx,
                    None => {
                        order.push(resid);
                        sums.push(0.0);
                        counts.push(0);
                        order.len() - 1
                    }
                };
                sums[idx] += values[pos] as f64;
                counts[idx] += 1;
            }
            let mut out = Array2::<f32>::zeros((order.len(), 2));
            for row in 0..order.len() {
                out[[row, 0]] = order[row] as f32;
                out[[row, 1]] = if counts[row] > 0 {
                    (sums[row] / counts[row] as f64) as f32
                } else {
                    0.0
                };
            }
            Ok(out)
        }
        "byatom" => {
            let mut out = Array2::<f32>::zeros((values.len(), 2));
            for row in 0..values.len() {
                out[[row, 0]] = indices[row] as f32;
                out[[row, 1]] = values[row];
            }
            Ok(out)
        }
        _ => Err(PyValueError::new_err(
            "mode must be byatom, byres, or bymask",
        )),
    }
}

#[pyfunction]
#[pyo3(signature = (values, indices, resids=None, mode="byatom"))]
fn fluct_aggregate_array<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'_, f32>,
    indices: PyReadonlyArray1<'_, i64>,
    resids: Option<PyReadonlyArray1<'_, i64>>,
    mode: &str,
) -> PyResult<&'py PyArray2<f32>> {
    let resids_view = resids.as_ref().map(|arr| arr.as_array());
    let out = compute_fluct_aggregate(values.as_array(), indices.as_array(), resids_view, mode)?;
    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

fn compute_pair_distances_array(
    coords: numpy::ndarray::ArrayView2<'_, f64>,
    pbc: &str,
    box_lengths: Option<numpy::ndarray::ArrayView1<'_, f64>>,
) -> PyResult<Vec<f64>> {
    if coords.ndim() != 2 || coords.shape()[1] != 3 {
        return Err(PyValueError::new_err("coords must have shape (n_atoms, 3)"));
    }
    let use_pbc = match pbc {
        "none" => false,
        "orthorhombic" => true,
        _ => {
            return Err(PyValueError::new_err(
                "pbc must be 'none' or 'orthorhombic'",
            ))
        }
    };
    let box_lengths = if use_pbc {
        let view = box_lengths
            .ok_or_else(|| PyValueError::new_err("pbc='orthorhombic' requires box lengths"))?;
        if view.len() != 3 || view.iter().any(|&v| v == 0.0) {
            return Err(PyValueError::new_err(
                "pbc='orthorhombic' requires nonzero box lengths",
            ));
        }
        Some([view[0], view[1], view[2]])
    } else {
        None
    };
    let n_atoms = coords.shape()[0];
    let mut out = Vec::with_capacity(n_atoms.saturating_mul(n_atoms.saturating_sub(1)) / 2);
    for i in 0..n_atoms {
        for j in (i + 1)..n_atoms {
            let mut dx = coords[[j, 0]] - coords[[i, 0]];
            let mut dy = coords[[j, 1]] - coords[[i, 1]];
            let mut dz = coords[[j, 2]] - coords[[i, 2]];
            if let Some([lx, ly, lz]) = box_lengths {
                dx -= (dx / lx).round_ties_even() * lx;
                dy -= (dy / ly).round_ties_even() * ly;
                dz -= (dz / lz).round_ties_even() * lz;
            }
            out.push((dx * dx + dy * dy + dz * dz).sqrt());
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(signature = (coords, pbc="none", box_lengths=None))]
fn pair_distances_array<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'_, f64>,
    pbc: &str,
    box_lengths: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<&'py PyArray1<f64>> {
    let box_view = box_lengths.as_ref().map(|arr| arr.as_array());
    let out = compute_pair_distances_array(coords.as_array(), pbc, box_view)?;
    Ok(PyArray1::from_vec_bound(py, out).into_gil_ref())
}

fn compute_closest_gather_array(
    coords: numpy::ndarray::ArrayView3<'_, f32>,
    keep_idx: numpy::ndarray::ArrayView2<'_, i64>,
) -> PyResult<Array3<f32>> {
    if coords.ndim() != 3 || coords.shape()[2] != 3 {
        return Err(PyValueError::new_err(
            "coords must have shape (n_frames, n_atoms, 3)",
        ));
    }
    if keep_idx.ndim() != 2 {
        return Err(PyValueError::new_err(
            "keep_idx must have shape (n_frames, n_keep)",
        ));
    }
    let n_frames = coords.shape()[0];
    let n_atoms = coords.shape()[1];
    if keep_idx.shape()[0] != n_frames {
        return Err(PyValueError::new_err(
            "keep_idx frame count must match coords",
        ));
    }
    let n_keep = keep_idx.shape()[1];
    let mut out = Array3::<f32>::zeros((n_frames, n_keep, 3));
    for frame in 0..n_frames {
        for keep in 0..n_keep {
            let atom_idx = keep_idx[[frame, keep]];
            if atom_idx < 0 || atom_idx as usize >= n_atoms {
                return Err(PyValueError::new_err(
                    "keep_idx contains out-of-range atom index",
                ));
            }
            let atom_idx = atom_idx as usize;
            for axis in 0..3 {
                out[[frame, keep, axis]] = coords[[frame, atom_idx, axis]];
            }
        }
    }
    Ok(out)
}

#[pyfunction]
fn closest_gather_array<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray3<'_, f32>,
    keep_idx: PyReadonlyArray2<'_, i64>,
) -> PyResult<&'py PyArray3<f32>> {
    let out = compute_closest_gather_array(coords.as_array(), keep_idx.as_array())?;
    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

#[pyclass]
struct PyProjectionPlan {
    plan: RefCell<ProjectionPlan>,
}

#[pymethods]
impl PyProjectionPlan {
    #[new]
    #[pyo3(signature = (selection, eigenvectors, n_components, n_features, mean=None, mass_weighted=false))]
    fn new(
        selection: &PySelection,
        eigenvectors: Vec<f64>,
        n_components: usize,
        n_features: usize,
        mean: Option<Vec<f64>>,
        mass_weighted: bool,
    ) -> PyResult<Self> {
        let plan = ProjectionPlan::new(
            selection.selection.clone(),
            eigenvectors,
            n_components,
            n_features,
            mean,
            mass_weighted,
        )
        .map_err(to_py_err)?;
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyRmsdPerResPlan {
    plan: RefCell<RmsdPerResPlan>,
}

#[pyclass]
struct PyDridPlan {
    plan: RefCell<DridPlan>,
}

#[pymethods]
impl PyDridPlan {
    #[new]
    #[pyo3(signature = (selection, exclude_bonds=true))]
    fn new(selection: &PySelection, exclude_bonds: bool) -> Self {
        let plan = DridPlan::new(selection.selection.clone()).with_exclude_bonds(exclude_bonds);
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyRmsdPerResPlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", align=true))]
    fn new(selection: &PySelection, reference: &str, align: bool) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let plan = RmsdPerResPlan::new(selection.selection.clone(), reference_mode, align);
        Ok(Self {
            plan: RefCell::new(plan),
        })
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
        let output = {
            let mut plan = self.plan.borrow_mut();
            let mut traj_ref = traj.inner.borrow_mut();
            run_plan(
                &mut *plan,
                &mut traj_ref,
                &system.system.borrow(),
                chunk_frames,
                device,
            )?
        };
        let resids = self.plan.borrow().resids().to_vec();
        match output {
            PlanOutput::Matrix { data, rows, cols } => {
                let mat = matrix_to_py(py, data, rows, cols)?;
                let resid_arr = PyArray1::from_vec_bound(py, resids);
                Ok((resid_arr, mat).into_py(py))
            }
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyRmsfPlan {
    plan: RefCell<RmsfPlan>,
}

#[pymethods]
impl PyRmsfPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = RmsfPlan::new(selection.selection.clone());
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyBfactorsPlan {
    plan: RefCell<BfactorsPlan>,
}

#[pymethods]
impl PyBfactorsPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = BfactorsPlan::new(selection.selection.clone());
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyAtomicFluctPlan {
    plan: RefCell<AtomicFluctPlan>,
}

#[pymethods]
impl PyAtomicFluctPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = AtomicFluctPlan::new(selection.selection.clone());
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyAtomicAdpPlan {
    plan: RefCell<AtomicAdpPlan>,
}

#[pymethods]
impl PyAtomicAdpPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = AtomicAdpPlan::new(selection.selection.clone());
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyDistancePlan {
    plan: RefCell<DistancePlan>,
}

#[pyclass]
struct PyDistanceVectorPlan {
    plan: RefCell<DistanceVectorPlan>,
}

#[pyclass]
struct PyMultiDistancePlan {
    plan: RefCell<MultiDistancePlan>,
}

#[pyclass]
struct PyDistanceCenterToPointPlan {
    plan: RefCell<DistanceCenterToPointPlan>,
}

#[pyclass]
struct PyDistanceCenterToReferencePlan {
    plan: RefCell<DistanceCenterToReferencePlan>,
}

#[pymethods]
impl PyDistancePlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, mass_weighted=true, pbc="none"))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        mass_weighted: bool,
        pbc: &str,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = DistancePlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            mass_weighted,
            pbc,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyDistanceVectorPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, mass_weighted=true, pbc="none"))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        mass_weighted: bool,
        pbc: &str,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = DistanceVectorPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            mass_weighted,
            pbc,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyMultiDistancePlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, mass_weighted=true, pbc="none"))]
    fn new(
        sel_a: Vec<Vec<u32>>,
        sel_b: Vec<Vec<u32>>,
        mass_weighted: bool,
        pbc: &str,
    ) -> PyResult<Self> {
        if sel_a.len() != sel_b.len() {
            return Err(PyValueError::new_err(
                "multi-distance selections must have matching lengths",
            ));
        }
        let pbc = parse_pbc(pbc)?;
        let pairs = sel_a
            .into_iter()
            .zip(sel_b)
            .map(|(left, right)| {
                (
                    Selection {
                        expr: "__indices__".to_string(),
                        indices: Arc::new(left),
                    },
                    Selection {
                        expr: "__indices__".to_string(),
                        indices: Arc::new(right),
                    },
                )
            })
            .collect();
        Ok(Self {
            plan: RefCell::new(MultiDistancePlan::new(pairs, mass_weighted, pbc)),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyDistanceCenterToPointPlan {
    #[new]
    #[pyo3(signature = (selection, point, mass_weighted=true, pbc="none"))]
    fn new(
        selection: &PySelection,
        point: (f64, f64, f64),
        mass_weighted: bool,
        pbc: &str,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = DistanceCenterToPointPlan::new(
            selection.selection.clone(),
            [point.0, point.1, point.2],
            mass_weighted,
            pbc,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyDistanceCenterToReferencePlan {
    #[new]
    #[pyo3(signature = (selection, reference="frame0", mass_weighted=true, pbc="none"))]
    fn new(
        selection: &PySelection,
        reference: &str,
        mass_weighted: bool,
        pbc: &str,
    ) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let pbc = parse_pbc(pbc)?;
        let plan = DistanceCenterToReferencePlan::new(
            selection.selection.clone(),
            reference_mode,
            mass_weighted,
            pbc,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLowestCurvePlan {
    plan: RefCell<LowestCurvePlan>,
}

#[pymethods]
impl PyLowestCurvePlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, mass_weighted=true, pbc="none"))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        mass_weighted: bool,
        pbc: &str,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = LowestCurvePlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            mass_weighted,
            pbc,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyVectorPlan {
    plan: RefCell<VectorPlan>,
}

#[pyclass]
struct PyMultiVectorPlan {
    plan: RefCell<MultiVectorPlan>,
}

#[pymethods]
impl PyVectorPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, mass_weighted=true, pbc="none"))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        mass_weighted: bool,
        pbc: &str,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = VectorPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            mass_weighted,
            pbc,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyMultiVectorPlan {
    #[new]
    fn new(commands: Vec<(Vec<u32>, Vec<u32>, bool, String, bool)>) -> PyResult<Self> {
        let mut native_commands = Vec::with_capacity(commands.len());
        for (sel_a, sel_b, mass_weighted, pbc, is_center) in commands {
            let selection_a = Selection {
                expr: String::new(),
                indices: Arc::new(sel_a),
            };
            if is_center {
                native_commands.push(MultiVectorCommand::Center {
                    selection: selection_a,
                    mass_weighted,
                });
            } else {
                let selection_b = Selection {
                    expr: String::new(),
                    indices: Arc::new(sel_b),
                };
                native_commands.push(MultiVectorCommand::Between {
                    sel_a: selection_a,
                    sel_b: selection_b,
                    mass_weighted,
                    pbc: parse_pbc(&pbc)?,
                });
            }
        }
        Ok(Self {
            plan: RefCell::new(MultiVectorPlan::new(native_commands)),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyGetVelocityPlan {
    plan: RefCell<GetVelocityPlan>,
}

#[pymethods]
impl PyGetVelocityPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = GetVelocityPlan::new(selection.selection.clone());
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PySetVelocityPlan {
    plan: RefCell<SetVelocityPlan>,
}

#[pyclass]
struct PyRunningAveragePlan {
    plan: RefCell<RunningAveragePlan>,
}

#[pyclass]
struct PyRunningAverageTrajectoryPlan {
    plan: RefCell<RunningAverageTrajectoryPlan>,
}

#[pymethods]
impl PyRunningAveragePlan {
    #[new]
    #[pyo3(signature = (selection, window=0))]
    fn new(selection: &PySelection, window: usize) -> Self {
        let plan = RunningAveragePlan::new(selection.selection.clone(), window);
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyRunningAverageTrajectoryPlan {
    #[new]
    #[pyo3(signature = (selection, window=0))]
    fn new(selection: &PySelection, window: usize) -> Self {
        let plan = RunningAverageTrajectoryPlan::new(selection.selection.clone(), window);
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<PyObject> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Trajectory(output) => trajectory_output_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PySetVelocityPlan {
    #[new]
    #[pyo3(signature = (selection, temperature=298.0, seed=10))]
    fn new(selection: &PySelection, temperature: f64, seed: u64) -> Self {
        let plan = SetVelocityPlan::new(selection.selection.clone(), temperature, seed);
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyPairwiseDistancePlan {
    plan: RefCell<PairwiseDistancePlan>,
}

#[pyclass]
struct PyPairListDistancePlan {
    plan: RefCell<PairListDistancePlan>,
}

#[pymethods]
impl PyPairwiseDistancePlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, pbc="none"))]
    fn new(sel_a: &PySelection, sel_b: &PySelection, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = PairwiseDistancePlan::new(sel_a.selection.clone(), sel_b.selection.clone(), pbc);
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray2<f32>> {
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
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyPairListDistancePlan {
    #[new]
    #[pyo3(signature = (pairs, pbc="none"))]
    fn new(pairs: PyReadonlyArray2<'_, i64>, pbc: &str) -> PyResult<Self> {
        let arr = pairs.as_array();
        let shape = arr.shape();
        if shape.len() != 2 || shape[1] != 2 {
            return Err(PyValueError::new_err(
                "pair-list distance pairs must have shape (n_pairs, 2)",
            ));
        }
        let mut parsed = Vec::with_capacity(shape[0]);
        for row in arr.outer_iter() {
            let left = row[0];
            let right = row[1];
            if left < 0 || right < 0 {
                return Err(PyValueError::new_err(
                    "pair-list distance atom indices must be >= 0",
                ));
            }
            parsed.push((left as u32, right as u32));
        }
        let pbc = parse_pbc(pbc)?;
        Ok(Self {
            plan: RefCell::new(PairListDistancePlan::new(parsed, pbc)),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyAnglePlan {
    plan: RefCell<AnglePlan>,
}

#[pyclass]
struct PyMultiAnglePlan {
    plan: RefCell<MultiAnglePlan>,
}

#[pymethods]
impl PyAnglePlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, sel_c, mass_weighted=false, pbc="none", degrees=true))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        sel_c: &PySelection,
        mass_weighted: bool,
        pbc: &str,
        degrees: bool,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = AnglePlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            sel_c.selection.clone(),
            mass_weighted,
            pbc,
            degrees,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyMultiAnglePlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, sel_c, mass_weighted=false, pbc="none", degrees=true))]
    fn new(
        sel_a: Vec<Vec<u32>>,
        sel_b: Vec<Vec<u32>>,
        sel_c: Vec<Vec<u32>>,
        mass_weighted: bool,
        pbc: &str,
        degrees: bool,
    ) -> PyResult<Self> {
        if sel_a.len() != sel_b.len() || sel_a.len() != sel_c.len() {
            return Err(PyValueError::new_err(
                "multi-angle selections must have matching lengths",
            ));
        }
        let pbc = parse_pbc(pbc)?;
        let angles = sel_a
            .into_iter()
            .zip(sel_b)
            .zip(sel_c)
            .map(|((left, center), right)| {
                (
                    Selection {
                        expr: "__indices__".to_string(),
                        indices: Arc::new(left),
                    },
                    Selection {
                        expr: "__indices__".to_string(),
                        indices: Arc::new(center),
                    },
                    Selection {
                        expr: "__indices__".to_string(),
                        indices: Arc::new(right),
                    },
                )
            })
            .collect();
        Ok(Self {
            plan: RefCell::new(MultiAnglePlan::new(angles, mass_weighted, pbc, degrees)),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyDihedralPlan {
    plan: RefCell<DihedralPlan>,
}

#[pymethods]
impl PyDihedralPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, sel_c, sel_d, mass_weighted=false, pbc="none", degrees=true, range360=false))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        sel_c: &PySelection,
        sel_d: &PySelection,
        mass_weighted: bool,
        pbc: &str,
        degrees: bool,
        range360: bool,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = DihedralPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            sel_c.selection.clone(),
            sel_d.selection.clone(),
            mass_weighted,
            pbc,
            degrees,
            range360,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}
#[pyclass]
struct PyMultiDihedralPlan {
    plan: RefCell<MultiDihedralPlan>,
}

#[pymethods]
impl PyMultiDihedralPlan {
    #[new]
    #[pyo3(signature = (groups, mass_weighted=false, pbc="none", degrees=true, range360=false))]
    fn new(
        py: Python<'_>,
        groups: Vec<(
            Py<PySelection>,
            Py<PySelection>,
            Py<PySelection>,
            Py<PySelection>,
        )>,
        mass_weighted: bool,
        pbc: &str,
        degrees: bool,
        range360: bool,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let mut defs = Vec::with_capacity(groups.len());
        for (a, b, c, d) in groups {
            let a_sel = a.borrow(py).selection.clone();
            let b_sel = b.borrow(py).selection.clone();
            let c_sel = c.borrow(py).selection.clone();
            let d_sel = d.borrow(py).selection.clone();
            defs.push((a_sel, b_sel, c_sel, d_sel));
        }
        let plan = MultiDihedralPlan::new(defs, mass_weighted, pbc, degrees, range360);
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyDihedralRmsPlan {
    plan: RefCell<DihedralRmsPlan>,
}

#[pymethods]
impl PyDihedralRmsPlan {
    #[new]
    #[pyo3(signature = (groups, reference="topology", mass_weighted=false, pbc="none", degrees=true, range360=false))]
    fn new(
        py: Python<'_>,
        groups: Vec<(
            Py<PySelection>,
            Py<PySelection>,
            Py<PySelection>,
            Py<PySelection>,
        )>,
        reference: &str,
        mass_weighted: bool,
        pbc: &str,
        degrees: bool,
        range360: bool,
    ) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let pbc = parse_pbc(pbc)?;
        let mut defs = Vec::with_capacity(groups.len());
        for (a, b, c, d) in groups {
            let a_sel = a.borrow(py).selection.clone();
            let b_sel = b.borrow(py).selection.clone();
            let c_sel = c.borrow(py).selection.clone();
            let d_sel = d.borrow(py).selection.clone();
            defs.push((a_sel, b_sel, c_sel, d_sel));
        }
        let plan =
            DihedralRmsPlan::new(defs, reference_mode, mass_weighted, pbc, degrees, range360);
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyPuckerPlan {
    plan: RefCell<PuckerPlan>,
    return_phase: bool,
}

#[pymethods]
impl PyPuckerPlan {
    #[new]
    #[pyo3(signature = (selection, metric="max_radius", return_phase=false))]
    fn new(selection: &PySelection, metric: &str, return_phase: bool) -> PyResult<Self> {
        let metric = match metric.to_ascii_lowercase().as_str() {
            "max_radius" => PuckerMetric::MaxRadius,
            "amplitude" => PuckerMetric::Amplitude,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "metric must be 'max_radius' or 'amplitude'",
                ))
            }
        };
        let plan = PuckerPlan::new(selection.selection.clone())
            .with_metric(metric)
            .with_return_phase(return_phase);
        Ok(Self {
            plan: RefCell::new(plan),
            return_phase,
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<PyObject> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_py(py)),
            PlanOutput::Matrix { data, rows, cols } if self.return_phase && cols == 2 => {
                let mut values = Vec::with_capacity(rows);
                let mut phases = Vec::with_capacity(rows);
                for frame in 0..rows {
                    values.push(data[frame * 2]);
                    phases.push(data[frame * 2 + 1]);
                }
                let v = PyArray1::from_vec_bound(py, values).into_py(py);
                let p = PyArray1::from_vec_bound(py, phases).into_py(py);
                Ok((v, p).into_py(py))
            }
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyRotateDihedralPlan {
    plan: RefCell<RotateDihedralPlan>,
}

#[pymethods]
impl PyRotateDihedralPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, sel_c, sel_d, rotate_selection, angle, mass_weighted=false, degrees=true))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        sel_c: &PySelection,
        sel_d: &PySelection,
        rotate_selection: &PySelection,
        angle: f64,
        mass_weighted: bool,
        degrees: bool,
    ) -> Self {
        let plan = RotateDihedralPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            sel_c.selection.clone(),
            sel_d.selection.clone(),
            rotate_selection.selection.clone(),
            angle,
            mass_weighted,
            degrees,
        );
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyRotateDihedralTrajectoryPlan {
    plan: RefCell<RotateDihedralTrajectoryPlan>,
}

#[pymethods]
impl PyRotateDihedralTrajectoryPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, sel_c, sel_d, rotate_selection, angle, mass_weighted=false, degrees=true))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        sel_c: &PySelection,
        sel_d: &PySelection,
        rotate_selection: &PySelection,
        angle: f64,
        mass_weighted: bool,
        degrees: bool,
    ) -> Self {
        let plan = RotateDihedralTrajectoryPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            sel_c.selection.clone(),
            sel_d.selection.clone(),
            rotate_selection.selection.clone(),
            angle,
            mass_weighted,
            degrees,
        );
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<PyObject> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Trajectory(output) => trajectory_output_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PySetDihedralPlan {
    plan: RefCell<SetDihedralPlan>,
}

#[pymethods]
impl PySetDihedralPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, sel_c, sel_d, rotate_selection, target, mass_weighted=false, pbc="none", degrees=true, range360=false))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        sel_c: &PySelection,
        sel_d: &PySelection,
        rotate_selection: &PySelection,
        target: f64,
        mass_weighted: bool,
        pbc: &str,
        degrees: bool,
        range360: bool,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = SetDihedralPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            sel_c.selection.clone(),
            sel_d.selection.clone(),
            rotate_selection.selection.clone(),
            target,
            mass_weighted,
            pbc,
            degrees,
            range360,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyCheckChiralityPlan {
    plan: RefCell<CheckChiralityPlan>,
}

#[pymethods]
impl PyCheckChiralityPlan {
    #[new]
    #[pyo3(signature = (groups, mass_weighted=false))]
    fn new(
        py: Python<'_>,
        groups: Vec<(
            Py<PySelection>,
            Py<PySelection>,
            Py<PySelection>,
            Py<PySelection>,
        )>,
        mass_weighted: bool,
    ) -> PyResult<Self> {
        let mut defs = Vec::with_capacity(groups.len());
        for (a, b, c, d) in groups {
            let a_sel = a.borrow(py).selection.clone();
            let b_sel = b.borrow(py).selection.clone();
            let c_sel = c.borrow(py).selection.clone();
            let d_sel = d.borrow(py).selection.clone();
            defs.push((a_sel, b_sel, c_sel, d_sel));
        }
        let plan = CheckChiralityPlan::new(defs, mass_weighted);
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyMindistPlan {
    plan: RefCell<MindistPlan>,
}

#[pymethods]
impl PyMindistPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, pbc="none"))]
    fn new(sel_a: &PySelection, sel_b: &PySelection, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = MindistPlan::new(sel_a.selection.clone(), sel_b.selection.clone(), pbc);
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyHausdorffPlan {
    plan: RefCell<HausdorffPlan>,
}

#[pymethods]
impl PyHausdorffPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, pbc="none"))]
    fn new(sel_a: &PySelection, sel_b: &PySelection, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = HausdorffPlan::new(sel_a.selection.clone(), sel_b.selection.clone(), pbc);
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyCheckStructurePlan {
    plan: RefCell<CheckStructurePlan>,
}

#[pymethods]
impl PyCheckStructurePlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = CheckStructurePlan::new(selection.selection.clone());
        Self {
            plan: RefCell::new(plan),
        }
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<&'py PyArray1<f32>> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyAtomMapPlan {
    plan: RefCell<AtomMapPlan>,
}

#[pymethods]
impl PyAtomMapPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, pbc="none"))]
    fn new(sel_a: &PySelection, sel_b: &PySelection, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = AtomMapPlan::new(sel_a.selection.clone(), sel_b.selection.clone(), pbc);
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray2<f32>> {
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
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyPermuteDihedralsPlan {
    plan: RefCell<PermuteDihedralsPlan>,
}

#[pymethods]
impl PyPermuteDihedralsPlan {
    #[new]
    #[pyo3(signature = (groups, mass_weighted=false, pbc="none", degrees=true, range360=false))]
    fn new(
        py: Python<'_>,
        groups: Vec<(
            Py<PySelection>,
            Py<PySelection>,
            Py<PySelection>,
            Py<PySelection>,
        )>,
        mass_weighted: bool,
        pbc: &str,
        degrees: bool,
        range360: bool,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let mut defs = Vec::with_capacity(groups.len());
        for (a, b, c, d) in groups {
            let a_sel = a.borrow(py).selection.clone();
            let b_sel = b.borrow(py).selection.clone();
            let c_sel = c.borrow(py).selection.clone();
            let d_sel = d.borrow(py).selection.clone();
            defs.push((a_sel, b_sel, c_sel, d_sel));
        }
        let plan = PermuteDihedralsPlan::new(defs, mass_weighted, pbc, degrees, range360);
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray2<f32>> {
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
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PySetDihedralTrajectoryPlan {
    plan: RefCell<SetDihedralTrajectoryPlan>,
}

#[pymethods]
impl PySetDihedralTrajectoryPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, sel_c, sel_d, rotate_selection, target, mass_weighted=false, pbc="none", degrees=true, range360=false))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        sel_c: &PySelection,
        sel_d: &PySelection,
        rotate_selection: &PySelection,
        target: f64,
        mass_weighted: bool,
        pbc: &str,
        degrees: bool,
        range360: bool,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = SetDihedralTrajectoryPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            sel_c.selection.clone(),
            sel_d.selection.clone(),
            rotate_selection.selection.clone(),
            target,
            mass_weighted,
            pbc,
            degrees,
            range360,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
        frame_indices: Option<Vec<i64>>,
    ) -> PyResult<PyObject> {
        let mut plan = self.plan.borrow_mut();
        let mut traj_ref = traj.inner.borrow_mut();
        let output = run_plan_with_frame_subset(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
            frame_indices,
        )?;
        match output {
            PlanOutput::Trajectory(output) => trajectory_output_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

fn trajectory_output_to_py<'py>(
    py: Python<'py>,
    output: traj_engine::TrajectoryOutput,
) -> PyResult<PyObject> {
    let coords = Array3::from_shape_vec((output.frames, output.atoms, 3), output.coords)
        .map_err(|_| PyRuntimeError::new_err("failed to build trajectory coords"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("coords", coords.into_pyarray_bound(py))?;

    let mut box_data = Vec::with_capacity(output.frames * 3);
    let mut box_ok = output.box_.len() == output.frames && output.frames > 0;
    if box_ok {
        for box_ in output.box_.iter() {
            match *box_ {
                Box3::Orthorhombic { lx, ly, lz } => {
                    box_data.extend_from_slice(&[lx, ly, lz]);
                }
                Box3::None | Box3::Triclinic { .. } => {
                    box_ok = false;
                    break;
                }
            }
        }
    }
    if box_ok {
        let box_arr = Array2::from_shape_vec((output.frames, 3), box_data)
            .map_err(|_| PyRuntimeError::new_err("failed to build trajectory box"))?;
        dict.set_item("box", box_arr.into_pyarray_bound(py))?;
    } else {
        dict.set_item("box", py.None())?;
    }

    if output.time.len() == output.frames {
        dict.set_item("time_ps", PyArray1::from_vec_bound(py, output.time))?;
    } else {
        dict.set_item("time_ps", py.None())?;
    }
    Ok(dict.into_py(py))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowestcurve_array_matches_python_contract() {
        let xy = numpy::ndarray::arr2(&[
            [0.00, 1.00],
            [0.05, 2.00],
            [0.15, 0.50],
            [0.25, 5.00],
            [0.35, 1.50],
        ]);
        let out = compute_lowestcurve_array(xy.view(), 2, 0.2).unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert!((out[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((out[[0, 1]] - 0.2).abs() < 1e-6);
        assert!((out[[1, 0]] - 0.75).abs() < 1e-6);
        assert!((out[[1, 1]] - 3.25).abs() < 1e-6);
    }

    #[test]
    fn lowestcurve_array_accepts_transposed_xy() {
        let xy = numpy::ndarray::arr2(&[[0.0, 0.2, 0.4], [1.0, 0.5, 2.0]]);
        let xy_t = xy.t();
        let out = compute_lowestcurve_array(xy.view(), 1, 0.2).unwrap();
        let out_t = compute_lowestcurve_array(xy_t, 1, 0.2).unwrap();
        assert_eq!(out, out_t);
    }

    #[test]
    fn apply_orthorhombic_pbc_vectors_wraps_components() {
        let vectors = numpy::ndarray::arr2(&[[6.0, -6.0, 2.0], [2.5, 7.6, -7.6]]);
        let boxes = numpy::ndarray::arr2(&[[10.0, 10.0, 0.0], [5.0, 5.0, 5.0]]);
        let out = compute_apply_orthorhombic_pbc_vectors(vectors.view(), boxes.view()).unwrap();
        assert_eq!(out[[0, 0]], -4.0);
        assert_eq!(out[[0, 1]], 4.0);
        assert_eq!(out[[0, 2]], 2.0);
        assert_eq!(out[[1, 0]], 2.5);
        assert!((out[[1, 1]] + 2.4).abs() < 1e-12);
        assert!((out[[1, 2]] - 2.4).abs() < 1e-12);
    }

    #[test]
    fn fluct_aggregate_byatom_and_bymask() {
        let values = numpy::ndarray::arr1(&[1.0_f32, 3.0]);
        let indices = numpy::ndarray::arr1(&[4_i64, 8]);
        let byatom =
            compute_fluct_aggregate(values.view(), indices.view(), None, "byatom").unwrap();
        assert_eq!(byatom[[0, 0]], 4.0);
        assert_eq!(byatom[[0, 1]], 1.0);
        assert_eq!(byatom[[1, 0]], 8.0);
        assert_eq!(byatom[[1, 1]], 3.0);
        let bymask =
            compute_fluct_aggregate(values.view(), indices.view(), None, "bymask").unwrap();
        assert_eq!(bymask.shape(), &[1, 2]);
        assert_eq!(bymask[[0, 0]], 0.0);
        assert_eq!(bymask[[0, 1]], 2.0);
    }

    #[test]
    fn fluct_aggregate_byres_preserves_first_residue_order() {
        let values = numpy::ndarray::arr1(&[1.0_f32, 3.0, 5.0]);
        let indices = numpy::ndarray::arr1(&[0_i64, 1, 2]);
        let resids = numpy::ndarray::arr1(&[7_i64, 7, 3]);
        let out =
            compute_fluct_aggregate(values.view(), indices.view(), Some(resids.view()), "byres")
                .unwrap();
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out[[0, 0]], 7.0);
        assert_eq!(out[[0, 1]], 2.0);
        assert_eq!(out[[1, 0]], 3.0);
        assert_eq!(out[[1, 1]], 5.0);
    }

    #[test]
    fn pair_distances_array_no_pbc() {
        let coords = numpy::ndarray::arr2(&[[0.0_f64, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]]);
        let out = compute_pair_distances_array(coords.view(), "none", None).unwrap();
        assert_eq!(out.len(), 3);
        assert!((out[0] - 3.0).abs() < 1e-12);
        assert!((out[1] - 4.0).abs() < 1e-12);
        assert!((out[2] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn pair_distances_array_orthorhombic_pbc() {
        let coords = numpy::ndarray::arr2(&[[0.0_f64, 0.0, 0.0], [9.0, 0.0, 0.0]]);
        let box_lengths = numpy::ndarray::arr1(&[10.0_f64, 10.0, 10.0]);
        let out =
            compute_pair_distances_array(coords.view(), "orthorhombic", Some(box_lengths.view()))
                .unwrap();
        assert_eq!(out.len(), 1);
        assert!((out[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn closest_gather_array_selects_per_frame_indices() {
        let coords = numpy::ndarray::arr3(&[
            [[0.0_f32, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[3.0_f32, 0.0, 0.0], [4.0, 0.0, 0.0], [5.0, 0.0, 0.0]],
        ]);
        let keep = numpy::ndarray::arr2(&[[2_i64, 0], [1, 2]]);
        let out = compute_closest_gather_array(coords.view(), keep.view()).unwrap();
        assert_eq!(out.shape(), &[2, 2, 3]);
        assert_eq!(out[[0, 0, 0]], 2.0);
        assert_eq!(out[[0, 1, 0]], 0.0);
        assert_eq!(out[[1, 0, 0]], 4.0);
        assert_eq!(out[[1, 1, 0]], 5.0);
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(lowestcurve_array, m)?)?;
    m.add_function(wrap_pyfunction!(apply_orthorhombic_pbc_vectors, m)?)?;
    m.add_function(wrap_pyfunction!(fluct_aggregate_array, m)?)?;
    m.add_function(wrap_pyfunction!(pair_distances_array, m)?)?;
    m.add_function(wrap_pyfunction!(closest_gather_array, m)?)?;
    m.add_class::<PyProjectionPlan>()?;
    m.add_class::<PyRmsdPerResPlan>()?;
    m.add_class::<PyDridPlan>()?;
    m.add_class::<PyRmsfPlan>()?;
    m.add_class::<PyBfactorsPlan>()?;
    m.add_class::<PyAtomicFluctPlan>()?;
    m.add_class::<PyAtomicAdpPlan>()?;
    m.add_class::<PyDistancePlan>()?;
    m.add_class::<PyDistanceVectorPlan>()?;
    m.add_class::<PyMultiDistancePlan>()?;
    m.add_class::<PyDistanceCenterToPointPlan>()?;
    m.add_class::<PyDistanceCenterToReferencePlan>()?;
    m.add_class::<PyLowestCurvePlan>()?;
    m.add_class::<PyVectorPlan>()?;
    m.add_class::<PyMultiVectorPlan>()?;
    m.add_class::<PyGetVelocityPlan>()?;
    m.add_class::<PyRunningAveragePlan>()?;
    m.add_class::<PyRunningAverageTrajectoryPlan>()?;
    m.add_class::<PySetVelocityPlan>()?;
    m.add_class::<PyPairwiseDistancePlan>()?;
    m.add_class::<PyPairListDistancePlan>()?;
    m.add_class::<PyAnglePlan>()?;
    m.add_class::<PyMultiAnglePlan>()?;
    m.add_class::<PyDihedralPlan>()?;
    m.add_class::<PyMultiDihedralPlan>()?;
    m.add_class::<PyDihedralRmsPlan>()?;
    m.add_class::<PyPuckerPlan>()?;
    m.add_class::<PyRotateDihedralPlan>()?;
    m.add_class::<PyRotateDihedralTrajectoryPlan>()?;
    m.add_class::<PySetDihedralPlan>()?;
    m.add_class::<PySetDihedralTrajectoryPlan>()?;
    m.add_class::<PyCheckChiralityPlan>()?;
    m.add_class::<PyMindistPlan>()?;
    m.add_class::<PyHausdorffPlan>()?;
    m.add_class::<PyCheckStructurePlan>()?;
    m.add_class::<PyAtomMapPlan>()?;
    m.add_class::<PyPermuteDihedralsPlan>()?;
    Ok(())
}
