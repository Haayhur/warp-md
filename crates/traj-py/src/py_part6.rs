#[pyclass]
struct PySuperposePlan {
    plan: RefCell<SuperposePlan>,
}

#[pymethods]
impl PySuperposePlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", mass=false, norotate=false))]
    fn new(selection: &PySelection, reference: &str, mass: bool, norotate: bool) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let plan = SuperposePlan::new(selection.selection.clone(), reference_mode, mass, norotate);
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
struct PyRotationMatrixPlan {
    plan: RefCell<RotationMatrixPlan>,
}

#[pymethods]
impl PyRotationMatrixPlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", mass=false))]
    fn new(selection: &PySelection, reference: &str, mass: bool) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let plan = RotationMatrixPlan::new(selection.selection.clone(), reference_mode, mass);
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
struct PyAlignPrincipalAxisPlan {
    plan: RefCell<AlignPrincipalAxisPlan>,
}

#[pymethods]
impl PyAlignPrincipalAxisPlan {
    #[new]
    #[pyo3(signature = (selection, mass=false))]
    fn new(selection: &PySelection, mass: bool) -> Self {
        let plan = AlignPrincipalAxisPlan::new(selection.selection.clone(), mass);
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
struct PyMsdPlan {
    plan: RefCell<MsdPlan>,
}

#[pymethods]
impl PyMsdPlan {
    #[new]
    #[pyo3(signature = (selection, group_by="resid", axis=None, length_scale=None, frame_decimation=None, dt_decimation=None, time_binning=None, group_types=None, lag_mode=None, max_lag=None, memory_budget_bytes=None, multi_tau_m=None, multi_tau_levels=None))]
    fn new(
        selection: &PySelection,
        group_by: &str,
        axis: Option<(f64, f64, f64)>,
        length_scale: Option<f64>,
        frame_decimation: Option<(usize, usize)>,
        dt_decimation: Option<(usize, usize, usize, usize)>,
        time_binning: Option<(f64, f64)>,
        group_types: Option<Vec<usize>>,
        lag_mode: Option<&str>,
        max_lag: Option<usize>,
        memory_budget_bytes: Option<usize>,
        multi_tau_m: Option<usize>,
        multi_tau_levels: Option<usize>,
    ) -> PyResult<Self> {
        let group_by = parse_group_by(group_by)?;
        let mut plan = MsdPlan::new(selection.selection.clone(), group_by);
        if let Some(axis) = axis {
            plan = plan.with_axis([axis.0, axis.1, axis.2]);
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
        if let Some((start, stride)) = frame_decimation {
            plan = plan.with_frame_decimation(FrameDecimation { start, stride });
        }
        if let Some((cut1, stride1, cut2, stride2)) = dt_decimation {
            plan = plan.with_dt_decimation(DtDecimation {
                cut1,
                stride1,
                cut2,
                stride2,
            });
        }
        if let Some((eps_num, eps_add)) = time_binning {
            plan = plan.with_time_binning(TimeBinning { eps_num, eps_add });
        }
        if let Some(types) = group_types {
            plan = plan.with_group_types(types);
        }
        if let Some(mode) = lag_mode {
            plan = plan.with_lag_mode(parse_lag_mode(mode)?);
        }
        if let Some(max_lag) = max_lag {
            plan = plan.with_max_lag(max_lag);
        }
        if let Some(budget) = memory_budget_bytes {
            plan = plan.with_memory_budget_bytes(budget);
        }
        if let Some(m) = multi_tau_m {
            plan = plan.with_multi_tau_m(m);
        }
        if let Some(levels) = multi_tau_levels {
            plan = plan.with_multi_tau_levels(levels);
        }
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
            PlanOutput::TimeSeries {
                time,
                data,
                rows,
                cols,
            } => timeseries_to_py(py, time, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyAtomicCorrPlan {
    plan: RefCell<AtomicCorrPlan>,
}

#[pymethods]
impl PyAtomicCorrPlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", lag_mode=None, max_lag=None, memory_budget_bytes=None, multi_tau_m=None, multi_tau_levels=None))]
    fn new(
        selection: &PySelection,
        reference: &str,
        lag_mode: Option<&str>,
        max_lag: Option<usize>,
        memory_budget_bytes: Option<usize>,
        multi_tau_m: Option<usize>,
        multi_tau_levels: Option<usize>,
    ) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let mut plan = AtomicCorrPlan::new(selection.selection.clone(), reference_mode);
        if let Some(mode) = lag_mode {
            plan = plan.with_lag_mode(parse_lag_mode(mode)?);
        }
        if let Some(max_lag) = max_lag {
            plan = plan.with_max_lag(max_lag);
        }
        if let Some(budget) = memory_budget_bytes {
            plan = plan.with_memory_budget_bytes(budget);
        }
        if let Some(m) = multi_tau_m {
            plan = plan.with_multi_tau_m(m);
        }
        if let Some(levels) = multi_tau_levels {
            plan = plan.with_multi_tau_levels(levels);
        }
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
            PlanOutput::TimeSeries {
                time,
                data,
                rows,
                cols,
            } => timeseries_to_py(py, time, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyVelocityAutoCorrPlan {
    plan: RefCell<VelocityAutoCorrPlan>,
}

#[pymethods]
impl PyVelocityAutoCorrPlan {
    #[new]
    #[pyo3(signature = (selection, normalize=false, include_zero=false, lag_mode=None, max_lag=None, memory_budget_bytes=None, multi_tau_m=None, multi_tau_levels=None))]
    fn new(
        selection: &PySelection,
        normalize: bool,
        include_zero: bool,
        lag_mode: Option<&str>,
        max_lag: Option<usize>,
        memory_budget_bytes: Option<usize>,
        multi_tau_m: Option<usize>,
        multi_tau_levels: Option<usize>,
    ) -> PyResult<Self> {
        let mut plan = VelocityAutoCorrPlan::new(selection.selection.clone())
            .with_normalize(normalize)
            .with_include_zero(include_zero);
        if let Some(mode) = lag_mode {
            plan = plan.with_lag_mode(parse_lag_mode(mode)?);
        }
        if let Some(max_lag) = max_lag {
            plan = plan.with_max_lag(max_lag);
        }
        if let Some(budget) = memory_budget_bytes {
            plan = plan.with_memory_budget_bytes(budget);
        }
        if let Some(m) = multi_tau_m {
            plan = plan.with_multi_tau_m(m);
        }
        if let Some(levels) = multi_tau_levels {
            plan = plan.with_multi_tau_levels(levels);
        }
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
            PlanOutput::TimeSeries {
                time,
                data,
                rows,
                cols,
            } => timeseries_to_py(py, time, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyXcorrPlan {
    plan: RefCell<XcorrPlan>,
}

#[pymethods]
impl PyXcorrPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, reference="topology", mass_weighted=false, lag_mode=None, max_lag=None, memory_budget_bytes=None, multi_tau_m=None, multi_tau_levels=None))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        reference: &str,
        mass_weighted: bool,
        lag_mode: Option<&str>,
        max_lag: Option<usize>,
        memory_budget_bytes: Option<usize>,
        multi_tau_m: Option<usize>,
        multi_tau_levels: Option<usize>,
    ) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let mut plan = XcorrPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            reference_mode,
            mass_weighted,
        );
        if let Some(mode) = lag_mode {
            plan = plan.with_lag_mode(parse_lag_mode(mode)?);
        }
        if let Some(max_lag) = max_lag {
            plan = plan.with_max_lag(max_lag);
        }
        if let Some(budget) = memory_budget_bytes {
            plan = plan.with_memory_budget_bytes(budget);
        }
        if let Some(m) = multi_tau_m {
            plan = plan.with_multi_tau_m(m);
        }
        if let Some(levels) = multi_tau_levels {
            plan = plan.with_multi_tau_levels(levels);
        }
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
            PlanOutput::TimeSeries {
                time,
                data,
                rows,
                cols,
            } => timeseries_to_py(py, time, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyWaveletPlan {
    plan: RefCell<WaveletPlan>,
}

#[pymethods]
impl PyWaveletPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, mass_weighted=false, pbc="none"))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        mass_weighted: bool,
        pbc: &str,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = WaveletPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            mass_weighted,
            pbc,
        );
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
struct PySurfPlan {
    plan: RefCell<SurfPlan>,
}

#[pymethods]
impl PySurfPlan {
    #[new]
    #[pyo3(signature = (selection, algorithm="bbox", probe_radius=1.4, n_sphere_points=64, radii=None))]
    fn new(
        selection: &PySelection,
        algorithm: &str,
        probe_radius: f64,
        n_sphere_points: usize,
        radii: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let algorithm = match algorithm {
            "bbox" => SurfAlgorithm::Bbox,
            "sasa" => SurfAlgorithm::Sasa,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "algorithm must be 'bbox' or 'sasa'",
                ))
            }
        };
        let radii = radii.map(|values| values.into_iter().map(|v| v as f32).collect());
        let plan = SurfPlan::new(selection.selection.clone())
            .with_algorithm(algorithm)
            .with_probe_radius(probe_radius as f32)
            .with_n_sphere_points(n_sphere_points)
            .with_radii(radii);
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
struct PyMolSurfPlan {
    plan: RefCell<MolSurfPlan>,
}

#[pymethods]
impl PyMolSurfPlan {
    #[new]
    #[pyo3(signature = (selection, algorithm="sasa", probe_radius=0.0, n_sphere_points=64, radii=None))]
    fn new(
        selection: &PySelection,
        algorithm: &str,
        probe_radius: f64,
        n_sphere_points: usize,
        radii: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let algorithm = match algorithm {
            "bbox" => SurfAlgorithm::Bbox,
            "sasa" => SurfAlgorithm::Sasa,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "algorithm must be 'bbox' or 'sasa'",
                ))
            }
        };
        let radii = radii.map(|values| values.into_iter().map(|v| v as f32).collect());
        let plan = MolSurfPlan::new(selection.selection.clone())
            .with_algorithm(algorithm)
            .with_probe_radius(probe_radius as f32)
            .with_n_sphere_points(n_sphere_points)
            .with_radii(radii);
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
struct PyTorsionDiffusionPlan {
    plan: RefCell<TorsionDiffusionPlan>,
}

#[pymethods]
impl PyTorsionDiffusionPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = TorsionDiffusionPlan::new(selection.selection.clone());
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
