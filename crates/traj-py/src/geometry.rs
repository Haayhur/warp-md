use super::*;

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

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray1<f32>> {
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
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyVectorPlan {
    plan: RefCell<VectorPlan>,
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
struct PySetVelocityPlan {
    plan: RefCell<SetVelocityPlan>,
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

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray1<f32>> {
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
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
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

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray1<f32>> {
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

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray1<f32>> {
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

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray1<f32>> {
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

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto"))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        traj: &PyTrajectory,
        system: &PySystem,
        chunk_frames: Option<usize>,
        device: &str,
    ) -> PyResult<&'py PyArray1<f32>> {
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProjectionPlan>()?;
    m.add_class::<PyRmsdPerResPlan>()?;
    m.add_class::<PyRmsfPlan>()?;
    m.add_class::<PyBfactorsPlan>()?;
    m.add_class::<PyAtomicFluctPlan>()?;
    m.add_class::<PyAtomicAdpPlan>()?;
    m.add_class::<PyDistancePlan>()?;
    m.add_class::<PyMultiDistancePlan>()?;
    m.add_class::<PyDistanceCenterToPointPlan>()?;
    m.add_class::<PyDistanceCenterToReferencePlan>()?;
    m.add_class::<PyLowestCurvePlan>()?;
    m.add_class::<PyVectorPlan>()?;
    m.add_class::<PyGetVelocityPlan>()?;
    m.add_class::<PySetVelocityPlan>()?;
    m.add_class::<PyPairwiseDistancePlan>()?;
    m.add_class::<PyPairListDistancePlan>()?;
    m.add_class::<PyAnglePlan>()?;
    m.add_class::<PyDihedralPlan>()?;
    m.add_class::<PyMultiDihedralPlan>()?;
    m.add_class::<PyDihedralRmsPlan>()?;
    m.add_class::<PyPuckerPlan>()?;
    m.add_class::<PyRotateDihedralPlan>()?;
    m.add_class::<PySetDihedralPlan>()?;
    m.add_class::<PyCheckChiralityPlan>()?;
    m.add_class::<PyMindistPlan>()?;
    m.add_class::<PyHausdorffPlan>()?;
    m.add_class::<PyCheckStructurePlan>()?;
    m.add_class::<PyAtomMapPlan>()?;
    m.add_class::<PyPermuteDihedralsPlan>()?;
    Ok(())
}
