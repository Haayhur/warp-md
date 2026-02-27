#[pyclass]
struct PyProjectionPlan {
    plan: RefCell<ProjectionPlan>,
}

#[pymethods]
impl PyProjectionPlan {
    #[new]
    #[pyo3(signature = (selection, eigenvectors, n_components, n_features, mean=None))]
    fn new(
        selection: &PySelection,
        eigenvectors: Vec<f64>,
        n_components: usize,
        n_features: usize,
        mean: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        let plan = ProjectionPlan::new(
            selection.selection.clone(),
            eigenvectors,
            n_components,
            n_features,
            mean,
        )
        .map_err(to_py_err)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
            run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyDistancePlan {
    plan: RefCell<DistancePlan>,
}

#[pymethods]
impl PyDistancePlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, mass_weighted=true, pbc="none"))]
    fn new(sel_a: &PySelection, sel_b: &PySelection, mass_weighted: bool, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = DistancePlan::new(sel_a.selection.clone(), sel_b.selection.clone(), mass_weighted, pbc);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
    fn new(sel_a: &PySelection, sel_b: &PySelection, mass_weighted: bool, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = LowestCurvePlan::new(sel_a.selection.clone(), sel_b.selection.clone(), mass_weighted, pbc);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
    fn new(sel_a: &PySelection, sel_b: &PySelection, mass_weighted: bool, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = VectorPlan::new(sel_a.selection.clone(), sel_b.selection.clone(), mass_weighted, pbc);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}
