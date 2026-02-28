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
        let plan = DihedralRmsPlan::new(defs, reference_mode, mass_weighted, pbc, degrees, range360);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}
