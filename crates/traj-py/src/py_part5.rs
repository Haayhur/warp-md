#[pyclass]
struct PyAutoImagePlan {
    plan: RefCell<AutoImagePlan>,
}

#[pymethods]
impl PyAutoImagePlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = AutoImagePlan::new(selection.selection.clone());
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
struct PyReplicateCellPlan {
    plan: RefCell<ReplicateCellPlan>,
}

#[pymethods]
impl PyReplicateCellPlan {
    #[new]
    #[pyo3(signature = (selection, repeats, pbc="orthorhombic"))]
    fn new(selection: &PySelection, repeats: (usize, usize, usize), pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        if !matches!(pbc, PbcMode::Orthorhombic) {
            return Err(PyRuntimeError::new_err(
                "replicate_cell requires pbc='orthorhombic'",
            ));
        }
        let plan = ReplicateCellPlan::new(
            selection.selection.clone(),
            [repeats.0, repeats.1, repeats.2],
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
struct PyVolumePlan {
    plan: RefCell<VolumePlan>,
}

#[pymethods]
impl PyVolumePlan {
    #[new]
    fn new() -> Self {
        let plan = VolumePlan::new();
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
struct PyXtalSymmPlan {
    plan: RefCell<XtalSymmPlan>,
}

fn parse_symmetry_ops(symmetry_ops: Option<Vec<Vec<f64>>>) -> PyResult<Option<Vec<[f64; 12]>>> {
    let Some(raw_ops) = symmetry_ops else {
        return Ok(None);
    };
    if raw_ops.is_empty() {
        return Err(PyRuntimeError::new_err(
            "symmetry_ops must contain at least one transform",
        ));
    }
    let mut ops = Vec::with_capacity(raw_ops.len());
    for op in raw_ops.into_iter() {
        let parsed = match op.len() {
            9 => [
                op[0], op[1], op[2], 0.0, op[3], op[4], op[5], 0.0, op[6], op[7], op[8], 0.0,
            ],
            12 => [
                op[0], op[1], op[2], op[3], op[4], op[5], op[6], op[7], op[8], op[9], op[10],
                op[11],
            ],
            16 => {
                if op[12].abs() > 1.0e-9
                    || op[13].abs() > 1.0e-9
                    || op[14].abs() > 1.0e-9
                    || (op[15] - 1.0).abs() > 1.0e-9
                {
                    return Err(PyRuntimeError::new_err(
                        "4x4 symmetry op must be affine with last row [0,0,0,1]",
                    ));
                }
                [
                    op[0], op[1], op[2], op[3], op[4], op[5], op[6], op[7], op[8], op[9], op[10],
                    op[11],
                ]
            }
            _ => {
                return Err(PyRuntimeError::new_err(
                    "each symmetry op must have 9 (3x3), 12 (3x4), or 16 (4x4) values",
                ))
            }
        };
        ops.push(parsed);
    }
    Ok(Some(ops))
}

#[pymethods]
impl PyXtalSymmPlan {
    #[new]
    #[pyo3(signature = (selection, repeats, pbc="orthorhombic", symmetry_ops=None))]
    fn new(
        selection: &PySelection,
        repeats: (usize, usize, usize),
        pbc: &str,
        symmetry_ops: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        let symmetry_ops = parse_symmetry_ops(symmetry_ops)?;
        let pbc = parse_pbc(pbc)?;
        if symmetry_ops.is_none() && !matches!(pbc, PbcMode::Orthorhombic) {
            return Err(PyRuntimeError::new_err(
                "xtalsymm requires pbc='orthorhombic'",
            ));
        }
        let plan = XtalSymmPlan::new(
            selection.selection.clone(),
            [repeats.0, repeats.1, repeats.2],
        )
        .with_symmetry_ops(symmetry_ops);
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
struct PyStripPlan {
    plan: RefCell<StripPlan>,
}

#[pymethods]
impl PyStripPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = StripPlan::new(selection.selection.clone());
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
struct PyMeanStructurePlan {
    plan: RefCell<MeanStructurePlan>,
}

#[pymethods]
impl PyMeanStructurePlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = MeanStructurePlan::new(selection.selection.clone());
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
struct PyAverageFramePlan {
    plan: RefCell<AverageFramePlan>,
}

#[pymethods]
impl PyAverageFramePlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = AverageFramePlan::new(selection.selection.clone());
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
struct PyMakeStructurePlan {
    plan: RefCell<MakeStructurePlan>,
}

#[pymethods]
impl PyMakeStructurePlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = MakeStructurePlan::new(selection.selection.clone());
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
struct PyCenterOfMassPlan {
    plan: RefCell<CenterOfMassPlan>,
}

#[pymethods]
impl PyCenterOfMassPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = CenterOfMassPlan::new(selection.selection.clone());
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
struct PyCenterOfGeometryPlan {
    plan: RefCell<CenterOfGeometryPlan>,
}

#[pymethods]
impl PyCenterOfGeometryPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = CenterOfGeometryPlan::new(selection.selection.clone());
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
struct PyDistanceToPointPlan {
    plan: RefCell<DistanceToPointPlan>,
}

#[pymethods]
impl PyDistanceToPointPlan {
    #[new]
    #[pyo3(signature = (selection, point, pbc="none"))]
    fn new(selection: &PySelection, point: (f64, f64, f64), pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = DistanceToPointPlan::new(
            selection.selection.clone(),
            [point.0, point.1, point.2],
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => matrix_to_py(py, data, rows, cols),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyDistanceToReferencePlan {
    plan: RefCell<DistanceToReferencePlan>,
}

#[pymethods]
impl PyDistanceToReferencePlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", pbc="none"))]
    fn new(selection: &PySelection, reference: &str, pbc: &str) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let pbc = parse_pbc(pbc)?;
        let plan = DistanceToReferencePlan::new(selection.selection.clone(), reference_mode, pbc);
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
struct PyPrincipalAxesPlan {
    plan: RefCell<PrincipalAxesPlan>,
}

#[pymethods]
impl PyPrincipalAxesPlan {
    #[new]
    #[pyo3(signature = (selection, mass_weighted=true))]
    fn new(selection: &PySelection, mass_weighted: bool) -> Self {
        let plan = PrincipalAxesPlan::new(selection.selection.clone(), mass_weighted);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => {
                if cols != 12 {
                    return Err(PyRuntimeError::new_err("unexpected principal axes output shape"));
                }
                let mut axes = Vec::with_capacity(rows * 9);
                let mut values = Vec::with_capacity(rows * 3);
                for r in 0..rows {
                    let start = r * cols;
                    axes.extend_from_slice(&data[start..start + 9]);
                    values.extend_from_slice(&data[start + 9..start + 12]);
                }
                let axes = matrix_to_py(py, axes, rows, 9)?;
                let values = matrix_to_py(py, values, rows, 3)?;
                Ok((axes, values).into_py(py))
            }
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyAlignPlan {
    plan: RefCell<AlignPlan>,
}

#[pymethods]
impl PyAlignPlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", mass=false, norotate=false))]
    fn new(selection: &PySelection, reference: &str, mass: bool, norotate: bool) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let plan = AlignPlan::new(selection.selection.clone(), reference_mode, mass, norotate);
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
