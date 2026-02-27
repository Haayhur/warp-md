#[pyclass]
struct PyFixImageBondsPlan {
    plan: RefCell<FixImageBondsPlan>,
}

#[pymethods]
impl PyFixImageBondsPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = FixImageBondsPlan::new(selection.selection.clone());
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
struct PyRandomizeIonsPlan {
    plan: RefCell<RandomizeIonsPlan>,
}

#[pymethods]
impl PyRandomizeIonsPlan {
    #[new]
    #[pyo3(signature = (selection, seed=None, around=None, by=0.0, overlap=0.0, noimage=false, max_attempts=1000))]
    fn new(
        selection: &PySelection,
        seed: Option<u64>,
        around: Option<&PySelection>,
        by: f64,
        overlap: f64,
        noimage: bool,
        max_attempts: usize,
    ) -> Self {
        let seed = seed.unwrap_or(0);
        let mut plan = RandomizeIonsPlan::new(selection.selection.clone(), seed);
        let around_sel = around.map(|sel| sel.selection.clone());
        plan = plan.with_around(around_sel, by, overlap, noimage);
        plan = plan.with_max_attempts(max_attempts);
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
struct PyClosestAtomPlan {
    plan: RefCell<ClosestAtomPlan>,
}

#[pymethods]
impl PyClosestAtomPlan {
    #[new]
    #[pyo3(signature = (selection, point, pbc="none"))]
    fn new(selection: &PySelection, point: (f64, f64, f64), pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = ClosestAtomPlan::new(
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
struct PySearchNeighborsPlan {
    plan: RefCell<SearchNeighborsPlan>,
}

#[pymethods]
impl PySearchNeighborsPlan {
    #[new]
    #[pyo3(signature = (target, probe, cutoff, pbc="none"))]
    fn new(target: &PySelection, probe: &PySelection, cutoff: f64, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = SearchNeighborsPlan::new(
            target.selection.clone(),
            probe.selection.clone(),
            cutoff,
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyWatershellPlan {
    plan: RefCell<WatershellPlan>,
}

#[pymethods]
impl PyWatershellPlan {
    #[new]
    #[pyo3(signature = (target, probe, cutoff, pbc="none"))]
    fn new(target: &PySelection, probe: &PySelection, cutoff: f64, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = WatershellPlan::new(
            target.selection.clone(),
            probe.selection.clone(),
            cutoff,
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyClosestPlan {
    plan: RefCell<ClosestPlan>,
}

#[pymethods]
impl PyClosestPlan {
    #[new]
    #[pyo3(signature = (target, probe, n_solvents, pbc="none"))]
    fn new(target: &PySelection, probe: &PySelection, n_solvents: usize, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = ClosestPlan::new(
            target.selection.clone(),
            probe.selection.clone(),
            n_solvents,
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
struct PyNativeContactsPlan {
    plan: RefCell<NativeContactsPlan>,
}

#[pymethods]
impl PyNativeContactsPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b=None, reference="frame0", cutoff=7.0, pbc="none"))]
    fn new(
        sel_a: &PySelection,
        sel_b: Option<&PySelection>,
        reference: &str,
        cutoff: f64,
        pbc: &str,
    ) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let pbc = parse_pbc(pbc)?;
        let sel_b = sel_b
            .map(|s| s.selection.clone())
            .unwrap_or_else(|| sel_a.selection.clone());
        let plan = NativeContactsPlan::new(
            sel_a.selection.clone(),
            sel_b,
            reference_mode,
            cutoff,
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Series(values) => Ok(PyArray1::from_vec_bound(py, values).into_gil_ref()),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyCenterTrajectoryPlan {
    plan: RefCell<CenterTrajectoryPlan>,
}

#[pymethods]
impl PyCenterTrajectoryPlan {
    #[new]
    #[pyo3(signature = (selection, mode="box", point=None, mass_weighted=false))]
    fn new(
        selection: &PySelection,
        mode: &str,
        point: Option<(f64, f64, f64)>,
        mass_weighted: bool,
    ) -> PyResult<Self> {
        let (mode_enum, center) = match mode.to_ascii_lowercase().as_str() {
            "origin" => (CenterMode::Origin, [0.0, 0.0, 0.0]),
            "point" => {
                let p = point.ok_or_else(|| PyRuntimeError::new_err("point required for mode=point"))?;
                (CenterMode::Point, [p.0, p.1, p.2])
            }
            "box" => (CenterMode::Box, [0.0, 0.0, 0.0]),
            _ => {
                return Err(PyRuntimeError::new_err(
                    "mode must be one of: 'box', 'origin', 'point'",
                ))
            }
        };
        let plan = CenterTrajectoryPlan::new(
            selection.selection.clone(),
            center,
            mode_enum,
            mass_weighted,
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
struct PyTranslatePlan {
    plan: RefCell<TranslatePlan>,
}

#[pymethods]
impl PyTranslatePlan {
    #[new]
    fn new(delta: (f64, f64, f64)) -> Self {
        let plan = TranslatePlan::new([delta.0, delta.1, delta.2]);
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
struct PyTransformPlan {
    plan: RefCell<TransformPlan>,
}

#[pymethods]
impl PyTransformPlan {
    #[new]
    fn new(rotation: (f64, f64, f64, f64, f64, f64, f64, f64, f64), translation: (f64, f64, f64)) -> Self {
        let rot = [
            rotation.0, rotation.1, rotation.2,
            rotation.3, rotation.4, rotation.5,
            rotation.6, rotation.7, rotation.8,
        ];
        let plan = TransformPlan::new(rot, [translation.0, translation.1, translation.2]);
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
struct PyRotatePlan {
    plan: RefCell<RotatePlan>,
}

#[pymethods]
impl PyRotatePlan {
    #[new]
    fn new(rotation: (f64, f64, f64, f64, f64, f64, f64, f64, f64)) -> Self {
        let rot = [
            rotation.0, rotation.1, rotation.2,
            rotation.3, rotation.4, rotation.5,
            rotation.6, rotation.7, rotation.8,
        ];
        let plan = RotatePlan::new(rot);
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
struct PyScalePlan {
    plan: RefCell<ScalePlan>,
}

#[pymethods]
impl PyScalePlan {
    #[new]
    fn new(scale: f64) -> Self {
        let plan = ScalePlan::new(scale);
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
struct PyImagePlan {
    plan: RefCell<ImagePlan>,
}

#[pymethods]
impl PyImagePlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = ImagePlan::new(selection.selection.clone());
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
