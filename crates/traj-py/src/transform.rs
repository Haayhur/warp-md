use super::*;

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
struct PyFixImageBondsTrajectoryPlan {
    plan: RefCell<FixImageBondsTrajectoryPlan>,
}

#[pymethods]
impl PyFixImageBondsTrajectoryPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = FixImageBondsTrajectoryPlan::new(selection.selection.clone());
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
struct PyRandomizeIonsTrajectoryPlan {
    plan: RefCell<RandomizeIonsTrajectoryPlan>,
}

#[pymethods]
impl PyRandomizeIonsTrajectoryPlan {
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
        let mut plan = RandomizeIonsTrajectoryPlan::new(selection.selection.clone(), seed);
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
struct PySearchNeighborListPlan {
    plan: RefCell<SearchNeighborListPlan>,
}

#[pymethods]
impl PySearchNeighborListPlan {
    #[new]
    #[pyo3(signature = (target, probe, cutoff, pbc="none"))]
    fn new(target: &PySelection, probe: &PySelection, cutoff: f64, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = SearchNeighborListPlan::new(
            target.selection.clone(),
            probe.selection.clone(),
            cutoff,
            pbc,
        );
        Ok(Self {
            plan: RefCell::new(plan),
        })
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run(
        &self,
        py: Python<'_>,
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
            PlanOutput::NeighborList(output) => {
                let dict = PyDict::new_bound(py);
                dict.set_item("offsets", PyArray1::from_vec_bound(py, output.offsets))?;
                dict.set_item("indices", PyArray1::from_vec_bound(py, output.indices))?;
                dict.set_item("counts", PyArray1::from_vec_bound(py, output.counts))?;
                dict.set_item("frames", output.frames)?;
                Ok(dict.into_py(py))
            }
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
struct PyClosestPlan {
    plan: RefCell<ClosestPlan>,
}

#[pymethods]
impl PyClosestPlan {
    #[new]
    #[pyo3(signature = (target, probe, n_solvents, pbc="none"))]
    fn new(
        target: &PySelection,
        probe: &PySelection,
        n_solvents: usize,
        pbc: &str,
    ) -> PyResult<Self> {
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
struct PyClosestCoordsPlan {
    plan: RefCell<ClosestCoordsPlan>,
}

#[pymethods]
impl PyClosestCoordsPlan {
    #[new]
    #[pyo3(signature = (target, probe, n_solvents, pbc="none"))]
    fn new(
        target: &PySelection,
        probe: &PySelection,
        n_solvents: usize,
        pbc: &str,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = ClosestCoordsPlan::new(
            target.selection.clone(),
            probe.selection.clone(),
            n_solvents,
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
            PlanOutput::Trajectory(output) => {
                let rows = output.frames;
                let cols = output.atoms * 3;
                let coords = Array2::from_shape_vec((rows, cols), output.coords)
                    .map_err(|_| PyRuntimeError::new_err("failed to build closest coords"))?;
                let dict = PyDict::new_bound(py);
                dict.set_item("coords", coords.into_pyarray_bound(py))?;

                let mut box_data = Vec::with_capacity(rows * 3);
                let mut box_ok = output.box_.len() == rows && rows > 0;
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
                    let box_arr = Array2::from_shape_vec((rows, 3), box_data)
                        .map_err(|_| PyRuntimeError::new_err("failed to build closest box"))?;
                    dict.set_item("box", box_arr.into_pyarray_bound(py))?;
                } else {
                    dict.set_item("box", py.None())?;
                }

                if output.time.len() == rows {
                    dict.set_item("time_ps", PyArray1::from_vec_bound(py, output.time))?;
                } else {
                    dict.set_item("time_ps", py.None())?;
                }
                Ok(dict.into_py(py))
            }
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
    #[pyo3(signature = (sel_a, sel_b=None, reference="frame0", cutoff=7.0, pbc="none", min_cutoff=None))]
    fn new(
        sel_a: &PySelection,
        sel_b: Option<&PySelection>,
        reference: &str,
        cutoff: f64,
        pbc: &str,
        min_cutoff: Option<f64>,
    ) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let pbc = parse_pbc(pbc)?;
        let sel_b = sel_b
            .map(|s| s.selection.clone())
            .unwrap_or_else(|| sel_a.selection.clone());
        let plan =
            NativeContactsPlan::new(sel_a.selection.clone(), sel_b, reference_mode, cutoff, pbc)
                .with_min_cutoff(min_cutoff);
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
struct PyCenterTrajectoryPlan {
    plan: RefCell<CenterTrajectoryPlan>,
}

fn parse_center_mode(
    mode: &str,
    point: Option<(f64, f64, f64)>,
) -> PyResult<(CenterMode, [f64; 3])> {
    match mode.to_ascii_lowercase().as_str() {
        "origin" => Ok((CenterMode::Origin, [0.0, 0.0, 0.0])),
        "point" => {
            let p =
                point.ok_or_else(|| PyRuntimeError::new_err("point required for mode=point"))?;
            Ok((CenterMode::Point, [p.0, p.1, p.2]))
        }
        "box" => Ok((CenterMode::Box, [0.0, 0.0, 0.0])),
        _ => Err(PyRuntimeError::new_err(
            "mode must be one of: 'box', 'origin', 'point'",
        )),
    }
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
        let (mode_enum, center) = parse_center_mode(mode, point)?;
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
struct PyCenterTrajectoryOutputPlan {
    plan: RefCell<CenterTrajectoryOutputPlan>,
}

#[pymethods]
impl PyCenterTrajectoryOutputPlan {
    #[new]
    #[pyo3(signature = (selection, mode="box", point=None, mass_weighted=false))]
    fn new(
        selection: &PySelection,
        mode: &str,
        point: Option<(f64, f64, f64)>,
        mass_weighted: bool,
    ) -> PyResult<Self> {
        let (mode_enum, center) = parse_center_mode(mode, point)?;
        let plan = CenterTrajectoryOutputPlan::new(
            selection.selection.clone(),
            center,
            mode_enum,
            mass_weighted,
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
struct PyTransformPlan {
    plan: RefCell<TransformPlan>,
}

#[pymethods]
impl PyTransformPlan {
    #[new]
    fn new(
        rotation: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
        translation: (f64, f64, f64),
    ) -> Self {
        let rot = [
            rotation.0, rotation.1, rotation.2, rotation.3, rotation.4, rotation.5, rotation.6,
            rotation.7, rotation.8,
        ];
        let plan = TransformPlan::new(rot, [translation.0, translation.1, translation.2]);
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
struct PyTransformTrajectoryPlan {
    plan: RefCell<TransformTrajectoryPlan>,
}

#[pymethods]
impl PyTransformTrajectoryPlan {
    #[new]
    fn new(
        rotation: (f64, f64, f64, f64, f64, f64, f64, f64, f64),
        translation: (f64, f64, f64),
    ) -> Self {
        let rot = [
            rotation.0, rotation.1, rotation.2, rotation.3, rotation.4, rotation.5, rotation.6,
            rotation.7, rotation.8,
        ];
        let plan = TransformTrajectoryPlan::new(rot, [translation.0, translation.1, translation.2]);
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
struct PyRotatePlan {
    plan: RefCell<RotatePlan>,
}

#[pymethods]
impl PyRotatePlan {
    #[new]
    fn new(rotation: (f64, f64, f64, f64, f64, f64, f64, f64, f64)) -> Self {
        let rot = [
            rotation.0, rotation.1, rotation.2, rotation.3, rotation.4, rotation.5, rotation.6,
            rotation.7, rotation.8,
        ];
        let plan = RotatePlan::new(rot);
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
struct PyImageTrajectoryPlan {
    plan: RefCell<ImageTrajectoryPlan>,
}

#[pymethods]
impl PyImageTrajectoryPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = ImageTrajectoryPlan::new(selection.selection.clone());
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
struct PyAutoImageTrajectoryPlan {
    plan: RefCell<AutoImageTrajectoryPlan>,
}

#[pymethods]
impl PyAutoImageTrajectoryPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = AutoImageTrajectoryPlan::new(selection.selection.clone());
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

#[pyclass]
struct PyStripTrajectoryPlan {
    plan: RefCell<StripTrajectoryPlan>,
}

#[pymethods]
impl PyStripTrajectoryPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = StripTrajectoryPlan::new(selection.selection.clone());
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
        let output = run_plan(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
        )?;
        match output {
            PlanOutput::Matrix { data, rows, cols } => {
                if cols != 12 {
                    return Err(PyRuntimeError::new_err(
                        "unexpected principal axes output shape",
                    ));
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

#[pyclass]
struct PySuperposeTrajectoryPlan {
    plan: RefCell<SuperposeTrajectoryPlan>,
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
impl PySuperposeTrajectoryPlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", mass=false, norotate=false))]
    fn new(selection: &PySelection, reference: &str, mass: bool, norotate: bool) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let plan = SuperposeTrajectoryPlan::new(
            selection.selection.clone(),
            reference_mode,
            mass,
            norotate,
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFixImageBondsPlan>()?;
    m.add_class::<PyFixImageBondsTrajectoryPlan>()?;
    m.add_class::<PyRandomizeIonsPlan>()?;
    m.add_class::<PyRandomizeIonsTrajectoryPlan>()?;
    m.add_class::<PyClosestAtomPlan>()?;
    m.add_class::<PySearchNeighborsPlan>()?;
    m.add_class::<PySearchNeighborListPlan>()?;
    m.add_class::<PyWatershellPlan>()?;
    m.add_class::<PyClosestPlan>()?;
    m.add_class::<PyClosestCoordsPlan>()?;
    m.add_class::<PyNativeContactsPlan>()?;
    m.add_class::<PyCenterTrajectoryPlan>()?;
    m.add_class::<PyCenterTrajectoryOutputPlan>()?;
    m.add_class::<PyTranslatePlan>()?;
    m.add_class::<PyTransformPlan>()?;
    m.add_class::<PyTransformTrajectoryPlan>()?;
    m.add_class::<PyRotatePlan>()?;
    m.add_class::<PyScalePlan>()?;
    m.add_class::<PyImagePlan>()?;
    m.add_class::<PyImageTrajectoryPlan>()?;
    m.add_class::<PyAutoImagePlan>()?;
    m.add_class::<PyAutoImageTrajectoryPlan>()?;
    m.add_class::<PyReplicateCellPlan>()?;
    m.add_class::<PyVolumePlan>()?;
    m.add_class::<PyXtalSymmPlan>()?;
    m.add_class::<PyStripPlan>()?;
    m.add_class::<PyStripTrajectoryPlan>()?;
    m.add_class::<PyMeanStructurePlan>()?;
    m.add_class::<PyAverageFramePlan>()?;
    m.add_class::<PyMakeStructurePlan>()?;
    m.add_class::<PyCenterOfMassPlan>()?;
    m.add_class::<PyCenterOfGeometryPlan>()?;
    m.add_class::<PyDistanceToPointPlan>()?;
    m.add_class::<PyDistanceToReferencePlan>()?;
    m.add_class::<PyPrincipalAxesPlan>()?;
    m.add_class::<PyAlignPlan>()?;
    m.add_class::<PySuperposeTrajectoryPlan>()?;
    Ok(())
}
