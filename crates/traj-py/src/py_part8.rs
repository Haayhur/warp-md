#[pyclass]
struct PyCountInVoxelPlan {
    plan: RefCell<CountInVoxelPlan>,
}

#[pymethods]
impl PyCountInVoxelPlan {
    #[new]
    #[pyo3(signature = (selection, center_selection, box_unit, region_size, shift=None, length_scale=None))]
    fn new(
        selection: &PySelection,
        center_selection: &PySelection,
        box_unit: (f64, f64, f64),
        region_size: (f64, f64, f64),
        shift: Option<(f64, f64, f64)>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let mut plan = CountInVoxelPlan::new(
            selection.selection.clone(),
            center_selection.selection.clone(),
            [box_unit.0, box_unit.1, box_unit.2],
            [region_size.0, region_size.1, region_size.2],
        );
        if let Some(shift) = shift {
            plan = plan.with_shift([shift.0, shift.1, shift.2]);
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Grid(output) => grid_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyDensityPlan {
    plan: RefCell<DensityPlan>,
}

#[pymethods]
impl PyDensityPlan {
    #[new]
    #[pyo3(signature = (selection, center_selection, box_unit, region_size, shift=None, length_scale=None))]
    fn new(
        selection: &PySelection,
        center_selection: &PySelection,
        box_unit: (f64, f64, f64),
        region_size: (f64, f64, f64),
        shift: Option<(f64, f64, f64)>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let mut plan = DensityPlan::new(
            selection.selection.clone(),
            center_selection.selection.clone(),
            [box_unit.0, box_unit.1, box_unit.2],
            [region_size.0, region_size.1, region_size.2],
        );
        if let Some(shift) = shift {
            plan = plan.with_shift([shift.0, shift.1, shift.2]);
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Grid(output) => grid_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyVolmapPlan {
    plan: RefCell<VolmapPlan>,
}

#[pymethods]
impl PyVolmapPlan {
    #[new]
    #[pyo3(signature = (selection, center_selection, box_unit, region_size, shift=None, length_scale=None))]
    fn new(
        selection: &PySelection,
        center_selection: &PySelection,
        box_unit: (f64, f64, f64),
        region_size: (f64, f64, f64),
        shift: Option<(f64, f64, f64)>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let mut plan = VolmapPlan::new(
            selection.selection.clone(),
            center_selection.selection.clone(),
            [box_unit.0, box_unit.1, box_unit.2],
            [region_size.0, region_size.1, region_size.2],
        );
        if let Some(shift) = shift {
            plan = plan.with_shift([shift.0, shift.1, shift.2]);
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Grid(output) => grid_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyEquipartitionPlan {
    plan: RefCell<EquipartitionPlan>,
}

#[pymethods]
impl PyEquipartitionPlan {
    #[new]
    #[pyo3(signature = (selection, group_by="resid", velocity_scale=None, length_scale=None, group_types=None))]
    fn new(
        selection: &PySelection,
        group_by: &str,
        velocity_scale: Option<f64>,
        length_scale: Option<f64>,
        group_types: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let group_by = parse_group_by(group_by)?;
        let mut plan = EquipartitionPlan::new(selection.selection.clone(), group_by);
        if let Some(scale) = velocity_scale {
            plan = plan.with_velocity_scale(scale);
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
        if let Some(types) = group_types {
            plan = plan.with_group_types(types);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::TimeSeries { time, data, rows, cols } => {
                timeseries_to_py(py, time, data, rows, cols)
            }
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyHbondPlan {
    plan: RefCell<HbondPlan>,
}

#[pymethods]
impl PyHbondPlan {
    #[new]
    #[pyo3(signature = (donors, acceptors, dist_cutoff, hydrogens=None, angle_cutoff=None))]
    fn new(
        donors: &PySelection,
        acceptors: &PySelection,
        dist_cutoff: f64,
        hydrogens: Option<&PySelection>,
        angle_cutoff: Option<f64>,
    ) -> PyResult<Self> {
        let mut plan = HbondPlan::new(donors.selection.clone(), acceptors.selection.clone(), dist_cutoff);
        if let Some(h_sel) = hydrogens {
            let cutoff = angle_cutoff.ok_or_else(|| {
                PyRuntimeError::new_err("angle_cutoff required when hydrogens are provided")
            })?;
            plan = plan.with_hydrogens(h_sel.selection.clone(), cutoff);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::TimeSeries { time, data, rows, cols } => {
                timeseries_to_py(py, time, data, rows, cols)
            }
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyRdfPlan {
    plan: RefCell<RdfPlan>,
}

#[pymethods]
impl PyRdfPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, bins, r_max, pbc="orthorhombic"))]
    fn new(sel_a: &PySelection, sel_b: &PySelection, bins: usize, r_max: f32, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = RdfPlan::new(sel_a.selection.clone(), sel_b.selection.clone(), bins, r_max, pbc);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Rdf(rdf) => {
                let r = PyArray1::from_vec_bound(py, rdf.r);
                let g = PyArray1::from_vec_bound(py, rdf.g_r);
                let counts = PyArray1::from_vec_bound(py, rdf.counts);
                Ok((r, g, counts).into_py(py))
            }
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyPairDistPlan {
    plan: RefCell<PairDistPlan>,
}

#[pymethods]
impl PyPairDistPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, bins, r_max, pbc="orthorhombic"))]
    fn new(sel_a: &PySelection, sel_b: &PySelection, bins: usize, r_max: f32, pbc: &str) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = PairDistPlan::new(sel_a.selection.clone(), sel_b.selection.clone(), bins, r_max, pbc);
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
        let output = run_plan(&mut *plan, &mut traj_ref, &system.system.borrow(), chunk_frames, device)?;
        match output {
            PlanOutput::Histogram { centers, counts } => Ok(hist_to_py(py, centers, counts)),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyEndToEndPlan {
    plan: RefCell<EndToEndPlan>,
}

#[pymethods]
impl PyEndToEndPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = EndToEndPlan::new(selection.selection.clone());
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
struct PyContourLengthPlan {
    plan: RefCell<ContourLengthPlan>,
}

#[pymethods]
impl PyContourLengthPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = ContourLengthPlan::new(selection.selection.clone());
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
struct PyChainRgPlan {
    plan: RefCell<ChainRgPlan>,
}

#[pymethods]
impl PyChainRgPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = ChainRgPlan::new(selection.selection.clone());
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
struct PyBondLengthDistributionPlan {
    plan: RefCell<BondLengthDistributionPlan>,
}

#[pymethods]
impl PyBondLengthDistributionPlan {
    #[new]
    fn new(selection: &PySelection, bins: usize, r_max: f32) -> Self {
        let plan = BondLengthDistributionPlan::new(selection.selection.clone(), bins, r_max);
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
            PlanOutput::Histogram { centers, counts } => Ok(hist_to_py(py, centers, counts)),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyBondAngleDistributionPlan {
    plan: RefCell<BondAngleDistributionPlan>,
}

#[pymethods]
impl PyBondAngleDistributionPlan {
    #[new]
    #[pyo3(signature = (selection, bins, degrees=true))]
    fn new(selection: &PySelection, bins: usize, degrees: bool) -> Self {
        let plan = BondAngleDistributionPlan::new(selection.selection.clone(), bins, degrees);
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
            PlanOutput::Histogram { centers, counts } => Ok(hist_to_py(py, centers, counts)),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}
