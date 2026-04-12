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
            PlanOutput::Grid(output) => grid_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}


#[pyclass]
struct PyDensityMapPlan {
    plan: RefCell<DensityMapPlan>,
}

#[pymethods]
impl PyDensityMapPlan {
    #[new]
    #[pyo3(signature = (selection, average="z", bin=0.25, n1=None, n2=None, xmin=None, xmax=None, unit="nm-3", length_scale=None))]
    fn new(
        selection: &PySelection,
        average: &str,
        bin: f64,
        n1: Option<usize>,
        n2: Option<usize>,
        xmin: Option<f64>,
        xmax: Option<f64>,
        unit: &str,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let average_axis = parse_axis(average)?;
        let unit = parse_density_map_unit(unit)?;
        let mut plan = DensityMapPlan::new(selection.selection.clone(), average_axis, bin, unit)
            .with_bins(n1, n2)
            .with_average_window(xmin, xmax);
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
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
            PlanOutput::DensityMap(output) => density_map_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyPotentialPlan {
    plan: RefCell<PotentialPlan>,
}

#[pymethods]
impl PyPotentialPlan {
    #[new]
    #[pyo3(signature = (selection, charges, axis="z", bin=0.25, n_slices=None, center_selection=None, symmetrize=false, correct=false, discard_start=0, discard_end=0, length_scale=None))]
    fn new(
        selection: &PySelection,
        charges: Vec<f64>,
        axis: &str,
        bin: f64,
        n_slices: Option<usize>,
        center_selection: Option<&PySelection>,
        symmetrize: bool,
        correct: bool,
        discard_start: usize,
        discard_end: usize,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let axis = parse_axis(axis)?;
        let mut plan = PotentialPlan::new(selection.selection.clone(), axis, bin, charges)
            .with_n_slices(n_slices)
            .with_correct(correct)
            .with_symmetrize(symmetrize)
            .with_integration_discard(discard_start, discard_end);
        if let Some(center_selection) = center_selection {
            plan = plan.with_center_selection(center_selection.selection.clone());
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
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
            PlanOutput::Potential(output) => potential_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyFreeVolumePlan {
    plan: RefCell<FreeVolumePlan>,
}

#[pymethods]
impl PyFreeVolumePlan {
    #[new]
    #[pyo3(signature = (selection, center_selection, box_unit=None, region_size=None, probe_radius=None, shift=None, length_scale=None))]
    fn new(
        selection: &PySelection,
        center_selection: &PySelection,
        box_unit: Option<(f64, f64, f64)>,
        region_size: Option<(f64, f64, f64)>,
        probe_radius: Option<f64>,
        shift: Option<(f64, f64, f64)>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let box_unit_arr = box_unit.map(|v| [v.0, v.1, v.2]);
        let region_size_arr = region_size.map(|v| [v.0, v.1, v.2]);

        let mut plan = FreeVolumePlan::new(
            selection.selection.clone(),
            center_selection.selection.clone(),
            box_unit_arr,
            region_size_arr,
        );
        if let Some(radius) = probe_radius {
            plan = plan.with_probe_radius(radius);
        }
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
struct PyBondiFfvPlan {
    plan: RefCell<BondiFfvPlan>,
}

#[pymethods]
impl PyBondiFfvPlan {
    #[new]
    #[pyo3(signature = (selection, bondi_scale=None, probe_radius=None, seed=None, ninsert_per_nm3=None, length_scale=None))]
    fn new(
        selection: &PySelection,
        bondi_scale: Option<f64>,
        probe_radius: Option<f64>,
        seed: Option<i64>,
        ninsert_per_nm3: Option<usize>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let mut plan = BondiFfvPlan::new(selection.selection.clone());
        if let Some(scale) = bondi_scale {
            plan = plan.with_bondi_scale(scale);
        }
        if let Some(radius) = probe_radius {
            plan = plan.with_probe_radius(radius);
        }
        if let Some(seed) = seed {
            plan = plan.with_seed(seed);
        }
        if let Some(ninsert) = ninsert_per_nm3 {
            plan = plan.with_ninsert_per_nm3(ninsert);
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
            PlanOutput::TimeSeries {
                time,
                data,
                rows,
                cols,
            } => bondi_ffv_to_py(
                py,
                time,
                data,
                rows,
                cols,
                plan.bondi_scale(),
                plan.molar_mass_dalton(),
                plan.probe_radius(),
                plan.ninsert_per_nm3(),
                plan.seed(),
            ),
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
