use super::*;

fn lipid_matrix_to_py<'py>(
    py: Python<'py>,
    output: traj_engine::LipidMatrixOutput,
) -> PyResult<PyObject> {
    let arr = Array2::from_shape_vec((output.rows, output.cols), output.values)
        .map_err(|_| PyRuntimeError::new_err("failed to build lipid matrix"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("values", arr.into_pyarray_bound(py))?;
    dict.set_item("residue_ids", output.residue_ids)?;
    dict.set_item("frames", output.frames)?;
    dict.set_item("kind", output.kind)?;
    Ok(dict.into_py(py))
}

fn lipid_flipflop_to_py<'py>(
    py: Python<'py>,
    output: traj_engine::LipidFlipFlopOutput,
) -> PyResult<PyObject> {
    let arr = Array2::from_shape_vec((output.rows, output.cols), output.events)
        .map_err(|_| PyRuntimeError::new_err("failed to build flip-flop events"))?;
    let dict = PyDict::new_bound(py);
    dict.set_item("events", arr.into_pyarray_bound(py))?;
    dict.set_item("success", output.success)?;
    dict.set_item("residue_ids", output.residue_ids)?;
    Ok(dict.into_py(py))
}

fn hydrophobic_defect_to_py<'py>(
    py: Python<'py>,
    output: traj_engine::HydrophobicDefectOutput,
) -> PyResult<PyObject> {
    let dict = PyDict::new_bound(py);
    dict.set_item("dims", vec![output.dims[0], output.dims[1], output.dims[2]])?;
    dict.set_item("voxel_size", output.voxel_size)?;
    dict.set_item("z_bounds", output.z_bounds.to_vec())?;
    dict.set_item("mean", PyArray1::from_vec_bound(py, output.mean))?;
    dict.set_item("first", PyArray1::from_vec_bound(py, output.first))?;
    dict.set_item("last", PyArray1::from_vec_bound(py, output.last))?;
    dict.set_item("min", PyArray1::from_vec_bound(py, output.min))?;
    dict.set_item("max", PyArray1::from_vec_bound(py, output.max))?;
    dict.set_item(
        "frame_counts",
        PyArray1::from_vec_bound(py, output.frame_counts),
    )?;
    dict.set_item(
        "frame_area",
        PyArray1::from_vec_bound(py, output.frame_area),
    )?;
    dict.set_item(
        "frame_volume",
        PyArray1::from_vec_bound(py, output.frame_volume),
    )?;
    dict.set_item(
        "frame_cluster_count",
        PyArray1::from_vec_bound(py, output.frame_cluster_count),
    )?;
    dict.set_item(
        "frame_largest_cluster",
        PyArray1::from_vec_bound(py, output.frame_largest_cluster),
    )?;
    dict.set_item(
        "max_lifetime",
        PyArray1::from_vec_bound(py, output.max_lifetime),
    )?;
    Ok(dict.into_py(py))
}

#[pyclass]
struct PyHydrophobicDefectPlan {
    plan: RefCell<HydrophobicDefectPlan>,
}

#[pymethods]
impl PyHydrophobicDefectPlan {
    #[new]
    #[pyo3(signature = (lipid_selection, reference_selection, voxel_size=1.0, z_bounds=None, probe_radius=None, defect_radius=None, length_scale=None, grid_mode="voxel_centers", leaflet="both", midplane_selection=None, leaflet_bins=1))]
    fn new(
        lipid_selection: &PySelection,
        reference_selection: &PySelection,
        voxel_size: f64,
        z_bounds: Option<(f64, f64)>,
        probe_radius: Option<f64>,
        defect_radius: Option<f64>,
        length_scale: Option<f64>,
        grid_mode: &str,
        leaflet: &str,
        midplane_selection: Option<&PySelection>,
        leaflet_bins: usize,
    ) -> PyResult<Self> {
        let mut plan = HydrophobicDefectPlan::new(
            lipid_selection.selection.clone(),
            reference_selection.selection.clone(),
            voxel_size,
            z_bounds.map(|bounds| [bounds.0, bounds.1]),
        );
        let grid_mode = match grid_mode {
            "voxel_centers" | "voxel-centers" | "centers" => {
                HydrophobicDefectGridMode::VoxelCenters
            }
            "lattice_nodes" | "lattice-nodes" | "nodes" => HydrophobicDefectGridMode::LatticeNodes,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown hydrophobic_defects grid_mode '{other}'"
                )))
            }
        };
        plan = plan.with_grid_mode(grid_mode);
        let leaflet = match leaflet {
            "both" | "all" => HydrophobicDefectLeaflet::Both,
            "upper" => HydrophobicDefectLeaflet::Upper,
            "lower" => HydrophobicDefectLeaflet::Lower,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown hydrophobic_defects leaflet '{other}'"
                )))
            }
        };
        plan = plan.with_leaflet(leaflet);
        if let Some(selection) = midplane_selection {
            plan = plan.with_midplane_selection(selection.selection.clone());
        }
        plan = plan.with_leaflet_bins(leaflet_bins);
        if let Some(radius) = probe_radius {
            plan = plan.with_probe_radius(radius);
        }
        if let Some(radius) = defect_radius {
            plan = plan.with_defect_radius(radius);
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
            PlanOutput::HydrophobicDefect(output) => hydrophobic_defect_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidLeafletPlan {
    plan: RefCell<LipidLeafletPlan>,
}

#[pymethods]
impl PyLipidLeafletPlan {
    #[new]
    #[pyo3(signature = (selection, midplane_selection=None, midplane_cutoff=0.0, bins=1, length_scale=None))]
    fn new(
        selection: &PySelection,
        midplane_selection: Option<&PySelection>,
        midplane_cutoff: f64,
        bins: usize,
        length_scale: Option<f64>,
    ) -> Self {
        let mut plan = LipidLeafletPlan::new(selection.selection.clone()).with_bins(bins);
        if let Some(mid) = midplane_selection {
            plan = plan.with_midplane(mid.selection.clone(), midplane_cutoff);
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidCurvedLeafletPlan {
    plan: RefCell<LipidCurvedLeafletPlan>,
}

#[pymethods]
impl PyLipidCurvedLeafletPlan {
    #[new]
    #[pyo3(signature = (selection, cutoff=15.0, midplane_selection=None, midplane_cutoff=0.0, length_scale=None))]
    fn new(
        selection: &PySelection,
        cutoff: f64,
        midplane_selection: Option<&PySelection>,
        midplane_cutoff: f64,
        length_scale: Option<f64>,
    ) -> Self {
        let mut plan = LipidCurvedLeafletPlan::new(selection.selection.clone(), cutoff);
        if let Some(mid) = midplane_selection {
            plan = plan.with_midplane(mid.selection.clone(), midplane_cutoff);
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidZPositionPlan {
    plan: RefCell<LipidZPositionPlan>,
}

#[pymethods]
impl PyLipidZPositionPlan {
    #[new]
    #[pyo3(signature = (membrane_selection, height_selection, bins=1, length_scale=None))]
    fn new(
        membrane_selection: &PySelection,
        height_selection: &PySelection,
        bins: usize,
        length_scale: Option<f64>,
    ) -> Self {
        let mut plan = LipidZPositionPlan::new(
            membrane_selection.selection.clone(),
            height_selection.selection.clone(),
        )
        .with_bins(bins);
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidZThicknessPlan {
    plan: RefCell<LipidZThicknessPlan>,
}

#[pymethods]
impl PyLipidZThicknessPlan {
    #[new]
    #[pyo3(signature = (selection, length_scale=None))]
    fn new(selection: &PySelection, length_scale: Option<f64>) -> Self {
        let mut plan = LipidZThicknessPlan::new(selection.selection.clone());
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidZAnglePlan {
    plan: RefCell<LipidZAnglePlan>,
}

#[pymethods]
impl PyLipidZAnglePlan {
    #[new]
    #[pyo3(signature = (atom_a, atom_b, degrees=true, length_scale=None))]
    fn new(
        atom_a: &PySelection,
        atom_b: &PySelection,
        degrees: bool,
        length_scale: Option<f64>,
    ) -> Self {
        let mut plan = LipidZAnglePlan::new(atom_a.selection.clone(), atom_b.selection.clone())
            .with_degrees(degrees);
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidAreaPlan {
    plan: RefCell<LipidAreaPlan>,
}

#[pymethods]
impl PyLipidAreaPlan {
    #[new]
    #[pyo3(signature = (selection, leaflets, length_scale=None))]
    fn new(
        selection: &PySelection,
        leaflets: PyReadonlyArray2<i8>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let view = leaflets.as_array();
        let rows = view.shape()[0];
        let cols = view.shape()[1];
        let mut plan = LipidAreaPlan::new(
            selection.selection.clone(),
            view.iter().copied().collect(),
            rows,
            cols,
        );
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidFlipFlopPlan {
    plan: RefCell<LipidFlipFlopPlan>,
}

#[pymethods]
impl PyLipidFlipFlopPlan {
    #[new]
    #[pyo3(signature = (leaflets, residue_ids=None, frame_cutoff=1))]
    fn new(
        leaflets: PyReadonlyArray2<i8>,
        residue_ids: Option<Vec<i32>>,
        frame_cutoff: usize,
    ) -> PyResult<Self> {
        let view = leaflets.as_array();
        let rows = view.shape()[0];
        let cols = view.shape()[1];
        let ids = residue_ids.unwrap_or_else(|| (0..rows).map(|i| i as i32).collect());
        if ids.len() != rows {
            return Err(PyValueError::new_err(
                "residue_ids length must match leaflets rows",
            ));
        }
        Ok(Self {
            plan: RefCell::new(LipidFlipFlopPlan::new(
                view.iter().copied().collect(),
                rows,
                cols,
                ids,
                frame_cutoff,
            )),
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
            PlanOutput::LipidFlipFlop(output) => lipid_flipflop_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidNeighbourPlan {
    plan: RefCell<LipidNeighbourPlan>,
}

#[pymethods]
impl PyLipidNeighbourPlan {
    #[new]
    #[pyo3(signature = (selection, cutoff=10.0, length_scale=None))]
    fn new(selection: &PySelection, cutoff: f64, length_scale: Option<f64>) -> Self {
        let mut plan = LipidNeighbourPlan::new(selection.selection.clone(), cutoff);
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidNeighbourMatrixPlan {
    plan: RefCell<LipidNeighbourMatrixPlan>,
}

#[pymethods]
impl PyLipidNeighbourMatrixPlan {
    #[new]
    #[pyo3(signature = (selection, cutoff=10.0, length_scale=None))]
    fn new(selection: &PySelection, cutoff: f64, length_scale: Option<f64>) -> Self {
        let mut plan = LipidNeighbourMatrixPlan::new(selection.selection.clone(), cutoff);
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidLargestClusterPlan {
    plan: RefCell<LipidLargestClusterPlan>,
}

#[pymethods]
impl PyLipidLargestClusterPlan {
    #[new]
    #[pyo3(signature = (selection, cutoff=10.0, length_scale=None))]
    fn new(selection: &PySelection, cutoff: f64, length_scale: Option<f64>) -> Self {
        let mut plan = LipidLargestClusterPlan::new(selection.selection.clone(), cutoff);
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidMembraneThicknessPlan {
    plan: RefCell<LipidMembraneThicknessPlan>,
}

#[pymethods]
impl PyLipidMembraneThicknessPlan {
    #[new]
    #[pyo3(signature = (selection, leaflets, bins=1, length_scale=None))]
    fn new(
        selection: &PySelection,
        leaflets: PyReadonlyArray2<i8>,
        bins: usize,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let view = leaflets.as_array();
        let rows = view.shape()[0];
        let cols = view.shape()[1];
        let mut plan = LipidMembraneThicknessPlan::new(
            selection.selection.clone(),
            view.iter().copied().collect(),
            rows,
            cols,
        )
        .with_bins(bins);
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidRegistrationPlan {
    plan: RefCell<LipidRegistrationPlan>,
}

#[pymethods]
impl PyLipidRegistrationPlan {
    #[new]
    #[pyo3(signature = (upper_selection, lower_selection, leaflets, bins=1, gaussian_sd=0.0, length_scale=None))]
    fn new(
        upper_selection: &PySelection,
        lower_selection: &PySelection,
        leaflets: PyReadonlyArray2<i8>,
        bins: usize,
        gaussian_sd: f64,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let view = leaflets.as_array();
        let rows = view.shape()[0];
        let cols = view.shape()[1];
        let mut plan = LipidRegistrationPlan::new(
            upper_selection.selection.clone(),
            lower_selection.selection.clone(),
            view.iter().copied().collect(),
            rows,
            cols,
        )
        .with_bins(bins)
        .with_gaussian_sd(gaussian_sd);
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidMsdPlan {
    plan: RefCell<LipidMsdPlan>,
}

#[pymethods]
impl PyLipidMsdPlan {
    #[new]
    #[pyo3(signature = (selection, com_removal_selection=None, length_scale=None))]
    fn new(
        selection: &PySelection,
        com_removal_selection: Option<&PySelection>,
        length_scale: Option<f64>,
    ) -> Self {
        let mut plan = LipidMsdPlan::new(selection.selection.clone());
        if let Some(sel) = com_removal_selection {
            plan = plan.with_com_removal(sel.selection.clone());
        }
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
        }
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyLipidSccPlan {
    plan: RefCell<LipidSccPlan>,
}

#[pymethods]
impl PyLipidSccPlan {
    #[new]
    #[pyo3(signature = (selection, normals=None, length_scale=None))]
    fn new(
        selection: &PySelection,
        normals: Option<PyReadonlyArray3<f32>>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let mut plan = LipidSccPlan::new(selection.selection.clone());
        if let Some(normals) = normals {
            let view = normals.as_array();
            let rows = view.shape()[0];
            let cols = view.shape()[1];
            if view.shape()[2] != 3 {
                return Err(PyValueError::new_err(
                    "normals must have shape (n_residues, n_frames, 3)",
                ));
            }
            plan = plan.with_normals(view.iter().copied().collect(), rows, cols);
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
            PlanOutput::LipidMatrix(output) => lipid_matrix_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

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
        let output = run_plan(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
        )?;
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
        let mut plan = HbondPlan::new(
            donors.selection.clone(),
            acceptors.selection.clone(),
            dist_cutoff,
        );
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
struct PyRdfPlan {
    plan: RefCell<RdfPlan>,
}

fn parse_rdf_dimension(value: &str) -> PyResult<RdfDimension> {
    match value.to_ascii_lowercase().as_str() {
        "3d" | "xyz" => Ok(RdfDimension::ThreeD),
        "xy" => Ok(RdfDimension::PlanarXY),
        _ => Err(PyValueError::new_err("rdf dimension must be '3d' or 'xy'")),
    }
}

#[pymethods]
impl PyRdfPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, bins, r_max, pbc="orthorhombic", center1=false, center2=false, byres1=false, byres2=false, bymol1=false, bymol2=false, no_intramol=false, mass_weighted=true, density=0.033456, volume=false, raw_rdf=false, intrdf=false, dimension="3d"))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        bins: usize,
        r_max: f32,
        pbc: &str,
        center1: bool,
        center2: bool,
        byres1: bool,
        byres2: bool,
        bymol1: bool,
        bymol2: bool,
        no_intramol: bool,
        mass_weighted: bool,
        density: f64,
        volume: bool,
        raw_rdf: bool,
        intrdf: bool,
        dimension: &str,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let dimension = parse_rdf_dimension(dimension)?;
        let plan = RdfPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            bins,
            r_max,
            pbc,
        )
        .with_radial_options(
            center1,
            center2,
            byres1,
            byres2,
            bymol1,
            bymol2,
            no_intramol,
            mass_weighted,
        )
        .with_output_options(density, volume, raw_rdf, intrdf)
        .with_dimension(dimension);
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
                let has_integral = !rdf.integral.is_empty();
                let integral = PyArray1::from_vec_bound(py, rdf.integral);
                if has_integral {
                    Ok((r, g, counts, integral).into_py(py))
                } else {
                    Ok((r, g, counts).into_py(py))
                }
            }
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyPairDistPlan {
    plan: RefCell<PairDistPlan>,
}

#[pyclass]
struct PyPairDistDynamicPlan {
    plan: RefCell<PairDistDynamicPlan>,
}

#[pyclass]
struct PyPairDistanceExtremaPlan {
    plan: RefCell<PairDistanceExtremaPlan>,
}

fn parse_pair_distance_extrema_mode(mode: &str) -> PyResult<PairDistanceExtremaMode> {
    match mode.to_ascii_lowercase().as_str() {
        "min" | "minimum" => Ok(PairDistanceExtremaMode::Min),
        "max" | "maximum" => Ok(PairDistanceExtremaMode::Max),
        _ => Err(PyValueError::new_err(
            "pairdist mode must be 'min' or 'max'",
        )),
    }
}

#[pymethods]
impl PyPairDistPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, bins, r_max, pbc="orthorhombic", output_distribution=false, unique_pairs=false, compact_output=false))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        bins: usize,
        r_max: f32,
        pbc: &str,
        output_distribution: bool,
        unique_pairs: bool,
        compact_output: bool,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let plan = PairDistPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            bins,
            r_max,
            pbc,
        )
        .with_output_options(output_distribution, unique_pairs, compact_output);
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
            PlanOutput::PairDistribution(pairdist) => Ok((
                PyArray1::from_vec_bound(py, pairdist.centers),
                PyArray1::from_vec_bound(py, pairdist.probability),
                PyArray1::from_vec_bound(py, pairdist.std),
                PyArray1::from_vec_bound(py, pairdist.counts),
                pairdist.frames,
            )
                .into_py(py)),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pymethods]
impl PyPairDistanceExtremaPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, mode="min", pbc="none", unique_pairs=false))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        mode: &str,
        pbc: &str,
        unique_pairs: bool,
    ) -> PyResult<Self> {
        let mode = parse_pair_distance_extrema_mode(mode)?;
        let pbc = parse_pbc(pbc)?;
        let plan = PairDistanceExtremaPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            mode,
            pbc,
        )
        .with_unique_pairs(unique_pairs);
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
impl PyPairDistDynamicPlan {
    #[new]
    #[pyo3(signature = (sel_a, sel_b, delta, pbc="orthorhombic", output_distribution=false, unique_pairs=false, compact_output=false))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        delta: f32,
        pbc: &str,
        output_distribution: bool,
        unique_pairs: bool,
        compact_output: bool,
    ) -> PyResult<Self> {
        if !delta.is_finite() || delta <= 0.0 {
            return Err(PyValueError::new_err(
                "pairdist delta must be finite and positive",
            ));
        }
        let pbc = parse_pbc(pbc)?;
        let plan =
            PairDistDynamicPlan::new(sel_a.selection.clone(), sel_b.selection.clone(), delta, pbc)
                .with_output_options(output_distribution, unique_pairs, compact_output);
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
            PlanOutput::PairDistribution(pairdist) => Ok((
                PyArray1::from_vec_bound(py, pairdist.centers),
                PyArray1::from_vec_bound(py, pairdist.probability),
                PyArray1::from_vec_bound(py, pairdist.std),
                PyArray1::from_vec_bound(py, pairdist.counts),
                pairdist.frames,
            )
                .into_py(py)),
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
        let output = run_plan(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
        )?;
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
        let output = run_plan(
            &mut *plan,
            &mut traj_ref,
            &system.system.borrow(),
            chunk_frames,
            device,
        )?;
        match output {
            PlanOutput::Histogram { centers, counts } => Ok(hist_to_py(py, centers, counts)),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyHydrophobicDefectPlan>()?;
    m.add_class::<PyLipidLeafletPlan>()?;
    m.add_class::<PyLipidCurvedLeafletPlan>()?;
    m.add_class::<PyLipidZPositionPlan>()?;
    m.add_class::<PyLipidZThicknessPlan>()?;
    m.add_class::<PyLipidZAnglePlan>()?;
    m.add_class::<PyLipidAreaPlan>()?;
    m.add_class::<PyLipidFlipFlopPlan>()?;
    m.add_class::<PyLipidNeighbourPlan>()?;
    m.add_class::<PyLipidNeighbourMatrixPlan>()?;
    m.add_class::<PyLipidLargestClusterPlan>()?;
    m.add_class::<PyLipidMembraneThicknessPlan>()?;
    m.add_class::<PyLipidRegistrationPlan>()?;
    m.add_class::<PyLipidMsdPlan>()?;
    m.add_class::<PyLipidSccPlan>()?;
    m.add_class::<PyCountInVoxelPlan>()?;
    m.add_class::<PyDensityPlan>()?;
    m.add_class::<PyVolmapPlan>()?;
    m.add_class::<PyDensityMapPlan>()?;
    m.add_class::<PyPotentialPlan>()?;
    m.add_class::<PyFreeVolumePlan>()?;
    m.add_class::<PyBondiFfvPlan>()?;
    m.add_class::<PyEquipartitionPlan>()?;
    m.add_class::<PyHbondPlan>()?;
    m.add_class::<PyRdfPlan>()?;
    m.add_class::<PyPairDistPlan>()?;
    m.add_class::<PyPairDistDynamicPlan>()?;
    m.add_class::<PyPairDistanceExtremaPlan>()?;
    m.add_class::<PyEndToEndPlan>()?;
    m.add_class::<PyContourLengthPlan>()?;
    m.add_class::<PyChainRgPlan>()?;
    m.add_class::<PyBondLengthDistributionPlan>()?;
    m.add_class::<PyBondAngleDistributionPlan>()?;
    Ok(())
}
