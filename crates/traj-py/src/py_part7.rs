#[pyclass]
struct PyToroidalDiffusionPlan {
    plan: RefCell<ToroidalDiffusionPlan>,
}

#[pymethods]
impl PyToroidalDiffusionPlan {
    #[new]
    #[pyo3(signature = (selection, mass_weighted=false, transition_lag=1, emit_transitions=false, store_transition_states=false))]
    fn new(
        selection: &PySelection,
        mass_weighted: bool,
        transition_lag: usize,
        emit_transitions: bool,
        store_transition_states: bool,
    ) -> Self {
        let plan = ToroidalDiffusionPlan::new(selection.selection.clone())
            .with_mass_weighted(mass_weighted)
            .with_transition_lag(transition_lag)
            .with_emit_transitions(emit_transitions)
            .with_store_transition_states(store_transition_states);
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

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run_full<'py>(
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
        let (data, rows, cols) = match output {
            PlanOutput::Matrix { data, rows, cols } => (data, rows, cols),
            _ => return Err(PyRuntimeError::new_err("unexpected output")),
        };
        let counts = plan.transition_counts_flat();
        let probs = plan.transition_matrix_flat();
        let rate = plan.transition_rate();
        drop(plan);

        let fractions = Array2::from_shape_vec((rows, cols), data)
            .map_err(|_| PyRuntimeError::new_err("invalid toroidal diffusion shape"))?;
        let counts_vec: Vec<f32> = counts.into_iter().map(|v| v as f32).collect();
        let counts_arr = Array2::from_shape_vec((4, 4), counts_vec)
            .map_err(|_| PyRuntimeError::new_err("invalid transition counts shape"))?;
        let probs_arr = Array2::from_shape_vec((4, 4), probs.to_vec())
            .map_err(|_| PyRuntimeError::new_err("invalid transition matrix shape"))?;

        let fractions_py = fractions.into_pyarray_bound(py).into_py(py);
        let counts_py = counts_arr.into_pyarray_bound(py).into_py(py);
        let probs_py = probs_arr.into_pyarray_bound(py).into_py(py);
        Ok((fractions_py, counts_py, probs_py, rate).into_py(py))
    }

    #[pyo3(signature = (traj, system, chunk_frames=None, device="auto", frame_indices=None))]
    fn run_full_with_states<'py>(
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
        let (data, rows, cols) = match output {
            PlanOutput::Matrix { data, rows, cols } => (data, rows, cols),
            _ => return Err(PyRuntimeError::new_err("unexpected output")),
        };
        let counts = plan.transition_counts_flat();
        let probs = plan.transition_matrix_flat();
        let rate = plan.transition_rate();
        let state_rows = plan.transition_state_rows();
        let state_cols = plan.transition_state_cols();
        let state_data = plan.transition_states_flat().to_vec();
        drop(plan);

        let fractions = Array2::from_shape_vec((rows, cols), data)
            .map_err(|_| PyRuntimeError::new_err("invalid toroidal diffusion shape"))?;
        let counts_vec: Vec<f32> = counts.into_iter().map(|v| v as f32).collect();
        let counts_arr = Array2::from_shape_vec((4, 4), counts_vec)
            .map_err(|_| PyRuntimeError::new_err("invalid transition counts shape"))?;
        let probs_arr = Array2::from_shape_vec((4, 4), probs.to_vec())
            .map_err(|_| PyRuntimeError::new_err("invalid transition matrix shape"))?;
        let states_arr = Array2::from_shape_vec((state_rows, state_cols), state_data)
            .map_err(|_| PyRuntimeError::new_err("invalid transition states shape"))?;

        let fractions_py = fractions.into_pyarray_bound(py).into_py(py);
        let counts_py = counts_arr.into_pyarray_bound(py).into_py(py);
        let probs_py = probs_arr.into_pyarray_bound(py).into_py(py);
        let states_py = states_arr.into_pyarray_bound(py).into_py(py);
        Ok((fractions_py, counts_py, probs_py, rate, states_py).into_py(py))
    }
}

#[pyclass]
struct PyMultiPuckerPlan {
    plan: RefCell<MultiPuckerPlan>,
}

#[pymethods]
impl PyMultiPuckerPlan {
    #[new]
    #[pyo3(signature = (selection, bins=10, mode="legacy", range_max=None, normalize=true))]
    fn new(
        selection: &PySelection,
        bins: usize,
        mode: &str,
        range_max: Option<f32>,
        normalize: bool,
    ) -> PyResult<Self> {
        let mode = match mode {
            "legacy" => MultiPuckerMode::Legacy,
            "histogram" => MultiPuckerMode::Histogram,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "mode must be 'legacy' or 'histogram'",
                ))
            }
        };
        let plan = MultiPuckerPlan::new(selection.selection.clone(), bins)
            .with_mode(mode)
            .with_range_max(range_max)
            .with_normalize(normalize);
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

#[derive(Clone, Copy)]
enum NmrCorrMode {
    Tensor,
    Timecorr,
}

#[pyclass]
struct PyNmrIredPlan {
    plan: RefCell<NmrIredPlan>,
    order: usize,
    return_corr: bool,
    corr_mode: NmrCorrMode,
    n_pairs: usize,
}

#[pymethods]
impl PyNmrIredPlan {
    #[new]
    #[pyo3(signature = (pairs, order=2, length_scale=0.1, pbc="none", corr_mode="tensor", return_corr=true))]
    fn new(
        pairs: Vec<(usize, usize)>,
        order: usize,
        length_scale: f64,
        pbc: &str,
        corr_mode: &str,
        return_corr: bool,
    ) -> PyResult<Self> {
        if order != 1 && order != 2 {
            return Err(PyRuntimeError::new_err("order must be 1 or 2"));
        }
        let corr_mode = match corr_mode {
            "tensor" => NmrCorrMode::Tensor,
            "timecorr" => NmrCorrMode::Timecorr,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "corr_mode must be 'tensor' or 'timecorr'",
                ))
            }
        };
        let pbc = parse_pbc(pbc)?;
        let mut pair_idx = Vec::with_capacity(pairs.len());
        for (a, b) in pairs.into_iter() {
            pair_idx.push([a as u32, b as u32]);
        }
        let n_pairs = pair_idx.len();
        let plan = NmrIredPlan::new(pair_idx, length_scale, pbc);
        Ok(Self {
            plan: RefCell::new(plan),
            order,
            return_corr,
            corr_mode,
            n_pairs,
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
        let (data, rows, cols) = match output {
            PlanOutput::Matrix { data, rows, cols } => (data, rows, cols),
            _ => return Err(PyRuntimeError::new_err("unexpected output")),
        };
        if cols != self.n_pairs * 3 {
            return Err(PyRuntimeError::new_err("nmr vector shape mismatch"));
        }
        let vectors = Array3::from_shape_vec((rows, self.n_pairs, 3), data.clone())
            .map_err(|_| PyRuntimeError::new_err("invalid nmr vector shape"))?;
        let vectors_py = vectors.into_pyarray_bound(py).into_py(py);
        if !self.return_corr {
            return Ok(vectors_py);
        }
        let (corr_data, corr_rows, corr_cols) =
            nmr_corr_matrix(&data, rows, self.n_pairs, self.order, self.corr_mode);
        let corr = Array2::from_shape_vec((corr_rows, corr_cols), corr_data)
            .map_err(|_| PyRuntimeError::new_err("invalid nmr correlation shape"))?;
        let corr_py = corr.into_pyarray_bound(py).into_py(py);
        Ok((vectors_py, corr_py).into_py(py))
    }
}

fn nmr_corr_matrix(
    vectors: &[f32],
    n_frames: usize,
    n_pairs: usize,
    order: usize,
    corr_mode: NmrCorrMode,
) -> (Vec<f32>, usize, usize) {
    if n_frames == 0 || n_pairs == 0 {
        return match corr_mode {
            NmrCorrMode::Tensor => (Vec::new(), 0, 0),
            NmrCorrMode::Timecorr => (Vec::new(), 0, n_pairs),
        };
    }
    let n_features = n_pairs * 3;
    match corr_mode {
        NmrCorrMode::Tensor => {
            let mut mat = vec![0.0f64; n_features * n_features];
            for frame in 0..n_frames {
                let row = &vectors[frame * n_features..(frame + 1) * n_features];
                for i in 0..n_features {
                    let vi = row[i] as f64;
                    let base = i * n_features;
                    for j in 0..n_features {
                        mat[base + j] += vi * row[j] as f64;
                    }
                }
            }
            let inv = 1.0f64 / n_frames as f64;
            for value in mat.iter_mut() {
                *value *= inv;
            }
            if order == 2 && !mat.is_empty() {
                let denom = mat[0];
                if denom != 0.0 {
                    for value in mat.iter_mut() {
                        *value /= denom;
                    }
                }
            }
            (
                mat.into_iter().map(|v| v as f32).collect(),
                n_features,
                n_features,
            )
        }
        NmrCorrMode::Timecorr => {
            let mut out = vec![0.0f32; n_frames * n_pairs];
            for lag in 0..n_frames {
                let span = n_frames - lag;
                for pair in 0..n_pairs {
                    let mut acc = 0.0f64;
                    for frame in 0..span {
                        let base0 = frame * n_features + pair * 3;
                        let base1 = (frame + lag) * n_features + pair * 3;
                        let mut dot = 0.0f64;
                        dot += vectors[base0] as f64 * vectors[base1] as f64;
                        dot += vectors[base0 + 1] as f64 * vectors[base1 + 1] as f64;
                        dot += vectors[base0 + 2] as f64 * vectors[base1 + 2] as f64;
                        if order == 2 {
                            dot = 1.5 * dot * dot - 0.5;
                        }
                        acc += dot;
                    }
                    out[lag * n_pairs + pair] = (acc / span as f64) as f32;
                }
            }
            (out, n_frames, n_pairs)
        }
    }
}

#[pyclass]
struct PyRotAcfPlan {
    plan: RefCell<RotAcfPlan>,
}

#[pymethods]
impl PyRotAcfPlan {
    #[new]
    #[pyo3(signature = (selection, group_by="resid", orientation=None, p2_legendre=true, length_scale=None, frame_decimation=None, dt_decimation=None, time_binning=None, group_types=None, lag_mode=None, max_lag=None, memory_budget_bytes=None, multi_tau_m=None, multi_tau_levels=None))]
    fn new(
        selection: &PySelection,
        group_by: &str,
        orientation: Option<Vec<usize>>,
        p2_legendre: bool,
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
        let orient = match orientation {
            Some(ref v) if v.len() == 2 && v.iter().all(|&idx| idx > 0) => {
                OrientationSpec::VectorIndices([v[0], v[1]])
            }
            Some(ref v) if v.len() == 3 && v.iter().all(|&idx| idx > 0) => {
                OrientationSpec::PlaneIndices([v[0], v[1], v[2]])
            }
            Some(ref v) if v.iter().any(|&idx| idx == 0) => {
                return Err(PyRuntimeError::new_err(
                    "orientation indices are 1-based and must be >= 1",
                ))
            }
            Some(_) => {
                return Err(PyRuntimeError::new_err(
                    "orientation must be length-2 (vector) or length-3 (plane) indices",
                ))
            }
            None => {
                return Err(PyRuntimeError::new_err(
                    "orientation indices required for rotacf plan",
                ))
            }
        };
        let mut plan = RotAcfPlan::new(selection.selection.clone(), group_by, orient)
            .with_p2_legendre(p2_legendre);
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
struct PyConductivityPlan {
    plan: RefCell<ConductivityPlan>,
}

#[pymethods]
impl PyConductivityPlan {
    #[new]
    #[pyo3(signature = (selection, charges, temperature, group_by="resid", transference=false, length_scale=None, frame_decimation=None, dt_decimation=None, time_binning=None, group_types=None, lag_mode=None, max_lag=None, memory_budget_bytes=None, multi_tau_m=None, multi_tau_levels=None))]
    fn new(
        selection: &PySelection,
        charges: Vec<f64>,
        temperature: f64,
        group_by: &str,
        transference: bool,
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
        let mut plan =
            ConductivityPlan::new(selection.selection.clone(), group_by, charges, temperature)
                .with_transference(transference);
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
struct PyDielectricPlan {
    plan: RefCell<DielectricPlan>,
}

#[pymethods]
impl PyDielectricPlan {
    #[new]
    #[pyo3(signature = (selection, charges, group_by="resid", length_scale=None, group_types=None))]
    fn new(
        selection: &PySelection,
        charges: Vec<f64>,
        group_by: &str,
        length_scale: Option<f64>,
        group_types: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let group_by = parse_group_by(group_by)?;
        let mut plan = DielectricPlan::new(selection.selection.clone(), group_by, charges);
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
            PlanOutput::Dielectric(output) => dielectric_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyDipoleAlignmentPlan {
    plan: RefCell<DipoleAlignmentPlan>,
}

#[pymethods]
impl PyDipoleAlignmentPlan {
    #[new]
    #[pyo3(signature = (selection, charges, group_by="resid", length_scale=None, group_types=None))]
    fn new(
        selection: &PySelection,
        charges: Vec<f64>,
        group_by: &str,
        length_scale: Option<f64>,
        group_types: Option<Vec<usize>>,
    ) -> PyResult<Self> {
        let group_by = parse_group_by(group_by)?;
        let mut plan = DipoleAlignmentPlan::new(selection.selection.clone(), group_by, charges);
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
struct PyIonPairCorrelationPlan {
    plan: RefCell<IonPairCorrelationPlan>,
}

#[pymethods]
impl PyIonPairCorrelationPlan {
    #[new]
    #[pyo3(signature = (selection, rclust_cat, rclust_ani, group_by="resid", cation_type=0, anion_type=1, max_cluster=10, length_scale=None, group_types=None, lag_mode=None, max_lag=None, memory_budget_bytes=None, multi_tau_m=None, multi_tau_levels=None))]
    fn new(
        selection: &PySelection,
        rclust_cat: f64,
        rclust_ani: f64,
        group_by: &str,
        cation_type: usize,
        anion_type: usize,
        max_cluster: usize,
        length_scale: Option<f64>,
        group_types: Option<Vec<usize>>,
        lag_mode: Option<&str>,
        max_lag: Option<usize>,
        memory_budget_bytes: Option<usize>,
        multi_tau_m: Option<usize>,
        multi_tau_levels: Option<usize>,
    ) -> PyResult<Self> {
        let group_by = parse_group_by(group_by)?;
        let mut plan = IonPairCorrelationPlan::new(
            selection.selection.clone(),
            group_by,
            rclust_cat,
            rclust_ani,
        )
        .with_types(cation_type, anion_type)
        .with_max_cluster(max_cluster);
        if let Some(scale) = length_scale {
            plan = plan.with_length_scale(scale);
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
struct PyStructureFactorPlan {
    plan: RefCell<StructureFactorPlan>,
}

#[pymethods]
impl PyStructureFactorPlan {
    #[new]
    #[pyo3(signature = (selection, bins, r_max, q_bins, q_max, pbc="orthorhombic", length_scale=None))]
    fn new(
        selection: &PySelection,
        bins: usize,
        r_max: f64,
        q_bins: usize,
        q_max: f64,
        pbc: &str,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let pbc = parse_pbc(pbc)?;
        let mut plan =
            StructureFactorPlan::new(selection.selection.clone(), bins, r_max, q_bins, q_max, pbc);
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
            PlanOutput::StructureFactor(output) => structure_factor_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyDockingPlan {
    plan: RefCell<DockingPlan>,
}

#[pymethods]
impl PyDockingPlan {
    #[new]
    #[pyo3(signature = (receptor, ligand, close_contact_cutoff=4.0, hydrophobic_cutoff=4.0, hydrogen_bond_cutoff=3.5, clash_cutoff=2.5, salt_bridge_cutoff=5.5, halogen_bond_cutoff=5.5, metal_coordination_cutoff=3.5, cation_pi_cutoff=6.0, pi_pi_cutoff=7.5, hbond_min_angle_deg=120.0, donor_hydrogen_cutoff=1.25, allow_missing_hydrogen=true, length_scale=1.0, max_events_per_frame=20000))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        receptor: &PySelection,
        ligand: &PySelection,
        close_contact_cutoff: f64,
        hydrophobic_cutoff: f64,
        hydrogen_bond_cutoff: f64,
        clash_cutoff: f64,
        salt_bridge_cutoff: f64,
        halogen_bond_cutoff: f64,
        metal_coordination_cutoff: f64,
        cation_pi_cutoff: f64,
        pi_pi_cutoff: f64,
        hbond_min_angle_deg: f64,
        donor_hydrogen_cutoff: f64,
        allow_missing_hydrogen: bool,
        length_scale: f64,
        max_events_per_frame: usize,
    ) -> PyResult<Self> {
        let plan = DockingPlan::new(
            receptor.selection.clone(),
            ligand.selection.clone(),
            close_contact_cutoff,
            hydrophobic_cutoff,
            hydrogen_bond_cutoff,
            clash_cutoff,
        )
        .map_err(to_py_err)?
        .with_salt_bridge_cutoff(salt_bridge_cutoff)
        .with_halogen_bond_cutoff(halogen_bond_cutoff)
        .with_metal_coordination_cutoff(metal_coordination_cutoff)
        .with_cation_pi_cutoff(cation_pi_cutoff)
        .with_pi_pi_cutoff(pi_pi_cutoff)
        .with_hbond_min_angle_deg(hbond_min_angle_deg)
        .with_donor_hydrogen_cutoff(donor_hydrogen_cutoff)
        .with_allow_missing_hydrogen(allow_missing_hydrogen)
        .with_length_scale(length_scale)
        .with_max_events_per_frame(max_events_per_frame);
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
struct PyWaterCountPlan {
    plan: RefCell<WaterCountPlan>,
}

#[pyclass]
struct PyDsspPlan {
    plan: RefCell<DsspPlan>,
}

#[pyclass]
struct PyGistGridPlan {
    plan: RefCell<GistGridPlan>,
}

#[pyclass]
struct PyGistDirectPlan {
    plan: RefCell<GistDirectPlan>,
    record_frame_energies: bool,
    record_pme_frame_totals: bool,
}

#[pymethods]
impl PyGistGridPlan {
    #[new]
    #[pyo3(signature = (oxygen_indices, hydrogen1_indices, hydrogen2_indices, orientation_valid, solute_selection, spacing, orientation_bins=12, length_scale=1.0, padding=0.5, origin=None, dims=None, frame_indices=None, max_frames=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        oxygen_indices: Vec<usize>,
        hydrogen1_indices: Vec<usize>,
        hydrogen2_indices: Vec<usize>,
        orientation_valid: Vec<bool>,
        solute_selection: &PySelection,
        spacing: f64,
        orientation_bins: usize,
        length_scale: f64,
        padding: f64,
        origin: Option<(f64, f64, f64)>,
        dims: Option<(usize, usize, usize)>,
        frame_indices: Option<Vec<i64>>,
        max_frames: Option<usize>,
    ) -> PyResult<Self> {
        let orientation_valid_u8: Vec<u8> = orientation_valid
            .into_iter()
            .map(|v| if v { 1u8 } else { 0u8 })
            .collect();
        let oxygen_indices_u32: Vec<u32> = oxygen_indices.into_iter().map(|v| v as u32).collect();
        let hydrogen1_indices_u32: Vec<u32> =
            hydrogen1_indices.into_iter().map(|v| v as u32).collect();
        let hydrogen2_indices_u32: Vec<u32> =
            hydrogen2_indices.into_iter().map(|v| v as u32).collect();
        let mut plan = match (origin, dims) {
            (Some(origin), Some(dims)) => GistGridPlan::new(
                oxygen_indices_u32,
                hydrogen1_indices_u32,
                hydrogen2_indices_u32,
                orientation_valid_u8,
                solute_selection.selection.clone(),
                [origin.0, origin.1, origin.2],
                [dims.0, dims.1, dims.2],
                spacing,
                orientation_bins,
            )
            .map_err(to_py_err)?,
            (None, None) => GistGridPlan::new_auto(
                oxygen_indices_u32,
                hydrogen1_indices_u32,
                hydrogen2_indices_u32,
                orientation_valid_u8,
                solute_selection.selection.clone(),
                spacing,
                padding,
                orientation_bins,
            )
            .map_err(to_py_err)?,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "gist grid requires both origin+dims or neither",
                ));
            }
        };
        plan = plan.with_length_scale(length_scale);
        if let Some(max_frames) = max_frames {
            plan = plan.with_max_frames(Some(max_frames));
        }
        if let Some(frame_indices) = frame_indices {
            let mut selected = Vec::with_capacity(frame_indices.len());
            for idx in frame_indices.into_iter() {
                if idx < 0 {
                    return Err(PyRuntimeError::new_err(
                        "gist frame_indices must be non-negative",
                    ));
                }
                selected.push(idx as usize);
            }
            plan = plan.with_frame_indices(Some(selected));
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
        let (data, rows, cols) = match output {
            PlanOutput::Matrix { data, rows, cols } => (data, rows, cols),
            _ => return Err(PyRuntimeError::new_err("unexpected output")),
        };
        let dims = plan.dims();
        let orientation_bins = plan.orientation_bins();
        let expected_rows = dims[0] * dims[1] * dims[2];
        let expected_cols = orientation_bins + 1;
        if rows != expected_rows || cols != expected_cols {
            return Err(PyRuntimeError::new_err("gist grid output shape mismatch"));
        }
        let mut counts = vec![0.0f32; expected_rows];
        let mut orient = vec![0.0f32; expected_rows * orientation_bins];
        for cell in 0..expected_rows {
            let src = cell * cols;
            counts[cell] = data[src];
            let dst = cell * orientation_bins;
            orient[dst..dst + orientation_bins]
                .copy_from_slice(&data[src + 1..src + 1 + orientation_bins]);
        }
        let counts_arr = Array3::from_shape_vec((dims[0], dims[1], dims[2]), counts)
            .map_err(|_| PyRuntimeError::new_err("invalid gist count shape"))?;
        let orient_arr =
            Array4::from_shape_vec((dims[0], dims[1], dims[2], orientation_bins), orient)
                .map_err(|_| PyRuntimeError::new_err("invalid gist orientation shape"))?;
        let n_frames = plan.n_frames();
        let origin = plan.origin();
        drop(plan);
        Ok((
            counts_arr.into_pyarray_bound(py).into_py(py),
            orient_arr.into_pyarray_bound(py).into_py(py),
            (origin[0], origin[1], origin[2]),
            n_frames,
        )
            .into_py(py))
    }
}

#[pymethods]
impl PyGistDirectPlan {
    #[new]
    #[pyo3(signature = (oxygen_indices, hydrogen1_indices, hydrogen2_indices, orientation_valid, water_offsets, water_atoms, solute_indices, charges, sigmas, epsilons, exceptions, spacing, cutoff, periodic, orientation_bins=12, length_scale=1.0, padding=0.5, frame_indices=None, max_frames=None, record_frame_energies=false, record_pme_frame_totals=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        oxygen_indices: Vec<usize>,
        hydrogen1_indices: Vec<usize>,
        hydrogen2_indices: Vec<usize>,
        orientation_valid: Vec<bool>,
        water_offsets: Vec<usize>,
        water_atoms: Vec<usize>,
        solute_indices: Vec<usize>,
        charges: Vec<f64>,
        sigmas: Vec<f64>,
        epsilons: Vec<f64>,
        exceptions: Vec<(usize, usize, f64, f64, f64)>,
        spacing: f64,
        cutoff: f64,
        periodic: bool,
        orientation_bins: usize,
        length_scale: f64,
        padding: f64,
        frame_indices: Option<Vec<i64>>,
        max_frames: Option<usize>,
        record_frame_energies: bool,
        record_pme_frame_totals: bool,
    ) -> PyResult<Self> {
        let orientation_valid_u8: Vec<u8> = orientation_valid
            .into_iter()
            .map(|v| if v { 1u8 } else { 0u8 })
            .collect();
        let oxygen_indices_u32: Vec<u32> = oxygen_indices.into_iter().map(|v| v as u32).collect();
        let hydrogen1_indices_u32: Vec<u32> =
            hydrogen1_indices.into_iter().map(|v| v as u32).collect();
        let hydrogen2_indices_u32: Vec<u32> =
            hydrogen2_indices.into_iter().map(|v| v as u32).collect();
        let water_offsets_u32: Vec<u32> = water_offsets.into_iter().map(|v| v as u32).collect();
        let water_atoms_u32: Vec<u32> = water_atoms.into_iter().map(|v| v as u32).collect();
        let solute_indices_u32: Vec<u32> = solute_indices.into_iter().map(|v| v as u32).collect();
        let exceptions_u32: Vec<(u32, u32, f64, f64, f64)> = exceptions
            .into_iter()
            .map(|(a, b, q, s, e)| (a as u32, b as u32, q, s, e))
            .collect();
        let mut plan = GistDirectPlan::new_auto(
            oxygen_indices_u32,
            hydrogen1_indices_u32,
            hydrogen2_indices_u32,
            orientation_valid_u8,
            water_offsets_u32,
            water_atoms_u32,
            solute_indices_u32,
            charges,
            sigmas,
            epsilons,
            exceptions_u32,
            spacing,
            padding,
            orientation_bins,
            cutoff,
            periodic,
        )
        .map_err(to_py_err)?;
        plan = plan.with_length_scale(length_scale);
        plan = plan.with_record_frame_energies(record_frame_energies);
        plan = plan.with_record_pme_frame_totals(record_pme_frame_totals);
        if let Some(max_frames) = max_frames {
            plan = plan.with_max_frames(Some(max_frames));
        }
        if let Some(frame_indices) = frame_indices {
            let mut selected = Vec::with_capacity(frame_indices.len());
            for idx in frame_indices.into_iter() {
                if idx < 0 {
                    return Err(PyRuntimeError::new_err(
                        "gist frame_indices must be non-negative",
                    ));
                }
                selected.push(idx as usize);
            }
            plan = plan.with_frame_indices(Some(selected));
        }
        Ok(Self {
            plan: RefCell::new(plan),
            record_frame_energies,
            record_pme_frame_totals,
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
        let (data, rows, cols) = match output {
            PlanOutput::Matrix { data, rows, cols } => (data, rows, cols),
            _ => return Err(PyRuntimeError::new_err("unexpected output")),
        };
        let dims = plan.dims();
        let orientation_bins = plan.orientation_bins();
        let expected_rows = dims[0] * dims[1] * dims[2];
        let expected_cols = orientation_bins + 1;
        if rows != expected_rows || cols != expected_cols {
            return Err(PyRuntimeError::new_err("gist direct output shape mismatch"));
        }
        let mut counts = vec![0.0f32; expected_rows];
        let mut orient = vec![0.0f32; expected_rows * orientation_bins];
        for cell in 0..expected_rows {
            let src = cell * cols;
            counts[cell] = data[src];
            let dst = cell * orientation_bins;
            orient[dst..dst + orientation_bins]
                .copy_from_slice(&data[src + 1..src + 1 + orientation_bins]);
        }
        let counts_arr = Array3::from_shape_vec((dims[0], dims[1], dims[2]), counts)
            .map_err(|_| PyRuntimeError::new_err("invalid gist direct count shape"))?;
        let orient_arr =
            Array4::from_shape_vec((dims[0], dims[1], dims[2], orientation_bins), orient)
                .map_err(|_| PyRuntimeError::new_err("invalid gist direct orientation shape"))?;
        let energy_sw_arr =
            Array3::from_shape_vec((dims[0], dims[1], dims[2]), plan.energy_sw().to_vec())
                .map_err(|_| PyRuntimeError::new_err("invalid gist direct sw shape"))?;
        let energy_ww_arr =
            Array3::from_shape_vec((dims[0], dims[1], dims[2]), plan.energy_ww().to_vec())
                .map_err(|_| PyRuntimeError::new_err("invalid gist direct ww shape"))?;
        let n_frames = plan.n_frames();
        let origin = plan.origin();
        let direct_sw_total = plan.direct_sw_total();
        let direct_ww_total = plan.direct_ww_total();
        let frame_direct_sw = if self.record_frame_energies {
            plan.frame_direct_sw().to_vec()
        } else {
            Vec::new()
        };
        let frame_direct_ww = if self.record_frame_energies {
            plan.frame_direct_ww().to_vec()
        } else {
            Vec::new()
        };
        let frame_offsets_u64: Vec<u64> = if self.record_frame_energies {
            plan.frame_offsets().iter().map(|&v| v as u64).collect()
        } else {
            Vec::new()
        };
        let frame_cells = if self.record_frame_energies {
            plan.frame_cells().to_vec()
        } else {
            Vec::new()
        };
        let frame_sw = if self.record_frame_energies {
            plan.frame_sw().to_vec()
        } else {
            Vec::new()
        };
        let frame_ww = if self.record_frame_energies {
            plan.frame_ww().to_vec()
        } else {
            Vec::new()
        };
        let frame_pme_sw = if self.record_pme_frame_totals {
            plan.frame_pme_sw().to_vec()
        } else {
            Vec::new()
        };
        let frame_pme_ww = if self.record_pme_frame_totals {
            plan.frame_pme_ww().to_vec()
        } else {
            Vec::new()
        };
        drop(plan);
        let items: Vec<PyObject> = vec![
            counts_arr.into_pyarray_bound(py).into_py(py),
            orient_arr.into_pyarray_bound(py).into_py(py),
            energy_sw_arr.into_pyarray_bound(py).into_py(py),
            energy_ww_arr.into_pyarray_bound(py).into_py(py),
            (origin[0], origin[1], origin[2]).into_py(py),
            n_frames.into_py(py),
            direct_sw_total.into_py(py),
            direct_ww_total.into_py(py),
            PyArray1::from_vec_bound(py, frame_direct_sw).into_py(py),
            PyArray1::from_vec_bound(py, frame_direct_ww).into_py(py),
            PyArray1::from_vec_bound(py, frame_offsets_u64).into_py(py),
            PyArray1::from_vec_bound(py, frame_cells).into_py(py),
            PyArray1::from_vec_bound(py, frame_sw).into_py(py),
            PyArray1::from_vec_bound(py, frame_ww).into_py(py),
            PyArray1::from_vec_bound(py, frame_pme_sw).into_py(py),
            PyArray1::from_vec_bound(py, frame_pme_ww).into_py(py),
        ];
        let tuple = pyo3::types::PyTuple::new_bound(py, items);
        Ok(tuple.into_py(py))
    }
}

#[pymethods]
impl PyDsspPlan {
    #[new]
    fn new(selection: &PySelection) -> Self {
        let plan = DsspPlan::new(selection.selection.clone());
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
        let (data, rows, cols) = match output {
            PlanOutput::Matrix { data, rows, cols } => (data, rows, cols),
            _ => return Err(PyRuntimeError::new_err("unexpected output")),
        };
        if rows * cols != data.len() {
            return Err(PyRuntimeError::new_err("dssp output shape mismatch"));
        }
        let labels = plan.labels().to_vec();
        drop(plan);
        let mut codes = Vec::with_capacity(data.len());
        for value in data.into_iter() {
            let code = (value.round() as i32).clamp(0, 2) as u8;
            codes.push(code);
        }
        let arr = Array2::from_shape_vec((rows, cols), codes)
            .map_err(|_| PyRuntimeError::new_err("invalid dssp matrix shape"))?;
        Ok((labels, arr.into_pyarray_bound(py).into_py(py)).into_py(py))
    }
}

#[pymethods]
impl PyWaterCountPlan {
    #[new]
    #[pyo3(signature = (water_selection, center_selection, box_unit, region_size, shift=None, length_scale=None))]
    fn new(
        water_selection: &PySelection,
        center_selection: &PySelection,
        box_unit: (f64, f64, f64),
        region_size: (f64, f64, f64),
        shift: Option<(f64, f64, f64)>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let mut plan = WaterCountPlan::new(
            water_selection.selection.clone(),
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
