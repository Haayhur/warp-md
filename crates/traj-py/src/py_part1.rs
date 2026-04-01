#[pyclass]
struct PySystem {
    system: RefCell<System>,
}

#[pymethods]
impl PySystem {
    #[staticmethod]
    fn from_pdb(path: &str) -> PyResult<Self> {
        let mut reader = PdbReader::new(path);
        let system = reader.read_system().map_err(to_py_err)?;
        Ok(Self {
            system: RefCell::new(system),
        })
    }

    #[staticmethod]
    fn from_pdb_permissive(path: &str) -> PyResult<Self> {
        let reader = PdbReader::new(path);
        let system = reader.read_system_permissive().map_err(to_py_err)?;
        Ok(Self {
            system: RefCell::new(system),
        })
    }

    #[staticmethod]
    fn from_pdbqt(path: &str) -> PyResult<Self> {
        let mut reader = PdbqtReader::new(path);
        let system = reader.read_system().map_err(to_py_err)?;
        Ok(Self {
            system: RefCell::new(system),
        })
    }

    #[staticmethod]
    fn from_gro(path: &str) -> PyResult<Self> {
        let mut reader = GroReader::new(path);
        let system = reader.read_system().map_err(to_py_err)?;
        Ok(Self {
            system: RefCell::new(system),
        })
    }

    fn select(&self, expr: &str) -> PyResult<PySelection> {
        let mut sys = self.system.borrow_mut();
        let selection = sys.select(expr).map_err(to_py_err)?;
        Ok(PySelection { selection })
    }

    fn select_indices(&self, indices: Vec<u32>) -> PyResult<PySelection> {
        let selection = Selection {
            expr: "__indices__".to_string(),
            indices: Arc::new(indices),
        };
        Ok(PySelection { selection })
    }

    fn n_atoms(&self) -> PyResult<usize> {
        Ok(self.system.borrow().n_atoms())
    }

    fn atom_table<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let sys = self.system.borrow();
        let atoms = &sys.atoms;
        let mut names = Vec::with_capacity(atoms.name_id.len());
        let mut resnames = Vec::with_capacity(atoms.resname_id.len());
        let mut chains = Vec::with_capacity(atoms.chain_id.len());
        let mut elements = Vec::with_capacity(atoms.element_id.len());
        for &id in atoms.name_id.iter() {
            names.push(sys.interner.resolve(id).unwrap_or("").to_string());
        }
        for &id in atoms.resname_id.iter() {
            resnames.push(sys.interner.resolve(id).unwrap_or("").to_string());
        }
        for &id in atoms.chain_id.iter() {
            chains.push(sys.interner.resolve(id).unwrap_or("").to_string());
        }
        for &id in atoms.element_id.iter() {
            elements.push(sys.interner.resolve(id).unwrap_or("").to_string());
        }
        let dict = PyDict::new_bound(py);
        dict.set_item("name", names)?;
        dict.set_item("resname", resnames)?;
        dict.set_item("resid", atoms.resid.clone())?;
        dict.set_item("chain", chains)?;
        dict.set_item("chain_id", atoms.chain_id.clone())?;
        dict.set_item("element", elements)?;
        dict.set_item("mass", atoms.mass.clone())?;
        Ok(dict.into_py(py))
    }

    fn positions0<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let sys = self.system.borrow();
        let Some(pos) = &sys.positions0 else {
            return Ok(py.None());
        };
        let mut data = Vec::with_capacity(pos.len() * 3);
        for p in pos.iter() {
            data.extend_from_slice(&p[0..3]);
        }
        let arr = Array2::from_shape_vec((pos.len(), 3), data)
            .map_err(|_| PyRuntimeError::new_err("failed to build positions0 array"))?;
        Ok(arr.into_pyarray_bound(py).into_py(py))
    }
}

#[pyclass]
struct PySelection {
    selection: Selection,
}

#[pymethods]
impl PySelection {
    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let vec: Vec<u32> = self.selection.indices.as_ref().clone();
        Ok(vec.into_py(py))
    }
}

fn selected_frames_to_py<'py>(
    py: Python<'py>,
    selected: &[traj_engine::executor::SelectedFrame],
    source_indices: &[usize],
    include_box: bool,
    include_box_matrix: bool,
    include_time: bool,
) -> PyResult<PyObject> {
    let n_frames = selected.len();
    let n_atoms = selected[0].coords.len();
    let mut coord_data4 = Vec::with_capacity(n_frames * n_atoms * 4);
    for frame in selected.iter() {
        for atom in frame.coords.iter() {
            coord_data4.extend_from_slice(atom);
        }
    }
    let coords4 = Array3::from_shape_vec((n_frames, n_atoms, 4), coord_data4)
        .map_err(|_| PyRuntimeError::new_err("failed to build coords array"))?;
    let coords = coords4.slice_move(numpy::ndarray::s![.., .., 0..3]);
    let coords_py = coords.into_pyarray_bound(py);

    let mut box_data = if include_box {
        Some(Vec::with_capacity(n_frames * 3))
    } else {
        None
    };
    let mut box_matrix_data = if include_box_matrix {
        Some(Vec::with_capacity(n_frames * 9))
    } else {
        None
    };
    let mut box_ok = include_box;
    let mut box_matrix_ok = include_box_matrix;
    if include_box || include_box_matrix {
        for frame in selected.iter() {
            match frame.box_ {
                Box3::Orthorhombic { lx, ly, lz } => {
                    if let Some(data) = box_data.as_mut() {
                        data.push(lx);
                        data.push(ly);
                        data.push(lz);
                    }
                    if let Some(data) = box_matrix_data.as_mut() {
                        data.extend_from_slice(&[
                            lx, 0.0, 0.0, //
                            0.0, ly, 0.0, //
                            0.0, 0.0, lz,
                        ]);
                    }
                }
                Box3::None => {
                    box_ok = false;
                    box_matrix_ok = false;
                    break;
                }
                Box3::Triclinic { m } => {
                    box_ok = false;
                    if let Some(data) = box_matrix_data.as_mut() {
                        data.extend_from_slice(&m);
                    }
                    if !box_ok && !box_matrix_ok {
                        break;
                    }
                }
            }
        }
    }
    let box_py = if include_box && box_ok {
        let box_arr = Array2::from_shape_vec((n_frames, 3), box_data.unwrap_or_default())
            .map_err(|_| PyRuntimeError::new_err("failed to build box array"))?;
        box_arr.into_pyarray_bound(py).into_py(py)
    } else {
        py.None()
    };
    let box_matrix_py = if include_box_matrix && box_matrix_ok {
        let box_arr = Array3::from_shape_vec((n_frames, 3, 3), box_matrix_data.unwrap_or_default())
            .map_err(|_| PyRuntimeError::new_err("failed to build box matrix array"))?;
        box_arr.into_pyarray_bound(py).into_py(py)
    } else {
        py.None()
    };
    let time_py = if include_time {
        let mut times = Vec::with_capacity(n_frames);
        for frame in selected.iter() {
            times.push(frame.time_ps.unwrap_or(0.0));
        }
        PyArray1::from_vec_bound(py, times).into_py(py)
    } else {
        py.None()
    };

    let dict = PyDict::new_bound(py);
    dict.set_item("coords", coords_py)?;
    dict.set_item("box", box_py)?;
    dict.set_item("box_matrix", box_matrix_py)?;
    dict.set_item("time_ps", time_py)?;
    dict.set_item("frames", n_frames)?;
    dict.set_item("source_indices", source_indices)?;
    Ok(dict.into_py(py))
}

fn selected_trr_frames_to_py<'py>(
    py: Python<'py>,
    selected: &[traj_io::trr::TrrFrameData],
    source_indices: &[usize],
    include_box: bool,
    include_box_matrix: bool,
    include_time: bool,
    include_velocities: bool,
    include_forces: bool,
    include_lambda: bool,
) -> PyResult<PyObject> {
    let n_frames = selected.len();
    let n_atoms = selected[0].coords.len();
    let mut coord_data4 = Vec::with_capacity(n_frames * n_atoms * 4);
    for frame in selected.iter() {
        for atom in frame.coords.iter() {
            coord_data4.extend_from_slice(atom);
        }
    }
    let coords4 = Array3::from_shape_vec((n_frames, n_atoms, 4), coord_data4)
        .map_err(|_| PyRuntimeError::new_err("failed to build coords array"))?;
    let coords = coords4.slice_move(numpy::ndarray::s![.., .., 0..3]);
    let coords_py = coords.into_pyarray_bound(py);

    let mut box_data = if include_box {
        Some(Vec::with_capacity(n_frames * 3))
    } else {
        None
    };
    let mut box_matrix_data = if include_box_matrix {
        Some(Vec::with_capacity(n_frames * 9))
    } else {
        None
    };
    let mut box_ok = include_box;
    let mut box_matrix_ok = include_box_matrix;
    if include_box || include_box_matrix {
        for frame in selected.iter() {
            match frame.box_ {
                Box3::Orthorhombic { lx, ly, lz } => {
                    if let Some(data) = box_data.as_mut() {
                        data.push(lx);
                        data.push(ly);
                        data.push(lz);
                    }
                    if let Some(data) = box_matrix_data.as_mut() {
                        data.extend_from_slice(&[
                            lx, 0.0, 0.0, //
                            0.0, ly, 0.0, //
                            0.0, 0.0, lz,
                        ]);
                    }
                }
                Box3::None => {
                    box_ok = false;
                    box_matrix_ok = false;
                    break;
                }
                Box3::Triclinic { m } => {
                    box_ok = false;
                    if let Some(data) = box_matrix_data.as_mut() {
                        data.extend_from_slice(&m);
                    }
                    if !box_ok && !box_matrix_ok {
                        break;
                    }
                }
            }
        }
    }

    let box_py = if include_box && box_ok {
        let box_arr = Array2::from_shape_vec((n_frames, 3), box_data.unwrap_or_default())
            .map_err(|_| PyRuntimeError::new_err("failed to build box array"))?;
        box_arr.into_pyarray_bound(py).into_py(py)
    } else {
        py.None()
    };
    let box_matrix_py = if include_box_matrix && box_matrix_ok {
        let box_arr = Array3::from_shape_vec((n_frames, 3, 3), box_matrix_data.unwrap_or_default())
            .map_err(|_| PyRuntimeError::new_err("failed to build box matrix array"))?;
        box_arr.into_pyarray_bound(py).into_py(py)
    } else {
        py.None()
    };
    let time_py = if include_time {
        PyArray1::from_vec_bound(
            py,
            selected
                .iter()
                .map(|frame| frame.time_ps.unwrap_or(0.0))
                .collect(),
        )
        .into_py(py)
    } else {
        py.None()
    };
    let velocities_py = if include_velocities {
        let mut velocity_data = Vec::with_capacity(n_frames * n_atoms * 3);
        for frame in selected.iter() {
            if let Some(velocities) = frame.velocities.as_ref() {
                for atom in velocities.iter() {
                    velocity_data.extend_from_slice(atom);
                }
            }
        }
        Array3::from_shape_vec((n_frames, n_atoms, 3), velocity_data)
            .map_err(|_| PyRuntimeError::new_err("failed to build velocities array"))?
            .into_pyarray_bound(py)
            .into_py(py)
    } else {
        py.None()
    };
    let forces_py = if include_forces {
        let mut force_data = Vec::with_capacity(n_frames * n_atoms * 3);
        for frame in selected.iter() {
            if let Some(forces) = frame.forces.as_ref() {
                for atom in forces.iter() {
                    force_data.extend_from_slice(atom);
                }
            }
        }
        Array3::from_shape_vec((n_frames, n_atoms, 3), force_data)
            .map_err(|_| PyRuntimeError::new_err("failed to build forces array"))?
            .into_pyarray_bound(py)
            .into_py(py)
    } else {
        py.None()
    };
    let lambda_py = if include_lambda {
        PyArray1::from_vec_bound(
            py,
            selected
                .iter()
                .map(|frame| frame.lambda_value.unwrap_or(0.0))
                .collect(),
        )
        .into_py(py)
    } else {
        py.None()
    };

    let dict = PyDict::new_bound(py);
    dict.set_item("coords", coords_py)?;
    dict.set_item("box", box_py)?;
    dict.set_item("box_matrix", box_matrix_py)?;
    dict.set_item("time_ps", time_py)?;
    dict.set_item("velocities", velocities_py)?;
    dict.set_item("forces", forces_py)?;
    dict.set_item("lambda_value", lambda_py)?;
    dict.set_item("frames", n_frames)?;
    dict.set_item("source_indices", source_indices)?;
    Ok(dict.into_py(py))
}

enum TrajKind {
    Dcd { reader: DcdReader },
    Xtc { reader: XtcReader },
    Trr { reader: TrrReader },
    Pdb { reader: PdbTrajReader },
}

#[pyclass(unsendable)]
struct PyTrajectory {
    inner: RefCell<TrajKind>,
    auto_chunk_frames: RefCell<Option<usize>>,
    chunk_builder: RefCell<Option<FrameChunkBuilder>>,
}

#[pymethods]
impl PyTrajectory {
    #[staticmethod]
    #[pyo3(signature = (path, system, length_scale=None))]
    fn open_dcd(path: &str, system: &PySystem, length_scale: Option<f32>) -> PyResult<Self> {
        let scale = length_scale.unwrap_or(1.0);
        let reader = DcdReader::open(path, scale).map_err(to_py_err)?;
        let sys = system.system.borrow();
        if reader.n_atoms() != sys.n_atoms() {
            return Err(PyRuntimeError::new_err(
                "trajectory atom count does not match system",
            ));
        }
        Ok(Self {
            inner: RefCell::new(TrajKind::Dcd { reader }),
            auto_chunk_frames: RefCell::new(None),
            chunk_builder: RefCell::new(None),
        })
    }

    #[staticmethod]
    fn open_xtc(path: &str, system: &PySystem) -> PyResult<Self> {
        let reader = XtcReader::open(path).map_err(to_py_err)?;
        let sys = system.system.borrow();
        if reader.n_atoms() != sys.n_atoms() {
            return Err(PyRuntimeError::new_err(
                "trajectory atom count does not match system",
            ));
        }
        Ok(Self {
            inner: RefCell::new(TrajKind::Xtc { reader }),
            auto_chunk_frames: RefCell::new(None),
            chunk_builder: RefCell::new(None),
        })
    }

    #[staticmethod]
    fn open_trr(path: &str, system: &PySystem) -> PyResult<Self> {
        let reader = TrrReader::open(path).map_err(to_py_err)?;
        let sys = system.system.borrow();
        if reader.n_atoms() != sys.n_atoms() {
            return Err(PyRuntimeError::new_err(
                "trajectory atom count does not match system",
            ));
        }
        Ok(Self {
            inner: RefCell::new(TrajKind::Trr { reader }),
            auto_chunk_frames: RefCell::new(None),
            chunk_builder: RefCell::new(None),
        })
    }

    #[staticmethod]
    fn open_pdb(path: &str, system: &PySystem) -> PyResult<Self> {
        let reader = PdbTrajReader::open(path).map_err(to_py_err)?;
        let sys = system.system.borrow();
        if reader.n_atoms() != sys.n_atoms() {
            return Err(PyRuntimeError::new_err(
                "trajectory atom count does not match system",
            ));
        }
        Ok(Self {
            inner: RefCell::new(TrajKind::Pdb { reader }),
            auto_chunk_frames: RefCell::new(None),
            chunk_builder: RefCell::new(None),
        })
    }

    #[staticmethod]
    fn open_pdbqt(path: &str, system: &PySystem) -> PyResult<Self> {
        Self::open_pdb(path, system)
    }

    fn n_atoms(&self) -> PyResult<usize> {
        let inner = self.inner.borrow();
        let n = match &*inner {
            TrajKind::Dcd { reader, .. } => reader.n_atoms(),
            TrajKind::Xtc { reader, .. } => reader.n_atoms(),
            TrajKind::Trr { reader, .. } => reader.n_atoms(),
            TrajKind::Pdb { reader, .. } => reader.n_atoms(),
        };
        Ok(n)
    }

    #[pyo3(signature = (chunk_frames=None))]
    fn count_frames(&self, chunk_frames: Option<usize>) -> PyResult<usize> {
        let mut inner = self.inner.borrow_mut();
        let chunk_frames = if let Some(frames) = chunk_frames {
            frames.max(1)
        } else {
            resolve_chunk_frames_for_streaming(&inner)?
        };
        let total = match &mut *inner {
            TrajKind::Dcd { reader, .. } => {
                if let Some(hint) = reader.n_frames_hint() {
                    hint
                } else {
                    let counted = traj_engine::count_frames(reader, chunk_frames).map_err(to_py_err)?;
                    reader.reset().map_err(to_py_err)?;
                    counted
                }
            }
            TrajKind::Xtc { reader, .. } => {
                let counted = traj_engine::count_frames(reader, chunk_frames).map_err(to_py_err)?;
                reader.reset().map_err(to_py_err)?;
                counted
            }
            TrajKind::Trr { reader, .. } => {
                if let Some(hint) = reader.n_frames_hint() {
                    hint
                } else {
                    let counted = traj_engine::count_frames(reader, chunk_frames).map_err(to_py_err)?;
                    reader.reset().map_err(to_py_err)?;
                    counted
                }
            }
            TrajKind::Pdb { reader, .. } => {
                if let Some(hint) = reader.n_frames_hint() {
                    hint
                } else {
                    let counted = traj_engine::count_frames(reader, chunk_frames).map_err(to_py_err)?;
                    reader.reset();
                    counted
                }
            }
        };
        Ok(total)
    }

    fn reset(&self) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();
        match &mut *inner {
            TrajKind::Dcd { reader, .. } => {
                reader.reset().map_err(to_py_err)?;
            }
            TrajKind::Xtc { reader, .. } => {
                reader.reset().map_err(to_py_err)?;
            }
            TrajKind::Trr { reader, .. } => {
                reader.reset().map_err(to_py_err)?;
            }
            TrajKind::Pdb { reader, .. } => {
                reader.reset();
            }
        }
        *self.auto_chunk_frames.borrow_mut() = None;
        self.chunk_builder.borrow_mut().take();
        Ok(())
    }

    #[pyo3(signature = (frame_indices, chunk_frames=None, include_box=true, include_box_matrix=true, include_time=false, include_velocities=false, include_forces=false, include_lambda=false))]
    fn read_frames<'py>(
        &self,
        py: Python<'py>,
        frame_indices: Vec<i64>,
        chunk_frames: Option<usize>,
        include_box: bool,
        include_box_matrix: bool,
        include_time: bool,
        include_velocities: bool,
        include_forces: bool,
        include_lambda: bool,
    ) -> PyResult<PyObject> {
        if frame_indices.is_empty() {
            return Ok(py.None());
        }

        let mut inner = self.inner.borrow_mut();
        let chunk_frames = if let Some(frames) = chunk_frames {
            frames.max(1)
        } else {
            resolve_chunk_frames_for_streaming(&inner)?
        };

        let source_indices = if frame_indices.iter().any(|&idx| idx < 0) {
            let n_frames = if let Some(hint) = traj_n_frames_hint(&inner) {
                hint
            } else {
                let counted = match &mut *inner {
                    TrajKind::Dcd { reader, .. } => traj_engine::count_frames(reader, chunk_frames),
                    TrajKind::Xtc { reader, .. } => traj_engine::count_frames(reader, chunk_frames),
                    TrajKind::Trr { reader, .. } => traj_engine::count_frames(reader, chunk_frames),
                    TrajKind::Pdb { reader, .. } => traj_engine::count_frames(reader, chunk_frames),
                }
                .map_err(to_py_err)?;
                reset_traj(&mut inner).map_err(to_py_err)?;
                counted
            };
            traj_engine::normalize_frame_indices(frame_indices, n_frames)
        } else {
            frame_indices
                .into_iter()
                .filter_map(|idx| usize::try_from(idx).ok())
                .collect::<Vec<_>>()
        };

        let requirements = traj_engine::executor::PlanRequirements::new(
            include_box || include_box_matrix,
            include_time,
        );
        if let TrajKind::Trr { reader, .. } = &mut *inner {
            let selected = reader
                .collect_selected_frames(
                    &source_indices,
                    include_box || include_box_matrix,
                    include_time,
                    include_velocities,
                    include_forces,
                    include_lambda,
                )
                .map_err(to_py_err)?;
            if selected.is_empty() {
                return Ok(py.None());
            }
            return selected_trr_frames_to_py(
                py,
                &selected,
                &source_indices,
                include_box,
                include_box_matrix,
                include_time,
                include_velocities,
                include_forces,
                include_lambda,
            );
        }

        let selected = match &mut *inner {
            TrajKind::Dcd { reader, .. } => traj_engine::executor::collect_selected_frames_with_requirements(
                reader,
                &source_indices,
                chunk_frames,
                requirements,
            ),
            TrajKind::Xtc { reader, .. } => traj_engine::executor::collect_selected_frames_with_requirements(
                reader,
                &source_indices,
                chunk_frames,
                requirements,
            ),
            TrajKind::Trr { .. } => unreachable!(),
            TrajKind::Pdb { reader, .. } => traj_engine::executor::collect_selected_frames_with_requirements(
                reader,
                &source_indices,
                chunk_frames,
                requirements,
            ),
        }
        .map_err(to_py_err)?;

        if selected.is_empty() {
            return Ok(py.None());
        }

        selected_frames_to_py(
            py,
            &selected,
            &source_indices,
            include_box,
            include_box_matrix,
            include_time,
        )
    }

    #[pyo3(signature = (begin, end, step, chunk_frames=None, include_box=true, include_box_matrix=true, include_time=false, include_velocities=false, include_forces=false, include_lambda=false))]
    fn read_frame_range<'py>(
        &self,
        py: Python<'py>,
        begin: usize,
        end: usize,
        step: usize,
        chunk_frames: Option<usize>,
        include_box: bool,
        include_box_matrix: bool,
        include_time: bool,
        include_velocities: bool,
        include_forces: bool,
        include_lambda: bool,
    ) -> PyResult<PyObject> {
        if begin >= end {
            return Ok(py.None());
        }
        if step == 0 {
            return Err(PyRuntimeError::new_err("step must be >= 1"));
        }

        let mut inner = self.inner.borrow_mut();
        let chunk_frames = if let Some(frames) = chunk_frames {
            frames.max(1)
        } else {
            resolve_chunk_frames_for_streaming(&inner)?
        };
        let requirements = traj_engine::executor::PlanRequirements::new(
            include_box || include_box_matrix,
            include_time,
        );
        if let TrajKind::Trr { reader, .. } = &mut *inner {
            let (selected, source_indices) = reader
                .collect_strided_frames(
                    begin,
                    end,
                    step,
                    include_box || include_box_matrix,
                    include_time,
                    include_velocities,
                    include_forces,
                    include_lambda,
                )
                .map_err(to_py_err)?;
            if selected.is_empty() {
                return Ok(py.None());
            }
            return selected_trr_frames_to_py(
                py,
                &selected,
                &source_indices,
                include_box,
                include_box_matrix,
                include_time,
                include_velocities,
                include_forces,
                include_lambda,
            );
        }

        let (selected, source_indices) = match &mut *inner {
            TrajKind::Dcd { reader, .. } => {
                traj_engine::executor::collect_strided_frames_with_requirements(
                    reader,
                    begin,
                    end,
                    step,
                    chunk_frames,
                    requirements,
                )
            }
            TrajKind::Xtc { reader, .. } => {
                traj_engine::executor::collect_strided_frames_with_requirements(
                    reader,
                    begin,
                    end,
                    step,
                    chunk_frames,
                    requirements,
                )
            }
            TrajKind::Trr { .. } => unreachable!(),
            TrajKind::Pdb { reader, .. } => {
                traj_engine::executor::collect_strided_frames_with_requirements(
                    reader,
                    begin,
                    end,
                    step,
                    chunk_frames,
                    requirements,
                )
            }
        }
        .map_err(to_py_err)?;

        if selected.is_empty() {
            return Ok(py.None());
        }

        selected_frames_to_py(
            py,
            &selected,
            &source_indices,
            include_box,
            include_box_matrix,
            include_time,
        )
    }

    #[pyo3(signature = (max_frames=None, include_box=true, include_box_matrix=true, include_time=true, atom_indices=None, include_velocities=false, include_forces=false, include_lambda=false))]
    fn read_chunk<'py>(
        &self,
        py: Python<'py>,
        max_frames: Option<usize>,
        include_box: bool,
        include_box_matrix: bool,
        include_time: bool,
        atom_indices: Option<Vec<u32>>,
        include_velocities: bool,
        include_forces: bool,
        include_lambda: bool,
    ) -> PyResult<PyObject> {
        if atom_indices.as_ref().is_some_and(|idx| idx.is_empty()) {
            return Err(PyRuntimeError::new_err("atom_indices must not be empty"));
        }
        let max_frames = if let Some(frames) = max_frames {
            frames.max(1)
        } else {
            if let Some(indices) = atom_indices.as_ref() {
                heuristic_chunk_frames(indices.len())
            } else {
                let cached_frames = { *self.auto_chunk_frames.borrow() };
                if let Some(frames) = cached_frames {
                    frames
                } else {
                    let frames = {
                        let inner = self.inner.borrow();
                        resolve_chunk_frames_for_streaming(&inner)?
                    };
                    self.auto_chunk_frames.replace(Some(frames));
                    frames
                }
            }
        };

        let mut inner = self.inner.borrow_mut();
        let traj_n_atoms = match &*inner {
            TrajKind::Dcd { reader, .. } => reader.n_atoms(),
            TrajKind::Xtc { reader, .. } => reader.n_atoms(),
            TrajKind::Trr { reader, .. } => reader.n_atoms(),
            TrajKind::Pdb { reader, .. } => reader.n_atoms(),
        };
        let n_atoms = atom_indices
            .as_ref()
            .map(|idx| idx.len())
            .unwrap_or(traj_n_atoms);
        let mut builder = FrameChunkBuilder::new(n_atoms, max_frames);
        builder.set_requirements(include_box || include_box_matrix, include_time);
        builder.set_optional_requirements(include_velocities, include_forces, include_lambda);
        let frames = match (&mut *inner, atom_indices.as_deref()) {
            (TrajKind::Dcd { reader, .. }, Some(selection)) => {
                reader.read_chunk_selected(max_frames, selection, &mut builder)
            }
            (TrajKind::Dcd { reader, .. }, None) => reader.read_chunk(max_frames, &mut builder),
            (TrajKind::Xtc { reader, .. }, Some(selection)) => {
                reader.read_chunk_selected(max_frames, selection, &mut builder)
            }
            (TrajKind::Xtc { reader, .. }, None) => reader.read_chunk(max_frames, &mut builder),
            (TrajKind::Trr { reader, .. }, Some(selection)) => {
                reader.read_chunk_selected(max_frames, selection, &mut builder)
            }
            (TrajKind::Trr { reader, .. }, None) => reader.read_chunk(max_frames, &mut builder),
            (TrajKind::Pdb { reader, .. }, Some(selection)) => {
                reader.read_chunk_selected(max_frames, selection, &mut builder)
            }
            (TrajKind::Pdb { reader, .. }, None) => reader.read_chunk(max_frames, &mut builder),
        }
        .map_err(to_py_err)?;
        if frames == 0 {
            return Ok(py.None());
        }
        let chunk = builder.finish_take().map_err(to_py_err)?;
        let traj_core::frame::FrameChunk {
            n_atoms: chunk_n_atoms,
            n_frames: chunk_n_frames,
            coords,
            box_,
            time_ps,
            velocities,
            forces,
            lambda_values,
        } = chunk;
        let coord_data4 = flatten_coords4_no_copy(coords);
        let coords4 = Array3::from_shape_vec((chunk_n_frames, chunk_n_atoms, 4), coord_data4)
            .map_err(|_| PyRuntimeError::new_err("failed to build coords array"))?;
        let coords = coords4.slice_move(numpy::ndarray::s![.., .., 0..3]);
        let coords_py = coords.into_pyarray_bound(py);

        let mut box_data = if include_box {
            Some(Vec::with_capacity(chunk_n_frames * 3))
        } else {
            None
        };
        let mut box_matrix_data = if include_box_matrix {
            Some(Vec::with_capacity(chunk_n_frames * 9))
        } else {
            None
        };
        let mut box_ok = include_box;
        let mut box_matrix_ok = include_box_matrix;
        if include_box || include_box_matrix {
            for b in box_.iter() {
                match *b {
                    Box3::Orthorhombic { lx, ly, lz } => {
                        if let Some(data) = box_data.as_mut() {
                            data.push(lx);
                            data.push(ly);
                            data.push(lz);
                        }
                        if let Some(data) = box_matrix_data.as_mut() {
                            data.extend_from_slice(&[
                                lx, 0.0, 0.0, //
                                0.0, ly, 0.0, //
                                0.0, 0.0, lz,
                            ]);
                        }
                    }
                    Box3::None => {
                        box_ok = false;
                        box_matrix_ok = false;
                        break;
                    }
                    Box3::Triclinic { m } => {
                        box_ok = false;
                        if let Some(data) = box_matrix_data.as_mut() {
                            data.extend_from_slice(&m);
                        }
                        if !box_ok && !box_matrix_ok {
                            break;
                        }
                    }
                }
            }
        }
        let box_py = if include_box && box_ok {
            let box_arr = Array2::from_shape_vec(
                (chunk_n_frames, 3),
                box_data.unwrap_or_default(),
            )
            .map_err(|_| PyRuntimeError::new_err("failed to build box array"))?;
            box_arr.into_pyarray_bound(py).into_py(py)
        } else {
            py.None()
        };
        let box_matrix_py = if include_box_matrix && box_matrix_ok {
            let box_arr = Array3::from_shape_vec(
                (chunk_n_frames, 3, 3),
                box_matrix_data.unwrap_or_default(),
            )
            .map_err(|_| PyRuntimeError::new_err("failed to build box matrix array"))?;
            box_arr.into_pyarray_bound(py).into_py(py)
        } else {
            py.None()
        };

        let time_py = if include_time {
            if let Some(times) = time_ps {
                PyArray1::from_vec_bound(py, times).into_py(py)
            } else {
                py.None()
            }
        } else {
            py.None()
        };
        let velocities_py = if include_velocities {
            if let Some(velocities) = velocities {
                let data = velocities
                    .into_iter()
                    .flat_map(|atom| atom.into_iter())
                    .collect::<Vec<_>>();
                let arr = Array3::from_shape_vec((chunk_n_frames, chunk_n_atoms, 3), data)
                    .map_err(|_| PyRuntimeError::new_err("failed to build velocities array"))?;
                arr.into_pyarray_bound(py).into_py(py)
            } else {
                py.None()
            }
        } else {
            py.None()
        };
        let forces_py = if include_forces {
            if let Some(forces) = forces {
                let data = forces
                    .into_iter()
                    .flat_map(|atom| atom.into_iter())
                    .collect::<Vec<_>>();
                let arr = Array3::from_shape_vec((chunk_n_frames, chunk_n_atoms, 3), data)
                    .map_err(|_| PyRuntimeError::new_err("failed to build forces array"))?;
                arr.into_pyarray_bound(py).into_py(py)
            } else {
                py.None()
            }
        } else {
            py.None()
        };
        let lambda_py = if include_lambda {
            if let Some(lambda_values) = lambda_values {
                PyArray1::from_vec_bound(py, lambda_values).into_py(py)
            } else {
                py.None()
            }
        } else {
            py.None()
        };

        let dict = PyDict::new_bound(py);
        dict.set_item("coords", coords_py)?;
        dict.set_item("box", box_py)?;
        dict.set_item("box_matrix", box_matrix_py)?;
        dict.set_item("time_ps", time_py)?;
        dict.set_item("velocities", velocities_py)?;
        dict.set_item("forces", forces_py)?;
        dict.set_item("lambda_value", lambda_py)?;
        dict.set_item("frames", chunk_n_frames)?;
        Ok(dict.into_py(py))
    }

    #[pyo3(signature = (coords, box_out=None, time_out=None, max_frames=None, atom_indices=None))]
    fn read_chunk_into(
        &self,
        coords: &PyArray3<f32>,
        box_out: Option<&PyArray2<f32>>,
        time_out: Option<&PyArray1<f32>>,
        max_frames: Option<usize>,
        atom_indices: Option<Vec<u32>>,
    ) -> PyResult<usize> {
        if !coords.is_c_contiguous() {
            return Err(PyRuntimeError::new_err(
                "coords must be a C-contiguous float32 array",
            ));
        }
        let shape = coords.shape();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(PyRuntimeError::new_err(
                "coords must have shape (frames, atoms, 3)",
            ));
        }
        let cap_frames = shape[0];
        let n_atoms = shape[1];
        if n_atoms == 0 {
            return Err(PyRuntimeError::new_err("coords atom dimension must be > 0"));
        }
        if cap_frames == 0 {
            return Err(PyRuntimeError::new_err("coords frame dimension must be > 0"));
        }
        let target_frames = max_frames.unwrap_or(cap_frames).max(1).min(cap_frames);

        if let Some(box_arr) = box_out {
            if !box_arr.is_c_contiguous() {
                return Err(PyRuntimeError::new_err(
                    "box_out must be a C-contiguous float32 array",
                ));
            }
            let box_shape = box_arr.shape();
            if box_shape.len() != 2 || box_shape[1] != 3 || box_shape[0] < target_frames {
                return Err(PyRuntimeError::new_err(
                    "box_out must have shape (frames, 3) with frames >= max_frames",
                ));
            }
        }
        if let Some(time_arr) = time_out {
            if !time_arr.is_c_contiguous() {
                return Err(PyRuntimeError::new_err(
                    "time_out must be a C-contiguous float32 array",
                ));
            }
            let time_shape = time_arr.shape();
            if time_shape.len() != 1 || time_shape[0] < target_frames {
                return Err(PyRuntimeError::new_err(
                    "time_out must have shape (frames,) with frames >= max_frames",
                ));
            }
        }

        if atom_indices.as_ref().is_some_and(|idx| idx.is_empty()) {
            return Err(PyRuntimeError::new_err("atom_indices must not be empty"));
        }
        let mut inner = self.inner.borrow_mut();
        let traj_n_atoms = match &*inner {
            TrajKind::Dcd { reader, .. } => reader.n_atoms(),
            TrajKind::Xtc { reader, .. } => reader.n_atoms(),
            TrajKind::Trr { reader, .. } => reader.n_atoms(),
            TrajKind::Pdb { reader, .. } => reader.n_atoms(),
        };
        let expected_atoms = atom_indices
            .as_ref()
            .map(|idx| idx.len())
            .unwrap_or(traj_n_atoms);
        if expected_atoms != n_atoms {
            return Err(PyRuntimeError::new_err(
                "coords atom dimension does not match requested trajectory atom count",
            ));
        }

        // Coords-only fast path for DCD/XTC streaming: decode directly into caller
        // buffer to avoid intermediate FrameChunk materialization and AoS->SoA copies.
        if box_out.is_none() && time_out.is_none() {
            let coords_slice = unsafe {
                coords
                    .as_slice_mut()
                    .map_err(|_| PyRuntimeError::new_err("coords must be contiguous"))?
            };
            match (&mut *inner, atom_indices.as_deref()) {
                (TrajKind::Dcd { reader, .. }, None) => {
                    let read = reader
                        .read_chunk_into_coords3(target_frames, coords_slice)
                        .map_err(to_py_err)?;
                    return Ok(read);
                }
                (TrajKind::Dcd { reader, .. }, Some(selection)) => {
                    let read = reader
                        .read_chunk_into_coords3_selected(target_frames, selection, coords_slice)
                        .map_err(to_py_err)?;
                    return Ok(read);
                }
                (TrajKind::Xtc { reader, .. }, None) => {
                    let read = reader
                        .read_chunk_into_coords3(target_frames, coords_slice)
                        .map_err(to_py_err)?;
                    return Ok(read);
                }
                (TrajKind::Xtc { reader, .. }, Some(selection)) => {
                    let read = reader
                        .read_chunk_into_coords3_selected(target_frames, selection, coords_slice)
                        .map_err(to_py_err)?;
                    return Ok(read);
                }
                (TrajKind::Trr { .. }, _) => {}
                (TrajKind::Pdb { .. }, _) => {}
            }
        }

        let mut builder = self.take_chunk_builder(
            expected_atoms,
            target_frames,
            box_out.is_some(),
            time_out.is_some(),
        );
        let read = match (&mut *inner, atom_indices.as_deref()) {
            (TrajKind::Dcd { reader, .. }, Some(selection)) => {
                reader.read_chunk_selected(target_frames, selection, &mut builder)
            }
            (TrajKind::Dcd { reader, .. }, None) => reader.read_chunk(target_frames, &mut builder),
            (TrajKind::Xtc { reader, .. }, Some(selection)) => {
                reader.read_chunk_selected(target_frames, selection, &mut builder)
            }
            (TrajKind::Xtc { reader, .. }, None) => reader.read_chunk(target_frames, &mut builder),
            (TrajKind::Trr { reader, .. }, Some(selection)) => {
                reader.read_chunk_selected(target_frames, selection, &mut builder)
            }
            (TrajKind::Trr { reader, .. }, None) => reader.read_chunk(target_frames, &mut builder),
            (TrajKind::Pdb { reader, .. }, Some(selection)) => {
                reader.read_chunk_selected(target_frames, selection, &mut builder)
            }
            (TrajKind::Pdb { reader, .. }, None) => reader.read_chunk(target_frames, &mut builder),
        }
        .map_err(to_py_err)?;
        if read == 0 {
            self.put_chunk_builder(builder);
            return Ok(0);
        }

        let chunk = builder.finish_take().map_err(to_py_err)?;
        let n_frames = chunk.n_frames;

        {
            let coords_slice = unsafe {
                coords
                    .as_slice_mut()
                    .map_err(|_| PyRuntimeError::new_err("coords must be contiguous"))?
            };
            for frame in 0..n_frames {
                let src_base = frame * chunk.n_atoms;
                let dst_frame_base = frame * chunk.n_atoms * 3;
                for atom in 0..chunk.n_atoms {
                    let src = chunk.coords[src_base + atom];
                    let dst_base = dst_frame_base + atom * 3;
                    coords_slice[dst_base] = src[0];
                    coords_slice[dst_base + 1] = src[1];
                    coords_slice[dst_base + 2] = src[2];
                }
            }
        }

        if let Some(box_arr) = box_out {
            let box_slice = unsafe {
                box_arr
                    .as_slice_mut()
                    .map_err(|_| PyRuntimeError::new_err("box_out must be contiguous"))?
            };
            if !chunk.box_.is_empty() {
                for frame in 0..n_frames {
                    let dst = &mut box_slice[frame * 3..frame * 3 + 3];
                    match chunk.box_[frame] {
                        Box3::Orthorhombic { lx, ly, lz } => {
                            dst[0] = lx;
                            dst[1] = ly;
                            dst[2] = lz;
                        }
                        Box3::Triclinic { m } => {
                            dst[0] = m[0];
                            dst[1] = m[4];
                            dst[2] = m[8];
                        }
                        Box3::None => {
                            dst[0] = 0.0;
                            dst[1] = 0.0;
                            dst[2] = 0.0;
                        }
                    }
                }
            } else {
                for v in box_slice.iter_mut().take(n_frames * 3) {
                    *v = 0.0;
                }
            }
        }

        if let Some(time_arr) = time_out {
            let time_slice = unsafe {
                time_arr
                    .as_slice_mut()
                    .map_err(|_| PyRuntimeError::new_err("time_out must be contiguous"))?
            };
            if let Some(times) = chunk.time_ps.as_ref() {
                time_slice[..n_frames].copy_from_slice(&times[..n_frames]);
            } else {
                for v in time_slice.iter_mut().take(n_frames) {
                    *v = 0.0;
                }
            }
        }

        builder.reclaim(chunk);
        self.put_chunk_builder(builder);
        Ok(n_frames)
    }
}

impl PyTrajectory {
    fn take_chunk_builder(
        &self,
        n_atoms: usize,
        max_frames: usize,
        needs_box: bool,
        needs_time: bool,
    ) -> FrameChunkBuilder {
        let mut slot = self.chunk_builder.borrow_mut();
        let mut builder = slot
            .take()
            .unwrap_or_else(|| FrameChunkBuilder::new(n_atoms, max_frames));
        builder.set_requirements(needs_box, needs_time);
        builder.set_optional_requirements(false, false, false);
        builder.reset(n_atoms, max_frames);
        builder
    }

    fn put_chunk_builder(&self, builder: FrameChunkBuilder) {
        self.chunk_builder.replace(Some(builder));
    }
}

enum TrajWriteKind {
    Dcd { writer: DcdWriter },
    Xtc { writer: XtcWriter },
    Trr { writer: TrrWriter },
}

#[pyclass(unsendable)]
struct PyTrajectoryWriter {
    inner: RefCell<TrajWriteKind>,
    n_atoms: usize,
    frames_written: RefCell<usize>,
}

#[pymethods]
impl PyTrajectoryWriter {
    #[staticmethod]
    fn open(path: &str, format: &str, n_atoms: usize, n_frames: Option<usize>) -> PyResult<Self> {
        if n_atoms == 0 {
            return Err(PyRuntimeError::new_err("n_atoms must be > 0"));
        }
        let inner = match format.trim().to_ascii_lowercase().as_str() {
            "dcd" => TrajWriteKind::Dcd {
                writer: DcdWriter::create(path, n_atoms, n_frames.unwrap_or(0)).map_err(to_py_err)?,
            },
            "xtc" => TrajWriteKind::Xtc {
                writer: XtcWriter::create(path, n_atoms).map_err(to_py_err)?,
            },
            "trr" => TrajWriteKind::Trr {
                writer: TrrWriter::create(path, n_atoms).map_err(to_py_err)?,
            },
            other => {
                return Err(PyRuntimeError::new_err(format!(
                    "unsupported trajectory writer format: {other}"
                )))
            }
        };
        Ok(Self {
            inner: RefCell::new(inner),
            n_atoms,
            frames_written: RefCell::new(0),
        })
    }

    #[pyo3(signature = (coords, box_lengths=None, box_matrix=None, step=None, time_ps=None, velocities=None, forces=None, lambda_value=None))]
    fn write_frame(
        &self,
        coords: PyReadonlyArray2<'_, f32>,
        box_lengths: Option<PyReadonlyArray1<'_, f32>>,
        box_matrix: Option<PyReadonlyArray2<'_, f32>>,
        step: Option<usize>,
        time_ps: Option<f32>,
        velocities: Option<PyReadonlyArray2<'_, f32>>,
        forces: Option<PyReadonlyArray2<'_, f32>>,
        lambda_value: Option<f32>,
    ) -> PyResult<()> {
        let coords = coords_to_vec(coords, self.n_atoms)?;
        let velocities = optional_coords_to_vec(velocities, self.n_atoms)?;
        let forces = optional_coords_to_vec(forces, self.n_atoms)?;
        let box_ = py_box_to_box3(box_lengths, box_matrix)?;
        let mut frames_written = self.frames_written.borrow_mut();
        let step = step.unwrap_or(*frames_written);
        let mut inner = self.inner.borrow_mut();
        match &mut *inner {
            TrajWriteKind::Dcd { writer } => writer.write_frame(&coords, box_).map_err(to_py_err)?,
            TrajWriteKind::Xtc { writer } => {
                writer
                    .write_frame(&coords, box_, step, time_ps)
                    .map_err(to_py_err)?
            }
            TrajWriteKind::Trr { writer } => {
                writer
                    .write_frame(
                        &coords,
                        box_,
                        step,
                        time_ps,
                        velocities.as_deref(),
                        forces.as_deref(),
                        lambda_value,
                    )
                    .map_err(to_py_err)?
            }
        }
        *frames_written += 1;
        Ok(())
    }

    fn flush(&self) -> PyResult<()> {
        let mut inner = self.inner.borrow_mut();
        match &mut *inner {
            TrajWriteKind::Dcd { writer } => writer.flush().map_err(to_py_err)?,
            TrajWriteKind::Xtc { writer } => writer.flush().map_err(to_py_err)?,
            TrajWriteKind::Trr { writer } => writer.flush().map_err(to_py_err)?,
        }
        Ok(())
    }
}

fn coords_to_vec(coords: PyReadonlyArray2<'_, f32>, n_atoms: usize) -> PyResult<Vec<[f32; 3]>> {
    let view = coords.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[0] != n_atoms || shape[1] != 3 {
        return Err(PyRuntimeError::new_err(format!(
            "coords must have shape ({n_atoms}, 3)"
        )));
    }
    let mut out = Vec::with_capacity(n_atoms);
    for row in view.outer_iter() {
        out.push([row[0], row[1], row[2]]);
    }
    Ok(out)
}

fn optional_coords_to_vec(
    coords: Option<PyReadonlyArray2<'_, f32>>,
    n_atoms: usize,
) -> PyResult<Option<Vec<[f32; 3]>>> {
    coords.map(|value| coords_to_vec(value, n_atoms)).transpose()
}

fn py_box_to_box3(
    box_lengths: Option<PyReadonlyArray1<'_, f32>>,
    box_matrix: Option<PyReadonlyArray2<'_, f32>>,
) -> PyResult<Box3> {
    if let Some(matrix) = box_matrix {
        let view = matrix.as_array();
        let shape = view.shape();
        if shape.len() != 2 || shape[0] != 3 || shape[1] != 3 {
            return Err(PyRuntimeError::new_err("box_matrix must have shape (3, 3)"));
        }
        return Ok(Box3::Triclinic {
            m: [
                view[[0, 0]],
                view[[0, 1]],
                view[[0, 2]],
                view[[1, 0]],
                view[[1, 1]],
                view[[1, 2]],
                view[[2, 0]],
                view[[2, 1]],
                view[[2, 2]],
            ],
        });
    }
    if let Some(lengths) = box_lengths {
        let view = lengths.as_array();
        if view.len() != 3 {
            return Err(PyRuntimeError::new_err("box_lengths must have shape (3,)"));
        }
        return Ok(Box3::Orthorhombic {
            lx: view[0],
            ly: view[1],
            lz: view[2],
        });
    }
    Ok(Box3::None)
}

fn flatten_coords4_no_copy(coords: Vec<[f32; 4]>) -> Vec<f32> {
    // SAFETY:
    // - `[f32; 4]` is a plain contiguous block of 4 `f32` values.
    // - `Vec<[f32; 4]>` and `Vec<f32>` share compatible allocation layout when
    //   pointer, len, and capacity are adjusted by a factor of 4.
    // - We use `ManuallyDrop` to prevent double-free of the original vector.
    let mut coords = std::mem::ManuallyDrop::new(coords);
    let ptr = coords.as_mut_ptr() as *mut f32;
    let len = coords.len().saturating_mul(4);
    let cap = coords.capacity().saturating_mul(4);
    unsafe { Vec::from_raw_parts(ptr, len, cap) }
}

#[pyclass]
struct PyRgPlan {
    plan: RefCell<RgPlan>,
}

#[pymethods]
impl PyRgPlan {
    #[new]
    fn new(selection: &PySelection, mass_weighted: Option<bool>) -> Self {
        let plan = RgPlan::new(selection.selection.clone(), mass_weighted.unwrap_or(false));
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
struct PyRadgyrTensorPlan {
    plan: RefCell<RadgyrTensorPlan>,
}

#[pymethods]
impl PyRadgyrTensorPlan {
    #[new]
    #[pyo3(signature = (selection, mass_weighted=false))]
    fn new(selection: &PySelection, mass_weighted: bool) -> Self {
        let plan = RadgyrTensorPlan::new(selection.selection.clone(), mass_weighted);
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
struct PyRmsdPlan {
    plan: RefCell<RmsdPlan>,
}

#[pymethods]
impl PyRmsdPlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", align=true))]
    fn new(selection: &PySelection, reference: &str, align: bool) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let plan = RmsdPlan::new(selection.selection.clone(), reference_mode, align);
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
struct PySymmRmsdPlan {
    plan: RefCell<SymmRmsdPlan>,
}

#[pymethods]
impl PySymmRmsdPlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", align=true))]
    fn new(selection: &PySelection, reference: &str, align: bool) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let plan = SymmRmsdPlan::new(selection.selection.clone(), reference_mode, align);
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
struct PyDistanceRmsdPlan {
    plan: RefCell<DistanceRmsdPlan>,
}

#[pymethods]
impl PyDistanceRmsdPlan {
    #[new]
    #[pyo3(signature = (selection, reference="topology", pbc="none"))]
    fn new(selection: &PySelection, reference: &str, pbc: &str) -> PyResult<Self> {
        let reference_mode = parse_reference(reference)?;
        let pbc = parse_pbc(pbc)?;
        let plan = DistanceRmsdPlan::new(selection.selection.clone(), reference_mode, pbc);
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
struct PyPairwiseRmsdPlan {
    plan: RefCell<PairwiseRmsdPlan>,
}

#[pymethods]
impl PyPairwiseRmsdPlan {
    #[new]
    #[pyo3(signature = (selection, metric="rms", pbc="none"))]
    fn new(selection: &PySelection, metric: &str, pbc: &str) -> PyResult<Self> {
        let metric = parse_pairwise_metric(metric)?;
        let pbc = parse_pbc(pbc)?;
        let plan = PairwiseRmsdPlan::new(selection.selection.clone(), metric, pbc);
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
struct PyTrajectoryClusterPlan {
    plan: RefCell<TrajectoryClusterPlan>,
}

#[pymethods]
impl PyTrajectoryClusterPlan {
    #[new]
    #[pyo3(signature = (
        selection,
        method="dbscan",
        eps=2.0,
        min_samples=5,
        n_clusters=8,
        max_iter=100,
        tol=1.0e-4,
        seed=0,
        memory_budget_bytes=None
    ))]
    fn new(
        selection: &PySelection,
        method: &str,
        eps: f32,
        min_samples: usize,
        n_clusters: usize,
        max_iter: usize,
        tol: f32,
        seed: u64,
        memory_budget_bytes: Option<usize>,
    ) -> PyResult<Self> {
        let method_l = method.trim().to_ascii_lowercase();
        let cluster_method = match method_l.as_str() {
            "dbscan" => ClusterMethod::Dbscan { eps, min_samples },
            "kmeans" => ClusterMethod::Kmeans {
                n_clusters,
                max_iter,
                tol,
                seed,
            },
            _ => {
                return Err(PyRuntimeError::new_err(
                    "method must be 'dbscan' or 'kmeans'",
                ))
            }
        };
        let plan = TrajectoryClusterPlan::new(selection.selection.clone(), cluster_method)
            .with_memory_budget_bytes(memory_budget_bytes);
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
            PlanOutput::Clustering(output) => clustering_to_py(py, output),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyMatrixPlan {
    plan: RefCell<MatrixPlan>,
}

#[pymethods]
impl PyMatrixPlan {
    #[new]
    #[pyo3(signature = (selection, mode, pbc="none"))]
    fn new(selection: &PySelection, mode: &str, pbc: &str) -> PyResult<Self> {
        let mode = parse_matrix_mode(mode)?;
        let pbc = parse_pbc(pbc)?;
        let plan = MatrixPlan::new(selection.selection.clone(), mode, pbc);
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
struct PyPcaPlan {
    plan: RefCell<PcaPlan>,
}

#[pymethods]
impl PyPcaPlan {
    #[new]
    #[pyo3(signature = (selection, n_components, mass_weighted=false))]
    fn new(selection: &PySelection, n_components: usize, mass_weighted: bool) -> Self {
        let plan = PcaPlan::new(selection.selection.clone(), n_components, mass_weighted);
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
            PlanOutput::Pca(pca) => pca_to_py(py, pca),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}

#[pyclass]
struct PyAnalyzeModesPlan {
    plan: RefCell<AnalyzeModesPlan>,
}

#[pymethods]
impl PyAnalyzeModesPlan {
    #[new]
    #[pyo3(signature = (selection, n_components, mass_weighted=false))]
    fn new(selection: &PySelection, n_components: usize, mass_weighted: bool) -> Self {
        let plan = AnalyzeModesPlan::new(selection.selection.clone(), n_components, mass_weighted);
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
            PlanOutput::Pca(pca) => pca_to_py(py, pca),
            _ => Err(PyRuntimeError::new_err("unexpected output")),
        }
    }
}
