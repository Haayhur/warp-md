use super::*;

fn bin_index(value: f64, edges: &[f64]) -> Option<usize> {
    if edges.len() < 2 || value < edges[0] || value > edges[edges.len() - 1] {
        return None;
    }
    if value == edges[edges.len() - 1] {
        return Some(edges.len() - 2);
    }
    let mut lo = 0usize;
    let mut hi = edges.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if edges[mid] <= value {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.checked_sub(1).filter(|&idx| idx + 1 < edges.len())
}

fn compute_binned_statistic_2d(
    x: numpy::ndarray::ArrayView1<'_, f64>,
    y: numpy::ndarray::ArrayView1<'_, f64>,
    values: numpy::ndarray::ArrayView1<'_, f64>,
    x_edges: numpy::ndarray::ArrayView1<'_, f64>,
    y_edges: numpy::ndarray::ArrayView1<'_, f64>,
    statistic: &str,
) -> PyResult<(Array2<f32>, Array2<i64>)> {
    if x.len() != y.len() || x.len() != values.len() {
        return Err(PyValueError::new_err(
            "x, y, and values must have the same shape",
        ));
    }
    if x_edges.len() < 2 || y_edges.len() < 2 {
        return Err(PyValueError::new_err(
            "bins must define at least one bin per dimension",
        ));
    }
    let nx = x_edges.len() - 1;
    let ny = y_edges.len() - 1;
    let x_edges_slice = x_edges
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("x_edges must be contiguous"))?;
    let y_edges_slice = y_edges
        .as_slice()
        .ok_or_else(|| PyValueError::new_err("y_edges must be contiguous"))?;
    let mut counts = Array2::<i64>::zeros((nx, ny));
    let mut total = Array2::<f64>::zeros((nx, ny));
    let mut total2 = Array2::<f64>::zeros((nx, ny));
    let mut grouped: Vec<Vec<f64>> = vec![Vec::new(); nx * ny];
    let needs_grouped = matches!(statistic, "median" | "min" | "max");
    if !matches!(
        statistic,
        "count" | "mean" | "sum" | "std" | "median" | "min" | "max"
    ) {
        return Err(PyValueError::new_err(
            "statistic must be mean, std, median, count, sum, min, or max",
        ));
    }

    for idx in 0..x.len() {
        let xv = x[idx];
        let yv = y[idx];
        let v = values[idx];
        if !xv.is_finite() || !yv.is_finite() || !v.is_finite() {
            continue;
        }
        let Some(ix) = bin_index(xv, x_edges_slice) else {
            continue;
        };
        let Some(iy) = bin_index(yv, y_edges_slice) else {
            continue;
        };
        counts[[ix, iy]] += 1;
        total[[ix, iy]] += v;
        if statistic == "std" {
            total2[[ix, iy]] += v * v;
        }
        if needs_grouped {
            grouped[ix * ny + iy].push(v);
        }
    }

    let mut out = Array2::<f32>::from_elem((nx, ny), f32::NAN);
    for ix in 0..nx {
        for iy in 0..ny {
            let count = counts[[ix, iy]];
            match statistic {
                "count" => out[[ix, iy]] = count as f32,
                "sum" => out[[ix, iy]] = total[[ix, iy]] as f32,
                "mean" => {
                    if count > 0 {
                        out[[ix, iy]] = (total[[ix, iy]] / count as f64) as f32;
                    }
                }
                "std" => {
                    if count > 0 {
                        let mean = total[[ix, iy]] / count as f64;
                        let variance = total2[[ix, iy]] / count as f64 - mean * mean;
                        out[[ix, iy]] = variance.max(0.0).sqrt() as f32;
                    }
                }
                "median" | "min" | "max" => {
                    let cell = &mut grouped[ix * ny + iy];
                    if cell.is_empty() {
                        continue;
                    }
                    cell.sort_by(|a, b| a.total_cmp(b));
                    out[[ix, iy]] = match statistic {
                        "min" => cell[0] as f32,
                        "max" => cell[cell.len() - 1] as f32,
                        _ => {
                            let mid = cell.len() / 2;
                            if cell.len() % 2 == 0 {
                                ((cell[mid - 1] + cell[mid]) * 0.5) as f32
                            } else {
                                cell[mid] as f32
                            }
                        }
                    };
                }
                _ => unreachable!(),
            }
        }
    }
    Ok((out, counts))
}

fn compute_nearest_fill_grid(grid: numpy::ndarray::ArrayView2<'_, f32>, tile: bool) -> Array2<f32> {
    let shape = grid.shape();
    let nx = shape[0];
    let ny = shape[1];
    let mut has_missing = false;
    let mut known_original = Vec::new();
    for ix in 0..nx {
        for iy in 0..ny {
            let value = grid[[ix, iy]];
            if value.is_finite() {
                known_original.push((ix, iy, value));
            } else {
                has_missing = true;
            }
        }
    }
    if !has_missing || known_original.is_empty() {
        return grid.to_owned();
    }

    let mut out = grid.to_owned();
    if !tile {
        for ix in 0..nx {
            for iy in 0..ny {
                if grid[[ix, iy]].is_finite() {
                    continue;
                }
                let mut best_dist = usize::MAX;
                let mut best_value = f32::NAN;
                for &(kx, ky, value) in &known_original {
                    let dx = ix.abs_diff(kx);
                    let dy = iy.abs_diff(ky);
                    let dist = dx * dx + dy * dy;
                    if dist < best_dist {
                        best_dist = dist;
                        best_value = value;
                    }
                }
                out[[ix, iy]] = best_value;
            }
        }
        return out;
    }

    let mut known_tiled = Vec::with_capacity(known_original.len() * 9);
    for tx in 0..3 {
        for ix in 0..nx {
            for ty in 0..3 {
                for iy in 0..ny {
                    let value = grid[[ix, iy]];
                    if value.is_finite() {
                        known_tiled.push((tx * nx + ix, ty * ny + iy, value));
                    }
                }
            }
        }
    }
    for ix in 0..nx {
        for iy in 0..ny {
            if grid[[ix, iy]].is_finite() {
                continue;
            }
            let px = nx + ix;
            let py = ny + iy;
            let mut best_dist = usize::MAX;
            let mut best_value = f32::NAN;
            for &(kx, ky, value) in &known_tiled {
                let dx = px.abs_diff(kx);
                let dy = py.abs_diff(ky);
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best_value = value;
                }
            }
            out[[ix, iy]] = best_value;
        }
    }
    out
}

fn compute_lipid_neighbour_composition(
    matrix: numpy::ndarray::ArrayView3<'_, i64>,
    labels: numpy::ndarray::ArrayView2<'_, i64>,
    label_values: numpy::ndarray::ArrayView1<'_, i64>,
) -> PyResult<Array3<i32>> {
    let shape = matrix.shape();
    let n_frames = shape[0];
    let n_residues = shape[1];
    if shape[2] != n_residues {
        return Err(PyValueError::new_err(
            "neighbour_matrix must have shape (n_frames, n_residues, n_residues)",
        ));
    }
    if labels.shape() != [n_residues, n_frames] {
        return Err(PyValueError::new_err(
            "labels must have shape (n_residues, n_frames)",
        ));
    }
    let n_labels = label_values.len();
    let mut counts = Array3::<i32>::zeros((n_frames, n_residues, n_labels));
    for frame in 0..n_frames {
        for residue in 0..n_residues {
            for neighbor in 0..n_residues {
                let edge = matrix[[frame, residue, neighbor]];
                if edge == 0 {
                    continue;
                }
                let label = labels[[neighbor, frame]];
                if let Some(label_idx) = label_values.iter().position(|&value| value == label) {
                    counts[[frame, residue, label_idx]] += edge as i32;
                }
            }
        }
    }
    Ok(counts)
}

fn compute_lipid_scd_chunk(
    coords: numpy::ndarray::ArrayView3<'_, f64>,
    idx_a: numpy::ndarray::ArrayView1<'_, i64>,
    idx_b: numpy::ndarray::ArrayView1<'_, i64>,
    axis: numpy::ndarray::ArrayView1<'_, f64>,
    pbc: &str,
    box_lengths: Option<numpy::ndarray::ArrayView2<'_, f64>>,
) -> PyResult<(Array2<f32>, Array2<i64>)> {
    if coords.ndim() != 3 || coords.shape()[2] != 3 {
        return Err(PyValueError::new_err(
            "coords must have shape (n_frames, n_atoms, 3)",
        ));
    }
    if idx_a.len() != idx_b.len() {
        return Err(PyValueError::new_err(
            "idx_a and idx_b must have the same length",
        ));
    }
    if axis.len() != 3 {
        return Err(PyValueError::new_err("axis must have length 3"));
    }
    let use_pbc = match pbc {
        "none" => false,
        "orthorhombic" => true,
        _ => {
            return Err(PyValueError::new_err(
                "pbc must be 'none' or 'orthorhombic'",
            ))
        }
    };
    let n_frames = coords.shape()[0];
    let n_atoms = coords.shape()[1];
    let n_pairs = idx_a.len();
    let box_lengths = if use_pbc {
        let view =
            box_lengths.ok_or_else(|| PyValueError::new_err("pbc='orthorhombic' requires box"))?;
        if view.shape() != [n_frames, 3] {
            return Err(PyValueError::new_err("box must have shape (n_frames, 3)"));
        }
        Some(view)
    } else {
        None
    };

    let mut out = Array2::<f32>::zeros((n_frames, n_pairs));
    let mut valid = Array2::<i64>::zeros((n_frames, n_pairs));
    for frame in 0..n_frames {
        for pair in 0..n_pairs {
            let a = idx_a[pair];
            let b = idx_b[pair];
            if a < 0 || b < 0 || a as usize >= n_atoms || b as usize >= n_atoms {
                return Err(PyValueError::new_err(
                    "bond index contains out-of-range atom index",
                ));
            }
            let a = a as usize;
            let b = b as usize;
            let mut dx = coords[[frame, b, 0]] - coords[[frame, a, 0]];
            let mut dy = coords[[frame, b, 1]] - coords[[frame, a, 1]];
            let mut dz = coords[[frame, b, 2]] - coords[[frame, a, 2]];
            if let Some(ref boxes) = box_lengths {
                let lx = boxes[[frame, 0]];
                let ly = boxes[[frame, 1]];
                let lz = boxes[[frame, 2]];
                if lx == 0.0 || ly == 0.0 || lz == 0.0 {
                    return Err(PyValueError::new_err(
                        "box lengths must be non-zero for pbc",
                    ));
                }
                dx -= (dx / lx).round_ties_even() * lx;
                dy -= (dy / ly).round_ties_even() * ly;
                dz -= (dz / lz).round_ties_even() * lz;
            }
            let norm = (dx * dx + dy * dy + dz * dz).sqrt();
            if norm <= 0.0 {
                continue;
            }
            let dot = dx * axis[0] + dy * axis[1] + dz * axis[2];
            let cos = dot / norm;
            out[[frame, pair]] = (0.5 * (3.0 * cos * cos - 1.0)) as f32;
            valid[[frame, pair]] = 1;
        }
    }
    Ok((out, valid))
}

fn compute_haar_details(series: numpy::ndarray::ArrayView1<'_, f32>) -> Array2<f32> {
    let n_frames = series.len();
    if n_frames < 2 {
        return Array2::<f32>::zeros((0, 0));
    }

    let max_cols = n_frames / 2;
    let mut levels = 0usize;
    let mut level_len = n_frames;
    while level_len >= 2 {
        levels += 1;
        level_len /= 2;
    }

    let mut out = Array2::<f32>::zeros((levels, max_cols));
    let mut current: Vec<f64> = series.iter().map(|&value| value as f64).collect();
    let mut row = 0usize;
    while current.len() >= 2 {
        let pairs = current.len() / 2;
        let mut next = Vec::with_capacity(pairs);
        for idx in 0..pairs {
            let a = current[2 * idx];
            let b = current[2 * idx + 1];
            next.push(0.5 * (a + b));
            out[[row, idx]] = (0.5 * (a - b)) as f32;
        }
        current = next;
        row += 1;
    }
    out
}

fn compute_mode_corr(
    vecs: numpy::ndarray::ArrayView2<'_, f64>,
    average_coords: numpy::ndarray::ArrayView2<'_, f64>,
    pairs: numpy::ndarray::ArrayView2<'_, i64>,
) -> PyResult<Array2<f32>> {
    if average_coords.shape()[1] != 3 {
        return Err(PyValueError::new_err("average_coords must be (n_atoms, 3)"));
    }
    if pairs.shape()[1] != 2 {
        return Err(PyValueError::new_err("pairs must have shape (n_pairs, 2)"));
    }
    let n_modes = vecs.shape()[0];
    let n_features = vecs.shape()[1];
    let n_atoms = average_coords.shape()[0];
    if n_features != n_atoms * 3 {
        return Err(PyValueError::new_err(
            "eigenvector feature size must match average_coords",
        ));
    }

    let n_pairs = pairs.shape()[0];
    let mut out = Array2::<f32>::zeros((n_pairs, n_modes));
    for pair_idx in 0..n_pairs {
        let a = pairs[[pair_idx, 0]];
        let b = pairs[[pair_idx, 1]];
        if a < 0 || b < 0 {
            continue;
        }
        let a = a as usize;
        let b = b as usize;
        if a >= n_atoms || b >= n_atoms {
            continue;
        }
        let dx = average_coords[[b, 0]] - average_coords[[a, 0]];
        let dy = average_coords[[b, 1]] - average_coords[[a, 1]];
        let dz = average_coords[[b, 2]] - average_coords[[a, 2]];
        let norm = (dx * dx + dy * dy + dz * dz).sqrt();
        if norm == 0.0 {
            continue;
        }
        let ux = dx / norm;
        let uy = dy / norm;
        let uz = dz / norm;
        let a0 = a * 3;
        let b0 = b * 3;
        for mode in 0..n_modes {
            let vx = vecs[[mode, b0]] - vecs[[mode, a0]];
            let vy = vecs[[mode, b0 + 1]] - vecs[[mode, a0 + 1]];
            let vz = vecs[[mode, b0 + 2]] - vecs[[mode, a0 + 2]];
            out[[pair_idx, mode]] = (vx * ux + vy * uy + vz * uz) as f32;
        }
    }
    Ok(out)
}

fn compute_mode_trajout(
    average_coords: numpy::ndarray::ArrayView2<'_, f64>,
    mode_vec: numpy::ndarray::ArrayView1<'_, f64>,
    pcmin: f64,
    pcmax: f64,
    nframes: usize,
    factor: f64,
) -> PyResult<Array3<f64>> {
    if average_coords.shape()[1] != 3 {
        return Err(PyValueError::new_err("average_coords must be (n_atoms, 3)"));
    }
    let n_atoms = average_coords.shape()[0];
    if mode_vec.len() != n_atoms * 3 {
        return Err(PyValueError::new_err(
            "mode vector length must match average_coords",
        ));
    }
    let nframes = nframes.max(2);
    let mut out = Array3::<f64>::zeros((nframes, n_atoms, 3));
    for frame in 0..nframes {
        let amp = if nframes == 1 {
            pcmin
        } else {
            pcmin + (pcmax - pcmin) * frame as f64 / (nframes - 1) as f64
        };
        let scale = factor * amp;
        for atom in 0..n_atoms {
            let base = atom * 3;
            for axis in 0..3 {
                out[[frame, atom, axis]] =
                    average_coords[[atom, axis]] + scale * mode_vec[base + axis];
            }
        }
    }
    Ok(out)
}

#[pyfunction]
fn binned_statistic_2d_array<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    values: PyReadonlyArray1<'_, f64>,
    x_edges: PyReadonlyArray1<'_, f64>,
    y_edges: PyReadonlyArray1<'_, f64>,
    statistic: &str,
) -> PyResult<PyObject> {
    let (out, counts) = compute_binned_statistic_2d(
        x.as_array(),
        y.as_array(),
        values.as_array(),
        x_edges.as_array(),
        y_edges.as_array(),
        statistic,
    )?;
    Ok((
        out.into_pyarray_bound(py).into_py(py),
        counts.into_pyarray_bound(py).into_py(py),
    )
        .into_py(py))
}

#[pyfunction]
fn nearest_fill_grid_array<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'_, f32>,
    tile: bool,
) -> PyResult<&'py PyArray2<f32>> {
    let out = compute_nearest_fill_grid(grid.as_array(), tile);
    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

#[pyfunction]
fn lipid_neighbour_composition_array<'py>(
    py: Python<'py>,
    matrix: PyReadonlyArray3<'_, i64>,
    labels: PyReadonlyArray2<'_, i64>,
    label_values: PyReadonlyArray1<'_, i64>,
) -> PyResult<&'py PyArray3<i32>> {
    let counts = compute_lipid_neighbour_composition(
        matrix.as_array(),
        labels.as_array(),
        label_values.as_array(),
    )?;
    Ok(counts.into_pyarray_bound(py).into_gil_ref())
}

#[pyfunction]
#[pyo3(signature = (coords, idx_a, idx_b, axis, pbc="none", box_lengths=None))]
fn lipid_scd_chunk_array<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray3<'_, f64>,
    idx_a: PyReadonlyArray1<'_, i64>,
    idx_b: PyReadonlyArray1<'_, i64>,
    axis: PyReadonlyArray1<'_, f64>,
    pbc: &str,
    box_lengths: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<PyObject> {
    let box_view = box_lengths.as_ref().map(|arr| arr.as_array());
    let (scd, valid) = compute_lipid_scd_chunk(
        coords.as_array(),
        idx_a.as_array(),
        idx_b.as_array(),
        axis.as_array(),
        pbc,
        box_view,
    )?;
    Ok((
        scd.into_pyarray_bound(py).into_py(py),
        valid.into_pyarray_bound(py).into_py(py),
    )
        .into_py(py))
}

#[pyfunction]
fn haar_details_array<'py>(
    py: Python<'py>,
    series: PyReadonlyArray1<'_, f32>,
) -> PyResult<&'py PyArray2<f32>> {
    let out = compute_haar_details(series.as_array());
    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

#[pyfunction]
fn mode_corr_array<'py>(
    py: Python<'py>,
    vecs: PyReadonlyArray2<'_, f64>,
    average_coords: PyReadonlyArray2<'_, f64>,
    pairs: PyReadonlyArray2<'_, i64>,
) -> PyResult<&'py PyArray2<f32>> {
    let out = compute_mode_corr(vecs.as_array(), average_coords.as_array(), pairs.as_array())?;
    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

#[pyfunction]
fn mode_trajout_array<'py>(
    py: Python<'py>,
    average_coords: PyReadonlyArray2<'_, f64>,
    mode_vec: PyReadonlyArray1<'_, f64>,
    pcmin: f64,
    pcmax: f64,
    nframes: usize,
    factor: f64,
) -> PyResult<&'py PyArray3<f64>> {
    let out = compute_mode_trajout(
        average_coords.as_array(),
        mode_vec.as_array(),
        pcmin,
        pcmax,
        nframes,
        factor,
    )?;
    Ok(out.into_pyarray_bound(py).into_gil_ref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binned_statistic_2d_mean_and_counts() {
        let x = numpy::ndarray::arr1(&[0.0, 0.2, 0.9, 1.0, f64::NAN]);
        let y = numpy::ndarray::arr1(&[0.0, 0.2, 0.9, 1.0, 0.1]);
        let values = numpy::ndarray::arr1(&[1.0, 3.0, 5.0, 7.0, 11.0]);
        let edges = numpy::ndarray::arr1(&[0.0, 0.5, 1.0]);
        let (out, counts) = compute_binned_statistic_2d(
            x.view(),
            y.view(),
            values.view(),
            edges.view(),
            edges.view(),
            "mean",
        )
        .unwrap();
        assert_eq!(counts[[0, 0]], 2);
        assert_eq!(counts[[1, 1]], 2);
        assert!((out[[0, 0]] - 2.0).abs() < 1e-6);
        assert!((out[[1, 1]] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn binned_statistic_2d_median() {
        let x = numpy::ndarray::arr1(&[0.1, 0.2, 0.3, 0.7]);
        let y = numpy::ndarray::arr1(&[0.1, 0.2, 0.3, 0.7]);
        let values = numpy::ndarray::arr1(&[5.0, 1.0, 9.0, 2.0]);
        let edges = numpy::ndarray::arr1(&[0.0, 0.5, 1.0]);
        let (out, counts) = compute_binned_statistic_2d(
            x.view(),
            y.view(),
            values.view(),
            edges.view(),
            edges.view(),
            "median",
        )
        .unwrap();
        assert_eq!(counts[[0, 0]], 3);
        assert!((out[[0, 0]] - 5.0).abs() < 1e-6);
        assert!((out[[1, 1]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn nearest_fill_grid_without_tiling() {
        let grid = numpy::ndarray::arr2(&[[1.0_f32, f32::NAN], [f32::NAN, 4.0]]);
        let out = compute_nearest_fill_grid(grid.view(), false);
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 1.0);
        assert_eq!(out[[1, 0]], 1.0);
        assert_eq!(out[[1, 1]], 4.0);
    }

    #[test]
    fn nearest_fill_grid_with_tiling_wraps_edges() {
        let grid = numpy::ndarray::arr2(&[
            [f32::NAN, f32::NAN, 3.0_f32],
            [f32::NAN, f32::NAN, f32::NAN],
            [1.0_f32, f32::NAN, f32::NAN],
        ]);
        let out = compute_nearest_fill_grid(grid.view(), true);
        assert_eq!(out[[0, 0]], 1.0);
        assert_eq!(out[[0, 1]], 3.0);
        assert_eq!(out[[2, 2]], 1.0);
    }

    #[test]
    fn lipid_neighbour_composition_counts_labels() {
        let matrix = numpy::ndarray::arr3(&[[[0_i64, 1, 1], [1, 0, 0], [1, 0, 0]]]);
        let labels = numpy::ndarray::arr2(&[[0_i64], [0], [1]]);
        let label_values = numpy::ndarray::arr1(&[0_i64, 1]);
        let counts =
            compute_lipid_neighbour_composition(matrix.view(), labels.view(), label_values.view())
                .unwrap();
        assert_eq!(counts.shape(), &[1, 3, 2]);
        assert_eq!(counts[[0, 0, 0]], 1);
        assert_eq!(counts[[0, 0, 1]], 1);
        assert_eq!(counts[[0, 1, 0]], 1);
        assert_eq!(counts[[0, 1, 1]], 0);
    }

    #[test]
    fn lipid_scd_chunk_handles_valid_pairs() {
        let coords = numpy::ndarray::arr3(&[
            [[0.0_f64, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]);
        let idx_a = numpy::ndarray::arr1(&[0_i64]);
        let idx_b = numpy::ndarray::arr1(&[1_i64]);
        let axis = numpy::ndarray::arr1(&[0.0_f64, 0.0, 1.0]);
        let (scd, valid) = compute_lipid_scd_chunk(
            coords.view(),
            idx_a.view(),
            idx_b.view(),
            axis.view(),
            "none",
            None,
        )
        .unwrap();
        assert_eq!(scd.shape(), &[2, 1]);
        assert_eq!(valid[[0, 0]], 1);
        assert_eq!(valid[[1, 0]], 1);
        assert_eq!(scd[[0, 0]], 1.0);
        assert_eq!(scd[[1, 0]], -0.5);
    }

    #[test]
    fn haar_details_match_padded_levels() {
        let series = numpy::ndarray::arr1(&[0.0_f32, 2.0, 4.0, 6.0, 8.0]);
        let out = compute_haar_details(series.view());
        assert_eq!(out.shape(), &[2, 2]);
        assert_eq!(out[[0, 0]], -1.0);
        assert_eq!(out[[0, 1]], -1.0);
        assert_eq!(out[[1, 0]], -2.0);
        assert_eq!(out[[1, 1]], 0.0);
    }

    #[test]
    fn mode_corr_projects_pair_displacements() {
        let vecs = numpy::ndarray::arr2(&[[1.0_f64, 0.0, 0.0, -1.0, 0.0, 0.0]]);
        let avg = numpy::ndarray::arr2(&[[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let pairs = numpy::ndarray::arr2(&[[0_i64, 1]]);
        let out = compute_mode_corr(vecs.view(), avg.view(), pairs.view()).unwrap();
        assert_eq!(out.shape(), &[1, 1]);
        assert_eq!(out[[0, 0]], -2.0);
    }

    #[test]
    fn mode_trajout_spans_pc_range() {
        let avg = numpy::ndarray::arr2(&[[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0]]);
        let mode = numpy::ndarray::arr1(&[1.0_f64, 0.0, 0.0, -1.0, 0.0, 0.0]);
        let out = compute_mode_trajout(avg.view(), mode.view(), -1.0, 1.0, 3, 1.0).unwrap();
        assert_eq!(out.shape(), &[3, 2, 3]);
        assert_eq!(out[[0, 0, 0]], -1.0);
        assert_eq!(out[[1, 0, 0]], 0.0);
        assert_eq!(out[[2, 0, 0]], 1.0);
    }
}

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

#[pyclass]
struct PyLinearDensityPlan {
    plan: RefCell<LinearDensityPlan>,
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

#[pymethods]
impl PyLinearDensityPlan {
    #[new]
    #[pyo3(signature = (selection, axis="z", bin=1.0, range=None, weight="number", norm="count", charges=None, cross_section_area=None, length_scale=None))]
    fn new(
        selection: &PySelection,
        axis: &str,
        bin: f64,
        range: Option<(f64, f64)>,
        weight: &str,
        norm: &str,
        charges: Option<Vec<f64>>,
        cross_section_area: Option<f64>,
        length_scale: Option<f64>,
    ) -> PyResult<Self> {
        let axis = parse_axis(axis)?;
        let weight = match weight.trim().to_ascii_lowercase().as_str() {
            "number" | "count" => LinearDensityWeight::Number,
            "mass" => LinearDensityWeight::Mass,
            "charge" => LinearDensityWeight::Charge,
            _ => {
                return Err(PyValueError::new_err(
                    "lineardensity weight must be one of number, mass, charge",
                ))
            }
        };
        let norm = match norm.trim().to_ascii_lowercase().as_str() {
            "count" | "raw" => LinearDensityNorm::Count,
            "density" | "volume" => LinearDensityNorm::Density,
            _ => {
                return Err(PyValueError::new_err(
                    "lineardensity norm must be one of count, density",
                ))
            }
        };
        let mut plan = LinearDensityPlan::new(selection.selection.clone(), axis, bin, weight, norm)
            .with_range(range.map(|value| [value.0, value.1]))
            .with_cross_section_area(cross_section_area);
        if let Some(charges) = charges {
            plan = plan.with_charges(charges);
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
    #[pyo3(signature = (sel_a, sel_b, bins, r_max, pbc="orthorhombic", center1=false, center2=false, byres1=false, byres2=false, bymol1=false, bymol2=false, no_intramol=false, mass_weighted=true, density=0.033456, volume=false, raw_rdf=false, intrdf=false, dimension="3d", number_density=false))]
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
        number_density: bool,
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
        .with_number_density_output(number_density)
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
    #[pyo3(signature = (sel_a, sel_b, mode="min", pbc="none", unique_pairs=false, cutoff=None, empty_value=None))]
    fn new(
        sel_a: &PySelection,
        sel_b: &PySelection,
        mode: &str,
        pbc: &str,
        unique_pairs: bool,
        cutoff: Option<f32>,
        empty_value: Option<f32>,
    ) -> PyResult<Self> {
        let mode = parse_pair_distance_extrema_mode(mode)?;
        let pbc = parse_pbc(pbc)?;
        if let Some(cutoff) = cutoff {
            if !cutoff.is_finite() || cutoff <= 0.0 {
                return Err(PyValueError::new_err(
                    "pairdist cutoff must be finite and positive",
                ));
            }
        }
        if let Some(empty_value) = empty_value {
            if !empty_value.is_finite() {
                return Err(PyValueError::new_err("pairdist empty value must be finite"));
            }
        }
        let plan = PairDistanceExtremaPlan::new(
            sel_a.selection.clone(),
            sel_b.selection.clone(),
            mode,
            pbc,
        )
        .with_unique_pairs(unique_pairs)
        .with_cutoff(cutoff, empty_value);
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
    m.add_function(wrap_pyfunction!(binned_statistic_2d_array, m)?)?;
    m.add_function(wrap_pyfunction!(nearest_fill_grid_array, m)?)?;
    m.add_function(wrap_pyfunction!(lipid_neighbour_composition_array, m)?)?;
    m.add_function(wrap_pyfunction!(lipid_scd_chunk_array, m)?)?;
    m.add_function(wrap_pyfunction!(haar_details_array, m)?)?;
    m.add_function(wrap_pyfunction!(mode_corr_array, m)?)?;
    m.add_function(wrap_pyfunction!(mode_trajout_array, m)?)?;
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
    m.add_class::<PyLinearDensityPlan>()?;
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
