#[cfg(feature = "cuda")]
fn rmsd_from_cov(cov: &[f32; 9], sum_x2: f64, sum_y2: f64, n_sel: usize) -> f32 {
    if n_sel == 0 {
        return 0.0;
    }
    let cov_f64 = [
        cov[0] as f64,
        cov[1] as f64,
        cov[2] as f64,
        cov[3] as f64,
        cov[4] as f64,
        cov[5] as f64,
        cov[6] as f64,
        cov[7] as f64,
        cov[8] as f64,
    ];
    let m = Matrix3::from_row_slice(&cov_f64);
    let svd = m.svd(true, true);
    let mut sigma_sum = svd.singular_values[0] + svd.singular_values[1] + svd.singular_values[2];
    if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
        let det = (v_t.transpose() * u.transpose()).determinant();
        if det < 0.0 {
            sigma_sum -= 2.0 * svd.singular_values[2];
        }
    }
    let n = n_sel as f64;
    let rmsd2 = (sum_x2 + sum_y2 - 2.0 * sigma_sum) / n;
    if rmsd2 <= 0.0 {
        0.0
    } else {
        rmsd2.sqrt() as f32
    }
}

fn centroid(points: &[Vector3<f64>]) -> Vector3<f64> {
    let mut c = Vector3::new(0.0, 0.0, 0.0);
    for p in points {
        c += p;
    }
    c / (points.len() as f64)
}
