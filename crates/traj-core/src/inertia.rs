use nalgebra::Matrix3;

pub fn principal_axes_from_inertia(
    i_xx: f64,
    i_yy: f64,
    i_zz: f64,
    i_xy: f64,
    i_xz: f64,
    i_yz: f64,
) -> (Matrix3<f64>, [f32; 3]) {
    let inertia = Matrix3::new(i_xx, i_xy, i_xz, i_xy, i_yy, i_yz, i_xz, i_yz, i_zz);
    let eigen = inertia.symmetric_eigen();
    let mut order = [0usize, 1, 2];
    order.sort_by(|&a, &b| {
        eigen.eigenvalues[a]
            .partial_cmp(&eigen.eigenvalues[b])
            .unwrap()
    });
    let mut axes = Matrix3::zeros();
    let mut vals = [0.0f32; 3];
    for (col_out, &col_in) in order.iter().enumerate() {
        axes.set_column(col_out, &eigen.eigenvectors.column(col_in));
        vals[col_out] = eigen.eigenvalues[col_in] as f32;
    }
    (axes, vals)
}
