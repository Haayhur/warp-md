use crate::geom::Vec3;

pub(crate) fn rotation_matrix(beta: f32, gamma: f32, teta: f32) -> [[f32; 3]; 3] {
    let cb = beta.cos();
    let sb = beta.sin();
    let cg = gamma.cos();
    let sg = gamma.sin();
    let ct = teta.cos();
    let st = teta.sin();
    let v1 = [-sb * sg * ct + cb * cg, -sb * cg * ct - cb * sg, sb * st];
    let v2 = [cb * sg * ct + sb * cg, cb * cg * ct - sb * sg, -cb * st];
    let v3 = [sg * st, cg * st, ct];
    [
        [v1[0], v2[0], v3[0]],
        [v1[1], v2[1], v3[1]],
        [v1[2], v2[2], v3[2]],
    ]
}

pub(crate) fn rotation_with_derivatives(
    beta: f32,
    gamma: f32,
    teta: f32,
) -> ([[f32; 3]; 3], [[f32; 3]; 3], [[f32; 3]; 3], [[f32; 3]; 3]) {
    let cb = beta.cos();
    let sb = beta.sin();
    let cg = gamma.cos();
    let sg = gamma.sin();
    let ct = teta.cos();
    let st = teta.sin();

    let v1 = [-sb * sg * ct + cb * cg, -sb * cg * ct - cb * sg, sb * st];
    let v2 = [cb * sg * ct + sb * cg, cb * cg * ct - sb * sg, -cb * st];
    let v3 = [sg * st, cg * st, ct];

    let dv1_db = [-cb * sg * ct - sb * cg, -cb * cg * ct + sb * sg, cb * st];
    let dv2_db = [-sb * sg * ct + cb * cg, -sb * cg * ct - cb * sg, sb * st];
    let dv3_db = [0.0, 0.0, 0.0];

    let dv1_dg = [-sb * cg * ct - cb * sg, sb * sg * ct - cb * cg, 0.0];
    let dv2_dg = [cb * cg * ct - sb * sg, -cb * sg * ct - sb * cg, 0.0];
    let dv3_dg = [cg * st, -sg * st, 0.0];

    let dv1_dt = [sb * sg * st, sb * cg * st, sb * ct];
    let dv2_dt = [-cb * sg * st, -cb * cg * st, -cb * ct];
    let dv3_dt = [sg * ct, cg * ct, -st];

    let mat = [
        [v1[0], v2[0], v3[0]],
        [v1[1], v2[1], v3[1]],
        [v1[2], v2[2], v3[2]],
    ];
    let db = [
        [dv1_db[0], dv2_db[0], dv3_db[0]],
        [dv1_db[1], dv2_db[1], dv3_db[1]],
        [dv1_db[2], dv2_db[2], dv3_db[2]],
    ];
    let dg = [
        [dv1_dg[0], dv2_dg[0], dv3_dg[0]],
        [dv1_dg[1], dv2_dg[1], dv3_dg[1]],
        [dv1_dg[2], dv2_dg[2], dv3_dg[2]],
    ];
    let dt = [
        [dv1_dt[0], dv2_dt[0], dv3_dt[0]],
        [dv1_dt[1], dv2_dt[1], dv3_dt[1]],
        [dv1_dt[2], dv2_dt[2], dv3_dt[2]],
    ];
    (mat, db, dg, dt)
}

pub(crate) fn mat_vec(m: [[f32; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}
