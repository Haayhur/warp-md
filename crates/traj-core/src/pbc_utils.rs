use crate::error::{TrajError, TrajResult};
use crate::frame::Box3;

pub fn box_lengths(box_: Box3) -> TrajResult<(f64, f64, f64)> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => Ok((lx as f64, ly as f64, lz as f64)),
        _ => Err(TrajError::Mismatch(
            "orthorhombic box required for PBC".into(),
        )),
    }
}

pub fn apply_pbc(dx: &mut f64, dy: &mut f64, dz: &mut f64, lx: f64, ly: f64, lz: f64) {
    if lx > 0.0 {
        *dx -= (*dx / lx).round() * lx;
    }
    if ly > 0.0 {
        *dy -= (*dy / ly).round() * ly;
    }
    if lz > 0.0 {
        *dz -= (*dz / lz).round() * lz;
    }
}

pub fn apply_pbc_triclinic(
    dx: &mut f64,
    dy: &mut f64,
    dz: &mut f64,
    cell: &[[f64; 3]; 3],
    inv: &[[f64; 3]; 3],
) {
    let fx = inv[0][0] * *dx + inv[1][0] * *dy + inv[2][0] * *dz;
    let fy = inv[0][1] * *dx + inv[1][1] * *dy + inv[2][1] * *dz;
    let fz = inv[0][2] * *dx + inv[1][2] * *dy + inv[2][2] * *dz;
    let fx = fx - fx.round();
    let fy = fy - fy.round();
    let fz = fz - fz.round();
    *dx = fx * cell[0][0] + fy * cell[1][0] + fz * cell[2][0];
    *dy = fx * cell[0][1] + fy * cell[1][1] + fz * cell[2][1];
    *dz = fx * cell[0][2] + fy * cell[1][2] + fz * cell[2][2];
}

pub fn cell_and_inv_from_box(box_: Box3) -> TrajResult<([[f64; 3]; 3], [[f64; 3]; 3])> {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => {
            if lx == 0.0 || ly == 0.0 || lz == 0.0 {
                return Err(TrajError::Mismatch(
                    "image requires nonzero box lengths".into(),
                ));
            }
            let cell = [
                [lx as f64, 0.0, 0.0],
                [0.0, ly as f64, 0.0],
                [0.0, 0.0, lz as f64],
            ];
            let inv = [
                [1.0 / lx as f64, 0.0, 0.0],
                [0.0, 1.0 / ly as f64, 0.0],
                [0.0, 0.0, 1.0 / lz as f64],
            ];
            Ok((cell, inv))
        }
        Box3::Triclinic { m } => {
            let m0 = m[0] as f64;
            let m1 = m[1] as f64;
            let m2 = m[2] as f64;
            let m3 = m[3] as f64;
            let m4 = m[4] as f64;
            let m5 = m[5] as f64;
            let m6 = m[6] as f64;
            let m7 = m[7] as f64;
            let m8 = m[8] as f64;
            let det =
                m0 * (m4 * m8 - m5 * m7) - m1 * (m3 * m8 - m5 * m6) + m2 * (m3 * m7 - m4 * m6);
            if det == 0.0 {
                return Err(TrajError::Mismatch("box matrix not invertible".into()));
            }
            let inv00 = (m4 * m8 - m5 * m7) / det;
            let inv01 = (m2 * m7 - m1 * m8) / det;
            let inv02 = (m1 * m5 - m2 * m4) / det;
            let inv10 = (m5 * m6 - m3 * m8) / det;
            let inv11 = (m0 * m8 - m2 * m6) / det;
            let inv12 = (m2 * m3 - m0 * m5) / det;
            let inv20 = (m3 * m7 - m4 * m6) / det;
            let inv21 = (m1 * m6 - m0 * m7) / det;
            let inv22 = (m0 * m4 - m1 * m3) / det;
            let cell = [[m0, m1, m2], [m3, m4, m5], [m6, m7, m8]];
            let inv = [
                [inv00, inv01, inv02],
                [inv10, inv11, inv12],
                [inv20, inv21, inv22],
            ];
            Ok((cell, inv))
        }
        Box3::None => Err(TrajError::Mismatch("box vectors required".into())),
    }
}
