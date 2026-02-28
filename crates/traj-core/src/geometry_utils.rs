pub fn angle_from_vectors(v1: [f64; 3], v2: [f64; 3], degrees: bool) -> f32 {
    let dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    let n1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let n2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    if n1 == 0.0 || n2 == 0.0 {
        return 0.0;
    }
    let mut cos = dot / (n1 * n2);
    if cos > 1.0 {
        cos = 1.0;
    } else if cos < -1.0 {
        cos = -1.0;
    }
    let mut angle = cos.acos();
    if degrees {
        angle = angle.to_degrees();
    }
    angle as f32
}

pub fn dihedral_from_vectors(
    b0: [f64; 3],
    b1: [f64; 3],
    b2: [f64; 3],
    degrees: bool,
    range360: bool,
) -> f32 {
    let b1_norm = {
        let n = (b1[0] * b1[0] + b1[1] * b1[1] + b1[2] * b1[2]).sqrt();
        if n == 0.0 {
            return 0.0;
        }
        [b1[0] / n, b1[1] / n, b1[2] / n]
    };
    let dot_b0 = b0[0] * b1_norm[0] + b0[1] * b1_norm[1] + b0[2] * b1_norm[2];
    let dot_b2 = b2[0] * b1_norm[0] + b2[1] * b1_norm[1] + b2[2] * b1_norm[2];
    let v = [
        b0[0] - dot_b0 * b1_norm[0],
        b0[1] - dot_b0 * b1_norm[1],
        b0[2] - dot_b0 * b1_norm[2],
    ];
    let w = [
        b2[0] - dot_b2 * b1_norm[0],
        b2[1] - dot_b2 * b1_norm[1],
        b2[2] - dot_b2 * b1_norm[2],
    ];
    let x = v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
    let m = [
        b1_norm[1] * v[2] - b1_norm[2] * v[1],
        b1_norm[2] * v[0] - b1_norm[0] * v[2],
        b1_norm[0] * v[1] - b1_norm[1] * v[0],
    ];
    let y = m[0] * w[0] + m[1] * w[1] + m[2] * w[2];
    let mut angle = y.atan2(x);
    if degrees {
        angle = angle.to_degrees();
        if range360 && angle < 0.0 {
            angle += 360.0;
        }
    } else if range360 && angle < 0.0 {
        angle += std::f64::consts::TAU;
    }
    angle as f32
}

pub fn angle_diff(current: f64, reference: f64, degrees: bool) -> f64 {
    let period = if degrees {
        360.0
    } else {
        std::f64::consts::TAU
    };
    let half = 0.5 * period;
    let mut diff = current - reference;
    diff = (diff + half).rem_euclid(period) - half;
    diff
}

pub fn rotate_about_axis(
    point: [f64; 3],
    origin: [f64; 3],
    axis: [f64; 3],
    angle: f64,
) -> [f64; 3] {
    let ax = axis[0];
    let ay = axis[1];
    let az = axis[2];
    let norm = (ax * ax + ay * ay + az * az).sqrt();
    if norm == 0.0 {
        return point;
    }
    let ux = ax / norm;
    let uy = ay / norm;
    let uz = az / norm;
    let vx = point[0] - origin[0];
    let vy = point[1] - origin[1];
    let vz = point[2] - origin[2];
    let cos = angle.cos();
    let sin = angle.sin();
    let dot = ux * vx + uy * vy + uz * vz;
    let cx = uy * vz - uz * vy;
    let cy = uz * vx - ux * vz;
    let cz = ux * vy - uy * vx;
    let rx = vx * cos + cx * sin + ux * dot * (1.0 - cos);
    let ry = vy * cos + cy * sin + uy * dot * (1.0 - cos);
    let rz = vz * cos + cz * sin + uz * dot * (1.0 - cos);
    [origin[0] + rx, origin[1] + ry, origin[2] + rz]
}
