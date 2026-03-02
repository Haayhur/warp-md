//! 3D coordinate calculation from internal coordinates (bond length, angle, dihedral).

use std::f64::consts::PI;

/// A 3D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn scale(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Self {
        let l = self.length();
        if l < 1e-15 {
            return Self::zero();
        }
        self.scale(1.0 / l)
    }
}

fn deg2rad(d: f64) -> f64 {
    d * PI / 180.0
}

fn rad2deg(r: f64) -> f64 {
    r * 180.0 / PI
}

/// Compute dihedral angle (degrees) for four points A-B-C-D.
pub fn calc_dihedral(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    let b1 = b.sub(a);
    let b2 = c.sub(b);
    let b3 = d.sub(c);
    let n1 = b1.cross(b2);
    let n2 = b2.cross(b3);
    let m1 = n1.cross(b2.normalized());
    let x = n1.dot(n2);
    let y = m1.dot(n2);
    rad2deg((-y).atan2(-x) + PI)
}

/// Rotate vector `v` around axis `axis` (must be unit) by `angle_rad`.
fn rotate_around_axis(v: Vec3, axis: Vec3, angle_rad: f64) -> Vec3 {
    let cos_a = angle_rad.cos();
    let sin_a = angle_rad.sin();
    let dot = axis.dot(v);
    // Rodrigues' rotation formula
    let term1 = v.scale(cos_a);
    let term2 = axis.cross(v).scale(sin_a);
    let term3 = axis.scale(dot * (1.0 - cos_a));
    term1.add(term2).add(term3)
}

/// Place a new atom at distance `length` from `ref_c`, with bond angle
/// `angle_deg` (A-B-C→D angle at C), and dihedral angle `dihedral_deg`
/// (A-B-C-D dihedral). Reference atoms are `ref_a`, `ref_b`, `ref_c`.
///
/// This is the Rust port of PeptideBuilder's `calculateCoordinates`.
pub fn calculate_coordinates(
    ref_a: Vec3,
    ref_b: Vec3,
    ref_c: Vec3,
    length: f64,
    angle_deg: f64,
    dihedral_deg: f64,
) -> Vec3 {
    let av = ref_a.sub(ref_c);
    let bv = ref_b.sub(ref_c);

    let ax = av.x;
    let ay = av.y;
    let az = av.z;
    let bx = bv.x;
    let by = bv.y;
    let bz = bv.z;

    // Plane normal
    let a = ay * bz - az * by;
    let b = az * bx - ax * bz;
    let g = ax * by - ay * bx;

    let ang_rad = deg2rad(angle_deg);
    let f_val = (bx * bx + by * by + bz * bz).sqrt() * length * ang_rad.cos();

    let denom = b * b * (bx * bx + bz * bz) + a * a * (by * by + bz * bz) - 2.0 * a * bx * bz * g
        + (bx * bx + by * by) * g * g
        - 2.0 * b * by * (a * bx + bz * g);

    let inner = (b * bz - by * g).powi(2)
        * (-(f_val * f_val) * (a * a + b * b + g * g)
            + (b * b * (bx * bx + bz * bz) + a * a * (by * by + bz * bz) - 2.0 * a * bx * bz * g
                + (bx * bx + by * by) * g * g
                - 2.0 * b * by * (a * bx + bz * g))
                * length
                * length);

    let cst = inner.abs().sqrt();

    let x_val =
        (b * b * bx * f_val - a * b * by * f_val + f_val * g * (-a * bz + bx * g) + cst) / denom;

    let (y_val, z_val);
    if (b == 0.0 || bz == 0.0) && (by == 0.0 || g == 0.0) {
        let c1_inner = g
            * g
            * (-a * a * x_val * x_val + (b * b + g * g) * (length - x_val) * (length + x_val));
        let c1 = c1_inner.abs().sqrt();
        y_val = (-a * b * x_val + c1) / (b * b + g * g);
        z_val = -(a * g * g * x_val + b * c1) / (g * (b * b + g * g));
    } else {
        let d_factor = b * bz - by * g;
        y_val = (a * a * by * f_val * d_factor + g * (-f_val * d_factor * d_factor + bx * cst)
            - a * (b * b * bx * bz * f_val - b * bx * by * f_val * g + bz * cst))
            / (d_factor * denom);
        z_val = (a * a * bz * f_val * d_factor
            + b * f_val * d_factor * d_factor
            + a * bx * f_val * g * (-b * bz + by * g)
            - b * bx * cst
            + a * by * cst)
            / (d_factor * denom);
    }

    let d_raw = Vec3::new(x_val, y_val, z_val).add(ref_c);

    // Compute current dihedral and correct it
    let temp = calc_dihedral(ref_a, ref_b, ref_c, d_raw);
    let correction = dihedral_deg - temp;
    let rot_axis = ref_c.sub(ref_b).normalized();
    let d_shifted = d_raw.sub(ref_b);
    let d_rotated = rotate_around_axis(d_shifted, rot_axis, deg2rad(correction));
    d_rotated.add(ref_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec3_basics() {
        let a = Vec3::new(1.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 1.0, 0.0);
        let c = a.cross(b);
        assert!((c.z - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_calculate_coordinates_alanine_first() {
        // Place CA at origin, C along x, N in xy-plane
        let n_ca_c_angle: f64 = 111.068;
        let ca_n_length: f64 = 1.46;
        let ca_c_length: f64 = 1.52;

        let ca = Vec3::zero();
        let c = Vec3::new(ca_c_length, 0.0, 0.0);
        let n = Vec3::new(
            ca_n_length * (n_ca_c_angle * PI / 180.0).cos(),
            ca_n_length * (n_ca_c_angle * PI / 180.0).sin(),
            0.0,
        );

        // Place O relative to N, CA, C
        let o = calculate_coordinates(n, ca, c, 1.23, 120.5, -60.5);
        // Sanity: O should be ~1.23 Å from C
        let dist = o.sub(c).length();
        assert!((dist - 1.23).abs() < 0.02, "O-C distance {dist} not ~1.23");
    }
}
