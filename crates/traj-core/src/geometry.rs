use rand::Rng;

#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn from_array(v: [f32; 3]) -> Self {
        Self::new(v[0], v[1], v[2])
    }

    pub fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn dot(self, other: Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn norm(self) -> f32 {
        self.dot(self).sqrt()
    }

    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Quaternion {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Quaternion {
    pub fn identity() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            w: 1.0,
        }
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        let u3: f32 = rng.gen();
        let s1 = (1.0 - u1).sqrt();
        let s2 = u1.sqrt();
        let t1 = 2.0 * std::f32::consts::PI * u2;
        let t2 = 2.0 * std::f32::consts::PI * u3;
        Self {
            x: s1 * t1.sin(),
            y: s1 * t1.cos(),
            z: s2 * t2.sin(),
            w: s2 * t2.cos(),
        }
    }

    pub fn rotate_vec(self, v: Vec3) -> Vec3 {
        let u = Vec3::new(self.x, self.y, self.z);
        let s = self.w;
        let term1 = u.scale(2.0 * u.dot(v));
        let term2 = v.scale(s * s - u.dot(u));
        let term3 = u.cross(v).scale(2.0 * s);
        term1.add(term2).add(term3)
    }

    pub fn from_packmol_euler(beta: f32, gamma: f32, teta: f32) -> Self {
        let m = matrix_from_packmol_euler(beta, gamma, teta);
        Self::from_matrix(m)
    }

    pub fn to_packmol_euler(self) -> (f32, f32, f32) {
        let m = self.to_matrix();
        euler_from_packmol_matrix(m)
    }

    fn to_matrix(self) -> [[f32; 3]; 3] {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let w = self.w;
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ]
    }

    fn from_matrix(m: [[f32; 3]; 3]) -> Self {
        let trace = m[0][0] + m[1][1] + m[2][2];
        let (w, x, y, z) = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            (
                0.25 * s,
                (m[2][1] - m[1][2]) / s,
                (m[0][2] - m[2][0]) / s,
                (m[1][0] - m[0][1]) / s,
            )
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt() * 2.0;
            (
                (m[2][1] - m[1][2]) / s,
                0.25 * s,
                (m[0][1] + m[1][0]) / s,
                (m[0][2] + m[2][0]) / s,
            )
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt() * 2.0;
            (
                (m[0][2] - m[2][0]) / s,
                (m[0][1] + m[1][0]) / s,
                0.25 * s,
                (m[1][2] + m[2][1]) / s,
            )
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt() * 2.0;
            (
                (m[1][0] - m[0][1]) / s,
                (m[0][2] + m[2][0]) / s,
                (m[1][2] + m[2][1]) / s,
                0.25 * s,
            )
        };
        let norm = (w * w + x * x + y * y + z * z).sqrt();
        if norm > 0.0 {
            Self {
                x: x / norm,
                y: y / norm,
                z: z / norm,
                w: w / norm,
            }
        } else {
            Self::identity()
        }
    }
}

pub fn center_of_geometry(points: &[Vec3]) -> Vec3 {
    if points.is_empty() {
        return Vec3::default();
    }
    let mut sum = Vec3::default();
    for p in points {
        sum = sum.add(*p);
    }
    sum.scale(1.0 / points.len() as f32)
}

fn matrix_from_packmol_euler(beta: f32, gamma: f32, teta: f32) -> [[f32; 3]; 3] {
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

fn euler_from_packmol_matrix(m: [[f32; 3]; 3]) -> (f32, f32, f32) {
    let ct = m[2][2].clamp(-1.0, 1.0);
    let teta = ct.acos();
    let st = (1.0 - ct * ct).sqrt();
    if st > 1.0e-6 {
        let gamma = m[0][2].atan2(m[1][2]);
        let beta = m[2][0].atan2(-m[2][1]);
        (beta, gamma, teta)
    } else {
        let gamma = 0.0;
        let beta = (-m[1][0]).atan2(m[0][0]);
        (beta, gamma, teta)
    }
}
