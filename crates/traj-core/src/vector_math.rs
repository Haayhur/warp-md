use crate::geometry::Vec3;

pub fn normalize_vec3(v: Vec3) -> Vec3 {
    let norm = v.norm();
    if norm <= 1.0e-12 {
        return Vec3::new(1.0, 0.0, 0.0);
    }
    v.scale(1.0 / norm)
}

pub fn rotate_from_to_vec3(v: Vec3, source_axis: Vec3, target_axis: Vec3) -> Vec3 {
    let source = normalize_vec3(source_axis);
    let target = normalize_vec3(target_axis);
    let dot = source.dot(target).clamp(-1.0, 1.0);
    if dot > 1.0 - 1.0e-6 {
        return v;
    }
    if dot < -1.0 + 1.0e-6 {
        let basis = if source.x.abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let axis = normalize_vec3(source.cross(basis));
        return axis.scale(2.0 * axis.dot(v)).sub(v);
    }
    let axis = normalize_vec3(source.cross(target));
    let theta = dot.acos();
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    v.scale(cos_t)
        .add(axis.cross(v).scale(sin_t))
        .add(axis.scale(axis.dot(v) * (1.0 - cos_t)))
}

pub fn rotate_about_axis_vec3(v: Vec3, axis: Vec3, theta: f32) -> Vec3 {
    if theta.abs() <= 1.0e-8 {
        return v;
    }
    let axis = normalize_vec3(axis);
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    v.scale(cos_t)
        .add(axis.cross(v).scale(sin_t))
        .add(axis.scale(axis.dot(v) * (1.0 - cos_t)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_fallback_axis() {
        let axis = normalize_vec3(Vec3::new(0.0, 0.0, 0.0));
        assert_eq!(axis.x, 1.0);
        assert_eq!(axis.y, 0.0);
        assert_eq!(axis.z, 0.0);
    }

    #[test]
    fn rotate_about_axis_preserves_distance() {
        let point = Vec3::new(1.0, 0.0, 0.0);
        let rotated =
            rotate_about_axis_vec3(point, Vec3::new(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
        assert!(rotated.x.abs() < 1.0e-5);
        assert!((rotated.y - 1.0).abs() < 1.0e-5);
    }
}
