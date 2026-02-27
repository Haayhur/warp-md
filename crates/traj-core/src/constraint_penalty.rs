use crate::constraints::{ConstraintMode, ConstraintSpec, ShapeSpec};
use crate::geom::Vec3;

const SCALE2: f32 = 1.0e-2;

#[derive(Clone, Copy, Debug, Default)]
pub struct PenaltyResult {
    pub value: f32,
    pub grad: Vec3,
    pub violation: f32,
}

pub fn penalty_and_grad(point: Vec3, constraint: &ConstraintSpec) -> PenaltyResult {
    match constraint.mode {
        ConstraintMode::Inside => penalty_inside(point, &constraint.shape),
        ConstraintMode::Outside => penalty_outside(point, &constraint.shape),
        ConstraintMode::Above | ConstraintMode::Over => penalty_above(point, &constraint.shape),
        ConstraintMode::Below => penalty_below(point, &constraint.shape),
    }
}

fn penalty_inside(point: Vec3, shape: &ShapeSpec) -> PenaltyResult {
    match shape {
        ShapeSpec::Box { min, max } => {
            penalty_inside_box(point, Vec3::from_array(*min), Vec3::from_array(*max))
        }
        ShapeSpec::Cube { center, side } => {
            let c = Vec3::from_array(*center);
            let half = *side * 0.5;
            let min = Vec3::new(c.x - half, c.y - half, c.z - half);
            let max = Vec3::new(c.x + half, c.y + half, c.z + half);
            penalty_inside_box(point, min, max)
        }
        ShapeSpec::Sphere { center, radius } => {
            penalty_inside_sphere(point, Vec3::from_array(*center), *radius)
        }
        ShapeSpec::Ellipsoid { center, radii } => {
            penalty_inside_ellipsoid(point, Vec3::from_array(*center), *radii)
        }
        ShapeSpec::Cylinder {
            base,
            axis,
            radius,
            height,
        } => penalty_inside_cylinder(
            point,
            Vec3::from_array(*base),
            Vec3::from_array(*axis),
            *radius,
            *height,
        ),
        ShapeSpec::Plane { point: p0, normal } => penalty_above(
            point,
            &ShapeSpec::Plane {
                point: *p0,
                normal: *normal,
            },
        ),
        ShapeSpec::XyGauss { .. } => penalty_above(point, shape),
    }
}

fn penalty_outside(point: Vec3, shape: &ShapeSpec) -> PenaltyResult {
    match shape {
        ShapeSpec::Box { min, max } => {
            penalty_outside_box(point, Vec3::from_array(*min), Vec3::from_array(*max))
        }
        ShapeSpec::Cube { center, side } => {
            let c = Vec3::from_array(*center);
            let half = *side * 0.5;
            let min = Vec3::new(c.x - half, c.y - half, c.z - half);
            let max = Vec3::new(c.x + half, c.y + half, c.z + half);
            penalty_outside_box(point, min, max)
        }
        ShapeSpec::Sphere { center, radius } => {
            penalty_outside_sphere(point, Vec3::from_array(*center), *radius)
        }
        ShapeSpec::Ellipsoid { center, radii } => {
            penalty_outside_ellipsoid(point, Vec3::from_array(*center), *radii)
        }
        ShapeSpec::Cylinder {
            base,
            axis,
            radius,
            height,
        } => penalty_outside_cylinder(
            point,
            Vec3::from_array(*base),
            Vec3::from_array(*axis),
            *radius,
            *height,
        ),
        ShapeSpec::Plane { point: p0, normal } => penalty_below(
            point,
            &ShapeSpec::Plane {
                point: *p0,
                normal: *normal,
            },
        ),
        ShapeSpec::XyGauss { .. } => penalty_below(point, shape),
    }
}

fn penalty_above(point: Vec3, shape: &ShapeSpec) -> PenaltyResult {
    match shape {
        ShapeSpec::Plane { point: p0, normal } => {
            let n = Vec3::from_array(*normal);
            let w = point.sub(Vec3::from_array(*p0)).dot(n);
            if w >= 0.0 {
                PenaltyResult::default()
            } else {
                let value = w * w;
                let grad = n.scale(2.0 * w);
                PenaltyResult {
                    value,
                    grad,
                    violation: -w,
                }
            }
        }
        ShapeSpec::XyGauss {
            center,
            sigma,
            z0,
            amplitude,
        } => {
            let dx = point.x - center[0];
            let dy = point.y - center[1];
            let sx = sigma[0];
            let sy = sigma[1];
            let expo = -(dx * dx) / (2.0 * sx * sx) - (dy * dy) / (2.0 * sy * sy);
            let surf = if expo <= -50.0 {
                *z0
            } else {
                z0 + amplitude * expo.exp()
            };
            let w = surf - point.z;
            if w <= 0.0 {
                PenaltyResult::default()
            } else {
                let value = w * w;
                let grad = Vec3::new(0.0, 0.0, -2.0 * w);
                PenaltyResult {
                    value,
                    grad,
                    violation: w,
                }
            }
        }
        _ => PenaltyResult::default(),
    }
}

fn penalty_below(point: Vec3, shape: &ShapeSpec) -> PenaltyResult {
    match shape {
        ShapeSpec::Plane { point: p0, normal } => {
            let n = Vec3::from_array(*normal);
            let w = point.sub(Vec3::from_array(*p0)).dot(n);
            if w <= 0.0 {
                PenaltyResult::default()
            } else {
                let value = w * w;
                let grad = n.scale(2.0 * w);
                PenaltyResult {
                    value,
                    grad,
                    violation: w,
                }
            }
        }
        ShapeSpec::XyGauss {
            center,
            sigma,
            z0,
            amplitude,
        } => {
            let dx = point.x - center[0];
            let dy = point.y - center[1];
            let sx = sigma[0];
            let sy = sigma[1];
            let expo = -(dx * dx) / (2.0 * sx * sx) - (dy * dy) / (2.0 * sy * sy);
            let surf = if expo <= -50.0 {
                *z0
            } else {
                z0 + amplitude * expo.exp()
            };
            let w = surf - point.z;
            if w >= 0.0 {
                PenaltyResult::default()
            } else {
                let value = w * w;
                let grad = Vec3::new(0.0, 0.0, -2.0 * w);
                PenaltyResult {
                    value,
                    grad,
                    violation: -w,
                }
            }
        }
        _ => PenaltyResult::default(),
    }
}

fn penalty_inside_box(point: Vec3, min: Vec3, max: Vec3) -> PenaltyResult {
    let mut value = 0.0;
    let mut grad = Vec3::default();
    let mut violation: f32 = 0.0;
    let dx_low = point.x - min.x;
    if dx_low < 0.0 {
        value += dx_low * dx_low;
        grad.x += 2.0 * dx_low;
        violation = violation.max(-dx_low);
    }
    let dx_high = point.x - max.x;
    if dx_high > 0.0 {
        value += dx_high * dx_high;
        grad.x += 2.0 * dx_high;
        violation = violation.max(dx_high);
    }
    let dy_low = point.y - min.y;
    if dy_low < 0.0 {
        value += dy_low * dy_low;
        grad.y += 2.0 * dy_low;
        violation = violation.max(-dy_low);
    }
    let dy_high = point.y - max.y;
    if dy_high > 0.0 {
        value += dy_high * dy_high;
        grad.y += 2.0 * dy_high;
        violation = violation.max(dy_high);
    }
    let dz_low = point.z - min.z;
    if dz_low < 0.0 {
        value += dz_low * dz_low;
        grad.z += 2.0 * dz_low;
        violation = violation.max(-dz_low);
    }
    let dz_high = point.z - max.z;
    if dz_high > 0.0 {
        value += dz_high * dz_high;
        grad.z += 2.0 * dz_high;
        violation = violation.max(dz_high);
    }
    PenaltyResult {
        value,
        grad,
        violation,
    }
}

fn penalty_outside_box(point: Vec3, min: Vec3, max: Vec3) -> PenaltyResult {
    let inside = point.x > min.x
        && point.x < max.x
        && point.y > min.y
        && point.y < max.y
        && point.z > min.z
        && point.z < max.z;
    if !inside {
        return PenaltyResult::default();
    }
    let mid = Vec3::new(
        (min.x + max.x) * 0.5,
        (min.y + max.y) * 0.5,
        (min.z + max.z) * 0.5,
    );
    let mut value = 0.0;
    let mut grad = Vec3::default();
    let mut violation: f32 = 0.0;
    let dx = if point.x <= mid.x {
        point.x - min.x
    } else {
        max.x - point.x
    };
    let sign_x = if point.x <= mid.x { 1.0 } else { -1.0 };
    value += dx;
    grad.x += sign_x;
    violation = violation.max(dx);
    let dy = if point.y <= mid.y {
        point.y - min.y
    } else {
        max.y - point.y
    };
    let sign_y = if point.y <= mid.y { 1.0 } else { -1.0 };
    value += dy;
    grad.y += sign_y;
    violation = violation.max(dy);
    let dz = if point.z <= mid.z {
        point.z - min.z
    } else {
        max.z - point.z
    };
    let sign_z = if point.z <= mid.z { 1.0 } else { -1.0 };
    value += dz;
    grad.z += sign_z;
    violation = violation.max(dz);
    PenaltyResult {
        value,
        grad,
        violation,
    }
}

fn penalty_inside_sphere(point: Vec3, center: Vec3, radius: f32) -> PenaltyResult {
    let d = point.sub(center);
    let w = d.dot(d) - radius * radius;
    if w <= 0.0 {
        return PenaltyResult::default();
    }
    let value = SCALE2 * w * w;
    let grad = d.scale(4.0 * SCALE2 * w);
    PenaltyResult {
        value,
        grad,
        violation: w.sqrt(),
    }
}

fn penalty_outside_sphere(point: Vec3, center: Vec3, radius: f32) -> PenaltyResult {
    let d = point.sub(center);
    let dist2 = d.dot(d);
    let w = dist2 - radius * radius;
    if w >= 0.0 {
        return PenaltyResult::default();
    }
    let value = SCALE2 * w * w;
    let grad = d.scale(4.0 * SCALE2 * w);
    PenaltyResult {
        value,
        grad,
        violation: (-w).sqrt(),
    }
}

fn penalty_inside_ellipsoid(point: Vec3, center: Vec3, radii: [f32; 3]) -> PenaltyResult {
    let dx = (point.x - center.x) / radii[0];
    let dy = (point.y - center.y) / radii[1];
    let dz = (point.z - center.z) / radii[2];
    let w = dx * dx + dy * dy + dz * dz - 1.0;
    if w <= 0.0 {
        return PenaltyResult::default();
    }
    let value = SCALE2 * w * w;
    let grad = Vec3::new(
        4.0 * SCALE2 * w * dx / radii[0],
        4.0 * SCALE2 * w * dy / radii[1],
        4.0 * SCALE2 * w * dz / radii[2],
    );
    PenaltyResult {
        value,
        grad,
        violation: w,
    }
}

fn penalty_outside_ellipsoid(point: Vec3, center: Vec3, radii: [f32; 3]) -> PenaltyResult {
    let dx = (point.x - center.x) / radii[0];
    let dy = (point.y - center.y) / radii[1];
    let dz = (point.z - center.z) / radii[2];
    let w = 1.0 - (dx * dx + dy * dy + dz * dz);
    if w <= 0.0 {
        return PenaltyResult::default();
    }
    let value = SCALE2 * w * w;
    let grad = Vec3::new(
        -4.0 * SCALE2 * w * dx / radii[0],
        -4.0 * SCALE2 * w * dy / radii[1],
        -4.0 * SCALE2 * w * dz / radii[2],
    );
    PenaltyResult {
        value,
        grad,
        violation: w,
    }
}

fn penalty_inside_cylinder(
    point: Vec3,
    base: Vec3,
    axis: Vec3,
    radius: f32,
    height: f32,
) -> PenaltyResult {
    let axis_len = axis.norm();
    if axis_len <= 1.0e-6 {
        return PenaltyResult::default();
    }
    let axis_unit = axis.scale(1.0 / axis_len);
    let v = point.sub(base);
    let proj = v.dot(axis_unit);
    let mut value = 0.0;
    let mut grad = Vec3::default();
    let mut violation: f32 = 0.0;
    if proj < 0.0 {
        let w = proj;
        value += SCALE2 * w * w;
        grad = grad.add(axis_unit.scale(2.0 * SCALE2 * w));
        violation = violation.max(-w);
    } else if proj > height {
        let w = proj - height;
        value += SCALE2 * w * w;
        grad = grad.add(axis_unit.scale(2.0 * SCALE2 * w));
        violation = violation.max(w);
    }
    let radial = v.sub(axis_unit.scale(proj));
    let dist = radial.norm();
    if dist > radius {
        let w = dist - radius;
        value += SCALE2 * w * w;
        if dist > 1.0e-6 {
            grad = grad.add(radial.scale(2.0 * SCALE2 * w / dist));
        }
        violation = violation.max(w);
    }
    PenaltyResult {
        value,
        grad,
        violation,
    }
}

fn penalty_outside_cylinder(
    point: Vec3,
    base: Vec3,
    axis: Vec3,
    radius: f32,
    height: f32,
) -> PenaltyResult {
    let axis_len = axis.norm();
    if axis_len <= 1.0e-6 {
        return PenaltyResult::default();
    }
    let axis_unit = axis.scale(1.0 / axis_len);
    let v = point.sub(base);
    let proj = v.dot(axis_unit);
    if proj < 0.0 || proj > height {
        return PenaltyResult::default();
    }
    let radial = v.sub(axis_unit.scale(proj));
    let dist = radial.norm();
    if dist >= radius {
        return PenaltyResult::default();
    }
    let w = radius - dist;
    let value = SCALE2 * w * w;
    let grad = if dist > 1.0e-6 {
        radial.scale(-2.0 * SCALE2 * w / dist)
    } else {
        Vec3::default()
    };
    PenaltyResult {
        value,
        grad,
        violation: w,
    }
}
