use serde::{Deserialize, Serialize};

use std::borrow::Cow;

use crate::error::{TrajError, TrajResult};
use crate::geom::Vec3;
use crate::pbc::PbcBox;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ConstraintMode {
    Inside,
    Outside,
    Above,
    Below,
    Over,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstraintSpec {
    pub mode: ConstraintMode,
    #[serde(flatten)]
    pub shape: ShapeSpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "lowercase")]
pub enum ShapeSpec {
    Box {
        min: [f32; 3],
        max: [f32; 3],
    },
    Cube {
        center: [f32; 3],
        side: f32,
    },
    Sphere {
        center: [f32; 3],
        radius: f32,
    },
    Ellipsoid {
        center: [f32; 3],
        radii: [f32; 3],
    },
    Cylinder {
        base: [f32; 3],
        axis: [f32; 3],
        radius: f32,
        height: f32,
    },
    Plane {
        point: [f32; 3],
        normal: [f32; 3],
    },
    XyGauss {
        center: [f32; 2],
        sigma: [f32; 2],
        z0: f32,
        amplitude: f32,
    },
}

impl ConstraintSpec {
    pub fn validate(&self) -> TrajResult<()> {
        match self.mode {
            ConstraintMode::Above | ConstraintMode::Below | ConstraintMode::Over => {
                if !matches!(
                    self.shape,
                    ShapeSpec::Plane { .. } | ShapeSpec::XyGauss { .. }
                ) {
                    return Err(TrajError::Invalid(
                        "above/below/over constraints require plane or xygauss shape".into(),
                    ));
                }
            }
            _ => {}
        }
        match &self.shape {
            ShapeSpec::Box { min, max } => {
                for i in 0..3 {
                    if max[i] <= min[i] {
                        return Err(TrajError::Invalid(
                            "box constraint requires max > min".into(),
                        ));
                    }
                }
            }
            ShapeSpec::Cube { side, .. } => {
                if *side <= 0.0 {
                    return Err(TrajError::Invalid(
                        "cube constraint side must be > 0".into(),
                    ));
                }
            }
            ShapeSpec::Sphere { radius, .. } => {
                if *radius <= 0.0 {
                    return Err(TrajError::Invalid(
                        "sphere constraint radius must be > 0".into(),
                    ));
                }
            }
            ShapeSpec::Ellipsoid { radii, .. } => {
                if radii.iter().any(|&v| v <= 0.0) {
                    return Err(TrajError::Invalid(
                        "ellipsoid constraint radii must be > 0".into(),
                    ));
                }
            }
            ShapeSpec::Cylinder {
                axis,
                radius,
                height,
                ..
            } => {
                if *radius <= 0.0 || *height <= 0.0 {
                    return Err(TrajError::Invalid(
                        "cylinder constraint radius/height must be > 0".into(),
                    ));
                }
                let v = Vec3::from_array(*axis);
                if v.norm() <= 1.0e-6 {
                    return Err(TrajError::Invalid(
                        "cylinder constraint axis must be non-zero".into(),
                    ));
                }
            }
            ShapeSpec::Plane { normal, .. } => {
                let v = Vec3::from_array(*normal);
                if v.norm() <= 1.0e-6 {
                    return Err(TrajError::Invalid(
                        "plane constraint normal must be non-zero".into(),
                    ));
                }
            }
            ShapeSpec::XyGauss {
                sigma, amplitude, ..
            } => {
                if sigma.iter().any(|&v| v <= 0.0) {
                    return Err(TrajError::Invalid(
                        "xygauss constraint sigma values must be > 0".into(),
                    ));
                }
                if *amplitude == 0.0 {
                    return Err(TrajError::Invalid(
                        "xygauss constraint amplitude must be non-zero".into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

pub fn satisfies_constraints(
    points: &[Vec3],
    constraints: &[ConstraintSpec],
    pbc: Option<PbcBox>,
) -> bool {
    let wrapped: Cow<[Vec3]> = if let Some(pbc_box) = pbc {
        Cow::Owned(points.iter().map(|p| pbc_box.wrap(*p)).collect())
    } else {
        Cow::Borrowed(points)
    };
    for constraint in constraints {
        match constraint.mode {
            ConstraintMode::Inside => {
                if !wrapped.iter().all(|p| contains(&constraint.shape, *p)) {
                    return false;
                }
            }
            ConstraintMode::Outside => {
                if !wrapped.iter().all(|p| !contains(&constraint.shape, *p)) {
                    return false;
                }
            }
            ConstraintMode::Above | ConstraintMode::Over => {
                if !wrapped
                    .iter()
                    .all(|p| plane_halfspace(&constraint.shape, *p, true))
                {
                    return false;
                }
            }
            ConstraintMode::Below => {
                if !wrapped
                    .iter()
                    .all(|p| plane_halfspace(&constraint.shape, *p, false))
                {
                    return false;
                }
            }
        }
    }
    true
}

fn contains(shape: &ShapeSpec, point: Vec3) -> bool {
    match shape {
        ShapeSpec::Box { min, max } => {
            let min = Vec3::from_array(*min);
            let max = Vec3::from_array(*max);
            point.x >= min.x
                && point.x <= max.x
                && point.y >= min.y
                && point.y <= max.y
                && point.z >= min.z
                && point.z <= max.z
        }
        ShapeSpec::Cube { center, side } => {
            let c = Vec3::from_array(*center);
            let half = *side * 0.5;
            point.x >= c.x - half
                && point.x <= c.x + half
                && point.y >= c.y - half
                && point.y <= c.y + half
                && point.z >= c.z - half
                && point.z <= c.z + half
        }
        ShapeSpec::Sphere { center, radius } => {
            let c = Vec3::from_array(*center);
            let d = point.sub(c);
            d.dot(d) <= radius * radius
        }
        ShapeSpec::Ellipsoid { center, radii } => {
            let c = Vec3::from_array(*center);
            let dx = (point.x - c.x) / radii[0];
            let dy = (point.y - c.y) / radii[1];
            let dz = (point.z - c.z) / radii[2];
            dx * dx + dy * dy + dz * dz <= 1.0
        }
        ShapeSpec::Cylinder {
            base,
            axis,
            radius,
            height,
        } => {
            let base = Vec3::from_array(*base);
            let axis = Vec3::from_array(*axis);
            let axis_len = axis.norm();
            if axis_len <= 1.0e-6 {
                return false;
            }
            let axis_unit = axis.scale(1.0 / axis_len);
            let v = point.sub(base);
            let proj = v.dot(axis_unit);
            if proj < 0.0 || proj > *height {
                return false;
            }
            let radial = v.sub(axis_unit.scale(proj));
            radial.dot(radial) <= radius * radius
        }
        ShapeSpec::Plane {
            point: plane_point,
            normal,
        } => {
            let p0 = Vec3::from_array(*plane_point);
            let n = Vec3::from_array(*normal);
            let d = point.sub(p0).dot(n);
            d >= -1.0e-6
        }
        ShapeSpec::XyGauss { .. } => false,
    }
}

fn plane_halfspace(shape: &ShapeSpec, point: Vec3, above: bool) -> bool {
    match shape {
        ShapeSpec::Plane {
            point: plane_point,
            normal,
        } => {
            let p0 = Vec3::from_array(*plane_point);
            let n = Vec3::from_array(*normal);
            let d = point.sub(p0).dot(n);
            if above {
                d >= -1.0e-6
            } else {
                d <= 1.0e-6
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
            if above {
                point.z >= surf
            } else {
                point.z <= surf
            }
        }
        _ => false,
    }
}
