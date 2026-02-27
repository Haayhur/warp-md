use warp_pack::constraints::{satisfies_constraints, ConstraintMode, ConstraintSpec, ShapeSpec};
use warp_pack::geom::Vec3;
use warp_pack::pbc::PbcBox;

#[test]
fn constraint_box_inside_outside() {
    let points = vec![Vec3::new(0.5, 0.5, 0.5)];
    let inside = ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Box {
            min: [0.0, 0.0, 0.0],
            max: [1.0, 1.0, 1.0],
        },
    };
    assert!(satisfies_constraints(&points, &[inside.clone()], None));
    let outside = ConstraintSpec {
        mode: ConstraintMode::Outside,
        shape: inside.shape.clone(),
    };
    assert!(!satisfies_constraints(&points, &[outside], None));
}

#[test]
fn constraint_shapes_inside() {
    let points = vec![Vec3::new(0.0, 0.0, 0.0)];
    let cube = ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Cube {
            center: [0.0, 0.0, 0.0],
            side: 2.0,
        },
    };
    let sphere = ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        },
    };
    let ellipsoid = ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Ellipsoid {
            center: [0.0, 0.0, 0.0],
            radii: [1.0, 2.0, 3.0],
        },
    };
    let cylinder = ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Cylinder {
            base: [0.0, 0.0, 0.0],
            axis: [0.0, 0.0, 1.0],
            radius: 1.0,
            height: 2.0,
        },
    };
    assert!(satisfies_constraints(&points, &[cube], None));
    assert!(satisfies_constraints(&points, &[sphere], None));
    assert!(satisfies_constraints(&points, &[ellipsoid], None));
    assert!(satisfies_constraints(&points, &[cylinder], None));
}

#[test]
fn constraint_plane_and_xygauss() {
    let above_plane = ConstraintSpec {
        mode: ConstraintMode::Above,
        shape: ShapeSpec::Plane {
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        },
    };
    let below_plane = ConstraintSpec {
        mode: ConstraintMode::Below,
        shape: ShapeSpec::Plane {
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
        },
    };
    let p_above = vec![Vec3::new(0.0, 0.0, 1.0)];
    let p_below = vec![Vec3::new(0.0, 0.0, -1.0)];
    assert!(satisfies_constraints(
        &p_above,
        &[above_plane.clone()],
        None
    ));
    assert!(!satisfies_constraints(&p_below, &[above_plane], None));
    assert!(satisfies_constraints(&p_below, &[below_plane], None));

    let gauss_above = ConstraintSpec {
        mode: ConstraintMode::Above,
        shape: ShapeSpec::XyGauss {
            center: [0.0, 0.0],
            sigma: [1.0, 1.0],
            z0: 0.0,
            amplitude: 1.0,
        },
    };
    let gauss_below = ConstraintSpec {
        mode: ConstraintMode::Below,
        shape: gauss_above.shape.clone(),
    };
    let p_gauss_above = vec![Vec3::new(0.0, 0.0, 1.1)];
    let p_gauss_below = vec![Vec3::new(0.0, 0.0, 0.9)];
    assert!(satisfies_constraints(&p_gauss_above, &[gauss_above], None));
    assert!(satisfies_constraints(&p_gauss_below, &[gauss_below], None));
}

#[test]
fn constraint_validation_rejects_invalid() {
    let bad = ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: -1.0,
        },
    };
    assert!(bad.validate().is_err());

    let bad_plane = ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Plane {
            point: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 0.0],
        },
    };
    assert!(bad_plane.validate().is_err());

    let bad_mode = ConstraintSpec {
        mode: ConstraintMode::Above,
        shape: ShapeSpec::Sphere {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
        },
    };
    assert!(bad_mode.validate().is_err());
}

#[test]
fn constraint_respects_pbc_wrapping() {
    let points = vec![Vec3::new(1.2, 0.2, 0.2)];
    let constraint = ConstraintSpec {
        mode: ConstraintMode::Inside,
        shape: ShapeSpec::Box {
            min: [0.0, 0.0, 0.0],
            max: [0.4, 0.4, 0.4],
        },
    };
    let pbc = PbcBox::from_size([1.0, 1.0, 1.0]).expect("pbc box");
    assert!(satisfies_constraints(&points, &[constraint], Some(pbc)));
}
