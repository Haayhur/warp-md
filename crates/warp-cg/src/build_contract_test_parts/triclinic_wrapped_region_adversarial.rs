use super::*;

#[test]
fn triclinic_wrapping_handles_heavily_clipped_rotated_eighth_neighbor_polygon() {
    let basis = LayoutBasis2D::new([0.0, 0.0], [10.0, 0.0], [5.0, 10.0]).unwrap();
    let bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 15.0,
        ymin: 0.0,
        ymax: 10.0,
    };
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let center = basis.cartesian([8.04, 0.97]);
    let local_points = [
        [-0.18, -0.035],
        [0.18, -0.035],
        [0.18, 0.035],
        [-0.18, 0.035],
    ];
    let region = LeafletRegion {
        name: Some("rotated-eighth-image".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: local_points
                .iter()
                .map(|[dx, dy]| [center[0] + dx, center[1] + dy])
                .collect(),
            scale_xy: Some([1.2, 0.75]),
            rotate_degrees: 27.0,
        },
    };
    let base_cell_point = basis.cartesian([0.04, 0.97]);

    assert!(region_contains_point_periodic_basis(
        &region,
        base_cell_point,
        bounds,
        periodicity,
        Some(basis)
    ));
    assert!(
        region_boundary_distance_periodic_basis(
            &region,
            base_cell_point,
            bounds,
            periodicity,
            Some(basis)
        ) < 0.15
    );
    let closest = closest_periodic_region_image_basis(
        &region,
        base_cell_point,
        bounds,
        periodicity,
        Some(basis),
    );
    let expected = basis.cartesian([8.04, 0.97]);
    assert!((closest[0] - expected[0]).abs() < 1.0e-5);
    assert!((closest[1] - expected[1]).abs() < 1.0e-5);
}

#[test]
fn triclinic_wrapping_handles_far_rotated_twentieth_image_polygon() {
    let basis = LayoutBasis2D::new([0.0, 0.0], [10.0, 0.0], [5.0, 10.0]).unwrap();
    let bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 15.0,
        ymin: 0.0,
        ymax: 10.0,
    };
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let center = basis.cartesian([20.03, -13.04]);
    let local_points = [
        [-0.22, -0.030],
        [0.19, -0.055],
        [0.24, 0.025],
        [-0.20, 0.045],
    ];
    let region = LeafletRegion {
        name: Some("far-rotated-twentieth-image".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: local_points
                .iter()
                .map(|[dx, dy]| [center[0] + dx, center[1] + dy])
                .collect(),
            scale_xy: Some([1.35, 0.70]),
            rotate_degrees: -38.0,
        },
    };
    let base_cell_point = basis.cartesian([0.03, 0.96]);

    assert!(
        !periodic_point_images_basis(base_cell_point, bounds, periodicity, Some(basis))
            .into_iter()
            .any(|image| region_contains_point(&region, image))
    );
    assert!(region_contains_point_periodic_basis(
        &region,
        base_cell_point,
        bounds,
        periodicity,
        Some(basis)
    ));
    assert!(
        region_boundary_distance_periodic_basis(
            &region,
            base_cell_point,
            bounds,
            periodicity,
            Some(basis)
        ) < 0.20
    );
    let closest = closest_periodic_region_image_basis(
        &region,
        base_cell_point,
        bounds,
        periodicity,
        Some(basis),
    );
    let expected = basis.cartesian([20.03, -13.04]);
    assert!((closest[0] - expected[0]).abs() < 1.0e-5);
    assert!((closest[1] - expected[1]).abs() < 1.0e-5);
}
