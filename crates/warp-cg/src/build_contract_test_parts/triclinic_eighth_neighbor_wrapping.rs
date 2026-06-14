use super::*;

#[test]
fn triclinic_region_wrapping_finds_eighth_neighbor_patch_image() {
    let basis = LayoutBasis2D::new([0.0, 0.0], [10.0, 0.0], [5.0, 10.0]).unwrap();
    let bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 15.0,
        ymin: 0.0,
        ymax: 10.0,
    };
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let region = LeafletRegion {
        name: Some("eighth-image".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                basis.cartesian([8.00, 0.46]),
                basis.cartesian([8.10, 0.46]),
                basis.cartesian([8.10, 0.54]),
                basis.cartesian([8.00, 0.54]),
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let base_cell_point = basis.cartesian([0.05, 0.50]);

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
        ) < 0.6
    );
    assert_eq!(
        closest_periodic_region_image_basis(
            &region,
            base_cell_point,
            bounds,
            periodicity,
            Some(basis)
        ),
        basis.cartesian([8.05, 0.50])
    );
}
