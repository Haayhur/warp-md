use super::*;

#[test]
fn clipped_circle_region_union_uses_exact_area_for_disjoint_bounds() {
    let bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let half = LeafletRegion {
        name: Some("half".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: 10.0,
        },
    };
    let clipped = LeafletRegion {
        name: Some("clipped".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [80.0, 95.0],
            radius_angstrom: 10.0,
        },
    };
    let half_estimate = region_union_area_angstrom2(&[&half], bounds);
    assert_eq!(half_estimate.method, "exact_circle_union");
    assert_eq!(half_estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((half_estimate.area_angstrom2 - 0.5 * std::f32::consts::PI * 100.0).abs() < 1.0e-3);

    let union_estimate = region_union_area_angstrom2(&[&half, &clipped], bounds);
    assert_eq!(union_estimate.method, "exact_circle_union");
    assert_eq!(union_estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!(union_estimate.area_angstrom2 > half_estimate.area_angstrom2);
    assert!(union_estimate.area_angstrom2 < std::f32::consts::PI * 200.0);
}

#[test]
fn overlapping_clipped_circle_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "hole".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-5.0, 0.0],
            radius_angstrom: 10.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "hole".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [5.0, 0.0],
            radius_angstrom: 10.0,
        },
    };
    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    let expected = 0.5
        * exact_unclipped_circle_union_area(&[
            CircleRegion {
                center: [-5.0, 0.0],
                radius: 10.0,
            },
            CircleRegion {
                center: [5.0, 0.0],
                radius: 10.0,
            },
        ]);

    assert_eq!(estimate.method, "exact_circle_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn separated_ellipse_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-40.0, 0.0],
            radius_angstrom: [10.0, 20.0],
            rotate_degrees: 30.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [40.0, 0.0],
            radius_angstrom: [15.0, 12.0],
            rotate_degrees: -20.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    let expected = std::f32::consts::PI * (10.0 * 20.0 + 15.0 * 12.0);

    assert_eq!(estimate.method, "exact_disjoint_ellipse_union");
    assert!((estimate.area_angstrom2 - expected).abs() < 1.0e-3);
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
}

#[test]
fn clipped_axis_aligned_ellipse_region_union_uses_exact_area_for_disjoint_bounds() {
    let bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let half = LeafletRegion {
        name: Some("half".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [20.0, 10.0],
            rotate_degrees: 0.0,
        },
    };
    let clipped = LeafletRegion {
        name: Some("clipped".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [80.0, 96.0],
            radius_angstrom: [10.0, 8.0],
            rotate_degrees: 0.0,
        },
    };

    let half_estimate = region_union_area_angstrom2(&[&half], bounds);
    assert_eq!(half_estimate.method, "exact_disjoint_ellipse_union");
    assert_eq!(half_estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!(
        (half_estimate.area_angstrom2 - 0.5 * std::f32::consts::PI * 20.0 * 10.0).abs() < 1.0e-3
    );

    let union_estimate = region_union_area_angstrom2(&[&half, &clipped], bounds);
    assert_eq!(union_estimate.method, "exact_disjoint_ellipse_union");
    assert_eq!(union_estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!(union_estimate.area_angstrom2 > half_estimate.area_angstrom2);
    assert!(union_estimate.area_angstrom2 < std::f32::consts::PI * (20.0 * 10.0 + 10.0 * 8.0));
}

#[test]
fn clipped_rotated_ellipse_region_union_uses_exact_area_for_disjoint_bounds() {
    let bounds = LayoutBounds {
        xmin: -18.0,
        xmax: 22.0,
        ymin: -12.0,
        ymax: 15.0,
    };
    let clipped = LeafletRegion {
        name: Some("rotated-clipped".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [28.0, 11.0],
            rotate_degrees: 32.0,
        },
    };
    let isolated = LeafletRegion {
        name: Some("isolated".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [80.0, 0.0],
            radius_angstrom: [7.0, 5.0],
            rotate_degrees: 47.0,
        },
    };
    let clipped_area = ellipse_rectangle_intersection_area([0.0, 0.0], [28.0, 11.0], 32.0, bounds);
    let numeric_area =
        numeric_rotated_ellipse_rectangle_area([0.0, 0.0], [28.0, 11.0], 32.0, bounds, 900);
    assert!((clipped_area - numeric_area).abs() < 1.0);

    let single = region_union_area_angstrom2(&[&clipped], bounds);
    assert_eq!(single.method, "exact_disjoint_ellipse_union");
    assert_eq!(single.reported_error_bound_angstrom2, Some(0.0));
    assert!((single.area_angstrom2 - clipped_area).abs() < 0.05);

    let wide_bounds = LayoutBounds {
        xmin: -18.0,
        xmax: 100.0,
        ymin: -12.0,
        ymax: 15.0,
    };
    let union = region_union_area_angstrom2(&[&clipped, &isolated], wide_bounds);
    let clipped_area_wide =
        ellipse_rectangle_intersection_area([0.0, 0.0], [28.0, 11.0], 32.0, wide_bounds);
    let isolated_area = std::f32::consts::PI * 7.0 * 5.0;
    assert_eq!(union.method, "exact_disjoint_ellipse_union");
    assert_eq!(union.reported_error_bound_angstrom2, Some(0.0));
    assert!((union.area_angstrom2 - (clipped_area_wide + isolated_area)).abs() < 0.05);
}

#[test]
fn overlapping_same_axis_aligned_ellipse_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-5.0, 0.0],
            radius_angstrom: [20.0, 10.0],
            rotate_degrees: 0.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [5.0, 0.0],
            radius_angstrom: [20.0, 10.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    let scaled_distance = 10.0_f32 / 20.0;
    let unit_lens = 2.0 * (scaled_distance * 0.5).acos()
        - 0.5 * scaled_distance * (4.0 - scaled_distance.powi(2)).sqrt();
    let expected = (2.0 * std::f32::consts::PI - unit_lens) * 20.0 * 10.0;

    assert_eq!(estimate.method, "exact_similar_oriented_ellipse_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_same_rotated_ellipse_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let theta = 30.0_f32.to_radians();
    let offset = [10.0 * theta.cos(), 10.0 * theta.sin()];
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-0.5 * offset[0], -0.5 * offset[1]],
            radius_angstrom: [20.0, 10.0],
            rotate_degrees: 30.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.5 * offset[0], 0.5 * offset[1]],
            radius_angstrom: [20.0, 10.0],
            rotate_degrees: 30.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    let scaled_distance = 10.0_f32 / 20.0;
    let unit_lens = 2.0 * (scaled_distance * 0.5).acos()
        - 0.5 * scaled_distance * (4.0 - scaled_distance.powi(2)).sqrt();
    let expected = (2.0 * std::f32::consts::PI - unit_lens) * 20.0 * 10.0;

    assert_eq!(estimate.method, "exact_similar_oriented_ellipse_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_proportional_ellipse_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let large = LeafletRegion {
        name: Some("large".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-10.0, 0.0],
            radius_angstrom: [20.0, 10.0],
            rotate_degrees: 0.0,
        },
    };
    let small = LeafletRegion {
        name: Some("small".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [10.0, 0.0],
            radius_angstrom: [10.0, 5.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&large, &small], bounds);
    let expected = exact_unclipped_circle_union_area(&[
        CircleRegion {
            center: [-5.0, 0.0],
            radius: 10.0,
        },
        CircleRegion {
            center: [5.0, 0.0],
            radius: 5.0,
        },
    ]) * 2.0;

    assert_eq!(estimate.method, "exact_similar_oriented_ellipse_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_axis_aligned_ellipses_with_different_aspects_use_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("wide".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-4.0, 0.0],
            radius_angstrom: [24.0, 9.0],
            rotate_degrees: 0.0,
        },
    };
    let right = LeafletRegion {
        name: Some("tall".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [5.0, 0.0],
            radius_angstrom: [12.0, 18.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    let overlap = axis_aligned_ellipse_pair_intersection_area_clipped(
        [-4.0, 0.0],
        [24.0, 9.0],
        [5.0, 0.0],
        [12.0, 18.0],
        bounds,
    );
    let expected = std::f32::consts::PI * (24.0 * 9.0 + 12.0 * 18.0) - overlap;
    let numeric = numeric_axis_aligned_ellipse_pair_union_area(
        [-4.0, 0.0],
        [24.0, 9.0],
        [5.0, 0.0],
        [12.0, 18.0],
        bounds,
        900,
    );

    assert_eq!(estimate.method, "exact_axis_aligned_ellipse_pair_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
    assert!((estimate.area_angstrom2 - numeric).abs() < 1.0);
}

#[test]
fn clipped_overlapping_axis_aligned_ellipses_with_different_aspects_use_exact_area() {
    let bounds = LayoutBounds {
        xmin: -8.0,
        xmax: 24.0,
        ymin: -10.0,
        ymax: 16.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [22.0, 8.0],
            rotate_degrees: 0.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [8.0, 3.0],
            radius_angstrom: [13.0, 18.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    let left_area =
        axis_aligned_ellipse_rectangle_intersection_area([0.0, 0.0], [22.0, 8.0], bounds);
    let right_area =
        axis_aligned_ellipse_rectangle_intersection_area([8.0, 3.0], [13.0, 18.0], bounds);
    let overlap = axis_aligned_ellipse_pair_intersection_area_clipped(
        [0.0, 0.0],
        [22.0, 8.0],
        [8.0, 3.0],
        [13.0, 18.0],
        bounds,
    );
    let expected = left_area + right_area - overlap;
    let numeric = numeric_axis_aligned_ellipse_pair_union_area(
        [0.0, 0.0],
        [22.0, 8.0],
        [8.0, 3.0],
        [13.0, 18.0],
        bounds,
        900,
    );

    assert_eq!(estimate.method, "exact_axis_aligned_ellipse_pair_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
    assert!((estimate.area_angstrom2 - numeric).abs() < 1.0);
}

#[test]
fn overlapping_rotated_ellipses_with_different_orientations_use_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-4.0, 1.0],
            radius_angstrom: [24.0, 9.0],
            rotate_degrees: 28.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [5.0, -2.0],
            radius_angstrom: [14.0, 18.0],
            rotate_degrees: -34.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    let overlap = rotated_ellipse_pair_intersection_area_clipped(
        [-4.0, 1.0],
        [24.0, 9.0],
        28.0,
        [5.0, -2.0],
        [14.0, 18.0],
        -34.0,
        bounds,
    );
    let expected = std::f32::consts::PI * (24.0 * 9.0 + 14.0 * 18.0) - overlap;
    let numeric = numeric_rotated_ellipse_pair_union_area(
        [-4.0, 1.0],
        [24.0, 9.0],
        28.0,
        [5.0, -2.0],
        [14.0, 18.0],
        -34.0,
        bounds,
        900,
    );

    assert_eq!(estimate.method, "exact_rotated_ellipse_pair_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
    assert!((estimate.area_angstrom2 - numeric).abs() < 1.5);
}

#[test]
fn clipped_overlapping_rotated_ellipses_with_different_orientations_use_exact_area() {
    let bounds = LayoutBounds {
        xmin: -18.0,
        xmax: 24.0,
        ymin: -12.0,
        ymax: 17.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-2.0, 0.0],
            radius_angstrom: [28.0, 10.0],
            rotate_degrees: 25.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [7.0, 3.0],
            radius_angstrom: [13.0, 20.0],
            rotate_degrees: -42.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    let left_area = ellipse_rectangle_intersection_area([-2.0, 0.0], [28.0, 10.0], 25.0, bounds);
    let right_area = ellipse_rectangle_intersection_area([7.0, 3.0], [13.0, 20.0], -42.0, bounds);
    let overlap = rotated_ellipse_pair_intersection_area_clipped(
        [-2.0, 0.0],
        [28.0, 10.0],
        25.0,
        [7.0, 3.0],
        [13.0, 20.0],
        -42.0,
        bounds,
    );
    let expected = left_area + right_area - overlap;
    let numeric = numeric_rotated_ellipse_pair_union_area(
        [-2.0, 0.0],
        [28.0, 10.0],
        25.0,
        [7.0, 3.0],
        [13.0, 20.0],
        -42.0,
        bounds,
        900,
    );

    assert_eq!(estimate.method, "exact_rotated_ellipse_pair_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
    assert!((estimate.area_angstrom2 - numeric).abs() < 1.5);
}
