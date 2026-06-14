use super::*;

#[test]
fn clipped_circle_axis_aligned_ellipse_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -8.0,
        xmax: 18.0,
        ymin: -6.0,
        ymax: 9.0,
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-4.0, 2.0],
            radius_angstrom: 13.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [5.0, -1.5],
            radius_angstrom: [18.0, 9.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &ellipse], bounds);
    let clipped_circle = CircleRegion {
        center: [-4.0, 2.0],
        radius: 13.0,
    };
    let circle_area = circle_rectangle_intersection_area(clipped_circle, bounds);
    let ellipse_area =
        axis_aligned_ellipse_rectangle_intersection_area([5.0, -1.5], [18.0, 9.0], bounds);
    let overlap = circle_axis_aligned_ellipse_intersection_area_clipped(
        clipped_circle,
        [5.0, -1.5],
        [18.0, 9.0],
        bounds,
    );
    let expected = circle_area + ellipse_area - overlap;
    assert_eq!(estimate.method, "exact_circle_oriented_ellipse_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);

    let numeric_overlap = numeric_circle_ellipse_overlap_area_clipped(
        clipped_circle,
        [5.0, -1.5],
        [18.0, 9.0],
        bounds,
        20_000,
    );
    assert!((overlap - numeric_overlap).abs() < 0.05);
}

#[test]
fn circle_multiple_disjoint_ellipses_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -60.0,
        xmax: 60.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: 16.0,
        },
    };
    let left_ellipse = LeafletRegion {
        name: Some("left_ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-12.0, 0.0],
            radius_angstrom: [5.0, 8.0],
            rotate_degrees: 0.0,
        },
    };
    let right_ellipse = LeafletRegion {
        name: Some("right_ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [12.0, 0.0],
            radius_angstrom: [5.0, 8.0],
            rotate_degrees: 28.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &left_ellipse, &right_ellipse], bounds);
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let left_area = exact_single_region_area(&left_ellipse, bounds).unwrap();
    let right_area = exact_single_region_area(&right_ellipse, bounds).unwrap();
    let left_pair =
        exact_circle_oriented_ellipse_region_union_area(&[&circle, &left_ellipse], bounds).unwrap();
    let right_pair =
        exact_circle_oriented_ellipse_region_union_area(&[&circle, &right_ellipse], bounds)
            .unwrap();
    let expected = circle_area + left_area + right_area
        - (circle_area + left_area - left_pair)
        - (circle_area + right_area - right_pair);

    assert!(regions_are_exactly_disjoint(
        &left_ellipse,
        region_bounds(&left_ellipse).unwrap(),
        &right_ellipse,
        region_bounds(&right_ellipse).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(estimate.method, "exact_circle_disjoint_ellipses_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn ellipse_multiple_disjoint_circles_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -60.0,
        xmax: 60.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [20.0, 11.0],
            rotate_degrees: 24.0,
        },
    };
    let left_circle = LeafletRegion {
        name: Some("left_circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-12.5, -1.0],
            radius_angstrom: 5.0,
        },
    };
    let right_circle = LeafletRegion {
        name: Some("right_circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [12.5, 1.0],
            radius_angstrom: 5.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &left_circle, &right_circle], bounds);
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let left_area = exact_single_region_area(&left_circle, bounds).unwrap();
    let right_area = exact_single_region_area(&right_circle, bounds).unwrap();
    let left_pair =
        exact_circle_oriented_ellipse_region_union_area(&[&left_circle, &ellipse], bounds).unwrap();
    let right_pair =
        exact_circle_oriented_ellipse_region_union_area(&[&right_circle, &ellipse], bounds)
            .unwrap();
    let expected = ellipse_area + left_area + right_area
        - (ellipse_area + left_area - left_pair)
        - (ellipse_area + right_area - right_pair);

    assert!(regions_are_exactly_disjoint(
        &left_circle,
        region_bounds(&left_circle).unwrap(),
        &right_circle,
        region_bounds(&right_circle).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(estimate.method, "exact_ellipse_disjoint_circles_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn circle_disjoint_mixed_shapes_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -60.0,
        xmax: 60.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: 18.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-12.0, 0.0],
            radius_angstrom: [5.0, 8.0],
            rotate_degrees: 18.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("polygon".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[9.0, -5.0], [17.0, -5.0], [17.0, 5.0], [9.0, 5.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &ellipse, &polygon], bounds);
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let polygon_area = exact_single_region_area(&polygon, bounds).unwrap();
    let circle_ellipse =
        exact_circle_oriented_ellipse_region_union_area(&[&circle, &ellipse], bounds).unwrap();
    let circle_polygon =
        exact_circle_convex_polygon_region_union_area(&[&circle, &polygon], bounds).unwrap();
    let expected = circle_area + ellipse_area + polygon_area
        - (circle_area + ellipse_area - circle_ellipse)
        - (circle_area + polygon_area - circle_polygon);

    assert!(regions_are_exactly_disjoint(
        &ellipse,
        region_bounds(&ellipse).unwrap(),
        &polygon,
        region_bounds(&polygon).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(estimate.method, "exact_circle_disjoint_mixed_shapes_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn ellipse_disjoint_mixed_shapes_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -60.0,
        xmax: 60.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [20.0, 11.0],
            rotate_degrees: 24.0,
        },
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-12.5, -1.0],
            radius_angstrom: 5.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("polygon".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[9.0, -5.0], [17.0, -5.0], [17.0, 5.0], [9.0, 5.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &circle, &polygon], bounds);
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let polygon_area = exact_single_region_area(&polygon, bounds).unwrap();
    let ellipse_circle =
        exact_circle_oriented_ellipse_region_union_area(&[&circle, &ellipse], bounds).unwrap();
    let ellipse_polygon =
        exact_ellipse_convex_polygon_region_union_area(&[&ellipse, &polygon], bounds).unwrap();
    let expected = ellipse_area + circle_area + polygon_area
        - (ellipse_area + circle_area - ellipse_circle)
        - (ellipse_area + polygon_area - ellipse_polygon);

    assert!(regions_are_exactly_disjoint(
        &circle,
        region_bounds(&circle).unwrap(),
        &polygon,
        region_bounds(&polygon).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(estimate.method, "exact_ellipse_disjoint_mixed_shapes_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn rectangle_disjoint_mixed_shapes_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -60.0,
        xmax: 60.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 0.0],
            size_angstrom: [42.0, 18.0],
            rotate_degrees: 0.0,
        },
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-13.0, 0.0],
            radius_angstrom: 5.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [12.0, 0.0],
            radius_angstrom: [5.0, 7.0],
            rotate_degrees: 18.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&rectangle, &circle, &ellipse], bounds);
    let rectangle_area = exact_single_region_area(&rectangle, bounds).unwrap();
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let rectangle_circle =
        exact_circle_axis_aligned_rectangle_region_union_area(&[&circle, &rectangle], bounds)
            .unwrap();
    let rectangle_ellipse = exact_rotated_ellipse_axis_aligned_rectangle_region_union_area(
        &[&ellipse, &rectangle],
        bounds,
    )
    .unwrap();
    let expected = rectangle_area + circle_area + ellipse_area
        - (rectangle_area + circle_area - rectangle_circle)
        - (rectangle_area + ellipse_area - rectangle_ellipse);

    assert!(regions_are_exactly_disjoint(
        &circle,
        region_bounds(&circle).unwrap(),
        &ellipse,
        region_bounds(&ellipse).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(
        estimate.method,
        "exact_rectangle_disjoint_mixed_shapes_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn convex_polygon_disjoint_mixed_shapes_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -60.0,
        xmax: 60.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let polygon = LeafletRegion {
        name: Some("polygon".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[0.0, -13.0], [23.0, 0.0], [0.0, 13.0], [-23.0, 0.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-13.0, 0.0],
            radius_angstrom: 5.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [13.0, 0.0],
            radius_angstrom: [5.0, 7.0],
            rotate_degrees: 18.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&polygon, &circle, &ellipse], bounds);
    let polygon_area = exact_single_region_area(&polygon, bounds).unwrap();
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let polygon_circle =
        exact_circle_convex_polygon_region_union_area(&[&circle, &polygon], bounds).unwrap();
    let polygon_ellipse =
        exact_ellipse_convex_polygon_region_union_area(&[&ellipse, &polygon], bounds).unwrap();
    let expected = polygon_area + circle_area + ellipse_area
        - (polygon_area + circle_area - polygon_circle)
        - (polygon_area + ellipse_area - polygon_ellipse);

    assert!(regions_are_exactly_disjoint(
        &circle,
        region_bounds(&circle).unwrap(),
        &ellipse,
        region_bounds(&ellipse).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(
        estimate.method,
        "exact_convex_polygon_disjoint_mixed_shapes_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn off_center_circle_rotated_ellipse_region_union_uses_analytic_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-4.0, 2.0],
            radius_angstrom: 13.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [5.0, -1.5],
            radius_angstrom: [18.0, 9.0],
            rotate_degrees: 35.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &ellipse], bounds);
    let local_circle = inverse_rotated_xy([-4.0, 2.0], [5.0, -1.5], 35.0);
    let overlap = circle_axis_aligned_ellipse_intersection_area(
        CircleRegion {
            center: local_circle,
            radius: 13.0,
        },
        [0.0, 0.0],
        [18.0, 9.0],
    );
    let expected = std::f32::consts::PI * 13.0 * 13.0 + std::f32::consts::PI * 18.0 * 9.0 - overlap;
    assert_eq!(estimate.method, "exact_circle_oriented_ellipse_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn clipped_circle_rotated_ellipse_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -12.0,
        xmax: 18.0,
        ymin: -11.0,
        ymax: 15.0,
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-3.0, 2.0],
            radius_angstrom: 14.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [5.0, -1.5],
            radius_angstrom: [19.0, 8.0],
            rotate_degrees: 37.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &ellipse], bounds);
    let circle_area = circle_rectangle_intersection_area(
        CircleRegion {
            center: [-3.0, 2.0],
            radius: 14.0,
        },
        bounds,
    );
    let ellipse_area = ellipse_rectangle_intersection_area([5.0, -1.5], [19.0, 8.0], 37.0, bounds);
    let overlap = rotated_ellipse_pair_intersection_area_clipped(
        [-3.0, 2.0],
        [14.0, 14.0],
        0.0,
        [5.0, -1.5],
        [19.0, 8.0],
        37.0,
        bounds,
    );
    let expected = circle_area + ellipse_area - overlap;
    let numeric = numeric_rotated_ellipse_pair_union_area(
        [-3.0, 2.0],
        [14.0, 14.0],
        0.0,
        [5.0, -1.5],
        [19.0, 8.0],
        37.0,
        bounds,
        900,
    );

    assert_eq!(
        estimate.method,
        "exact_clipped_circle_rotated_ellipse_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
    assert!((estimate.area_angstrom2 - numeric).abs() < 1.5);
}

#[test]
fn circle_region_union_uses_exact_lens_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
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
    let r = 10.0_f32;
    let d = 10.0_f32;
    let lens =
        2.0 * r.powi(2) * (d / (2.0 * r)).acos() - 0.5 * d * (4.0 * r.powi(2) - d.powi(2)).sqrt();
    let expected = 2.0 * std::f32::consts::PI * r.powi(2) - lens;
    assert_eq!(estimate.method, "exact_circle_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}
