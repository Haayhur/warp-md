use super::*;

#[test]
fn clipped_ellipse_convex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -14.0,
        xmax: 14.0,
        ymin: -9.0,
        ymax: 9.0,
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [24.0, 10.0],
            rotate_degrees: 27.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("polygon".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[0.0, -8.0], [30.0, -8.0], [30.0, 8.0], [0.0, 8.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &polygon], bounds);
    let clipped_polygon_bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 14.0,
        ymin: -8.0,
        ymax: 8.0,
    };
    let ellipse_area = ellipse_rectangle_intersection_area([0.0, 0.0], [24.0, 10.0], 27.0, bounds);
    let polygon_area = 14.0 * 16.0;
    let overlap_area =
        ellipse_rectangle_intersection_area([0.0, 0.0], [24.0, 10.0], 27.0, clipped_polygon_bounds);
    let expected = ellipse_area + polygon_area - overlap_area;
    assert_eq!(estimate.method, "exact_ellipse_convex_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn circle_simple_nonconvex_polygon_region_union_uses_exact_area() {
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
            center_angstrom: [0.0, 0.0],
            radius_angstrom: 10.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("l-shape".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [0.0, -8.0],
                [20.0, -8.0],
                [20.0, -2.0],
                [6.0, -2.0],
                [6.0, 14.0],
                [0.0, 14.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &polygon], bounds);
    let circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 10.0,
    };
    let horizontal = LayoutBounds {
        xmin: 0.0,
        xmax: 20.0,
        ymin: -8.0,
        ymax: -2.0,
    };
    let vertical = LayoutBounds {
        xmin: 0.0,
        xmax: 6.0,
        ymin: -2.0,
        ymax: 14.0,
    };
    let polygon_area = 20.0 * 6.0 + 6.0 * 16.0;
    let overlap_area = circle_rectangle_intersection_area(circle, horizontal)
        + circle_rectangle_intersection_area(circle, vertical);
    let expected = std::f32::consts::PI * 10.0 * 10.0 + polygon_area - overlap_area;
    assert_eq!(estimate.method, "exact_circle_simple_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn ellipse_simple_nonconvex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [24.0, 10.0],
            rotate_degrees: 27.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("l-shape".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [0.0, -8.0],
                [20.0, -8.0],
                [20.0, -2.0],
                [6.0, -2.0],
                [6.0, 14.0],
                [0.0, 14.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &polygon], bounds);
    let horizontal = LayoutBounds {
        xmin: 0.0,
        xmax: 20.0,
        ymin: -8.0,
        ymax: -2.0,
    };
    let vertical = LayoutBounds {
        xmin: 0.0,
        xmax: 6.0,
        ymin: -2.0,
        ymax: 14.0,
    };
    let polygon_area = 20.0 * 6.0 + 6.0 * 16.0;
    let overlap_area =
        ellipse_rectangle_intersection_area([0.0, 0.0], [24.0, 10.0], 27.0, horizontal)
            + ellipse_rectangle_intersection_area([0.0, 0.0], [24.0, 10.0], 27.0, vertical);
    let expected = std::f32::consts::PI * 24.0 * 10.0 + polygon_area - overlap_area;
    assert_eq!(estimate.method, "exact_ellipse_simple_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn clipped_circle_simple_nonconvex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -10.0,
        xmax: 14.0,
        ymin: -10.0,
        ymax: 10.0,
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: 10.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("l-shape".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [0.0, -8.0],
                [20.0, -8.0],
                [20.0, -2.0],
                [6.0, -2.0],
                [6.0, 14.0],
                [0.0, 14.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &polygon], bounds);
    let circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 10.0,
    };
    let horizontal = LayoutBounds {
        xmin: 0.0,
        xmax: 14.0,
        ymin: -8.0,
        ymax: -2.0,
    };
    let vertical = LayoutBounds {
        xmin: 0.0,
        xmax: 6.0,
        ymin: -2.0,
        ymax: 10.0,
    };
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let polygon_area = 14.0 * 6.0 + 6.0 * 12.0;
    let overlap_area = circle_rectangle_intersection_area(circle, horizontal)
        + circle_rectangle_intersection_area(circle, vertical);
    let expected = circle_area + polygon_area - overlap_area;
    assert_eq!(estimate.method, "exact_circle_simple_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn clipped_ellipse_simple_nonconvex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -10.0,
        xmax: 14.0,
        ymin: -10.0,
        ymax: 10.0,
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [24.0, 10.0],
            rotate_degrees: 27.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("l-shape".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [0.0, -8.0],
                [20.0, -8.0],
                [20.0, -2.0],
                [6.0, -2.0],
                [6.0, 14.0],
                [0.0, 14.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &polygon], bounds);
    let horizontal = LayoutBounds {
        xmin: 0.0,
        xmax: 14.0,
        ymin: -8.0,
        ymax: -2.0,
    };
    let vertical = LayoutBounds {
        xmin: 0.0,
        xmax: 6.0,
        ymin: -2.0,
        ymax: 10.0,
    };
    let ellipse_area = ellipse_rectangle_intersection_area([0.0, 0.0], [24.0, 10.0], 27.0, bounds);
    let polygon_area = 14.0 * 6.0 + 6.0 * 12.0;
    let overlap_area =
        ellipse_rectangle_intersection_area([0.0, 0.0], [24.0, 10.0], 27.0, horizontal)
            + ellipse_rectangle_intersection_area([0.0, 0.0], [24.0, 10.0], 27.0, vertical);
    let expected = ellipse_area + polygon_area - overlap_area;
    assert_eq!(estimate.method, "exact_ellipse_simple_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_axis_aligned_ellipse_axis_aligned_rectangle_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [20.0, 10.0],
            rotate_degrees: 0.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [10.0, 0.0],
            size_angstrom: [30.0, 12.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &rectangle], bounds);
    let ellipse_area = std::f32::consts::PI * 20.0 * 10.0;
    let rectangle_area = 30.0 * 12.0;
    let rectangle_bounds = LayoutBounds {
        xmin: -5.0,
        xmax: 25.0,
        ymin: -6.0,
        ymax: 6.0,
    };
    let overlap_area = axis_aligned_ellipse_rectangle_intersection_area(
        [0.0, 0.0],
        [20.0, 10.0],
        rectangle_bounds,
    );
    let expected = ellipse_area + rectangle_area - overlap_area;
    assert_eq!(
        estimate.method,
        "exact_axis_aligned_ellipse_axis_aligned_rectangle_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_rotated_ellipse_axis_aligned_rectangle_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -22.0,
        xmax: 28.0,
        ymin: -18.0,
        ymax: 16.0,
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [28.0, 11.0],
            rotate_degrees: 32.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [6.0, -1.0],
            size_angstrom: [30.0, 14.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &rectangle], bounds);
    let rectangle_bounds = LayoutBounds {
        xmin: -9.0,
        xmax: 21.0,
        ymin: -8.0,
        ymax: 6.0,
    };
    let ellipse_area = ellipse_rectangle_intersection_area([0.0, 0.0], [28.0, 11.0], 32.0, bounds);
    let rectangle_area = 30.0 * 14.0;
    let overlap_area =
        ellipse_rectangle_intersection_area([0.0, 0.0], [28.0, 11.0], 32.0, rectangle_bounds);
    let expected = ellipse_area + rectangle_area - overlap_area;

    assert_eq!(
        estimate.method,
        "exact_rotated_ellipse_axis_aligned_rectangle_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_oriented_ellipse_rotated_rectangle_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [28.0, 11.0],
            rotate_degrees: 32.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [8.0, -3.0],
            size_angstrom: [34.0, 14.0],
            rotate_degrees: -24.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &rectangle], bounds);
    let local_ellipse_center = inverse_rotated_xy([0.0, 0.0], [8.0, -3.0], -24.0);
    let local_rectangle = LayoutBounds {
        xmin: -17.0,
        xmax: 17.0,
        ymin: -7.0,
        ymax: 7.0,
    };
    let overlap_area = ellipse_rectangle_intersection_area(
        local_ellipse_center,
        [28.0, 11.0],
        56.0,
        local_rectangle,
    );
    let expected = std::f32::consts::PI * 28.0 * 11.0 + 34.0 * 14.0 - overlap_area;
    let numeric = numeric_oriented_ellipse_rotated_rectangle_union_area(
        [0.0, 0.0],
        [28.0, 11.0],
        32.0,
        [8.0, -3.0],
        [34.0, 14.0],
        -24.0,
        bounds,
        900,
    );

    assert_eq!(
        estimate.method,
        "exact_oriented_ellipse_rotated_rectangle_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
    assert!((estimate.area_angstrom2 - numeric).abs() < 3.0);
}

#[test]
fn overlapping_centered_circle_axis_aligned_ellipse_region_union_uses_exact_area() {
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
            center_angstrom: [0.0, 0.0],
            radius_angstrom: 12.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: [20.0, 8.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &ellipse], bounds);
    let circle_area = std::f32::consts::PI * 12.0 * 12.0;
    let ellipse_area = std::f32::consts::PI * 20.0 * 8.0;
    let crossing =
        ((8.0_f32.powi(2) - 12.0_f32.powi(2)) / (8.0_f32.powi(2) / 20.0_f32.powi(2) - 1.0)).sqrt();
    let overlap = 4.0
        * ((ellipse_upper_arc_integral(20.0, 8.0, crossing as f64)
            - ellipse_upper_arc_integral(20.0, 8.0, 0.0))
            + (circle_upper_arc_integral(12.0, 12.0)
                - circle_upper_arc_integral(12.0, crossing as f64)));
    let expected = circle_area + ellipse_area - overlap as f32;
    assert_eq!(estimate.method, "exact_circle_oriented_ellipse_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn centered_circle_axis_aligned_ellipse_overlap_handles_tall_ellipse() {
    let overlap = circle_axis_aligned_ellipse_intersection_area(
        CircleRegion {
            center: [0.0, 0.0],
            radius: 12.0,
        },
        [0.0, 0.0],
        [8.0, 20.0],
    );
    let crossing =
        ((20.0_f32.powi(2) - 12.0_f32.powi(2)) / (20.0_f32.powi(2) / 8.0_f32.powi(2) - 1.0)).sqrt();
    let expected = 4.0
        * ((circle_upper_arc_integral(12.0, crossing as f64)
            - circle_upper_arc_integral(12.0, 0.0))
            + (ellipse_upper_arc_integral(8.0, 20.0, 8.0)
                - ellipse_upper_arc_integral(8.0, 20.0, crossing as f64)));
    assert!((overlap - expected as f32).abs() < 0.05);
}

#[test]
fn off_center_circle_axis_aligned_ellipse_region_union_uses_analytic_area() {
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
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &ellipse], bounds);
    let circle_area = std::f32::consts::PI * 13.0 * 13.0;
    let ellipse_area = std::f32::consts::PI * 18.0 * 9.0;
    let overlap = circle_axis_aligned_ellipse_intersection_area(
        CircleRegion {
            center: [-4.0, 2.0],
            radius: 13.0,
        },
        [5.0, -1.5],
        [18.0, 9.0],
    );
    let expected = circle_area + ellipse_area - overlap;
    assert_eq!(estimate.method, "exact_circle_oriented_ellipse_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);

    let numeric_overlap = numeric_circle_ellipse_overlap_area(
        CircleRegion {
            center: [-4.0, 2.0],
            radius: 13.0,
        },
        [5.0, -1.5],
        [18.0, 9.0],
        20_000,
    );
    assert!((overlap - numeric_overlap).abs() < 0.05);
}
