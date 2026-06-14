use super::*;

#[test]
fn circle_multiple_simple_polygons_use_exact_pair_proof_for_overlapping_bounds() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let circle_region = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [0.0, 0.0],
            radius_angstrom: 16.0,
        },
    };
    let top_left_l = LeafletRegion {
        name: Some("top-left-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-20.0, -10.0],
                [-16.0, -10.0],
                [-16.0, 6.0],
                [4.0, 6.0],
                [4.0, 10.0],
                [-20.0, 10.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let bottom_right_l = LeafletRegion {
        name: Some("bottom-right-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-4.0, -10.0],
                [20.0, -10.0],
                [20.0, 10.0],
                [16.0, 10.0],
                [16.0, -6.0],
                [-4.0, -6.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let top_polygon = simple_polygon_for_region_clipped_to_bounds(&top_left_l, bounds).unwrap();
    let bottom_polygon =
        simple_polygon_for_region_clipped_to_bounds(&bottom_right_l, bounds).unwrap();
    assert!(axis_aligned_bounds_overlap(
        polygon_bounds(&top_polygon).unwrap(),
        polygon_bounds(&bottom_polygon).unwrap()
    ));

    let estimate =
        region_union_area_angstrom2(&[&circle_region, &top_left_l, &bottom_right_l], bounds);
    let circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 16.0,
    };
    let expected = circle_rectangle_intersection_area(circle, bounds)
        + polygon_area(&top_polygon)
        + polygon_area(&bottom_polygon)
        - circle_polygon_intersection_area(circle, &top_polygon)
        - circle_polygon_intersection_area(circle, &bottom_polygon);

    assert_eq!(estimate.method, "exact_circle_simple_polygons_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_ellipse_multiple_convex_polygons_region_union_uses_exact_area() {
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
            center_angstrom: [2.0, -1.0],
            radius_angstrom: [26.0, 11.0],
            rotate_degrees: 31.0,
        },
    };
    let tilted = LeafletRegion {
        name: Some("tilted".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [6.0, 0.0],
            size_angstrom: [34.0, 10.0],
            rotate_degrees: 24.0,
        },
    };
    let diamond = LeafletRegion {
        name: Some("diamond".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 7.0],
            size_angstrom: [24.0, 14.0],
            rotate_degrees: -37.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &tilted, &diamond], bounds);
    let bounds_polygon = layout_bounds_polygon(bounds);
    let polygons = [&tilted, &diamond]
        .iter()
        .map(|region| {
            convex_polygon_intersection(
                &convex_polygon_for_region(region).unwrap(),
                &bounds_polygon,
            )
        })
        .collect::<Vec<_>>();
    let transformed_polygons = polygons
        .iter()
        .map(|polygon| {
            polygon
                .iter()
                .map(|point| {
                    let local = inverse_rotated_xy(*point, [2.0, -1.0], 31.0);
                    [local[0] / 26.0, local[1] / 11.0]
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let expected = ellipse_rectangle_intersection_area([2.0, -1.0], [26.0, 11.0], 31.0, bounds)
        + exact_convex_polygon_union_area_from_polygons(&polygons).unwrap()
        - circle_convex_polygon_union_intersection_area(
            CircleRegion {
                center: [0.0, 0.0],
                radius: 1.0,
            },
            &transformed_polygons,
        )
        .unwrap()
            * 26.0
            * 11.0;
    assert_eq!(estimate.method, "exact_ellipse_convex_polygons_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_ellipse_multiple_simple_polygons_region_union_uses_exact_area() {
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
            center_angstrom: [2.0, -1.0],
            radius_angstrom: [24.0, 10.0],
            rotate_degrees: 27.0,
        },
    };
    let left_l = LeafletRegion {
        name: Some("left-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-22.0, -9.0],
                [-8.0, -9.0],
                [-8.0, -3.0],
                [-16.0, -3.0],
                [-16.0, 11.0],
                [-22.0, 11.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let right_l = LeafletRegion {
        name: Some("right-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [8.0, -11.0],
                [22.0, -11.0],
                [22.0, 9.0],
                [16.0, 9.0],
                [16.0, -5.0],
                [8.0, -5.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &left_l, &right_l], bounds);
    let polygons = [&left_l, &right_l]
        .iter()
        .map(|region| simple_polygon_for_region_clipped_to_bounds(region, bounds).unwrap())
        .collect::<Vec<_>>();
    let unit_circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 1.0,
    };
    let overlap = polygons
        .iter()
        .map(|polygon| {
            let transformed = polygon
                .iter()
                .map(|point| {
                    let local = inverse_rotated_xy(*point, [2.0, -1.0], 27.0);
                    [local[0] / 24.0, local[1] / 10.0]
                })
                .collect::<Vec<_>>();
            circle_polygon_intersection_area(unit_circle, &transformed) * 24.0 * 10.0
        })
        .sum::<f32>();
    let expected = ellipse_rectangle_intersection_area([2.0, -1.0], [24.0, 10.0], 27.0, bounds)
        + polygons
            .iter()
            .map(|polygon| polygon_area(polygon))
            .sum::<f32>()
        - overlap;

    assert_eq!(estimate.method, "exact_ellipse_simple_polygons_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn ellipse_multiple_simple_polygons_use_exact_pair_proof_for_overlapping_bounds() {
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
            center_angstrom: [2.0, -1.0],
            radius_angstrom: [24.0, 10.0],
            rotate_degrees: 27.0,
        },
    };
    let top_left_l = LeafletRegion {
        name: Some("top-left-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-20.0, -10.0],
                [-16.0, -10.0],
                [-16.0, 6.0],
                [4.0, 6.0],
                [4.0, 10.0],
                [-20.0, 10.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let bottom_right_l = LeafletRegion {
        name: Some("bottom-right-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-4.0, -10.0],
                [20.0, -10.0],
                [20.0, 10.0],
                [16.0, 10.0],
                [16.0, -6.0],
                [-4.0, -6.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let top_polygon = simple_polygon_for_region_clipped_to_bounds(&top_left_l, bounds).unwrap();
    let bottom_polygon =
        simple_polygon_for_region_clipped_to_bounds(&bottom_right_l, bounds).unwrap();
    assert!(axis_aligned_bounds_overlap(
        polygon_bounds(&top_polygon).unwrap(),
        polygon_bounds(&bottom_polygon).unwrap()
    ));

    let estimate = region_union_area_angstrom2(&[&ellipse, &top_left_l, &bottom_right_l], bounds);
    let unit_circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 1.0,
    };
    let overlap = [&top_polygon, &bottom_polygon]
        .iter()
        .map(|polygon| {
            let transformed = polygon
                .iter()
                .map(|point| {
                    let local = inverse_rotated_xy(*point, [2.0, -1.0], 27.0);
                    [local[0] / 24.0, local[1] / 10.0]
                })
                .collect::<Vec<_>>();
            circle_polygon_intersection_area(unit_circle, &transformed) * 24.0 * 10.0
        })
        .sum::<f32>();
    let expected = ellipse_rectangle_intersection_area([2.0, -1.0], [24.0, 10.0], 27.0, bounds)
        + polygon_area(&top_polygon)
        + polygon_area(&bottom_polygon)
        - overlap;

    assert_eq!(estimate.method, "exact_ellipse_simple_polygons_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_ellipse_multiple_axis_aligned_rectangles_region_union_uses_exact_area() {
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
            center_angstrom: [2.0, -1.0],
            radius_angstrom: [26.0, 11.0],
            rotate_degrees: 31.0,
        },
    };
    let horizontal = LeafletRegion {
        name: Some("horizontal".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [8.0, 0.0],
            size_angstrom: [34.0, 10.0],
            rotate_degrees: 0.0,
        },
    };
    let vertical = LeafletRegion {
        name: Some("vertical".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 7.0],
            size_angstrom: [12.0, 30.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &horizontal, &vertical], bounds);
    let rectangles = [(-9.0, 25.0, -5.0, 5.0), (-6.0, 6.0, -8.0, 22.0)];
    let expected = ellipse_rectangle_intersection_area([2.0, -1.0], [26.0, 11.0], 31.0, bounds)
        + exact_axis_aligned_rectangle_bounds_union_area(&rectangles)
        - ellipse_axis_aligned_rectangle_bounds_union_intersection_area(
            [2.0, -1.0],
            [26.0, 11.0],
            31.0,
            &rectangles,
            bounds,
        )
        .unwrap();
    assert_eq!(
        estimate.method,
        "exact_ellipse_axis_aligned_rectangles_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn disjoint_ellipses_multiple_axis_aligned_rectangles_region_union_uses_exact_area() {
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
            center_angstrom: [-35.0, 0.0],
            radius_angstrom: [12.0, 8.0],
            rotate_degrees: 20.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [35.0, 0.0],
            radius_angstrom: [10.0, 14.0],
            rotate_degrees: -25.0,
        },
    };
    let horizontal = LeafletRegion {
        name: Some("horizontal".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 0.0],
            size_angstrom: [86.0, 8.0],
            rotate_degrees: 0.0,
        },
    };
    let vertical = LeafletRegion {
        name: Some("vertical".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 0.0],
            size_angstrom: [10.0, 32.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right, &horizontal, &vertical], bounds);
    let rectangles = [(-43.0, 43.0, -4.0, 4.0), (-5.0, 5.0, -16.0, 16.0)];
    let left_overlap = ellipse_axis_aligned_rectangle_bounds_union_intersection_area(
        [-35.0, 0.0],
        [12.0, 8.0],
        20.0,
        &rectangles,
        bounds,
    )
    .unwrap();
    let right_overlap = ellipse_axis_aligned_rectangle_bounds_union_intersection_area(
        [35.0, 0.0],
        [10.0, 14.0],
        -25.0,
        &rectangles,
        bounds,
    )
    .unwrap();
    let expected = exact_axis_aligned_rectangle_bounds_union_area(&rectangles)
        + ellipse_rectangle_intersection_area([-35.0, 0.0], [12.0, 8.0], 20.0, bounds)
        + ellipse_rectangle_intersection_area([35.0, 0.0], [10.0, 14.0], -25.0, bounds)
        - left_overlap
        - right_overlap;

    assert_eq!(
        estimate.method,
        "exact_disjoint_ellipses_axis_aligned_rectangles_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_circle_rotated_rectangle_region_union_uses_exact_area() {
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
            radius_angstrom: 15.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [7.0, -3.0],
            size_angstrom: [32.0, 12.0],
            rotate_degrees: 28.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &rectangle], bounds);
    let local_circle_center = inverse_rotated_xy([-4.0, 2.0], [7.0, -3.0], 28.0);
    let overlap_area = circle_rectangle_intersection_area(
        CircleRegion {
            center: local_circle_center,
            radius: 15.0,
        },
        LayoutBounds {
            xmin: -16.0,
            xmax: 16.0,
            ymin: -6.0,
            ymax: 6.0,
        },
    );
    let expected = std::f32::consts::PI * 15.0 * 15.0 + 32.0 * 12.0 - overlap_area;
    assert_eq!(estimate.method, "exact_circle_rotated_rectangle_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn clipped_circle_convex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -12.0,
        xmax: 12.0,
        ymin: -8.0,
        ymax: 8.0,
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
        name: Some("polygon".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[0.0, -6.0], [16.0, -6.0], [16.0, 6.0], [0.0, 6.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &polygon], bounds);
    let circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 10.0,
    };
    let circle_area = circle_rectangle_intersection_area(circle, bounds);
    let clipped_polygon_area = 12.0 * 12.0;
    let overlap_area = circle_rectangle_intersection_area(
        circle,
        LayoutBounds {
            xmin: 0.0,
            xmax: 12.0,
            ymin: -6.0,
            ymax: 6.0,
        },
    );
    let expected = circle_area + clipped_polygon_area - overlap_area;
    assert_eq!(estimate.method, "exact_circle_convex_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_ellipse_convex_polygon_region_union_uses_exact_area() {
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
        name: Some("polygon".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[0.0, -8.0], [28.0, -8.0], [28.0, 8.0], [0.0, 8.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&ellipse, &polygon], bounds);
    let polygon_bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 28.0,
        ymin: -8.0,
        ymax: 8.0,
    };
    let ellipse_area = std::f32::consts::PI * 24.0 * 10.0;
    let polygon_area = 28.0 * 16.0;
    let overlap_area =
        ellipse_rectangle_intersection_area([0.0, 0.0], [24.0, 10.0], 27.0, polygon_bounds);
    let expected = ellipse_area + polygon_area - overlap_area;
    assert_eq!(estimate.method, "exact_ellipse_convex_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}
