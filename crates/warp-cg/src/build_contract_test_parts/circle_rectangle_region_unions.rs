use super::*;

#[test]
fn component_mixed_region_union_ignores_false_bounding_box_edges() {
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
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [-8.0, 0.0],
            size_angstrom: [12.0, 12.0],
            rotate_degrees: 0.0,
        },
    };
    let l_shape = LeafletRegion {
        name: Some("corner-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [8.0, 8.0],
                [16.0, 8.0],
                [16.0, 10.0],
                [10.0, 10.0],
                [10.0, 16.0],
                [8.0, 16.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &rectangle, &l_shape], bounds);
    let circle_rectangle =
        exact_circle_axis_aligned_rectangle_region_union_area(&[&circle, &rectangle], bounds)
            .unwrap();
    let expected = circle_rectangle + polygon_area(&transformed_polygon_points(&l_shape));

    assert!(axis_aligned_bounds_overlap(
        region_bounds(&circle).unwrap(),
        region_bounds(&l_shape).unwrap()
    ));
    assert_eq!(estimate.method, "exact_component_mixed_region_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 1.0e-3);
}

#[test]
fn component_mixed_region_union_reuses_multiple_circle_rectangle_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-8.0, 0.0],
            radius_angstrom: 14.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [8.0, 0.0],
            radius_angstrom: 14.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 8.0],
            size_angstrom: [36.0, 16.0],
            rotate_degrees: 0.0,
        },
    };
    let island = LeafletRegion {
        name: Some("island".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[42.0, -8.0], [54.0, -8.0], [54.0, 4.0], [42.0, 4.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right, &rectangle, &island], bounds);
    let exact_cluster = exact_circles_axis_aligned_rectangle_region_union_area(
        &[&left, &right, &rectangle],
        bounds,
    )
    .unwrap();
    let expected = exact_cluster + polygon_area(&transformed_polygon_points(&island));

    assert_eq!(estimate.method, "exact_component_mixed_region_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_circle_axis_aligned_rectangle_region_union_uses_exact_area() {
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
            radius_angstrom: 15.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [10.0, 0.0],
            size_angstrom: [30.0, 20.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &rectangle], bounds);
    let circle_area = std::f32::consts::PI * 15.0 * 15.0;
    let rectangle_area = 30.0 * 20.0;
    let rectangle_bounds = LayoutBounds {
        xmin: -5.0,
        xmax: 25.0,
        ymin: -10.0,
        ymax: 10.0,
    };
    let overlap_area = circle_rectangle_intersection_area(
        CircleRegion {
            center: [0.0, 0.0],
            radius: 15.0,
        },
        rectangle_bounds,
    );
    let expected = circle_area + rectangle_area - overlap_area;
    assert_eq!(estimate.method, "exact_circle_axis_aligned_rectangle_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_circle_multiple_axis_aligned_rectangles_region_union_uses_exact_area() {
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
            radius_angstrom: 18.0,
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

    let estimate = region_union_area_angstrom2(&[&circle, &horizontal, &vertical], bounds);
    let circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 18.0,
    };
    let rectangles = [(-9.0, 25.0, -5.0, 5.0), (-6.0, 6.0, -8.0, 22.0)];
    let expected = std::f32::consts::PI * 18.0 * 18.0
        + exact_axis_aligned_rectangle_bounds_union_area(&rectangles)
        - circle_axis_aligned_rectangle_bounds_union_intersection_area(circle, &rectangles, bounds)
            .unwrap();
    assert_eq!(
        estimate.method,
        "exact_circle_axis_aligned_rectangles_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_multiple_circles_axis_aligned_rectangle_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-8.0, 0.0],
            radius_angstrom: 14.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [8.0, 0.0],
            radius_angstrom: 14.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 8.0],
            size_angstrom: [36.0, 16.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right, &rectangle], bounds);
    let circles = [
        CircleRegion {
            center: [-8.0, 0.0],
            radius: 14.0,
        },
        CircleRegion {
            center: [8.0, 0.0],
            radius: 14.0,
        },
    ];
    let rectangle_bounds = LayoutBounds {
        xmin: -18.0,
        xmax: 18.0,
        ymin: 0.0,
        ymax: 16.0,
    };
    let expected = exact_clipped_circle_union_area(&circles, bounds) + 36.0 * 16.0
        - exact_clipped_circle_union_area(&circles, rectangle_bounds);

    assert_eq!(
        estimate.method,
        "exact_circles_axis_aligned_rectangle_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_multiple_circles_multiple_axis_aligned_rectangles_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-8.0, 0.0],
            radius_angstrom: 14.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [8.0, 0.0],
            radius_angstrom: 14.0,
        },
    };
    let horizontal = LeafletRegion {
        name: Some("horizontal".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 8.0],
            size_angstrom: [36.0, 16.0],
            rotate_degrees: 0.0,
        },
    };
    let vertical = LeafletRegion {
        name: Some("vertical".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 0.0],
            size_angstrom: [12.0, 36.0],
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right, &horizontal, &vertical], bounds);
    let circles = [
        CircleRegion {
            center: [-8.0, 0.0],
            radius: 14.0,
        },
        CircleRegion {
            center: [8.0, 0.0],
            radius: 14.0,
        },
    ];
    let rectangles = [(-18.0, 18.0, 0.0, 16.0), (-6.0, 6.0, -18.0, 18.0)];
    let expected = exact_clipped_circle_union_area(&circles, bounds)
        + exact_axis_aligned_rectangle_bounds_union_area(&rectangles)
        - clipped_circle_union_axis_aligned_rectangle_bounds_union_intersection_area(
            &circles,
            &rectangles,
            bounds,
        )
        .unwrap();

    assert_eq!(
        estimate.method,
        "exact_circles_axis_aligned_rectangles_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_circle_multiple_convex_polygons_region_union_uses_exact_area() {
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
            radius_angstrom: 18.0,
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

    let estimate = region_union_area_angstrom2(&[&circle, &tilted, &diamond], bounds);
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
    let circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 18.0,
    };
    let expected = std::f32::consts::PI * 18.0 * 18.0
        + exact_convex_polygon_union_area_from_polygons(&polygons).unwrap()
        - circle_convex_polygon_union_intersection_area(circle, &polygons).unwrap();
    assert_eq!(estimate.method, "exact_circle_convex_polygons_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn disjoint_circles_multiple_convex_polygons_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left_circle = LeafletRegion {
        name: Some("left-circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-35.0, 0.0],
            radius_angstrom: 10.0,
        },
    };
    let right_circle = LeafletRegion {
        name: Some("right-circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [35.0, 0.0],
            radius_angstrom: 9.0,
        },
    };
    let tilted = LeafletRegion {
        name: Some("tilted".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 0.0],
            size_angstrom: [86.0, 9.0],
            rotate_degrees: 18.0,
        },
    };
    let diamond = LeafletRegion {
        name: Some("diamond".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 8.0],
            size_angstrom: [24.0, 14.0],
            rotate_degrees: -36.0,
        },
    };

    let estimate =
        region_union_area_angstrom2(&[&left_circle, &right_circle, &tilted, &diamond], bounds);
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
    let left = CircleRegion {
        center: [-35.0, 0.0],
        radius: 10.0,
    };
    let right = CircleRegion {
        center: [35.0, 0.0],
        radius: 9.0,
    };
    let expected = circle_rectangle_intersection_area(left, bounds)
        + circle_rectangle_intersection_area(right, bounds)
        + exact_convex_polygon_union_area_from_polygons(&polygons).unwrap()
        - circle_convex_polygon_union_intersection_area(left, &polygons).unwrap()
        - circle_convex_polygon_union_intersection_area(right, &polygons).unwrap();

    assert_eq!(
        estimate.method,
        "exact_disjoint_circles_convex_polygons_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn overlapping_circle_multiple_simple_polygons_region_union_uses_exact_area() {
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

    let estimate = region_union_area_angstrom2(&[&circle_region, &left_l, &right_l], bounds);
    let circle = CircleRegion {
        center: [0.0, 0.0],
        radius: 16.0,
    };
    let polygons = [&left_l, &right_l]
        .iter()
        .map(|region| simple_polygon_for_region_clipped_to_bounds(region, bounds).unwrap())
        .collect::<Vec<_>>();
    let expected = circle_rectangle_intersection_area(circle, bounds)
        + polygons
            .iter()
            .map(|polygon| polygon_area(polygon))
            .sum::<f32>()
        - polygons
            .iter()
            .map(|polygon| circle_polygon_intersection_area(circle, polygon))
            .sum::<f32>();

    assert_eq!(estimate.method, "exact_circle_simple_polygons_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}
