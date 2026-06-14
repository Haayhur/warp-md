use super::*;

#[test]
fn circle_overlapping_convex_polygons_disjoint_ellipse_union_uses_exact_area() {
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
    let horizontal = LeafletRegion {
        name: Some("horizontal".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [-6.0, 0.0],
            size_angstrom: [20.0, 10.0],
            rotate_degrees: 0.0,
        },
    };
    let vertical = LeafletRegion {
        name: Some("vertical".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [-3.0, 2.0],
            size_angstrom: [10.0, 18.0],
            rotate_degrees: 0.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [12.0, 0.0],
            radius_angstrom: [5.0, 7.0],
            rotate_degrees: 25.0,
        },
    };

    let estimate =
        region_union_area_angstrom2(&[&circle, &horizontal, &vertical, &ellipse], bounds);
    let circle_polygons =
        exact_circle_convex_polygons_region_union_area(&[&circle, &horizontal, &vertical], bounds)
            .unwrap();
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let circle_ellipse =
        exact_circle_oriented_ellipse_region_union_area(&[&circle, &ellipse], bounds).unwrap();
    let expected = circle_polygons + ellipse_area - (circle_area + ellipse_area - circle_ellipse);

    assert!(regions_are_exactly_disjoint(
        &ellipse,
        region_bounds(&ellipse).unwrap(),
        &horizontal,
        region_bounds(&horizontal).unwrap(),
        bounds,
    )
    .unwrap());
    assert!(!regions_are_exactly_disjoint(
        &horizontal,
        region_bounds(&horizontal).unwrap(),
        &vertical,
        region_bounds(&vertical).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(
        estimate.method,
        "exact_circle_convex_polygons_disjoint_ellipses_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn ellipse_overlapping_convex_polygons_disjoint_circle_union_uses_exact_area() {
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
            rotate_degrees: 20.0,
        },
    };
    let horizontal = LeafletRegion {
        name: Some("horizontal".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [-7.0, 0.0],
            size_angstrom: [20.0, 10.0],
            rotate_degrees: 0.0,
        },
    };
    let vertical = LeafletRegion {
        name: Some("vertical".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [-4.0, 2.0],
            size_angstrom: [10.0, 18.0],
            rotate_degrees: 0.0,
        },
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [13.0, 0.0],
            radius_angstrom: 5.0,
        },
    };

    let estimate =
        region_union_area_angstrom2(&[&ellipse, &horizontal, &vertical, &circle], bounds);
    let ellipse_polygons = exact_ellipse_convex_polygons_region_union_area(
        &[&ellipse, &horizontal, &vertical],
        bounds,
    )
    .unwrap();
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let ellipse_circle =
        exact_circle_oriented_ellipse_region_union_area(&[&circle, &ellipse], bounds).unwrap();
    let expected = ellipse_polygons + circle_area - (ellipse_area + circle_area - ellipse_circle);

    assert!(regions_are_exactly_disjoint(
        &circle,
        region_bounds(&circle).unwrap(),
        &horizontal,
        region_bounds(&horizontal).unwrap(),
        bounds,
    )
    .unwrap());
    assert!(!regions_are_exactly_disjoint(
        &horizontal,
        region_bounds(&horizontal).unwrap(),
        &vertical,
        region_bounds(&vertical).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(
        estimate.method,
        "exact_ellipse_convex_polygons_disjoint_circles_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn circle_overlapping_convex_polygons_disjoint_mixed_shapes_union_uses_exact_area() {
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
            radius_angstrom: 20.0,
        },
    };
    let lower_left = LeafletRegion {
        name: Some("lower_left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[-18.0, -10.0], [-2.0, -10.0], [-2.0, 6.0], [-18.0, 6.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let upper_left = LeafletRegion {
        name: Some("upper_left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[-14.0, -4.0], [2.0, -4.0], [2.0, 12.0], [-14.0, 12.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [12.0, -8.0],
            radius_angstrom: [4.0, 3.0],
            rotate_degrees: 25.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [12.0, 8.0],
            size_angstrom: [6.0, 4.0],
            rotate_degrees: 20.0,
        },
    };

    let regions = [&circle, &lower_left, &upper_left, &ellipse, &rectangle];
    let estimate = region_union_area_angstrom2(&regions, bounds);
    let curve_polygons = exact_circle_convex_polygons_region_union_area(
        &[&circle, &lower_left, &upper_left],
        bounds,
    )
    .unwrap();
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let rectangle_area = exact_single_region_area(&rectangle, bounds).unwrap();
    let circle_ellipse_union =
        exact_pair_region_union_area_without_disjoint(&circle, &ellipse, bounds).unwrap();
    let circle_rectangle_union =
        exact_pair_region_union_area_without_disjoint(&circle, &rectangle, bounds).unwrap();
    let expected = curve_polygons + ellipse_area + rectangle_area
        - (circle_area + ellipse_area - circle_ellipse_union)
        - (circle_area + rectangle_area - circle_rectangle_union);

    assert!(!regions_are_exactly_disjoint(
        &lower_left,
        region_bounds(&lower_left).unwrap(),
        &upper_left,
        region_bounds(&upper_left).unwrap(),
        bounds,
    )
    .unwrap());
    assert!(regions_are_exactly_disjoint(
        &ellipse,
        region_bounds(&ellipse).unwrap(),
        &rectangle,
        region_bounds(&rectangle).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(
        estimate.method,
        "exact_circle_convex_polygons_disjoint_mixed_shapes_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}

#[test]
fn ellipse_overlapping_convex_polygons_disjoint_mixed_shapes_union_uses_exact_area() {
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
            radius_angstrom: [21.0, 12.0],
            rotate_degrees: 15.0,
        },
    };
    let lower_left = LeafletRegion {
        name: Some("lower_left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[-18.0, -10.0], [-2.0, -10.0], [-2.0, 6.0], [-18.0, 6.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let upper_left = LeafletRegion {
        name: Some("upper_left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[-14.0, -4.0], [2.0, -4.0], [2.0, 12.0], [-14.0, 12.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [13.0, -7.0],
            radius_angstrom: 4.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [13.0, 8.0],
            size_angstrom: [5.0, 4.0],
            rotate_degrees: 18.0,
        },
    };

    let regions = [&ellipse, &lower_left, &upper_left, &circle, &rectangle];
    let estimate = region_union_area_angstrom2(&regions, bounds);
    let curve_polygons = exact_ellipse_convex_polygons_region_union_area(
        &[&ellipse, &lower_left, &upper_left],
        bounds,
    )
    .unwrap();
    let ellipse_area = exact_single_region_area(&ellipse, bounds).unwrap();
    let circle_area = exact_single_region_area(&circle, bounds).unwrap();
    let rectangle_area = exact_single_region_area(&rectangle, bounds).unwrap();
    let ellipse_circle_union =
        exact_pair_region_union_area_without_disjoint(&ellipse, &circle, bounds).unwrap();
    let ellipse_rectangle_union =
        exact_pair_region_union_area_without_disjoint(&ellipse, &rectangle, bounds).unwrap();
    let expected = curve_polygons + circle_area + rectangle_area
        - (ellipse_area + circle_area - ellipse_circle_union)
        - (ellipse_area + rectangle_area - ellipse_rectangle_union);

    assert!(!regions_are_exactly_disjoint(
        &lower_left,
        region_bounds(&lower_left).unwrap(),
        &upper_left,
        region_bounds(&upper_left).unwrap(),
        bounds,
    )
    .unwrap());
    assert!(regions_are_exactly_disjoint(
        &circle,
        region_bounds(&circle).unwrap(),
        &rectangle,
        region_bounds(&rectangle).unwrap(),
        bounds,
    )
    .unwrap());
    assert_eq!(
        estimate.method,
        "exact_ellipse_convex_polygons_disjoint_mixed_shapes_union"
    );
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 0.05);
}
