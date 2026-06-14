use super::*;

#[test]
fn leaflet_regions_adjust_apl_counts() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 100.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [
                {
                    "name": "upper",
                    "side": "upper",
                    "apl_angstrom2": 50.0,
                    "regions": [{
                        "name": "round-hole",
                        "role": "hole",
                        "geometry": {
                            "shape": "circle",
                            "center_angstrom": [0.0, 0.0],
                            "radius_angstrom": 10.0
                        }
                    }],
                    "composition": [{"lipid": "POPC"}]
                },
                {
                    "name": "lower",
                    "side": "lower",
                    "apl_angstrom2": 50.0,
                    "regions": [{
                        "name": "patch",
                        "role": "patch",
                        "geometry": {
                            "shape": "rectangle",
                            "center_angstrom": [0.0, 0.0],
                            "size_angstrom": [50.0, 20.0]
                        }
                    }],
                    "composition": [{"lipid": "POPC"}]
                }
            ]
        }],
        "outputs": {
            "coordinates": temp.path().join("m.gro"),
            "topology": temp.path().join("t.top"),
            "manifest": temp.path().join("m.json")
        }
    });
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 214);
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["exclusion_count"],
        1
    );
}

#[test]
fn polygon_regions_validate_and_adjust_counts() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 100.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 50.0,
                "regions": [{
                    "name": "triangle",
                    "role": "hole",
                    "geometry": {
                        "shape": "polygon",
                        "points_angstrom": [[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]]
                    }
                }],
                "composition": [{"lipid": "POPC"}]
            }]
        }],
        "outputs": {
            "coordinates": temp.path().join("m.gro"),
            "topology": temp.path().join("t.top"),
            "manifest": temp.path().join("m.json")
        }
    });
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 196);
}

#[test]
fn overlapping_patch_regions_use_union_area_for_counts() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [200.0, 200.0, 100.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 60.0,
                "regions": [
                    {"name": "ellipse", "role": "patch", "geometry": {"shape": "ellipse", "center_angstrom": [0.0, 10.0], "radius_angstrom": [30.0, 40.0]}},
                    {"name": "rectangle", "role": "patch", "geometry": {"shape": "rectangle", "center_angstrom": [0.0, 60.0], "size_angstrom": [100.0, 40.0]}},
                    {"name": "triangle", "role": "patch", "geometry": {"shape": "polygon", "points_angstrom": [[0.0, 0.0], [70.0, -70.0], [-70.0, -70.0]]}},
                    {"name": "square", "role": "patch", "geometry": {"shape": "rectangle", "center_angstrom": [-70.0, 0.0], "size_angstrom": [40.0, 40.0]}},
                    {"name": "circle", "role": "patch", "geometry": {"shape": "circle", "center_angstrom": [70.0, 0.0], "radius_angstrom": 20.0}}
                ],
                "composition": [{"lipid": "POPC"}]
            }]
        }],
        "outputs": {
            "coordinates": temp.path().join("m.gro"),
            "topology": temp.path().join("t.top"),
            "manifest": temp.path().join("m.json")
        }
    });
    let text = serde_json::to_string(&request).unwrap();
    let (code, value) = run_request_json(&text, false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 243);
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["method"],
        "patch:exact_ellipse_convex_polygons_disjoint_circles_union"
    );
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["is_exact"],
        true
    );
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["reported_error_bound_angstrom2"],
        0.0
    );
}

#[test]
fn overlapping_rectangular_patch_regions_use_exact_union_area() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [200.0, 200.0, 100.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 60.0,
                "regions": [
                    {"name": "left", "role": "patch", "geometry": {"shape": "rectangle", "center_angstrom": [-15.0, 0.0], "size_angstrom": [60.0, 60.0]}},
                    {"name": "right", "role": "patch", "geometry": {"shape": "rectangle", "center_angstrom": [15.0, 0.0], "size_angstrom": [60.0, 60.0]}}
                ],
                "composition": [{"lipid": "POPC"}]
            }]
        }],
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 90);
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["method"],
        "patch:exact_axis_aligned_rectangle_union"
    );
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["is_exact"],
        true
    );
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["reported_error_bound_angstrom2"],
        0.0
    );
}

#[test]
fn convex_polygon_region_union_uses_exact_intersection_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[-30.0, -20.0], [10.0, -20.0], [10.0, 20.0], [-30.0, 20.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[-10.0, -20.0], [30.0, -20.0], [30.0, 20.0], [-10.0, 20.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    assert_eq!(estimate.method, "exact_convex_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - 2400.0).abs() < 1.0e-3);
}

#[test]
fn overlapping_polygon_patch_regions_use_exact_union_area() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 80.0]},
        "membranes": [{
            "name": "poly",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 100.0,
                "regions": [
                    {"name": "left", "role": "patch", "geometry": {"shape": "polygon", "points_angstrom": [[-30.0, -20.0], [10.0, -20.0], [10.0, 20.0], [-30.0, 20.0]]}},
                    {"name": "right", "role": "patch", "geometry": {"shape": "polygon", "points_angstrom": [[-10.0, -20.0], [30.0, -20.0], [30.0, 20.0], [-10.0, 20.0]]}}
                ],
                "composition": [{"lipid": "LIP", "charge_e": 0.0}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("m.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["LIP"], 24);
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["method"],
        "patch:exact_convex_polygon_union"
    );
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["reported_error_bound_angstrom2"],
        0.0
    );
}

#[test]
fn disjoint_nonconvex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-40.0, -20.0],
                [-10.0, -20.0],
                [-10.0, -10.0],
                [-30.0, -10.0],
                [-30.0, 20.0],
                [-40.0, 20.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [10.0, -20.0],
                [40.0, -20.0],
                [40.0, -10.0],
                [20.0, -10.0],
                [20.0, 20.0],
                [10.0, 20.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    assert_eq!(estimate.method, "exact_disjoint_simple_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - 1200.0).abs() < 1.0e-3);
}

#[test]
fn overlapping_nonconvex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -100.0,
        xmax: 100.0,
        ymin: -100.0,
        ymax: 100.0,
    };
    let left = LeafletRegion {
        name: Some("left-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-30.0, -20.0],
                [0.0, -20.0],
                [0.0, -10.0],
                [-20.0, -10.0],
                [-20.0, 20.0],
                [-30.0, 20.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right-l".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-10.0, -20.0],
                [20.0, -20.0],
                [20.0, -10.0],
                [0.0, -10.0],
                [0.0, 20.0],
                [-10.0, 20.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&left, &right], bounds);
    assert_eq!(estimate.method, "exact_simple_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - 1100.0).abs() < 1.0e-2);
}

#[test]
fn disjoint_mixed_region_union_uses_exact_area() {
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
            center_angstrom: [-70.0, 0.0],
            radius_angstrom: 10.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-30.0, 0.0],
            radius_angstrom: [10.0, 5.0],
            rotate_degrees: 20.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("nonconvex".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [0.0, -20.0],
                [30.0, -20.0],
                [30.0, -10.0],
                [10.0, -10.0],
                [10.0, 20.0],
                [0.0, 20.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rotated-rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [65.0, 0.0],
            size_angstrom: [20.0, 10.0],
            rotate_degrees: 30.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &ellipse, &polygon, &rectangle], bounds);
    let expected = std::f32::consts::PI * (10.0 * 10.0 + 10.0 * 5.0) + 600.0 + 200.0;
    assert_eq!(estimate.method, "exact_disjoint_mixed_region_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 1.0e-3);
}

#[test]
fn disjoint_mixed_region_union_uses_exact_pair_proof_for_overlapping_bounds() {
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
        name: Some("corner-square".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[8.0, 8.0], [12.0, 8.0], [12.0, 12.0], [8.0, 12.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-40.0, 0.0],
            radius_angstrom: [6.0, 4.0],
            rotate_degrees: 25.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &polygon, &ellipse], bounds);
    let expected = std::f32::consts::PI * 10.0 * 10.0
        + polygon_area(&transformed_polygon_points(&polygon))
        + std::f32::consts::PI * 6.0 * 4.0;

    assert!(axis_aligned_bounds_overlap(
        region_bounds(&circle).unwrap(),
        region_bounds(&polygon).unwrap()
    ));
    assert_eq!(estimate.method, "exact_disjoint_mixed_region_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 1.0e-3);
}

#[test]
fn rectangle_simple_nonconvex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -50.0,
        xmax: 50.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [0.0, 0.0],
            size_angstrom: [16.0, 8.0],
            rotate_degrees: 0.0,
        },
    };
    let l_shape = LeafletRegion {
        name: Some("l-shape".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-4.0, -8.0],
                [12.0, -8.0],
                [12.0, -4.0],
                [0.0, -4.0],
                [0.0, 8.0],
                [-4.0, 8.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&rectangle, &l_shape], bounds);

    assert_eq!(estimate.method, "exact_rectangle_simple_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - 208.0).abs() < 1.0e-3);
}

#[test]
fn rotated_rectangle_simple_nonconvex_polygon_region_union_uses_exact_area() {
    let bounds = LayoutBounds {
        xmin: -50.0,
        xmax: 50.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let rectangle = LeafletRegion {
        name: Some("rotated-rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [1.0, -1.0],
            size_angstrom: [18.0, 8.0],
            rotate_degrees: 28.0,
        },
    };
    let l_shape = LeafletRegion {
        name: Some("l-shape".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-6.0, -9.0],
                [11.0, -9.0],
                [11.0, -4.0],
                [-1.0, -4.0],
                [-1.0, 9.0],
                [-6.0, 9.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&rectangle, &l_shape], bounds);
    let rectangle_points = convex_polygon_for_region(&rectangle).unwrap();
    let polygon_points = transformed_polygon_points(&l_shape);
    let expected =
        exact_simple_polygon_union_area_from_polygons(&[rectangle_points, polygon_points], bounds)
            .unwrap();

    assert_eq!(estimate.method, "exact_rectangle_simple_polygon_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - expected).abs() < 1.0e-3);
}

#[test]
fn component_mixed_region_union_uses_exact_area_per_overlap_group() {
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
            center_angstrom: [-50.0, 0.0],
            radius_angstrom: 15.0,
        },
    };
    let rectangle = LeafletRegion {
        name: Some("rectangle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [-40.0, 0.0],
            size_angstrom: [30.0, 20.0],
            rotate_degrees: 0.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [35.0, 0.0],
            radius_angstrom: [12.0, 8.0],
            rotate_degrees: 15.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("diamond".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[35.0, 12.0], [47.0, 0.0], [35.0, -12.0], [23.0, 0.0]],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &rectangle, &ellipse, &polygon], bounds);
    let left_expected =
        exact_circle_axis_aligned_rectangle_region_union_area(&[&circle, &rectangle], bounds)
            .unwrap();
    let right_expected =
        exact_ellipse_convex_polygon_region_union_area(&[&ellipse, &polygon], bounds).unwrap();

    assert_eq!(estimate.method, "exact_component_mixed_region_union");
    assert_eq!(estimate.reported_error_bound_angstrom2, Some(0.0));
    assert!((estimate.area_angstrom2 - (left_expected + right_expected)).abs() < 1.0e-3);
}
