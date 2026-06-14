use super::*;

#[test]
fn mixed_region_union_uses_geo_boolean_backend_before_grid_fallback() {
    let bounds = LayoutBounds {
        xmin: -15.0,
        xmax: 15.0,
        ymin: -14.0,
        ymax: 16.0,
    };
    let left = LeafletRegion {
        name: Some("left".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [-5.0, 0.0],
            radius_angstrom: [20.0, 10.0],
            rotate_degrees: 15.0,
        },
    };
    let right = LeafletRegion {
        name: Some("right".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-2.0, -8.0],
                [18.0, -8.0],
                [18.0, -2.0],
                [4.0, -2.0],
                [4.0, 14.0],
                [-2.0, 14.0],
            ],
            scale_xy: None,
            rotate_degrees: 0.0,
        },
    };
    let top = LeafletRegion {
        name: Some("top".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [5.0, 8.0],
            radius_angstrom: 9.0,
        },
    };

    let regions = [&left, &right, &top];
    let estimate = region_union_area_angstrom2(&regions, bounds);
    let domain = region_union_bounds(&regions).unwrap();
    let clipped_domain = clipped_bounds(domain, bounds).unwrap();
    let fine = grid_region_union_area(
        &regions,
        clipped_domain.0,
        clipped_domain.1,
        clipped_domain.2,
        clipped_domain.3,
        REGION_UNION_FINE_GRID_SPACING_ANGSTROM,
    );

    assert_eq!(estimate.method, "geo_polygonized_boolean_region_union");
    assert!(!estimate.is_exact);
    assert!(estimate.reported_error_bound_angstrom2.is_some());
    assert!(estimate.area_angstrom2 > 0.0);
    assert!((estimate.area_angstrom2 - fine).abs() > 1.0e-3);
}

#[test]
fn overlapping_circle_holes_use_exact_union_area() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 80.0]},
        "membranes": [{
            "name": "circle-holes",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 100.0,
                "regions": [
                    {"name": "left", "role": "hole", "geometry": {"shape": "circle", "center_angstrom": [-5.0, 0.0], "radius_angstrom": 10.0}},
                    {"name": "right", "role": "hole", "geometry": {"shape": "circle", "center_angstrom": [5.0, 0.0], "radius_angstrom": 10.0}}
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
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["method"],
        "hole:exact_circle_union"
    );
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["reported_error_bound_angstrom2"],
        0.0
    );
    assert_eq!(value["summary"]["lipid_counts"]["LIP"], 95);
}

#[test]
fn overlapping_region_holes_subtract_union_area_once() {
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
                "apl_angstrom2": 100.0,
                "regions": [
                    {"name": "left-hole", "role": "hole", "geometry": {"shape": "rectangle", "center_angstrom": [-10.0, 0.0], "size_angstrom": [40.0, 40.0]}},
                    {"name": "right-hole", "role": "hole", "geometry": {"shape": "rectangle", "center_angstrom": [10.0, 0.0], "size_angstrom": [40.0, 40.0]}}
                ],
                "composition": [{"lipid": "POPC"}]
            }]
        }],
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 76);
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["method"],
        "hole:exact_axis_aligned_rectangle_union"
    );
}

#[test]
fn membrane_xy_domain_limits_apl_counts_and_layout_bounds() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("domain.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [200.0, 200.0, 100.0], "placement": {"relaxation": false}},
        "membranes": [{
            "name": "top-right-domain",
            "center_xy_angstrom": [50.0, 50.0],
            "size_xy_angstrom": [100.0, 100.0],
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 60.0,
                "composition": [{"lipid": "POPC"}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"coordinates": gro, "manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 167);

    let text = std::fs::read_to_string(gro).unwrap();
    for line in text.lines().skip(2).take(167) {
        let x = line[20..28].trim().parse::<f32>().unwrap() * 10.0;
        let y = line[28..36].trim().parse::<f32>().unwrap() * 10.0;
        assert!((0.0..=100.0).contains(&x), "x {x}");
        assert!((0.0..=100.0).contains(&y), "y {y}");
    }
}

#[test]
fn polygon_region_scaling_changes_area_about_center() {
    let region = LeafletRegion {
        name: Some("scaled".into()),
        role: "patch".into(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]],
            scale_xy: Some([3.0, 2.0]),
            rotate_degrees: 45.0,
        },
    };
    assert!((region_area_estimate(&region).area_angstrom2 - 300.0).abs() < 1.0e-3);
}

#[test]
fn periodic_leaflet_regions_wrap_non_circular_constraints() {
    let bounds = LayoutBounds {
        xmin: -50.0,
        xmax: 50.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let periodicity = LayoutPeriodicity { x: true, y: false };
    let patch = LeafletRegion {
        name: Some("edge-patch".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Rectangle {
            center_angstrom: [45.0, 0.0],
            size_angstrom: [20.0, 20.0],
            rotate_degrees: 0.0,
        },
    };
    let hole = LeafletRegion {
        name: Some("edge-hole".to_string()),
        role: "hole".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [45.0, 0.0],
            radius_angstrom: [10.0, 8.0],
            rotate_degrees: 20.0,
        },
    };

    let patch_leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: None,
        exclusions: Vec::new(),
        regions: vec![patch],
        composition: Vec::new(),
    };
    assert!(!leaflet_allows_point(&patch_leaflet, [-49.0, 0.0]));
    assert!(leaflet_allows_point_periodic(
        &patch_leaflet,
        [-49.0, 0.0],
        bounds,
        periodicity
    ));

    let hole_leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: None,
        exclusions: Vec::new(),
        regions: vec![hole],
        composition: Vec::new(),
    };
    assert!(leaflet_allows_point(&hole_leaflet, [-49.0, 0.0]));
    assert!(!leaflet_allows_point_periodic(
        &hole_leaflet,
        [-49.0, 0.0],
        bounds,
        periodicity
    ));
}

#[test]
fn periodic_leaflet_projection_wraps_non_circular_patch_boundary() {
    let bounds = LayoutBounds {
        xmin: -50.0,
        xmax: 50.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let periodicity = LayoutPeriodicity { x: true, y: false };
    let leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: None,
        exclusions: Vec::new(),
        regions: vec![LeafletRegion {
            name: Some("edge-patch".to_string()),
            role: "patch".to_string(),
            geometry: RegionGeometry::Rectangle {
                center_angstrom: [45.0, 0.0],
                size_angstrom: [20.0, 20.0],
                rotate_degrees: 0.0,
            },
        }],
        composition: Vec::new(),
    };
    let point = LayoutPoint {
        x: -35.0,
        y: 0.0,
        radius: 1.0,
    };

    let projected = analytic_allowed_leaflet_projection_with_boundary(
        point,
        &leaflet,
        bounds,
        periodicity,
        None,
        None,
    )
    .unwrap();

    assert!(point_inside_bounds_with_radius_periodic(
        &projected,
        bounds,
        periodicity
    ));
    assert!(leaflet_allows_point_periodic(
        &leaflet,
        [projected.x, projected.y],
        bounds,
        periodicity
    ));
}

#[test]
fn periodic_leaflet_geometry_diagnostics_wrap_non_circular_holes() {
    let bounds = LayoutBounds {
        xmin: -50.0,
        xmax: 50.0,
        ymin: -50.0,
        ymax: 50.0,
    };
    let periodicity = LayoutPeriodicity { x: true, y: false };
    let membrane = MembraneRequest {
        name: "geom".to_string(),
        center_xy_angstrom: None,
        size_xy_angstrom: None,
        center_z_angstrom: 0.0,
        solvate_voids: true,
        solvent_exclusion_half_thickness_angstrom:
            default_membrane_solvent_exclusion_half_thickness(),
        leaflets: Vec::new(),
        protein_boundary: None,
    };
    let leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: None,
        exclusions: Vec::new(),
        regions: vec![LeafletRegion {
            name: Some("edge-hole".to_string()),
            role: "hole".to_string(),
            geometry: RegionGeometry::Polygon {
                points_angstrom: vec![[40.0, -8.0], [60.0, -8.0], [60.0, 8.0], [40.0, 8.0]],
                scale_xy: None,
                rotate_degrees: 0.0,
            },
        }],
        composition: Vec::new(),
    };
    let points = vec![LayoutPoint {
        x: -45.0,
        y: 0.0,
        radius: 1.0,
    }];

    let diagnostics =
        leaflet_geometry_diagnostics(&membrane, &leaflet, &[], &points, bounds, periodicity, None)
            .unwrap();

    assert_eq!(diagnostics.constraint_count, 1);
    assert_eq!(diagnostics.violation_count, 1);
    assert_eq!(diagnostics.constraints[0].name, "edge-hole");
}

#[test]
fn triclinic_region_wrapping_finds_second_neighbor_patch_image() {
    let basis = LayoutBasis2D::new([0.0, 0.0], [10.0, 0.0], [5.0, 10.0]).unwrap();
    let bounds = LayoutBounds {
        xmin: 0.0,
        xmax: 15.0,
        ymin: 0.0,
        ymax: 10.0,
    };
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let region = LeafletRegion {
        name: Some("second-image".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                basis.cartesian([2.00, 0.46]),
                basis.cartesian([2.10, 0.46]),
                basis.cartesian([2.10, 0.54]),
                basis.cartesian([2.00, 0.54]),
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
}

#[test]
fn patch_regions_constrain_emitted_lipid_coordinates() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("patch.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 80.0], "placement": {"relaxation": false}},
        "membranes": [{
            "name": "patch",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "regions": [{
                    "name": "circle",
                    "role": "patch",
                    "geometry": {"shape": "circle", "center_angstrom": [20.0, 0.0], "radius_angstrom": 20.0}
                }],
                "composition": [{"lipid": "DUM", "count": 8}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"coordinates": gro, "manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    let geometry = &value["placement"]["leaflet_metrics"][0]["geometry"];
    assert_eq!(geometry["constraint_count"], 1);
    assert_eq!(geometry["violation_count"], 0);
    assert_eq!(geometry["constraints"][0]["role"], "patch");
    let text = std::fs::read_to_string(gro).unwrap();
    for line in text.lines().skip(2).take(8) {
        let x = line[20..28].trim().parse::<f32>().unwrap() * 10.0;
        let y = line[28..36].trim().parse::<f32>().unwrap() * 10.0;
        let dx = x - 20.0;
        assert!(dx * dx + y * y <= 20.0_f32.powi(2), "x {x} y {y}");
    }
}

#[test]
fn relaxed_non_circular_holes_remain_excluded() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("rectangle-hole.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {
            "box_size_angstrom": [100.0, 100.0, 80.0],
            "placement": {"relaxation": true, "max_steps": 50}
        },
        "membranes": [{
            "name": "holed",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "regions": [{
                    "name": "rect-hole",
                    "role": "hole",
                    "geometry": {
                        "shape": "rectangle",
                        "center_angstrom": [0.0, 0.0],
                        "size_angstrom": [30.0, 20.0]
                    }
                }],
                "composition": [{"lipid": "DUM", "count": 40}]
            }]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"coordinates": gro, "manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    let geometry = &value["placement"]["leaflet_metrics"][0]["geometry"];
    assert_eq!(geometry["constraint_count"], 1);
    assert_eq!(geometry["violation_count"], 0);
    assert_eq!(geometry["constraints"][0]["name"], "rect-hole");
    assert_eq!(geometry["constraints"][0]["role"], "hole");
    for [x, y, _z] in read_gro_positions(&gro) {
        assert!(
            !(x.abs() <= 15.0 && y.abs() <= 10.0),
            "relaxed lipid drifted into rectangle hole at {x},{y}"
        );
    }
}
