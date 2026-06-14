use super::*;

#[test]
fn stacked_membranes_expand_to_membranes_solvent_zones_and_box_height() {
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [200.0, 100.0, 1.0]},
        "membranes": [],
        "stacked_membranes": [{
            "name": "three_bilayers",
            "pbc": "split",
            "distance_angstrom": [30.0, 50.0, 50.0],
            "distance_type": ["surface"],
            "layers": [
                {
                    "membrane": {
                        "name": "outer_a",
                        "leaflets": [
                            {"name": "upper", "side": "upper", "composition": [{"lipid": "POPC", "fraction": 5.0}, {"lipid": "CHOL", "fraction": 1.0}]},
                            {"name": "lower", "side": "lower", "composition": [{"lipid": "POPC", "fraction": 5.0}, {"lipid": "CHOL", "fraction": 1.0}]}
                        ]
                    },
                    "solvent": {"name": "pbc_gap"}
                },
                {
                    "membrane": {
                        "name": "middle",
                        "leaflets": [
                            {"name": "upper", "side": "upper", "composition": [{"lipid": "POPE", "fraction": 3.0}, {"lipid": "CHOL", "fraction": 2.0}]},
                            {"name": "lower", "side": "lower", "composition": [{"lipid": "POPE", "fraction": 3.0}, {"lipid": "CHOL", "fraction": 2.0}]}
                        ]
                    },
                    "solvent": {"name": "middle_gap_a"}
                },
                {
                    "membrane": {
                        "name": "outer_b",
                        "leaflets": [
                            {"name": "upper", "side": "upper", "composition": [{"lipid": "POPC", "fraction": 5.0}, {"lipid": "CHOL", "fraction": 1.0}]},
                            {"name": "lower", "side": "lower", "composition": [{"lipid": "POPC", "fraction": 5.0}, {"lipid": "CHOL", "fraction": 1.0}]}
                        ]
                    },
                    "solvent": {"name": "middle_gap_b"}
                }
            ]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false, "molarity_mol_l": 0.0}
        },
        "outputs": {"manifest": "unused.json"}
    });

    let (code, value) = validate_request_json(&serde_json::to_string(&request).unwrap());
    assert_eq!(code, 0, "{value}");
    let normalized = &value["normalized_request"];
    assert_eq!(normalized["membranes"].as_array().unwrap().len(), 3);
    assert_eq!(
        normalized["environment"]["solvent"]["zones"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert!(normalized["environment"]["solvent"]["enabled"]
        .as_bool()
        .unwrap());
    let box_z = normalized["system"]["box_size_angstrom"][2]
        .as_f64()
        .unwrap();
    assert!(box_z > 0.0);
    assert_eq!(
        normalized["membranes"][1]["name"].as_str().unwrap(),
        "three_bilayers:middle"
    );
    let center_z = normalized["membranes"][1]["center_z_angstrom"]
        .as_f64()
        .unwrap();
    assert!(center_z.abs() < box_z * 0.5);
}

#[test]
fn solvent_per_lipid_counts_lipids_inside_zone_by_reference_rule() {
    let temp = tempfile::tempdir().unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [{
            "name": "small",
            "leaflets": [
                {
                    "name": "upper",
                    "side": "upper",
                    "composition": [{"lipid": "POPC", "count": 2}]
                },
                {
                    "name": "lower",
                    "side": "lower",
                    "composition": [{"lipid": "POPC", "count": 2}]
                }
            ]
        }],
        "environment": {
            "ions": {"neutralize": false, "salt_molarity_mol_l": 0.0},
            "solvent": {
                "enabled": true,
                "molarity_mol_l": 0.0,
                "solvent_per_lipid": 2.0,
                "box_size_angstrom": [80.0, 80.0, 80.0],
                "center_angstrom": [0.0, 0.0, 0.0]
            }
        },
        "outputs": {"manifest": temp.path().join("m.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 4);
    assert_eq!(value["summary"]["solvent_counts"]["W"], 8);
}

#[test]
fn protein_boundary_clips_lipid_counts_and_coordinates() {
    let temp = tempfile::tempdir().unwrap();
    let gro = temp.path().join("m.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [100.0, 100.0, 80.0]},
        "membranes": [{
            "name": "disc",
            "protein_boundary": {
                "mode": "inside",
                "center_angstrom": [0.0, 0.0],
                "radius_angstrom": 20.0
            },
            "leaflets": [
                {"name": "upper", "side": "upper", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]},
                {"name": "lower", "side": "lower", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]}
            ]
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {
            "coordinates": gro,
            "manifest": temp.path().join("m.json")
        }
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 50);
    let atoms = read_gro_positions(&gro);
    assert!(!atoms.is_empty());
    for [x, y, _z] in atoms {
        assert!(
            x * x + y * y <= 20.1_f32.powi(2),
            "lipid outside boundary at {x},{y}"
        );
    }
}

#[test]
fn protein_boundary_can_be_derived_from_inserted_protein_coordinates() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("ring.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  A   RIM A   1      10.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  B   RIM A   1     -10.000   0.000   0.000  1.00  0.00           C\n\
ATOM      3  C   RIM A   1       0.000  10.000   0.000  1.00  0.00           C\n\
ATOM      4  D   RIM A   1       0.000 -10.000   0.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [{
            "name": "disc",
            "protein_boundary": {"mode": "inside", "protein": "RIM"},
            "leaflets": [
                {"name": "upper", "side": "upper", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]},
                {"name": "lower", "side": "lower", "apl_angstrom2": 50.0, "composition": [{"lipid": "POPC"}]}
            ]
        }],
        "proteins": [{
            "name": "RIM",
            "coordinates": protein_path,
            "format": "pdb",
            "net_charge_e": 0.0,
            "placement": {"center_method": "cog", "center_angstrom": [20.0, 0.0, 0.0]}
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("m.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["POPC"], 12);
    assert_eq!(value["summary"]["inserted_counts"]["RIM"], 1);
}

#[test]
fn protein_boundary_convex_hull_clips_non_circular_coordinates() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("bar.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  A   BAR A   1     -30.000  -5.000   0.000  1.00  0.00           C\n\
ATOM      2  B   BAR A   1      30.000  -5.000   0.000  1.00  0.00           C\n\
ATOM      3  C   BAR A   1      30.000   5.000   0.000  1.00  0.00           C\n\
ATOM      4  D   BAR A   1     -30.000   5.000   0.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let gro = temp.path().join("m.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0], "placement": {"relaxation": true, "max_steps": 80}},
        "membranes": [{
            "name": "disc",
            "protein_boundary": {"mode": "inside", "geometry": "convex_hull", "protein": "BAR"},
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 100.0,
                "composition": [{"lipid": "LIP", "charge_e": 0.0}]
            }]
        }],
        "proteins": [{
            "name": "BAR",
            "coordinates": protein_path,
            "format": "pdb",
            "net_charge_e": 0.0,
            "placement": {"center_method": "cog", "center_angstrom": [0.0, 0.0, 0.0]}
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"coordinates": gro, "manifest": temp.path().join("m.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["lipid_counts"]["LIP"], 6);
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["method"],
        "protein_boundary_area"
    );
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["is_exact"],
        true
    );
    let lipids = read_gro_residue_positions(&gro, "LIP");
    assert_eq!(lipids.len(), 6);
    for [x, y, _z] in lipids {
        assert!(x.abs() <= 30.1, "lipid outside hull x bounds at {x},{y}");
        assert!(y.abs() <= 5.1, "lipid outside hull y bounds at {x},{y}");
    }
}

#[test]
fn polygon_protein_boundary_enforces_buffer_and_point_radius() {
    let boundary = ProteinBoundaryGeometry::Polygon {
        points: vec![[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
        inset_angstrom: 2.0,
    };

    assert!(boundary.contains_point_with_margin([0.0, 0.0], 1.0));
    assert!(!boundary.contains_point_with_margin([8.5, 0.0], 0.0));
    assert!(!boundary.contains_point_with_margin([7.5, 0.0], 1.0));

    let projected = boundary.project_point([9.5, 0.0], 1.0);

    assert!(boundary.contains_point_with_margin(projected, 1.0));
    assert!(
        polygon_boundary_distance(
            projected,
            &[[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]
        ) >= 3.0 - 1.0e-4
    );
    assert_eq!(boundary.area_estimate().0, 256.0);

    let unbuffered = ProteinBoundaryGeometry::Polygon {
        points: vec![[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
        inset_angstrom: 0.0,
    };
    assert!(unbuffered.contains_point_with_margin([10.0, 0.0], 0.0));
    assert!(unbuffered.contains_point_with_margin([10.0, 0.0], 0.5));

    let collapsed = ProteinBoundaryGeometry::Polygon {
        points: vec![[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
        inset_angstrom: 12.0,
    };
    assert_eq!(collapsed.area_estimate().0, 0.0);

    let non_rect_points = [[-10.0, -10.0], [10.0, -10.0], [8.0, 10.0], [-10.0, 10.0]];
    let non_rect = ProteinBoundaryGeometry::Polygon {
        points: non_rect_points.to_vec(),
        inset_angstrom: 2.0,
    };
    assert!(exact_axis_aligned_rectangle_polygon_inset_area(&non_rect_points, 2.0).is_none());
    assert!(exact_convex_polygon_inset_area(&non_rect_points, 2.0).is_some());
    let (non_rect_area, non_rect_exact) = non_rect.area_estimate();
    assert!(non_rect_exact);
    assert!(non_rect_area > 0.0);
    assert!(non_rect_area < polygon_area(&non_rect_points));

    let concave_points = [
        [-10.0, -10.0],
        [10.0, -10.0],
        [10.0, 10.0],
        [0.0, 0.0],
        [-10.0, 10.0],
    ];
    let concave = ProteinBoundaryGeometry::Polygon {
        points: concave_points.to_vec(),
        inset_angstrom: 2.0,
    };
    assert!(exact_convex_polygon_inset_area(&concave_points, 2.0).is_none());
    let (concave_area, concave_exact) = concave.area_estimate();
    assert!(concave_exact);
    assert!(concave_area > 0.0);
    assert!(concave_area < polygon_area(&concave_points));
}

#[test]
fn multipolygon_protein_boundary_preserves_disconnected_footprints() {
    let left = vec![[-15.0, -5.0], [-5.0, -5.0], [-5.0, 5.0], [-15.0, 5.0]];
    let right = vec![[5.0, -5.0], [15.0, -5.0], [15.0, 5.0], [5.0, 5.0]];
    let boundary = ProteinBoundaryGeometry::MultiPolygon {
        polygons: vec![left.clone(), right.clone()],
        inset_angstrom: 1.0,
    };

    let (area, exact) = boundary.area_estimate();
    assert!(exact);
    assert!((area - 128.0).abs() < 1.0e-3);
    assert!(boundary.contains_point_with_margin([-10.0, 0.0], 1.0));
    assert!(boundary.contains_point_with_margin([10.0, 0.0], 1.0));
    assert!(!boundary.contains_point_with_margin([0.0, 0.0], 0.0));

    let projected = boundary.project_point([0.0, 0.0], 1.0);
    assert!(boundary.contains_point_with_margin(projected, 1.0));
    assert!(
        squared_distance2(projected, [-10.0, 0.0]) < 25.0
            || squared_distance2(projected, [10.0, 0.0]) < 25.0
    );
    assert_eq!(boundary.bounds(), (-15.0, 15.0, -5.0, 5.0));
}

#[test]
fn overlapping_multipolygon_protein_boundary_uses_exact_union_area() {
    let left = vec![[-10.0, -5.0], [0.0, -5.0], [0.0, 5.0], [-10.0, 5.0]];
    let right = vec![[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]];
    let boundary = ProteinBoundaryGeometry::MultiPolygon {
        polygons: vec![left.clone(), right.clone()],
        inset_angstrom: 0.0,
    };

    let (area, exact) = boundary.area_estimate();

    assert!(exact);
    assert!((area - 150.0).abs() < 1.0e-3);
    assert!(
        (exact_simple_polygon_union_area_from_polygons(
            &[left, right],
            LayoutBounds {
                xmin: -10.0,
                xmax: 5.0,
                ymin: -5.0,
                ymax: 5.0,
            }
        )
        .unwrap()
            - area)
            .abs()
            < 1.0e-3
    );
}

#[test]
fn buffered_overlapping_multipolygon_protein_boundary_uses_union_estimate() {
    let left = vec![
        [-10.0, -5.0],
        [0.0, -5.0],
        [0.0, 0.0],
        [-5.0, 0.0],
        [-5.0, 5.0],
        [-10.0, 5.0],
    ];
    let right = vec![[-7.0, -2.0], [3.0, -2.0], [3.0, 8.0], [-7.0, 8.0]];
    let boundary = ProteinBoundaryGeometry::MultiPolygon {
        polygons: vec![left.clone(), right.clone()],
        inset_angstrom: 1.0,
    };

    let (area, exact) = boundary.area_estimate();
    let raw_component_area = polygon_area(&left) + polygon_area(&right);
    let summed_components =
        polygon_area_with_inset(&left, 1.0) + polygon_area_with_inset(&right, 1.0);
    let union_estimate = multipolygon_area_with_inset_union(&[left, right], 1.0);

    assert!(!exact);
    assert!(area < raw_component_area);
    assert!((area - summed_components).abs() > 1.0e-3);
    assert!((area - union_estimate).abs() > 1.0e-3);
}

#[test]
fn buffered_overlapping_convex_multipolygon_protein_boundary_uses_exact_union_area() {
    let left = vec![[-10.0, -5.0], [0.0, -5.0], [0.0, 5.0], [-10.0, 5.0]];
    let right = vec![[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]];
    let boundary = ProteinBoundaryGeometry::MultiPolygon {
        polygons: vec![left.clone(), right.clone()],
        inset_angstrom: 1.0,
    };

    let (area, exact) = boundary.area_estimate();
    let left_inset = convex_polygon_inset_polygon(&left, 1.0).unwrap();
    let right_inset = convex_polygon_inset_polygon(&right, 1.0).unwrap();
    let expected = exact_simple_polygon_union_area_from_polygons(
        &[left_inset, right_inset],
        LayoutBounds {
            xmin: -9.0,
            xmax: 4.0,
            ymin: -4.0,
            ymax: 4.0,
        },
    )
    .unwrap();

    assert!(exact);
    assert!((area - expected).abs() < 1.0e-3);
    assert!((area - 104.0).abs() < 1.0e-3);
}

#[test]
fn nested_polygon_protein_boundary_classifies_inner_holes() {
    let outer = vec![[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]];
    let hole = vec![[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]];
    let boundary = ProteinBoundaryGeometry::NestedPolygons {
        outer: outer.clone(),
        holes: vec![hole.clone()],
        inset_angstrom: 0.0,
    };

    let (area, exact) = boundary.area_estimate();
    assert!(exact);
    assert!((area - 1500.0).abs() < 1.0e-3);
    assert!(boundary.contains_point([-10.0, 0.0]));
    assert!(!boundary.contains_point([0.0, 0.0]));
    assert!(!boundary.contains_point([30.0, 0.0]));

    let projected = boundary.project_point([0.0, 0.0], 0.0);
    assert!(boundary.contains_point(projected));
    assert!(!point_in_polygon(projected, &hole));

    let (nested_outer, nested_holes) =
        nested_polygon_from_components(&[outer.clone(), hole.clone()]).unwrap();
    assert_eq!(nested_outer, outer);
    assert_eq!(nested_holes, vec![hole]);
}

#[test]
fn nested_polygon_forest_classifies_multi_level_rings() {
    let outer = vec![[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]];
    let hole = vec![[-12.0, -12.0], [12.0, -12.0], [12.0, 12.0], [-12.0, 12.0]];
    let island = vec![[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]];
    let rings =
        nested_polygon_forest_from_components(&[outer.clone(), island.clone(), hole.clone()])
            .unwrap();
    let boundary = ProteinBoundaryGeometry::NestedPolygonForest {
        rings,
        inset_angstrom: 0.0,
    };

    let (area, exact) = boundary.area_estimate();
    assert!(exact);
    assert!((area - (1600.0 - 576.0 + 64.0)).abs() < 1.0e-3);
    assert!(boundary.contains_point([-16.0, 0.0]));
    assert!(!boundary.contains_point([8.0, 0.0]));
    assert!(boundary.contains_point([0.0, 0.0]));
    assert!(!boundary.contains_point([30.0, 0.0]));
    assert!(boundary.contains_point([-20.0, 0.0]));
    assert!(!boundary.contains_point([12.0, 0.0]));
    assert!(boundary.contains_point([4.0, 0.0]));
    assert!(boundary.contains_point(boundary.project_point([8.0, 0.0], 0.0)));
}

#[test]
fn rectangular_nested_polygon_forest_has_exact_buffered_area() {
    let outer = vec![[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]];
    let hole = vec![[-12.0, -12.0], [12.0, -12.0], [12.0, 12.0], [-12.0, 12.0]];
    let island = vec![[-4.0, -4.0], [4.0, -4.0], [4.0, 4.0], [-4.0, 4.0]];
    let rings =
        nested_polygon_forest_from_components(&[outer.clone(), island.clone(), hole.clone()])
            .unwrap();
    let boundary = ProteinBoundaryGeometry::NestedPolygonForest {
        rings: rings.clone(),
        inset_angstrom: 1.0,
    };

    let (area, exact) = boundary.area_estimate();
    let expected =
        38.0 * 38.0 - rounded_rectangle_dilation_area((-12.0, 12.0, -12.0, 12.0), 1.0) + 6.0 * 6.0;

    assert!(exact);
    assert!((area - expected).abs() < 1.0e-3);
    assert_eq!(
        exact_axis_aligned_rectangle_nested_forest_inset_area(
            &rings,
            &nested_polygon_ring_depths(&rings),
            4.1
        ),
        None
    );
    assert!(boundary.contains_point([-15.0, 0.0]));
    assert!(!boundary.contains_point([10.5, 0.0]));
    assert!(boundary.contains_point([0.0, 0.0]));
}

#[test]
fn convex_nested_polygon_forest_has_exact_buffered_area() {
    let outer = vec![[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]];
    let hole = vec![[0.0, 20.0], [20.0, 0.0], [0.0, -20.0], [-20.0, 0.0]];
    let island = vec![[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]];
    let rings =
        nested_polygon_forest_from_components(&[outer.clone(), island.clone(), hole.clone()])
            .unwrap();
    let boundary = ProteinBoundaryGeometry::NestedPolygonForest {
        rings: rings.clone(),
        inset_angstrom: 2.0,
    };

    let (area, exact) = boundary.area_estimate();
    let expected_hole_area =
        polygon_area(&hole) + polygon_perimeter(&hole) * 2.0 + std::f32::consts::PI * 4.0;
    let expected = 56.0 * 56.0 - expected_hole_area + 6.0 * 6.0;

    assert!(exact);
    assert!((area - expected).abs() < 1.0e-3);
    assert_eq!(
        exact_convex_nested_forest_inset_area(&rings, &nested_polygon_ring_depths(&rings), 4.0),
        None
    );
    assert!(boundary.contains_point([-25.0, 0.0]));
    assert!(!boundary.contains_point([10.0, 0.0]));
    assert!(boundary.contains_point([0.0, 0.0]));
}

#[test]
fn simple_nested_polygon_forest_has_exact_buffered_area_before_topology_event() {
    let outer = vec![[-40.0, -40.0], [40.0, -40.0], [40.0, 40.0], [-40.0, 40.0]];
    let hole = vec![[-22.0, -22.0], [22.0, -22.0], [22.0, 22.0], [-22.0, 22.0]];
    let island = vec![
        [-8.0, -8.0],
        [8.0, -8.0],
        [8.0, -4.0],
        [-4.0, -4.0],
        [-4.0, 8.0],
        [-8.0, 8.0],
    ];
    let rings =
        nested_polygon_forest_from_components(&[outer.clone(), island.clone(), hole.clone()])
            .unwrap();
    let boundary = ProteinBoundaryGeometry::NestedPolygonForest {
        rings: rings.clone(),
        inset_angstrom: 1.0,
    };

    let (area, exact) = boundary.area_estimate();
    let depths = nested_polygon_ring_depths(&rings);
    let expected = exact_simple_polygon_inset_area_before_topology_event(&outer, 1.0).unwrap()
        - (polygon_area(&hole) + polygon_perimeter(&hole) * 1.0 + std::f32::consts::PI)
        + exact_simple_polygon_inset_area_before_topology_event(&island, 1.0).unwrap();

    assert!(exact);
    assert!((area - expected).abs() < 1.0e-3);
    assert!(exact_axis_aligned_rectangle_nested_forest_inset_area(&rings, &depths, 1.0).is_none());
    assert!(exact_convex_nested_forest_inset_area(&rings, &depths, 1.0).is_none());
    assert!(
        (exact_simple_nested_forest_inset_area_before_topology_event(&rings, &depths, 1.0)
            .unwrap()
            - expected)
            .abs()
            < 1.0e-3
    );
}
