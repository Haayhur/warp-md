use super::*;

#[test]
fn geo_backend_unions_overlapping_protein_multipolygon_insets() {
    let boundary = ProteinBoundaryGeometry::MultiPolygon {
        polygons: vec![
            vec![[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]],
            vec![[0.0, -10.0], [20.0, -10.0], [20.0, 10.0], [0.0, 10.0]],
        ],
        inset_angstrom: 2.0,
    };

    let (area, exact) = boundary.area_estimate();

    assert!(exact);
    assert!((area - 416.0).abs() < 1.0e-3, "area={area}");
}

#[test]
fn geo_backend_subtracts_overlapping_nested_polygon_holes_once() {
    let outer = vec![[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]];
    let left_hole = vec![[-9.0, -6.0], [3.0, -6.0], [3.0, 6.0], [-9.0, 6.0]];
    let right_hole = vec![[-3.0, -6.0], [9.0, -6.0], [9.0, 6.0], [-3.0, 6.0]];
    let boundary = ProteinBoundaryGeometry::NestedPolygons {
        outer,
        holes: vec![left_hole, right_hole],
        inset_angstrom: 0.0,
    };

    let (area, exact) = boundary.area_estimate();

    assert!(exact);
    assert!((area - 1384.0).abs() < 1.0e-3, "area={area}");
}

#[test]
fn geo_backend_replaces_grid_for_overlapping_mixed_curve_polygon_union() {
    let bounds = LayoutBounds {
        xmin: -18.0,
        xmax: 18.0,
        ymin: -14.0,
        ymax: 14.0,
    };
    let circle = LeafletRegion {
        name: Some("circle".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: [-3.0, 0.0],
            radius_angstrom: 9.0,
        },
    };
    let ellipse = LeafletRegion {
        name: Some("ellipse".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Ellipse {
            center_angstrom: [4.0, 1.0],
            radius_angstrom: [10.0, 5.0],
            rotate_degrees: 27.0,
        },
    };
    let polygon = LeafletRegion {
        name: Some("notched".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Polygon {
            points_angstrom: vec![
                [-12.0, -8.0],
                [10.0, -8.0],
                [10.0, -2.0],
                [1.0, -2.0],
                [1.0, 8.0],
                [-12.0, 8.0],
            ],
            scale_xy: None,
            rotate_degrees: 9.0,
        },
    };

    let estimate = region_union_area_angstrom2(&[&circle, &ellipse, &polygon], bounds);

    assert_eq!(estimate.method, "geo_polygonized_boolean_region_union");
    assert!(!estimate.is_exact);
    assert!(
        estimate.area_angstrom2 > 250.0,
        "area={}",
        estimate.area_angstrom2
    );
    assert!(
        estimate.area_angstrom2 < 700.0,
        "area={}",
        estimate.area_angstrom2
    );
}
