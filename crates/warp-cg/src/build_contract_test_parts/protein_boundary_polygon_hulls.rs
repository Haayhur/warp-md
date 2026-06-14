use super::*;

#[test]
fn rectangular_nested_polygon_boundary_has_exact_buffered_area() {
    let outer = vec![[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]];
    let left_hole = vec![[-10.0, -5.0], [-2.0, -5.0], [-2.0, 5.0], [-10.0, 5.0]];
    let right_hole = vec![[2.0, -5.0], [10.0, -5.0], [10.0, 5.0], [2.0, 5.0]];
    let boundary = ProteinBoundaryGeometry::NestedPolygons {
        outer: outer.clone(),
        holes: vec![left_hole.clone(), right_hole.clone()],
        inset_angstrom: 2.0,
    };

    let (area, exact) = boundary.area_estimate();
    let expected = 36.0 * 36.0
        - rounded_rectangle_dilation_area((-10.0, -2.0, -5.0, 5.0), 2.0)
        - rounded_rectangle_dilation_area((2.0, 10.0, -5.0, 5.0), 2.0);

    assert!(exact);
    assert!((area - expected).abs() < 1.0e-3);
    assert!(boundary.contains_point([-15.0, 0.0]));
    assert!(!boundary.contains_point([1.0, 0.0]));
    assert!(!boundary.contains_point([18.5, 0.0]));
    let exact_area =
        exact_axis_aligned_rectangle_nested_inset_area(&outer, &[left_hole, right_hole], 2.0)
            .unwrap();
    assert!((exact_area - expected).abs() < 1.0e-3);
}

#[test]
fn convex_nested_polygon_boundary_has_exact_buffered_area() {
    let outer = vec![[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]];
    let triangle_hole = vec![[0.0, 6.0], [6.0, -6.0], [-6.0, -6.0]];
    let boundary = ProteinBoundaryGeometry::NestedPolygons {
        outer: outer.clone(),
        holes: vec![triangle_hole.clone()],
        inset_angstrom: 2.0,
    };

    let (area, exact) = boundary.area_estimate();
    let expected_hole_area = polygon_area(&triangle_hole)
        + polygon_perimeter(&triangle_hole) * 2.0
        + std::f32::consts::PI * 4.0;
    let expected = 36.0 * 36.0 - expected_hole_area;

    assert!(exact);
    assert!((area - expected).abs() < 1.0e-3);
    assert!(boundary.contains_point([-15.0, 0.0]));
    assert!(!boundary.contains_point([0.0, 0.0]));
    assert!(!boundary.contains_point([18.5, 0.0]));
    assert!(
        (exact_convex_nested_inset_area(&outer, &[triangle_hole], 2.0).unwrap() - expected).abs()
            < 1.0e-3
    );
}

#[test]
fn concave_polygon_boundary_has_exact_buffered_area_before_topology_event() {
    let points = vec![
        [0.0, 0.0],
        [4.0, 0.0],
        [4.0, 2.0],
        [2.0, 2.0],
        [2.0, 4.0],
        [0.0, 4.0],
    ];
    let boundary = ProteinBoundaryGeometry::Polygon {
        points: points.clone(),
        inset_angstrom: 0.5,
    };

    let (area, exact) = boundary.area_estimate();
    let expected = 5.0 + 0.25 - std::f32::consts::PI * 0.25 * 0.25;

    assert!(exact);
    assert!((area - expected).abs() < 1.0e-4);
    assert_eq!(
        exact_simple_polygon_inset_area_before_topology_event(&points, 1.0),
        None
    );
}

#[test]
fn nested_concave_polygon_boundary_has_exact_buffered_area_before_topology_event() {
    let outer = vec![
        [-30.0, -20.0],
        [-10.0, -20.0],
        [-10.0, 0.0],
        [10.0, 0.0],
        [10.0, -20.0],
        [30.0, -20.0],
        [30.0, 20.0],
        [-30.0, 20.0],
    ];
    let hole = vec![
        [-20.0, 5.0],
        [-14.0, 5.0],
        [-14.0, 7.0],
        [-18.0, 7.0],
        [-18.0, 12.0],
        [-20.0, 12.0],
    ];
    let inset_angstrom = 0.5;
    let boundary = ProteinBoundaryGeometry::NestedPolygons {
        outer: outer.clone(),
        holes: vec![hole.clone()],
        inset_angstrom,
    };

    let (area, exact) = boundary.area_estimate();
    let outer_area =
        exact_simple_polygon_inset_area_before_topology_event(&outer, inset_angstrom).unwrap();
    let hole_area = polygon_area(&hole)
        + polygon_perimeter(&hole) * inset_angstrom
        + std::f32::consts::PI * inset_angstrom.powi(2);
    let expected = outer_area - hole_area;

    assert!(exact);
    assert!((area - expected).abs() < 1.0e-4);
}

#[test]
fn concave_hull_preserves_notched_protein_footprint() {
    let points = vec![
        [-30.0, -20.0],
        [-10.0, -20.0],
        [-10.0, 0.0],
        [10.0, 0.0],
        [10.0, -20.0],
        [30.0, -20.0],
        [30.0, 20.0],
        [-30.0, 20.0],
    ];
    let convex = convex_hull(points.clone());
    let concave = concave_hull(points);
    assert!(polygon_area(&concave) < polygon_area(&convex));
    assert!(!point_in_polygon([0.0, -10.0], &concave));
    assert!(point_in_polygon([0.0, 10.0], &concave));
}

#[test]
fn alpha_shape_preserves_unordered_notched_footprint() {
    let points = vec![
        [15.0, 15.0],
        [-15.0, -15.0],
        [5.0, -5.0],
        [-15.0, 15.0],
        [15.0, 0.0],
        [-5.0, 15.0],
        [0.0, -15.0],
        [5.0, 15.0],
        [-5.0, -5.0],
        [15.0, -15.0],
        [-15.0, 0.0],
        [5.0, 5.0],
        [-5.0, 5.0],
    ];
    let convex = convex_hull({
        let mut sorted = points.clone();
        sorted.sort_by(|left, right| {
            left[0]
                .partial_cmp(&right[0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left[1]
                        .partial_cmp(&right[1])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        sorted
    });
    let alpha = alpha_shape(&points, 8.0).unwrap();
    assert!(polygon_area(&alpha) < polygon_area(&convex));
    assert!(polygon_area(&alpha) > 1.0);
    assert!(!point_in_polygon([0.0, 10.0], &alpha));
}

#[test]
fn alpha_shape_components_preserve_disconnected_footprints() {
    let points = vec![
        [-14.0, -3.0],
        [-8.0, -3.0],
        [-8.0, 3.0],
        [-14.0, 3.0],
        [8.0, -3.0],
        [14.0, -3.0],
        [14.0, 3.0],
        [8.0, 3.0],
    ];
    let components = alpha_shape_components(&points, 5.0).unwrap();
    assert_eq!(components.len(), 2);
    assert!(components
        .iter()
        .all(|polygon| polygon_area(polygon) > 30.0));
    assert!(components
        .iter()
        .all(|polygon| !point_in_polygon([0.0, 0.0], polygon)));
}

#[test]
fn protein_boundary_concave_hull_excludes_notch_coordinates() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("notched.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  A   CNC A   1     -30.000 -20.000   0.000  1.00  0.00           C\n\
ATOM      2  B   CNC A   1     -10.000 -20.000   0.000  1.00  0.00           C\n\
ATOM      3  C   CNC A   1     -10.000   0.000   0.000  1.00  0.00           C\n\
ATOM      4  D   CNC A   1      10.000   0.000   0.000  1.00  0.00           C\n\
ATOM      5  E   CNC A   1      10.000 -20.000   0.000  1.00  0.00           C\n\
ATOM      6  F   CNC A   1      30.000 -20.000   0.000  1.00  0.00           C\n\
ATOM      7  G   CNC A   1      30.000  20.000   0.000  1.00  0.00           C\n\
ATOM      8  H   CNC A   1     -30.000  20.000   0.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let gro = temp.path().join("m.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [90.0, 90.0, 80.0], "placement": {"relaxation": true, "max_steps": 80}},
        "membranes": [{
            "name": "disc",
            "protein_boundary": {"mode": "inside", "geometry": "concave_hull", "protein": "CNC"},
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 100.0,
                "composition": [{"lipid": "LIP", "charge_e": 0.0}]
            }]
        }],
        "proteins": [{
            "name": "CNC",
            "coordinates": protein_path,
            "format": "pdb",
            "net_charge_e": 0.0,
            "placement": {"center_method": "none", "center_angstrom": [0.0, 0.0, 0.0]}
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"coordinates": gro, "manifest": temp.path().join("m.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["method"],
        "protein_boundary_area"
    );
    assert_eq!(value["summary"]["lipid_counts"]["LIP"], 20);
    let lipids = read_gro_residue_positions(&gro, "LIP");
    assert_eq!(lipids.len(), 20);
    for [x, y, _z] in lipids {
        assert!(
            !(x.abs() < 9.9 && y < -0.1),
            "lipid leaked into concave notch at {x},{y}"
        );
    }
}

#[test]
fn protein_boundary_alpha_shape_excludes_unordered_notch_coordinates() {
    let temp = tempfile::tempdir().unwrap();
    let protein_path = temp.path().join("alpha_notched.pdb");
    std::fs::write(
        &protein_path,
        "ATOM      1  A   ALP A   1      15.000  15.000   0.000  1.00  0.00           C\n\
ATOM      2  B   ALP A   1     -15.000 -15.000   0.000  1.00  0.00           C\n\
ATOM      3  C   ALP A   1       5.000  -5.000   0.000  1.00  0.00           C\n\
ATOM      4  D   ALP A   1     -15.000  15.000   0.000  1.00  0.00           C\n\
ATOM      5  E   ALP A   1      15.000   0.000   0.000  1.00  0.00           C\n\
ATOM      6  F   ALP A   1      -5.000  15.000   0.000  1.00  0.00           C\n\
ATOM      7  G   ALP A   1       0.000 -15.000   0.000  1.00  0.00           C\n\
ATOM      8  H   ALP A   1       5.000  15.000   0.000  1.00  0.00           C\n\
ATOM      9  I   ALP A   1      -5.000  -5.000   0.000  1.00  0.00           C\n\
ATOM     10  J   ALP A   1      15.000 -15.000   0.000  1.00  0.00           C\n\
ATOM     11  K   ALP A   1     -15.000   0.000   0.000  1.00  0.00           C\n\
ATOM     12  L   ALP A   1       5.000   5.000   0.000  1.00  0.00           C\n\
ATOM     13  M   ALP A   1      -5.000   5.000   0.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let gro = temp.path().join("m.gro");
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [90.0, 90.0, 80.0], "placement": {"relaxation": true, "max_steps": 80}},
        "membranes": [{
            "name": "disc",
            "protein_boundary": {"mode": "inside", "geometry": "alpha_shape", "alpha_radius_angstrom": 8.0, "protein": "ALP"},
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "apl_angstrom2": 100.0,
                "composition": [{"lipid": "LIP", "charge_e": 0.0}]
            }]
        }],
        "proteins": [{
            "name": "ALP",
            "coordinates": protein_path,
            "format": "pdb",
            "net_charge_e": 0.0,
            "placement": {"center_method": "none", "center_angstrom": [0.0, 0.0, 0.0]}
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"coordinates": gro, "manifest": temp.path().join("m.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(
        value["placement"]["leaflet_metrics"][0]["area"]["method"],
        "protein_boundary_area"
    );
    let lipids = read_gro_residue_positions(&gro, "LIP");
    assert!(!lipids.is_empty());
    for [x, y, _z] in lipids {
        assert!(
            !(x.abs() < 4.9 && y > -4.9),
            "lipid leaked into alpha-shape notch at {x},{y}"
        );
    }
}

#[test]
fn inserted_component_can_account_multiple_molecule_types_from_topologies() {
    let temp = tempfile::tempdir().unwrap();
    let coords = temp.path().join("rings.pdb");
    std::fs::write(
        &coords,
        "ATOM      1  A   R1  A   1       0.000   0.000   0.000  1.00  0.00           C\n\
ATOM      2  B   R2  B   1       1.000   0.000   0.000  1.00  0.00           C\n\
END\n",
    )
    .unwrap();
    let ring1 = temp.path().join("ring1.itp");
    let ring2 = temp.path().join("ring2.itp");
    std::fs::write(
        &ring1,
        "[ moleculetype ]\nring1 1\n\n[ atoms ]\n1 Q 1 R1 A 1 1.0\n",
    )
    .unwrap();
    std::fs::write(
        &ring2,
        "[ moleculetype ]\nring2 1\n\n[ atoms ]\n1 Q 1 R2 B 1 -2.0\n",
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [],
        "proteins": [{
            "name": "nanodisc",
            "coordinates": coords,
            "format": "pdb",
            "molecule_types": ["ring1", "ring2"],
            "charge_topologies": [ring1, ring2],
            "placement": {"center_method": "cog"}
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("m.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["summary"]["inserted_counts"]["ring1"], 1);
    assert_eq!(value["summary"]["inserted_counts"]["ring2"], 1);
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], -1.0);
}

#[test]
fn inserted_component_charge_can_derive_from_gromacs_topology() {
    let temp = tempfile::tempdir().unwrap();
    let topology = temp.path().join("solute.itp");
    std::fs::write(
        &topology,
        r#"
[ moleculetype ]
  SOL 1

[ atoms ]
  1 P1 1 SOL A 1 -1.0
  2 P1 1 SOL B 2 -0.5
"#,
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "POPC"}]
            }]
        }],
        "solutes": [{
            "name": "SOL",
            "count": 2,
            "charge_topology": topology
        }],
        "environment": {
            "ions": {"neutralize": true, "salt_molarity_mol_l": 0.0},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });
    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], -3.0);
    assert_eq!(
        value["charge"]["component_charges"][1]["source"],
        format!("gromacs_topology:{}:SOL", topology.display())
    );
}

#[test]
fn inserted_component_charge_topology_follows_local_includes() {
    let temp = tempfile::tempdir().unwrap();
    let topology = temp.path().join("topol.top");
    let included = temp.path().join("molecules.itp");
    std::fs::write(
        &included,
        r#"
[ moleculetype ]
  SOL 1

[ atoms ]
  1 P1 1 SOL A 1 -1.0
  2 P1 1 SOL B 2  0.25
"#,
    )
    .unwrap();
    std::fs::write(
        &topology,
        r#"
#include "molecules.itp"

[ system ]
included molecule definitions
"#,
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [80.0, 80.0, 80.0]},
        "membranes": [{
            "name": "bilayer",
            "leaflets": [{
                "name": "upper",
                "side": "upper",
                "composition": [{"lipid": "POPC"}]
            }]
        }],
        "solutes": [{
            "name": "SOL",
            "count": 2,
            "charge_topology": topology
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"manifest": temp.path().join("manifest.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["charge"]["net_charge_before_neutralization_e"], -1.5);
    assert_eq!(
        value["charge"]["component_charges"][1]["source"],
        format!("gromacs_topology:{}:SOL", topology.display())
    );
}

#[test]
fn inserted_component_charge_topology_emits_atoms_bonds_angles_and_dihedrals() {
    let temp = tempfile::tempdir().unwrap();
    let topology = temp.path().join("topol.top");
    let included = temp.path().join("molecule.itp");
    let out_top = temp.path().join("out.top");
    std::fs::write(
        &included,
        r#"
[ moleculetype ]
  SOL 1

[ atoms ]
  1 P1 1 SOL A 1 -0.5
  2 P2 1 SOL B 2  0.25
  3 P3 1 SOL C 3  0.25
  4 P4 1 SOL D 4  0.00

[ bonds ]
  1 2 1 0.33 900

[ angles ]
  1 2 3 2 150.0 100.0

[ dihedrals ]
  1 2 3 4 1 180.0 5.0 2
"#,
    )
    .unwrap();
    std::fs::write(
        &topology,
        r#"
#include "molecule.itp"

[ system ]
included molecule definitions
"#,
    )
    .unwrap();
    let request = json!({
        "schema_version": BUILD_SCHEMA_VERSION,
        "mode": "membrane",
        "system": {"box_size_angstrom": [40.0, 40.0, 40.0]},
        "membranes": [],
        "solutes": [{
            "name": "SOL",
            "count": 1,
            "charge_topology": topology
        }],
        "environment": {
            "ions": {"neutralize": false},
            "solvent": {"enabled": false}
        },
        "outputs": {"topology": out_top, "manifest": temp.path().join("manifest.json")}
    });

    let (code, value) = run_request_json(&serde_json::to_string(&request).unwrap(), false);
    assert_eq!(code, 0, "{value}");
    let emitted = std::fs::read_to_string(&out_top).unwrap();
    assert!(emitted.contains("[ moleculetype ]"));
    assert!(emitted.contains("[ atoms ]"));
    assert!(emitted.contains("[ bonds ]"));
    assert!(emitted.contains("[ angles ]"));
    assert!(emitted.contains("[ dihedrals ]"));
    assert!(emitted.contains("P1"));
    assert!(emitted.contains("1     2     1    0.33000    900.000"));
    assert!(emitted.contains("1     2     3     2    150.000    100.000"));
    assert!(emitted.contains("1     2     3     4     1    180.000      5.000     2"));
    assert!(emitted.contains("[ molecules ]"));
    assert!(emitted.contains("SOL              1"));
}
