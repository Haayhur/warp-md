use super::*;

#[test]
fn example_request_validates() {
    let text = serde_json::to_string(&example_request()).unwrap();
    let (code, value) = validate_request_json(&text);
    assert_eq!(code, 0, "{value}");
    assert_eq!(value["valid"], true);
}

#[test]
fn capabilities_report_exact_mixed_geometry_contracts() {
    let value = capabilities();
    let implemented = value["membrane"]["implemented_now"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|entry| entry.as_str())
        .collect::<Vec<_>>();

    for expected in [
        "exact multiple-circle plus axis-aligned rectangle region union area planning",
        "exact multiple-circle plus multiple axis-aligned rectangles region union area planning",
        "exact circle or ellipse plus multiple axis-aligned rectangles region union area planning",
        "exact multiple mutually disjoint ellipses plus multiple axis-aligned rectangles region union area planning",
        "exact circle plus multiple mutually disjoint ellipses region union area planning",
        "exact ellipse plus multiple mutually disjoint circles region union area planning",
        "exact multiple mutually disjoint circles plus multiple convex polygons region union area planning",
        "exact circle plus mutually disjoint mixed secondary shapes region union area planning",
        "exact ellipse plus mutually disjoint mixed secondary shapes region union area planning",
        "exact rectangle plus mutually disjoint mixed secondary shapes region union area planning",
        "exact convex polygon plus mutually disjoint mixed secondary shapes region union area planning",
        "exact circle or ellipse plus multiple convex polygons region union area planning",
        "exact circle plus overlapping convex polygons and polygon-disjoint ellipses region union area planning",
        "exact ellipse plus overlapping convex polygons and polygon-disjoint circles region union area planning",
        "exact circle plus overlapping convex polygons and polygon-disjoint mixed secondary shapes region union area planning",
        "exact ellipse plus overlapping convex polygons and polygon-disjoint mixed secondary shapes region union area planning",
        "exact circle or ellipse plus multiple actual-disjoint simple non-convex polygons using exact pair-overlap proofs",
        "exact rectangle plus simple non-convex polygon region union area planning",
        "exact mixed-shape region component partitioning using exact pair-disjoint proofs",
        "scaled triclinic XY membrane subdomain basis for explicit size_xy_angstrom deterministic leaflet placement",
        "scaled triclinic XY membrane subdomain basis for seeded random leaflet candidate placement",
        "adaptive far-image triclinic wrapped-region queries for heavily clipped rotated polygon constraints",
        "atomistic SOL water aliases for TIP3/TIP4/TIP5 residue-name output",
        "case-sensitive atomistic Na/Cl ion aliases while preserving Martini NA/CL names",
        "library-backed multi-bead Martini diacyl with LTF release tail tables, named LTF single-chain/monoglyceride/diglyceride/triglyceride, generated sphingomyelin, and sterol templates",
        "library-backed standard Martini 3 solvents including DMSO, ACN, alkanes, alkenes, alkynes, dienes, haloalkanes, alcohols, ethers, sulfides, ketones, aldehydes, esters, amines, carboxylic acids, and amides, reusable as coordinate-less inserted solutes",
        "conservative boundary-band error bounds for remaining grid-estimated mixed region unions",
    ] {
        assert!(
            implemented.contains(&expected),
            "missing capability: {expected}"
        );
    }
}

#[test]
fn minimum_image_distance_respects_enabled_axes() {
    let left = [-4.8, -4.8, 0.0];
    let right = [4.8, 4.8, 0.0];
    assert!(squared_distance3_pbc(left, right, [10.0, 10.0, 10.0], "none") > 180.0);
    assert!(squared_distance3_pbc(left, right, [10.0, 10.0, 10.0], "x") > 90.0);
    assert!(squared_distance3_pbc(left, right, [10.0, 10.0, 10.0], "xy") < 0.4);
}

#[test]
fn triclinic_minimum_image_distance_uses_box_vectors() {
    let left = [0.2, 0.2, 0.0];
    let right = [5.1, 8.8, 0.0];
    assert!(
        squared_distance3_for_system(left, right, &test_triclinic_system_with_pbc("none")) > 95.0
    );
    assert!(
        squared_distance3_for_system(left, right, &test_triclinic_system_with_pbc("xy")) < 0.02
    );
}

#[test]
fn triclinic_unit_cell_extents_drive_default_placement_bounds() {
    let mut system = test_triclinic_system_with_pbc("xy");
    system.box_size_angstrom = [11.0, 12.0, 13.0];
    system.unit_cell_angstrom = Some([10.0, 20.0, 30.0, 90.0, 90.0, 60.0]);

    let placement_box = placement_box_size_angstrom(&system);
    assert!((placement_box[0] - 20.0).abs() < 1.0e-4);
    assert!((placement_box[1] - 17.320_507).abs() < 1.0e-4);
    assert!((placement_box[2] - 30.0).abs() < 1.0e-4);

    let membrane = MembraneRequest {
        name: "triclinic-default".to_string(),
        center_xy_angstrom: None,
        size_xy_angstrom: None,
        center_z_angstrom: 0.0,
        solvate_voids: true,
        solvent_exclusion_half_thickness_angstrom:
            default_membrane_solvent_exclusion_half_thickness(),
        protein_boundary: None,
        leaflets: Vec::new(),
    };
    let bounds = membrane_layout_bounds(&system, &membrane);
    assert!((bounds.xmin + 10.0).abs() < 1.0e-4);
    assert!((bounds.xmax - 10.0).abs() < 1.0e-4);
    assert!((bounds.ymin + 8.660_254).abs() < 1.0e-4);
    assert!((bounds.ymax - 8.660_254).abs() < 1.0e-4);

    let solvent = SolventPolicy::default();
    assert_eq!(solvent_box_size_angstrom(&system, &solvent), placement_box);

    assert!(inserted_center_inside_box([8.0, 6.0, 0.0], &system, 0.1));
    assert!(!inserted_center_inside_box([9.8, 0.0, 0.0], &system, 0.1));
    assert!(inserted_center_inside_box([0.0, 6.1, 0.0], &system, 0.1));
    assert!(!inserted_center_inside_box([0.0, 8.6, 0.0], &system, 0.1));

    let centers = inserted_center_candidates(&system, 0.1, 8.0, &[[0.0; 3]]);
    assert!(!centers.is_empty());
    assert!(centers
        .iter()
        .all(|center| inserted_center_inside_box(*center, &system, 0.1)));
}

#[test]
fn triclinic_leaflet_relaxation_uses_xy_cell_basis_for_default_domain() {
    let mut system = test_triclinic_system_with_pbc("xy");
    system.unit_cell_angstrom = Some([10.0, 20.0, 30.0, 90.0, 90.0, 60.0]);
    let mut membrane = MembraneRequest {
        name: "triclinic-default".to_string(),
        center_xy_angstrom: None,
        size_xy_angstrom: None,
        center_z_angstrom: 0.0,
        solvate_voids: true,
        solvent_exclusion_half_thickness_angstrom:
            default_membrane_solvent_exclusion_half_thickness(),
        protein_boundary: None,
        leaflets: Vec::new(),
    };
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let bounds = membrane_layout_bounds(&system, &membrane);
    let basis = membrane_layout_basis(&system, &membrane, bounds, periodicity).unwrap();
    let left = basis.cartesian([0.99, 0.5]);
    let right = basis.cartesian([0.01, 0.5]);
    let fractional_delta = (left[0] - right[0]).hypot(left[1] - right[1]);
    assert!(fractional_delta > 9.0);

    let wrapped_delta = basis.minimum_image_delta(left, right);
    assert!(wrapped_delta[0].hypot(wrapped_delta[1]) < 0.25);

    membrane.size_xy_angstrom = Some([10.0, 8.660_254]);
    let explicit_bounds = membrane_layout_bounds(&system, &membrane);
    let explicit_basis =
        membrane_layout_basis(&system, &membrane, explicit_bounds, periodicity).unwrap();
    assert!((explicit_basis.a[0].hypot(explicit_basis.a[1]) - 5.0).abs() < 1.0e-4);
    assert!((explicit_basis.b[0].hypot(explicit_basis.b[1]) - 10.0).abs() < 1.0e-4);

    let radii = vec![0.4; 6];
    let grid = initial_leaflet_grid(
        &system,
        &radii,
        60.0,
        explicit_bounds,
        Some(explicit_basis),
        periodicity,
        &membrane,
        &LeafletRequest {
            name: "upper".to_string(),
            side: "upper".to_string(),
            apl_angstrom2: Some(60.0),
            composition: Vec::new(),
            exclusions: Vec::new(),
            regions: Vec::new(),
        },
        &[],
    )
    .unwrap();
    assert_eq!(grid.len(), radii.len());
    for point in grid {
        let fractional = explicit_basis.fractional([point.x, point.y]);
        assert!(fractional[0] > 0.0 && fractional[0] < 1.0);
        assert!(fractional[1] > 0.0 && fractional[1] < 1.0);
    }
}

#[test]
fn triclinic_constrained_leaflet_grid_stays_inside_xy_cell_basis() {
    let mut system = test_triclinic_system_with_pbc("xy");
    system.unit_cell_angstrom = Some([10.0, 20.0, 30.0, 90.0, 90.0, 60.0]);
    let membrane = MembraneRequest {
        name: "triclinic-constrained".to_string(),
        center_xy_angstrom: Some([2.0, -1.0]),
        size_xy_angstrom: Some([10.0, 8.660_254]),
        center_z_angstrom: 0.0,
        solvate_voids: true,
        solvent_exclusion_half_thickness_angstrom:
            default_membrane_solvent_exclusion_half_thickness(),
        protein_boundary: None,
        leaflets: Vec::new(),
    };
    let leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: Some(18.0),
        exclusions: Vec::new(),
        regions: vec![LeafletRegion {
            name: Some("wide-patch".to_string()),
            role: "patch".to_string(),
            geometry: RegionGeometry::Circle {
                center_angstrom: [2.0, -1.0],
                radius_angstrom: 100.0,
            },
        }],
        composition: Vec::new(),
    };
    let bounds = membrane_layout_bounds(&system, &membrane);
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let basis = membrane_layout_basis(&system, &membrane, bounds, periodicity).unwrap();
    let radii = vec![0.25; 18];

    let points = initial_leaflet_grid(
        &system,
        &radii,
        18.0,
        bounds,
        Some(basis),
        periodicity,
        &membrane,
        &leaflet,
        &[],
    )
    .unwrap();

    assert_eq!(points.len(), radii.len());
    for point in points {
        assert!(point_inside_layout_domain(
            &point,
            bounds,
            periodicity,
            Some(basis)
        ));
        let fractional = basis.fractional([point.x, point.y]);
        assert!(fractional[0] > 0.0 && fractional[0] < 1.0);
        assert!(fractional[1] > 0.0 && fractional[1] < 1.0);
    }
}

#[test]
fn triclinic_leaflet_regions_wrap_across_xy_cell_basis_edges() {
    let mut system = test_triclinic_system_with_pbc("xy");
    system.unit_cell_angstrom = Some([10.0, 20.0, 30.0, 90.0, 90.0, 60.0]);
    let membrane = MembraneRequest {
        name: "triclinic-region".to_string(),
        center_xy_angstrom: Some([0.0, 0.0]),
        size_xy_angstrom: Some([10.0, 8.660_254]),
        center_z_angstrom: 0.0,
        solvate_voids: true,
        solvent_exclusion_half_thickness_angstrom:
            default_membrane_solvent_exclusion_half_thickness(),
        protein_boundary: None,
        leaflets: Vec::new(),
    };
    let bounds = membrane_layout_bounds(&system, &membrane);
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let basis = membrane_layout_basis(&system, &membrane, bounds, periodicity).unwrap();
    let center = basis.cartesian([0.95, 0.50]);
    let wrapped_point = basis.cartesian([-0.04, 0.50]);
    let region = LeafletRegion {
        name: Some("basis-edge-patch".to_string()),
        role: "patch".to_string(),
        geometry: RegionGeometry::Circle {
            center_angstrom: center,
            radius_angstrom: 0.5,
        },
    };
    let leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: Some(60.0),
        exclusions: Vec::new(),
        regions: vec![region],
        composition: Vec::new(),
    };

    assert!(!leaflet_allows_point_periodic(
        &leaflet,
        wrapped_point,
        bounds,
        periodicity
    ));
    assert!(leaflet_allows_point_periodic_basis(
        &leaflet,
        wrapped_point,
        bounds,
        periodicity,
        Some(basis)
    ));
}

#[test]
fn explicit_solvent_box_overrides_triclinic_placement_extents() {
    let mut system = test_triclinic_system_with_pbc("xyz");
    system.box_size_angstrom = [11.0, 12.0, 13.0];
    system.unit_cell_angstrom = Some([10.0, 20.0, 30.0, 90.0, 90.0, 60.0]);
    let solvent = SolventPolicy {
        box_size_angstrom: Some([4.0, 5.0, 6.0]),
        ..SolventPolicy::default()
    };
    assert_eq!(
        solvent_box_size_angstrom(&system, &solvent),
        [4.0, 5.0, 6.0]
    );
}

#[test]
fn triclinic_solvent_defaults_use_fractional_cell_containment() {
    let mut request: BuildRequest = serde_json::from_value(example_request()).unwrap();
    request.system = test_triclinic_system_with_pbc("xyz");
    request.system.box_size_angstrom = [11.0, 12.0, 13.0];
    request.system.unit_cell_angstrom = Some([10.0, 20.0, 30.0, 90.0, 90.0, 60.0]);
    request.environment.solvent = SolventPolicy::default();

    assert!(point_inside_solvent_zone(
        [8.0, 6.0, 0.0],
        &request,
        &request.environment.solvent
    ));
    assert!(!point_inside_solvent_zone(
        [9.8, 0.0, 0.0],
        &request,
        &request.environment.solvent
    ));
    assert!(solvent_candidate_allowed(
        &request,
        &request.environment.solvent,
        [8.0, 6.0, 0.0]
    ));
    assert!(!solvent_candidate_allowed(
        &request,
        &request.environment.solvent,
        [9.8, 0.0, 0.0]
    ));

    let candidates = solvent_candidates(
        &request,
        &[],
        &request.environment.solvent,
        8.0,
        &[[0.0; 3]],
    );
    assert!(!candidates.is_empty());
    assert!(candidates.iter().all(|candidate| point_inside_solvent_zone(
        *candidate,
        &request,
        &request.environment.solvent
    )));
}

#[test]
fn inserted_center_clearance_uses_periodic_edges() {
    let occupied = vec![test_bead([4.8, 0.0, 0.0])];
    let candidate = [-4.8, 0.0, 0.0];
    let molecule_radius = 0.2;
    let occupied_radius = 0.5;
    let centers = Vec::new();
    assert!(inserted_center_is_clear(
        candidate,
        molecule_radius,
        &occupied,
        occupied_radius,
        &centers,
        &test_system_with_pbc("none")
    ));
    assert!(!inserted_center_is_clear(
        candidate,
        molecule_radius,
        &occupied,
        occupied_radius,
        &centers,
        &test_system_with_pbc("x")
    ));
}

#[test]
fn solvent_candidate_clearance_uses_periodic_edges() {
    let occupied = vec![test_bead([4.8, 0.0, 0.0])];
    let bins = occupied_position_bins(&occupied, 1.0);
    let candidate = [-4.8, 0.0, 0.0];
    assert!(candidate_overlaps_occupied_for_system(
        candidate,
        &bins,
        &pbc_occupied_points(&occupied, &test_system_with_pbc("x")),
        &[],
        1.0,
        1.0,
        &test_system_with_pbc("x")
    ));
    assert!(!candidate_overlaps_occupied_for_system(
        candidate,
        &bins,
        &pbc_occupied_points(&occupied, &test_system_with_pbc("none")),
        &[],
        1.0,
        1.0,
        &test_system_with_pbc("none")
    ));
    assert!(candidate_overlaps_occupied_for_system(
        candidate,
        &bins,
        &pbc_occupied_points(&occupied, &test_system_with_pbc("x")),
        &[],
        1.0,
        0.25,
        &test_system_with_pbc("x")
    ));
}

#[test]
fn placement_diagnostics_report_minimum_image_exclusion_margins() {
    let beads = vec![
        test_bead_with_residue(1, [4.8, 0.0, 0.0]),
        test_bead_with_residue(2, [-4.8, 0.0, 0.0]),
    ];
    let non_periodic = placement_diagnostics(&test_system_with_pbc("none"), &beads);
    assert_eq!(non_periodic.exclusion_violation_count, 0);
    assert_eq!(non_periodic.pbc_axes, [false, false, false]);
    assert!(!non_periodic.uses_minimum_image);
    assert!(non_periodic.min_exclusion_margin_angstrom.unwrap() > 4.0);

    let periodic = placement_diagnostics(&test_system_with_pbc("x"), &beads);
    assert_eq!(periodic.exclusion_violation_count, 1);
    assert_eq!(periodic.exclusion_violation_examples.len(), 1);
    assert_eq!(periodic.pbc_axes, [true, false, false]);
    assert!(periodic.uses_minimum_image);
    assert!(periodic.min_inter_residue_distance_angstrom.unwrap() < 0.5);
    assert!(periodic.min_exclusion_margin_angstrom.unwrap() < -4.0);
    let example = &periodic.exclusion_violation_examples[0];
    assert_eq!(example.left_residue_id, 1);
    assert_eq!(example.right_residue_id, 2);
    assert!(example.distance_angstrom < 0.5);
    assert!(example.margin_angstrom < -4.0);
}

#[test]
fn placement_diagnostics_caps_exclusion_violation_examples() {
    let beads = (0..6)
        .map(|idx| test_bead_with_residue(idx + 1, [0.0, 0.0, idx as f32 * 0.1]))
        .collect::<Vec<_>>();
    let diagnostics = placement_diagnostics(&test_system_with_pbc("none"), &beads);
    assert!(diagnostics.exclusion_violation_count > 4);
    assert_eq!(diagnostics.exclusion_violation_examples.len(), 4);
}

#[test]
fn leaflet_geometry_diagnostics_report_region_violations() {
    let membrane = MembraneRequest {
        name: "geom".to_string(),
        center_xy_angstrom: None,
        size_xy_angstrom: None,
        center_z_angstrom: 0.0,
        solvate_voids: true,
        solvent_exclusion_half_thickness_angstrom:
            default_membrane_solvent_exclusion_half_thickness(),
        protein_boundary: None,
        leaflets: Vec::new(),
    };
    let leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: None,
        exclusions: Vec::new(),
        regions: vec![LeafletRegion {
            name: Some("rect-hole".to_string()),
            role: "hole".to_string(),
            geometry: RegionGeometry::Rectangle {
                center_angstrom: [0.0, 0.0],
                size_angstrom: [20.0, 10.0],
                rotate_degrees: 0.0,
            },
        }],
        composition: Vec::new(),
    };
    let points = vec![LayoutPoint {
        x: 0.0,
        y: 0.0,
        radius: 1.0,
    }];
    let diagnostics = leaflet_geometry_diagnostics(
        &membrane,
        &leaflet,
        &[],
        &points,
        LayoutBounds {
            xmin: -50.0,
            xmax: 50.0,
            ymin: -50.0,
            ymax: 50.0,
        },
        LayoutPeriodicity::default(),
        None,
    )
    .unwrap();
    assert_eq!(diagnostics.constraint_count, 1);
    assert_eq!(diagnostics.violation_count, 1);
    assert!(diagnostics.max_violation_angstrom > 4.9);
    assert_eq!(diagnostics.constraints[0].name, "rect-hole");
    assert_eq!(diagnostics.constraints[0].role, "hole");
}

#[test]
fn nearest_allowed_leaflet_point_projects_from_ellipse_hole_boundary() {
    let membrane = MembraneRequest {
        name: "geom".to_string(),
        center_xy_angstrom: None,
        size_xy_angstrom: None,
        center_z_angstrom: 0.0,
        solvate_voids: true,
        solvent_exclusion_half_thickness_angstrom:
            default_membrane_solvent_exclusion_half_thickness(),
        protein_boundary: None,
        leaflets: Vec::new(),
    };
    let leaflet = LeafletRequest {
        name: "upper".to_string(),
        side: "upper".to_string(),
        apl_angstrom2: None,
        exclusions: Vec::new(),
        regions: vec![LeafletRegion {
            name: Some("ellipse-hole".to_string()),
            role: "hole".to_string(),
            geometry: RegionGeometry::Ellipse {
                center_angstrom: [0.0, 0.0],
                radius_angstrom: [10.0, 5.0],
                rotate_degrees: 0.0,
            },
        }],
        composition: Vec::new(),
    };
    let bounds = LayoutBounds {
        xmin: -30.0,
        xmax: 30.0,
        ymin: -30.0,
        ymax: 30.0,
    };
    let point = LayoutPoint {
        x: 0.0,
        y: 0.0,
        radius: 1.0,
    };
    let projected = nearest_allowed_leaflet_point(
        point,
        &membrane,
        &leaflet,
        &[],
        bounds,
        LayoutPeriodicity::default(),
        None,
    )
    .unwrap();
    let projected = projected.unwrap();
    assert!(!region_contains_point(
        &leaflet.regions[0],
        [projected.x, projected.y]
    ));
    assert!((projected.x.abs() - 11.001).abs() < 1.0e-3);
    assert!(projected.y.abs() < 1.0e-3);
}
