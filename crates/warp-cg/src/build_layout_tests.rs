use super::*;

#[test]
fn relaxation_reduces_pair_overlap() {
    let bounds = LayoutBounds {
        xmin: -20.0,
        xmax: 20.0,
        ymin: -20.0,
        ymax: 20.0,
    };
    let mut points = vec![
        LayoutPoint {
            x: 0.0,
            y: 0.0,
            radius: 4.0,
        },
        LayoutPoint {
            x: 1.0,
            y: 0.0,
            radius: 4.0,
        },
    ];
    let before = min_pair_clearance(&points, bounds, LayoutPeriodicity::default(), None).unwrap();
    let metrics = relax_leaflet_points_periodic(
        &mut points,
        bounds,
        &[],
        RelaxationConfig {
            enabled: true,
            max_steps: 50,
            push_tolerance: 0.001,
            lipid_push_multiplier: 0.25,
            edge_push_multiplier: 0.5,
            occupation_modifier: 0.05,
        },
        LayoutPeriodicity::default(),
    );
    assert!(metrics.min_pair_clearance_angstrom.unwrap() > before);
    assert!(metrics.relaxation_steps > 0);
    assert_eq!(metrics.trajectory_frame_count, metrics.relaxation_steps + 1);
    assert_eq!(
        metrics.optimizer_trace.len(),
        metrics.trajectory_frame_count
    );
    assert_eq!(metrics.optimizer_trace[0].step, 0);
    assert_eq!(metrics.optimizer_trace[0].max_push_angstrom, 0.0);
    assert_eq!(
        metrics.optimizer_trace.last().unwrap().step,
        metrics.relaxation_steps
    );
    assert_eq!(
        metrics.optimizer_trace.last().unwrap().max_push_angstrom,
        metrics.max_push_angstrom
    );
    assert_eq!(metrics.optimizer_neighbor_cutoff_angstrom, Some(16.0));
    assert!(metrics.neighbor_search_rebuild_count <= metrics.relaxation_steps);
    assert_eq!(metrics.neighbor_search_rebuild_count, 0);
    assert!(metrics.max_total_displacement_angstrom > 0.0);
    assert!(metrics.mean_total_displacement_angstrom > 0.0);
}

#[test]
fn periodic_relaxation_uses_minimum_image_across_x_boundary() {
    let bounds = LayoutBounds {
        xmin: -20.0,
        xmax: 20.0,
        ymin: -20.0,
        ymax: 20.0,
    };
    let mut points = vec![
        LayoutPoint {
            x: -19.0,
            y: 0.0,
            radius: 2.0,
        },
        LayoutPoint {
            x: 19.0,
            y: 0.0,
            radius: 2.0,
        },
    ];
    let before = min_pair_clearance(
        &points,
        bounds,
        LayoutPeriodicity { x: true, y: false },
        None,
    )
    .unwrap();
    assert!(before < 0.0);

    let metrics = relax_leaflet_points_periodic(
        &mut points,
        bounds,
        &[],
        RelaxationConfig {
            enabled: true,
            max_steps: 80,
            push_tolerance: 0.001,
            lipid_push_multiplier: 0.5,
            edge_push_multiplier: 0.5,
            occupation_modifier: 0.05,
        },
        LayoutPeriodicity { x: true, y: false },
    );

    assert!(metrics.min_pair_clearance_angstrom.unwrap() > before);
    assert!(metrics.max_edge_violation_angstrom <= 0.0);
    assert!(points
        .iter()
        .all(|point| point.x >= bounds.xmin && point.x < bounds.xmax));
}

#[test]
fn periodic_relaxation_pushes_points_out_of_wrapped_circular_exclusions() {
    let bounds = LayoutBounds {
        xmin: -20.0,
        xmax: 20.0,
        ymin: -20.0,
        ymax: 20.0,
    };
    let mut points = vec![LayoutPoint {
        x: -19.0,
        y: 0.0,
        radius: 2.0,
    }];
    let exclusions = [CircularExclusion {
        x: 19.0,
        y: 0.0,
        radius: 4.0,
    }];
    let periodicity = LayoutPeriodicity { x: true, y: false };
    let before = max_exclusion_violation(&points, &exclusions, bounds, periodicity, None);
    assert!(before > 0.0);

    let metrics = relax_leaflet_points_periodic(
        &mut points,
        bounds,
        &exclusions,
        RelaxationConfig {
            enabled: true,
            max_steps: 80,
            push_tolerance: 0.001,
            lipid_push_multiplier: 0.25,
            edge_push_multiplier: 0.75,
            occupation_modifier: 0.0,
        },
        periodicity,
    );

    assert!(metrics.max_exclusion_violation_angstrom <= 0.01);
    assert!(metrics.max_total_displacement_angstrom > 0.0);
    assert!(points
        .iter()
        .all(|point| point.x >= bounds.xmin && point.x < bounds.xmax));
}

#[test]
fn skew_basis_relaxation_uses_fractional_minimum_image() {
    let bounds = LayoutBounds {
        xmin: -10.0,
        xmax: 10.0,
        ymin: -8.660_254,
        ymax: 8.660_254,
    };
    let basis = LayoutBasis2D::new([-10.0, -8.660_254], [10.0, 0.0], [10.0, 17.320_508]).unwrap();
    let periodicity = LayoutPeriodicity { x: true, y: true };
    let mut points = vec![
        LayoutPoint {
            x: 4.9,
            y: 0.0,
            radius: 1.2,
        },
        LayoutPoint {
            x: -4.9,
            y: 0.0,
            radius: 1.2,
        },
    ];
    let rectangular_before = min_pair_clearance(&points, bounds, periodicity, None).unwrap();
    let fractional_before = min_pair_clearance(&points, bounds, periodicity, Some(basis)).unwrap();
    assert!(rectangular_before > 7.0);
    assert!(fractional_before < 0.0);

    let metrics = relax_leaflet_points_with_projector_basis(
        &mut points,
        bounds,
        &[],
        RelaxationConfig {
            enabled: true,
            max_steps: 80,
            push_tolerance: 0.001,
            lipid_push_multiplier: 0.5,
            edge_push_multiplier: 0.5,
            occupation_modifier: 0.0,
        },
        periodicity,
        Some(basis),
        |_| None,
    );

    assert!(metrics.min_pair_clearance_angstrom.unwrap() > fractional_before);
    assert!(metrics.max_edge_violation_angstrom <= 0.0);
    for point in points {
        let fractional = basis.fractional([point.x, point.y]);
        assert!(fractional[0] >= 0.0 && fractional[0] < 1.0);
        assert!(fractional[1] >= 0.0 && fractional[1] < 1.0);
    }
}

#[test]
fn relaxation_pushes_points_out_of_exclusions() {
    let bounds = LayoutBounds {
        xmin: -20.0,
        xmax: 20.0,
        ymin: -20.0,
        ymax: 20.0,
    };
    let mut points = vec![LayoutPoint {
        x: 0.0,
        y: 0.0,
        radius: 2.0,
    }];
    let exclusions = [CircularExclusion {
        x: 0.0,
        y: 0.0,
        radius: 5.0,
    }];
    let metrics = relax_leaflet_points_periodic(
        &mut points,
        bounds,
        &exclusions,
        RelaxationConfig {
            enabled: true,
            max_steps: 100,
            push_tolerance: 0.001,
            lipid_push_multiplier: 0.25,
            edge_push_multiplier: 0.75,
            occupation_modifier: 0.0,
        },
        LayoutPeriodicity::default(),
    );
    assert!(metrics.max_exclusion_violation_angstrom <= 0.01);
    assert!(metrics.max_total_displacement_angstrom > 0.0);
}

#[test]
fn rectangular_grid_respects_count() {
    let points = rectangular_leaflet_grid(
        &[3.0; 7],
        64.0,
        LayoutBounds {
            xmin: -20.0,
            xmax: 20.0,
            ymin: -20.0,
            ymax: 20.0,
        },
    );
    assert_eq!(points.len(), 7);
}

#[test]
fn disabled_relaxation_reports_static_trajectory_metrics() {
    let bounds = LayoutBounds {
        xmin: -20.0,
        xmax: 20.0,
        ymin: -20.0,
        ymax: 20.0,
    };
    let mut points = vec![LayoutPoint {
        x: 0.0,
        y: 0.0,
        radius: 2.0,
    }];
    let metrics = relax_leaflet_points_periodic(
        &mut points,
        bounds,
        &[],
        RelaxationConfig {
            enabled: false,
            max_steps: 50,
            push_tolerance: 0.001,
            lipid_push_multiplier: 0.25,
            edge_push_multiplier: 0.5,
            occupation_modifier: 0.05,
        },
        LayoutPeriodicity::default(),
    );
    assert_eq!(metrics.relaxation_steps, 0);
    assert_eq!(metrics.trajectory_frame_count, 1);
    assert_eq!(metrics.optimizer_trace.len(), 1);
    assert_eq!(metrics.optimizer_trace[0].step, 0);
    assert_eq!(metrics.optimizer_trace[0].step_multiplier, 0.0);
    assert_eq!(metrics.optimizer_neighbor_cutoff_angstrom, None);
    assert_eq!(metrics.neighbor_search_rebuild_count, 0);
    assert_eq!(metrics.max_total_displacement_angstrom, 0.0);
    assert_eq!(metrics.mean_total_displacement_angstrom, 0.0);
}

#[test]
fn relaxation_projector_confines_points_during_steps() {
    let bounds = LayoutBounds {
        xmin: -20.0,
        xmax: 20.0,
        ymin: -20.0,
        ymax: 20.0,
    };
    let mut points = vec![LayoutPoint {
        x: 8.0,
        y: 0.0,
        radius: 1.0,
    }];
    let metrics = relax_leaflet_points_with_projector_periodic(
        &mut points,
        bounds,
        &[],
        RelaxationConfig {
            enabled: true,
            max_steps: 3,
            push_tolerance: 0.01,
            lipid_push_multiplier: 1.0,
            edge_push_multiplier: 1.0,
            occupation_modifier: 0.0,
        },
        LayoutPeriodicity::default(),
        |point| {
            (point.x > 2.0).then_some(LayoutPoint {
                x: 2.0,
                y: point.y,
                radius: point.radius,
            })
        },
    );
    assert_eq!(points[0].x, 2.0);
    assert!(metrics.max_total_displacement_angstrom >= 6.0);
}
