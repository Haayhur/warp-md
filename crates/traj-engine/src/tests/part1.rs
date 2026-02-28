#[test]
fn rg_plan_basic() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = RgPlan::new(sel, false);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 2);
            assert!((vals[0] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn rmsd_plan_align_zero() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = RmsdPlan::new(sel, ReferenceMode::Topology, true);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert!((vals[0]).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn symmrmsd_plan_align_zero() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = SymmRmsdPlan::new(sel, ReferenceMode::Topology, true);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert!((vals[0]).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn center_of_mass_plan_basic() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = CenterOfMassPlan::new(sel);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 3);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1]).abs() < 1e-6);
            assert!((data[2]).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn volume_plan_orthorhombic() {
    let system = build_system();
    let frames = vec![system.positions0.clone().unwrap()];
    let box_ = Box3::Orthorhombic {
        lx: 2.0,
        ly: 3.0,
        lz: 4.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut plan = VolumePlan::new();
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 24.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn volume_plan_triclinic() {
    let system = build_system();
    let frames = vec![system.positions0.clone().unwrap()];
    let box_ = Box3::Triclinic {
        m: [2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 4.0],
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut plan = VolumePlan::new();
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 24.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn superpose_plan_basic() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = SuperposePlan::new(sel, ReferenceMode::Topology, false, false);
    let frames = vec![vec![[1.0, 1.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            let dx0 = vals[0] as f64 - 0.0;
            let dy0 = vals[1] as f64 - 0.0;
            let dz0 = vals[2] as f64 - 0.0;
            let dx1 = vals[3] as f64 - 1.0;
            let dy1 = vals[4] as f64 - 0.0;
            let dz1 = vals[5] as f64 - 0.0;
            let err0 = (dx0 * dx0 + dy0 * dy0 + dz0 * dz0).sqrt();
            let err1 = (dx1 * dx1 + dy1 * dy1 + dz1 * dz1).sqrt();
            assert!(err0 < 1e-5);
            assert!(err1 < 1e-5);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn atomiccorr_plan_basic() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = AtomicCorrPlan::new(sel, ReferenceMode::Frame0)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(1);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0]], vec![[1.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            time,
            data,
            rows,
            cols,
        } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 1);
            assert_eq!(time.len(), 1);
            assert_eq!(data.len(), 1);
            assert!(data[0].abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn velocity_autocorr_plan_basic() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = VelocityAutoCorrPlan::new(sel)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(1);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            time,
            data,
            rows,
            cols,
        } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 1);
            assert_eq!(time.len(), 1);
            assert_eq!(data.len(), 1);
            assert!((data[0] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn hausdorff_plan_basic() {
    let mut system = build_two_resid_system();
    let sel_a = system.select("name DA or name AC").unwrap();
    let sel_b = system.select("name DA").unwrap();
    let mut plan = HausdorffPlan::new(sel_a, sel_b, PbcMode::None);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn hausdorff_plan_triclinic_pbc() {
    let mut system = build_two_resid_system();
    let sel_a = system.select("name DA").unwrap();
    let sel_b = system.select("name AC").unwrap();
    let mut plan = HausdorffPlan::new(sel_a, sel_b, PbcMode::Orthorhombic);
    let frames = vec![vec![[9.0, 9.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]];
    let box_ = Box3::Triclinic {
        m: [10.0, 0.0, 0.0, 2.0, 10.0, 0.0, 0.0, 0.0, 10.0],
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            let expected = 20.0f32.sqrt();
            assert!((vals[0] - expected).abs() < 1e-5);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn check_structure_plan_nan() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = CheckStructurePlan::new(sel);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0]], vec![[f32::NAN, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 2);
            assert_eq!(vals[0], 1.0);
            assert_eq!(vals[1], 0.0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn check_chirality_plan_basic() {
    let mut system = build_four_resid_system();
    let a = system.select("resid 1").unwrap();
    let b = system.select("resid 2").unwrap();
    let c = system.select("resid 3").unwrap();
    let d = system.select("resid 4").unwrap();
    let mut plan = CheckChiralityPlan::new(vec![(a, b, c, d)], false);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 1);
            assert!(data[0] > 0.0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn atom_map_plan_basic() {
    let mut system = build_two_resid_system();
    let sel_a = system.select("name DA").unwrap();
    let sel_b = system.select("name AC").unwrap();
    let mut plan = AtomMapPlan::new(sel_a, sel_b, PbcMode::None);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 1);
            assert_eq!(data[0], 1.0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn rotate_dihedral_plan_basic() {
    let mut system = build_four_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let sel_c = system.select("resid 3").unwrap();
    let sel_d = system.select("resid 4").unwrap();
    let rotate_sel = system.select("resid 4").unwrap();
    let mut plan =
        RotateDihedralPlan::new(sel_a, sel_b, sel_c, sel_d, rotate_sel, 90.0, false, true);
    let frames = vec![vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 12);
            let x = vals[9];
            let y = vals[10];
            let z = vals[11];
            assert!(x.abs() < 1e-5);
            assert!((y - 1.0).abs() < 1e-5);
            assert!((z + 1.0).abs() < 1e-5);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn set_dihedral_plan_basic() {
    let mut system = build_four_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let sel_c = system.select("resid 3").unwrap();
    let sel_d = system.select("resid 4").unwrap();
    let rotate_sel = system.select("resid 4").unwrap();
    let mut plan = SetDihedralPlan::new(
        sel_a,
        sel_b,
        sel_c,
        sel_d,
        rotate_sel,
        90.0,
        false,
        PbcMode::None,
        true,
        false,
    );
    let frames = vec![vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 12);
            let x = vals[9];
            let y = vals[10];
            let z = vals[11];
            assert!(x.abs() < 1e-5);
            assert!((y - 1.0).abs() < 1e-5);
            assert!((z + 1.0).abs() < 1e-5);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn dihedral_rms_plan_basic() {
    let mut system = build_four_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let sel_c = system.select("resid 3").unwrap();
    let sel_d = system.select("resid 4").unwrap();
    let mut plan = DihedralRmsPlan::new(
        vec![(sel_a, sel_b, sel_c, sel_d)],
        ReferenceMode::Frame0,
        false,
        PbcMode::None,
        true,
        false,
    );
    let frames = vec![
        vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 1.0],
        ],
        vec![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0, 1.0],
        ],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 2);
            assert!(vals[0].abs() < 1e-6);
            assert!((vals[1] - 90.0).abs() < 1e-3);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn xcorr_plan_basic() {
    let mut system = build_two_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let mut plan = XcorrPlan::new(sel_a, sel_b, ReferenceMode::Topology, false)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(1);
    let frames = vec![
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            time,
            data,
            rows,
            cols,
        } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 1);
            assert_eq!(time.len(), 1);
            assert_eq!(data.len(), 1);
            assert!((data[0] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn wavelet_plan_basic() {
    let mut system = build_two_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let mut plan = WaveletPlan::new(sel_a, sel_b, false, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 2);
            assert!((data[0] + 0.5).abs() < 1e-6);
            assert!((data[1] + 0.5).abs() < 1e-6);
            assert!((data[2] + 1.0).abs() < 1e-6);
            assert!(data[3].abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_basic() {
    let mut system = build_plane_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = SurfPlan::new(sel);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_sasa_single_atom() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_probe_radius(0.0)
        .with_n_sphere_points(64)
        .with_radii(Some(vec![1.0]));
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            let expected = 4.0f32 * std::f32::consts::PI;
            assert!((vals[0] - expected).abs() < 0.2);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn nmr_ired_plan_basic() {
    let system = build_two_resid_system();
    let mut plan = NmrIredPlan::new(vec![[0, 1]], 1.0, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 3);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!(data[1].abs() < 1e-6);
            assert!(data[2].abs() < 1e-6);
            assert!(data[3].abs() < 1e-6);
            assert!((data[4] - 1.0).abs() < 1e-6);
            assert!(data[5].abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn nmr_ired_plan_pbc_orthorhombic() {
    let system = build_two_resid_system();
    let mut plan = NmrIredPlan::new(vec![[0, 1]], 1.0, PbcMode::Orthorhombic);
    let frames = vec![vec![[9.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTrajWithBox::new(
        frames,
        Box3::Orthorhombic {
            lx: 10.0,
            ly: 10.0,
            lz: 10.0,
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 3);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!(data[1].abs() < 1e-6);
            assert!(data[2].abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}
