#[test]
fn torsion_diffusion_plan_basic() {
    let mut system = build_four_resid_system();
    let sel = system
        .select("resid 1 or resid 2 or resid 3 or resid 4")
        .unwrap();
    let mut plan = TorsionDiffusionPlan::new(sel);
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
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 4);
            let sum: f32 = data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn toroidal_diffusion_plan_basic() {
    let mut system = build_four_resid_system();
    let sel = system
        .select("resid 1 or resid 2 or resid 3 or resid 4")
        .unwrap();
    let mut plan = ToroidalDiffusionPlan::new(sel);
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
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 4);
            let sum: f32 = data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn toroidal_diffusion_plan_mass_weighted() {
    let mut interner = StringInterner::new();
    let name = interner.intern_upper("CA");
    let res = interner.intern_upper("ALA");
    let atoms = AtomTable {
        name_id: vec![name; 8],
        resname_id: vec![res; 8],
        resid: vec![1, 2, 3, 4, 5, 6, 7, 8],
        chain_id: vec![0; 8],
        element_id: vec![0; 8],
        mass: vec![1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0],
    };
    let positions0 = Some(vec![
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [5.0, 1.0, 0.0, 1.0],
        [5.0, 0.0, 0.0, 1.0],
        [6.0, 0.0, 0.0, 1.0],
        [6.0, 1.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let sel = system.select("resid 1:8").unwrap();
    let mut plan = ToroidalDiffusionPlan::new(sel).with_mass_weighted(true);
    let frame = vec![
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0], // trans
        [5.0, 1.0, 0.0, 1.0],
        [5.0, 0.0, 0.0, 1.0],
        [6.0, 0.0, 0.0, 1.0],
        [6.0, 1.0, 0.0, 1.0], // cis
    ];
    let mut traj = InMemoryTraj::new(vec![frame]);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 4);
            assert!(data[0] < 0.1);
            assert!(data[1] > 0.9);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn toroidal_diffusion_plan_transition_counts() {
    let mut system = build_four_resid_system();
    let sel = system
        .select("resid 1 or resid 2 or resid 3 or resid 4")
        .unwrap();
    let mut plan = ToroidalDiffusionPlan::new(sel)
        .with_emit_transitions(true)
        .with_store_transition_states(true)
        .with_transition_lag(1);

    let frame_for_angle = |deg: f32| -> Vec<[f32; 4]> {
        let rad = deg.to_radians();
        vec![
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, rad.cos(), rad.sin(), 1.0],
        ]
    };

    let mut traj = InMemoryTraj::new(vec![frame_for_angle(180.0), frame_for_angle(0.0)]);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 4);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[5] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
    let counts = plan.transition_counts_flat();
    assert_eq!(counts[1], 1);
    let probs = plan.transition_matrix_flat();
    assert!((probs[1] - 1.0).abs() < 1e-6);
    assert!((plan.transition_rate() - 1.0).abs() < 1e-6);
    assert_eq!(plan.transition_state_rows(), 2);
    assert_eq!(plan.transition_state_cols(), 1);
    assert_eq!(plan.transition_states_flat(), &[0, 1]);
}

#[test]
fn gist_grid_plan_basic() {
    let mut system = build_four_resid_system();
    let solute = system.select("resid 1").unwrap();
    let mut plan =
        GistGridPlan::new_auto(vec![1], vec![2], vec![3], vec![1], solute, 1.0, 0.0, 4).unwrap();

    let frame = vec![
        [0.0, 0.0, 0.0, 1.0], // solute center
        [0.2, 0.0, 0.0, 1.0], // oxygen
        [0.3, 0.0, 0.0, 1.0], // hydrogen 1
        [0.3, 0.0, 0.0, 1.0], // hydrogen 2
    ];
    let mut traj = InMemoryTraj::new(vec![frame]);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 5);
            assert_eq!(data[0], 1.0);
            assert_eq!(data[4], 1.0);
        }
        _ => panic!("unexpected output"),
    }
    assert_eq!(plan.n_frames(), 1);
    assert_eq!(plan.dims(), [1, 1, 1]);
}

#[test]
fn gist_direct_plan_frame_sparse_basic() {
    let system = build_four_resid_system();
    let mut plan = GistDirectPlan::new_auto(
        vec![1, 2],                // oxygen indices
        vec![1, 2],                // h1 (unused when orientation_valid=0)
        vec![1, 2],                // h2 (unused when orientation_valid=0)
        vec![0, 0],                // orientation_valid
        vec![0, 1, 2],             // water offsets
        vec![1, 2],                // water atom indices
        vec![0],                   // solute atom index
        vec![1.0, -1.0, 1.0, 0.0], // charges
        vec![0.0, 0.0, 0.0, 0.0],  // sigmas
        vec![0.0, 0.0, 0.0, 0.0],  // epsilons
        vec![],                    // exceptions
        10.0,                      // spacing
        0.0,                       // padding
        4,                         // orientation bins
        100.0,                     // cutoff
        false,                     // periodic
    )
    .unwrap()
    .with_record_frame_energies(true);

    let frame = vec![
        [0.0, 0.0, 0.0, 1.0], // solute
        [1.0, 0.0, 0.0, 1.0], // water 1
        [2.0, 0.0, 0.0, 1.0], // water 2
        [0.0, 0.0, 0.0, 1.0], // unused
    ];
    let mut traj = InMemoryTraj::new(vec![frame]);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 5);
            assert_eq!(data[0], 2.0);
        }
        _ => panic!("unexpected output"),
    }
    assert_eq!(plan.n_frames(), 1);
    assert_eq!(plan.frame_offsets(), &[0, 1]);
    assert_eq!(plan.frame_cells(), &[0]);
    assert_eq!(plan.frame_sw().len(), 1);
    assert_eq!(plan.frame_ww().len(), 1);
    assert!((plan.frame_direct_sw()[0] - plan.direct_sw_total()).abs() < 1e-6);
    assert!((plan.frame_direct_ww()[0] - plan.direct_ww_total()).abs() < 1e-6);
}

#[test]
fn gist_direct_plan_native_pme_totals_capture_out_of_grid_waters() {
    let system = build_four_resid_system();
    let mut plan = GistDirectPlan::new_auto(
        vec![1, 2],                // oxygen indices
        vec![1, 2],                // h1 (unused when orientation_valid=0)
        vec![1, 2],                // h2 (unused when orientation_valid=0)
        vec![0, 0],                // orientation_valid
        vec![0, 1, 2],             // water offsets
        vec![1, 2],                // water atom indices
        vec![0],                   // solute atom index
        vec![1.0, 0.0, -1.0, 0.0], // charges
        vec![0.0, 0.0, 0.0, 0.0],  // sigmas
        vec![0.0, 0.0, 0.0, 0.0],  // epsilons
        vec![],                    // exceptions
        1.0,                       // spacing
        0.0,                       // padding
        4,                         // orientation bins
        2.0,                       // cutoff
        false,                     // periodic
    )
    .unwrap()
    .with_record_frame_energies(true)
    .with_record_pme_frame_totals(true);

    let frame = vec![
        [0.0, 0.0, 0.0, 1.0], // solute
        [0.2, 0.0, 0.0, 1.0], // water 1 (inside voxel; charge=0)
        [1.5, 0.0, 0.0, 1.0], // water 2 (outside voxel; contributes to PME totals)
        [0.0, 0.0, 0.0, 1.0], // unused
    ];
    let mut traj = InMemoryTraj::new(vec![frame]);
    let mut exec = Executor::new(system);
    let _ = exec.run_plan(&mut plan, &mut traj).unwrap();

    assert_eq!(plan.n_frames(), 1);
    assert_eq!(plan.frame_direct_sw().len(), 1);
    assert_eq!(plan.frame_pme_sw().len(), 1);
    assert!(plan.frame_direct_sw()[0].abs() < 1e-6);
    assert!(plan.frame_pme_sw()[0].abs() > 1e-3);
}

#[test]
fn gist_direct_plan_periodic_triclinic_basic() {
    let system = build_four_resid_system();
    let mut plan = GistDirectPlan::new_auto(
        vec![1, 2],                // oxygen indices
        vec![1, 2],                // h1
        vec![1, 2],                // h2
        vec![0, 0],                // orientation_valid
        vec![0, 1, 2],             // water offsets
        vec![1, 2],                // water atom indices
        vec![0],                   // solute atom index
        vec![1.0, -1.0, 1.0, 0.0], // charges
        vec![0.0, 0.0, 0.0, 0.0],  // sigmas
        vec![0.0, 0.0, 0.0, 0.0],  // epsilons
        vec![],                    // exceptions
        10.0,                      // spacing
        0.0,                       // padding
        4,                         // orientation bins
        100.0,                     // cutoff
        true,                      // periodic
    )
    .unwrap();

    let frame = vec![
        [0.0, 0.0, 0.0, 1.0], // solute
        [1.0, 0.0, 0.0, 1.0], // water 1
        [2.0, 0.0, 0.0, 1.0], // water 2
        [0.0, 0.0, 0.0, 1.0], // unused
    ];
    let box_ = Box3::Triclinic {
        m: [
            10.0, 0.5, 0.0, //
            0.0, 10.0, 0.0, //
            0.0, 0.0, 10.0,
        ],
    };
    let mut traj = InMemoryTrajWithBox::new(vec![frame], box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 5);
            assert_eq!(data[0], 2.0);
        }
        _ => panic!("unexpected output"),
    }
    assert_eq!(plan.n_frames(), 1);
}

#[cfg(feature = "cuda")]
#[test]
fn gist_direct_plan_native_pme_totals_cuda_matches_cpu() {
    let gpu = match std::panic::catch_unwind(|| traj_gpu::GpuContext::new(0)) {
        Ok(Ok(ctx)) => ctx,
        _ => return,
    };
    let build_plan = || {
        GistDirectPlan::new_auto(
            vec![1, 2],                // oxygen indices
            vec![1, 2],                // h1
            vec![1, 2],                // h2
            vec![0, 0],                // orientation_valid
            vec![0, 1, 2],             // water offsets
            vec![1, 2],                // water atom indices
            vec![0],                   // solute atom index
            vec![1.0, 0.0, -1.0, 0.0], // charges
            vec![0.0, 0.0, 0.0, 0.0],  // sigmas
            vec![0.0, 0.0, 0.0, 0.0],  // epsilons
            vec![],                    // exceptions
            1.0,                       // spacing
            0.0,                       // padding
            4,                         // orientation bins
            2.0,                       // cutoff
            false,                     // periodic
        )
        .unwrap()
        .with_record_frame_energies(true)
        .with_record_pme_frame_totals(true)
    };

    let frames = vec![
        vec![
            [0.0, 0.0, 0.0, 1.0], // solute
            [0.2, 0.0, 0.0, 1.0], // water 1
            [1.5, 0.0, 0.0, 1.0], // water 2
            [0.0, 0.0, 0.0, 1.0], // unused
        ],
        vec![
            [0.0, 0.0, 0.0, 1.0], // solute
            [0.3, 0.0, 0.0, 1.0], // water 1
            [1.4, 0.0, 0.0, 1.0], // water 2
            [0.0, 0.0, 0.0, 1.0], // unused
        ],
    ];

    let mut plan_cpu = build_plan();
    let mut plan_gpu = build_plan();
    let mut traj_cpu = InMemoryTraj::new(frames.clone());
    let mut traj_gpu = InMemoryTraj::new(frames);
    let mut exec_cpu = Executor::new(build_four_resid_system());
    let mut exec_gpu = Executor::new(build_four_resid_system()).with_device(Device::Cuda(gpu));
    let _ = exec_cpu.run_plan(&mut plan_cpu, &mut traj_cpu).unwrap();
    let _ = exec_gpu.run_plan(&mut plan_gpu, &mut traj_gpu).unwrap();

    assert_eq!(plan_cpu.frame_pme_sw().len(), plan_gpu.frame_pme_sw().len());
    assert_eq!(plan_cpu.frame_pme_ww().len(), plan_gpu.frame_pme_ww().len());
    for (a, b) in plan_cpu
        .frame_pme_sw()
        .iter()
        .zip(plan_gpu.frame_pme_sw().iter())
    {
        assert!((a - b).abs() < 1e-4);
    }
    for (a, b) in plan_cpu
        .frame_pme_ww()
        .iter()
        .zip(plan_gpu.frame_pme_ww().iter())
    {
        assert!((a - b).abs() < 1e-4);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn gist_direct_plan_periodic_triclinic_cuda_matches_cpu() {
    let gpu = match std::panic::catch_unwind(|| traj_gpu::GpuContext::new(0)) {
        Ok(Ok(ctx)) => ctx,
        _ => return,
    };

    let build_plan = || {
        GistDirectPlan::new_auto(
            vec![1, 2],                // oxygen indices
            vec![1, 2],                // h1
            vec![1, 2],                // h2
            vec![0, 0],                // orientation_valid
            vec![0, 1, 2],             // water offsets
            vec![1, 2],                // water atom indices
            vec![0],                   // solute atom index
            vec![1.0, -1.0, 1.0, 0.0], // charges
            vec![0.0, 0.0, 0.0, 0.0],  // sigmas
            vec![0.0, 0.0, 0.0, 0.0],  // epsilons
            vec![],                    // exceptions
            10.0,                      // spacing
            0.0,                       // padding
            4,                         // orientation bins
            100.0,                     // cutoff
            true,                      // periodic
        )
        .unwrap()
        .with_record_frame_energies(true)
    };

    let frames = vec![
        vec![
            [0.0, 0.0, 0.0, 1.0], // solute
            [1.0, 0.0, 0.0, 1.0], // water 1
            [2.0, 0.0, 0.0, 1.0], // water 2
            [0.0, 0.0, 0.0, 1.0], // unused
        ],
        vec![
            [0.0, 0.0, 0.0, 1.0], // solute
            [1.2, 0.1, 0.0, 1.0], // water 1
            [2.1, 0.0, 0.2, 1.0], // water 2
            [0.0, 0.0, 0.0, 1.0], // unused
        ],
    ];
    let box_ = Box3::Triclinic {
        m: [
            10.0, 0.5, 0.0, //
            0.0, 10.0, 0.2, //
            0.0, 0.0, 10.0,
        ],
    };

    let mut plan_cpu = build_plan();
    let mut plan_gpu = build_plan();

    let mut traj_cpu = InMemoryTrajWithBox::new(frames.clone(), box_);
    let mut traj_gpu = InMemoryTrajWithBox::new(frames, box_);

    let mut exec_cpu = Executor::new(build_four_resid_system());
    let mut exec_gpu = Executor::new(build_four_resid_system()).with_device(Device::Cuda(gpu));

    let out_cpu = exec_cpu.run_plan(&mut plan_cpu, &mut traj_cpu).unwrap();
    let out_gpu = exec_gpu.run_plan(&mut plan_gpu, &mut traj_gpu).unwrap();

    match (out_cpu, out_gpu) {
        (
            PlanOutput::Matrix {
                data: cpu_data,
                rows: cpu_rows,
                cols: cpu_cols,
            },
            PlanOutput::Matrix {
                data: gpu_data,
                rows: gpu_rows,
                cols: gpu_cols,
            },
        ) => {
            assert_eq!(cpu_rows, gpu_rows);
            assert_eq!(cpu_cols, gpu_cols);
            assert_eq!(cpu_data.len(), gpu_data.len());
            for (a, b) in cpu_data.iter().zip(gpu_data.iter()) {
                assert!((a - b).abs() < 1e-4);
            }
        }
        _ => panic!("unexpected output"),
    }

    assert_eq!(plan_cpu.n_frames(), plan_gpu.n_frames());
    assert_eq!(plan_cpu.frame_offsets(), plan_gpu.frame_offsets());
    assert_eq!(plan_cpu.frame_cells(), plan_gpu.frame_cells());
    assert_eq!(plan_cpu.frame_sw().len(), plan_gpu.frame_sw().len());
    assert_eq!(plan_cpu.frame_ww().len(), plan_gpu.frame_ww().len());
    for (a, b) in plan_cpu.frame_sw().iter().zip(plan_gpu.frame_sw().iter()) {
        assert!((a - b).abs() < 1e-4);
    }
    for (a, b) in plan_cpu.frame_ww().iter().zip(plan_gpu.frame_ww().iter()) {
        assert!((a - b).abs() < 1e-4);
    }
    assert!((plan_cpu.direct_sw_total() - plan_gpu.direct_sw_total()).abs() < 1e-4);
    assert!((plan_cpu.direct_ww_total() - plan_gpu.direct_ww_total()).abs() < 1e-4);
    assert_eq!(plan_cpu.energy_sw().len(), plan_gpu.energy_sw().len());
    assert_eq!(plan_cpu.energy_ww().len(), plan_gpu.energy_ww().len());
    for (a, b) in plan_cpu.energy_sw().iter().zip(plan_gpu.energy_sw().iter()) {
        assert!((a - b).abs() < 1e-4);
    }
    for (a, b) in plan_cpu.energy_ww().iter().zip(plan_gpu.energy_ww().iter()) {
        assert!((a - b).abs() < 1e-4);
    }
}

#[test]
fn dssp_plan_basic_shape_and_labels() {
    let mut interner = StringInterner::new();
    let n = interner.intern_upper("N");
    let ca = interner.intern_upper("CA");
    let c = interner.intern_upper("C");
    let ala = interner.intern_upper("ALA");
    let gly = interner.intern_upper("GLY");
    let atoms = AtomTable {
        name_id: vec![n, ca, c, n, ca, c],
        resname_id: vec![ala, ala, ala, gly, gly, gly],
        resid: vec![1, 1, 1, 2, 2, 2],
        chain_id: vec![0, 0, 0, 0, 0, 0],
        element_id: vec![0; 6],
        mass: vec![1.0; 6],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 1.0],
        [5.0, 0.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = DsspPlan::new(sel);
    let frame = vec![[0.0, 0.0, 0.0, 1.0]; 6];
    let mut traj = InMemoryTraj::new(vec![frame]);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert_eq!(data, vec![0.0, 0.0]);
        }
        _ => panic!("unexpected output"),
    }
    assert_eq!(plan.labels(), &["ALA:1".to_string(), "GLY:2".to_string()]);
}

#[test]
fn multipucker_plan_basic() {
    let mut system = build_plane_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = MultiPuckerPlan::new(sel, 4);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 4);
            assert_eq!(data.len(), 4);
            let sum: f32 = data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn multipucker_histogram_plan_basic() {
    let mut system = build_plane_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = MultiPuckerPlan::new(sel, 2)
        .with_mode(MultiPuckerMode::Histogram)
        .with_range_max(Some(1.0))
        .with_normalize(false);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert_eq!(data.len(), 2);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn multipucker_histogram_plan_auto_range() {
    let mut system = build_plane_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = MultiPuckerPlan::new(sel, 2)
        .with_mode(MultiPuckerMode::Histogram)
        .with_range_max(None)
        .with_normalize(false);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert_eq!(data.len(), 2);
            assert!((data[0] - 0.0).abs() < 1e-6);
            assert!((data[1] - 3.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn multipucker_histogram_legacy_parity_zero_radius() {
    let mut system_legacy = build_plane_system();
    let sel_legacy = system_legacy.select("name CA").unwrap();
    let mut plan_legacy = MultiPuckerPlan::new(sel_legacy, 4).with_mode(MultiPuckerMode::Legacy);
    let zero_frame = vec![
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let mut traj_legacy = InMemoryTraj::new(vec![zero_frame.clone()]);
    let mut exec_legacy = Executor::new(system_legacy);
    let out_legacy = exec_legacy
        .run_plan(&mut plan_legacy, &mut traj_legacy)
        .unwrap();

    let mut system_hist = build_plane_system();
    let sel_hist = system_hist.select("name CA").unwrap();
    let mut plan_hist = MultiPuckerPlan::new(sel_hist, 4)
        .with_mode(MultiPuckerMode::Histogram)
        .with_range_max(Some(1.0))
        .with_normalize(true);
    let mut traj_hist = InMemoryTraj::new(vec![zero_frame]);
    let mut exec_hist = Executor::new(system_hist);
    let out_hist = exec_hist.run_plan(&mut plan_hist, &mut traj_hist).unwrap();

    match (out_legacy, out_hist) {
        (
            PlanOutput::Matrix {
                data: legacy_data,
                rows: legacy_rows,
                cols: legacy_cols,
            },
            PlanOutput::Matrix {
                data: hist_data,
                rows: hist_rows,
                cols: hist_cols,
            },
        ) => {
            assert_eq!(legacy_rows, 1);
            assert_eq!(hist_rows, 1);
            assert_eq!(legacy_cols, 4);
            assert_eq!(hist_cols, 4);
            assert_eq!(legacy_data, hist_data);
            assert_eq!(legacy_data, vec![1.0, 0.0, 0.0, 0.0]);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn multipucker_histogram_auto_range_matches_explicit() {
    let frame = vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
    ];

    let mut system_auto = build_plane_system();
    let sel_auto = system_auto.select("name CA").unwrap();
    let mut plan_auto = MultiPuckerPlan::new(sel_auto, 2)
        .with_mode(MultiPuckerMode::Histogram)
        .with_range_max(None)
        .with_normalize(false);
    let mut traj_auto = InMemoryTraj::new(vec![frame.clone()]);
    let mut exec_auto = Executor::new(system_auto);
    let out_auto = exec_auto.run_plan(&mut plan_auto, &mut traj_auto).unwrap();

    let mut system_explicit = build_plane_system();
    let sel_explicit = system_explicit.select("name CA").unwrap();
    let mut plan_explicit = MultiPuckerPlan::new(sel_explicit, 2)
        .with_mode(MultiPuckerMode::Histogram)
        .with_range_max(Some(1.0))
        .with_normalize(false);
    let mut traj_explicit = InMemoryTraj::new(vec![frame]);
    let mut exec_explicit = Executor::new(system_explicit);
    let out_explicit = exec_explicit
        .run_plan(&mut plan_explicit, &mut traj_explicit)
        .unwrap();

    match (out_auto, out_explicit) {
        (
            PlanOutput::Matrix {
                data: auto_data,
                rows: auto_rows,
                cols: auto_cols,
            },
            PlanOutput::Matrix {
                data: explicit_data,
                rows: explicit_rows,
                cols: explicit_cols,
            },
        ) => {
            assert_eq!(auto_rows, 1);
            assert_eq!(explicit_rows, 1);
            assert_eq!(auto_cols, 2);
            assert_eq!(explicit_cols, 2);
            assert_eq!(auto_data, explicit_data);
            assert_eq!(auto_data, vec![1.0, 2.0]);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn pucker_plan_basic() {
    let mut system = build_plane_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = PuckerPlan::new(sel);
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
            assert!(vals[0] > 0.0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn pucker_plan_return_phase() {
    let mut system = build_plane_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = PuckerPlan::new(sel)
        .with_metric(PuckerMetric::Amplitude)
        .with_return_phase(true);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.2, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert_eq!(data.len(), 2);
            assert!(data[0].is_finite());
            assert!(data[1].is_finite());
            assert!(data[1] >= 0.0 && data[1] <= 360.0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn randomize_ions_plan_basic() {
    let mut interner = StringInterner::new();
    let ion = interner.intern_upper("NA");
    let ow = interner.intern_upper("OW");
    let hw = interner.intern_upper("HW");
    let ca = interner.intern_upper("CA");
    let res_ion = interner.intern_upper("ION");
    let res_wat = interner.intern_upper("WAT");
    let res_sol = interner.intern_upper("SOL");
    let atoms = AtomTable {
        name_id: vec![ion, ow, hw, ca],
        resname_id: vec![res_ion, res_wat, res_wat, res_sol],
        resid: vec![1, 2, 2, 3],
        chain_id: vec![0, 0, 0, 0],
        element_id: vec![0, 0, 0, 0],
        mass: vec![1.0, 1.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [5.0, 5.0, 5.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let sel = system.select("name NA").unwrap();
    let around = system.select("resid 3").unwrap();
    let mut plan = RandomizeIonsPlan::new(sel, 123).with_around(Some(around), 0.5, 0.0, false);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [5.0, 5.0, 5.0, 1.0],
    ]];
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
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 12);
            let coords: Vec<[f32; 3]> = vals.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            assert_eq!(coords.len(), 4);
            assert!((coords[0][0] - 1.0).abs() < 1e-6);
            assert!((coords[0][1]).abs() < 1e-6);
            assert!((coords[1][0]).abs() < 1e-6);
            assert!((coords[2][0]).abs() < 1e-6);
            assert!((coords[2][1] - 1.0).abs() < 1e-6);
            assert!((coords[3][0] - 5.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn randomize_ions_plan_triclinic() {
    let mut interner = StringInterner::new();
    let ion = interner.intern_upper("NA");
    let ow = interner.intern_upper("OW");
    let hw = interner.intern_upper("HW");
    let res_ion = interner.intern_upper("ION");
    let res_wat = interner.intern_upper("WAT");
    let atoms = AtomTable {
        name_id: vec![ion, ow, hw],
        resname_id: vec![res_ion, res_wat, res_wat],
        resid: vec![1, 2, 2],
        chain_id: vec![0, 0, 0],
        element_id: vec![0, 0, 0],
        mass: vec![1.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let sel = system.select("name NA").unwrap();
    let mut plan = RandomizeIonsPlan::new(sel, 7);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]];
    let mut traj = InMemoryTrajWithBox::new(
        frames,
        Box3::Triclinic {
            m: [2.0, 0.0, 0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 4.0],
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 9);
            let coords: Vec<[f32; 3]> = vals.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
            assert_eq!(coords.len(), 3);
            assert!((coords[0][0] - 1.0).abs() < 1e-6);
            assert!((coords[1][0]).abs() < 1e-6);
            assert!((coords[2][1] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn image_plan_triclinic_selection() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = ImagePlan::new(sel);
    let frames = vec![vec![[2.9, 1.5, 1.0, 1.0], [5.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTrajWithBox::new(
        frames,
        Box3::Triclinic {
            m: [2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 4.0],
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 0.9).abs() < 1e-6);
            assert!((vals[1] - 1.5).abs() < 1e-6);
            assert!((vals[2] - 1.0).abs() < 1e-6);
            assert!((vals[3] - 5.0).abs() < 1e-6);
            assert!((vals[4]).abs() < 1e-6);
            assert!((vals[5]).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn fiximagedbonds_plan_basic() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = FixImageBondsPlan::new(sel);
    let frames = vec![vec![[2.5, -1.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTrajWithBox::new(
        frames,
        Box3::Orthorhombic {
            lx: 3.0,
            ly: 3.0,
            lz: 3.0,
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn count_in_voxel_basic() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = CountInVoxelPlan::new(sel.clone(), sel, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
    let frames = vec![vec![[0.2, 0.2, 0.2, 1.0], [0.2, 0.2, 0.2, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Grid(grid) => {
            assert_eq!(grid.dims, [2, 2, 2]);
            assert_eq!(grid.mean.len(), 8);
            assert!((grid.mean[0] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn distance_to_point_plan_basic() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = DistanceToPointPlan::new(sel, [0.0, 0.0, 0.0], PbcMode::None);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [3.0, 4.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert!((data[0]).abs() < 1e-6);
            assert!((data[1] - 5.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn align_plan_translation_only() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = AlignPlan::new(sel, ReferenceMode::Topology, false, true);
    let frames = vec![vec![[10.0, 0.0, 0.0, 1.0], [11.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 12);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[4] - 1.0).abs() < 1e-6);
            assert!((data[8] - 1.0).abs() < 1e-6);
            let tx = data[9];
            assert!((tx + 10.0).abs() < 1e-5);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn angle_plan_basic() {
    let mut system = build_four_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let sel_c = system.select("resid 3").unwrap();
    let mut plan = AnglePlan::new(sel_a, sel_b, sel_c, false, PbcMode::None, true);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 90.0).abs() < 1e-4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn dihedral_plan_basic() {
    let mut system = build_four_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let sel_c = system.select("resid 3").unwrap();
    let sel_d = system.select("resid 4").unwrap();
    let mut plan = DihedralPlan::new(
        sel_a,
        sel_b,
        sel_c,
        sel_d,
        false,
        PbcMode::None,
        true,
        false,
    );
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] + 90.0).abs() < 1e-4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn mindist_plan_basic() {
    let mut system = build_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 1").unwrap();
    let mut plan = MindistPlan::new(sel_a, sel_b, PbcMode::None);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [3.0, 4.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0]).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn closest_atom_plan_basic() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = ClosestAtomPlan::new(sel, [0.1, 0.0, 0.0], PbcMode::None);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [3.0, 4.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 0.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn search_neighbors_plan_basic() {
    let mut system = build_system();
    let target = system.select("resid 1").unwrap();
    let probe = system.select("resid 1").unwrap();
    let mut plan = SearchNeighborsPlan::new(target, probe, 0.5, PbcMode::None);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [3.0, 4.0, 0.0, 1.0]]];
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
fn native_contacts_plan_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan =
        NativeContactsPlan::new(sel.clone(), sel, ReferenceMode::Frame0, 1.5, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 2);
            assert!((vals[0] - 1.0).abs() < 1e-6);
            assert!((vals[1]).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn strip_plan_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = StripPlan::new(sel);
    let frames = vec![vec![[1.0, 2.0, 3.0, 1.0], [9.0, 9.0, 9.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 3);
            assert!((vals[0] - 1.0).abs() < 1e-6);
            assert!((vals[1] - 2.0).abs() < 1e-6);
            assert!((vals[2] - 3.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn scale_plan_basic() {
    let system = build_system();
    let mut plan = ScalePlan::new(2.0);
    let frames = vec![vec![[1.0, -2.0, 3.0, 1.0], [0.5, 0.0, -1.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 2.0).abs() < 1e-6);
            assert!((vals[1] + 4.0).abs() < 1e-6);
            assert!((vals[2] - 6.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn rotate_plan_basic() {
    let system = build_system();
    let rot = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    let mut plan = RotatePlan::new(rot);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[3]).abs() < 1e-6);
            assert!((vals[4] - 1.0).abs() < 1e-6);
            assert!((vals[5]).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}
