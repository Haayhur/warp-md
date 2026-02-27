#[test]
fn rotation_matrix_plan_basic() {
    let mut system = build_plane_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = RotationMatrixPlan::new(sel, ReferenceMode::Topology, false);
    let mut frame = system.positions0.clone().unwrap();
    for p in frame.iter_mut() {
        p[0] += 10.0;
    }
    let frames = vec![frame];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 9);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[4] - 1.0).abs() < 1e-6);
            assert!((data[8] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn bfactors_plan_basic() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = BfactorsPlan::new(sel);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 2);
            let expected = (8.0 * std::f64::consts::PI * std::f64::consts::PI / 3.0) * 0.25;
            assert!((vals[0] as f64 - expected).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn multidihedral_plan_basic() {
    let mut system = build_four_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let sel_c = system.select("resid 3").unwrap();
    let sel_d = system.select("resid 4").unwrap();
    let defs = vec![
        (sel_a.clone(), sel_b.clone(), sel_c.clone(), sel_d.clone()),
        (sel_a, sel_b, sel_c, sel_d),
    ];
    let mut plan = MultiDihedralPlan::new(defs, false, PbcMode::None, true, false);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert!((data[0] + 90.0).abs() < 1e-4);
            assert!((data[1] + 90.0).abs() < 1e-4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn multidihedral_selected_io_path() {
    struct SelectedOnlyTraj {
        n_atoms: usize,
        frames: Vec<Vec<[f32; 4]>>,
        cursor: usize,
    }

    impl SelectedOnlyTraj {
        fn new(frames: Vec<Vec<[f32; 4]>>) -> Self {
            let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
            Self {
                n_atoms,
                frames,
                cursor: 0,
            }
        }
    }

    impl TrajReader for SelectedOnlyTraj {
        fn n_atoms(&self) -> usize {
            self.n_atoms
        }

        fn n_frames_hint(&self) -> Option<usize> {
            Some(self.frames.len())
        }

        fn read_chunk(
            &mut self,
            _max_frames: usize,
            _out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            Err(traj_core::error::TrajError::Unsupported(
                "selected path expected".into(),
            ))
        }

        fn read_chunk_selected(
            &mut self,
            max_frames: usize,
            selection: &[u32],
            out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            out.reset(selection.len(), max_frames);
            let mut count = 0usize;
            while self.cursor < self.frames.len() && count < max_frames {
                let src = &self.frames[self.cursor];
                let dst = out.start_frame(Box3::None, None);
                for (dst_atom, &src_idx) in dst.iter_mut().zip(selection.iter()) {
                    *dst_atom = src[src_idx as usize];
                }
                self.cursor += 1;
                count += 1;
            }
            Ok(count)
        }
    }

    let mut system = build_four_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let sel_c = system.select("resid 3").unwrap();
    let sel_d = system.select("resid 4").unwrap();
    let defs = vec![
        (sel_a.clone(), sel_b.clone(), sel_c.clone(), sel_d.clone()),
        (sel_a, sel_b, sel_c, sel_d),
    ];
    let mut plan = MultiDihedralPlan::new(defs, false, PbcMode::None, true, false);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = SelectedOnlyTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert!((data[0] + 90.0).abs() < 1e-4);
            assert!((data[1] + 90.0).abs() < 1e-4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn closest_plan_basic() {
    let mut system = build_four_resid_system();
    let target = system.select("resid 1").unwrap();
    let probe = system.select("resid 2:4").unwrap();
    let mut plan = ClosestPlan::new(target, probe, 2, PbcMode::None);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn replicate_cell_plan_basic() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = ReplicateCellPlan::new(sel, [2, 1, 1]);
    let chunk = traj_core::frame::FrameChunk {
        n_atoms: 2,
        n_frames: 1,
        coords: vec![[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]],
        box_: vec![traj_core::frame::Box3::Orthorhombic {
            lx: 10.0,
            ly: 10.0,
            lz: 10.0,
        }],
        time_ps: None,
    };
    let device = Device::cpu();
    plan.init(&system, &device).unwrap();
    plan.process_chunk(&chunk, &system, &device).unwrap();
    let out = plan.finalize().unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 12);
            assert!((vals[0] - 1.0).abs() < 1e-6);
            assert!((vals[3] - 4.0).abs() < 1e-6);
            assert!((vals[6] - 11.0).abs() < 1e-6);
            assert!((vals[9] - 14.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn replicate_cell_plan_triclinic() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = ReplicateCellPlan::new(sel, [1, 2, 1]);
    let chunk = traj_core::frame::FrameChunk {
        n_atoms: 1,
        n_frames: 1,
        coords: vec![[0.0, 0.0, 0.0, 1.0]],
        box_: vec![traj_core::frame::Box3::Triclinic {
            m: [2.0, 0.0, 0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 4.0],
        }],
        time_ps: None,
    };
    let device = Device::cpu();
    plan.init(&system, &device).unwrap();
    plan.process_chunk(&chunk, &system, &device).unwrap();
    let out = plan.finalize().unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 0.0).abs() < 1e-6);
            assert!((vals[1] - 0.0).abs() < 1e-6);
            assert!((vals[2] - 0.0).abs() < 1e-6);
            assert!((vals[3] - 0.5).abs() < 1e-6);
            assert!((vals[4] - 3.0).abs() < 1e-6);
            assert!((vals[5] - 0.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn matrix_plan_distance_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = MatrixPlan::new(sel, MatrixMode::Distance, PbcMode::None);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [3.0, 4.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 2);
            assert!((data[1] - 5.0).abs() < 1e-6);
            assert!((data[2] - 5.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn pca_plan_basic() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = PcaPlan::new(sel, 2, false);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Pca(pca) => {
            assert_eq!(pca.n_components, 2);
            assert_eq!(pca.n_features, 6);
            assert_eq!(pca.eigenvalues.len(), 2);
            assert_eq!(pca.eigenvectors.len(), 12);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn trajectory_cluster_dbscan_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = TrajectoryClusterPlan::new(
        sel,
        ClusterMethod::Dbscan {
            eps: 0.1,
            min_samples: 2,
        },
    );
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.1, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Clustering(c) => {
            assert_eq!(c.labels, vec![0, 0, -1]);
            assert_eq!(c.centroids, vec![0u32]);
            assert_eq!(c.sizes, vec![2u32]);
            assert_eq!(c.method, "dbscan");
            assert_eq!(c.n_frames, 3);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn trajectory_cluster_kmeans_seed_deterministic() {
    let mut system_a = build_two_resid_system();
    let mut system_b = build_two_resid_system();
    let sel_a = system_a.select("resid 1:2").unwrap();
    let sel_b = system_b.select("resid 1:2").unwrap();
    let mut plan_a = TrajectoryClusterPlan::new(
        sel_a,
        ClusterMethod::Kmeans {
            n_clusters: 2,
            max_iter: 64,
            tol: 1.0e-6,
            seed: 42,
        },
    );
    let mut plan_b = TrajectoryClusterPlan::new(
        sel_b,
        ClusterMethod::Kmeans {
            n_clusters: 2,
            max_iter: 64,
            tol: 1.0e-6,
            seed: 42,
        },
    );
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.2, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [5.2, 0.0, 0.0, 1.0]],
    ];
    let mut traj_a = InMemoryTraj::new(frames.clone());
    let mut traj_b = InMemoryTraj::new(frames);
    let mut exec_a = Executor::new(system_a);
    let mut exec_b = Executor::new(system_b);

    let out_a = exec_a.run_plan(&mut plan_a, &mut traj_a).unwrap();
    let out_b = exec_b.run_plan(&mut plan_b, &mut traj_b).unwrap();
    match (out_a, out_b) {
        (PlanOutput::Clustering(a), PlanOutput::Clustering(b)) => {
            assert_eq!(a.labels, b.labels);
            assert_eq!(a.centroids, b.centroids);
            assert_eq!(a.sizes, b.sizes);
            assert_eq!(a.method, "kmeans");
            let mut sizes = a.sizes.clone();
            sizes.sort_unstable();
            assert_eq!(sizes, vec![2u32, 2u32]);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn trajectory_cluster_memory_budget_rejects_large_matrix() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = TrajectoryClusterPlan::new(
        sel,
        ClusterMethod::Dbscan {
            eps: 0.2,
            min_samples: 2,
        },
    )
    .with_memory_budget_bytes(Some(8));
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.1, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.2, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.3, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let err = match exec.run_plan(&mut plan, &mut traj) {
        Ok(_) => panic!("expected memory budget error"),
        Err(err) => err,
    };
    assert!(err.to_string().contains("memory_budget_bytes"));
}

#[test]
fn trajectory_cluster_selected_frame_subset() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = TrajectoryClusterPlan::new(
        sel,
        ClusterMethod::Kmeans {
            n_clusters: 1,
            max_iter: 32,
            tol: 1.0e-6,
            seed: 9,
        },
    );
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.5, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec
        .run_plan_on_selected_frames(&mut plan, &mut traj, &[1, 3])
        .unwrap();
    match out {
        PlanOutput::Clustering(c) => {
            assert_eq!(c.n_frames, 2);
            assert_eq!(c.labels.len(), 2);
            assert_eq!(c.method, "kmeans");
        }
        _ => panic!("unexpected output"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn trajectory_cluster_dbscan_cuda_device() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = TrajectoryClusterPlan::new(
        sel,
        ClusterMethod::Dbscan {
            eps: 0.1,
            min_samples: 2,
        },
    );
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.1, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut exec = Executor::new(system).with_device_spec("cuda:0")?;
        exec.run_plan(&mut plan, &mut traj)
    }));
    let out = match run {
        Ok(Ok(out)) => out,
        Ok(Err(err)) => {
            eprintln!(
                "CUDA unavailable; skipping trajectory_cluster_dbscan_cuda_device ({err})"
            );
            return;
        }
        Err(_) => {
            eprintln!(
                "CUDA runtime unavailable (e.g., missing nvrtc); skipping trajectory_cluster_dbscan_cuda_device"
            );
            return;
        }
    };
    match out {
        PlanOutput::Clustering(c) => {
            assert_eq!(c.labels, vec![0, 0, -1]);
            assert_eq!(c.centroids, vec![0u32]);
            assert_eq!(c.sizes, vec![2u32]);
            assert_eq!(c.method, "dbscan");
            assert_eq!(c.n_frames, 3);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn vector_plan_basic() {
    let mut system = build_two_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let mut plan = VectorPlan::new(sel_a, sel_b, false, PbcMode::None);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 3);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 2.0).abs() < 1e-6);
            assert!((data[2] - 3.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn get_velocity_plan_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = GetVelocityPlan::new(sel);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 6);
            assert_eq!(&data[0..6], &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            assert!((data[6] - 1.0).abs() < 1e-6);
            assert!((data[9] - 0.0).abs() < 1e-6);
            assert!((data[10] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn set_velocity_plan_deterministic() {
    let system = build_two_resid_system();
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 1.0, 0.0, 1.0], [2.0, 1.0, 0.0, 1.0]],
    ];
    let mut traj_a = InMemoryTraj::new(frames.clone());
    let mut traj_b = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let sel = exec.system_mut().select("resid 1:2").unwrap();
    let mut plan_a = SetVelocityPlan::new(sel.clone(), 100.0, 7);
    let mut plan_b = SetVelocityPlan::new(sel, 100.0, 7);
    let out_a = exec.run_plan(&mut plan_a, &mut traj_a).unwrap();
    let out_b = exec.run_plan(&mut plan_b, &mut traj_b).unwrap();
    match (out_a, out_b) {
        (
            PlanOutput::Matrix {
                data: a,
                rows: ra,
                cols: ca,
            },
            PlanOutput::Matrix {
                data: b,
                rows: rb,
                cols: cb,
            },
        ) => {
            assert_eq!(ra, 2);
            assert_eq!(ca, 6);
            assert_eq!(ra, rb);
            assert_eq!(ca, cb);
            assert_eq!(a, b);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn mean_structure_plan_basic() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = MeanStructurePlan::new(sel);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 1.0).abs() < 1e-6);
            assert!((vals[3] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn molsurf_plan_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = MolSurfPlan::new(sel).with_algorithm(SurfAlgorithm::Bbox);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [2.0, 3.0, 4.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 52.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn xtalsymm_plan_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = XtalSymmPlan::new(sel, [2, 1, 1]);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]];
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
            assert!((vals[0] - 0.0).abs() < 1e-6);
            assert!((vals[3] - 1.0).abs() < 1e-6);
            assert!((vals[6] - 10.0).abs() < 1e-6);
            assert!((vals[9] - 11.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn xtalsymm_plan_with_symmetry_ops() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let op = [1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, -2.0, 0.0, 0.0, 1.0, 1.5];
    let mut plan = XtalSymmPlan::new(sel, [1, 1, 1]).with_symmetry_ops(Some(vec![op]));
    let frames = vec![vec![[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 6.0).abs() < 1e-6);
            assert!((vals[1] - 0.0).abs() < 1e-6);
            assert!((vals[2] - 4.5).abs() < 1e-6);
            assert!((vals[3] - 9.0).abs() < 1e-6);
            assert!((vals[4] - 3.0).abs() < 1e-6);
            assert!((vals[5] - 7.5).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn make_structure_plan_basic() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = MakeStructurePlan::new(sel);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 1.0).abs() < 1e-6);
            assert!((vals[3] - 3.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn mean_and_make_structure_selected_io_path() {
    struct SelectedOnlyTraj {
        n_atoms: usize,
        frames: Vec<Vec<[f32; 4]>>,
        cursor: usize,
    }

    impl SelectedOnlyTraj {
        fn new(frames: Vec<Vec<[f32; 4]>>) -> Self {
            let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
            Self {
                n_atoms,
                frames,
                cursor: 0,
            }
        }
    }

    impl TrajReader for SelectedOnlyTraj {
        fn n_atoms(&self) -> usize {
            self.n_atoms
        }

        fn n_frames_hint(&self) -> Option<usize> {
            Some(self.frames.len())
        }

        fn read_chunk(
            &mut self,
            _max_frames: usize,
            _out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            Err(traj_core::error::TrajError::Unsupported(
                "selected path expected".into(),
            ))
        }

        fn read_chunk_selected(
            &mut self,
            max_frames: usize,
            selection: &[u32],
            out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            out.reset(selection.len(), max_frames);
            let mut count = 0usize;
            while self.cursor < self.frames.len() && count < max_frames {
                let src = &self.frames[self.cursor];
                let dst = out.start_frame(Box3::None, None);
                for (dst_atom, &src_idx) in dst.iter_mut().zip(selection.iter()) {
                    *dst_atom = src[src_idx as usize];
                }
                self.cursor += 1;
                count += 1;
            }
            Ok(count)
        }
    }

    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 1.0]],
    ];

    let mut system_mean = build_system();
    let sel_mean = system_mean.select("resid 1").unwrap();
    let mut mean = MeanStructurePlan::new(sel_mean);
    let mut traj_mean = SelectedOnlyTraj::new(frames.clone());
    let mut exec_mean = Executor::new(system_mean);
    let out_mean = exec_mean.run_plan(&mut mean, &mut traj_mean).unwrap();
    match out_mean {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 1.0).abs() < 1e-6);
            assert!((vals[3] - 3.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }

    let mut system_make = build_system();
    let sel_make = system_make.select("resid 1").unwrap();
    let mut make = MakeStructurePlan::new(sel_make);
    let mut traj_make = SelectedOnlyTraj::new(frames);
    let mut exec_make = Executor::new(system_make);
    let out_make = exec_make.run_plan(&mut make, &mut traj_make).unwrap();
    match out_make {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 6);
            assert!((vals[0] - 1.0).abs() < 1e-6);
            assert!((vals[3] - 3.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lowestcurve_plan_basic() {
    let mut system = build_two_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let mut plan = LowestCurvePlan::new(sel_a, sel_b, false, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 0.5).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn permute_dihedrals_plan_basic() {
    let mut system = build_four_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let sel_c = system.select("resid 3").unwrap();
    let sel_d = system.select("resid 4").unwrap();
    let defs = vec![(sel_a, sel_b, sel_c, sel_d)];
    let mut plan = PermuteDihedralsPlan::new(defs, false, PbcMode::None, true, false);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 1);
            assert!((data[0] + 90.0).abs() < 1e-4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn distance_rmsd_plan_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = DistanceRmsdPlan::new(sel, ReferenceMode::Frame0, PbcMode::None);
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
            assert!((vals[0]).abs() < 1e-6);
            assert!((vals[1] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn pairwise_distance_plan_basic() {
    let mut system = build_two_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let mut plan = PairwiseDistancePlan::new(sel_a, sel_b, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 1);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn pairwise_distance_selected_io_path() {
    struct SelectedOnlyTraj {
        n_atoms: usize,
        frames: Vec<Vec<[f32; 4]>>,
        cursor: usize,
    }

    impl SelectedOnlyTraj {
        fn new(frames: Vec<Vec<[f32; 4]>>) -> Self {
            let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
            Self {
                n_atoms,
                frames,
                cursor: 0,
            }
        }
    }

    impl TrajReader for SelectedOnlyTraj {
        fn n_atoms(&self) -> usize {
            self.n_atoms
        }

        fn n_frames_hint(&self) -> Option<usize> {
            Some(self.frames.len())
        }

        fn read_chunk(
            &mut self,
            _max_frames: usize,
            _out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            Err(traj_core::error::TrajError::Unsupported(
                "selected path expected".into(),
            ))
        }

        fn read_chunk_selected(
            &mut self,
            max_frames: usize,
            selection: &[u32],
            out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            out.reset(selection.len(), max_frames);
            let mut count = 0usize;
            while count < max_frames && self.cursor < self.frames.len() {
                let frame = &self.frames[self.cursor];
                let dst = out.start_frame(Box3::None, None);
                for (dst_atom, &src_idx) in dst.iter_mut().zip(selection.iter()) {
                    *dst_atom = frame[src_idx as usize];
                }
                self.cursor += 1;
                count += 1;
            }
            Ok(count)
        }
    }

    let mut system = build_two_resid_system();
    let sel_a = system.select("resid 1").unwrap();
    let sel_b = system.select("resid 2").unwrap();
    let mut plan = PairwiseDistancePlan::new(sel_a, sel_b, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = SelectedOnlyTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 1);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn pairwise_rmsd_plan_basic() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = PairwiseRmsdPlan::new(sel, PairwiseMetric::Nofit, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 2);
            let rmsd = (0.5f32).sqrt();
            assert!((data[1] - rmsd).abs() < 1e-6);
            assert!((data[2] - rmsd).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn pairwise_rmsd_selected_io_path() {
    struct SelectedOnlyTraj {
        n_atoms: usize,
        frames: Vec<Vec<[f32; 4]>>,
        cursor: usize,
    }

    impl SelectedOnlyTraj {
        fn new(frames: Vec<Vec<[f32; 4]>>) -> Self {
            let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
            Self {
                n_atoms,
                frames,
                cursor: 0,
            }
        }
    }

    impl TrajReader for SelectedOnlyTraj {
        fn n_atoms(&self) -> usize {
            self.n_atoms
        }

        fn n_frames_hint(&self) -> Option<usize> {
            Some(self.frames.len())
        }

        fn read_chunk(
            &mut self,
            _max_frames: usize,
            _out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            Err(traj_core::error::TrajError::Unsupported(
                "selected path expected".into(),
            ))
        }

        fn read_chunk_selected(
            &mut self,
            max_frames: usize,
            selection: &[u32],
            out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            out.reset(selection.len(), max_frames);
            let mut count = 0usize;
            while count < max_frames && self.cursor < self.frames.len() {
                let frame = &self.frames[self.cursor];
                let dst = out.start_frame(Box3::None, None);
                for (dst_atom, &src_idx) in dst.iter_mut().zip(selection.iter()) {
                    *dst_atom = frame[src_idx as usize];
                }
                self.cursor += 1;
                count += 1;
            }
            Ok(count)
        }
    }

    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = PairwiseRmsdPlan::new(sel, PairwiseMetric::Nofit, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = SelectedOnlyTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 2);
            let rmsd = (0.5f32).sqrt();
            assert!((data[1] - rmsd).abs() < 1e-6);
            assert!((data[2] - rmsd).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn pairwise_rmsd_plan_dme() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = PairwiseRmsdPlan::new(sel, PairwiseMetric::Dme, PbcMode::None);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 2);
            assert_eq!(cols, 2);
            assert!((data[1] - 1.0).abs() < 1e-6);
            assert!((data[2] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn radgyr_tensor_plan_basic() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let mut plan = RadgyrTensorPlan::new(sel, false);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 7);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 1.0).abs() < 1e-6);
            assert!(data[2].abs() < 1e-6);
            assert!(data[3].abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn msd_plan_simple() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = MsdPlan::new(sel, GroupBy::Resid);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            rows, cols, data, ..
        } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 8);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 1.0).abs() < 1e-6);
            assert!((data[6] - 1.0).abs() < 1e-6);
            assert!((data[7] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn msd_plan_atom_grouping_with_types() {
    let mut system = build_system();
    for resid in system.atoms.resid.iter_mut() {
        *resid = 1;
    }
    let sel = system.select("name CA").unwrap();
    let mut plan = MsdPlan::new(sel, GroupBy::Atom).with_group_types(vec![0, 1]);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            rows, cols, data, ..
        } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 12);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 1.0).abs() < 1e-6);
            assert!((data[11] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn msd_plan_rejects_nonuniform_time_spacing() {
    struct TimedTraj {
        n_atoms: usize,
        frames: Vec<Vec<[f32; 4]>>,
        times: Vec<f32>,
        cursor: usize,
    }

    impl TimedTraj {
        fn new(frames: Vec<Vec<[f32; 4]>>, times: Vec<f32>) -> Self {
            let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
            Self {
                n_atoms,
                frames,
                times,
                cursor: 0,
            }
        }
    }

    impl TrajReader for TimedTraj {
        fn n_atoms(&self) -> usize {
            self.n_atoms
        }

        fn n_frames_hint(&self) -> Option<usize> {
            Some(self.frames.len())
        }

        fn read_chunk(
            &mut self,
            max_frames: usize,
            out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            out.reset(self.n_atoms, max_frames.max(1));
            let mut count = 0usize;
            while self.cursor < self.frames.len() && count < max_frames {
                let coords = out.start_frame(Box3::None, self.times.get(self.cursor).copied());
                coords.copy_from_slice(&self.frames[self.cursor]);
                self.cursor += 1;
                count += 1;
            }
            Ok(count)
        }
    }

    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = MsdPlan::new(sel, GroupBy::Resid)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(2);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = TimedTraj::new(frames, vec![0.0, 1.0, 2.5]);
    let mut exec = Executor::new(system);
    match exec.run_plan(&mut plan, &mut traj) {
        Ok(_) => panic!("expected non-uniform time error"),
        Err(err) => assert!(err.to_string().contains("uniform frame spacing")),
    }
}
