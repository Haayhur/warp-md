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
    let mut plan = AtomicCorrelationPlan::new(sel, ReferenceMode::Frame0)
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
fn vanhove_plan_basic() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = VanHovePlan::new(sel, 1.0, 3.0);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::VanHove(output) => {
            assert_eq!(output.rows, 3);
            assert_eq!(output.cols, 3);
            assert_eq!(output.time, vec![0.0, 1.0, 2.0]);
            assert_eq!(output.r, vec![0.0, 1.0, 2.0]);
            assert!((output.matrix[0] - 1.0).abs() < 1e-6);
            assert!((output.matrix[1 * output.cols + 1] - 1.0).abs() < 1e-6);
            assert!((output.matrix[2 * output.cols + 2] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn vanhove_plan_unwraps_pbc_jumps() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = VanHovePlan::new(sel, 1.0, 4.0);
    let frames = vec![vec![[9.5, 0.0, 0.0, 1.0]], vec![[0.5, 0.0, 0.0, 1.0]]];
    let box_ = Box3::Orthorhombic {
        lx: 10.0,
        ly: 10.0,
        lz: 10.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::VanHove(output) => {
            assert_eq!(output.rows, 2);
            assert_eq!(output.cols, 4);
            assert!((output.matrix[1 * output.cols + 1] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn vanhove_plan_keeps_upper_shell_counts() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = VanHovePlan::new(sel, 1.0, 3.0);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0]], vec![[2.6, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::VanHove(output) => {
            assert_eq!(output.cols, 3);
            assert!((output.matrix[1 * output.cols + 2] - 1.0).abs() < 1e-6);
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
    let mut plan = CrossCorrelationPlan::new(sel_a, sel_b, ReferenceMode::Topology, false)
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
    let mut plan = SurfPlan::new(sel).with_algorithm(SurfAlgorithm::Bbox);
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
fn surf_plan_bbox_and_sasa_use_selected_chunks() {
    use std::sync::Arc;

    struct SurfaceSelectedOnlyTraj {
        n_atoms: usize,
        frames: Vec<Vec<[f32; 4]>>,
        selection: Vec<u32>,
        cursor: usize,
        selected_reads: usize,
    }

    impl SurfaceSelectedOnlyTraj {
        fn new(n_atoms: usize, frames: Vec<Vec<[f32; 4]>>, selection: Vec<u32>) -> Self {
            Self {
                n_atoms,
                frames,
                selection,
                cursor: 0,
                selected_reads: 0,
            }
        }
    }

    impl TrajReader for SurfaceSelectedOnlyTraj {
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
            panic!("surface plan should use selected trajectory reads")
        }

        fn read_chunk_selected(
            &mut self,
            max_frames: usize,
            selection: &[u32],
            out: &mut FrameChunkBuilder,
        ) -> TrajResult<usize> {
            assert_eq!(selection, self.selection.as_slice());
            self.selected_reads += 1;
            out.reset(selection.len(), max_frames.max(1));
            if self.cursor >= self.frames.len() {
                return Ok(0);
            }

            let mut written = 0usize;
            while written < max_frames && self.cursor < self.frames.len() {
                let dst = out.start_frame(Box3::None, None);
                let frame = &self.frames[self.cursor];
                for (dst_atom, &src_idx) in dst.iter_mut().zip(selection.iter()) {
                    *dst_atom = frame[src_idx as usize];
                }
                self.cursor += 1;
                written += 1;
            }
            Ok(written)
        }
    }

    let mut interner = StringInterner::new();
    let selected_name = interner.intern_upper("SX");
    let other_name = interner.intern_upper("OX");
    let res = interner.intern_upper("MOL");
    let atoms = AtomTable {
        name_id: vec![selected_name, other_name, selected_name],
        resname_id: vec![res, res, res],
        resid: vec![1, 2, 3],
        chain_id: vec![0, 0, 0],
        element_id: vec![0, 0, 0],
        mass: vec![1.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1000.0, 1000.0, 1000.0, 1.0],
        [1.0, 2.0, 3.0, 1.0],
    ]);
    let system = System::with_atoms(atoms, interner, positions0);
    let selection = Selection {
        expr: "selected".into(),
        indices: Arc::new(vec![0, 2]),
    };
    let selected_indices = selection.indices.as_ref().clone();

    let bbox_frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1000.0, 1000.0, 1000.0, 1.0],
        [1.0, 2.0, 3.0, 1.0],
    ]];
    let mut bbox_traj = SurfaceSelectedOnlyTraj::new(3, bbox_frames, selected_indices.clone());
    let mut bbox_plan = SurfPlan::new(selection.clone()).with_algorithm(SurfAlgorithm::Bbox);
    let mut bbox_exec = Executor::new(system.clone());
    let bbox_out = bbox_exec.run_plan(&mut bbox_plan, &mut bbox_traj).unwrap();
    match bbox_out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            assert!((vals[0] - 22.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
    assert!(bbox_traj.selected_reads > 0);

    let sasa_frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [0.5, 0.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 1.0],
    ]];
    let mut sasa_traj = SurfaceSelectedOnlyTraj::new(3, sasa_frames, selected_indices);
    let mut sasa_plan = SurfPlan::new(selection)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_probe_radius(0.0)
        .with_n_sphere_points(64)
        .with_radii(Some(vec![1.0, 1.0]))
        .with_atom_area(true)
        .with_residue_area(true);
    let mut sasa_exec = Executor::new(system);
    let sasa_out = sasa_exec.run_plan(&mut sasa_plan, &mut sasa_traj).unwrap();
    match sasa_out {
        PlanOutput::Surface(output) => {
            assert_eq!(output.frames, 1);
            assert_eq!(output.atoms, 2);
            assert_eq!(output.residues, 2);
            assert_eq!(output.residue_ids, vec![1, 3]);
            assert_eq!(output.atom_area.len(), 2);
            assert_eq!(output.residue_area.len(), 2);
            let expected_atom = 4.0f32 * std::f32::consts::PI;
            let expected_total = 2.0 * expected_atom;
            assert!((output.total[0] - expected_total).abs() < 0.2);
            for area in output.atom_area.iter().chain(output.residue_area.iter()) {
                assert!((*area - expected_atom).abs() < 0.2);
            }
        }
        _ => panic!("unexpected output"),
    }
    assert!(sasa_traj.selected_reads > 0);
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
fn surf_plan_sasa_atom_area_output_matches_totals() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_probe_radius(0.0)
        .with_n_sphere_points(64)
        .with_radii(Some(vec![1.0, 1.0]))
        .with_atom_area(true);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Surface(output) => {
            assert_eq!(output.frames, 2);
            assert_eq!(output.atoms, 2);
            assert_eq!(output.total.len(), 2);
            assert_eq!(output.atom_area.len(), 4);
            let expected_atom = 4.0f32 * std::f32::consts::PI;
            for area in output.atom_area.iter() {
                assert!((*area - expected_atom).abs() < 0.2);
            }
            for frame in 0..output.frames {
                let row_start = frame * output.atoms;
                let row_sum: f32 = output.atom_area[row_start..row_start + output.atoms]
                    .iter()
                    .sum();
                assert!((output.total[frame] - row_sum).abs() < 1e-6);
            }
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_sasa_residue_area_output_aggregates_selected_atoms() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_probe_radius(0.0)
        .with_n_sphere_points(64)
        .with_radii(Some(vec![1.0, 1.0]))
        .with_residue_area(true);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [4.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0]],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Surface(output) => {
            assert_eq!(output.frames, 2);
            assert_eq!(output.residues, 2);
            assert_eq!(output.residue_ids, vec![1, 2]);
            assert_eq!(output.residue_area.len(), 4);
            let expected_atom = 4.0f32 * std::f32::consts::PI;
            for area in output.residue_area.iter() {
                assert!((*area - expected_atom).abs() < 0.2);
            }
            for frame in 0..output.frames {
                let row_start = frame * output.residues;
                let row_sum: f32 = output.residue_area[row_start..row_start + output.residues]
                    .iter()
                    .sum();
                assert!((output.total[frame] - row_sum).abs() < 1e-6);
            }
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_sasa_volume_output_matches_single_sphere() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_probe_radius(0.0)
        .with_n_sphere_points(96)
        .with_radii(Some(vec![1.5]))
        .with_volume(true);
    let frames = vec![vec![[3.0, 4.0, 5.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Surface(output) => {
            assert_eq!(output.frames, 1);
            assert_eq!(output.atoms, 1);
            assert!(output.atom_area.is_empty());
            assert_eq!(output.volume.len(), 1);
            let expected = 4.0f32 * std::f32::consts::PI * 1.5 * 1.5 * 1.5 / 3.0;
            assert!((output.volume[0] - expected).abs() < 0.2);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_sasa_prefers_system_gb_radius() {
    let mut system = build_single_atom_system();
    system.set_gb_radii(vec![2.0]).unwrap();
    let sel = system.select("name CA").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_probe_radius(0.0)
        .with_n_sphere_points(64);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            let expected = 4.0f32 * std::f32::consts::PI * 2.0 * 2.0;
            assert!((vals[0] - expected).abs() < 0.8);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_sasa_can_use_system_parse_radius() {
    let mut system = build_single_atom_system();
    system.set_parse_radii(vec![1.0]).unwrap();
    let sel = system.select("name CA").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_probe_radius(0.0)
        .with_n_sphere_points(64)
        .with_radii_mode(SurfaceRadiiMode::Parse);
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
fn surf_plan_sasa_can_use_system_vdw_radius() {
    let mut system = build_single_atom_system();
    system.set_vdw_radii(vec![1.25]).unwrap();
    let sel = system.select("name CA").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_probe_radius(0.0)
        .with_n_sphere_points(64)
        .with_radii_mode(SurfaceRadiiMode::Vdw);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            let expected = 4.0f32 * std::f32::consts::PI * 1.25 * 1.25;
            assert!((vals[0] - expected).abs() < 0.4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_sasa_errors_when_vdw_radii_missing() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Sasa)
        .with_radii_mode(SurfaceRadiiMode::Vdw);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    match exec.run_plan(&mut plan, &mut traj) {
        Ok(_) => panic!("expected missing vdw radii error"),
        Err(err) => assert!(err.to_string().contains("vdw radii requested")),
    }
}

#[test]
fn surf_plan_lcpo_single_atom_matches_lcpo_carbon_term() {
    let mut system = build_single_atom_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Lcpo)
        .with_radius_offset(1.4)
        .with_neighbor_cutoff(2.5);
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            let radius = 1.70f32 + 1.4;
            let expected = 0.77887f32 * 4.0 * std::f32::consts::PI * radius * radius;
            assert!((vals[0] - expected).abs() < 1e-3);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_lcpo_constant_term_handles_all_small_selection() {
    let mut interner = StringInterner::new();
    let o_name = interner.intern_upper("O");
    let res = interner.intern_upper("HOH");
    let oxygen = interner.intern_upper("O");
    let atoms = AtomTable {
        name_id: vec![o_name],
        resname_id: vec![res],
        resid: vec![1],
        chain_id: vec![0],
        element_id: vec![oxygen],
        mass: vec![16.0],
    };
    let positions0 = Some(vec![[0.0, 0.0, 0.0, 1.0]]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let sel = system.select("resid 1").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Lcpo)
        .with_radius_offset(0.0)
        .with_neighbor_cutoff(2.5);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            let radius = 1.60f32;
            let expected = 0.68563f32 * 4.0 * std::f32::consts::PI * radius * radius;
            assert!((vals[0] - expected).abs() < 1e-3);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_lcpo_constant_term_is_added_once_with_heavy_atoms() {
    let mut interner = StringInterner::new();
    let c1_name = interner.intern_upper("C1");
    let c2_name = interner.intern_upper("C2");
    let o_name = interner.intern_upper("O");
    let res = interner.intern_upper("MIX");
    let carbon = interner.intern_upper("C");
    let oxygen = interner.intern_upper("O");
    let atoms = AtomTable {
        name_id: vec![c1_name, c2_name, o_name],
        resname_id: vec![res, res, res],
        resid: vec![1, 2, 3],
        chain_id: vec![0, 0, 0],
        element_id: vec![carbon, carbon, oxygen],
        mass: vec![12.0, 12.0, 16.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [20.0, 0.0, 0.0, 1.0],
        [40.0, 0.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let sel = system.select("resid 1:3").unwrap();
    let mut plan = SurfPlan::new(sel)
        .with_algorithm(SurfAlgorithm::Lcpo)
        .with_radius_offset(0.0)
        .with_neighbor_cutoff(1.6);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Series(vals) => {
            assert_eq!(vals.len(), 1);
            let c_radius = 1.70f32;
            let o_radius = 1.60f32;
            let c_term =
                0.77887f32 * 4.0 * std::f32::consts::PI * c_radius * c_radius;
            let o_term =
                0.68563f32 * 4.0 * std::f32::consts::PI * o_radius * o_radius;
            let expected = 2.0 * c_term + o_term;
            assert!((vals[0] - expected).abs() < 1e-3);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn surf_plan_lcpo_uses_solute_neighbors_in_rust_loop() {
    fn carbon_pair_system() -> System {
        let mut interner = StringInterner::new();
        let ca = interner.intern_upper("CA");
        let cb = interner.intern_upper("CB");
        let ala = interner.intern_upper("ALA");
        let carbon = interner.intern_upper("C");
        let atoms = AtomTable {
            name_id: vec![ca, cb],
            resname_id: vec![ala, ala],
            resid: vec![1, 2],
            chain_id: vec![0, 0],
            element_id: vec![carbon, carbon],
            mass: vec![12.0, 12.0],
        };
        let positions0 = Some(vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]);
        System::with_atoms(atoms, interner, positions0)
    }

    let mut system_all = carbon_pair_system();
    let sel_all = system_all.select("resid 1").unwrap();
    let mut plan_all = SurfPlan::new(sel_all)
        .with_algorithm(SurfAlgorithm::Lcpo)
        .with_radius_offset(1.4);
    let frames = vec![system_all.positions0.clone().unwrap()];
    let mut traj_all = InMemoryTraj::new(frames);
    let mut exec_all = Executor::new(system_all);
    let with_neighbor = match exec_all.run_plan(&mut plan_all, &mut traj_all).unwrap() {
        PlanOutput::Series(vals) => vals[0],
        _ => panic!("unexpected output"),
    };

    let mut system_one = carbon_pair_system();
    let sel_one = system_one.select("resid 1").unwrap();
    let solute_one = system_one.select("resid 1").unwrap();
    let mut plan_one = SurfPlan::new(sel_one)
        .with_algorithm(SurfAlgorithm::Lcpo)
        .with_radius_offset(1.4)
        .with_solute_selection(Some(solute_one));
    let mut traj_one = InMemoryTraj::new(vec![system_one.positions0.clone().unwrap()]);
    let mut exec_one = Executor::new(system_one);
    let isolated = match exec_one.run_plan(&mut plan_one, &mut traj_one).unwrap() {
        PlanOutput::Series(vals) => vals[0],
        _ => panic!("unexpected output"),
    };

    assert!(with_neighbor < isolated);
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
