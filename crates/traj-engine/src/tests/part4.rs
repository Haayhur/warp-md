#[test]
fn rotacf_plan_constant_orientation() {
    let mut system = build_plane_system();
    let sel = system.select("resid 1").unwrap();
    let orient = crate::plans::analysis::rotacf::OrientationSpec::PlaneIndices([1, 2, 3]);
    let mut plan = RotAcfPlan::new(sel, GroupBy::Resid, orient);
    let frames = vec![
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries { data, cols, .. } => {
            assert_eq!(cols, 4);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 1.0).abs() < 1e-6);
            assert!((data[2] - 1.0).abs() < 1e-6);
            assert!((data[3] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn rotacf_plan_ring_emits_lag_one() {
    let mut system = build_plane_system();
    let sel = system.select("resid 1").unwrap();
    let orient = crate::plans::analysis::rotacf::OrientationSpec::PlaneIndices([1, 2, 3]);
    let mut plan = RotAcfPlan::new(sel, GroupBy::Resid, orient)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(2);
    let frames = vec![
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            time,
            data,
            cols,
            rows,
            ..
        } => {
            assert_eq!(cols, 4);
            assert_eq!(rows, 3);
            assert_eq!(time.len(), 3);
            assert!((time[0] - 0.0).abs() < 1e-6);
            assert!((time[1] - 1.0).abs() < 1e-6);
            assert!((time[2] - 2.0).abs() < 1e-6);
            for &idx in &[0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] {
                assert!((data[idx] - 1.0).abs() < 1e-6);
            }
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn rotacf_plan_rejects_nonuniform_time_spacing() {
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

    let mut system = build_plane_system();
    let sel = system.select("resid 1").unwrap();
    let orient = crate::plans::analysis::rotacf::OrientationSpec::PlaneIndices([1, 2, 3]);
    let mut plan = RotAcfPlan::new(sel, GroupBy::Resid, orient)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(2);
    let frames = vec![
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
    ];
    let mut traj = TimedTraj::new(frames, vec![0.0, 1.0, 2.4]);
    let mut exec = Executor::new(system);
    match exec.run_plan(&mut plan, &mut traj) {
        Ok(_) => panic!("expected non-uniform time error"),
        Err(err) => assert!(err.to_string().contains("uniform frame spacing")),
    }
}

#[test]
fn rotacf_plan_rejects_zero_orientation_index() {
    let mut system = build_plane_system();
    let sel = system.select("resid 1").unwrap();
    let orient = crate::plans::analysis::rotacf::OrientationSpec::VectorIndices([0, 1]);
    let mut plan = RotAcfPlan::new(sel, GroupBy::Resid, orient);
    let frames = vec![
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    match exec.run_plan(&mut plan, &mut traj) {
        Ok(_) => panic!("expected invalid orientation index error"),
        Err(err) => assert!(err.to_string().contains("1-based")),
    }
}

#[test]
fn conductivity_plan_basic() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let charges = vec![1.0f64, 1.0f64];
    let mut plan =
        ConductivityPlan::new(sel, GroupBy::Resid, charges, 1.0).with_lag_mode(LagMode::MultiTau);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
    ];
    let box_ = Box3::Orthorhombic {
        lx: 10.0,
        ly: 10.0,
        lz: 10.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            cols, data, time, ..
        } => {
            assert_eq!(cols, 1);
            let qelec = 1.60217656535e-19_f64;
            let kb = 1.380648813e-23_f64;
            let vol_nm3 = 10.0 * 10.0 * 10.0;
            let vol = vol_nm3 * 1.0e-27;
            let multi = qelec * qelec / (6.0 * kb * 1.0 * vol);
            let expected = 4.0 * multi;
            let (idx, _) = time
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    let da = (a.1 - 1.0).abs();
                    let db = (b.1 - 1.0).abs();
                    da.partial_cmp(&db).unwrap()
                })
                .unwrap();
            let got = data[idx] as f64;
            let rel = (got - expected).abs() / expected.abs().max(1.0);
            assert!(rel < 1.0e-3, "got {got} expected {expected}");
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn conductivity_plan_rejects_nonuniform_time_spacing() {
    struct TimedTrajWithBox {
        n_atoms: usize,
        frames: Vec<Vec<[f32; 4]>>,
        times: Vec<f32>,
        box_: Box3,
        cursor: usize,
    }

    impl TimedTrajWithBox {
        fn new(frames: Vec<Vec<[f32; 4]>>, times: Vec<f32>, box_: Box3) -> Self {
            let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
            Self {
                n_atoms,
                frames,
                times,
                box_,
                cursor: 0,
            }
        }
    }

    impl TrajReader for TimedTrajWithBox {
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
                let coords = out.start_frame(self.box_, self.times.get(self.cursor).copied());
                coords.copy_from_slice(&self.frames[self.cursor]);
                self.cursor += 1;
                count += 1;
            }
            Ok(count)
        }
    }

    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let charges = vec![1.0f64, 1.0f64];
    let mut plan = ConductivityPlan::new(sel, GroupBy::Resid, charges, 1.0)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(2);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]],
    ];
    let box_ = Box3::Orthorhombic {
        lx: 10.0,
        ly: 10.0,
        lz: 10.0,
    };
    let mut traj = TimedTrajWithBox::new(frames, vec![0.0, 1.0, 2.3], box_);
    let mut exec = Executor::new(system);
    match exec.run_plan(&mut plan, &mut traj) {
        Ok(_) => panic!("expected non-uniform time error"),
        Err(err) => assert!(err.to_string().contains("uniform frame spacing")),
    }
}

#[test]
fn conductivity_plan_atom_selected_io_path() {
    struct SelectedOnlyTrajWithBox {
        n_atoms: usize,
        frames: Vec<Vec<[f32; 4]>>,
        box_: Box3,
        cursor: usize,
    }

    impl SelectedOnlyTrajWithBox {
        fn new(frames: Vec<Vec<[f32; 4]>>, box_: Box3) -> Self {
            let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
            Self {
                n_atoms,
                frames,
                box_,
                cursor: 0,
            }
        }
    }

    impl TrajReader for SelectedOnlyTrajWithBox {
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
                let dst = out.start_frame(self.box_, None);
                for (dst_atom, &src_idx) in dst.iter_mut().zip(selection.iter()) {
                    *dst_atom = src[src_idx as usize];
                }
                self.cursor += 1;
                count += 1;
            }
            Ok(count)
        }
    }

    let mut system = build_two_resid_system();
    let sel = system.select("resid 1").unwrap();
    let charges = vec![1.0f64, 0.0f64];
    let mut plan = ConductivityPlan::new(sel, GroupBy::Atom, charges, 1.0)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(1);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [5.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [6.0, 0.0, 0.0, 1.0]],
    ];
    let box_ = Box3::Orthorhombic {
        lx: 10.0,
        ly: 10.0,
        lz: 10.0,
    };
    let mut traj = SelectedOnlyTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system).with_chunk_frames(1);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            cols, data, time, ..
        } => {
            assert_eq!(cols, 1);
            assert_eq!(time.len(), 1);
            let qelec = 1.60217656535e-19_f64;
            let kb = 1.380648813e-23_f64;
            let vol_nm3 = 10.0 * 10.0 * 10.0;
            let vol = vol_nm3 * 1.0e-27;
            let multi = qelec * qelec / (6.0 * kb * 1.0 * vol);
            let expected = multi;
            let got = data[0] as f64;
            let rel = (got - expected).abs() / expected.abs().max(1.0);
            assert!(rel < 1.0e-3, "got {got} expected {expected}");
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn ion_pair_corr_constant_pairs() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let group_types = vec![0usize, 1usize];
    let mut plan =
        IonPairCorrelationPlan::new(sel, GroupBy::Resid, 2.0, 2.0).with_group_types(group_types);
    let frames = vec![
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            data, cols, rows, ..
        } => {
            assert_eq!(cols, 6);
            assert_eq!(rows, 2);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[3] - 1.0).abs() < 1e-6);
            assert!((data[6] - 1.0).abs() < 1e-6);
            assert!((data[9] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn ion_pair_corr_ring_emits_lag_one() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let group_types = vec![0usize, 1usize];
    let mut plan = IonPairCorrelationPlan::new(sel, GroupBy::Resid, 2.0, 2.0)
        .with_group_types(group_types)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(2);
    let frames = vec![
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
        system.positions0.clone().unwrap(),
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            time,
            data,
            cols,
            rows,
            ..
        } => {
            assert_eq!(cols, 6);
            assert_eq!(rows, 3);
            assert_eq!(time.len(), 3);
            assert!((time[0] - 0.0).abs() < 1e-6);
            assert!((time[1] - 1.0).abs() < 1e-6);
            assert!((time[2] - 2.0).abs() < 1e-6);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[6] - 1.0).abs() < 1e-6);
            assert!((data[12] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn structure_factor_shapes() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = StructureFactorPlan::new(sel, 5, 5.0, 4, 2.0, PbcMode::None);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::StructureFactor(output) => {
            assert_eq!(output.r.len(), 5);
            assert_eq!(output.g_r.len(), 5);
            assert_eq!(output.q.len(), 4);
            assert_eq!(output.s_q.len(), 4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn water_count_basic() {
    let mut system = build_two_resid_system();
    let water = system.select("resid 1").unwrap();
    let center = system.select("resid 1").unwrap();
    let mut plan = WaterCountPlan::new(water, center, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Grid(output) => {
            assert_eq!(output.dims, [3, 3, 3]);
            assert!((output.mean[0] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn hbond_distance_only() {
    let mut system = build_two_resid_system();
    let donors = system.select("resid 1").unwrap();
    let acceptors = system.select("resid 2").unwrap();
    let mut plan = HbondPlan::new(donors, acceptors, 2.0);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::TimeSeries {
            data, cols, rows, ..
        } => {
            assert_eq!(cols, 1);
            assert_eq!(rows, 1);
            assert!((data[0] - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn rdf_plan_counts() {
    let mut system = build_system();
    let sel = system.select("name CA").unwrap();
    let mut plan = RdfPlan::new(sel.clone(), sel, 5, 5.0, PbcMode::None);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Rdf(rdf) => {
            let total: u64 = rdf.counts.iter().sum();
            assert!(total > 0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn polymer_end_to_end_basic() {
    let mut system = build_polymer_system(2, 3);
    let sel = system.select("name C").unwrap();
    let mut plan = EndToEndPlan::new(sel);
    let frames = vec![linear_frame(2, 3, 1.0, 1.0)];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert_eq!(data.len(), 2);
            assert!((data[0] - 2.0).abs() < 1e-6);
            assert!((data[1] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn polymer_contour_length_basic() {
    let mut system = build_polymer_system(2, 3);
    let sel = system.select("name C").unwrap();
    let mut plan = ContourLengthPlan::new(sel);
    let frames = vec![linear_frame(2, 3, 1.0, 1.0)];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert_eq!(data.len(), 2);
            assert!((data[0] - 2.0).abs() < 1e-6);
            assert!((data[1] - 2.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn polymer_chain_rg_basic() {
    let mut system = build_polymer_system(2, 3);
    let sel = system.select("name C").unwrap();
    let mut plan = ChainRgPlan::new(sel);
    let frames = vec![linear_frame(2, 3, 1.0, 1.0)];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            let expected = (2.0f32 / 3.0f32).sqrt();
            assert_eq!(rows, 1);
            assert_eq!(cols, 2);
            assert!((data[0] - expected).abs() < 1e-6);
            assert!((data[1] - expected).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn polymer_bond_length_histogram() {
    let mut system = build_polymer_system(2, 3);
    let sel = system.select("name C").unwrap();
    let mut plan = BondLengthDistributionPlan::new(sel, 2, 2.0);
    let frames = vec![linear_frame(2, 3, 1.0, 1.0)];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Histogram { centers: _, counts } => {
            assert_eq!(counts.len(), 2);
            assert_eq!(counts[0], 0);
            assert_eq!(counts[1], 4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn polymer_bond_length_selected_io_path() {
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

    let mut system = build_polymer_system(2, 3);
    let sel = system.select("name C").unwrap();
    let mut plan = BondLengthDistributionPlan::new(sel, 2, 2.0);
    let frames = vec![linear_frame(2, 3, 1.0, 1.0)];
    let mut traj = SelectedOnlyTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Histogram { centers: _, counts } => {
            assert_eq!(counts.len(), 2);
            assert_eq!(counts[0], 0);
            assert_eq!(counts[1], 4);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn polymer_bond_angle_histogram() {
    let mut system = build_polymer_system(2, 3);
    let sel = system.select("name C").unwrap();
    let mut plan = BondAngleDistributionPlan::new(sel, 3, true);
    let frames = vec![right_angle_frame(2, 1.0)];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Histogram { centers: _, counts } => {
            assert_eq!(counts.len(), 3);
            assert_eq!(counts[1], 2);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn polymer_persistence_basic() {
    let mut system = build_polymer_system(1, 3);
    let sel = system.select("name C").unwrap();
    let mut plan = PersistenceLengthPlan::new(sel);
    let frames = vec![linear_frame(1, 3, 1.0, 0.0)];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Persistence(p) => {
            assert_eq!(p.bond_autocorrelation.len(), 2);
            assert!((p.bond_autocorrelation[0] - 1.0).abs() < 1e-6);
            assert!((p.lb - 1.0).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn docking_plan_detects_interaction_classes() {
    let mut interner = StringInterner::new();
    let rec_o = interner.intern_upper("O1");
    let rec_c = interner.intern_upper("C1");
    let lig_n = interner.intern_upper("N1");
    let lig_c = interner.intern_upper("C2");
    let res_rec = interner.intern_upper("REC");
    let res_lig = interner.intern_upper("LIG");
    let elem_o = interner.intern_upper("O");
    let elem_c = interner.intern_upper("C");
    let elem_n = interner.intern_upper("N");
    let atoms = AtomTable {
        name_id: vec![rec_o, rec_c, lig_n, lig_c],
        resname_id: vec![res_rec, res_rec, res_lig, res_lig],
        resid: vec![1, 1, 2, 2],
        chain_id: vec![0, 0, 0, 0],
        element_id: vec![elem_o, elem_c, elem_n, elem_c],
        mass: vec![16.0, 12.0, 14.0, 12.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [6.0, 0.0, 0.0, 1.0],
        [2.8, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let receptor = system.select("resid 1").unwrap();
    let ligand = system.select("resid 2").unwrap();
    let mut plan = DockingPlan::new(receptor, ligand, 4.0, 4.0, 3.5, 1.5)
        .unwrap()
        .with_max_events_per_frame(64);
    let frames = vec![
        vec![
            [0.0, 0.0, 0.0, 1.0],
            [6.0, 0.0, 0.0, 1.0],
            [2.8, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
        ],
        vec![
            [0.0, 0.0, 0.0, 1.0],
            [6.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [5.0, 0.0, 0.0, 1.0],
        ],
    ];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(cols, 6);
            assert_eq!(rows, 6);
            assert_eq!(data.len(), rows * cols);
            let mut codes = Vec::with_capacity(rows);
            for row in 0..rows {
                codes.push(data[row * cols + 3] as i32);
            }
            codes.sort_unstable();
            assert_eq!(codes, vec![1, 2, 3, 3, 4, 4]);
            assert_eq!(data[0] as usize, 0);
            assert_eq!(data[6 * 4] as usize, 1);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn docking_plan_rejects_empty_selection() {
    let mut system = build_four_resid_system();
    let receptor = system.select("resid 99").unwrap();
    let ligand = system.select("resid 3:4").unwrap();
    let err = DockingPlan::new(receptor, ligand, 4.0, 4.0, 3.5, 2.5)
        .err()
        .expect("expected constructor failure");
    assert!(err.to_string().contains("receptor selection"));
}

#[test]
fn docking_plan_hbond_angle_filter_blocks_bent_geometry() {
    let mut interner = StringInterner::new();
    let n_name = interner.intern_upper("N1");
    let h_name = interner.intern_upper("H1");
    let o_name = interner.intern_upper("O1");
    let res_rec = interner.intern_upper("REC");
    let res_lig = interner.intern_upper("LIG");
    let elem_n = interner.intern_upper("N");
    let elem_h = interner.intern_upper("H");
    let elem_o = interner.intern_upper("O");
    let atoms = AtomTable {
        name_id: vec![n_name, h_name, o_name],
        resname_id: vec![res_rec, res_rec, res_lig],
        resid: vec![1, 1, 2],
        chain_id: vec![0, 0, 0],
        element_id: vec![elem_n, elem_h, elem_o],
        mass: vec![14.0, 1.0, 16.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let receptor = system.select("name N1").unwrap();
    let ligand = system.select("name O1").unwrap();
    let mut plan = DockingPlan::new(receptor, ligand, 4.0, 4.0, 3.5, 0.5)
        .unwrap()
        .with_allow_missing_hydrogen(false)
        .with_hbond_min_angle_deg(140.0)
        .with_donor_hydrogen_cutoff(1.2);
    let frames = vec![vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 6);
            assert_eq!(data[3] as i32, 3);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn docking_plan_salt_bridge_precedence_over_hbond() {
    let mut interner = StringInterner::new();
    let nz = interner.intern_upper("NZ");
    let od1 = interner.intern_upper("OD1");
    let lys = interner.intern_upper("LYS");
    let asp = interner.intern_upper("ASP");
    let elem_n = interner.intern_upper("N");
    let elem_o = interner.intern_upper("O");
    let atoms = AtomTable {
        name_id: vec![nz, od1],
        resname_id: vec![lys, asp],
        resid: vec![1, 2],
        chain_id: vec![0, 0],
        element_id: vec![elem_n, elem_o],
        mass: vec![14.0, 16.0],
    };
    let positions0 = Some(vec![[0.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let receptor = system.select("name NZ").unwrap();
    let ligand = system.select("name OD1").unwrap();
    let mut plan = DockingPlan::new(receptor, ligand, 4.0, 4.0, 3.5, 0.5).unwrap();
    let frames = vec![vec![[0.0, 0.0, 0.0, 1.0], [3.0, 0.0, 0.0, 1.0]]];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Matrix { data, rows, cols } => {
            assert_eq!(rows, 1);
            assert_eq!(cols, 6);
            assert_eq!(data[3] as i32, 5);
        }
        _ => panic!("unexpected output"),
    }
}
