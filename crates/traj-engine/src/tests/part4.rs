use std::sync::Arc;

use crate::plans::analysis::secondary_structure::{add, cross, mul, normalize, sub};
use traj_core::selection::Selection;

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
fn dielectric_plan_constant_dipole_gives_unity() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let charges = vec![1.0f64, -1.0f64];
    let mut plan = DielectricPlan::new(sel, GroupBy::Resid, charges);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
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
        PlanOutput::Dielectric(output) => {
            assert!((output.dielectric_total - 1.0).abs() < 1.0e-6);
            assert!((output.dielectric_rot - 1.0).abs() < 1.0e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn dielectric_plan_fluctuating_dipole_matches_neumann_formula() {
    let mut system = build_system();
    let sel = system.select("resid 1").unwrap();
    let charges = vec![1.0f64, -1.0f64];
    let temperature = 300.0f64;
    let mut plan =
        DielectricPlan::new(sel, GroupBy::Resid, charges).with_temperature(temperature);
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
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
        PlanOutput::Dielectric(output) => {
            let fluct_ea2 = 0.25f64;
            let elementary_charge = 1.602_176_634e-19_f64;
            let angstrom_to_m = 1.0e-10_f64;
            let angstrom3_to_m3 = 1.0e-30_f64;
            let eps0 = 8.854_187_812_8e-12_f64;
            let kb = 1.380_649e-23_f64;
            let volume_ang3 = 10.0f64 * 10.0 * 10.0;
            let expected = 1.0
                + fluct_ea2 * (elementary_charge * angstrom_to_m).powi(2)
                    / (3.0 * eps0 * kb * temperature * volume_ang3 * angstrom3_to_m3);
            let got = output.dielectric_total as f64;
            let rel = (got - expected).abs() / expected.abs().max(1.0);
            assert!(rel < 1.0e-6, "got {got} expected {expected}");
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn current_plan_matches_conductivity_and_dielectric_components() {
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]],
    ];
    let box_ = Box3::Orthorhombic {
        lx: 10.0,
        ly: 10.0,
        lz: 10.0,
    };
    let charges = vec![1.0f64, -1.0f64];

    let mut system_cond = build_two_resid_system();
    let sel_cond = system_cond.select("resid 1:2").unwrap();
    let mut cond_plan = ConductivityPlan::new(sel_cond, GroupBy::Resid, charges.clone(), 300.0)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(1);
    let mut cond_traj = InMemoryTrajWithBox::new(frames.clone(), box_);
    let mut cond_exec = Executor::new(system_cond);
    let cond_out = cond_exec.run_plan(&mut cond_plan, &mut cond_traj).unwrap();

    let mut system_dielectric = build_two_resid_system();
    let sel_dielectric = system_dielectric.select("resid 1:2").unwrap();
    let mut dielectric_plan =
        DielectricPlan::new(sel_dielectric, GroupBy::Resid, charges.clone()).with_temperature(300.0);
    let mut dielectric_traj = InMemoryTrajWithBox::new(frames.clone(), box_);
    let mut dielectric_exec = Executor::new(system_dielectric);
    let dielectric_out = dielectric_exec
        .run_plan(&mut dielectric_plan, &mut dielectric_traj)
        .unwrap();

    let mut system_current = build_two_resid_system();
    let sel_current = system_current.select("resid 1:2").unwrap();
    let mut current_plan = CurrentPlan::new(sel_current, GroupBy::Resid, charges, 300.0)
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(1);
    let mut current_traj = InMemoryTrajWithBox::new(frames, box_);
    let mut current_exec = Executor::new(system_current);
    let current_out = current_exec.run_plan(&mut current_plan, &mut current_traj).unwrap();

    match (cond_out, dielectric_out, current_out) {
        (
            PlanOutput::TimeSeries {
                time: cond_time,
                data: cond_data,
                rows: cond_rows,
                cols: cond_cols,
            },
            PlanOutput::Dielectric(dielectric),
            PlanOutput::Current(current),
        ) => {
            assert_eq!(current.conductivity_time, cond_time);
            assert_eq!(current.conductivity, cond_data);
            assert_eq!(current.conductivity_rows, cond_rows);
            assert_eq!(current.conductivity_cols, cond_cols);
            assert_eq!(current.frame_time, dielectric.time);
            assert_eq!(current.md_sq, dielectric.rot_sq);
            assert_eq!(current.mj_sq, dielectric.trans_sq);
            assert_eq!(current.md_mj, dielectric.rot_trans);
            assert!((current.dielectric_rot - dielectric.dielectric_rot).abs() < 1.0e-6);
            assert!((current.dielectric_total - dielectric.dielectric_total).abs() < 1.0e-6);
            assert!((current.mu_avg - dielectric.mu_avg).abs() < 1.0e-6);
            assert_eq!(current.conductivity_static, cond_data.last().copied());
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
fn salt_bridge_plan_filters_pairs_and_tracks_contacts() {
    let mut interner = StringInterner::new();
    let nz = interner.intern_upper("NZ");
    let oe1 = interner.intern_upper("OE1");
    let nh1 = interner.intern_upper("NH1");
    let od1 = interner.intern_upper("OD1");
    let lys = interner.intern_upper("LYS");
    let glu = interner.intern_upper("GLU");
    let arg = interner.intern_upper("ARG");
    let asp = interner.intern_upper("ASP");
    let atoms = AtomTable {
        name_id: vec![nz, oe1, nh1, od1],
        resname_id: vec![lys, glu, arg, asp],
        resid: vec![1, 2, 3, 4],
        chain_id: vec![0; 4],
        element_id: vec![0; 4],
        mass: vec![1.0; 4],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 1.0],
        [4.0, 0.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let sel = system.select("resid 1:4").unwrap();
    let charges = vec![1.0, -1.0, 1.0, -1.0];
    let mut plan = SaltBridgePlan::new(sel, GroupBy::Atom, charges)
        .with_truncate(Some(1.5))
        .with_contact_cutoff(Some(1.1));
    let frames = vec![
        vec![
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
            [4.0, 0.0, 0.0, 1.0],
        ],
        vec![
            [0.0, 0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0, 1.0],
            [0.8, 0.0, 0.0, 1.0],
            [3.8, 0.0, 0.0, 1.0],
        ],
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
            assert_eq!(rows, 2);
            assert_eq!(cols, 4);
            assert_eq!(time, vec![0.0, 1.0]);
            assert!((data[0] - 1.0).abs() < 1e-6);
            assert!((data[1] - 3.0).abs() < 1e-6);
            assert!((data[2] - 3.0).abs() < 1e-6);
            assert!((data[3] - 1.0).abs() < 1e-6);
            assert!((data[5] - 0.8).abs() < 1e-6);
            assert!((data[6] - 0.8).abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
    assert_eq!(
        plan.pair_labels(),
        &[
            "LYS1-1:GLU2-2".to_string(),
            "LYS1-1:ARG3-3".to_string(),
            "GLU2-2:ASP4-4".to_string(),
            "ARG3-3:ASP4-4".to_string(),
        ]
    );
    assert_eq!(plan.pair_classes(), &[2, 0, 1, 2]);
    assert_eq!(plan.contact_count(), &[1, 1, 1, 1]);
    assert!((plan.min_distance()[0] - 1.0).abs() < 1e-6);
    assert!((plan.min_distance()[1] - 0.8).abs() < 1e-6);
    assert!((plan.min_distance()[2] - 0.8).abs() < 1e-6);
    assert!((plan.min_distance()[3] - 1.0).abs() < 1e-6);
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
fn densmap_plan_box_count_and_axes() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan =
        DensityMapPlan::new(sel, 2, 1.0, DensityMapUnit::Count).with_bins(Some(2), Some(2));
    let frames = vec![system.positions0.clone().unwrap()];
    let box_ = Box3::Orthorhombic {
        lx: 2.0,
        ly: 2.0,
        lz: 2.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::DensityMap(output) => {
            assert_eq!(output.rows, 2);
            assert_eq!(output.cols, 2);
            assert_eq!(output.plane_axes, [0, 1]);
            assert_eq!(output.average_axis, 2);
            assert_eq!(output.unit, "count");
            assert_eq!(output.n_frames, 1);
            assert_eq!(output.axis1, vec![0.5, 1.5]);
            assert_eq!(output.axis2, vec![0.5, 1.5]);
            assert_eq!(output.matrix, vec![1.0, 0.0, 1.0, 0.0]);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn densmap_plan_number_density_uses_box_volume_and_filter() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = DensityMapPlan::new(sel, 0, 1.0, DensityMapUnit::NumberDensity)
        .with_bins(Some(2), Some(2))
        .with_average_window(Some(0.5), Some(1.5));
    let frames = vec![system.positions0.clone().unwrap()];
    let box_ = Box3::Orthorhombic {
        lx: 2.0,
        ly: 2.0,
        lz: 2.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::DensityMap(output) => {
            assert_eq!(output.plane_axes, [1, 2]);
            assert_eq!(output.average_axis, 0);
            assert_eq!(output.unit, "nm-3");
            assert_eq!(output.n_frames, 1);
            assert!((output.matrix[0] - 0.5).abs() < 1e-6);
            assert!(output.matrix[1].abs() < 1e-6);
            assert!(output.matrix[2].abs() < 1e-6);
            assert!(output.matrix[3].abs() < 1e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn potential_plan_charge_density_and_integrals() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = PotentialPlan::new(sel, 2, 1.0, vec![1.0, -1.0]).with_n_slices(Some(3));
    let frames = vec![vec![[0.0, 0.0, 0.25, 1.0], [0.0, 0.0, 2.25, 1.0]]];
    let box_ = Box3::Orthorhombic {
        lx: 1.0,
        ly: 1.0,
        lz: 3.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Potential(output) => {
            let eps0 = 8.854_187_812_8e-12_f64;
            let elementary_charge = 1.602_176_634e-19_f64;
            let field_factor = elementary_charge * 1.0e9 / eps0;
            let potential_factor = -elementary_charge * 1.0e9 / eps0;
            assert_eq!(output.axis, 2);
            assert_eq!(output.coordinate, vec![0.5, 1.5, 2.5]);
            assert!((output.slice_width - 1.0).abs() < 1.0e-6);
            assert!((output.charge_density[0] - 1.0).abs() < 1.0e-6);
            assert!(output.charge_density[1].abs() < 1.0e-6);
            assert!((output.charge_density[2] + 1.0).abs() < 1.0e-6);
            assert!(output.field[0].abs() < 1.0e-3);
            assert!((output.field[1] as f64 - 0.5 * field_factor).abs() < 5.0e-3);
            assert!(output.field[2].abs() < 1.0e-3);
            assert!(output.potential[0].abs() < 1.0e-3);
            assert!((output.potential[1] as f64 - 0.25 * potential_factor).abs() < 5.0e-3);
            assert!((output.potential[2] as f64 - 0.5 * potential_factor).abs() < 5.0e-3);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn potential_plan_center_selection_recenters_profile() {
    let mut system = build_two_resid_system();
    let sel = system.select("resid 1").unwrap();
    let center = system.select("resid 1").unwrap();
    let mut plan = PotentialPlan::new(sel, 0, 1.0, vec![1.0, 0.0])
        .with_n_slices(Some(4))
        .with_center_selection(center);
    let frames = vec![vec![[3.0, 0.0, 0.0, 1.0], [0.5, 0.0, 0.0, 1.0]]];
    let box_ = Box3::Orthorhombic {
        lx: 4.0,
        ly: 1.0,
        lz: 1.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Potential(output) => {
            let peak = output
                .charge_density
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            assert!(output.centered);
            assert_eq!(output.axis, 0);
            assert!(output.coordinate[peak].abs() <= 0.6);
            assert_eq!(output.n_frames, 1);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn h2order_plan_reports_slice_order_and_dipole() {
    let system = build_plane_system();
    let charges = vec![-0.834, 0.417, 0.417];
    let mut plan = H2OrderPlan::new(vec![0], vec![1], vec![2], charges, 2, 1.0)
        .with_n_slices(Some(2))
        .with_length_scale(1.0);
    let frames = vec![vec![
        [0.0, 0.0, 0.25, 1.0],
        [0.0, 0.0, 0.75, 1.0],
        [0.0, 0.0, 0.75, 1.0],
    ]];
    let box_ = Box3::Orthorhombic {
        lx: 2.0,
        ly: 2.0,
        lz: 2.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::H2Order(output) => {
            let expected_dipole = 0.417 * 48.026898;
            assert_eq!(output.axis, 2);
            assert_eq!(output.rows, 2);
            assert_eq!(output.cols, 3);
            assert_eq!(output.coordinate, vec![0.5, 1.5]);
            assert_eq!(output.counts, vec![1, 0]);
            assert!((output.order[0] - 1.0).abs() < 1.0e-6);
            assert!(output.order[1].abs() < 1.0e-6);
            assert!(output.dipole[0].abs() < 1.0e-5);
            assert!(output.dipole[1].abs() < 1.0e-5);
            assert!((output.dipole[2] as f64 - expected_dipole).abs() < 1.0e-3);
            assert_eq!(output.dipole_unit, "debye");
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn h2order_plan_wraps_hydrogens_with_pbc() {
    let system = build_plane_system();
    let charges = vec![-0.834, 0.417, 0.417];
    let mut plan = H2OrderPlan::new(vec![0], vec![1], vec![2], charges, 2, 1.0)
        .with_n_slices(Some(2))
        .with_length_scale(1.0);
    let frames = vec![vec![
        [0.0, 0.0, 1.75, 1.0],
        [0.0, 0.0, 0.25, 1.0],
        [0.0, 0.0, 0.25, 1.0],
    ]];
    let box_ = Box3::Orthorhombic {
        lx: 2.0,
        ly: 2.0,
        lz: 2.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::H2Order(output) => {
            assert_eq!(output.counts, vec![0, 1]);
            assert!((output.order[1] - 1.0).abs() < 1.0e-6);
            assert!(output.dipole[5] > 0.0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn hydorder_plan_reports_tetrahedral_grid_values() {
    let mut interner = StringInterner::new();
    let ow = interner.intern_upper("OW");
    let sol = interner.intern_upper("SOL");
    let oxygen = interner.intern_upper("O");
    let atoms = AtomTable {
        name_id: vec![ow; 5],
        resname_id: vec![sol; 5],
        resid: vec![1, 2, 3, 4, 5],
        chain_id: vec![0; 5],
        element_id: vec![oxygen; 5],
        mass: vec![16.0; 5],
    };
    let scale = 0.4f32;
    let positions0 = Some(vec![
        [2.0, 2.0, 2.0, 1.0],
        [2.0 + scale, 2.0 + scale, 2.0 + scale, 1.0],
        [2.0 + scale, 2.0 - scale, 2.0 - scale, 1.0],
        [2.0 - scale, 2.0 + scale, 2.0 - scale, 1.0],
        [2.0 - scale, 2.0 - scale, 2.0 + scale, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let sel = system.select("resid 1:5").unwrap();
    let mut plan = HydOrderPlan::new(sel, 2, 0.25).with_length_scale(1.0);
    let frames = vec![system.positions0.clone().unwrap()];
    let box_ = Box3::Orthorhombic {
        lx: 4.0,
        ly: 4.0,
        lz: 4.0,
    };
    let mut traj = InMemoryTrajWithBox::new(frames, box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::HydOrder(output) => {
            let center_idx = (8 * 16 + 8) * 16 + 8;
            assert_eq!(output.dims, [16, 16, 16]);
            assert_eq!(output.n_frames, 1);
            assert_eq!(output.axis, 2);
            assert_eq!(output.plane_axes, [0, 1]);
            assert_eq!(output.counts.iter().sum::<u64>(), 5);
            assert_eq!(output.counts[center_idx], 1);
            assert!(output.sg_grid[center_idx].abs() < 1.0e-6);
            assert!(output.sk_grid[center_idx].abs() < 1.0e-6);
            assert_eq!(output.interface_blocks, 0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn helixorient_plan_reports_local_axis_metrics_and_rotation() {
    let mut interner = StringInterner::new();
    let ca = interner.intern_upper("CA");
    let ala = interner.intern_upper("ALA");
    let carbon = interner.intern_upper("C");
    let atoms = AtomTable {
        name_id: vec![ca; 5],
        resname_id: vec![ala; 5],
        resid: vec![1, 2, 3, 4, 5],
        chain_id: vec![0; 5],
        element_id: vec![carbon; 5],
        mass: vec![12.0; 5],
    };
    let radius = 2.3f32;
    let rise = 1.5f32;
    let twist_deg = 100.0f32;
    let frame0: Vec<[f32; 4]> = (0..5)
        .map(|i| {
            let angle = (i as f32 * twist_deg).to_radians();
            [
                radius * angle.cos(),
                radius * angle.sin(),
                i as f32 * rise,
                1.0,
            ]
        })
        .collect();
    let rotation_deg = 20.0f32;
    let frame1: Vec<[f32; 4]> = frame0
        .iter()
        .map(|coord| {
            let angle = rotation_deg.to_radians();
            [
                coord[0] * angle.cos() - coord[1] * angle.sin(),
                coord[0] * angle.sin() + coord[1] * angle.cos(),
                coord[2],
                1.0,
            ]
        })
        .collect();
    let mut system = System::with_atoms(atoms, interner, Some(frame0.clone()));
    let sel = system.select("resid 1:5").unwrap();
    let mut plan = HelixOrientPlan::new(sel).with_length_scale(1.0);
    let frames = vec![frame0, frame1];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::HelixOrient(output) => {
            let idx0 = 2usize;
            let idx1 = 5 + 2;
            let axis_base = idx0 * 3;
            let center_base = idx0 * 3;
            assert_eq!(output.frames, 2);
            assert_eq!(output.residues, 5);
            assert_eq!(output.labels.len(), 5);
            assert_eq!(output.time, vec![0.0, 1.0]);
            assert!((output.axis[axis_base + 2] - 1.0).abs() < 5.0e-3);
            assert!(output.center[center_base].abs() < 5.0e-2);
            assert!(output.center[center_base + 1].abs() < 5.0e-2);
            assert!((output.rise[idx0] - rise).abs() < 5.0e-2);
            assert!((output.radius[idx0] - radius).abs() < 5.0e-2);
            assert!((output.twist[idx0] - twist_deg).abs() < 1.0);
            assert!(output.tilt[idx1].abs() < 5.0e-2);
            assert!((output.rotation[idx1].abs() - rotation_deg).abs() < 1.0);
            assert!((output.theta3[idx1].abs() - rotation_deg).abs() < 1.0);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn bundle_plan_reports_axis_metrics_and_kink_geometry() {
    let mut interner = StringInterner::new();
    let ca = interner.intern_upper("CA");
    let ala = interner.intern_upper("ALA");
    let carbon = interner.intern_upper("C");
    let atoms = AtomTable {
        name_id: vec![ca; 6],
        resname_id: vec![ala; 6],
        resid: vec![1, 2, 3, 4, 5, 6],
        chain_id: vec![0; 6],
        element_id: vec![carbon; 6],
        mass: vec![12.0; 6],
    };
    let frame = vec![
        [0.0, 0.0, 2.0, 1.0],
        [2.0, 0.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [2.0, 0.0, 1.0, 1.0],
    ];
    let mut system = System::with_atoms(atoms, interner, Some(frame.clone()));
    let top = system.select("resid 1:2").unwrap();
    let bottom = system.select("resid 3:4").unwrap();
    let kink = system.select("resid 5:6").unwrap();
    let mut plan = BundlePlan::new(top, bottom, 2)
        .with_kink_selection(kink)
        .with_length_scale(1.0);
    let box_ = Box3::Orthorhombic {
        lx: 10.0,
        ly: 10.0,
        lz: 10.0,
    };
    let mut traj = InMemoryTrajWithBox::new(vec![frame], box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    let approx = |lhs: f32, rhs: f32| (lhs - rhs).abs() < 1.0e-5;
    match out {
        PlanOutput::Bundle(output) => {
            assert_eq!(output.labels, vec!["axis_1".to_string(), "axis_2".to_string()]);
            assert_eq!(output.frames, 1);
            assert_eq!(output.axes, 2);
            assert_eq!(output.time, vec![0.0]);
            assert!(output.has_kink);
            assert!(output.mass_weighted);
            assert!(output.used_box);
            assert!(approx(output.length_scale, 1.0));
            assert_eq!(output.reference_axis.len(), 3);
            assert!(approx(output.reference_axis[0], 0.0));
            assert!(approx(output.reference_axis[1], 0.0));
            assert!(approx(output.reference_axis[2], 1.0));
            assert_eq!(output.top.len(), 6);
            assert!(approx(output.top[0], -1.0));
            assert!(approx(output.top[2], 1.0));
            assert!(approx(output.top[3], 1.0));
            assert!(approx(output.top[5], 1.0));
            assert_eq!(output.bottom.len(), 6);
            assert!(approx(output.bottom[0], -1.0));
            assert!(approx(output.bottom[2], -1.0));
            assert!(approx(output.bottom[3], 1.0));
            assert!(approx(output.bottom[5], -1.0));
            assert_eq!(output.direction.len(), 6);
            assert!(approx(output.direction[2], 1.0));
            assert!(approx(output.direction[5], 1.0));
            assert_eq!(output.length.len(), 2);
            assert!(approx(output.length[0], 2.0));
            assert!(approx(output.length[1], 2.0));
            assert!(approx(output.distance[0], 1.0));
            assert!(approx(output.distance[1], 1.0));
            assert!(approx(output.z_shift[0], 0.0));
            assert!(approx(output.z_shift[1], 0.0));
            assert!(approx(output.tilt[0], 0.0));
            assert!(approx(output.tilt[1], 0.0));
            assert!(approx(output.radial_tilt[0], 0.0));
            assert!(approx(output.lateral_tilt[1], 0.0));
            assert_eq!(output.kink.len(), 6);
            assert!(approx(output.kink[0], -1.0));
            assert!(approx(output.kink[3], 1.0));
            assert!(approx(output.kink_angle[0], 0.0));
            assert!(approx(output.kink_angle[1], 0.0));
            assert!(approx(output.kink_radial[0], 0.0));
            assert!(approx(output.kink_lateral[1], 0.0));
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn mdmat_plan_reports_mean_matrix_frames_and_contact_summary() {
    let mut interner = StringInterner::new();
    let a1 = interner.intern_upper("A1");
    let a2 = interner.intern_upper("A2");
    let ala = interner.intern_upper("ALA");
    let carbon = interner.intern_upper("C");
    let atoms = AtomTable {
        name_id: vec![a1, a2, a1, a2],
        resname_id: vec![ala; 4],
        resid: vec![1, 1, 2, 2],
        chain_id: vec![0; 4],
        element_id: vec![carbon; 4],
        mass: vec![12.0; 4],
    };
    let frame0 = vec![
        [0.0, 0.0, 0.0, 1.0],
        [0.2, 0.0, 0.0, 1.0],
        [0.55, 0.0, 0.0, 1.0],
        [1.4, 0.0, 0.0, 1.0],
    ];
    let frame1 = vec![
        [0.0, 0.0, 0.0, 1.0],
        [0.2, 0.0, 0.0, 1.0],
        [0.8, 0.0, 0.0, 1.0],
        [0.9, 0.0, 0.0, 1.0],
    ];
    let mut system = System::with_atoms(atoms, interner, Some(frame0.clone()));
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = MdmatPlan::new(sel)
        .with_truncate(0.5)
        .with_include_contacts(true)
        .with_include_frames(true)
        .with_length_scale(1.0);
    let mut traj = InMemoryTraj::new(vec![frame0, frame1]);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Mdmat(output) => {
            assert_eq!(output.labels, vec!["ALA:1".to_string(), "ALA:2".to_string()]);
            assert_eq!(output.frames, 2);
            assert_eq!(output.residues, 2);
            assert_eq!(output.time, vec![0.0, 1.0]);
            assert_eq!(output.mean_matrix.len(), 4);
            assert!((output.mean_matrix[1] - 0.475).abs() < 1.0e-6);
            assert!((output.mean_matrix[2] - 0.475).abs() < 1.0e-6);
            assert_eq!(output.frame_matrices.len(), 8);
            assert!((output.frame_matrices[1] - 0.35).abs() < 1.0e-6);
            assert!((output.frame_matrices[5] - 0.6).abs() < 1.0e-6);
            assert!(output.used_box == false);
            assert_eq!(output.distinct_contact_atoms, vec![1, 1]);
            assert_eq!(output.residue_atom_counts, vec![2, 2]);
            assert!((output.mean_contact_atoms[0] - 0.5).abs() < 1.0e-6);
            assert!((output.mean_contact_atoms[1] - 0.5).abs() < 1.0e-6);
            assert!((output.contact_ratio[0] - 2.0).abs() < 1.0e-6);
            assert!((output.contact_ratio[1] - 2.0).abs() < 1.0e-6);
            assert!((output.mean_contact_atoms_per_residue_atom[0] - 0.25).abs() < 1.0e-6);
            assert!((output.mean_contact_atoms_per_residue_atom[1] - 0.25).abs() < 1.0e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn mdmat_plan_uses_only_selected_atoms_inside_each_residue() {
    let mut interner = StringInterner::new();
    let a1 = interner.intern_upper("A1");
    let a2 = interner.intern_upper("A2");
    let ala = interner.intern_upper("ALA");
    let carbon = interner.intern_upper("C");
    let atoms = AtomTable {
        name_id: vec![a1, a2, a1, a2],
        resname_id: vec![ala; 4],
        resid: vec![1, 1, 2, 2],
        chain_id: vec![0; 4],
        element_id: vec![carbon; 4],
        mass: vec![12.0; 4],
    };
    let frame = vec![
        [0.0, 0.0, 0.0, 1.0],
        [5.0, 0.0, 0.0, 1.0],
        [6.0, 0.0, 0.0, 1.0],
        [0.1, 0.0, 0.0, 1.0],
    ];
    let system = System::with_atoms(atoms, interner, Some(frame.clone()));
    let selection = Selection {
        expr: "manual".into(),
        indices: Arc::new(vec![1, 2]),
    };
    let mut plan = MdmatPlan::new(selection).with_length_scale(1.0);
    let mut traj = InMemoryTraj::new(vec![frame]);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Mdmat(output) => {
            assert_eq!(output.labels, vec!["ALA:1".to_string(), "ALA:2".to_string()]);
            assert_eq!(output.residues, 2);
            assert!((output.mean_matrix[1] - 1.0).abs() < 1.0e-6);
            assert!((output.mean_matrix[2] - 1.0).abs() < 1.0e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn mdmat_plan_wraps_orthorhombic_distances() {
    let mut interner = StringInterner::new();
    let ca = interner.intern_upper("CA");
    let ala = interner.intern_upper("ALA");
    let carbon = interner.intern_upper("C");
    let atoms = AtomTable {
        name_id: vec![ca, ca],
        resname_id: vec![ala, ala],
        resid: vec![1, 2],
        chain_id: vec![0, 0],
        element_id: vec![carbon, carbon],
        mass: vec![12.0, 12.0],
    };
    let frame = vec![[0.1, 0.0, 0.0, 1.0], [1.9, 0.0, 0.0, 1.0]];
    let mut system = System::with_atoms(atoms, interner, Some(frame.clone()));
    let sel = system.select("resid 1:2").unwrap();
    let mut plan = MdmatPlan::new(sel).with_length_scale(1.0);
    let box_ = Box3::Orthorhombic {
        lx: 2.0,
        ly: 2.0,
        lz: 2.0,
    };
    let mut traj = InMemoryTrajWithBox::new(vec![frame], box_);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Mdmat(output) => {
            assert!(output.used_box);
            assert!((output.mean_matrix[1] - 0.2).abs() < 1.0e-6);
            assert!((output.mean_matrix[2] - 0.2).abs() < 1.0e-6);
        }
        _ => panic!("unexpected output"),
    }
}

fn place_internal(
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    length: f64,
    angle_deg: f64,
    dihedral_deg: f64,
) -> [f64; 3] {
    let bc = normalize(sub(c, b));
    let ba = normalize(sub(a, b));
    let normal = normalize(cross(ba, bc));
    let binormal = cross(normal, bc);
    let theta = angle_deg.to_radians();
    let tau = dihedral_deg.to_radians();
    add(
        c,
        add(
            mul(bc, -length * theta.cos()),
            mul(
                add(mul(binormal, tau.cos()), mul(normal, tau.sin())),
                length * theta.sin(),
            ),
        ),
    )
}

fn build_alpha_helix_backbone_system(n_res: usize) -> System {
    let mut interner = StringInterner::new();
    let n_name = interner.intern_upper("N");
    let h_name = interner.intern_upper("H");
    let ca_name = interner.intern_upper("CA");
    let c_name = interner.intern_upper("C");
    let o_name = interner.intern_upper("O");
    let ala = interner.intern_upper("ALA");
    let n_el = interner.intern_upper("N");
    let h_el = interner.intern_upper("H");
    let c_el = interner.intern_upper("C");
    let o_el = interner.intern_upper("O");

    let phi = -57.0;
    let psi = -47.0;
    let omega = 180.0;
    let oxygen_dihedral = -60.0;
    let hydrogen_dihedral = 0.0;

    let mut name_id = Vec::with_capacity(n_res * 5);
    let mut resname_id = Vec::with_capacity(n_res * 5);
    let mut resid = Vec::with_capacity(n_res * 5);
    let mut chain_id = Vec::with_capacity(n_res * 5);
    let mut element_id = Vec::with_capacity(n_res * 5);
    let mut mass = Vec::with_capacity(n_res * 5);
    let mut coords = Vec::with_capacity(n_res * 5);

    let n0 = [0.0, 0.0, 0.0];
    let ca0 = [1.458, 0.0, 0.0];
    let angle = 110.4f64.to_radians();
    let c0 = [
        ca0[0] + 1.525 * (std::f64::consts::PI - angle).cos(),
        ca0[1] + 1.525 * (std::f64::consts::PI - angle).sin(),
        0.0,
    ];
    let o0 = place_internal(n0, ca0, c0, 1.231, 120.8, oxygen_dihedral);

    let mut n_positions = vec![n0];
    let mut ca_positions = vec![ca0];
    let mut c_positions = vec![c0];
    let mut o_positions = vec![o0];
    let mut h_positions = vec![[-1.0, 0.0, 0.0]];

    for _ in 1..n_res {
        let prev = n_positions.len() - 1;
        let n_pos = place_internal(
            n_positions[prev],
            ca_positions[prev],
            c_positions[prev],
            1.329,
            116.2,
            psi + 180.0,
        );
        let h_pos = place_internal(
            c_positions[prev],
            n_pos,
            ca_positions[prev],
            1.0,
            120.0,
            hydrogen_dihedral,
        );
        let ca_pos = place_internal(
            ca_positions[prev],
            c_positions[prev],
            n_pos,
            1.458,
            121.7,
            omega + 180.0,
        );
        let c_pos = place_internal(
            c_positions[prev],
            n_pos,
            ca_pos,
            1.525,
            110.4,
            phi + 180.0,
        );
        let o_pos = place_internal(n_pos, ca_pos, c_pos, 1.231, 120.8, oxygen_dihedral);
        n_positions.push(n_pos);
        h_positions.push(h_pos);
        ca_positions.push(ca_pos);
        c_positions.push(c_pos);
        o_positions.push(o_pos);
    }

    for i in 0..n_res {
        for (name, element, atom_mass, point) in [
            (n_name, n_el, 14.0f32, n_positions[i]),
            (h_name, h_el, 1.0f32, h_positions[i]),
            (ca_name, c_el, 12.0f32, ca_positions[i]),
            (c_name, c_el, 12.0f32, c_positions[i]),
            (o_name, o_el, 16.0f32, o_positions[i]),
        ] {
            name_id.push(name);
            resname_id.push(ala);
            resid.push(i as i32 + 1);
            chain_id.push(0);
            element_id.push(element);
            mass.push(atom_mass);
            coords.push([point[0] as f32, point[1] as f32, point[2] as f32, 1.0]);
        }
    }

    let atoms = AtomTable {
        name_id,
        resname_id,
        resid,
        chain_id,
        element_id,
        mass,
    };
    System::with_atoms(atoms, interner, Some(coords))
}

#[test]
fn helix_plan_reports_fitted_alpha_helix_metrics() {
    let mut system = build_alpha_helix_backbone_system(7);
    let sel = system.select("resid 1:7").unwrap();
    let mut plan = HelixPlan::new(sel).with_length_scale(0.1);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Helix(output) => {
            assert_eq!(output.frames, 1);
            assert_eq!(output.residues, 7);
            assert_eq!(output.fragment_start, vec![2]);
            assert_eq!(output.fragment_end, vec![6]);
            assert!((output.radius[0] - 0.23).abs() < 1.0e-2);
            assert!((output.rise[0] - 0.15).abs() < 1.0e-2);
            assert!((output.twist[0] - 100.0).abs() < 2.0);
            assert!(!output.fragment_mask[0]);
            assert!(output.fragment_mask[1..6].iter().all(|&value| value));
            assert!(!output.fragment_mask[6]);
            assert!(output.helicity_fraction[0].abs() < 1.0e-6);
            assert!(output.helicity_fraction[1..6]
                .iter()
                .all(|&value| (value - 1.0).abs() < 1.0e-6));
            assert!(output.helicity_fraction[6].abs() < 1.0e-6);
            assert_eq!(output.labels.first().map(String::as_str), Some("ALA:1"));
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn sorient_plan_profiles_orientation_shells() {
    let mut interner = StringInterner::new();
    let ca = interner.intern_upper("CA");
    let ow = interner.intern_upper("OW");
    let hw1 = interner.intern_upper("HW1");
    let hw2 = interner.intern_upper("HW2");
    let ala = interner.intern_upper("ALA");
    let sol = interner.intern_upper("SOL");
    let carbon = interner.intern_upper("C");
    let oxygen = interner.intern_upper("O");
    let hydrogen = interner.intern_upper("H");
    let atoms = AtomTable {
        name_id: vec![ca, ow, hw1, hw2],
        resname_id: vec![ala, sol, sol, sol],
        resid: vec![1, 2, 2, 2],
        chain_id: vec![0, 0, 0, 0],
        element_id: vec![carbon, oxygen, hydrogen, hydrogen],
        mass: vec![12.0, 16.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let solute = system.select("resid 1").unwrap();
    let mut plan = SOrientPlan::new(solute, vec![1], vec![2], vec![3], 0.0, 1.5, 0.5, 2.0)
        .with_r_profile_max(Some(2.0));
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::SOrient(output) => {
            let expected = std::f64::consts::FRAC_1_SQRT_2 as f32;
            assert_eq!(output.n_frames, 1);
            assert_eq!(output.n_reference_positions, 1);
            assert_eq!(output.window_count, 1);
            assert!((output.average_shell_size - 1.0).abs() < 1.0e-6);
            assert_eq!(output.counts, vec![1]);
            assert!((output.mean_cos_theta1[0] - expected).abs() < 1.0e-6);
            assert!((output.mean_p2_theta2[0] + 1.0).abs() < 1.0e-6);
            assert!((output.cumulative_cos_theta1[0] - expected).abs() < 1.0e-6);
            assert!((output.cumulative_p2_theta2[0] + 1.0).abs() < 1.0e-6);
            assert!((output.count_density[0] - 0.5).abs() < 1.0e-6);
            assert_eq!(output.cos_theta1_distribution.len(), 4);
            assert_eq!(output.abs_cos_theta2_distribution.len(), 2);
            assert_eq!(output.use_vector23, false);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn sorient_plan_supports_com_and_vector23() {
    let mut interner = StringInterner::new();
    let ca = interner.intern_upper("CA");
    let ow = interner.intern_upper("OW");
    let hw1 = interner.intern_upper("HW1");
    let hw2 = interner.intern_upper("HW2");
    let ala = interner.intern_upper("ALA");
    let sol = interner.intern_upper("SOL");
    let carbon = interner.intern_upper("C");
    let oxygen = interner.intern_upper("O");
    let hydrogen = interner.intern_upper("H");
    let atoms = AtomTable {
        name_id: vec![ca, ca, ow, hw1, hw2],
        resname_id: vec![ala, ala, sol, sol, sol],
        resid: vec![1, 1, 2, 2, 2],
        chain_id: vec![0, 0, 0, 0, 0],
        element_id: vec![carbon, carbon, oxygen, hydrogen, hydrogen],
        mass: vec![12.0, 12.0, 16.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 1.0],
        [2.0, 1.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let solute = system.select("resid 1").unwrap();
    let mut plan = SOrientPlan::new(solute, vec![2], vec![3], vec![4], 0.0, 1.5, 0.5, 2.0)
        .with_r_profile_max(Some(2.0))
        .with_use_com(true)
        .with_use_vector23(true);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::SOrient(output) => {
            assert_eq!(output.n_reference_positions, 1);
            assert!(output.use_com);
            assert!(output.use_vector23);
            assert_eq!(output.window_count, 1);
            assert!((output.mean_p2_theta2[0] - 0.5).abs() < 1.0e-6);
            assert!((output.window_mean_p2_theta2 - 0.5).abs() < 1.0e-6);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn spol_plan_reports_cumulative_distribution_and_dipole_stats() {
    let mut interner = StringInterner::new();
    let ca = interner.intern_upper("CA");
    let ow = interner.intern_upper("OW");
    let hw1 = interner.intern_upper("HW1");
    let hw2 = interner.intern_upper("HW2");
    let ala = interner.intern_upper("ALA");
    let sol = interner.intern_upper("SOL");
    let carbon = interner.intern_upper("C");
    let oxygen = interner.intern_upper("O");
    let hydrogen = interner.intern_upper("H");
    let atoms = AtomTable {
        name_id: vec![ca, ow, hw1, hw2],
        resname_id: vec![ala, sol, sol, sol],
        resid: vec![1, 2, 2, 2],
        chain_id: vec![0, 0, 0, 0],
        element_id: vec![carbon, oxygen, hydrogen, hydrogen],
        mass: vec![12.0, 16.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let solute = system.select("resid 1").unwrap();
    let charges = vec![0.0, -0.834, 0.417, 0.417];
    let mut plan = SpolPlan::new(solute, vec![1], vec![2], vec![3], charges, 0.0, 1.5, 0.5)
        .with_r_hist_max(Some(2.0));
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Spol(output) => {
            let debye = 0.417f32 * 48.026_898f32;
            let expected_mag = debye * (2.0f32).sqrt();
            assert_eq!(output.window_count, 1);
            assert_eq!(output.shell_count, vec![0, 0, 1, 0]);
            assert_eq!(output.r, vec![0.5, 1.0, 1.5, 2.0]);
            assert_eq!(output.cumulative_count, vec![0.0, 0.0, 1.0, 1.0]);
            assert!((output.average_shell_size - 1.0).abs() < 1.0e-6);
            assert!((output.average_dipole - expected_mag).abs() < 1.0e-3);
            assert!(output.dipole_std.abs() < 1.0e-6);
            assert!((output.average_radial_dipole - debye).abs() < 1.0e-3);
            assert!((output.average_radial_polarization - debye).abs() < 1.0e-3);
            assert_eq!(output.dipole_unit, "debye");
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn spol_plan_supports_com_refdip_and_reference_atom() {
    let mut interner = StringInterner::new();
    let ca = interner.intern_upper("CA");
    let ow = interner.intern_upper("OW");
    let hw1 = interner.intern_upper("HW1");
    let hw2 = interner.intern_upper("HW2");
    let ala = interner.intern_upper("ALA");
    let sol = interner.intern_upper("SOL");
    let carbon = interner.intern_upper("C");
    let oxygen = interner.intern_upper("O");
    let hydrogen = interner.intern_upper("H");
    let hx = interner.intern_upper("HX");
    let atoms = AtomTable {
        name_id: vec![ca, ca, ow, hw1, hw2, hx],
        resname_id: vec![ala, ala, sol, sol, sol, sol],
        resid: vec![1, 1, 2, 2, 2, 2],
        chain_id: vec![0, 0, 0, 0, 0, 0],
        element_id: vec![carbon, carbon, oxygen, hydrogen, hydrogen, hydrogen],
        mass: vec![12.0, 12.0, 16.0, 1.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [3.0, 0.0, 0.0, 1.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, 0.0, 1.0, 1.0],
    ]);
    let mut system = System::with_atoms(atoms, interner, positions0);
    let solute = system.select("resid 1").unwrap();
    let charges = vec![0.0, 0.0, -0.834, 0.417, 0.417, 0.1];
    let mut plan =
        SpolPlan::new(solute, vec![], vec![], vec![], charges, 1.5, 2.5, 0.5)
            .with_r_hist_max(Some(3.0))
            .with_use_com(true)
            .with_reference_atom(3)
            .with_refdip(1.0)
            .with_molecules(vec![2, 3, 4, 5], vec![0, 4], [0, 1, 2]);
    let frames = vec![system.positions0.clone().unwrap()];
    let mut traj = InMemoryTraj::new(frames);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::Spol(output) => {
            let debye = 48.026_898f64;
            let dipole = [0.467 * debye, 0.392 * debye, 0.075 * debye];
            let dipole_mag =
                (dipole[0] * dipole[0] + dipole[1] * dipole[1] + dipole[2] * dipole[2]).sqrt();
            let radial_unit = [
                2.0 / 5.0f64.sqrt(),
                0.0,
                1.0 / 5.0f64.sqrt(),
            ];
            let direction = [std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2, 0.0];
            let expected_radial_dipole =
                radial_unit[0] * dipole[0] + radial_unit[1] * dipole[1] + radial_unit[2] * dipole[2];
            let expected = expected_radial_dipole
                - (radial_unit[0] * direction[0] + radial_unit[1] * direction[1] + radial_unit[2] * direction[2]);
            assert!(output.use_com);
            assert_eq!(output.reference_atom, 3);
            assert_eq!(output.window_count, 1);
            assert_eq!(output.shell_count, vec![0, 0, 0, 0, 1, 0]);
            assert!((output.average_shell_size - 1.0).abs() < 1.0e-6);
            assert!((output.average_dipole as f64 - dipole_mag).abs() < 1.0e-3);
            assert!((output.average_radial_dipole as f64 - expected_radial_dipole).abs() < 1.0e-3);
            assert!((output.average_radial_polarization as f64 - expected).abs() < 1.0e-3);
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
            assert_eq!(total, 2);
            assert_eq!(rdf.counts[0], 0);
            assert_eq!(rdf.counts[1], 2);
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
