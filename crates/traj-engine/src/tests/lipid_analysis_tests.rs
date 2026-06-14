use crate::{
    Executor, LipidAreaPlan, LipidCurvedLeafletPlan, LipidFlipFlopPlan, LipidLargestClusterPlan,
    LipidLeafletPlan, LipidMembraneThicknessPlan, LipidMsdPlan, LipidNeighbourMatrixPlan,
    LipidNeighbourPlan, LipidRegistrationPlan, LipidSccPlan, LipidZAnglePlan, LipidZPositionPlan,
    LipidZThicknessPlan, PlanOutput,
};

fn lipid_system(n_residues: usize, atoms_per_residue: usize) -> System {
    let mut interner = StringInterner::new();
    let name_l = interner.intern_upper("L");
    let name_c = interner.intern_upper("C");
    let res_lip = interner.intern_upper("LIP");
    let mut name_id = Vec::new();
    let mut resname_id = Vec::new();
    let mut resid = Vec::new();
    let mut chain_id = Vec::new();
    let mut element_id = Vec::new();
    let mut mass = Vec::new();
    let mut positions0 = Vec::new();
    for res in 0..n_residues {
        for atom in 0..atoms_per_residue {
            name_id.push(if atom == 0 { name_l } else { name_c });
            resname_id.push(res_lip);
            resid.push((res + 1) as i32);
            chain_id.push(0);
            element_id.push(0);
            mass.push(1.0);
            positions0.push([0.0, 0.0, 0.0, 1.0]);
        }
    }
    System::with_atoms(
        AtomTable {
            name_id,
            resname_id,
            resid,
            chain_id,
            element_id,
            mass,
        },
        interner,
        Some(positions0),
    )
}

fn two_leaflet_frame() -> Vec<[f32; 4]> {
    vec![
        [25.0, 25.0, 60.0, 1.0],
        [25.0, 25.0, 50.0, 1.0],
        [75.0, 25.0, 60.0, 1.0],
        [75.0, 25.0, 50.0, 1.0],
        [25.0, 75.0, 40.0, 1.0],
        [25.0, 75.0, 50.0, 1.0],
        [75.0, 75.0, 40.0, 1.0],
        [75.0, 75.0, 50.0, 1.0],
    ]
}

#[test]
fn lipid_leaflets_and_z_positions_follow_midpoint() {
    let mut system = lipid_system(4, 2);
    let sel = system.select("name L C").unwrap();
    let mut leaflets = LipidLeafletPlan::new(sel.clone());
    let mut traj = InMemoryTrajWithBox::new(
        vec![two_leaflet_frame()],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut leaflets, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            assert_eq!(out.rows, 4);
            assert_eq!(out.cols, 1);
            assert_eq!(out.values, vec![1.0, 1.0, -1.0, -1.0]);
        }
        _ => panic!("unexpected output"),
    }

    let mem = system.select("name L C").unwrap();
    let height = system.select("name L").unwrap();
    let mut zpos = LipidZPositionPlan::new(mem, height);
    let mut traj = InMemoryTrajWithBox::new(
        vec![two_leaflet_frame()],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut zpos, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            assert_eq!(out.values, vec![10.0, 10.0, -10.0, -10.0]);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_curved_leaflets_use_contact_components_and_midplane_cutoff() {
    let mut system = lipid_system(5, 1);
    let sel = system.select("name L").unwrap();
    let mid = system.select("resid 5").unwrap();
    let frame = vec![
        [80.0, 50.0, 50.0, 1.0],
        [85.0, 50.0, 50.0, 1.0],
        [45.0, 50.0, 50.0, 1.0],
        [40.0, 50.0, 50.0, 1.0],
        [62.0, 50.0, 50.0, 1.0],
    ];
    let mut plan = LipidCurvedLeafletPlan::new(sel, 8.0).with_midplane(mid, 8.0);
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            assert_eq!(out.values, vec![1.0, 1.0, -1.0, -1.0, 0.0]);
            assert_eq!(out.kind, "curved_leaflets");
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_z_thickness_and_angle_are_per_residue() {
    let mut system = lipid_system(2, 2);
    let sel = system.select("name L C").unwrap();
    let mut thickness = LipidZThicknessPlan::new(sel);
    let frame = vec![
        [10.0, 10.0, 10.0, 1.0],
        [10.0, 10.0, 15.0, 1.0],
        [20.0, 20.0, 95.0, 1.0],
        [20.0, 20.0, 5.0, 1.0],
    ];
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame.clone()],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut thickness, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            assert_eq!(out.values, vec![5.0, 10.0]);
        }
        _ => panic!("unexpected output"),
    }

    let atom_a = system.select("name L").unwrap();
    let atom_b = system.select("name C").unwrap();
    let mut angles = LipidZAnglePlan::new(atom_a, atom_b);
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut angles, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            assert!((out.values[0] - 180.0).abs() < 1.0e-5);
            assert!((out.values[1] - 0.0).abs() < 1.0e-5);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_flip_flop_matches_reference_state_machine() {
    let leaflets = vec![
        -1, -1, -1, -1, -1, -1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, -1, -1,
        -1, -1, -1, -1,
    ];
    let mut plan = LipidFlipFlopPlan::new(leaflets, 1, 25, vec![0], 1);
    let system = lipid_system(1, 1);
    let mut traj = InMemoryTrajWithBox::new(
        vec![vec![[0.0, 0.0, 0.0, 1.0]]],
        Box3::Orthorhombic {
            lx: 10.0,
            ly: 10.0,
            lz: 10.0,
        },
    );
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::LipidFlipFlop(out) => {
            assert_eq!(out.events, vec![0, 5, 8, 1, 0, 8, 11, 1, 0, 17, 19, -1]);
            assert_eq!(out.success, vec!["Success", "Fail", "Success"]);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_area_periodic_square_lattice() {
    let mut system = lipid_system(4, 1);
    let sel = system.select("name L").unwrap();
    let leaflets = vec![1, 1, 1, 1];
    let mut plan = LipidAreaPlan::new(sel, leaflets, 4, 1);
    let frame = vec![
        [25.0, 25.0, 60.0, 1.0],
        [75.0, 25.0, 60.0, 1.0],
        [25.0, 75.0, 60.0, 1.0],
        [75.0, 75.0, 60.0, 1.0],
    ];
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            for area in out.values {
                assert!((area - 2500.0).abs() < 1.0e-3);
            }
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_neighbours_and_largest_cluster_use_cutoff_contacts() {
    let mut system = lipid_system(4, 1);
    let sel = system.select("name L").unwrap();
    let frame = vec![
        [10.0, 10.0, 50.0, 1.0],
        [15.0, 10.0, 50.0, 1.0],
        [80.0, 80.0, 50.0, 1.0],
        [86.0, 80.0, 50.0, 1.0],
    ];
    let mut neighbours = LipidNeighbourPlan::new(sel.clone(), 7.0);
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame.clone()],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut neighbours, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => assert_eq!(out.values, vec![1.0, 1.0, 1.0, 1.0]),
        _ => panic!("unexpected output"),
    }

    let mut cluster = LipidLargestClusterPlan::new(sel, 7.0);
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut cluster, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => assert_eq!(out.values, vec![2.0]),
        _ => panic!("unexpected output"),
    }

    let sel = system.select("name L").unwrap();
    let mut matrix = LipidNeighbourMatrixPlan::new(sel, 7.0);
    let mut traj = InMemoryTrajWithBox::new(
        vec![vec![
            [10.0, 10.0, 50.0, 1.0],
            [15.0, 10.0, 50.0, 1.0],
            [80.0, 80.0, 50.0, 1.0],
            [86.0, 80.0, 50.0, 1.0],
        ]],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut matrix, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            assert_eq!(out.rows, 16);
            assert_eq!(out.cols, 1);
            assert_eq!(
                out.values,
                vec![
                    0.0, 1.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, 0.0, //
                    0.0, 0.0, 0.0, 1.0, //
                    0.0, 0.0, 1.0, 0.0,
                ]
            );
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_membrane_thickness_averages_leaflet_surfaces() {
    let mut system = lipid_system(4, 1);
    let sel = system.select("name L").unwrap();
    let leaflets = vec![1, 1, -1, -1];
    let mut plan = LipidMembraneThicknessPlan::new(sel, leaflets, 4, 1);
    let frame = vec![
        [25.0, 25.0, 60.0, 1.0],
        [75.0, 25.0, 62.0, 1.0],
        [25.0, 75.0, 40.0, 1.0],
        [75.0, 75.0, 38.0, 1.0],
    ];
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => assert!((out.values[0] - 22.0).abs() < 1.0e-6),
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_registration_correlates_leaflet_density_grids() {
    let mut system = lipid_system(4, 1);
    let sel = system.select("name L").unwrap();
    let leaflets = vec![1, 1, -1, -1];
    let mut plan = LipidRegistrationPlan::new(sel.clone(), sel, leaflets, 4, 1).with_bins(2);
    let frame = vec![
        [25.0, 25.0, 60.0, 1.0],
        [75.0, 75.0, 60.0, 1.0],
        [25.0, 25.0, 40.0, 1.0],
        [75.0, 75.0, 40.0, 1.0],
    ];
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => assert!((out.values[0] - 1.0).abs() < 1.0e-6),
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_msd_reports_per_lipid_lag_displacements() {
    let mut system = lipid_system(2, 1);
    let sel = system.select("name L").unwrap();
    let frames = vec![
        vec![[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]],
        vec![[1.0, 0.0, 0.0, 1.0], [10.0, 2.0, 0.0, 1.0]],
        vec![[2.0, 0.0, 0.0, 1.0], [10.0, 4.0, 0.0, 1.0]],
    ];
    let mut plan = LipidMsdPlan::new(sel);
    let mut traj = InMemoryTrajWithBox::new(
        frames,
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            assert_eq!(out.rows, 2);
            assert_eq!(out.cols, 3);
            assert_eq!(out.values, vec![0.0, 1.0, 4.0, 0.0, 4.0, 16.0]);
        }
        _ => panic!("unexpected output"),
    }
}

#[test]
fn lipid_scc_averages_consecutive_tail_vectors() {
    let mut system = lipid_system(2, 3);
    let sel = system.select("name L C").unwrap();
    let frame = vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
        [10.0, 0.0, 0.0, 1.0],
        [10.0, 0.0, 1.0, 1.0],
        [10.0, 0.0, 2.0, 1.0],
    ];
    let normals = vec![0.0, 0.0, 1.0, 0.0, 0.0, -1.0];
    let mut plan = LipidSccPlan::new(sel).with_normals(normals, 2, 1);
    let mut traj = InMemoryTrajWithBox::new(
        vec![frame],
        Box3::Orthorhombic {
            lx: 100.0,
            ly: 100.0,
            lz: 100.0,
        },
    );
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut plan, &mut traj).unwrap();
    match out {
        PlanOutput::LipidMatrix(out) => {
            assert!((out.values[0] + 0.5).abs() < 1.0e-6);
            assert!((out.values[1] - 1.0).abs() < 1.0e-6);
        }
        _ => panic!("unexpected output"),
    }
}
