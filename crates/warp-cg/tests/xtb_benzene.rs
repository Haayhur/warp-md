use tempfile::tempdir;
use warp_cg::mapping::map_molecule;
use warp_cg::molecule::Molecule;
use warp_cg::parameters::calculate_bond_stats;
use warp_cg::trajectory::{map_trajectory, BeadMapping};
use warp_cg::xtb::run_xtb_pipeline;

#[test]
#[ignore = "requires external xTB/chemfiles behavior; run manually for validation"]
fn test_full_xtb_benzene() {
    let smiles = "c1ccccc1";
    let name = "benzene_test";
    let tmp_dir = tempdir().unwrap();
    let out_dir = tmp_dir.path();

    let xtb_res = run_xtb_pipeline(name, smiles, out_dir).expect("xTB pipeline failed");
    let mol = Molecule::from_smiles(smiles).unwrap();
    let mapping_res = map_molecule(&mol);

    assert_eq!(mapping_res.bead_names.len(), 3);

    let ref_path = if let Some(trj) = &xtb_res.trajectory_trj {
        trj.to_str().unwrap().to_string()
    } else {
        xtb_res.opt_xyz.to_str().unwrap().to_string()
    };

    let out_traj = out_dir.join("cg.xtc");
    let bead_mapping = BeadMapping {
        bead_names: mapping_res.bead_names.clone(),
        atom_indices: mapping_res.atom_groups.clone(),
    };
    map_trajectory(&ref_path, out_traj.to_str().unwrap(), &bead_mapping)
        .expect("Trajectory mapping failed");

    let stats = calculate_bond_stats(out_traj.to_str().unwrap(), &mapping_res.connections)
        .expect("Parameter calculation failed");

    for bond in &stats {
        println!(
            "Benzene CG bond {}-{}: {:.4} nm",
            bond.bead_i,
            bond.bead_j,
            bond.mean / 10.0
        );
    }

    // This validates the xTB -> CG mapping -> bonded-statistics path, not
    // Martini/OpenMM/Gromacs parity. The automated benzene bead centers can be
    // closer than production Martini equilibrium distances.
    assert!(!stats.is_empty());
    for bond in &stats {
        let bond_nm = bond.mean / 10.0;
        assert!(
            bond_nm.is_finite() && bond_nm > 0.05 && bond_nm < 0.35,
            "Bond length {} nm out of expected range",
            bond_nm
        );
        assert!(bond.samples > 0);
    }
}
