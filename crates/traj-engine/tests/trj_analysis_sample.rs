use std::path::Path;

use traj_core::error::TrajResult;
use traj_engine::plans::analysis::msd::{DtDecimation, FrameDecimation};
#[cfg(feature = "cuda")]
use traj_engine::{Device, GroupSpec, IonPairCorrelationPlan, LagMode};
use traj_engine::{Executor, GroupBy, MsdPlan, PbcMode, PlanOutput, StructureFactorPlan};
use traj_io::gro::GroReader;
use traj_io::xtc::XtcReader;
use traj_io::TopologyReader;

#[cfg(all(feature = "cuda", target_os = "linux"))]
use libloading::Library;
#[cfg(feature = "cuda")]
use traj_gpu::GpuContext;

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn cuda_driver_has_symbol(name: &[u8]) -> bool {
    unsafe {
        let lib = Library::new("libcuda.so")
            .or_else(|_| Library::new("libcuda.so.1"))
            .ok();
        let Some(lib) = lib else {
            return false;
        };
        lib.get::<*const ()>(name).is_ok()
    }
}

#[cfg(not(all(feature = "cuda", target_os = "linux")))]
#[allow(dead_code)]
fn cuda_driver_has_symbol(_name: &[u8]) -> bool {
    false
}

#[allow(dead_code)]
fn assert_close(a: &[f32], b: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(a.len(), b.len());
    for (idx, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (va - vb).abs();
        let tol = abs_tol + rel_tol * vb.abs();
        assert!(
            diff <= tol,
            "idx {idx} diff {diff} > tol {tol} (a={va}, b={vb})"
        );
    }
}

#[cfg(feature = "cuda")]
fn build_lag_index(time: &[f32], max_lag: usize) -> Vec<Option<usize>> {
    let dt = time
        .iter()
        .skip(1)
        .copied()
        .find(|v| *v > 0.0)
        .unwrap_or(1.0);
    let mut map = vec![None; max_lag + 1];
    for (idx, &t) in time.iter().enumerate() {
        let lag = if t <= 0.0 {
            0
        } else {
            (t / dt).round() as usize
        };
        if lag <= max_lag {
            map[lag] = Some(idx);
        }
    }
    map
}

#[test]
#[ignore]
fn sample_msd_structure_factor() -> TrajResult<()> {
    let gro_path = Path::new("/tmp/trj_analysis_repo/data/bmimC1Hnopol_bf4nopol.gro");
    let xtc_path = Path::new("/tmp/trj_analysis_repo/data/bmimC1Hnopol_bf4nopol_pbc_20.xtc");
    if !gro_path.exists() || !xtc_path.exists() {
        return Ok(());
    }

    let mut gro = GroReader::new(gro_path);
    let mut system = gro.read_system()?;
    let sel = system.select("resname BMIM or resname BF4")?;

    let mut traj = XtcReader::open(xtc_path)?;
    let mut msd = MsdPlan::new(sel.clone(), GroupBy::Resid)
        .with_frame_decimation(FrameDecimation {
            start: 100,
            stride: 100,
        })
        .with_dt_decimation(DtDecimation {
            cut1: 1000,
            stride1: 10,
            cut2: 10000,
            stride2: 100,
        });
    let mut exec = Executor::new(system.clone());
    let out = exec.run_plan(&mut msd, &mut traj)?;
    match out {
        PlanOutput::TimeSeries {
            rows, cols, data, ..
        } => {
            assert!(rows > 0);
            assert_eq!(cols, 8);
            assert!(data.iter().all(|v| v.is_finite()));
        }
        _ => panic!("unexpected output"),
    }

    let mut traj = XtcReader::open(xtc_path)?;
    let mut sf = StructureFactorPlan::new(sel, 200, 6.0, 100, 20.0, PbcMode::Orthorhombic);
    let mut exec = Executor::new(system);
    let out = exec.run_plan(&mut sf, &mut traj)?;
    match out {
        PlanOutput::StructureFactor(output) => {
            assert_eq!(output.r.len(), 200);
            assert_eq!(output.g_r.len(), 200);
            assert_eq!(output.q.len(), 100);
            assert_eq!(output.s_q.len(), 100);
            assert!(output
                .g_r
                .iter()
                .chain(output.s_q.iter())
                .all(|v| v.is_finite()));
        }
        _ => panic!("unexpected output"),
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn sample_msd_structure_factor_cuda_compare() -> TrajResult<()> {
    let gro_path = Path::new("/tmp/trj_analysis_repo/data/bmimC1Hnopol_bf4nopol.gro");
    let xtc_path = Path::new("/tmp/trj_analysis_repo/data/bmimC1Hnopol_bf4nopol_pbc_20.xtc");
    if !gro_path.exists() || !xtc_path.exists() {
        return Ok(());
    }
    if !cuda_driver_has_symbol(b"cuCtxRecordEvent") {
        return Ok(());
    }
    let _gpu = match std::panic::catch_unwind(|| GpuContext::new(0)) {
        Ok(Ok(ctx)) => ctx,
        _ => return Ok(()),
    };

    let mut gro = GroReader::new(gro_path);
    let mut system = gro.read_system()?;
    let sel = system.select("resname BMIM or resname BF4")?;

    let mut traj = XtcReader::open(xtc_path)?;
    let mut msd_cpu = MsdPlan::new(sel.clone(), GroupBy::Resid)
        .with_frame_decimation(FrameDecimation {
            start: 100,
            stride: 100,
        })
        .with_dt_decimation(DtDecimation {
            cut1: 1000,
            stride1: 10,
            cut2: 10000,
            stride2: 100,
        });
    let mut exec_cpu = Executor::new(system.clone());
    let out_cpu = exec_cpu.run_plan(&mut msd_cpu, &mut traj)?;

    let mut traj = XtcReader::open(xtc_path)?;
    let mut msd_gpu = MsdPlan::new(sel.clone(), GroupBy::Resid)
        .with_frame_decimation(FrameDecimation {
            start: 100,
            stride: 100,
        })
        .with_dt_decimation(DtDecimation {
            cut1: 1000,
            stride1: 10,
            cut2: 10000,
            stride2: 100,
        });
    if !cuda_driver_has_symbol(b"cuCtxRecordEvent") {
        return Ok(());
    }
    let gpu = match std::panic::catch_unwind(|| GpuContext::new(0)) {
        Ok(Ok(ctx)) => ctx,
        _ => return Ok(()),
    };
    let mut exec_gpu = Executor::new(system.clone()).with_device(Device::Cuda(gpu));
    let out_gpu = exec_gpu.run_plan(&mut msd_gpu, &mut traj)?;

    match (out_cpu, out_gpu) {
        (
            PlanOutput::TimeSeries {
                time: t1,
                data: d1,
                rows: r1,
                cols: c1,
            },
            PlanOutput::TimeSeries {
                time: t2,
                data: d2,
                rows: r2,
                cols: c2,
            },
        ) => {
            assert_eq!(r1, r2);
            assert_eq!(c1, c2);
            assert_close(&t1, &t2, 1.0e-5, 1.0e-5);
            assert_close(&d1, &d2, 1.0e-3, 1.0e-3);
        }
        _ => panic!("unexpected output"),
    }

    let mut traj = XtcReader::open(xtc_path)?;
    let mut sf_cpu =
        StructureFactorPlan::new(sel.clone(), 200, 6.0, 100, 20.0, PbcMode::Orthorhombic);
    let mut exec_cpu = Executor::new(system.clone());
    let out_cpu = exec_cpu.run_plan(&mut sf_cpu, &mut traj)?;

    let mut traj = XtcReader::open(xtc_path)?;
    let mut sf_gpu = StructureFactorPlan::new(sel, 200, 6.0, 100, 20.0, PbcMode::Orthorhombic);
    let gpu = GpuContext::new(0)?;
    let mut exec_gpu = Executor::new(system).with_device(Device::Cuda(gpu));
    let out_gpu = exec_gpu.run_plan(&mut sf_gpu, &mut traj)?;

    match (out_cpu, out_gpu) {
        (PlanOutput::StructureFactor(cpu), PlanOutput::StructureFactor(gpu)) => {
            assert_close(&cpu.r, &gpu.r, 1.0e-5, 1.0e-5);
            assert_close(&cpu.q, &gpu.q, 1.0e-5, 1.0e-5);
            assert_close(&cpu.g_r, &gpu.g_r, 1.0e-3, 1.0e-3);
            assert_close(&cpu.s_q, &gpu.s_q, 1.0e-3, 1.0e-3);
        }
        _ => panic!("unexpected output"),
    }
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn sample_ion_pair_cuda_compare() -> TrajResult<()> {
    let gro_path = Path::new("/tmp/trj_analysis_repo/data/bmimC1Hnopol_bf4nopol.gro");
    let xtc_path = Path::new("/tmp/trj_analysis_repo/data/bmimC1Hnopol_bf4nopol_pbc_20.xtc");
    if !gro_path.exists() || !xtc_path.exists() {
        return Ok(());
    }
    if !cuda_driver_has_symbol(b"cuCtxRecordEvent") {
        return Ok(());
    }
    let _gpu = match std::panic::catch_unwind(|| GpuContext::new(0)) {
        Ok(Ok(ctx)) => ctx,
        _ => return Ok(()),
    };

    let mut gro = GroReader::new(gro_path);
    let mut system = gro.read_system()?;
    let sel = system.select("resname BMIM or resname BF4")?;

    let groups = GroupSpec::new(sel.clone(), GroupBy::Resid).build(&system)?;
    let mut group_types = Vec::with_capacity(groups.n_groups());
    for atoms in groups.groups.iter() {
        let atom_idx = atoms[0];
        let res_id = system.atoms.resname_id[atom_idx];
        let resname = system.interner.resolve(res_id).unwrap_or("");
        let t = if resname.eq_ignore_ascii_case("BMIM") {
            0usize
        } else if resname.eq_ignore_ascii_case("BF4") {
            1usize
        } else {
            0usize
        };
        group_types.push(t);
    }

    let max_lag = 10usize;

    let mut traj = XtcReader::open(xtc_path)?;
    let mut plan_cpu = IonPairCorrelationPlan::new(sel.clone(), GroupBy::Resid, 2.0, 2.0)
        .with_group_types(group_types.clone())
        .with_lag_mode(LagMode::Ring)
        .with_max_lag(max_lag);
    let mut exec_cpu = Executor::new(system.clone());
    let out_cpu = exec_cpu.run_plan(&mut plan_cpu, &mut traj)?;

    let mut traj = XtcReader::open(xtc_path)?;
    let mut plan_gpu = IonPairCorrelationPlan::new(sel, GroupBy::Resid, 2.0, 2.0)
        .with_group_types(group_types)
        .with_lag_mode(LagMode::Fft);
    let gpu = GpuContext::new(0)?;
    let mut exec_gpu = Executor::new(system).with_device(Device::Cuda(gpu));
    let out_gpu = exec_gpu.run_plan(&mut plan_gpu, &mut traj)?;

    match (out_cpu, out_gpu) {
        (
            PlanOutput::TimeSeries {
                time: t1,
                data: d1,
                rows: r1,
                cols: c1,
            },
            PlanOutput::TimeSeries {
                time: t2,
                data: d2,
                rows: r2,
                cols: c2,
            },
        ) => {
            if r1 == 0 || r2 == 0 {
                return Ok(());
            }
            assert_eq!(c1, 6);
            assert_eq!(c2, 6);
            let map_cpu = build_lag_index(&t1, max_lag);
            let map_gpu = build_lag_index(&t2, max_lag);
            for lag in 0..=max_lag {
                let (Some(i_cpu), Some(i_gpu)) = (map_cpu[lag], map_gpu[lag]) else {
                    continue;
                };
                let base_cpu = i_cpu * c1;
                let base_gpu = i_gpu * c2;
                for c in 0..c1 {
                    let a = d1[base_cpu + c];
                    let b = d2[base_gpu + c];
                    let diff = (a - b).abs();
                    let tol = 1.0e-3 + 1.0e-3 * b.abs();
                    assert!(
                        diff <= tol,
                        "lag {lag} col {c} diff {diff} > tol {tol} (cpu={a}, gpu={b})"
                    );
                }
            }
        }
        _ => panic!("unexpected output"),
    }
    Ok(())
}
