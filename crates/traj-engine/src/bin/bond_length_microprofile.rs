use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{FrameChunk, FrameChunkBuilder};
use traj_core::selection::Selection;
use traj_core::system::System;
use traj_engine::{BondLengthDistributionPlan, Device, Executor, Plan, PlanOutput};
use traj_io::dcd::DcdReader;
use traj_io::gro::GroReader;
use traj_io::pdb::{PdbReader, PdbqtReader};
use traj_io::pdb_traj::PdbTrajReader;
use traj_io::xtc::XtcReader;
use traj_io::{TopologyReader, TrajReader};

#[derive(Debug, Clone)]
struct Args {
    top: PathBuf,
    traj: PathBuf,
    selection: String,
    bins: usize,
    r_max: f32,
    chunk_frames: usize,
    repeats: usize,
    warmup_repeats: usize,
    dcd_length_scale: f32,
    json_out: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            top: PathBuf::new(),
            traj: PathBuf::new(),
            selection: "protein and name CA".to_string(),
            bins: 64,
            r_max: 10.0,
            chunk_frames: 128,
            repeats: 5,
            warmup_repeats: 1,
            dcd_length_scale: 1.0,
            json_out: None,
        }
    }
}

#[derive(Debug, Serialize)]
struct TimingSummary {
    sec_mean: f64,
    sec_median: f64,
    sec_std: f64,
    sec_min: f64,
    sec_max: f64,
    sec_samples: usize,
    warmup_repeats: usize,
    fps_mean: f64,
    fps_median: f64,
}

#[derive(Debug, Serialize)]
struct PhaseReport {
    timing: TimingSummary,
    checksum_mean: f64,
    checksum_std: f64,
}

#[derive(Debug, Serialize)]
struct MicroprofileReport {
    dataset: DatasetInfo,
    config: ConfigInfo,
    selection: SelectionInfo,
    baseline_cache: BaselineCacheInfo,
    phases: PhaseInfo,
    derived: DerivedInfo,
}

#[derive(Debug, Serialize)]
struct DatasetInfo {
    top: String,
    traj: String,
}

#[derive(Debug, Serialize)]
struct ConfigInfo {
    selection_expr: String,
    bins: usize,
    r_max: f32,
    chunk_frames: usize,
    repeats: usize,
    warmup_repeats: usize,
    dcd_length_scale: f32,
}

#[derive(Debug, Serialize)]
struct SelectionInfo {
    n_selected_atoms: usize,
    n_chains: usize,
    bonds_per_frame: usize,
}

#[derive(Debug, Serialize)]
struct BaselineCacheInfo {
    n_frames: usize,
    n_chunks: usize,
    coords_checksum: f64,
}

#[derive(Debug, Serialize)]
struct PhaseInfo {
    e2e_executor: PhaseReport,
    cache_selected: PhaseReport,
    compute_cached_chunks: PhaseReport,
}

#[derive(Debug, Serialize)]
struct DerivedInfo {
    e2e_minus_compute_sec_mean: f64,
    cache_fraction_of_e2e: f64,
    compute_fraction_of_e2e: f64,
}

fn usage() {
    println!("BondLengthDistributionPlan microprofiler (engine-only, CPU)");
    println!("Usage:");
    println!("  cargo run -p traj-engine --bin bond_length_microprofile -- \\");
    println!("    --top <topology.pdb|gro|pdbqt> --traj <traj.xtc|dcd|pdb|pdbqt> [options]");
    println!();
    println!("Options:");
    println!("  --selection <expr>           default: \"protein and name CA\"");
    println!("  --bins <int>                 default: 64");
    println!("  --r-max <float>              default: 10.0");
    println!("  --chunk <int>                default: 128");
    println!("  --repeats <int>              default: 5");
    println!("  --warmup-repeats <int>       default: 1");
    println!("  --dcd-length-scale <float>   default: 1.0");
    println!("  --json-out <path>            optional JSON output");
    println!("  --help                       show this message");
}

fn parse_args() -> TrajResult<Args> {
    let mut args = Args::default();
    let mut it = env::args().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            "--top" => args.top = PathBuf::from(next_value(&mut it, "--top")?),
            "--traj" => args.traj = PathBuf::from(next_value(&mut it, "--traj")?),
            "--selection" => args.selection = next_value(&mut it, "--selection")?,
            "--bins" => args.bins = parse_usize(&next_value(&mut it, "--bins")?, "--bins")?,
            "--r-max" => args.r_max = parse_f32(&next_value(&mut it, "--r-max")?, "--r-max")?,
            "--chunk" => {
                args.chunk_frames = parse_usize(&next_value(&mut it, "--chunk")?, "--chunk")?
            }
            "--repeats" => {
                args.repeats = parse_usize(&next_value(&mut it, "--repeats")?, "--repeats")?
            }
            "--warmup-repeats" => {
                args.warmup_repeats = parse_usize(
                    &next_value(&mut it, "--warmup-repeats")?,
                    "--warmup-repeats",
                )?
            }
            "--dcd-length-scale" => {
                args.dcd_length_scale = parse_f32(
                    &next_value(&mut it, "--dcd-length-scale")?,
                    "--dcd-length-scale",
                )?
            }
            "--json-out" => args.json_out = Some(PathBuf::from(next_value(&mut it, "--json-out")?)),
            _ => {
                return Err(TrajError::Parse(format!(
                    "unknown argument '{flag}' (use --help for usage)"
                )))
            }
        }
    }
    if args.top.as_os_str().is_empty() || args.traj.as_os_str().is_empty() {
        return Err(TrajError::Parse(
            "missing required --top and/or --traj (use --help)".into(),
        ));
    }
    if args.bins == 0 {
        return Err(TrajError::Parse("--bins must be > 0".into()));
    }
    if args.r_max <= 0.0 {
        return Err(TrajError::Parse("--r-max must be > 0".into()));
    }
    if args.chunk_frames == 0 {
        return Err(TrajError::Parse("--chunk must be > 0".into()));
    }
    if args.repeats == 0 {
        return Err(TrajError::Parse("--repeats must be > 0".into()));
    }
    Ok(args)
}

fn next_value(it: &mut impl Iterator<Item = String>, flag: &str) -> TrajResult<String> {
    it.next()
        .ok_or_else(|| TrajError::Parse(format!("missing value for {flag}")))
}

fn parse_usize(value: &str, flag: &str) -> TrajResult<usize> {
    value
        .parse::<usize>()
        .map_err(|_| TrajError::Parse(format!("invalid integer for {flag}: {value}")))
}

fn parse_f32(value: &str, flag: &str) -> TrajResult<f32> {
    value
        .parse::<f32>()
        .map_err(|_| TrajError::Parse(format!("invalid float for {flag}: {value}")))
}

fn extension_lower(path: &Path) -> Option<String> {
    path.extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_ascii_lowercase())
}

fn load_system(top: &Path) -> TrajResult<System> {
    match extension_lower(top).as_deref() {
        Some("pdb") => {
            let mut reader = PdbReader::new(top);
            reader.read_system()
        }
        Some("pdbqt") => {
            let mut reader = PdbqtReader::new(top);
            reader.read_system()
        }
        Some("gro") => {
            let mut reader = GroReader::new(top);
            reader.read_system()
        }
        _ => Err(TrajError::Unsupported(format!(
            "unsupported topology format for '{}'",
            top.display()
        ))),
    }
}

fn open_reader(traj: &Path, dcd_length_scale: f32) -> TrajResult<Box<dyn TrajReader>> {
    match extension_lower(traj).as_deref() {
        Some("xtc") => Ok(Box::new(XtcReader::open(traj)?)),
        Some("dcd") => Ok(Box::new(DcdReader::open(traj, dcd_length_scale)?)),
        Some("pdb") | Some("pdbqt") => Ok(Box::new(PdbTrajReader::open(traj)?)),
        _ => Err(TrajError::Unsupported(format!(
            "unsupported trajectory format for '{}'",
            traj.display()
        ))),
    }
}

fn coords_checksum(coords: &[[f32; 4]]) -> f64 {
    let mut acc = 0.0f64;
    for (i, c) in coords.iter().enumerate() {
        let w = (i as f64 % 7.0) + 1.0;
        acc += w * ((c[0] as f64) + 0.5 * (c[1] as f64) + 0.25 * (c[2] as f64));
    }
    acc
}

fn output_checksum(out: &PlanOutput) -> f64 {
    match out {
        PlanOutput::Histogram { counts, .. } => counts.iter().map(|&v| v as f64).sum(),
        PlanOutput::Series(v) => v.iter().map(|&x| x as f64).sum(),
        PlanOutput::Matrix { data, .. } => data.iter().map(|&x| x as f64).sum(),
        PlanOutput::TimeSeries { data, .. } => data.iter().map(|&x| x as f64).sum(),
        PlanOutput::Rdf(r) => r.counts.iter().map(|&v| v as f64).sum(),
        PlanOutput::Persistence(p) => p.bond_autocorrelation.iter().map(|&x| x as f64).sum(),
        PlanOutput::Dielectric(d) => d.rot_sq.iter().map(|&x| x as f64).sum(),
        PlanOutput::StructureFactor(s) => s.s_q.iter().map(|&x| x as f64).sum(),
        PlanOutput::Grid(g) => g.mean.iter().map(|&x| x as f64).sum(),
        PlanOutput::Pca(p) => p.eigenvalues.iter().map(|&x| x as f64).sum(),
        PlanOutput::Clustering(c) => c.labels.iter().map(|&x| x as f64).sum(),
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn stddev(values: &[f64], mu: f64) -> f64 {
    if values.len() < 2 {
        0.0
    } else {
        let var = values
            .iter()
            .map(|&x| {
                let d = x - mu;
                d * d
            })
            .sum::<f64>()
            / values.len() as f64;
        var.sqrt()
    }
}

fn timing_summary(
    samples_sec: &[f64],
    warmup_repeats: usize,
    n_frames: usize,
) -> TrajResult<TimingSummary> {
    if samples_sec.is_empty() {
        return Err(TrajError::Parse("no timing samples collected".into()));
    }
    let start = warmup_repeats.min(samples_sec.len());
    let active = &samples_sec[start..];
    if active.is_empty() {
        return Err(TrajError::Parse(
            "warmup_repeats consumed all samples; increase repeats".into(),
        ));
    }
    let mut sorted = active.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let sec_median = if sorted.len() % 2 == 0 {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    };
    let sec_mean = mean(active);
    let sec_std = stddev(active, sec_mean);
    let sec_min = sorted[0];
    let sec_max = sorted[sorted.len() - 1];
    let fps_mean = if sec_mean > 0.0 {
        n_frames as f64 / sec_mean
    } else {
        0.0
    };
    let fps_median = if sec_median > 0.0 {
        n_frames as f64 / sec_median
    } else {
        0.0
    };
    Ok(TimingSummary {
        sec_mean,
        sec_median,
        sec_std,
        sec_min,
        sec_max,
        sec_samples: active.len(),
        warmup_repeats,
        fps_mean,
        fps_median,
    })
}

fn polymer_shape(system: &System, selection: &[u32]) -> (usize, usize) {
    let mut by_chain: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for &idx in selection {
        by_chain
            .entry(system.atoms.chain_id[idx as usize])
            .or_default()
            .push(idx as usize);
    }
    let mut bonds = 0usize;
    for atoms in by_chain.values_mut() {
        atoms.sort_by_key(|&idx| (system.atoms.resid[idx], idx));
        if atoms.len() > 1 {
            bonds += atoms.len() - 1;
        }
    }
    (by_chain.len(), bonds)
}

fn read_selected_chunks(
    traj: &Path,
    selection: &[u32],
    chunk_frames: usize,
    dcd_length_scale: f32,
) -> TrajResult<(Vec<FrameChunk>, usize, f64)> {
    let mut reader = open_reader(traj, dcd_length_scale)?;
    let max_frames = chunk_frames.max(1);
    let mut builder = FrameChunkBuilder::new(selection.len(), max_frames);
    builder.set_requirements(false, false);
    let mut chunks = Vec::<FrameChunk>::new();
    let mut total_frames = 0usize;
    let mut checksum = 0.0f64;
    loop {
        let read = reader.read_chunk_selected(max_frames, selection, &mut builder)?;
        if read == 0 {
            break;
        }
        let chunk = builder.finish_take()?;
        total_frames += chunk.n_frames;
        checksum += coords_checksum(&chunk.coords);
        chunks.push(chunk);
        builder = FrameChunkBuilder::new(selection.len(), max_frames);
        builder.set_requirements(false, false);
    }
    Ok((chunks, total_frames, checksum))
}

fn run_e2e_once(system: &System, selection: &Selection, args: &Args) -> TrajResult<(f64, f64)> {
    let mut reader = open_reader(&args.traj, args.dcd_length_scale)?;
    let mut plan = BondLengthDistributionPlan::new(selection.clone(), args.bins, args.r_max);
    let mut exec = Executor::new(system.clone()).with_chunk_frames(args.chunk_frames);
    let t0 = Instant::now();
    let out = exec.run_plan(&mut plan, reader.as_mut())?;
    let elapsed = t0.elapsed().as_secs_f64();
    Ok((elapsed, output_checksum(&out)))
}

fn run_compute_cached_once(
    system: &System,
    selection: &Selection,
    selection_indices: &[u32],
    cached_chunks: &[FrameChunk],
    args: &Args,
) -> TrajResult<(f64, f64)> {
    let mut plan = BondLengthDistributionPlan::new(selection.clone(), args.bins, args.r_max);
    let device = Device::cpu();
    let t0 = Instant::now();
    plan.init(system, &device)?;
    for chunk in cached_chunks {
        plan.process_chunk_selected(chunk, selection_indices, system, &device)?;
    }
    let out = plan.finalize()?;
    let elapsed = t0.elapsed().as_secs_f64();
    Ok((elapsed, output_checksum(&out)))
}

fn phase_report(
    samples_sec: &[f64],
    checksums: &[f64],
    warmup: usize,
    n_frames: usize,
) -> TrajResult<PhaseReport> {
    let timing = timing_summary(samples_sec, warmup, n_frames)?;
    let start = warmup.min(checksums.len());
    let active = &checksums[start..];
    let checksum_mean = mean(active);
    let checksum_std = stddev(active, checksum_mean);
    Ok(PhaseReport {
        timing,
        checksum_mean,
        checksum_std,
    })
}

fn run() -> TrajResult<()> {
    let args = parse_args()?;
    let mut system = load_system(&args.top)?;
    let selection = system.select(&args.selection)?;
    if selection.indices.len() < 2 {
        return Err(TrajError::Parse(
            "selection must contain at least 2 atoms".into(),
        ));
    }
    let selection_indices = selection.indices.as_ref().clone();
    let (n_chains, bonds_per_frame) = polymer_shape(&system, &selection_indices);
    if bonds_per_frame == 0 {
        return Err(TrajError::Parse(
            "selection has zero adjacent bond pairs for bond-length histogram".into(),
        ));
    }

    let (baseline_chunks, n_frames, baseline_checksum) = read_selected_chunks(
        &args.traj,
        &selection_indices,
        args.chunk_frames,
        args.dcd_length_scale,
    )?;
    if n_frames == 0 {
        return Err(TrajError::Parse(
            "trajectory resolved to zero frames".into(),
        ));
    }
    let total_repeats = args.repeats + args.warmup_repeats;
    let mut e2e_sec = Vec::<f64>::with_capacity(total_repeats);
    let mut e2e_checksum = Vec::<f64>::with_capacity(total_repeats);
    for _ in 0..total_repeats {
        let (dt, checksum) = run_e2e_once(&system, &selection, &args)?;
        e2e_sec.push(dt);
        e2e_checksum.push(checksum);
    }

    let mut cache_sec = Vec::<f64>::with_capacity(total_repeats);
    let mut cache_checksum = Vec::<f64>::with_capacity(total_repeats);
    for _ in 0..total_repeats {
        let t0 = Instant::now();
        let (_chunks, frames, checksum) = read_selected_chunks(
            &args.traj,
            &selection_indices,
            args.chunk_frames,
            args.dcd_length_scale,
        )?;
        let dt = t0.elapsed().as_secs_f64();
        if frames != n_frames {
            return Err(TrajError::Parse(format!(
                "frame count mismatch during cache stage: expected {n_frames}, got {frames}"
            )));
        }
        cache_sec.push(dt);
        cache_checksum.push(checksum);
    }

    let mut compute_sec = Vec::<f64>::with_capacity(total_repeats);
    let mut compute_checksum = Vec::<f64>::with_capacity(total_repeats);
    for _ in 0..total_repeats {
        let (dt, checksum) = run_compute_cached_once(
            &system,
            &selection,
            &selection_indices,
            &baseline_chunks,
            &args,
        )?;
        compute_sec.push(dt);
        compute_checksum.push(checksum);
    }

    let e2e = phase_report(&e2e_sec, &e2e_checksum, args.warmup_repeats, n_frames)?;
    let cache = phase_report(&cache_sec, &cache_checksum, args.warmup_repeats, n_frames)?;
    let compute = phase_report(
        &compute_sec,
        &compute_checksum,
        args.warmup_repeats,
        n_frames,
    )?;

    let e2e_mean = e2e.timing.sec_mean;
    let cache_mean = cache.timing.sec_mean;
    let compute_mean = compute.timing.sec_mean;
    let derived = DerivedInfo {
        e2e_minus_compute_sec_mean: e2e_mean - compute_mean,
        cache_fraction_of_e2e: if e2e_mean > 0.0 {
            cache_mean / e2e_mean
        } else {
            0.0
        },
        compute_fraction_of_e2e: if e2e_mean > 0.0 {
            compute_mean / e2e_mean
        } else {
            0.0
        },
    };

    println!("=== BondLength Microprofile ===");
    println!("top={} traj={}", args.top.display(), args.traj.display());
    println!(
        "selection=\"{}\" selected_atoms={} chains={} bonds/frame={} frames={} chunks={}",
        args.selection,
        selection_indices.len(),
        n_chains,
        bonds_per_frame,
        n_frames,
        baseline_chunks.len()
    );
    println!(
        "e2e_executor: fps_mean={:.2} sec_mean={:.6} checksum_mean={:.3}",
        e2e.timing.fps_mean, e2e.timing.sec_mean, e2e.checksum_mean
    );
    println!(
        "cache_selected: fps_mean={:.2} sec_mean={:.6} checksum_mean={:.3}",
        cache.timing.fps_mean, cache.timing.sec_mean, cache.checksum_mean
    );
    println!(
        "compute_cached_chunks: fps_mean={:.2} sec_mean={:.6} checksum_mean={:.3}",
        compute.timing.fps_mean, compute.timing.sec_mean, compute.checksum_mean
    );
    println!(
        "derived: e2e_minus_compute_sec_mean={:.6} cache_frac={:.4} compute_frac={:.4}",
        derived.e2e_minus_compute_sec_mean,
        derived.cache_fraction_of_e2e,
        derived.compute_fraction_of_e2e
    );

    let report = MicroprofileReport {
        dataset: DatasetInfo {
            top: args.top.display().to_string(),
            traj: args.traj.display().to_string(),
        },
        config: ConfigInfo {
            selection_expr: args.selection.clone(),
            bins: args.bins,
            r_max: args.r_max,
            chunk_frames: args.chunk_frames,
            repeats: args.repeats,
            warmup_repeats: args.warmup_repeats,
            dcd_length_scale: args.dcd_length_scale,
        },
        selection: SelectionInfo {
            n_selected_atoms: selection_indices.len(),
            n_chains,
            bonds_per_frame,
        },
        baseline_cache: BaselineCacheInfo {
            n_frames,
            n_chunks: baseline_chunks.len(),
            coords_checksum: baseline_checksum,
        },
        phases: PhaseInfo {
            e2e_executor: e2e,
            cache_selected: cache,
            compute_cached_chunks: compute,
        },
        derived,
    };

    if let Some(path) = &args.json_out {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent)?;
            }
        }
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| TrajError::Parse(format!("json serialize error: {e}")))?;
        fs::write(path, json)?;
        println!("wrote json: {}", path.display());
    }

    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
