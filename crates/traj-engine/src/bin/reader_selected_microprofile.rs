use std::cmp::Ordering;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{FrameChunk, FrameChunkBuilder};
use traj_core::selection::Selection;
use traj_core::system::System;
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
    chunk_frames: usize,
    repeats: usize,
    warmup_repeats: usize,
    dcd_length_scale: f32,
    cache_frames: usize,
    json_out: Option<PathBuf>,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            top: PathBuf::new(),
            traj: PathBuf::new(),
            selection: "protein and name CA".to_string(),
            chunk_frames: 128,
            repeats: 6,
            warmup_repeats: 1,
            dcd_length_scale: 1.0,
            cache_frames: 1024,
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
struct ReaderSelectedReport {
    dataset: DatasetInfo,
    config: ConfigInfo,
    selection: SelectionInfo,
    cache: CacheInfo,
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
    chunk_frames: usize,
    repeats: usize,
    warmup_repeats: usize,
    dcd_length_scale: f32,
    cache_frames_target: usize,
}

#[derive(Debug, Serialize)]
struct SelectionInfo {
    n_total_atoms: usize,
    n_selected_atoms: usize,
    selected_fraction: f64,
}

#[derive(Debug, Serialize)]
struct CacheInfo {
    cached_frames: usize,
    cached_chunks: usize,
    cached_coords_checksum: f64,
}

#[derive(Debug, Serialize)]
struct PhaseInfo {
    read_selected_e2e: PhaseReport,
    read_full_e2e: PhaseReport,
    copy_selected_from_cached_full: PhaseReport,
}

#[derive(Debug, Serialize)]
struct DerivedInfo {
    total_frames: usize,
    selected_sec_per_frame: f64,
    full_sec_per_frame: f64,
    copy_sec_per_cached_frame: f64,
    copy_sec_est_for_total_frames: f64,
    estimated_copy_fraction_of_selected: f64,
    estimated_decode_fraction_of_selected: f64,
    selected_over_full_sec_ratio: f64,
}

fn usage() {
    println!("Reader selected-path microprofiler (engine IO split)");
    println!("Usage:");
    println!("  cargo run --release -p traj-engine --bin reader_selected_microprofile -- \\");
    println!("    --top <topology.pdb|gro|pdbqt> --traj <traj.xtc|dcd|pdb|pdbqt> [options]");
    println!();
    println!("Options:");
    println!("  --selection <expr>           default: \"protein and name CA\"");
    println!("  --chunk <int>                default: 128");
    println!("  --repeats <int>              default: 6");
    println!("  --warmup-repeats <int>       default: 1");
    println!("  --cache-frames <int>         default: 1024");
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
            "--cache-frames" => {
                args.cache_frames =
                    parse_usize(&next_value(&mut it, "--cache-frames")?, "--cache-frames")?
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
    if args.chunk_frames == 0 {
        return Err(TrajError::Parse("--chunk must be > 0".into()));
    }
    if args.repeats == 0 {
        return Err(TrajError::Parse("--repeats must be > 0".into()));
    }
    if args.cache_frames == 0 {
        return Err(TrajError::Parse("--cache-frames must be > 0".into()));
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
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
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

fn read_selected_stream(
    traj: &Path,
    selection_u32: &[u32],
    chunk_frames: usize,
    dcd_length_scale: f32,
) -> TrajResult<(f64, usize, f64)> {
    let mut reader = open_reader(traj, dcd_length_scale)?;
    let mut builder = FrameChunkBuilder::new(selection_u32.len(), chunk_frames);
    builder.set_requirements(false, false);
    let t0 = Instant::now();
    let mut n_frames = 0usize;
    let mut checksum = 0.0f64;
    loop {
        let read = reader.read_chunk_selected(chunk_frames, selection_u32, &mut builder)?;
        if read == 0 {
            break;
        }
        let chunk = builder.finish_take()?;
        n_frames += chunk.n_frames;
        checksum += coords_checksum(&chunk.coords);
        builder.reclaim(chunk);
    }
    Ok((t0.elapsed().as_secs_f64(), n_frames, checksum))
}

fn read_full_stream(
    traj: &Path,
    n_atoms: usize,
    chunk_frames: usize,
    dcd_length_scale: f32,
) -> TrajResult<(f64, usize, f64)> {
    let mut reader = open_reader(traj, dcd_length_scale)?;
    let mut builder = FrameChunkBuilder::new(n_atoms, chunk_frames);
    builder.set_requirements(false, false);
    let t0 = Instant::now();
    let mut n_frames = 0usize;
    let mut checksum = 0.0f64;
    loop {
        let read = reader.read_chunk(chunk_frames, &mut builder)?;
        if read == 0 {
            break;
        }
        let chunk = builder.finish_take()?;
        n_frames += chunk.n_frames;
        checksum += coords_checksum(&chunk.coords);
        builder.reclaim(chunk);
    }
    Ok((t0.elapsed().as_secs_f64(), n_frames, checksum))
}

fn cache_full_chunks(
    traj: &Path,
    n_atoms: usize,
    chunk_frames: usize,
    cache_frames_target: usize,
    dcd_length_scale: f32,
) -> TrajResult<(Vec<FrameChunk>, usize, f64)> {
    let mut reader = open_reader(traj, dcd_length_scale)?;
    let mut builder = FrameChunkBuilder::new(n_atoms, chunk_frames);
    builder.set_requirements(false, false);
    let mut chunks = Vec::<FrameChunk>::new();
    let mut cached_frames = 0usize;
    let mut checksum = 0.0f64;
    while cached_frames < cache_frames_target {
        let read = reader.read_chunk(chunk_frames, &mut builder)?;
        if read == 0 {
            break;
        }
        let chunk = builder.finish_take()?;
        cached_frames += chunk.n_frames;
        checksum += coords_checksum(&chunk.coords);
        chunks.push(chunk);
        builder = FrameChunkBuilder::new(n_atoms, chunk_frames);
        builder.set_requirements(false, false);
    }
    if cached_frames == 0 {
        return Err(TrajError::Parse(
            "failed to cache full chunks: trajectory had zero frames".into(),
        ));
    }
    Ok((chunks, cached_frames, checksum))
}

fn copy_selected_from_cached(
    cached_chunks: &[FrameChunk],
    selection_usize: &[usize],
) -> TrajResult<(f64, usize, f64)> {
    if cached_chunks.is_empty() {
        return Err(TrajError::Parse("cached_chunks is empty".into()));
    }
    if selection_usize.is_empty() {
        return Err(TrajError::Parse("selection_usize is empty".into()));
    }
    let t0 = Instant::now();
    let mut total_frames = 0usize;
    let mut checksum = 0.0f64;
    let mut out = vec![[0.0f32; 4]; selection_usize.len() * cached_chunks[0].n_frames.max(1)];
    for chunk in cached_chunks {
        let needed = selection_usize.len() * chunk.n_frames;
        if out.len() < needed {
            out.resize(needed, [0.0; 4]);
        }
        let mut dst_off = 0usize;
        for frame_coords in chunk
            .coords
            .chunks_exact(chunk.n_atoms)
            .take(chunk.n_frames)
        {
            for &src_idx in selection_usize {
                out[dst_off] = frame_coords[src_idx];
                dst_off += 1;
            }
        }
        checksum += coords_checksum(&out[..dst_off]);
        total_frames += chunk.n_frames;
    }
    Ok((t0.elapsed().as_secs_f64(), total_frames, checksum))
}

fn run() -> TrajResult<()> {
    let args = parse_args()?;
    let mut system = load_system(&args.top)?;
    let selection: Selection = system.select(&args.selection)?;
    let selection_u32 = selection.indices.as_ref().clone();
    if selection_u32.is_empty() {
        return Err(TrajError::Parse("selection resolved to zero atoms".into()));
    }
    let selection_usize: Vec<usize> = selection_u32.iter().map(|&x| x as usize).collect();
    let n_atoms = system.n_atoms();
    let total_repeats = args.repeats + args.warmup_repeats;

    let mut selected_sec = Vec::<f64>::with_capacity(total_repeats);
    let mut selected_checksum = Vec::<f64>::with_capacity(total_repeats);
    let mut total_frames = None::<usize>;
    for _ in 0..total_repeats {
        let (sec, frames, checksum) = read_selected_stream(
            &args.traj,
            &selection_u32,
            args.chunk_frames,
            args.dcd_length_scale,
        )?;
        if let Some(expected) = total_frames {
            if expected != frames {
                return Err(TrajError::Parse(format!(
                    "selected read frame count mismatch: expected {expected}, got {frames}"
                )));
            }
        } else {
            total_frames = Some(frames);
        }
        selected_sec.push(sec);
        selected_checksum.push(checksum);
    }
    let total_frames = total_frames.unwrap_or(0);
    if total_frames == 0 {
        return Err(TrajError::Parse(
            "selected read produced zero frames".into(),
        ));
    }

    let mut full_sec = Vec::<f64>::with_capacity(total_repeats);
    let mut full_checksum = Vec::<f64>::with_capacity(total_repeats);
    for _ in 0..total_repeats {
        let (sec, frames, checksum) = read_full_stream(
            &args.traj,
            n_atoms,
            args.chunk_frames,
            args.dcd_length_scale,
        )?;
        if frames != total_frames {
            return Err(TrajError::Parse(format!(
                "full read frame count mismatch: expected {total_frames}, got {frames}"
            )));
        }
        full_sec.push(sec);
        full_checksum.push(checksum);
    }

    let (cached_chunks, cached_frames, cached_checksum) = cache_full_chunks(
        &args.traj,
        n_atoms,
        args.chunk_frames,
        args.cache_frames,
        args.dcd_length_scale,
    )?;

    let mut copy_sec = Vec::<f64>::with_capacity(total_repeats);
    let mut copy_checksum = Vec::<f64>::with_capacity(total_repeats);
    for _ in 0..total_repeats {
        let (sec, frames, checksum) = copy_selected_from_cached(&cached_chunks, &selection_usize)?;
        if frames != cached_frames {
            return Err(TrajError::Parse(format!(
                "cached copy frame count mismatch: expected {cached_frames}, got {frames}"
            )));
        }
        copy_sec.push(sec);
        copy_checksum.push(checksum);
    }

    let read_selected = phase_report(
        selected_sec.as_slice(),
        selected_checksum.as_slice(),
        args.warmup_repeats,
        total_frames,
    )?;
    let read_full = phase_report(
        full_sec.as_slice(),
        full_checksum.as_slice(),
        args.warmup_repeats,
        total_frames,
    )?;
    let copy_selected = phase_report(
        copy_sec.as_slice(),
        copy_checksum.as_slice(),
        args.warmup_repeats,
        cached_frames,
    )?;

    let selected_sec_per_frame = read_selected.timing.sec_mean / total_frames as f64;
    let full_sec_per_frame = read_full.timing.sec_mean / total_frames as f64;
    let copy_sec_per_cached_frame = copy_selected.timing.sec_mean / cached_frames as f64;
    let copy_sec_est_for_total_frames = copy_sec_per_cached_frame * total_frames as f64;
    let estimated_copy_fraction = if read_selected.timing.sec_mean > 0.0 {
        copy_sec_est_for_total_frames / read_selected.timing.sec_mean
    } else {
        0.0
    };
    let estimated_decode_fraction = (1.0 - estimated_copy_fraction).max(0.0);
    let selected_over_full_sec_ratio = if read_full.timing.sec_mean > 0.0 {
        read_selected.timing.sec_mean / read_full.timing.sec_mean
    } else {
        0.0
    };

    println!("=== Reader Selected-Path Microprofile ===");
    println!("top={} traj={}", args.top.display(), args.traj.display());
    println!(
        "selection=\"{}\" selected_atoms={} total_atoms={} frames={} cache_frames={} cache_chunks={}",
        args.selection,
        selection_u32.len(),
        n_atoms,
        total_frames,
        cached_frames,
        cached_chunks.len()
    );
    println!(
        "read_selected_e2e: fps_mean={:.2} sec_mean={:.6}",
        read_selected.timing.fps_mean, read_selected.timing.sec_mean
    );
    println!(
        "read_full_e2e: fps_mean={:.2} sec_mean={:.6}",
        read_full.timing.fps_mean, read_full.timing.sec_mean
    );
    println!(
        "copy_selected_from_cached_full: fps_mean={:.2} sec_mean={:.6}",
        copy_selected.timing.fps_mean, copy_selected.timing.sec_mean
    );
    println!(
        "derived: selected/full sec ratio={:.4} | est copy frac={:.4} | est decode frac={:.4}",
        selected_over_full_sec_ratio, estimated_copy_fraction, estimated_decode_fraction
    );

    let report = ReaderSelectedReport {
        dataset: DatasetInfo {
            top: args.top.display().to_string(),
            traj: args.traj.display().to_string(),
        },
        config: ConfigInfo {
            selection_expr: args.selection.clone(),
            chunk_frames: args.chunk_frames,
            repeats: args.repeats,
            warmup_repeats: args.warmup_repeats,
            dcd_length_scale: args.dcd_length_scale,
            cache_frames_target: args.cache_frames,
        },
        selection: SelectionInfo {
            n_total_atoms: n_atoms,
            n_selected_atoms: selection_u32.len(),
            selected_fraction: selection_u32.len() as f64 / n_atoms as f64,
        },
        cache: CacheInfo {
            cached_frames,
            cached_chunks: cached_chunks.len(),
            cached_coords_checksum: cached_checksum,
        },
        phases: PhaseInfo {
            read_selected_e2e: read_selected,
            read_full_e2e: read_full,
            copy_selected_from_cached_full: copy_selected,
        },
        derived: DerivedInfo {
            total_frames,
            selected_sec_per_frame,
            full_sec_per_frame,
            copy_sec_per_cached_frame,
            copy_sec_est_for_total_frames,
            estimated_copy_fraction_of_selected: estimated_copy_fraction,
            estimated_decode_fraction_of_selected: estimated_decode_fraction,
            selected_over_full_sec_ratio,
        },
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
