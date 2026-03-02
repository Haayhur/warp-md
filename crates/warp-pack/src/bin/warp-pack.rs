use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use warp_pack::config::{OutputSpec, PackConfig};
use warp_pack::error::{PackError, PackResult};
use warp_pack::io::write_output;
use warp_pack::pack::run_with_stream;
use warp_pack::streaming::{PackCompleteEvent, PackProfile, StreamEmitter};

#[derive(Parser)]
#[command(name = "warp-pack", version, about = "CPU packing utility")]
struct Cli {
    #[arg(short, long)]
    config: PathBuf,
    #[arg(short, long)]
    output: Option<PathBuf>,
    #[arg(short, long)]
    format: Option<String>,
    #[arg(long)]
    /// Enable NDJSON streaming progress events to stderr
    stream: bool,
}

fn main() -> Result<(), String> {
    if let Err(err) = run_cli() {
        return Err(err.to_string());
    }
    Ok(())
}

fn run_cli() -> PackResult<()> {
    let t_start = Instant::now();
    let cli = Cli::parse();
    let cfg = load_config(&cli.config)?;
    let emitter = StreamEmitter::new(cli.stream);
    let out = run_with_stream(&cfg, emitter)?;
    let output = if let Some(path) = cli.output {
        Some(OutputSpec {
            path: path.to_string_lossy().to_string(),
            format: cli.format.unwrap_or_else(|| "pdb".to_string()),
            scale: None,
        })
    } else {
        cfg.output.clone()
    };
    if let Some(ref spec) = output {
        let add_box_sides = cfg.add_box_sides || cfg.pbc;
        let box_fix = if cfg.add_box_sides {
            cfg.add_box_sides_fix.unwrap_or(0.0)
        } else {
            0.0
        };
        let write_conect = !cfg.ignore_conect;
        write_output(
            &out,
            &spec,
            add_box_sides,
            box_fix,
            write_conect,
            cfg.hexadecimal_indices,
        )?;
    }
    if let Some(path) = &cfg.write_crd {
        let scale = output.as_ref().and_then(|spec| spec.scale).unwrap_or(1.0);
        let box_fix = cfg.add_box_sides_fix.unwrap_or(0.0);
        warp_pack::io::write_crd(&out, path, scale, box_fix)?;
    }
    if emitter.is_enabled() {
        let elapsed_ms: u64 = t_start.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
        let total_molecules: usize = cfg.structures.iter().map(|s| s.count).sum();
        emitter.emit_pack_complete(&PackCompleteEvent {
            total_atoms: out.atoms.len(),
            total_molecules,
            final_box_size: out.box_size,
            output_path: output.as_ref().map(|spec| spec.path.clone()),
            elapsed_ms,
            profile: PackProfile::default(),
        });
    }
    Ok(())
}

fn load_config(path: &PathBuf) -> PackResult<PackConfig> {
    let content = fs::read_to_string(path)?;
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    if ext == "inp" {
        warp_pack::inp::parse_packmol_inp(&content)
    } else if ext == "yaml" || ext == "yml" {
        serde_yaml::from_str(&content)
            .map_err(|e| PackError::Parse(format!("yaml parse error: {e}")))
    } else {
        serde_json::from_str(&content)
            .map_err(|e| PackError::Parse(format!("json parse error: {e}")))
    }
}
