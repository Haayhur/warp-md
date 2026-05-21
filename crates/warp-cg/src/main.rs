use anyhow::Result;
use clap::Parser;
use warp_cg::{agent, mapping, molecule, trajectory, xtb};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,

    /// SMILES string of the molecule
    #[arg(short, long)]
    smiles: Option<String>,

    /// Name of the compound
    #[arg(short, long, default_value = "molecule")]
    name: String,

    /// Path to a trajectory file (XTC, TRR, PDB, etc.)
    #[arg(short, long)]
    trajectory: Option<String>,

    /// Path to a topology file (if needed)
    #[arg(long)]
    topology: Option<String>,

    /// Output directory
    #[arg(short, long, default_value = ".")]
    out_dir: String,

    /// Run xTB MD to generate a reference trajectory
    #[arg(long)]
    run_xtb: bool,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Run a first-class agent request JSON.
    Run {
        /// Path to request JSON.
        request: Option<String>,
        /// Read request JSON from stdin.
        #[arg(long)]
        stdin: bool,
        /// Emit NDJSON progress events to stderr.
        #[arg(long, value_parser = ["none", "ndjson"], default_value = "none")]
        stream: String,
    },
    /// Validate a first-class agent request JSON.
    Validate {
        request: Option<String>,
        #[arg(long)]
        stdin: bool,
    },
    /// Print JSON schema.
    Schema {
        #[arg(long, value_parser = ["request", "result", "event"], default_value = "request")]
        kind: String,
    },
    /// Print an example request.
    Example,
    /// Print capabilities.
    Capabilities,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(command) = args.command {
        return run_agent_command(command);
    }
    let Some(smiles) = args.smiles.as_deref() else {
        anyhow::bail!("--smiles is required outside contract subcommands");
    };
    let out_path = std::path::Path::new(&args.out_dir);

    println!("Mapping molecule: {} with SMILES: {}", args.name, smiles);

    let mol = molecule::Molecule::from_smiles(smiles)?;
    println!("Parsed molecule with {} atoms", mol.graph.node_count());

    let mapping_res = mapping::map_molecule(&mol);

    let mut final_traj_path = args.trajectory.clone();

    if args.run_xtb {
        let xtb_res = xtb::run_xtb_pipeline(&args.name, smiles, out_path)?;
        println!("xTB pipeline completed.");
        if let Some(trj) = xtb_res.trajectory_trj {
            final_traj_path = Some(trj.to_string_lossy().to_string());
        } else {
            final_traj_path = Some(xtb_res.opt_xyz.to_string_lossy().to_string());
            println!("Using optimized structure as reference (MD failed or skipped).");
        }
    }

    if let Some(traj_path) = final_traj_path {
        let out_traj = format!("{}/{}_cg.xtc", args.out_dir, args.name);
        let bead_mapping = trajectory::BeadMapping {
            bead_names: mapping_res.bead_names,
            atom_indices: mapping_res.atom_groups,
        };
        trajectory::map_trajectory(&traj_path, &out_traj, &bead_mapping)?;
        println!("Mapped trajectory to {}", out_traj);
    }

    Ok(())
}

fn run_agent_command(command: Command) -> Result<()> {
    let read_payload = |request: Option<String>, stdin: bool| -> Result<String> {
        if stdin {
            use std::io::Read;
            let mut text = String::new();
            std::io::stdin().read_to_string(&mut text)?;
            return Ok(text);
        }
        let Some(path) = request else {
            anyhow::bail!("request path is required unless --stdin is used");
        };
        Ok(std::fs::read_to_string(path)?)
    };

    match command {
        Command::Run {
            request,
            stdin,
            stream,
        } => {
            let payload = read_payload(request, stdin)?;
            let (exit_code, result) = agent::run_request_json(&payload, stream == "ndjson");
            println!("{}", serde_json::to_string_pretty(&result)?);
            std::process::exit(exit_code);
        }
        Command::Validate { request, stdin } => {
            let payload = read_payload(request, stdin)?;
            let (exit_code, result) = agent::validate_request_json(&payload);
            println!("{}", serde_json::to_string_pretty(&result)?);
            std::process::exit(exit_code);
        }
        Command::Schema { kind } => {
            println!("{}", agent::schema_json(&kind)?);
        }
        Command::Example => {
            println!(
                "{}",
                serde_json::to_string_pretty(&agent::example_request())?
            );
        }
        Command::Capabilities => {
            println!("{}", serde_json::to_string_pretty(&agent::capabilities())?);
        }
    }
    Ok(())
}
