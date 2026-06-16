use anyhow::Result;
use clap::Parser;
use warp_cg::{agent, build_contract, forcefield, simulate_contract};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
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
    /// Build coarse-grained systems from first-class build contracts.
    Build {
        #[command(subcommand)]
        command: BuildCommand,
    },
    /// Plan and inspect CG simulation handoffs without owning execution.
    Simulate {
        #[command(subcommand)]
        command: SimulateCommand,
    },
    /// Inspect or install bundled CG force-field snapshots.
    Forcefield {
        #[command(subcommand)]
        command: ForcefieldCommand,
    },
}

#[derive(clap::Subcommand, Debug)]
enum BuildCommand {
    /// Run a warp-cg build request JSON.
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
    /// Validate a warp-cg build request JSON.
    Validate {
        request: Option<String>,
        #[arg(long)]
        stdin: bool,
    },
    /// Print build JSON schema.
    Schema {
        #[arg(long, value_parser = ["request", "result", "event"], default_value = "request")]
        kind: String,
    },
    /// Print an example build request.
    Example,
    /// Print build capabilities.
    Capabilities,
}

#[derive(clap::Subcommand, Debug)]
enum SimulateCommand {
    /// Print simulate JSON schema.
    Schema {
        #[arg(long, value_parser = ["request", "plan", "result", "status", "manifest"], default_value = "request")]
        kind: String,
    },
    /// Print an example simulate request.
    Example {
        #[arg(long, value_parser = ["gromacs", "openmm"], default_value = "gromacs")]
        engine: String,
    },
    /// Print simulate capabilities.
    Capabilities,
    /// Validate a simulate request JSON.
    Validate {
        request: Option<String>,
        #[arg(long)]
        stdin: bool,
    },
    /// Emit an execution handoff plan.
    Plan {
        request: Option<String>,
        #[arg(long)]
        stdin: bool,
        #[arg(long, value_parser = ["gromacs", "openmm"])]
        engine: Option<String>,
    },
    /// Inspect a run directory for artifacts/checkpoints/status.
    Status { run_dir: String },
}

#[derive(clap::Subcommand, Debug)]
enum ForcefieldCommand {
    /// Print the bundled forcefield manifest.
    Inspect {
        #[arg(long, default_value = "martini3")]
        kind: String,
    },
    /// Copy a bundled forcefield snapshot into a project-local directory.
    Install {
        #[arg(long, default_value = "martini3")]
        kind: String,
        #[arg(long)]
        dest: String,
        #[arg(long)]
        overwrite: bool,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(command) = args.command {
        return run_agent_command(command);
    }
    anyhow::bail!(
        "warp-cg requires a contract subcommand: run, validate, schema, example, or capabilities"
    )
}

fn run_agent_command(command: Command) -> Result<()> {
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
        Command::Build { command } => return run_build_command(command),
        Command::Simulate { command } => return run_simulate_command(command),
        Command::Forcefield { command } => return run_forcefield_command(command),
    }
    Ok(())
}

fn run_forcefield_command(command: ForcefieldCommand) -> Result<()> {
    match command {
        ForcefieldCommand::Inspect { kind } => {
            let manifest = forcefield::bundled_manifest_json(&kind)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        ForcefieldCommand::Install {
            kind,
            dest,
            overwrite,
        } => {
            let manifest = forcefield::install_bundled_forcefield(
                &kind,
                std::path::Path::new(&dest),
                overwrite,
            )?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
    }
    Ok(())
}

fn run_build_command(command: BuildCommand) -> Result<()> {
    match command {
        BuildCommand::Run {
            request,
            stdin,
            stream,
        } => {
            let payload = read_payload(request, stdin)?;
            let (exit_code, result) =
                build_contract::run_request_json(&payload, stream == "ndjson");
            println!("{}", serde_json::to_string_pretty(&result)?);
            std::process::exit(exit_code);
        }
        BuildCommand::Validate { request, stdin } => {
            let payload = read_payload(request, stdin)?;
            let (exit_code, result) = build_contract::validate_request_json(&payload);
            println!("{}", serde_json::to_string_pretty(&result)?);
            std::process::exit(exit_code);
        }
        BuildCommand::Schema { kind } => {
            println!("{}", build_contract::schema_json(&kind)?);
        }
        BuildCommand::Example => {
            println!(
                "{}",
                serde_json::to_string_pretty(&build_contract::example_request())?
            );
        }
        BuildCommand::Capabilities => {
            println!(
                "{}",
                serde_json::to_string_pretty(&build_contract::capabilities())?
            );
        }
    }
    Ok(())
}

fn run_simulate_command(command: SimulateCommand) -> Result<()> {
    match command {
        SimulateCommand::Schema { kind } => {
            println!("{}", simulate_contract::schema_json(&kind)?);
        }
        SimulateCommand::Example { engine } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&simulate_contract::example_request(&engine)?)?
            );
        }
        SimulateCommand::Capabilities => {
            println!(
                "{}",
                serde_json::to_string_pretty(&simulate_contract::capabilities())?
            );
        }
        SimulateCommand::Validate { request, stdin } => {
            let payload = read_payload(request, stdin)?;
            let (exit_code, result) = simulate_contract::validate_request_json(&payload);
            println!("{}", serde_json::to_string_pretty(&result)?);
            std::process::exit(exit_code);
        }
        SimulateCommand::Plan {
            request,
            stdin,
            engine,
        } => {
            let payload = read_payload(request, stdin)?;
            let (exit_code, result) =
                simulate_contract::plan_request_json(&payload, engine.as_deref());
            println!("{}", serde_json::to_string_pretty(&result)?);
            std::process::exit(exit_code);
        }
        SimulateCommand::Status { run_dir } => {
            let (exit_code, result) = simulate_contract::status_json(run_dir);
            println!("{}", serde_json::to_string_pretty(&result)?);
            std::process::exit(exit_code);
        }
    }
    Ok(())
}

fn read_payload(request: Option<String>, stdin: bool) -> Result<String> {
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
}
