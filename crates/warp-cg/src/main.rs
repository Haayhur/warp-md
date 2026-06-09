use anyhow::Result;
use clap::Parser;
use warp_cg::agent;

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
