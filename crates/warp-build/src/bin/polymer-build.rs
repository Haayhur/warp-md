use std::fs;
use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "polymer-build",
    version,
    about = "Native polymer build stage CLI"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    Schema {
        #[arg(long, default_value = "request")]
        kind: String,
    },
    Example {
        #[arg(long, default_value = "random_walk")]
        mode: String,
    },
    ExampleBundle,
    Capabilities,
    InspectSource {
        source: PathBuf,
    },
    Validate {
        request: PathBuf,
    },
    Run {
        request: PathBuf,
        #[arg(long)]
        stream: bool,
    },
}

fn main() -> Result<(), String> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Schema { kind } => {
            let text = warp_build::schema_json(&kind).map_err(|err| err.to_string())?;
            println!("{text}");
            Ok(())
        }
        Command::Example { mode } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&warp_build::example_request(&mode))
                    .map_err(|err| err.to_string())?
            );
            Ok(())
        }
        Command::ExampleBundle => {
            println!(
                "{}",
                serde_json::to_string_pretty(&warp_build::example_bundle())
                    .map_err(|err| err.to_string())?
            );
            Ok(())
        }
        Command::Capabilities => {
            println!(
                "{}",
                serde_json::to_string_pretty(&warp_build::capabilities())
                    .map_err(|err| err.to_string())?
            );
            Ok(())
        }
        Command::InspectSource { source } => {
            let (code, value) = warp_build::inspect_source_json(&source.to_string_lossy());
            println!(
                "{}",
                serde_json::to_string_pretty(&value).map_err(|err| err.to_string())?
            );
            if code == 0 {
                Ok(())
            } else {
                Err(format!("inspect-source failed with exit code {code}"))
            }
        }
        Command::Validate { request } => {
            let text = fs::read_to_string(request).map_err(|err| err.to_string())?;
            let (code, value) = warp_build::validate_request_json(&text);
            println!(
                "{}",
                serde_json::to_string_pretty(&value).map_err(|err| err.to_string())?
            );
            if code == 0 {
                Ok(())
            } else {
                Err(format!("validate failed with exit code {code}"))
            }
        }
        Command::Run { request, stream } => {
            let text = fs::read_to_string(request).map_err(|err| err.to_string())?;
            let (code, value) = warp_build::run_request_json(&text, stream);
            if !stream {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&value).map_err(|err| err.to_string())?
                );
            }
            if code == 0 {
                Ok(())
            } else {
                Err(format!("run failed with exit code {code}"))
            }
        }
    }
}
