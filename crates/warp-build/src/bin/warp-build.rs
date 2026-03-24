use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "warp-build", version, about = "Native warp-build stage CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Json,
    Yaml,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum SchemaKind {
    Request,
    Result,
    Event,
    SourceBundle,
    BuildManifest,
    ChargeManifest,
    TopologyGraph,
}

impl SchemaKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Result => "result",
            Self::Event => "event",
            Self::SourceBundle => "source_bundle",
            Self::BuildManifest => "build_manifest",
            Self::ChargeManifest => "charge_manifest",
            Self::TopologyGraph => "topology_graph",
        }
    }
}

#[derive(Subcommand)]
enum Command {
    Schema {
        #[arg(long, value_enum, default_value_t = SchemaKind::Request)]
        kind: SchemaKind,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Example {
        #[arg(long, default_value = "random_walk")]
        mode: String,
        #[arg(long, default_value = "source.bundle.json")]
        bundle_path: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    ExampleBundle {
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    Capabilities {
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    InspectSource {
        source: PathBuf,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    Validate {
        request: PathBuf,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    Run {
        request: PathBuf,
        #[arg(long)]
        stream: bool,
    },
}

fn selected_format(format: OutputFormat, json: bool) -> OutputFormat {
    if json {
        OutputFormat::Json
    } else {
        format
    }
}

fn render_value<T: Serialize>(value: &T, format: OutputFormat) -> Result<String, String> {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(value).map_err(|err| err.to_string()),
        OutputFormat::Yaml => serde_yaml::to_string(value).map_err(|err| err.to_string()),
    }
}

fn print_value<T: Serialize>(value: &T, format: OutputFormat) -> Result<(), String> {
    println!("{}", render_value(value, format)?);
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}

fn run() -> Result<u8, String> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Schema {
            kind,
            format,
            json,
            out,
        } => {
            let value: serde_json::Value = serde_json::from_str(
                &warp_build::schema_json(kind.as_str()).map_err(|err| err.to_string())?,
            )
            .map_err(|err| err.to_string())?;
            let text = render_value(&value, selected_format(format, json))?;
            if let Some(path) = out {
                fs::write(&path, format!("{text}\n")).map_err(|err| err.to_string())?;
                println!("{}", path.display());
            } else {
                print!("{text}");
            }
            Ok(0)
        }
        Command::Example {
            mode,
            bundle_path,
            format,
            json,
        } => {
            print_value(
                &warp_build::example_request_for_bundle(&mode, &bundle_path),
                selected_format(format, json),
            )?;
            Ok(0)
        }
        Command::ExampleBundle { out, format, json } => {
            if let Some(path) = out {
                warp_build::write_example_bundle(path.to_string_lossy().as_ref())?;
                println!("{}", path.display());
            } else {
                print_value(&warp_build::example_bundle(), selected_format(format, json))?;
            }
            Ok(0)
        }
        Command::Capabilities { format, json } => {
            print_value(&warp_build::capabilities(), selected_format(format, json))?;
            Ok(0)
        }
        Command::InspectSource {
            source,
            format,
            json,
        } => {
            let (code, value) = warp_build::inspect_source_json(&source.to_string_lossy());
            print_value(&value, selected_format(format, json))?;
            Ok(code as u8)
        }
        Command::Validate {
            request,
            format,
            json,
        } => {
            let text = fs::read_to_string(request).map_err(|err| err.to_string())?;
            let (code, value) = warp_build::validate_request_json(&text);
            print_value(&value, selected_format(format, json))?;
            Ok(code as u8)
        }
        Command::Run { request, stream } => {
            let text = fs::read_to_string(request).map_err(|err| err.to_string())?;
            let (code, value) = warp_build::run_request_json(&text, stream);
            if !stream {
                print_value(&value, OutputFormat::Json)?;
            }
            Ok(code as u8)
        }
    }
}
