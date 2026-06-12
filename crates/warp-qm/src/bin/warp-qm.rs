use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;

#[derive(Parser)]
#[command(name = "warp-qm", version, about = "Native warp-qm agent contract CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Json,
    Yaml,
}

#[derive(Subcommand)]
enum Command {
    Schema {
        #[arg(long, default_value = "request")]
        kind: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Example {
        #[arg(long, default_value = "psi4")]
        engine: String,
        #[arg(long, default_value = "single_point")]
        task: String,
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
    Validate {
        request: PathBuf,
        #[arg(long, conflicts_with = "shallow")]
        deep: bool,
        #[arg(long, conflicts_with = "deep")]
        shallow: bool,
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
    InspectOutput {
        output: PathBuf,
        #[arg(long)]
        engine: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    ProjectCharges {
        charge_manifest: PathBuf,
        #[arg(long, short = 'n')]
        repeat_count: usize,
        #[arg(long, default_value = "mid")]
        repeat_set: String,
        #[arg(long, default_value = "repeat_tiled_no_terminal_specific_charges")]
        terminal_policy: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
        #[arg(long)]
        out: Option<PathBuf>,
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

fn with_validation_depth(text: &str, depth: &str) -> Result<String, String> {
    let mut value: serde_json::Value = serde_json::from_str(text).map_err(|err| err.to_string())?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| "request must decode to a JSON object".to_string())?;
    let validation = object
        .entry("validation".to_string())
        .or_insert_with(|| serde_json::json!({}));
    let validation_obj = validation
        .as_object_mut()
        .ok_or_else(|| "request.validation must decode to a JSON object".to_string())?;
    validation_obj.insert("depth".into(), serde_json::Value::String(depth.into()));
    serde_json::to_string(&value).map_err(|err| err.to_string())
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
            let kind = kind.replace('-', "_");
            let value: serde_json::Value = serde_json::from_str(&warp_qm::schema_json(&kind)?)
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
            engine,
            task,
            format,
            json,
        } => {
            print_value(
                &warp_qm::example_request(&engine, &task),
                selected_format(format, json),
            )?;
            Ok(0)
        }
        Command::Capabilities { format, json } => {
            print_value(&warp_qm::capabilities(), selected_format(format, json))?;
            Ok(0)
        }
        Command::Validate {
            request,
            deep,
            shallow,
            format,
            json,
        } => {
            let mut text = fs::read_to_string(request).map_err(|err| err.to_string())?;
            if deep {
                text = with_validation_depth(&text, "deep")?;
            } else if shallow {
                text = with_validation_depth(&text, "shallow")?;
            }
            let (code, value) = warp_qm::validate_request_json(&text);
            print_value(&value, selected_format(format, json))?;
            Ok(code as u8)
        }
        Command::Run { request, stream } => {
            let text = fs::read_to_string(request).map_err(|err| err.to_string())?;
            if !stream {
                eprintln!("warp-qm: run started; use --stream for structured NDJSON progress");
            }
            let (code, value) = warp_qm::run_request_json(&text, stream);
            if !stream {
                print_value(&value, OutputFormat::Json)?;
                eprintln!("warp-qm: run finished exit_code={code}");
            }
            Ok(code as u8)
        }
        Command::InspectOutput {
            output,
            engine,
            format,
            json,
        } => {
            let (code, value) = warp_qm::inspect_output_json(&output.to_string_lossy(), &engine);
            print_value(&value, selected_format(format, json))?;
            Ok(code as u8)
        }
        Command::ProjectCharges {
            charge_manifest,
            repeat_count,
            repeat_set,
            terminal_policy,
            format,
            json,
            out,
        } => {
            let (code, value) = warp_qm::project_polymer_charges_json(
                &charge_manifest.to_string_lossy(),
                repeat_count,
                &repeat_set,
                &terminal_policy,
            );
            let text = render_value(&value, selected_format(format, json))?;
            if let Some(path) = out {
                fs::write(&path, format!("{text}\n")).map_err(|err| err.to_string())?;
                println!("{}", path.display());
            } else {
                println!("{text}");
            }
            Ok(code as u8)
        }
    }
}
