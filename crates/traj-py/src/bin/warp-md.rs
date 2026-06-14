use std::ffi::OsString;
use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgAction, Parser, Subcommand, ValueEnum};
use traj_py::contract;

#[derive(Parser)]
#[command(name = "warp-md", version, about = "Native warp-md agent contract CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum OutputFormat {
    Json,
    Yaml,
}

#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
enum SchemaKind {
    Request,
    Result,
    Event,
    PlotManifest,
}

impl SchemaKind {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Result => "result",
            Self::Event => "event",
            Self::PlotManifest => "plot-manifest",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum ValidateKind {
    Request,
    Result,
    Event,
}

#[derive(Subcommand)]
enum Command {
    /// Print a JSON schema for agent contracts.
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
    /// Validate an agent request/result/event payload.
    Validate {
        request: Option<PathBuf>,
        #[arg(long)]
        stdin: bool,
        #[arg(long, value_enum, default_value_t = ValidateKind::Request)]
        kind: ValidateKind,
        #[arg(long)]
        strict: bool,
        #[arg(long)]
        check_selections: bool,
    },
    /// Print warp-md agent capabilities.
    Capabilities {
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    /// Print the full analysis contract catalog.
    Catalog {
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    /// Print one analysis contract schema by name or alias.
    PlanSchema {
        name: String,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    /// Print an agent request template for one analysis.
    Template {
        analysis: String,
        #[arg(long)]
        fill_defaults: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    /// Normalize request aliases and optional defaults.
    Normalize {
        request: Option<PathBuf>,
        #[arg(long)]
        stdin: bool,
        #[arg(long)]
        strip_unknown: bool,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    /// Lint a selection expression, optionally against a topology path.
    LintSelection {
        expr: String,
        #[arg(long, default_value = "selection")]
        field_type: String,
        #[arg(long)]
        system_path: Option<String>,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    /// Suggest analyses for a natural-language goal.
    Suggest {
        goal: String,
        #[arg(long = "field")]
        provided_fields: Vec<String>,
        #[arg(long, default_value_t = 5)]
        top_n: usize,
        #[arg(long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
        #[arg(long)]
        json: bool,
    },
    /// Delegate Python runtime commands, including analysis execution.
    #[command(disable_help_flag = true)]
    Run {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, action = ArgAction::Append)]
        args: Vec<OsString>,
    },
    /// Delegate legacy Python CLI commands such as rg, frames, list-plans, plot, and mcp.
    #[command(external_subcommand)]
    Python(Vec<OsString>),
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, String> {
    match cli.command {
        Command::Schema {
            kind,
            format,
            json,
            out,
        } => {
            let value = contract::agent_schema_value(kind.as_str())?;
            write_value(&value, resolve_format(format, json), out.as_ref())?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Validate {
            request,
            stdin,
            kind,
            strict,
            check_selections,
        } => {
            let text = read_payload(request.as_ref(), stdin)?;
            let value_result = match kind {
                ValidateKind::Request => {
                    contract::agent_validate_request_json(&text, strict, check_selections)
                }
                ValidateKind::Result => contract::agent_validate_result_json(&text),
                ValidateKind::Event => contract::agent_validate_event_json(&text),
            };
            let value = match value_result {
                Ok(value) => value,
                Err(message) => invalid_json_validation_error(kind, message),
            };
            print_value(&value, OutputFormat::Json)?;
            Ok(
                if value.get("valid").and_then(serde_json::Value::as_bool) == Some(true) {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(2)
                },
            )
        }
        Command::Capabilities { format, json } => {
            print_value(
                &contract::agent_capabilities_value(),
                resolve_format(format, json),
            )?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Catalog { format, json } => {
            let value = contract::agent_contract_catalog_value()?;
            print_value(&value, resolve_format(format, json))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::PlanSchema { name, format, json } => {
            let value = contract::agent_plan_schema_value(&name)?;
            print_value(&value, resolve_format(format, json))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Template {
            analysis,
            fill_defaults,
            format,
            json,
        } => {
            let value = contract::agent_generate_template_value(&analysis, fill_defaults)?;
            print_value(&value, resolve_format(format, json))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Normalize {
            request,
            stdin,
            strip_unknown,
            format,
            json,
        } => {
            let text = read_payload(request.as_ref(), stdin)?;
            let value = contract::agent_normalize_request_json(&text, strip_unknown)?;
            print_value(&value, resolve_format(format, json))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::LintSelection {
            expr,
            field_type,
            system_path,
            format,
            json,
        } => {
            let value =
                contract::agent_lint_selection_value(&expr, &field_type, system_path.as_deref());
            print_value(&value, resolve_format(format, json))?;
            Ok(
                if value.get("valid").and_then(serde_json::Value::as_bool) == Some(true) {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from(2)
                },
            )
        }
        Command::Suggest {
            goal,
            provided_fields,
            top_n,
            format,
            json,
        } => {
            let value = contract::agent_suggest_analyses_value(&goal, &provided_fields, top_n);
            print_value(&value, resolve_format(format, json))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Run { args } => delegate_python_cli("run", args),
        Command::Python(args) => delegate_python_external(args),
    }
}

fn resolve_format(format: OutputFormat, json: bool) -> OutputFormat {
    if json {
        OutputFormat::Json
    } else {
        format
    }
}

fn read_payload(path: Option<&PathBuf>, stdin: bool) -> Result<String, String> {
    if stdin {
        let mut text = String::new();
        io::stdin()
            .read_to_string(&mut text)
            .map_err(|err| err.to_string())?;
        return Ok(text);
    }
    let Some(path) = path else {
        return Err("request path is required unless --stdin is used".into());
    };
    fs::read_to_string(path).map_err(|err| format!("failed to read {}: {err}", path.display()))
}

fn write_value(
    value: &serde_json::Value,
    format: OutputFormat,
    out: Option<&PathBuf>,
) -> Result<(), String> {
    let text = render_value(value, format)?;
    if let Some(path) = out {
        fs::write(path, text).map_err(|err| format!("failed to write {}: {err}", path.display()))
    } else {
        println!("{text}");
        Ok(())
    }
}

fn print_value(value: &serde_json::Value, format: OutputFormat) -> Result<(), String> {
    write_value(value, format, None)
}

fn render_value(value: &serde_json::Value, format: OutputFormat) -> Result<String, String> {
    match format {
        OutputFormat::Json => serde_json::to_string_pretty(value).map_err(|err| err.to_string()),
        OutputFormat::Yaml => serde_yaml::to_string(value).map_err(|err| err.to_string()),
    }
}

fn invalid_json_validation_error(
    kind: ValidateKind,
    message: impl Into<String>,
) -> serde_json::Value {
    let normalized_key = match kind {
        ValidateKind::Request => "normalized_request",
        ValidateKind::Result => "normalized_result",
        ValidateKind::Event => "normalized_event",
    };
    let mut value = serde_json::json!({
        "schema_version": "warp-md.agent.v1",
        "status": "error",
        "valid": false,
        "errors": [{
            "code": "E_SCHEMA_VALIDATION",
            "path": "root",
            "message": message.into(),
            "context": {},
        }],
        "warnings": [],
    });
    value
        .as_object_mut()
        .expect("validation error is an object")
        .insert(normalized_key.into(), serde_json::Value::Null);
    value
}

fn delegate_python_external(args: Vec<OsString>) -> Result<ExitCode, String> {
    let Some((command, rest)) = args.split_first() else {
        return Err("python runtime command is required".into());
    };
    delegate_python_cli(command, rest.to_vec())
}

fn delegate_python_cli(
    command: impl Into<OsString>,
    args: Vec<OsString>,
) -> Result<ExitCode, String> {
    let status =
        std::process::Command::new(std::env::var_os("PYTHON").unwrap_or_else(|| "python".into()))
            .arg("-m")
            .arg("warp_md.cli")
            .arg(command.into())
            .args(args)
            .status()
            .map_err(|err| format!("failed to delegate to Python warp_md.cli: {err}"))?;
    Ok(ExitCode::from(status.code().unwrap_or(1) as u8))
}
