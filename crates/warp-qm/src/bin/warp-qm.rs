use std::process::ExitCode;

fn main() -> ExitCode {
    match warp_qm::cli::run_from_env() {
        Ok(code) => ExitCode::from(code),
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}
