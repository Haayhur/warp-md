use anyhow::{anyhow, Result};
use sci_form::{embed, ConformerResult};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::molecule::atomic_number_to_symbol;

pub struct XtbResult {
    pub opt_xyz: PathBuf,
    pub trajectory_trj: Option<PathBuf>,
}

pub fn run_xtb_pipeline(name: &str, smiles: &str, out_dir: &Path) -> Result<XtbResult> {
    run_xtb_pipeline_with_config(name, smiles, out_dir, &XtbRunConfig::default())
}

#[derive(Debug, Clone)]
pub struct XtbRunConfig {
    pub temperature_k: f64,
    pub time_ps: f64,
    pub timestep_fs: f64,
    pub dump_fs: f64,
    pub gfn: String,
    pub seed: u64,
}

impl Default for XtbRunConfig {
    fn default() -> Self {
        Self {
            temperature_k: 298.15,
            time_ps: 10.0,
            timestep_fs: 1.0,
            dump_fs: 100.0,
            gfn: "gfnff".to_string(),
            seed: 42,
        }
    }
}

pub fn run_xtb_pipeline_with_config(
    name: &str,
    smiles: &str,
    out_dir: &Path,
    config: &XtbRunConfig,
) -> Result<XtbResult> {
    // 1. Generate initial 3D coords with sci-form
    let res = embed(smiles, config.seed);
    if res.coords.is_empty() {
        return Err(anyhow!("sci-form failed to embed SMILES: {}", smiles));
    }

    let work_dir = out_dir.join(format!("{}_xtb_work", name));
    fs::create_dir_all(&work_dir)?;

    let initial_xyz = work_dir.join("initial.xyz");
    write_xyz_from_res(&res, &initial_xyz)?;

    // 2. Optimization
    let xtb_exe = find_xtb()?;

    println!("Running xtb optimization...");
    let mut opt_cmd = Command::new(&xtb_exe);
    opt_cmd.arg("initial.xyz");
    add_gfn_args(&mut opt_cmd, &config.gfn);
    let status = opt_cmd
        .arg("--opt")
        .arg("normal")
        .env("OMP_NUM_THREADS", "1")
        .env("OMP_STACKSIZE", "2G")
        .current_dir(&work_dir)
        .status()?;

    if !status.success() {
        return Err(anyhow!("xtb optimization failed"));
    }

    let opt_xyz = work_dir.join("xtbopt.xyz");
    if !opt_xyz.exists() {
        return Err(anyhow!("xtbopt.xyz not found after optimization"));
    }

    // 3. MD (Try, but don't fail if MD segfaults but opt succeeded)
    let md_inp = work_dir.join("md.inp");
    fs::write(
        &md_inp,
        format!(
            "$md\n  temp={:.6}\n  time={:.6}\n  dump={:.6}\n  step={:.6}\n  nvt=true\n$end\n",
            config.temperature_k, config.time_ps, config.dump_fs, config.timestep_fs
        ),
    )?;

    println!("Running xtb MD...");
    let mut md_cmd = Command::new(&xtb_exe);
    md_cmd.arg("xtbopt.xyz");
    add_gfn_args(&mut md_cmd, &config.gfn);
    let status = md_cmd
        .arg("--input")
        .arg("md.inp")
        .arg("--md")
        .env("OMP_NUM_THREADS", "1")
        .env("OMP_STACKSIZE", "2G")
        .current_dir(&work_dir)
        .status();

    let trj = work_dir.join("xtb.trj");
    let trajectory_trj = if let Ok(s) = status {
        if s.success() && trj.exists() {
            Some(trj)
        } else {
            None
        }
    } else {
        None
    };

    if trajectory_trj.is_none() {
        println!("Warning: xTB MD failed or was skipped. Only optimization result available.");
    }

    Ok(XtbResult {
        opt_xyz,
        trajectory_trj,
    })
}

fn add_gfn_args(command: &mut Command, gfn: &str) {
    match gfn.trim().to_ascii_lowercase().as_str() {
        "" | "gfnff" => {
            command.arg("--gfnff");
        }
        "0" | "gfn0" | "gfn0-xtb" => {
            command.arg("--gfn").arg("0");
        }
        "1" | "gfn1" | "gfn1-xtb" => {
            command.arg("--gfn").arg("1");
        }
        "2" | "gfn2" | "gfn2-xtb" => {
            command.arg("--gfn").arg("2");
        }
        other => {
            command.arg("--gfn").arg(other);
        }
    }
}

fn find_xtb() -> Result<PathBuf> {
    which::which("xtb").map_err(|_| anyhow!("xtb executable not found in PATH"))
}

fn write_xyz_from_res(res: &ConformerResult, path: &Path) -> Result<()> {
    let mut content = format!("{}\n\n", res.num_atoms);
    for i in 0..res.num_atoms {
        let symbol = atomic_number_to_symbol(res.elements[i]);
        let x = res.coords[i * 3];
        let y = res.coords[i * 3 + 1];
        let z = res.coords[i * 3 + 2];
        content.push_str(&format!("{} {:10.6} {:10.6} {:10.6}\n", symbol, x, y, z));
    }
    fs::write(path, content)?;
    Ok(())
}
