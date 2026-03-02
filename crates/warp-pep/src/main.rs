//! warp-pep CLI: peptide builder and mutation tool.

use clap::{Parser, Subcommand};
use warp_pep::builder;
use warp_pep::builder::RamaPreset;
use warp_pep::convert;
use warp_pep::disulfide;
use warp_pep::json_spec::BuildSpec;
use warp_pep::mutation;
use warp_pep::streaming::{PepOperation, StreamEmitter};

#[derive(Parser)]
#[command(
    name = "warp-pep",
    version,
    about = "Peptide builder & mutation CLI — construct, mutate, and export peptide structures",
    long_about = "
warp-pep builds all-atom peptide structures from amino acid sequences using \
internal-coordinate geometry (bond lengths, angles, dihedrals) for all 20 \
standard amino acids. Supports Amber force field naming conventions \
(CYX, HID, HIE, HIP, ASH, GLH, LYN), Ramachandran angle presets \
(alpha-helix, beta-sheet, polyproline-II), multi-chain construction, \
disulfide bond detection, point mutations, and 7 output formats \
(PDB, PDBx/CIF, XYZ, GRO, MOL2, CRD, LAMMPS).

INPUT MODES (mutually exclusive):
  --sequence / -s     One-letter codes: ACDEFGHIKLMNPQRSTVWY
  --three-letter / -t Dash-separated three-letter codes with Amber variants:
                      ALA-CYX-HID-GLU (standard + CYX HID HIE HIP ASH GLH LYN)
  --json / -j         JSON spec file for full control (angles, presets,
                      mutations, multi-chain — see 'build --help' for schema)

RAMACHANDRAN PRESETS (--preset):
  extended     φ=180°  ψ=180°   (default when no angles given)
  alpha-helix  φ=−57°  ψ=−47°
  beta-sheet   φ=−120° ψ=+130°  (anti-parallel)
  polyproline  φ=−75°  ψ=+145°  (PPII helix)

MUTATION SPEC FORMAT:
  <from><position><to>  e.g. A5G = mutate residue 5 from Ala to Gly
  Comma-separated for multiple: A5G,L10W",
    after_long_help = "\
EXAMPLES:
  # Build 5-residue alpha-helix, write PDB to stdout:
  warp-pep build -s AAAAA --preset alpha-helix --oxt

  # Build from three-letter codes with Amber naming:
  warp-pep build -t ALA-CYX-HID-GLU --oxt --detect-ss -o out.pdb

  # Build from JSON spec (single-chain):
  echo '{\"residues\":[\"ALA\",\"CYX\",\"HID\"],\"preset\":\"alpha-helix\",\"oxt\":true}' > spec.json
  warp-pep build -j spec.json

  # Multi-chain JSON spec:
  echo '{\"chains\":[{\"id\":\"A\",\"residues\":[\"ALA\",\"CYS\"],\"preset\":\"alpha-helix\"},{\"id\":\"B\",\"residues\":[\"GLY\",\"VAL\"]}],\"oxt\":true}' > mc.json
  warp-pep build -j mc.json -o multi.pdb

  # Mutate residue 2 from Ala to Gly in existing PDB:
  warp-pep mutate -i input.pdb -m A2G -o mutated.pdb

  # Build + mutate in one shot (no input file needed):
  warp-pep mutate -s ACDEF -m C2G,D3W -o out.pdb

  # Custom backbone angles (phi/psi per inter-residue junction):
  warp-pep build -s AAA --phi=-60,-60 --psi=-45,-45 --oxt"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, global = true)]
    /// Enable NDJSON streaming progress events to stderr
    stream: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a peptide from sequence, three-letter codes, or JSON spec.
    ///
    /// Exactly one input mode required: --sequence, --three-letter, or --json.
    /// JSON mode is self-contained (may include angles, mutations, output path).
    #[command(after_long_help = "JSON SPEC SCHEMA (single-chain):\n\
  {\n\
    \"residues\": [\"ALA\", \"CYX\", \"HID\", \"GLU\"],\n\
    \"preset\":   \"alpha-helix\",  // optional, overrides phi/psi\n\
    \"phi\":      [-60.0, -60.0, -60.0],  // optional, length = N-1\n\
    \"psi\":      [-45.0, -45.0, -45.0],  // optional, length = N-1\n\
    \"omega\":    [180.0, 180.0, 180.0],  // optional, length = N-1\n\
    \"oxt\":      true,\n\
    \"detect_ss\": true,\n\
    \"mutations\": [\"A1G\"],\n\
    \"output\":   \"out.pdb\",\n\
    \"format\":   \"pdb\"\n\
  }\n\
JSON SPEC SCHEMA (multi-chain):\n\
  {\n\
    \"chains\": [\n\
      { \"id\": \"A\", \"residues\": [\"ALA\", \"CYS\"], \"preset\": \"alpha-helix\" },\n\
      { \"id\": \"B\", \"residues\": [\"GLY\", \"VAL\"] }\n\
    ],\n\
    \"oxt\": true, \"detect_ss\": true\n\
  }")]
    Build {
        /// One-letter amino acid sequence.
        /// Valid codes: G A S C V I L T R K D E N Q M H P F Y W.
        /// Example: "ACDEFG"
        #[arg(short, long, group = "input_mode")]
        sequence: Option<String>,

        /// Three-letter dash-separated sequence; supports Amber ff variants.
        /// Standard: ALA, ARG, ASN, ASP, CYS, GLN, GLU, GLY, HIS, ILE,
        ///           LEU, LYS, MET, PHE, PRO, SER, THR, TRP, TYR, VAL.
        /// Amber:    CYX (SS-bonded Cys), HID (Nδ-H), HIE (Nε-H),
        ///           HIP (doubly protonated), ASH, GLH, LYN.
        /// Example: "ALA-CYX-HID-GLU"
        #[arg(short = 't', long, group = "input_mode")]
        three_letter: Option<String>,

        /// Path to a JSON build spec file. Self-contained: may include
        /// residues, preset, phi/psi/omega angles, mutations, multi-chain
        /// definitions, output path, and format. See after_long_help for schema.
        #[arg(short = 'j', long, group = "input_mode")]
        json: Option<String>,

        /// Output file path. If omitted, writes PDB to stdout.
        /// Format auto-detected from extension:
        ///   .pdb .cif .mmcif .xyz .gro .mol2 .crd .lmp .lammps
        #[arg(short, long)]
        output: Option<String>,

        /// Explicit output format string (overrides extension detection).
        /// Values: pdb, pdbx, xyz, gro, mol2, crd, lammps.
        #[arg(short, long)]
        format: Option<String>,

        /// Append terminal OXT oxygen to the last residue of each chain.
        #[arg(long, default_value_t = false)]
        oxt: bool,

        /// Ramachandran angle preset. Overrides --phi/--psi if given.
        ///   extended:    φ=180°  ψ=180°   (default when no angles given)
        ///   alpha-helix: φ=−57°  ψ=−47°
        ///   beta-sheet:  φ=−120° ψ=+130°
        ///   polyproline: φ=−75°  ψ=+145°
        /// Aliases accepted: alpha, helix, beta, sheet, ppii, ext.
        #[arg(long)]
        preset: Option<String>,

        /// Per-junction phi angles in degrees, comma-separated.
        /// Length must equal (num_residues − 1).
        /// Example for 3 residues: --phi=-60,-60
        #[arg(long)]
        phi: Option<String>,

        /// Per-junction psi angles in degrees, comma-separated.
        /// Length must equal (num_residues − 1). Must pair with --phi.
        /// Example for 3 residues: --psi=-45,-45
        #[arg(long)]
        psi: Option<String>,

        /// Per-junction omega (peptide bond) angles in degrees, comma-separated.
        /// Length must equal (num_residues − 1). Defaults to 180° (trans).
        /// Use ~0° for cis peptide bonds (e.g. preceding proline).
        #[arg(long)]
        omega: Option<String>,

        /// Scan CYS pairs for Sγ–Sγ distance < 2.5 Å and relabel as CYX
        /// (Amber disulfide convention). Also records SSBOND metadata.
        #[arg(long, default_value_t = false)]
        detect_ss: bool,
    },
    /// Mutate residue(s) in an existing or freshly-built peptide.
    ///
    /// Source structure: --input (file), --sequence (one-letter), or --three-letter.
    /// Mutation spec format: <from><position><to> where from/to are one-letter AA codes
    /// and position is the 1-based residue index. Comma-separate multiples.
    /// Example: A5G = Ala→Gly at position 5.  A5G,L10W = two mutations.
    #[command(after_long_help = "EXAMPLES:\n\
  warp-pep mutate -i input.pdb -m A5G -o out.pdb\n\
  warp-pep mutate -s ACDEF -m C2G,D3W --oxt -o out.pdb\n\
  warp-pep mutate -t ALA-CYX-HID -m H3W --detect-ss")]
    Mutate {
        /// Input structure file to mutate. Supported formats:
        ///   PDB, PDBx/CIF, XYZ, GRO, MOL2, Amber CRD, LAMMPS.
        /// If omitted, build from --sequence or --three-letter instead.
        #[arg(short, long)]
        input: Option<String>,

        /// One-letter amino acid sequence to build before mutating.
        /// Used when --input is not given.
        #[arg(short = 'S', long)]
        sequence: Option<String>,

        /// Three-letter dash-separated sequence (Amber variants OK).
        /// Used when --input is not given.
        #[arg(short = 't', long)]
        three_letter: Option<String>,

        /// Mutation spec(s): <from><pos><to>, comma-separated.
        /// from/to = one-letter AA code, pos = 1-based residue index.
        /// Examples: "A5G" or "A5G,L10W"
        #[arg(short, long)]
        mutations: String,

        /// Output file path. If omitted, writes PDB to stdout.
        #[arg(short, long)]
        output: Option<String>,

        /// Explicit output format (overrides extension detection).
        /// Values: pdb, pdbx, xyz, gro, mol2, crd, lammps.
        #[arg(short, long)]
        format: Option<String>,

        /// Append terminal OXT oxygen to the last residue.
        #[arg(long, default_value_t = false)]
        oxt: bool,

        /// Ramachandran preset for --sequence / --three-letter source.
        /// Ignored when --input is used.
        #[arg(long)]
        preset: Option<String>,

        /// Detect disulfide bonds (Sγ–Sγ < 2.5 Å → CYX).
        #[arg(long, default_value_t = false)]
        detect_ss: bool,
    },
}

fn parse_angles(s: &str) -> Result<Vec<f64>, String> {
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<f64>()
                .map_err(|e| format!("invalid angle '{}': {}", v, e))
        })
        .collect()
}

fn emit(
    struc: &warp_pep::residue::Structure,
    output: &Option<String>,
    format: &Option<String>,
) -> Result<(), String> {
    match output {
        Some(path) => convert::write_structure(struc, path, format.as_deref()),
        None => {
            let fmt = format.as_deref().unwrap_or("pdb");
            convert::write_structure_stdout(struc, fmt)
        }
    }
}

fn structure_total_atoms(struc: &warp_pep::residue::Structure) -> usize {
    struc
        .chains
        .iter()
        .map(|chain| {
            chain
                .residues
                .iter()
                .map(|res| res.atoms.len())
                .sum::<usize>()
        })
        .sum()
}

fn main() {
    let cli = Cli::parse();

    let emitter = StreamEmitter::new(cli.stream);

    let result = match cli.command {
        Commands::Build {
            sequence,
            three_letter,
            json,
            output,
            format,
            oxt,
            preset,
            phi,
            psi,
            omega,
            detect_ss,
        } => run_build(
            sequence.as_deref(),
            three_letter.as_deref(),
            json.as_deref(),
            &output,
            &format,
            oxt,
            detect_ss,
            preset.as_deref(),
            phi.as_deref(),
            psi.as_deref(),
            omega.as_deref(),
            emitter,
        ),
        Commands::Mutate {
            input,
            sequence,
            three_letter,
            mutations,
            output,
            format,
            oxt,
            preset,
            detect_ss,
        } => run_mutate(
            input.as_deref(),
            sequence.as_deref(),
            three_letter.as_deref(),
            &mutations,
            &output,
            &format,
            oxt,
            detect_ss,
            preset.as_deref(),
            emitter,
        ),
    };

    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run_build(
    sequence: Option<&str>,
    three_letter: Option<&str>,
    json: Option<&str>,
    output: &Option<String>,
    format: &Option<String>,
    oxt: bool,
    detect_ss: bool,
    preset: Option<&str>,
    phi: Option<&str>,
    psi: Option<&str>,
    omega: Option<&str>,
    emitter: StreamEmitter,
) -> Result<(), String> {
    use std::time::Instant;

    let t_start = Instant::now();

    // Emit operation started event
    if emitter.is_enabled() {
        // Try to get input path for context
        let input_path = json.or(sequence).or(three_letter).map(|s| s.to_string());
        let total_residues = sequence
            .as_ref()
            .map(|s| s.len())
            .or_else(|| three_letter.map(|s| s.split('-').count()))
            .unwrap_or(0);
        emitter.emit_operation_started(&warp_pep::streaming::OperationStartedEvent {
            operation: PepOperation::Build,
            input_path,
            total_chains: 1,
            total_residues,
            total_mutations: None,
        });
    }

    // JSON path: self-contained, ignores other flags
    if let Some(json_path) = json {
        let spec = BuildSpec::from_file(json_path)?;
        let struc = spec.execute()?;
        let out = spec.output.as_ref().or(output.as_ref()).cloned();
        let fmt = spec.format.as_ref().or(format.as_ref()).cloned();

        emit(&struc, &out, &fmt)?;

        if emitter.is_enabled() {
            let total_atoms = structure_total_atoms(&struc);
            let total_residues = struc.total_residues();
            emitter.emit_operation_complete(&warp_pep::streaming::OperationCompleteEvent {
                operation: PepOperation::Build,
                total_atoms,
                total_residues,
                total_chains: struc.chains.len(),
                output_path: out,
                elapsed_ms: warp_pep::streaming::duration_ms(t_start.elapsed()),
            });
        }

        return Ok(());
    }

    // Resolve preset if given
    let rama = match preset {
        Some(p) => Some(RamaPreset::from_str(p).ok_or_else(|| format!("unknown preset '{}'", p))?),
        None => None,
    };

    // Parse omega once if supplied
    let omega_v = omega.map(|o| parse_angles(o)).transpose()?;

    // Three-letter path
    if let Some(tl) = three_letter {
        let specs = builder::parse_three_letter_sequence(&tl.to_uppercase())?;
        let mut struc = if let Some(preset) = rama {
            builder::make_preset_structure_from_specs(&specs, preset)?
        } else {
            match (phi, psi) {
                (Some(phi_s), Some(psi_s)) => {
                    let phi_v = parse_angles(phi_s)?;
                    let psi_v = parse_angles(psi_s)?;
                    builder::make_structure_from_specs(&specs, &phi_v, &psi_v, omega_v.as_deref())?
                }
                (None, None) => builder::make_extended_structure_from_specs(&specs)?,
                _ => return Err("must provide both --phi and --psi or neither".into()),
            }
        };
        if oxt {
            builder::add_terminal_oxt(&mut struc);
        }
        if detect_ss {
            disulfide::detect_disulfides(&mut struc);
        }

        emit(&struc, output, format)?;

        if emitter.is_enabled() {
            let total_atoms = structure_total_atoms(&struc);
            let total_residues = struc.total_residues();
            emitter.emit_operation_complete(&warp_pep::streaming::OperationCompleteEvent {
                operation: PepOperation::Build,
                total_atoms,
                total_residues,
                total_chains: struc.chains.len(),
                output_path: output.clone(),
                elapsed_ms: warp_pep::streaming::duration_ms(t_start.elapsed()),
            });
        }
        return Ok(());
    }

    // One-letter path — preserve case for D-amino support (lowercase = D-form)
    let seq = sequence.ok_or("provide --sequence, --three-letter, or --json")?;

    let mut struc = if let Some(preset) = rama {
        builder::make_preset_structure(seq, preset)?
    } else {
        match (phi, psi) {
            (Some(phi_s), Some(psi_s)) => {
                let phi_v = parse_angles(phi_s)?;
                let psi_v = parse_angles(psi_s)?;
                builder::make_structure(&seq, &phi_v, &psi_v, omega_v.as_deref())?
            }
            (None, None) => builder::make_extended_structure(&seq)?,
            _ => return Err("must provide both --phi and --psi or neither".into()),
        }
    };

    if oxt {
        builder::add_terminal_oxt(&mut struc);
    }
    if detect_ss {
        disulfide::detect_disulfides(&mut struc);
    }

    emit(&struc, output, format)?;

    if emitter.is_enabled() {
        let total_atoms = structure_total_atoms(&struc);
        let total_residues = struc.total_residues();
        emitter.emit_operation_complete(&warp_pep::streaming::OperationCompleteEvent {
            operation: PepOperation::Build,
            total_atoms,
            total_residues,
            total_chains: struc.chains.len(),
            output_path: output.clone(),
            elapsed_ms: warp_pep::streaming::duration_ms(t_start.elapsed()),
        });
    }
    Ok(())
}

fn run_mutate(
    input: Option<&str>,
    sequence: Option<&str>,
    three_letter: Option<&str>,
    mutations: &str,
    output: &Option<String>,
    format: &Option<String>,
    oxt: bool,
    detect_ss: bool,
    preset: Option<&str>,
    emitter: StreamEmitter,
) -> Result<(), String> {
    use std::time::Instant;

    let t_start = Instant::now();

    // Count mutations for progress tracking
    let mutation_specs: Vec<&str> = mutations
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect();
    let total_mutations = mutation_specs.len();

    // Emit operation started event
    if emitter.is_enabled() {
        let input_path = input.or(sequence).or(three_letter).map(|s| s.to_string());
        emitter.emit_operation_started(&warp_pep::streaming::OperationStartedEvent {
            operation: PepOperation::Mutate,
            input_path,
            total_chains: 1,
            total_residues: 0,
            total_mutations: Some(total_mutations),
        });
    }

    let rama = match preset {
        Some(p) => Some(RamaPreset::from_str(p).ok_or_else(|| format!("unknown preset '{}'", p))?),
        None => None,
    };

    let mut struc = if let Some(path) = input {
        convert::read_structure(path)?
    } else if let Some(tl) = three_letter {
        let specs = builder::parse_three_letter_sequence(&tl.to_uppercase())?;
        match rama {
            Some(preset) => builder::make_preset_structure_from_specs(&specs, preset)?,
            None => builder::make_extended_structure_from_specs(&specs)?,
        }
    } else if let Some(seq) = sequence {
        match rama {
            Some(preset) => builder::make_preset_structure(seq, preset)?,
            None => builder::make_extended_structure(seq)?,
        }
    } else {
        return Err("provide --input, --sequence, or --three-letter".into());
    };

    for (idx, spec) in mutation_specs.iter().enumerate() {
        let t_mutation = Instant::now();
        let (from, pos, to) = mutation::parse_mutation_spec(spec)?;
        let result = mutation::mutate_residue_checked(&mut struc, Some(from), pos, to);

        if emitter.is_enabled() {
            let successful = result.is_ok();
            emitter.emit_mutation_complete(&warp_pep::streaming::MutationCompleteEvent {
                mutation_index: idx + 1,
                total_mutations,
                mutation_spec: spec.to_string(),
                successful,
                elapsed_ms: warp_pep::streaming::duration_ms(t_mutation.elapsed()),
            });
        }

        result?;
    }

    if oxt {
        builder::add_terminal_oxt(&mut struc);
    }
    if detect_ss {
        disulfide::detect_disulfides(&mut struc);
    }

    emit(&struc, output, format)?;

    if emitter.is_enabled() {
        let total_atoms = structure_total_atoms(&struc);
        let total_residues = struc.total_residues();
        emitter.emit_operation_complete(&warp_pep::streaming::OperationCompleteEvent {
            operation: PepOperation::Mutate,
            total_atoms,
            total_residues,
            total_chains: struc.chains.len(),
            output_path: output.clone(),
            elapsed_ms: warp_pep::streaming::duration_ms(t_start.elapsed()),
        });
    }

    Ok(())
}
