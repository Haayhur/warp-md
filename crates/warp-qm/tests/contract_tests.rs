use std::fs;
use std::os::unix::fs::PermissionsExt;

use serde_json::Value;

#[test]
fn schema_targets_include_qm_handoff_manifests() {
    for kind in [
        "request",
        "result",
        "event",
        "job_manifest",
        "molecule",
        "charge_manifest",
        "polymer_charge_manifest",
        "esp_manifest",
        "cube_manifest",
        "engine_capabilities",
    ] {
        let schema = warp_qm::schema_json(kind).expect("schema");
        let payload: Value = serde_json::from_str(&schema).expect("json schema");
        assert_eq!(
            payload["$schema"],
            "http://json-schema.org/draft-07/schema#"
        );
    }
}

#[test]
fn project_polymer_charges_tiles_repeat_set() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_polymer_project_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let manifest = dir.join("charge_manifest.json");
    fs::write(
        &manifest,
        serde_json::json!({
            "schema_version": "warp-qm.charge-manifest.v1",
            "model": "resp",
            "charge_unit": "elementary_charge",
            "total_charge_e": 0.0,
            "atom_charges_e": [0.1, -0.1, 0.2, -0.2],
            "projection": {
                "policy": "fake_caps_redistributed_repeat_unit",
                "projected_charges_e": [0.0, 0.0, 0.2, -0.2],
                "deployable_sets": [
                    {"name": "mid", "atom_indices": [2, 3], "charges_e": [0.2, -0.2], "role": "interior_repeat"}
                ],
                "redistribution": [],
                "provenance": {}
            },
            "provenance": {}
        })
        .to_string(),
    )
    .expect("write manifest");
    let (code, payload) =
        warp_qm::project_polymer_charges_json(&manifest.to_string_lossy(), 3, "mid", "none");
    assert_eq!(code, 0, "{payload}");
    assert_eq!(
        payload["schema_version"],
        "warp-qm.polymer-charge-manifest.v1"
    );
    assert_eq!(payload["repeat_count"], 3);
    assert_eq!(payload["atom_charges_e"].as_array().unwrap().len(), 6);
    assert_eq!(payload["atom_charges_e"][0], 0.2);
    assert_eq!(payload["atom_charges_e"][5], -0.2);
    assert!(payload["total_charge_e"].as_f64().unwrap().abs() < 1e-12);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn example_request_validates_shallow() {
    let request = warp_qm::example_request("psi4", "single_point");
    let text = serde_json::to_string(&request).expect("request json");
    let (code, payload) = warp_qm::validate_request_json(&text);
    assert_eq!(code, 0);
    assert_eq!(payload["schema_version"], "warp-qm.agent.v1");
    assert_eq!(payload["valid"], true);
}

#[test]
fn promoted_reliable_tasks_validate_shallow() {
    for task in [
        "charges",
        "orca_molden_export",
        "population",
        "orbital_cube",
        "electron_density_cube",
        "elf_cube",
        "lol_cube",
        "laplacian_cube",
        "nmr_shielding",
        "binding_energy",
        "solvation_energy",
        "proton_affinity",
    ] {
        let request = warp_qm::example_request("orca", task);
        let text = serde_json::to_string(&request).expect("request json");
        let (code, payload) = warp_qm::validate_request_json(&text);
        assert_eq!(code, 0, "{task}: {payload}");
    }
}

#[test]
fn inspect_orca_output_returns_structured_energy() {
    let path = std::env::temp_dir().join(format!(
        "warp_qm_orca_test_{}_{}.out",
        std::process::id(),
        "energy"
    ));
    fs::write(
        &path,
        "FINAL SINGLE POINT ENERGY     -154.123456\nORCA TERMINATED NORMALLY\n",
    )
    .expect("write output");
    let (code, payload) = warp_qm::inspect_output_json(&path.to_string_lossy(), "orca");
    let _ = fs::remove_file(&path);
    assert_eq!(code, 0);
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["final_energy_hartree"], -154.123456);
    assert_eq!(payload["convergence_status"], "terminated_normally");
}

#[test]
fn inspect_orca_output_parses_frequency_charge_and_nmr_sections() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_orca_parse_sections_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let path = dir.join("job.out");
    fs::write(
        &path,
        "FINAL SINGLE POINT ENERGY     -1.0\nORCA TERMINATED NORMALLY\nVIBRATIONAL FREQUENCIES\n\n     0:      -12.34 cm**-1\n     1:        0.00 cm**-1\n     2:     1234.50 cm**-1\n\nMULLIKEN ATOMIC CHARGES\n   0 H :    0.100000\n   1 H :   -0.100000\nSum of atomic charges:    0.0000000\n\nHIRSHFELD ANALYSIS\n\n  ATOM     CHARGE      SPIN\n   0 H    0.000001    0.000000\n   1 H   -0.000001    0.000000\n\nCHEMICAL SHIELDING SUMMARY (ppm)\n\n  Nucleus  Element    Isotropic     Anisotropy\n  -------  -------  ------------   ------------\n      0       H           27.769          0.587\n",
    )
    .expect("write output");
    fs::write(
        &dir.join("job.xyz"),
        "2\njob\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n",
    )
    .expect("write xyz");
    let (code, payload) = warp_qm::inspect_output_json(&path.to_string_lossy(), "orca");
    assert_eq!(code, 0);
    assert_eq!(payload["frequencies_cm1"][0], -12.34);
    assert_eq!(payload["imaginary_frequency_count"], 1);
    assert_eq!(payload["charge_analysis"][0]["model"], "mulliken");
    assert_eq!(payload["charge_analysis"][1]["model"], "hirshfeld");
    assert_eq!(payload["nmr_shielding"][0]["isotropic_ppm"], 27.769);
    assert_eq!(payload["optimized_geometry"]["atoms"][1]["z"], 0.74);
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn orca_run_renders_input_and_manifest_before_missing_binary_error() {
    let dir = std::env::temp_dir().join(format!("warp_qm_orca_run_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let xyz = dir.join("water.xyz");
    fs::write(
        &xyz,
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.0 0.75 -0.24\n",
    )
    .expect("write xyz");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "orca-missing",
        "engine": {
            "name": "orca",
            "executable": dir.join("missing-orca").to_string_lossy(),
            "settings": {"basename": "water", "keywords": ["TightSCF"]}
        },
        "molecule": {
            "source": {"kind": "file", "path": xyz, "format": "xyz"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {
            "kind": "optimize",
            "method": "b3lyp",
            "basis": "def2-svp",
            "properties": ["charges"],
            "charge_model": "hirshfeld"
        },
        "runtime": {"work_dir": dir.join("work"), "threads": 2, "memory_mb": 4000},
        "output": {"out_dir": dir.join("work"), "write_json": true}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 4);
    assert_eq!(payload["status"], "error");
    let input = fs::read_to_string(dir.join("work/water.inp")).expect("input");
    assert!(input.contains("! b3lyp def2-svp Opt TightSCF"));
    assert!(input.contains("%output"));
    assert!(input.contains("%pal"));
    assert!(input.contains("* xyz 0 1"));
    assert!(dir.join("work/job_manifest.json").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn orca_generic_run_executes_agent_supplied_input() {
    let dir =
        std::env::temp_dir().join(format!("warp_qm_orca_generic_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let fake_orca = dir.join("fake_orca");
    fs::write(
        &fake_orca,
        "#!/bin/sh\nprintf 'FINAL SINGLE POINT ENERGY     -1.234567\\nORCA TERMINATED NORMALLY\\n'\n",
    )
    .expect("write fake orca");
    let mut perms = fs::metadata(&fake_orca).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&fake_orca, perms).expect("chmod");
    let input_text = "! HF 6-31G* TightSCF\n* xyz 0 1\nH 0 0 0\nH 0 0 0.74\n*\n";
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "orca-generic",
        "engine": {
            "name": "orca",
            "executable": fake_orca,
            "settings": {
                "input_name": "agent_job.inp",
                "input_text": input_text
            }
        },
        "molecule": {
            "source": {"kind": "engine_input", "format": "orca"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "generic_run", "method": "custom"},
        "runtime": {"work_dir": dir.join("work")},
        "output": {"out_dir": dir.join("work"), "write_json": true}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 0, "{payload}");
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["summary"]["energy_hartree"], -1.234567);
    let input = fs::read_to_string(dir.join("work/agent_job.inp")).expect("generic input");
    assert_eq!(input, input_text);
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(dir.join("work/job_manifest.json")).expect("manifest"),
    )
    .expect("manifest json");
    assert_eq!(manifest["provenance"]["input_mode"], "generic_engine_input");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn psi4_generic_run_executes_agent_supplied_input() {
    let dir =
        std::env::temp_dir().join(format!("warp_qm_psi4_generic_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let fake_psi4 = dir.join("fake_psi4");
    fs::write(
        &fake_psi4,
        "#!/bin/sh\nprintf 'SCF has converged\\nFinal Energy = -76.123456\\nPsi4 exiting successfully\\n' > \"$2\"\n",
    )
    .expect("write fake psi4");
    let mut perms = fs::metadata(&fake_psi4).expect("metadata").permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&fake_psi4, perms).expect("chmod");
    let input_text = "memory 1 GB\nmolecule { 0 1\nO 0 0 0\nH 0 0 0.96\nH 0 0.75 -0.24\n}\nenergy('scf/sto-3g')\n";
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "psi4-generic",
        "engine": {
            "name": "psi4",
            "executable": fake_psi4,
            "settings": {
                "input_name": "agent_job.dat",
                "output_name": "agent_job.out",
                "input_text": input_text
            }
        },
        "molecule": {
            "source": {"kind": "engine_input", "format": "psi4"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "generic_run", "method": "custom"},
        "runtime": {"work_dir": dir.join("work")},
        "output": {"out_dir": dir.join("work"), "write_json": true}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 0, "{payload}");
    assert_eq!(payload["status"], "ok");
    assert_eq!(payload["summary"]["energy_hartree"], -76.123456);
    let input = fs::read_to_string(dir.join("work/agent_job.dat")).expect("generic input");
    assert_eq!(input, input_text);
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(dir.join("work/job_manifest.json")).expect("manifest"),
    )
    .expect("manifest json");
    assert_eq!(manifest["provenance"]["input_mode"], "generic_engine_input");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn psi4_non_generic_tasks_are_not_implicitly_rendered() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_psi4_non_generic_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let xyz = dir.join("water.xyz");
    fs::write(
        &xyz,
        "3\nwater\nO 0.0 0.0 0.0\nH 0.0 0.0 0.96\nH 0.0 0.75 -0.24\n",
    )
    .expect("write xyz");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "psi4-rendered-not-ready",
        "engine": {"name": "psi4", "executable": dir.join("missing-psi4")},
        "molecule": {
            "source": {"kind": "file", "path": xyz, "format": "xyz"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "single_point", "method": "hf", "basis": "sto-3g"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 4);
    assert!(payload["warnings"][0]
        .as_str()
        .unwrap()
        .contains("supports only task.kind generic_run"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_resp_fit_writes_deterministic_script() {
    let dir = std::env::temp_dir().join(format!("warp_qm_multiwfn_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let fchk = dir.join("molecule.fch");
    fs::write(&fchk, "stub").expect("write fch");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-missing",
        "engine": {
            "name": "multiwfn",
            "executable": dir.join("missing-multiwfn").to_string_lossy()
        },
        "molecule": {
            "source": {"kind": "file", "path": fchk, "format": "fch"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "resp_fit", "method": "multiwfn"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 4);
    assert_eq!(payload["status"], "error");
    let script = fs::read_to_string(dir.join("work/multiwfn_input.txt")).expect("script");
    assert_eq!(script, "7\n18\n1\ny\n0\n0\nq\n");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_generic_run_writes_inline_menu_script() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_multiwfn_generic_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-generic-missing",
        "engine": {
            "name": "multiwfn",
            "executable": dir.join("missing-multiwfn").to_string_lossy(),
            "settings": {
                "menu_script": "7\n1\n1\ny\n0\nq\n",
                "expected_outputs": ["h2.chg"]
            }
        },
        "molecule": {
            "source": {"kind": "file", "path": molden, "format": "molden"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "generic_run", "method": "multiwfn"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 4);
    assert_eq!(payload["status"], "error");
    let script = fs::read_to_string(dir.join("work/multiwfn_input.txt")).expect("script");
    assert_eq!(script, "7\n1\n1\ny\n0\nq\n");
    assert_eq!(
        payload["properties"]["multiwfn_recipe"]["name"],
        "custom_menu_script"
    );
    assert_eq!(
        payload["properties"]["multiwfn_recipe"]["expected_outputs"][0],
        "h2.chg"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_generic_run_accepts_menu_script_file() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_multiwfn_generic_file_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    let script_path = dir.join("commands.txt");
    fs::write(&script_path, "q").expect("write script");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-generic-file-missing",
        "engine": {
            "name": "multiwfn",
            "executable": dir.join("missing-multiwfn").to_string_lossy(),
            "settings": {
                "menu_script_file": script_path
            }
        },
        "molecule": {
            "source": {"kind": "file", "path": molden, "format": "molden"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "generic_run", "method": "multiwfn"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 4);
    assert_eq!(payload["status"], "error");
    let script = fs::read_to_string(dir.join("work/multiwfn_input.txt")).expect("script");
    assert_eq!(script, "q\n");
    assert_eq!(
        payload["properties"]["multiwfn_recipe"]["name"],
        "custom_menu_script_file"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_generic_run_requires_menu_script() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_multiwfn_generic_missing_script_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-generic-no-script",
        "engine": {
            "name": "multiwfn",
            "executable": dir.join("missing-multiwfn").to_string_lossy()
        },
        "molecule": {
            "source": {"kind": "file", "path": molden, "format": "molden"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "generic_run", "method": "multiwfn"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 2);
    assert_eq!(payload["status"], "error");
    assert!(payload["warnings"][0]
        .as_str()
        .unwrap()
        .contains("generic_run requires"));
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_resp2_mixes_gas_and_solvent_charge_tables() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_multiwfn_resp2_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let fake_multiwfn = dir.join("fake_multiwfn");
    fs::write(
        &fake_multiwfn,
        "#!/bin/sh\nbase=$(basename \"$1\")\nstem=${base%.*}\nif [ \"$stem\" = gas ]; then\n  cat > gas.chg <<'EOF'\nC 0.0 0.0 0.0 -0.2000000000\nH 0.0 0.0 1.0 0.2000000000\nEOF\nelse\n  cat > solvent.chg <<'EOF'\nC 0.0 0.0 0.0 -0.6000000000\nH 0.0 0.0 1.0 0.6000000000\nEOF\nfi\nprintf 'ok\\n'\n",
    )
    .expect("write fake multiwfn");
    let mut perms = fs::metadata(&fake_multiwfn)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&fake_multiwfn, perms).expect("chmod");
    let gas = dir.join("gas.fchk");
    let solvent = dir.join("solvent.fchk");
    fs::write(&gas, "gas").expect("write gas");
    fs::write(&solvent, "solvent").expect("write solvent");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-resp2",
        "engine": {
            "name": "multiwfn",
            "executable": fake_multiwfn,
            "settings": {
                "gas_input_file": gas,
                "solvent_input_file": solvent,
                "delta": 0.25
            }
        },
        "molecule": {
            "source": {"kind": "engine_input", "format": "multiwfn"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "resp_fit", "method": "multiwfn", "charge_model": "resp2"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 0, "{payload}");
    assert_eq!(payload["status"], "ok");
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(dir.join("work/charge_manifest.json")).expect("manifest"),
    )
    .expect("manifest json");
    assert_eq!(manifest["model"], "resp2");
    assert_eq!(manifest["provenance"]["delta"], 0.25);
    assert!((manifest["atom_charges_e"][0].as_f64().unwrap() + 0.3).abs() < 1e-12);
    assert!((manifest["atom_charges_e"][1].as_f64().unwrap() - 0.3).abs() < 1e-12);
    assert!(dir.join("work/RESP2.chg").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_charge_manifest_projects_fake_caps_and_region_sets() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_multiwfn_projection_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let fake_multiwfn = dir.join("fake_multiwfn");
    fs::write(
        &fake_multiwfn,
        "#!/bin/sh\ncat > molecule.chg <<'EOF'\nX 0 0 0 0.1000000000\nC 1 0 0 -0.2000000000\nC 2 0 0 0.0000000000\nC 3 0 0 -0.3000000000\nX 4 0 0 0.4000000000\nEOF\n",
    )
    .expect("write fake multiwfn");
    let mut perms = fs::metadata(&fake_multiwfn)
        .expect("metadata")
        .permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&fake_multiwfn, perms).expect("chmod");
    let input = dir.join("molecule.fchk");
    fs::write(&input, "stub").expect("write input");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-projection",
        "engine": {
            "name": "multiwfn",
            "executable": fake_multiwfn,
            "settings": {
                "menu_script": "q\n",
                "expected_outputs": ["molecule.chg"],
                "charge_projection": {
                    "policy": "fake_caps_redistributed_region_sets",
                    "redistribution": [
                        {"source_atom": 0, "target_atoms": [1]},
                        {"source_atom": 4, "target_atoms": [3]}
                    ],
                    "deployable_sets": [
                        {"name": "head", "role": "head_cap_plus_first_repeat", "atom_indices": [1, 2]},
                        {"name": "mid", "role": "interior_repeat", "atom_indices": [2]},
                        {"name": "tail", "role": "last_repeat_plus_tail_cap", "atom_indices": [2, 3]}
                    ]
                }
            }
        },
        "molecule": {
            "source": {"kind": "file", "path": input, "format": "fchk"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "generic_run", "method": "multiwfn", "charge_model": "resp"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 0, "{payload}");
    let manifest: Value = serde_json::from_str(
        &fs::read_to_string(dir.join("work/charge_manifest.json")).expect("manifest"),
    )
    .expect("manifest json");
    assert_eq!(manifest["atom_labels"][0], "X");
    assert_eq!(
        manifest["projection"]["policy"],
        "fake_caps_redistributed_region_sets"
    );
    assert_eq!(manifest["projection"]["projected_charges_e"][0], 0.0);
    assert!(
        (manifest["projection"]["projected_charges_e"][1]
            .as_f64()
            .unwrap()
            + 0.1)
            .abs()
            < 1e-12
    );
    assert!(
        (manifest["projection"]["projected_charges_e"][3]
            .as_f64()
            .unwrap()
            - 0.1)
            .abs()
            < 1e-12
    );
    assert_eq!(manifest["projection"]["deployable_sets"][0]["name"], "head");
    assert_eq!(manifest["projection"]["deployable_sets"][1]["name"], "mid");
    assert_eq!(manifest["projection"]["deployable_sets"][2]["name"], "tail");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_esp_writes_current_cube_recipe() {
    let dir =
        std::env::temp_dir().join(format!("warp_qm_multiwfn_esp_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-esp-missing",
        "engine": {
            "name": "multiwfn",
            "executable": dir.join("missing-multiwfn").to_string_lossy(),
            "settings": {"grid_quality": "medium"}
        },
        "molecule": {
            "source": {"kind": "file", "path": molden, "format": "molden"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "esp", "method": "multiwfn"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 4);
    assert_eq!(payload["status"], "error");
    let script = fs::read_to_string(dir.join("work/multiwfn_input.txt")).expect("script");
    assert_eq!(script, "5\n12\n2\n2\n0\nq\n");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_population_hirshfeld_writes_verified_charge_recipe() {
    let dir =
        std::env::temp_dir().join(format!("warp_qm_multiwfn_pop_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-pop-missing",
        "engine": {
            "name": "multiwfn",
            "executable": dir.join("missing-multiwfn").to_string_lossy()
        },
        "molecule": {
            "source": {"kind": "file", "path": molden, "format": "molden"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "population", "method": "multiwfn", "charge_model": "hirshfeld"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 4);
    assert_eq!(payload["status"], "error");
    let script = fs::read_to_string(dir.join("work/multiwfn_input.txt")).expect("script");
    assert_eq!(script, "7\n1\n1\ny\n0\nq\n");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_orbital_cube_requires_orbital_number() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_multiwfn_orb_req_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-orbital-missing-setting",
        "engine": {"name": "multiwfn"},
        "molecule": {
            "source": {"kind": "file", "path": molden, "format": "molden"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "orbital_cube", "method": "multiwfn"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 2);
    assert_eq!(
        payload["warnings"][0],
        "orbital_cube requires engine.settings.orbital or engine.settings.orbital_number"
    );
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn multiwfn_orbital_cube_writes_current_recipe() {
    let dir =
        std::env::temp_dir().join(format!("warp_qm_multiwfn_orb_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-orbital-missing",
        "engine": {
            "name": "multiwfn",
            "executable": dir.join("missing-multiwfn").to_string_lossy(),
            "settings": {"orbital": 1, "grid_quality": "low"}
        },
        "molecule": {
            "source": {"kind": "file", "path": molden, "format": "molden"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "orbital_cube", "method": "multiwfn"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 4);
    assert_eq!(payload["status"], "error");
    let script = fs::read_to_string(dir.join("work/multiwfn_input.txt")).expect("script");
    assert_eq!(script, "5\n4\n1\n1\n2\n0\nq\n");
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn orca_binding_energy_requires_fragments_before_execution() {
    let dir =
        std::env::temp_dir().join(format!("warp_qm_orca_binding_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let xyz = dir.join("h2.xyz");
    fs::write(&xyz, "2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n").expect("write xyz");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "binding-no-fragments",
        "engine": {
            "name": "orca",
            "executable": dir.join("missing-orca").to_string_lossy()
        },
        "molecule": {
            "source": {"kind": "file", "path": xyz, "format": "xyz"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "binding_energy", "method": "HF", "basis": "STO-3G"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 2);
    assert_eq!(payload["status"], "error");
    assert_eq!(
        payload["warnings"][0],
        "binding_energy requires engine.settings.fragments as a non-empty array"
    );
    assert!(!dir.join("work/complex").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn orca_binding_energy_computes_delta_from_subjobs() {
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_orca_binding_compute_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let fake_orca = dir.join("fake-orca");
    fs::write(
        &fake_orca,
        "#!/bin/sh\nbase=${1%.inp}\ncase \"$PWD\" in\n  *frag_a*) e=-0.250000 ;;\n  *frag_b*) e=-0.500000 ;;\n  *) e=-1.000000 ;;\nesac\ntouch \"$base.gbw\"\nprintf 'FINAL SINGLE POINT ENERGY     %s\\nORCA TERMINATED NORMALLY\\n' \"$e\"\n",
    )
    .expect("write fake orca");
    let mut permissions = fs::metadata(&fake_orca).expect("metadata").permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_orca, permissions).expect("chmod fake orca");
    let complex = dir.join("complex.xyz");
    let frag_a = dir.join("frag_a.xyz");
    let frag_b = dir.join("frag_b.xyz");
    fs::write(&complex, "2\ncomplex\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n").expect("complex");
    fs::write(&frag_a, "1\nfrag_a\nH 0.0 0.0 0.0\n").expect("frag_a");
    fs::write(&frag_b, "1\nfrag_b\nH 0.0 0.0 0.0\n").expect("frag_b");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "binding-compute",
        "engine": {
            "name": "orca",
            "executable": fake_orca,
            "settings": {
                "fragments": [
                    {"label": "frag_a", "path": frag_a, "format": "xyz", "charge": 0, "multiplicity": 1},
                    {"label": "frag_b", "path": frag_b, "format": "xyz", "charge": 0, "multiplicity": 1}
                ]
            }
        },
        "molecule": {
            "source": {"kind": "file", "path": complex, "format": "xyz"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "binding_energy", "method": "HF", "basis": "STO-3G"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 0, "{payload}");
    assert_eq!(payload["status"], "ok");
    assert_eq!(
        payload["properties"]["binding_energy"]["delta_hartree"],
        -0.25
    );
    assert_eq!(payload["summary"]["energy_hartree"], -0.25);
    assert!(dir.join("work/complex/complex.out").exists());
    assert!(dir.join("work/frag_a/frag_a.out").exists());
    assert!(dir.join("work/frag_b/frag_b.out").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires local ORCA installation; set WARP_QM_TEST_ORCA"]
fn orca_real_h2_single_point_smoke() {
    let orca = match std::env::var("WARP_QM_TEST_ORCA") {
        Ok(path) => path,
        Err(_) => return,
    };
    let dir = std::env::temp_dir().join(format!("warp_qm_orca_real_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let xyz = dir.join("h2.xyz");
    fs::write(&xyz, "2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n").expect("write xyz");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "orca-real-h2",
        "engine": {
            "name": "orca",
            "executable": orca,
            "settings": {"basename": "h2", "keywords": ["MiniPrint"]}
        },
        "molecule": {
            "source": {"kind": "file", "path": xyz, "format": "xyz"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "single_point", "method": "HF", "basis": "STO-3G"},
        "runtime": {"work_dir": dir.join("work"), "threads": 1, "memory_mb": 1000},
        "output": {"out_dir": dir.join("work"), "write_json": true}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 0, "{payload}");
    assert_eq!(payload["status"], "ok");
    assert!(payload["summary"]["energy_hartree"].as_f64().is_some());
    assert!(dir.join("work/h2.property.json").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires local Multiwfn installation; set WARP_QM_TEST_MULTIWFN and WARP_QM_TEST_MULTIWFN_LIB_DIR"]
fn multiwfn_real_resp_smoke() {
    let multiwfn = match std::env::var("WARP_QM_TEST_MULTIWFN") {
        Ok(path) => path,
        Err(_) => return,
    };
    let lib_dir = match std::env::var("WARP_QM_TEST_MULTIWFN_LIB_DIR") {
        Ok(path) => path,
        Err(_) => return,
    };
    let dir =
        std::env::temp_dir().join(format!("warp_qm_multiwfn_real_test_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    let request = serde_json::json!({
        "schema_version": "warp-qm.agent.v1",
        "request_id": "multiwfn-real-resp",
        "engine": {
            "name": "multiwfn",
            "executable": multiwfn,
            "settings": {"lib_dir": lib_dir}
        },
        "molecule": {
            "source": {"kind": "file", "path": molden, "format": "molden"},
            "charge": 0,
            "multiplicity": 1
        },
        "task": {"kind": "resp_fit", "method": "multiwfn", "charge_model": "resp"},
        "runtime": {"work_dir": dir.join("work")}
    });
    let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
    assert_eq!(code, 0, "{payload}");
    assert_eq!(payload["status"], "ok");
    assert!(dir.join("work/h2.chg").exists());
    assert!(dir.join("work/charge_manifest.json").exists());
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires local ORCA installation; set WARP_QM_TEST_ORCA"]
fn orca_real_promoted_lanes_smoke() {
    let orca = match std::env::var("WARP_QM_TEST_ORCA") {
        Ok(path) => path,
        Err(_) => return,
    };
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_orca_promoted_real_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let xyz = dir.join("h2.xyz");
    fs::write(&xyz, "2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n").expect("write xyz");
    for task in [
        "optimize",
        "frequency",
        "nmr_shielding",
        "charges",
        "orca_molden_export",
    ] {
        let work = dir.join(task);
        let request = serde_json::json!({
            "schema_version": "warp-qm.agent.v1",
            "request_id": format!("orca-real-{task}"),
            "engine": {
                "name": "orca",
                "executable": orca,
                "settings": {"basename": "h2", "keywords": ["MiniPrint"], "export_molden": task == "orca_molden_export"}
            },
            "molecule": {
                "source": {"kind": "file", "path": xyz, "format": "xyz"},
                "charge": 0,
                "multiplicity": 1
            },
            "task": {"kind": task, "method": "HF", "basis": "STO-3G", "charge_model": "hirshfeld"},
            "runtime": {"work_dir": work, "threads": 1, "memory_mb": 1000},
            "output": {"out_dir": work, "write_json": true}
        });
        let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
        assert_eq!(code, 0, "{task}: {payload}");
        assert_eq!(payload["status"], "ok", "{task}: {payload}");
        if task == "frequency" {
            assert_eq!(
                payload["properties"]["inspection"]["imaginary_frequency_count"],
                0
            );
            assert!(
                payload["properties"]["inspection"]["frequencies_cm1"]
                    .as_array()
                    .unwrap()
                    .len()
                    > 0
            );
        }
        if task == "nmr_shielding" {
            assert!(
                payload["properties"]["inspection"]["nmr_shielding"]
                    .as_array()
                    .unwrap()
                    .len()
                    > 0
            );
        }
        if task == "charges" {
            assert!(dir.join(task).join("charge_manifest.json").exists());
        }
        if task == "orca_molden_export" {
            assert!(dir.join(task).join("h2.molden.input").exists());
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

#[test]
#[ignore = "requires local Multiwfn installation; set WARP_QM_TEST_MULTIWFN and WARP_QM_TEST_MULTIWFN_LIB_DIR"]
fn multiwfn_real_promoted_lanes_smoke() {
    let multiwfn = match std::env::var("WARP_QM_TEST_MULTIWFN") {
        Ok(path) => path,
        Err(_) => return,
    };
    let lib_dir = match std::env::var("WARP_QM_TEST_MULTIWFN_LIB_DIR") {
        Ok(path) => path,
        Err(_) => return,
    };
    let dir = std::env::temp_dir().join(format!(
        "warp_qm_multiwfn_promoted_real_test_{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create dir");
    let molden = dir.join("h2.molden.input");
    fs::write(&molden, h2_molden()).expect("write molden");
    for task in [
        "esp",
        "electron_density_cube",
        "elf_cube",
        "lol_cube",
        "laplacian_cube",
        "population",
        "orbital_cube",
    ] {
        let work = dir.join(task);
        let request = serde_json::json!({
            "schema_version": "warp-qm.agent.v1",
            "request_id": format!("multiwfn-real-{task}"),
            "engine": {
                "name": "multiwfn",
                "executable": multiwfn,
                "settings": {"lib_dir": lib_dir, "grid_quality": "low", "orbital": 1}
            },
            "molecule": {
                "source": {"kind": "file", "path": molden, "format": "molden"},
                "charge": 0,
                "multiplicity": 1
            },
            "task": {"kind": task, "method": "multiwfn", "charge_model": "hirshfeld"},
            "runtime": {"work_dir": work}
        });
        let (code, payload) = warp_qm::run_request_json(&request.to_string(), false);
        assert_eq!(code, 0, "{task}: {payload}");
        assert_eq!(payload["status"], "ok", "{task}: {payload}");
        if task.ends_with("_cube") || task == "esp" {
            assert!(work.join("cube_manifest.json").exists(), "{task}");
        }
        if task == "population" {
            assert!(work.join("charge_manifest.json").exists());
        }
    }
    let _ = fs::remove_dir_all(&dir);
}

fn h2_molden() -> &'static str {
    "[Molden Format]\n[Title]\n Molden file created by orca_2mkl for BaseName=h2\n\n[Atoms] AU\nH    1   1          0.0000000000         0.0000000000         0.0000000000 \nH    2   1          0.0000000000         0.0000000000         1.3983973580 \n[GTO]\n  1 0\ns   3 1.0 \n        3.4252509100         0.2769343610\n        0.6239137300         0.2678388518\n        0.1688554000         0.0834736696\n\n  2 0\ns   3 1.0 \n        3.4252509100         0.2769343610\n        0.6239137300         0.2678388518\n        0.1688554000         0.0834736696\n\n[5D]\n[7F]\n[9G]\n[MO]\n Sym=     1a\n Ene= -5.78553854175481E-01\n Spin= Alpha\n Occup= 2.000000\n  1      -0.548842276567\n  2      -0.548842276567\n Sym=     1a\n Ene= 6.71143477795428E-01\n Spin= Alpha\n Occup= 0.000000\n  1      -1.212451904244\n  2       1.212451904244\n"
}
