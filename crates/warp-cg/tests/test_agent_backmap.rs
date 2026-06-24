use std::process::Command;
use warp_cg::backmap::{BackmapArtifact, BackmapPlan, BeadSite, Link, ResidueTemplate};
use warp_pep::builder;
use warp_pep::residue::Residue;

fn build_template_from_residue(
    res: &Residue,
    bead_index: usize,
) -> (ResidueTemplate, Option<usize>, Option<usize>) {
    let mut ref_coords = Vec::new();
    let mut atom_names = Vec::new();
    let mut ca_idx = None;
    let mut c_idx = None;
    let mut n_idx = None;

    for (i, atom) in res.atoms.iter().enumerate() {
        ref_coords.push([atom.coord.x, atom.coord.y, atom.coord.z]);
        atom_names.push(atom.name.clone());
        match atom.name.as_str() {
            "CA" => ca_idx = Some(i),
            "C" => c_idx = Some(i),
            "N" => n_idx = Some(i),
            _ => {}
        }
    }

    let template = ResidueTemplate {
        name: format!("{:?}", res.name),
        atom_names,
        elements: Vec::new(),
        residue_names: Vec::new(),
        residue_ids: Vec::new(),
        chains: Vec::new(),
        source_atom_indices: Vec::new(),
        reference_coords: ref_coords,
        bead_sites: vec![BeadSite {
            target_bead_index: bead_index,
            atom_indices: vec![ca_idx.expect("missing CA")],
            weights: None,
        }],
        bonds: Vec::new(),
        chirality: Vec::new(),
    };

    (template, c_idx, n_idx)
}

#[test]
fn test_peptide_backmap_agent_workflow() {
    let structure = builder::make_extended_structure("AA").expect("failed to build peptide");
    let chain = &structure.chains[0];

    let mut templates = Vec::new();
    let mut links = Vec::new();
    let mut c_indices = Vec::new();
    let mut n_indices = Vec::new();
    let mut cg_coords = Vec::new();

    for (res_idx, res) in chain.residues.iter().enumerate() {
        let ca = res.atom_coord("CA").expect("missing CA");
        cg_coords.push([ca.x, ca.y, ca.z]);

        let (template, c_idx, n_idx) = build_template_from_residue(res, res_idx);
        c_indices.push(c_idx);
        n_indices.push(n_idx);
        templates.push(template);
    }

    // Links: residue i's C → residue i+1's N.
    for i in 0..templates.len() - 1 {
        if let (Some(c_idx), Some(n_idx)) = (c_indices[i], n_indices[i + 1]) {
            links.push(Link {
                from_template: i,
                from_atom: c_idx,
                to_template: i + 1,
                to_atom: n_idx,
                target_distance: None,
            });
        }
    }

    let plan = BackmapPlan {
        templates,
        links,
        fudge_factor: 1.0,
    };

    let aa_coords = plan.execute(&cg_coords).expect("backmap failed");
    let ref_len = plan.templates[0].reference_coords.len();

    assert_eq!(aa_coords.len(), ref_len * 2);

    // CA of first residue (no prior neighbour) should exactly match its CG bead.
    let ca1_local = plan.templates[0]
        .atom_names
        .iter()
        .position(|n| n == "CA")
        .unwrap();
    let rebuilt_ca1 = aa_coords[ca1_local];
    let d1 = ((rebuilt_ca1[0] - cg_coords[0][0]).powi(2)
        + (rebuilt_ca1[1] - cg_coords[0][1]).powi(2)
        + (rebuilt_ca1[2] - cg_coords[0][2]).powi(2))
    .sqrt();
    assert!(d1 < 1e-5, "first CA deviates {:.6}", d1);

    // CA of second residue (pulled by connection) should be NEAR its CG bead.
    let ca2_local = plan.templates[1]
        .atom_names
        .iter()
        .position(|n| n == "CA")
        .unwrap();
    let rebuilt_ca2 = aa_coords[ref_len + ca2_local];
    let d2 = ((rebuilt_ca2[0] - cg_coords[1][0]).powi(2)
        + (rebuilt_ca2[1] - cg_coords[1][1]).powi(2)
        + (rebuilt_ca2[2] - cg_coords[1][2]).powi(2))
    .sqrt();
    assert!(
        d2 < 0.3,
        "second CA deviates {:.6} from CG (pulled by connection)",
        d2
    );

    // Backbone bond C1–N2 should be close to peptide bond length (~1.3-1.5 Å).
    let c1_local = plan.templates[0]
        .atom_names
        .iter()
        .position(|n| n == "C")
        .unwrap();
    let n2_local = plan.templates[1]
        .atom_names
        .iter()
        .position(|n| n == "N")
        .unwrap();
    let c1 = aa_coords[c1_local];
    let n2 = aa_coords[ref_len + n2_local];
    let c1n2 = ((c1[0] - n2[0]).powi(2) + (c1[1] - n2[1]).powi(2) + (c1[2] - n2[2]).powi(2)).sqrt();
    assert!(
        c1n2 < 1.6,
        "C1–N2 distance {:.3} Å should be peptide-bond-like",
        c1n2
    );
}

#[test]
fn test_long_peptide_backmap_agent_workflow() {
    let sequence = "ACDEFGHIKLMNPQRSTVWY";
    let structure = builder::make_extended_structure(sequence).expect("failed to build peptide");
    let chain = &structure.chains[0];

    let mut templates = Vec::new();
    let mut links = Vec::new();
    let mut c_indices = Vec::new();
    let mut n_indices = Vec::new();
    let mut cg_coords = Vec::new();
    let mut total_expected_atoms = 0;

    for (res_idx, res) in chain.residues.iter().enumerate() {
        let ca = res.atom_coord("CA").expect("missing CA");
        // Exact CG positions (no displacement) — tests that the connection
        // constraints don't break perfect alignment.
        cg_coords.push([ca.x, ca.y, ca.z]);

        let (template, c_idx, n_idx) = build_template_from_residue(res, res_idx);
        total_expected_atoms += template.reference_coords.len();
        c_indices.push(c_idx);
        n_indices.push(n_idx);
        templates.push(template);
    }

    for i in 0..templates.len() - 1 {
        if let (Some(c_idx), Some(n_idx)) = (c_indices[i], n_indices[i + 1]) {
            links.push(Link {
                from_template: i,
                from_atom: c_idx,
                to_template: i + 1,
                to_atom: n_idx,
                target_distance: None,
            });
        }
    }

    let plan = BackmapPlan {
        templates,
        links,
        fudge_factor: 1.0,
    };

    let aa_coords = plan.execute(&cg_coords).expect("backmap failed");
    assert_eq!(aa_coords.len(), total_expected_atoms);

    let mut atom_offset = 0;
    for (res_idx, template) in plan.templates.iter().enumerate() {
        let ca_local = template.atom_names.iter().position(|n| n == "CA").unwrap();
        let rebuilt_ca = aa_coords[atom_offset + ca_local];
        let target_cg = cg_coords[res_idx];
        let ca_dist = ((rebuilt_ca[0] - target_cg[0]).powi(2)
            + (rebuilt_ca[1] - target_cg[1]).powi(2)
            + (rebuilt_ca[2] - target_cg[2]).powi(2))
        .sqrt();
        // With zero displacement, CG and template geometry are consistent,
        // so CA should map exactly.
        assert!(
            ca_dist < 1e-5,
            "Residue {} CA deviates {:.6} from CG bead",
            res_idx,
            ca_dist
        );

        // Check backbone bond from this residue's C to next residue's N.
        if res_idx + 1 < plan.templates.len() {
            let c_local = template.atom_names.iter().position(|n| n == "C").unwrap();
            let next_n_local = plan.templates[res_idx + 1]
                .atom_names
                .iter()
                .position(|n| n == "N")
                .unwrap();
            let next_offset = atom_offset + template.reference_coords.len();
            let c_pos = aa_coords[atom_offset + c_local];
            let n_pos = aa_coords[next_offset + next_n_local];
            let c_n_dist = ((c_pos[0] - n_pos[0]).powi(2)
                + (c_pos[1] - n_pos[1]).powi(2)
                + (c_pos[2] - n_pos[2]).powi(2))
            .sqrt();
            assert!(
                c_n_dist < 1.7,
                "Residue {}-{} C-N distance {:.3} Å too large",
                res_idx,
                res_idx + 1,
                c_n_dist
            );
        }

        atom_offset += template.reference_coords.len();
    }
}

#[test]
fn test_backmap_cli_consumes_generated_contract() {
    let tmp = tempfile::tempdir().unwrap();
    let plan_path = tmp.path().join("plan.json");
    let request_path = tmp.path().join("request.json");
    let output_path = tmp.path().join("aa.json");
    let plan = BackmapPlan {
        templates: vec![ResidueTemplate {
            name: "MOL".into(),
            atom_names: vec!["A".into(), "B".into()],
            elements: vec!["C".into(), "C".into()],
            residue_names: vec!["MOL".into(), "MOL".into()],
            residue_ids: vec![1, 1],
            chains: vec!["A".into(), "A".into()],
            source_atom_indices: vec![0, 1],
            reference_coords: vec![[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            bead_sites: vec![BeadSite {
                target_bead_index: 0,
                atom_indices: vec![0, 1],
                weights: None,
            }],
            bonds: Vec::new(),
            chirality: Vec::new(),
        }],
        links: vec![],
        fudge_factor: 1.0,
    };
    std::fs::write(
        &plan_path,
        serde_json::to_vec_pretty(&BackmapArtifact::new(plan)).unwrap(),
    )
    .unwrap();
    std::fs::write(
        &request_path,
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema_version": "warp-cg.backmap.v1",
            "plan_path": plan_path,
            "frames": [[[5.0, 6.0, 7.0]]],
            "output": {
                "out_dir": tmp.path(),
                "prefix": "aa",
                "formats": ["json"]
            }
        }))
        .unwrap(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_warp-cg"))
        .args(["backmap", "run", request_path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["schema_version"], "warp-cg.backmap-result.v1");
    assert_eq!(result["atom_count"], 2);
    assert!(output_path.is_file());
}
