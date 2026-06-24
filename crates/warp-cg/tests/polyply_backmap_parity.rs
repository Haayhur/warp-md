use warp_cg::backmap::{BackmapPlan, BeadSite, Link, ResidueTemplate};

fn template(bead_idx: usize, source_offset: usize) -> ResidueTemplate {
    ResidueTemplate {
        name: "test".into(),
        atom_names: vec!["A".into(), "B".into(), "C".into()],
        elements: vec!["C".into(), "C".into(), "C".into()],
        residue_names: vec!["test".into(); 3],
        residue_ids: vec![bead_idx as i32 + 1; 3],
        chains: vec!["A".into(); 3],
        source_atom_indices: vec![source_offset, source_offset + 1, source_offset + 2],
        reference_coords: vec![[0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
        bead_sites: vec![BeadSite {
            target_bead_index: bead_idx,
            atom_indices: vec![1],
            weights: None,
        }],
        bonds: vec![],
        chirality: vec![],
    }
}

#[test]
fn warp_cg_meets_or_improves_polyply_seed42_invariants() {
    let reference: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/polyply_backmap_seed42.json")).unwrap();
    let polyply_link_distance = reference["metrics"]["link_distance"].as_f64().unwrap();
    let plan = BackmapPlan {
        templates: vec![template(0, 0), template(1, 3)],
        links: vec![Link {
            from_template: 0,
            from_atom: 2,
            to_template: 1,
            to_atom: 0,
            target_distance: Some(polyply_link_distance),
        }],
        fudge_factor: 0.4,
    };
    let frame = plan
        .execute_frame(&[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        .unwrap();

    assert!(frame.diagnostics.finite);
    assert!(frame.diagnostics.mapped_bead_max_error <= 1.0e-12);
    assert!(
        frame.diagnostics.link_bond_max_error <= 1.0e-6,
        "warp-cg link error {} exceeds parity tolerance",
        frame.diagnostics.link_bond_max_error
    );
    assert_eq!(frame.diagnostics.chirality_inversion_count, 0);
}
