use super::{LipidTemplate, TemplateBead};

pub(super) fn ltf_complex_lipid_template(name: &str) -> Option<LipidTemplate> {
    match name {
        "TMCL" => Some(build_complex_template(
            "TMCL",
            "warp-cg.lipid-template.martini-ltf-cardiolipin.v1",
            0.0,
            &[
                "GLC", "PO41", "GL11", "GL21", "C1A1", "C2A1", "C3A1", "C1B1", "C2B1", "C3B1",
                "PO42", "GL12", "GL22", "C1A2", "C2A2", "C3A2", "C1B2", "C2B2", "C3C2",
            ],
        )),
        "TOCL" => Some(build_complex_template(
            "TOCL",
            "warp-cg.lipid-template.martini-ltf-cardiolipin.v1",
            0.0,
            &[
                "GLC", "PO41", "GL11", "GL21", "C1A1", "C2A1", "C3A1", "C4A1", "C1B1", "C2B1",
                "C3B1", "C4B1", "PO42", "GL12", "GL22", "C1A2", "C2A2", "C3A2", "C4A2", "C1B2",
                "C2B2", "C3C2", "C4B2",
            ],
        )),
        "DO2B" => Some(build_complex_template(
            "DO2B",
            "warp-cg.lipid-template.martini-ltf-bmp.v1",
            -1.0,
            &[
                "PO4", "OH1", "GL1", "C1A", "D2A", "C3A", "C4A", "OH2", "GL2", "C1B", "D2B", "C3B",
                "C4B",
            ],
        )),
        "DO3B" => Some(build_complex_template(
            "DO3B",
            "warp-cg.lipid-template.martini-ltf-bmp.v1",
            -1.0,
            &[
                "PO4", "GL1", "OH1", "C1A", "D2A", "C3A", "C4A", "GL2", "OH2", "C1B", "D2B", "C3B",
                "C4B",
            ],
        )),
        _ => None,
    }
}

pub(super) fn ltf_complex_lipid_names() -> &'static [&'static str] {
    &["TMCL", "TOCL", "DO2B", "DO3B"]
}

fn build_complex_template(
    name: &str,
    source: &'static str,
    net_charge_e: f32,
    bead_names: &[&str],
) -> LipidTemplate {
    let center = (bead_names.len().saturating_sub(1)) as f32 * 0.5;
    LipidTemplate {
        name: name.to_string(),
        source,
        radius_angstrom: 4.5,
        net_charge_e,
        beads: bead_names
            .iter()
            .enumerate()
            .map(|(idx, bead_name)| TemplateBead {
                name: (*bead_name).to_string(),
                offset_angstrom: [((idx % 4) as f32 - 1.5) * 1.5, 0.0, center - idx as f32],
                charge_e: complex_lipid_bead_charge(name, bead_name),
            })
            .collect(),
    }
}

fn complex_lipid_bead_charge(lipid_name: &str, bead_name: &str) -> f32 {
    match (lipid_name, bead_name) {
        ("DO2B" | "DO3B", "PO4") => -1.0,
        _ => 0.0,
    }
}
