use super::{LipidTemplate, TemplateBead};

const MONOACYL_X: [f32; 20] = [
    0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,
];
const MONOACYL_Y: [f32; 20] = [0.0; 20];
const MONOACYL_Z: [f32; 20] = [
    10.0, 9.0, 9.0, 8.0, 8.0, 7.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0,
];

pub(super) fn monoacyl_template(name: &str) -> Option<LipidTemplate> {
    match name {
        "GMO" => Some(monoacyl_lipid_template(
            "GMO",
            &[
                (6, "GL1"),
                (7, "GL2"),
                (8, "C1A"),
                (9, "C2A"),
                (10, "D3A"),
                (11, "C4A"),
                (12, "C5A"),
            ],
        )),
        "MO" => Some(ltf_monoglyceride_template(
            "MO",
            &["C1A", "D2A", "C3A", "C4A"],
        )),
        "ML" => Some(ltf_monoglyceride_template(
            "ML",
            &["C1A", "D2A", "D3A", "C4A"],
        )),
        "MLN" => Some(ltf_monoglyceride_template(
            "MLN",
            &["C1A", "D2A", "D3A", "D4A"],
        )),
        "MS" => Some(ltf_monoglyceride_template(
            "MS",
            &["C1A", "C2A", "C3A", "C4A"],
        )),
        "MP" => Some(ltf_monoglyceride_template(
            "MP",
            &["C1A", "C2A", "C3A", "C4A"],
        )),
        "MM" => Some(ltf_monoglyceride_template("MM", &["C1A", "C2A", "C3A"])),
        "MLA" => Some(ltf_monoglyceride_template("MLA", &["C1A", "C2A", "C3A"])),
        _ => None,
    }
}

pub(super) fn monoacyl_lipid_names() -> Vec<&'static str> {
    vec!["GMO", "MO", "ML", "MLN", "MS", "MP", "MM", "MLA"]
}

fn monoacyl_lipid_template(name: &str, beads: &[(usize, &str)]) -> LipidTemplate {
    LipidTemplate {
        name: name.to_string(),
        source: "warp-cg.lipid-template.martini-monoacyl.v1",
        radius_angstrom: 4.0,
        net_charge_e: 0.0,
        beads: beads
            .iter()
            .map(|(idx, bead)| TemplateBead {
                name: (*bead).to_string(),
                offset_angstrom: [MONOACYL_X[*idx], MONOACYL_Y[*idx], MONOACYL_Z[*idx] - 5.0],
                charge_e: 0.0,
            })
            .collect(),
    }
}

fn ltf_monoglyceride_template(name: &str, tail: &[&str]) -> LipidTemplate {
    let mut beads = Vec::with_capacity(tail.len() + 2);
    beads.push(TemplateBead {
        name: "DOH".to_string(),
        offset_angstrom: [0.0, 0.0, 0.3],
        charge_e: 0.0,
    });
    beads.push(TemplateBead {
        name: "GL1".to_string(),
        offset_angstrom: [0.0, 0.0, 0.0],
        charge_e: 0.0,
    });
    beads.extend(tail.iter().enumerate().map(|(idx, bead)| TemplateBead {
        name: (*bead).to_string(),
        offset_angstrom: [0.0, 0.0, -0.47 * (idx as f32 + 1.0)],
        charge_e: 0.0,
    }));
    LipidTemplate {
        name: name.to_string(),
        source: "warp-cg.lipid-template.martini-ltf-monoglyceride.v1",
        radius_angstrom: 4.0,
        net_charge_e: 0.0,
        beads,
    }
}
