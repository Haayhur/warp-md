use super::{LipidTemplate, TemplateBead};

pub(super) fn triglyceride_template(name: &str) -> Option<LipidTemplate> {
    let tail_code = match name {
        "TO" => "CDCC",
        "TL" => "CDDC",
        "TLN" => "CDDD",
        "TS" | "TP" => "CCCC",
        "TM" | "TLA" => "CCC",
        _ => return None,
    };
    Some(ltf_triglyceride_template(name, tail_code))
}

pub(super) fn triglyceride_lipid_names() -> &'static [&'static str] {
    &["TO", "TL", "TLN", "TS", "TP", "TM", "TLA"]
}

fn ltf_triglyceride_template(name: &str, tail_code: &str) -> LipidTemplate {
    let mut beads = Vec::new();
    append_linker_tail(&mut beads, "GL1", tail_code, 'A', 0.0);
    append_linker_tail(&mut beads, "GL2", tail_code, 'B', 1.0);
    append_linker_tail(&mut beads, "GL3", tail_code, 'C', 2.0);
    LipidTemplate {
        name: name.to_string(),
        source: "warp-cg.lipid-template.martini-ltf-triglyceride.v1",
        radius_angstrom: 5.0,
        net_charge_e: 0.0,
        beads,
    }
}

fn append_linker_tail(
    beads: &mut Vec<TemplateBead>,
    linker: &str,
    tail_code: &str,
    chain: char,
    x: f32,
) {
    beads.push(TemplateBead {
        name: linker.to_string(),
        offset_angstrom: [x, 0.0, 0.0],
        charge_e: 0.0,
    });
    for (idx, code) in tail_code.chars().enumerate() {
        beads.push(TemplateBead {
            name: format!("{}{}{}", code, idx + 1, chain),
            offset_angstrom: [x, 0.0, -0.47 * (idx as f32 + 1.0)],
            charge_e: 0.0,
        });
    }
}
