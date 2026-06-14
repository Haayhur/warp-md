use super::{LipidTemplate, TemplateBead};

pub(super) fn ltf_diglyceride_template(name: &str) -> Option<LipidTemplate> {
    let (tail_a, tail_b) = ltf_diglyceride_tailcodes(name)?;
    Some(build_diglyceride_template(name, tail_a, tail_b))
}

pub(super) fn ltf_diglyceride_lipid_names() -> &'static [&'static str] {
    &[
        "DTDG", "DJDG", "DUDG", "DMDG", "MPDG", "MSDG", "PMDG", "SMDG", "DPDG", "DSDG", "PSDG",
        "DKDG", "DBDG", "DXDG", "DCDG", "DRDG", "DYDG", "DODG", "DVDG", "YODG", "OEDG", "DGDG",
        "DEDG", "DNDG", "DLDG", "DFDG", "DADG", "DDDG", "PYDG", "PODG", "SODG", "PEDG", "PLDG",
        "SLDG", "PIDG", "PFDG", "PQDG", "PADG", "SADG", "PDDG", "SDDG", "OLDG", "ODDG", "LFDG",
    ]
}

fn build_diglyceride_template(name: &str, tail_a: &str, tail_b: &str) -> LipidTemplate {
    let mut beads = vec![
        bead("OH", [0.0, 0.0, 2.0]),
        bead("GL1", [0.0, 0.0, 1.0]),
        bead("GL2", [0.5, 0.0, 1.0]),
    ];
    append_tail(&mut beads, tail_a, 'A', 0.0);
    append_tail(&mut beads, tail_b, 'B', 1.0);
    LipidTemplate {
        name: name.to_string(),
        source: "warp-cg.lipid-template.martini-ltf-diglyceride.v1",
        radius_angstrom: 4.0,
        net_charge_e: 0.0,
        beads,
    }
}

fn append_tail(beads: &mut Vec<TemplateBead>, tail_code: &str, chain: char, x: f32) {
    for (idx, code) in tail_code.chars().enumerate() {
        beads.push(TemplateBead {
            name: format!("{}{}{}", code, idx + 1, chain),
            offset_angstrom: [x, 0.0, -0.47 * idx as f32],
            charge_e: 0.0,
        });
    }
}

fn bead(name: &str, offset_angstrom: [f32; 3]) -> TemplateBead {
    TemplateBead {
        name: name.to_string(),
        offset_angstrom,
        charge_e: 0.0,
    }
}

fn ltf_diglyceride_tailcodes(name: &str) -> Option<(&'static str, &'static str)> {
    match name.strip_suffix("DG")? {
        "DT" | "DJ" => Some(("CC", "CC")),
        "DU" | "DM" => Some(("CCC", "CCC")),
        "MP" | "MS" => Some(("CCCC", "CCC")),
        "PM" | "SM" => Some(("CCC", "CCCC")),
        "DP" | "DS" | "PS" => Some(("CCCC", "CCCC")),
        "DK" | "DB" => Some(("CCCCC", "CCCCC")),
        "DX" => Some(("CCCCC", "CCCCCC")),
        "DC" => Some(("CCCCCC", "CCCCCC")),
        "DR" => Some(("CDC", "CDC")),
        "DY" | "DV" => Some(("CCDC", "CCDC")),
        "DO" => Some(("CDCC", "CDCC")),
        "YO" => Some(("CDCC", "CCDC")),
        "OE" => Some(("CCDCC", "CDCC")),
        "DG" | "DE" => Some(("CCDCC", "CCDCC")),
        "DN" => Some(("CCCDCC", "CCCDCC")),
        "DL" => Some(("CDDC", "CDDC")),
        "DF" => Some(("CDDD", "CDDD")),
        "DA" => Some(("CDDDC", "CDDDC")),
        "DD" => Some(("DDDDD", "DDDDD")),
        "PY" => Some(("CCDC", "CCCC")),
        "PO" | "SO" => Some(("CDCC", "CCCC")),
        "PE" => Some(("CCDCC", "CCCC")),
        "PL" | "SL" => Some(("CDDC", "CCCC")),
        "PI" => Some(("CCDDC", "CCCC")),
        "PF" => Some(("CDDD", "CCCC")),
        "PQ" | "PA" | "SA" => Some(("CDDDC", "CCCC")),
        "PD" | "SD" => Some(("DDDDD", "CCCC")),
        "OL" => Some(("CDDC", "CDCC")),
        "OD" => Some(("DDDDD", "CDCC")),
        "LF" => Some(("CDDD", "CDDC")),
        _ => None,
    }
}
