use crate::build_lipids_diacyl::{
    diacyl_bead_names, diacyl_bead_names_from_tail, diacyl_bead_names_with_linker,
    expanded_diacyl_tail_names, known_diacyl_heads, known_plasmalogen_heads,
    known_plasmalogen_tails, linker_bead_names, parse_linked_diacyl_name, parse_plasmalogen_name,
    plasmalogen_tail_bead_names,
};
#[path = "build_lipids_ltf_complex.rs"]
mod build_lipids_ltf_complex;
#[path = "build_lipids_ltf_glycerides.rs"]
mod build_lipids_ltf_glycerides;
#[path = "build_lipids_ltf_single_chain.rs"]
mod build_lipids_ltf_single_chain;
#[path = "build_lipids_monoacyl.rs"]
mod build_lipids_monoacyl;
#[path = "build_lipids_sphingomyelin.rs"]
mod build_lipids_sphingomyelin;
#[path = "build_lipids_triglyceride.rs"]
mod build_lipids_triglyceride;
use build_lipids_ltf_complex::{ltf_complex_lipid_names, ltf_complex_lipid_template};
use build_lipids_ltf_glycerides::{ltf_diglyceride_lipid_names, ltf_diglyceride_template};
use build_lipids_ltf_single_chain::{ltf_single_chain_lipid_names, ltf_single_chain_template};
use build_lipids_monoacyl::{monoacyl_lipid_names, monoacyl_template};
use build_lipids_sphingomyelin::{
    generated_sphingomyelin_tail_names, generated_sphingomyelin_template,
};
use build_lipids_triglyceride::{triglyceride_lipid_names, triglyceride_template};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LipidTemplate {
    pub name: String,
    pub source: &'static str,
    pub radius_angstrom: f32,
    pub net_charge_e: f32,
    pub beads: Vec<TemplateBead>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TemplateBead {
    pub name: String,
    pub offset_angstrom: [f32; 3],
    pub charge_e: f32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinkerKind {
    Glycerol,
    Ether,
    Plasmalogen,
}

const DIACYL_X: [f32; 20] = [
    0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,
];
const DIACYL_Y: [f32; 20] = [0.0; 20];
const DIACYL_Z: [f32; 20] = [
    10.0, 9.0, 9.0, 8.0, 8.0, 7.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0,
];

const CHOL_X: [f32; 10] = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.0];
const CHOL_Y: [f32; 10] = [0.0; 10];
const CHOL_Z: [f32; 10] = [5.3, 4.5, 3.9, 3.3, 3.0, 2.6, 4.5, 2.6, 1.4, 0.0];

const INOSITOL_X: [f32; 22] = [
    0.5, 0.8, -0.3, 0.25, 0.0, 1.0, 0.5, -0.3, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
];
const INOSITOL_Y: [f32; 22] = [0.0; 22];
const INOSITOL_Z: [f32; 22] = [
    8.0, 9.1, 8.8, 8.5, 7.0, 10.0, 10.0, 10.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0,
    3.0, 2.0, 1.0, 0.0,
];

pub(crate) fn lookup_lipid_template(name: &str, force_field: &str) -> Option<LipidTemplate> {
    let normalized = normalize_name(name);
    let force_field = force_field.to_ascii_lowercase();
    if !(force_field.contains("martini") || force_field == "default") {
        return None;
    }
    let canonical = match normalized.as_str() {
        "DMPC" => "DLPC",
        "DMPE" => "DLPE",
        other => other,
    };
    match canonical {
        "CHOL" => Some(chol_template()),
        _ => ltf_complex_lipid_template(canonical).or_else(|| {
            ltf_single_chain_template(canonical).or_else(|| {
                monoacyl_template(canonical).or_else(|| {
                    ltf_diglyceride_template(canonical).or_else(|| {
                        triglyceride_template(canonical).or_else(|| {
                            ltf_sphingolipid_template(canonical).or_else(|| {
                                generated_sphingomyelin_template(canonical).or_else(|| {
                                    parse_linked_diacyl_name(canonical)
                                        .and_then(|(base_name, head_group, tail_group, linker)| {
                                            linked_diacyl_template(
                                                canonical.to_string(),
                                                base_name,
                                                head_group,
                                                tail_group,
                                                linker,
                                            )
                                        })
                                        .or_else(|| {
                                            parse_plasmalogen_name(canonical)
                                                .map(plasmalogen_template)
                                        })
                                })
                            })
                        })
                    })
                })
            })
        }),
    }
}

pub(crate) fn known_lipids() -> Vec<String> {
    let mut lipids = Vec::new();
    for tail in expanded_diacyl_tail_names() {
        for head in known_diacyl_heads() {
            lipids.push(format!("{tail}{head}"));
            if *head != "DG" {
                lipids.push(format!("{tail}{head}.GL"));
                lipids.push(format!("{tail}{head}.ET"));
            }
        }
    }
    for tail in known_plasmalogen_tails() {
        for head in known_plasmalogen_heads() {
            lipids.push(format!("{tail}PL{head}"));
        }
    }
    for name in ltf_complex_lipid_names() {
        lipids.push((*name).to_string());
    }
    for name in ltf_sphingolipid_names() {
        lipids.push((*name).to_string());
    }
    for tail in generated_sphingomyelin_tail_names() {
        lipids.push(format!("{tail}SM"));
    }
    for name in monoacyl_lipid_names() {
        lipids.push(name.to_string());
    }
    for name in ltf_diglyceride_lipid_names() {
        lipids.push((*name).to_string());
    }
    for name in triglyceride_lipid_names() {
        lipids.push((*name).to_string());
    }
    for name in ltf_single_chain_lipid_names() {
        lipids.push((*name).to_string());
    }
    lipids.push("CHOL".to_string());
    lipids
}

fn diacyl_template_with_linker(
    name: String,
    head_group: &str,
    tail_group: &str,
    net_charge_e: f32,
    linker: LinkerKind,
) -> LipidTemplate {
    let bead_names = diacyl_bead_names_with_linker(head_group, tail_group, linker);
    let source = match linker {
        LinkerKind::Glycerol => "warp-cg.lipid-template.martini-diacyl.v1",
        LinkerKind::Ether => "warp-cg.lipid-template.martini-ether.v1",
        LinkerKind::Plasmalogen => "warp-cg.lipid-template.martini-plasmalogen.v1",
    };
    LipidTemplate {
        name,
        source,
        radius_angstrom: 4.0,
        net_charge_e,
        beads: bead_names
            .iter()
            .enumerate()
            .filter_map(|(idx, bead)| {
                bead.map(|name| TemplateBead {
                    name: name.to_string(),
                    offset_angstrom: [DIACYL_X[idx], DIACYL_Y[idx], DIACYL_Z[idx] - 5.0],
                    charge_e: bead_charge(name),
                })
            })
            .collect(),
    }
}

fn linked_diacyl_template(
    name: String,
    base_name: &str,
    head_group: &str,
    tail_group: &str,
    linker: LinkerKind,
) -> Option<LipidTemplate> {
    if is_phosphoinositide_variant(head_group) {
        Some(inositol_template_with_linker(
            name, head_group, tail_group, linker,
        ))
    } else {
        let _ = base_name;
        Some(diacyl_template_with_linker(
            name,
            head_group,
            tail_group,
            head_group_charge(head_group),
            linker,
        ))
    }
}

fn plasmalogen_template((head_group, tail_group): (&str, &str)) -> LipidTemplate {
    let diacyl_head_group = match head_group {
        "C" => "PC",
        "E" => "PE",
        _ => "PC",
    };
    diacyl_template_with_tail_beads(
        format!("{tail_group}PL{head_group}"),
        diacyl_head_group,
        plasmalogen_tail_bead_names(tail_group),
        head_group_charge(diacyl_head_group),
        LinkerKind::Plasmalogen,
    )
}

fn diacyl_template_with_tail_beads(
    name: String,
    head_group: &str,
    tail: [Option<&'static str>; 12],
    net_charge_e: f32,
    linker: LinkerKind,
) -> LipidTemplate {
    let bead_names = diacyl_bead_names_from_tail(head_group, tail, linker);
    let source = match linker {
        LinkerKind::Glycerol => "warp-cg.lipid-template.martini-diacyl.v1",
        LinkerKind::Ether => "warp-cg.lipid-template.martini-ether.v1",
        LinkerKind::Plasmalogen => "warp-cg.lipid-template.martini-plasmalogen.v1",
    };
    LipidTemplate {
        name,
        source,
        radius_angstrom: 4.0,
        net_charge_e,
        beads: bead_names
            .iter()
            .enumerate()
            .filter_map(|(idx, bead)| {
                bead.map(|name| TemplateBead {
                    name: name.to_string(),
                    offset_angstrom: [DIACYL_X[idx], DIACYL_Y[idx], DIACYL_Z[idx] - 5.0],
                    charge_e: bead_charge(name),
                })
            })
            .collect(),
    }
}

fn head_group_charge(head_group: &str) -> f32 {
    match head_group {
        "PC" | "PE" | "DG" => 0.0,
        "PA" | "PG" | "PS" | "PI" => -1.0,
        "P1" | "P4" | "P5" => -3.0,
        "P2" | "P6" | "P7" => -4.0,
        "P3" => -5.0,
        _ => 0.0,
    }
}

fn is_phosphoinositide_variant(head_group: &str) -> bool {
    matches!(head_group, "P1" | "P2" | "P3" | "P4" | "P5" | "P6" | "P7")
}

fn inositol_template_with_linker(
    name: String,
    head_group: &str,
    tail_group: &str,
    linker: LinkerKind,
) -> LipidTemplate {
    let bead_names = inositol_bead_names(head_group, tail_group);
    let source = match linker {
        LinkerKind::Glycerol => "warp-cg.lipid-template.martini-inositol.v1",
        LinkerKind::Ether => "warp-cg.lipid-template.martini-inositol-ether.v1",
        LinkerKind::Plasmalogen => "warp-cg.lipid-template.martini-inositol.v1",
    };
    LipidTemplate {
        name,
        source,
        radius_angstrom: 4.0,
        net_charge_e: head_group_charge(head_group),
        beads: bead_names
            .iter()
            .enumerate()
            .filter_map(|(idx, bead)| {
                let bead_name = if idx == 8 || idx == 9 {
                    linker_bead_names(linker)[idx - 8]
                } else {
                    *bead
                };
                bead_name.map(|name| TemplateBead {
                    name: name.to_string(),
                    offset_angstrom: [INOSITOL_X[idx], INOSITOL_Y[idx], INOSITOL_Z[idx] - 5.0],
                    charge_e: inositol_bead_charge(head_group, name),
                })
            })
            .collect(),
    }
}

fn inositol_bead_names(head_group: &str, tail_group: &str) -> [Option<&'static str>; 22] {
    let head = match head_group {
        "P1" => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            Some("PO4"),
            Some("P3"),
            None,
            None,
        ],
        "P2" => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            Some("PO4"),
            Some("P3"),
            Some("P4"),
            None,
        ],
        "P3" => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            Some("PO4"),
            Some("P3"),
            Some("P4"),
            Some("P5"),
        ],
        "P4" => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            Some("PO4"),
            None,
            Some("P4"),
            None,
        ],
        "P5" => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            Some("PO4"),
            None,
            None,
            Some("P5"),
        ],
        "P6" => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            Some("PO4"),
            None,
            Some("P4"),
            Some("P5"),
        ],
        "P7" => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            Some("PO4"),
            Some("P3"),
            None,
            Some("P5"),
        ],
        _ => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            Some("PO4"),
            None,
            None,
            None,
        ],
    };
    let tail = tail_bead_names(tail_group);
    [
        head[0],
        head[1],
        head[2],
        head[3],
        head[4],
        head[5],
        head[6],
        head[7],
        Some("GL1"),
        Some("GL2"),
        tail[0],
        tail[1],
        tail[2],
        tail[3],
        tail[4],
        tail[5],
        tail[6],
        tail[7],
        tail[8],
        tail[9],
        tail[10],
        tail[11],
    ]
}

pub(crate) fn tail_bead_names(tail_group: &str) -> [Option<&'static str>; 12] {
    let full = diacyl_bead_names("PC", tail_group);
    [
        full[8], full[9], full[10], full[11], full[12], full[13], full[14], full[15], full[16],
        full[17], full[18], full[19],
    ]
}

fn inositol_bead_charge(head_group: &str, bead_name: &str) -> f32 {
    match bead_name {
        "PO4" => -1.0,
        "P3" if matches!(head_group, "P1") => -2.0,
        "P3" if matches!(head_group, "P2" | "P7") => -1.5,
        "P4" if matches!(head_group, "P2" | "P6") => -1.5,
        "P5" if matches!(head_group, "P6" | "P7") => -1.5,
        "P3" if matches!(head_group, "P3") => -1.3,
        "P4" if matches!(head_group, "P3") => -1.4,
        "P5" if matches!(head_group, "P3") => -1.3,
        "P4" if matches!(head_group, "P4") => -2.0,
        "P5" if matches!(head_group, "P5") => -2.0,
        _ => 0.0,
    }
}

fn chol_template() -> LipidTemplate {
    let bead_names = [
        Some("ROH"),
        Some("R1"),
        Some("R2"),
        Some("R3"),
        Some("R4"),
        None,
        Some("R5"),
        Some("R6"),
        Some("C1"),
        Some("C2"),
    ];
    LipidTemplate {
        name: "CHOL".to_string(),
        source: "warp-cg.lipid-template.martini-sterol.v1",
        radius_angstrom: 3.5,
        net_charge_e: 0.0,
        beads: bead_names
            .iter()
            .enumerate()
            .filter_map(|(idx, bead)| {
                bead.map(|name| TemplateBead {
                    name: name.to_string(),
                    offset_angstrom: [CHOL_X[idx], CHOL_Y[idx], CHOL_Z[idx] - 2.65],
                    charge_e: 0.0,
                })
            })
            .collect(),
    }
}

fn ltf_named_lipid_template(
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
                charge_e: ltf_named_lipid_bead_charge(name, bead_name),
            })
            .collect(),
    }
}

fn ltf_sphingolipid_template(name: &str) -> Option<LipidTemplate> {
    let (tail_key, head) = name
        .strip_suffix("CER")
        .map(|tail| (tail, "CER"))
        .or_else(|| name.strip_suffix("SM").map(|tail| (tail, "SM")))?;
    let tail = ltf_sphingolipid_tail_beads(tail_key)?;
    let mut beads = match head {
        "CER" => vec!["COH", "OH1", "AM2"],
        "SM" => vec!["NC3", "PO4", "OH1", "AM2"],
        _ => return None,
    };
    beads.extend(tail);
    Some(ltf_named_lipid_template(
        name,
        "warp-cg.lipid-template.martini-ltf-sphingolipid.v1",
        0.0,
        &beads,
    ))
}

fn ltf_sphingolipid_tail_beads(tail_key: &str) -> Option<&'static [&'static str]> {
    match tail_key {
        "U" | "M" => Some(&["T1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B"]),
        "P" | "S" => Some(&["T1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B"]),
        "K" | "B" | "C" => Some(&[
            "T1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B", "C5B",
        ]),
        "X" => Some(&[
            "T1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B", "C5B", "C6B",
        ]),
        "O" => Some(&["T1A", "C2A", "C3A", "C4A", "C1B", "D2B", "C3B", "C4B"]),
        "N" => Some(&[
            "T1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B", "D4B", "C5B", "C6B",
        ]),
        _ => None,
    }
}

fn ltf_sphingolipid_names() -> &'static [&'static str] {
    &[
        "BCER", "BSM", "CCER", "CSM", "KCER", "KSM", "MCER", "MSM", "NCER", "NSM", "OCER", "OSM",
        "PCER", "PSM", "SCER", "SSM", "UCER", "USM", "XCER", "XSM",
    ]
}

fn ltf_named_lipid_bead_charge(lipid_name: &str, bead_name: &str) -> f32 {
    match (lipid_name, bead_name) {
        ("DO2B" | "DO3B", "PO4") => -1.0,
        (name, "NC3") if name.ends_with("SM") => 1.0,
        (name, "PO4") if name.ends_with("SM") => -1.0,
        _ => 0.0,
    }
}

fn bead_charge(name: &str) -> f32 {
    match name {
        "NC3" | "NH3" => 1.0,
        "PO4" => -1.0,
        _ => 0.0,
    }
}

fn normalize_name(name: &str) -> String {
    name.trim().to_ascii_uppercase()
}

#[cfg(test)]
#[path = "build_lipids_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "build_lipids_registry_coverage_tests.rs"]
mod registry_coverage_tests;
