use super::{LipidTemplate, TemplateBead};
use crate::build_lipids_diacyl::{coby_tail_code_names, generated_sphingomyelin_tail_bead_names};

const SM_X: [f32; 20] = [
    0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0,
];
const SM_Y: [f32; 20] = [0.0; 20];
const SM_Z: [f32; 20] = [
    10.0, 9.0, 9.0, 8.0, 8.0, 7.0, 6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    0.0,
];

pub(super) fn generated_sphingomyelin_template(name: &str) -> Option<LipidTemplate> {
    let tail_group = name.strip_suffix("SM")?;
    if !is_generated_sphingomyelin_tail(tail_group) {
        return None;
    }
    let tail = generated_sphingomyelin_tail_bead_names(tail_group)?;
    let bead_names = [
        None,
        None,
        None,
        Some("NC3"),
        None,
        Some("PO4"),
        Some("AM1"),
        Some("AM2"),
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
    ];
    Some(LipidTemplate {
        name: name.to_string(),
        source: "warp-cg.lipid-template.martini-generated-sphingomyelin.v1",
        radius_angstrom: 4.0,
        net_charge_e: 0.0,
        beads: bead_names
            .iter()
            .enumerate()
            .filter_map(|(idx, bead)| {
                bead.map(|name| TemplateBead {
                    name: name.to_string(),
                    offset_angstrom: [SM_X[idx], SM_Y[idx], SM_Z[idx] - 5.0],
                    charge_e: bead_charge(name),
                })
            })
            .collect(),
    })
}

pub(super) fn generated_sphingomyelin_tail_names() -> Vec<String> {
    let mut tails = Vec::new();
    for sn1 in ['P', 'B', 'X'] {
        for sn2 in coby_tail_code_names() {
            let name = if sn1 == *sn2 {
                format!("D{sn1}")
            } else {
                format!("{sn2}{sn1}")
            };
            tails.push(name);
        }
    }
    tails
}

fn is_generated_sphingomyelin_tail(tail_group: &str) -> bool {
    let mut chars = tail_group.chars();
    let first = chars.next();
    let second = chars.next();
    if chars.next().is_some() {
        return false;
    }
    match (first, second) {
        (Some('D'), Some(sn1)) => matches!(sn1, 'P' | 'B' | 'X'),
        (Some(_sn2), Some(sn1)) => matches!(sn1, 'P' | 'B' | 'X'),
        _ => false,
    }
}

fn bead_charge(name: &str) -> f32 {
    match name {
        "NC3" => 1.0,
        "PO4" => -1.0,
        _ => 0.0,
    }
}
