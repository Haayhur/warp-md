use super::{LipidTemplate, TemplateBead};

pub(super) fn ltf_single_chain_template(name: &str) -> Option<LipidTemplate> {
    let (family, tail_code) = match name {
        "C16" => (SingleChainFamily::Hydrocarbon, "CCCC"),
        "C20" => (SingleChainFamily::Hydrocarbon, "CCCCC"),
        "C24" => (SingleChainFamily::Hydrocarbon, "CCCCCC"),
        "IC1" => (SingleChainFamily::Hydrocarbon, "CCDCC"),
        "IC2" => (SingleChainFamily::Hydrocarbon, "CCDDC"),
        "IC3" => (SingleChainFamily::Hydrocarbon, "CCDDD"),
        "IC4" | "IC5" => (SingleChainFamily::Hydrocarbon, "CDDDD"),
        "IC6" => (SingleChainFamily::Hydrocarbon, "CDCCC"),
        "IC7" => (SingleChainFamily::Hydrocarbon, "CDCDD"),
        "IC8" => (SingleChainFamily::Hydrocarbon, "CDCDC"),
        "IC9" => (SingleChainFamily::Hydrocarbon, "CDCCD"),
        "OA" => (SingleChainFamily::FattyAcid, "CDCC"),
        "LA" => (SingleChainFamily::FattyAcid, "CDDC"),
        "LNA" => (SingleChainFamily::FattyAcid, "CDDD"),
        "MA" => (SingleChainFamily::FattyAcid, "CCC"),
        "PA" | "SA" => (SingleChainFamily::FattyAcid, "CCCC"),
        _ => return None,
    };
    Some(build_single_chain_template(name, family, tail_code))
}

pub(super) fn ltf_single_chain_lipid_names() -> &'static [&'static str] {
    &[
        "C16", "C20", "C24", "IC1", "IC2", "IC3", "IC4", "IC5", "IC6", "IC7", "IC8", "IC9", "OA",
        "LA", "LNA", "MA", "PA", "SA",
    ]
}

#[derive(Clone, Copy)]
enum SingleChainFamily {
    Hydrocarbon,
    FattyAcid,
}

fn build_single_chain_template(
    name: &str,
    family: SingleChainFamily,
    tail_code: &str,
) -> LipidTemplate {
    let mut beads = Vec::new();
    if matches!(family, SingleChainFamily::FattyAcid) {
        beads.push(TemplateBead {
            name: "COO".to_string(),
            offset_angstrom: [0.0, 0.0, 0.0],
            charge_e: -1.0,
        });
    }
    for (idx, code) in tail_code.chars().enumerate() {
        beads.push(TemplateBead {
            name: format!("{}{}A", code, idx + 1),
            offset_angstrom: [0.0, 0.0, -0.47 * (idx as f32 + 1.0)],
            charge_e: 0.0,
        });
    }
    LipidTemplate {
        name: name.to_string(),
        source: match family {
            SingleChainFamily::Hydrocarbon => "warp-cg.lipid-template.martini-ltf-hydrocarbon.v1",
            SingleChainFamily::FattyAcid => "warp-cg.lipid-template.martini-ltf-fatty-acid.v1",
        },
        radius_angstrom: 3.5,
        net_charge_e: if matches!(family, SingleChainFamily::FattyAcid) {
            -1.0
        } else {
            0.0
        },
        beads,
    }
}
