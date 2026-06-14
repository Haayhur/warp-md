use super::*;

pub(super) fn lookup_tailcode_solvent_library(name: &str) -> Option<SolventLibraryEntry> {
    if let Some(entry) = lookup_ltf_named_tailcode_solvent_library(name) {
        return Some(entry);
    }
    let (family, tailcode) = split_tailcode_solvent_name(name)?;
    let family = family.trim().to_ascii_lowercase();
    let tailcode = tailcode.trim();
    if tailcode.is_empty() {
        return None;
    }
    match family.as_str() {
        "hc" | "hydrocarbon" => build_tailcode_solvent_library(
            &format!("HC{tailcode}"),
            &[tailcode],
            TailcodeFamily::Hydrocarbon,
        ),
        "fa" | "fattyacid" | "fatty-acid" => build_tailcode_solvent_library(
            &format!("FA{tailcode}"),
            &[tailcode],
            TailcodeFamily::FattyAcid,
        ),
        "mg" | "monoglyceride" | "mono-glyceride" => build_tailcode_solvent_library(
            &format!("MG{}", tailcode.replace(',', "")),
            &[tailcode],
            TailcodeFamily::Monoglyceride,
        ),
        "dg" | "diglyceride" | "di-glyceride" => {
            let tails = split_generated_tailcodes(tailcode, 2)?;
            build_tailcode_solvent_library(
                &format!("DG{}", tails.join("")),
                &tails,
                TailcodeFamily::Diglyceride,
            )
        }
        "tg" | "triglyceride" | "tri-glyceride" => {
            let tails = split_generated_tailcodes(tailcode, 3)?;
            build_tailcode_solvent_library(
                &format!("TG{}", tails.join("")),
                &tails,
                TailcodeFamily::Triglyceride,
            )
        }
        "bmp2" => {
            let tails = split_generated_tailcodes(tailcode, 2)?;
            build_tailcode_solvent_library(
                &format!("B2{}", tails.join("")),
                &tails,
                TailcodeFamily::Bmp2,
            )
        }
        "bmp3" => {
            let tails = split_generated_tailcodes(tailcode, 2)?;
            build_tailcode_solvent_library(
                &format!("B3{}", tails.join("")),
                &tails,
                TailcodeFamily::Bmp3,
            )
        }
        "cl" | "cardiolipin" | "cardio-lipin" => {
            let tails = split_generated_tailcodes(tailcode, 4)?;
            build_tailcode_solvent_library(
                &format!(
                    "L{}",
                    tails
                        .iter()
                        .filter_map(|tail| tail.chars().next())
                        .collect::<String>()
                ),
                &tails,
                TailcodeFamily::Cardiolipin,
            )
        }
        "sm" | "sphingolipid" | "sphingo-lipid" => {
            let parts = split_generated_tailcodes(tailcode, 3)?;
            build_sphingolipid_solvent_library(parts[0], parts[1], parts[2])
        }
        _ => None,
    }
}

pub(super) fn known_ltf_named_tailcode_solvent_names() -> &'static [&'static str] {
    &[
        "C16", "C20", "C24", "IC1", "IC2", "IC3", "IC4", "IC5", "IC6", "IC7", "IC8", "IC9", "OA",
        "LA", "LNA", "MA", "PA", "SA", "MO", "ML", "MLN", "MS", "MP", "MM", "MLA", "DTDG", "DJDG",
        "DUDG", "DMDG", "MPDG", "MSDG", "PMDG", "SMDG", "DPDG", "DSDG", "PSDG", "DKDG", "DBDG",
        "DXDG", "DCDG", "DRDG", "DYDG", "DODG", "DVDG", "YODG", "OEDG", "DGDG", "DEDG", "DNDG",
        "DLDG", "DFDG", "DADG", "DDDG", "PYDG", "PODG", "SODG", "PEDG", "PLDG", "SLDG", "PIDG",
        "PFDG", "PQDG", "PADG", "SADG", "PDDG", "SDDG", "OLDG", "ODDG", "LFDG", "TO", "TL", "TLN",
        "TS", "TP", "TM", "TLA", "DO2B", "DO3B", "TMCL", "TOCL", "BCER", "BSM", "CCER", "CSM",
        "KCER", "KSM", "MCER", "MSM", "NCER", "NSM", "OCER", "OSM", "PCER", "PSM", "SCER", "SSM",
        "UCER", "USM", "XCER", "XSM",
    ]
}

fn lookup_ltf_named_tailcode_solvent_library(name: &str) -> Option<SolventLibraryEntry> {
    let normalized = normalize_library_name(name);
    if let Some((family, tails)) = ltf_named_complex_tailcodes(&normalized) {
        return build_tailcode_solvent_library(&normalized, &tails, family);
    }
    if let Some((head, tail1, tail2)) = ltf_named_sphingolipid_parts(&normalized) {
        let mut entry = build_sphingolipid_solvent_library(head, tail1, tail2)?;
        entry.name = normalized;
        return Some(entry);
    }
    if let Some((tail1, tail2)) = ltf_named_diglyceride_tailcodes(&normalized) {
        return build_tailcode_solvent_library(
            &normalized,
            &[tail1, tail2],
            TailcodeFamily::Diglyceride,
        );
    }
    let (family, tails): (TailcodeFamily, &[&str]) = match normalized.as_str() {
        "C16" => (TailcodeFamily::Hydrocarbon, &["CCCC"]),
        "C20" => (TailcodeFamily::Hydrocarbon, &["CCCCC"]),
        "C24" => (TailcodeFamily::Hydrocarbon, &["CCCCCC"]),
        "IC1" => (TailcodeFamily::Hydrocarbon, &["CCDCC"]),
        "IC2" => (TailcodeFamily::Hydrocarbon, &["CCDDC"]),
        "IC3" | "IC4" | "IC5" => (TailcodeFamily::Hydrocarbon, &["CDDDD"]),
        "IC6" => (TailcodeFamily::Hydrocarbon, &["CDCCC"]),
        "IC7" => (TailcodeFamily::Hydrocarbon, &["CDCDD"]),
        "IC8" => (TailcodeFamily::Hydrocarbon, &["CDCDC"]),
        "IC9" => (TailcodeFamily::Hydrocarbon, &["CDCCD"]),
        "OA" | "MO" | "TO" => (named_ltf_family(&normalized)?, &["CDCC"]),
        "LA" | "ML" | "TL" => (named_ltf_family(&normalized)?, &["CDDC"]),
        "LNA" | "MLN" | "TLN" => (named_ltf_family(&normalized)?, &["CDDD"]),
        "MA" | "MM" | "MLA" | "TM" | "TLA" => (named_ltf_family(&normalized)?, &["CCC"]),
        "PA" | "SA" | "MS" | "MP" | "TS" | "TP" => (named_ltf_family(&normalized)?, &["CCCC"]),
        _ => return None,
    };
    let expanded_tails = match family {
        TailcodeFamily::Triglyceride => vec![tails[0], tails[0], tails[0]],
        _ => tails.to_vec(),
    };
    build_tailcode_solvent_library(&normalized, &expanded_tails, family)
}

fn named_ltf_family(name: &str) -> Option<TailcodeFamily> {
    match name {
        "OA" | "LA" | "LNA" | "MA" | "PA" | "SA" => Some(TailcodeFamily::FattyAcid),
        "MO" | "ML" | "MLN" | "MS" | "MP" | "MM" | "MLA" => Some(TailcodeFamily::Monoglyceride),
        "TO" | "TL" | "TLN" | "TS" | "TP" | "TM" | "TLA" => Some(TailcodeFamily::Triglyceride),
        _ => None,
    }
}

fn ltf_named_sphingolipid_parts(name: &str) -> Option<(&'static str, &'static str, &'static str)> {
    let (tail_key, head) = name
        .strip_suffix("CER")
        .map(|tail| (tail, "CER"))
        .or_else(|| name.strip_suffix("SM").map(|tail| (tail, "PC")))?;
    let (tail1, tail2) = match tail_key {
        "U" | "M" => ("TCCC", "CCC"),
        "P" | "S" => ("TCCC", "CCCC"),
        "K" | "B" | "C" => ("TCCC", "CCCCC"),
        "X" => ("TCCC", "CCCCCC"),
        "O" => ("TCCC", "CDCC"),
        "N" => ("TCCC", "CCCDCC"),
        _ => return None,
    };
    Some((head, tail1, tail2))
}

fn ltf_named_diglyceride_tailcodes(name: &str) -> Option<(&'static str, &'static str)> {
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

fn ltf_named_complex_tailcodes(name: &str) -> Option<(TailcodeFamily, Vec<&'static str>)> {
    match name {
        "DO2B" => Some((TailcodeFamily::Bmp2, vec!["CDCC", "CDCC"])),
        "DO3B" => Some((TailcodeFamily::Bmp3, vec!["CDCC", "CDCC"])),
        "TMCL" => Some((
            TailcodeFamily::Cardiolipin,
            vec!["CCC", "CCC", "CCC", "CCC"],
        )),
        "TOCL" => Some((
            TailcodeFamily::Cardiolipin,
            vec!["CCCC", "CCCC", "CCCC", "CCCC"],
        )),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum TailcodeFamily {
    Hydrocarbon,
    FattyAcid,
    Monoglyceride,
    Diglyceride,
    Triglyceride,
    Bmp2,
    Bmp3,
    Cardiolipin,
}

fn build_tailcode_solvent_library(
    name: &str,
    tailcodes: &[&str],
    family: TailcodeFamily,
) -> Option<SolventLibraryEntry> {
    let mut beads = Vec::new();
    match family {
        TailcodeFamily::Hydrocarbon => {
            if tailcodes.len() != 1 {
                return None;
            }
            append_tailcode_beads(&mut beads, tailcodes[0], "A", [0.0, 0.0, 0.0])?;
        }
        TailcodeFamily::FattyAcid => {
            if tailcodes.len() != 1 {
                return None;
            }
            beads.push(solvent_library_bead("COO", [0.0, 0.0, 0.0], -1.0));
            append_tailcode_beads(&mut beads, tailcodes[0], "A", [0.0, 0.0, -3.0])?;
        }
        TailcodeFamily::Monoglyceride => {
            if tailcodes.len() != 1 {
                return None;
            }
            beads.push(solvent_library_bead("DOH", [0.0, 0.0, 3.0], 0.0));
            beads.push(solvent_library_bead("GL1", [0.0, 0.0, 0.0], 0.0));
            append_tailcode_beads(&mut beads, tailcodes[0], "A", [0.0, 0.0, -3.0])?;
        }
        TailcodeFamily::Diglyceride => {
            if tailcodes.len() != 2 {
                return None;
            }
            beads.push(solvent_library_bead("COH", [0.0, 0.0, 3.0], 0.0));
            beads.push(solvent_library_bead("GL1", [0.0, 0.0, 0.0], 0.0));
            beads.push(solvent_library_bead("GL2", [2.5, 0.0, 0.0], 0.0));
            append_tailcode_beads(&mut beads, tailcodes[0], "A", [0.0, 0.0, -3.0])?;
            append_tailcode_beads(&mut beads, tailcodes[1], "B", [2.5, 0.0, -3.0])?;
        }
        TailcodeFamily::Triglyceride => {
            if tailcodes.len() != 3 {
                return None;
            }
            beads.push(solvent_library_bead("GL1", [0.0, 0.0, 0.0], 0.0));
            beads.push(solvent_library_bead("GL2", [2.5, 0.0, 0.0], 0.0));
            beads.push(solvent_library_bead("GL3", [5.0, 0.0, 0.0], 0.0));
            append_tailcode_beads(&mut beads, tailcodes[0], "A", [0.0, 0.0, -3.0])?;
            append_tailcode_beads(&mut beads, tailcodes[1], "B", [2.5, 0.0, -3.0])?;
            append_tailcode_beads(&mut beads, tailcodes[2], "C", [5.0, 0.0, -3.0])?;
        }
        TailcodeFamily::Bmp2 => {
            if tailcodes.len() != 2 {
                return None;
            }
            append_bmp_linkers_and_tails(&mut beads, tailcodes, BmpLinkerVariant::Two)?;
        }
        TailcodeFamily::Bmp3 => {
            if tailcodes.len() != 2 {
                return None;
            }
            append_bmp_linkers_and_tails(&mut beads, tailcodes, BmpLinkerVariant::Three)?;
        }
        TailcodeFamily::Cardiolipin => {
            if tailcodes.len() != 4 {
                return None;
            }
            append_cardiolipin_linkers_and_tails(&mut beads, tailcodes)?;
        }
    }
    Some(SolventLibraryEntry {
        name: name.to_string(),
        mapping_ratio: 1.0,
        molar_mass_g_mol: default_solvent_molar_mass(),
        density_kg_m3: default_solvent_density(),
        beads,
    })
}

#[derive(Clone, Copy)]
enum BmpLinkerVariant {
    Two,
    Three,
}

fn append_bmp_linkers_and_tails(
    beads: &mut Vec<SolventLibraryBead>,
    tailcodes: &[&str],
    variant: BmpLinkerVariant,
) -> Option<()> {
    beads.push(solvent_library_bead("PO4", [0.0, 0.0, 0.0], -1.0));
    match variant {
        BmpLinkerVariant::Two => {
            beads.push(solvent_library_bead("OH1", [2.0, 0.0, -2.5], -1.0));
            beads.push(solvent_library_bead("GL1", [2.0, 0.0, -5.5], -1.0));
            append_tailcode_beads(beads, tailcodes[0], "A", [2.0, 0.0, -8.5])?;
            beads.push(solvent_library_bead("OH2", [-2.0, 0.0, -2.5], -1.0));
            beads.push(solvent_library_bead("GL2", [-2.0, 0.0, -5.5], -1.0));
            append_tailcode_beads(beads, tailcodes[1], "B", [-2.0, 0.0, -8.5])?;
        }
        BmpLinkerVariant::Three => {
            beads.push(solvent_library_bead("GL1", [2.0, 0.0, -2.5], -1.0));
            beads.push(solvent_library_bead("OH1", [4.5, 0.0, -2.5], -1.0));
            append_tailcode_beads(beads, tailcodes[0], "A", [2.0, 0.0, -5.5])?;
            beads.push(solvent_library_bead("GL2", [-2.0, 0.0, -2.5], -1.0));
            beads.push(solvent_library_bead("OH2", [-4.5, 0.0, -2.5], -1.0));
            append_tailcode_beads(beads, tailcodes[1], "B", [-2.0, 0.0, -5.5])?;
        }
    }
    Some(())
}

fn append_cardiolipin_linkers_and_tails(
    beads: &mut Vec<SolventLibraryBead>,
    tailcodes: &[&str],
) -> Option<()> {
    beads.push(solvent_library_bead("GLC", [0.0, 0.0, 0.0], 0.0));
    beads.push(solvent_library_bead("PO41", [-1.5, 0.0, -3.0], 0.0));
    beads.push(solvent_library_bead("GL11", [-1.5, 0.0, -6.0], 0.0));
    beads.push(solvent_library_bead("GL21", [-4.0, 0.0, -6.0], 0.0));
    append_tailcode_beads(beads, tailcodes[0], "A1", [-1.5, 0.0, -9.0])?;
    append_tailcode_beads(beads, tailcodes[1], "B1", [-4.0, 0.0, -9.0])?;
    beads.push(solvent_library_bead("PO42", [1.5, 0.0, -3.0], 0.0));
    beads.push(solvent_library_bead("GL12", [1.5, 0.0, -6.0], 0.0));
    beads.push(solvent_library_bead("GL22", [4.0, 0.0, -6.0], 0.0));
    append_tailcode_beads(beads, tailcodes[2], "A2", [1.5, 0.0, -9.0])?;
    append_tailcode_beads(beads, tailcodes[3], "B2", [4.0, 0.0, -9.0])?;
    Some(())
}

fn build_sphingolipid_solvent_library(
    head_name: &str,
    tail1: &str,
    tail2: &str,
) -> Option<SolventLibraryEntry> {
    let normalized_head = normalize_library_name(head_name);
    let mut beads = Vec::new();
    append_sphingolipid_head(&mut beads, &normalized_head)?;
    beads.push(solvent_library_bead("OH1", [0.0, 0.0, 0.0], 0.0));
    beads.push(solvent_library_bead("AM2", [2.5, 0.0, 0.0], 0.0));
    append_tailcode_beads(&mut beads, tail1, "A", [0.0, 0.0, -3.0])?;
    append_tailcode_beads(&mut beads, tail2, "B", [2.5, 0.0, -3.0])?;
    let head_label = if normalized_head.len() <= 2 {
        normalized_head.clone()
    } else {
        normalized_head.chars().take(1).collect::<String>()
    };
    let tail_label = [tail1, tail2]
        .iter()
        .filter_map(|tail| tail.chars().next())
        .collect::<String>();
    Some(SolventLibraryEntry {
        name: format!("S{head_label}{tail_label}"),
        mapping_ratio: 1.0,
        molar_mass_g_mol: default_solvent_molar_mass(),
        density_kg_m3: default_solvent_density(),
        beads,
    })
}

fn append_sphingolipid_head(beads: &mut Vec<SolventLibraryBead>, head: &str) -> Option<()> {
    match head {
        "CER" => beads.push(solvent_library_bead("COH", [0.0, 0.0, 3.0], 0.0)),
        "PC" => {
            beads.push(solvent_library_bead("NC3", [0.0, 0.0, 6.0], 1.0));
            beads.push(solvent_library_bead("PO4", [0.0, 0.0, 3.0], -1.0));
        }
        "PE" => {
            beads.push(solvent_library_bead("NH3", [0.0, 0.0, 6.0], 1.0));
            beads.push(solvent_library_bead("PO4", [0.0, 0.0, 3.0], -1.0));
        }
        "PG" => {
            beads.push(solvent_library_bead("GL0", [0.0, 0.0, 6.0], 0.0));
            beads.push(solvent_library_bead("PO4", [0.0, 0.0, 3.0], -1.0));
        }
        "PA" => beads.push(solvent_library_bead("PO4", [0.0, 0.0, 3.0], -1.0)),
        "PS" => {
            beads.push(solvent_library_bead("CNO", [0.0, 0.0, 6.0], 0.0));
            beads.push(solvent_library_bead("PO4", [0.0, 0.0, 3.0], -1.0));
        }
        "PI" | "P1" | "P2" | "P3" | "P4" | "P5" | "P6" | "P7" => {
            append_sphingolipid_inositol_head(beads, head)?;
        }
        _ => return None,
    }
    Some(())
}

fn append_sphingolipid_inositol_head(
    beads: &mut Vec<SolventLibraryBead>,
    head: &str,
) -> Option<()> {
    const NAMES: [&str; 8] = ["C1", "C2", "C3", "C4", "PO4", "P3", "P4", "P5"];
    const OFFSETS: [[f32; 3]; 8] = [
        [0.0, 0.0, 6.3333335],
        [-2.0, 0.0, 9.666667],
        [2.0, 0.0, 8.833333],
        [0.0, 0.0, 8.0],
        [0.0, 0.0, 3.0],
        [2.75, 0.0, 10.5],
        [1.75, 0.0, 11.333333],
        [-2.0, 0.0, 12.166667],
    ];
    let present = match head {
        "PI" => [true, true, true, true, true, false, false, false],
        "P1" => [true, true, true, true, true, true, false, false],
        "P2" => [true, true, true, true, true, true, true, false],
        "P3" => [true, true, true, true, true, true, true, true],
        "P4" => [true, true, true, true, true, false, true, false],
        "P5" => [true, true, true, true, true, false, false, true],
        "P6" => [true, true, true, true, true, false, true, true],
        "P7" => [true, true, true, true, true, true, false, true],
        _ => return None,
    };
    for (idx, bead_name) in NAMES.iter().enumerate() {
        if present[idx] {
            beads.push(solvent_library_bead(
                *bead_name,
                OFFSETS[idx],
                sphingolipid_inositol_head_charge(head, bead_name),
            ));
        }
    }
    Some(())
}

fn sphingolipid_inositol_head_charge(head: &str, bead_name: &str) -> f32 {
    match (head, bead_name) {
        ("PI", "PO4") => -1.0,
        ("P1", "PO4") => -1.0,
        ("P1", "P3") => -2.0,
        ("P2", "PO4") => -1.0,
        ("P2", "P3" | "P4") => -1.5,
        ("P3", "PO4") => -1.0,
        ("P3", "P3" | "P5") => -1.3,
        ("P3", "P4") => -1.4,
        ("P4", "PO4") => -1.0,
        ("P4", "P4") => -2.0,
        ("P5", "PO4") => -1.0,
        ("P5", "P5") => -2.0,
        ("P6", "PO4") => -1.0,
        ("P6", "P4" | "P5") => -1.5,
        ("P7", "PO4") => -1.0,
        ("P7", "P3" | "P5") => -1.5,
        _ => 0.0,
    }
}

fn split_generated_tailcodes(tailcode: &str, expected_count: usize) -> Option<Vec<&str>> {
    let tails = tailcode
        .split([',', '+'])
        .map(str::trim)
        .collect::<Vec<_>>();
    if tails.len() == expected_count && tails.iter().all(|tail| !tail.is_empty()) {
        Some(tails)
    } else {
        None
    }
}

fn append_tailcode_beads(
    beads: &mut Vec<SolventLibraryBead>,
    tailcode: &str,
    suffix: &str,
    origin_angstrom: [f32; 3],
) -> Option<()> {
    if tailcode.is_empty() {
        return None;
    }
    for (idx, code) in tailcode.chars().enumerate() {
        let bead_prefix = match code {
            'C' | 'c' => "C",
            'D' => "D",
            'T' => "T",
            't' => "t",
            'F' => "F",
            _ => return None,
        };
        beads.push(solvent_library_bead(
            format!("{bead_prefix}{}{suffix}", idx + 1),
            [
                origin_angstrom[0],
                origin_angstrom[1],
                origin_angstrom[2] - 3.0 * idx as f32,
            ],
            0.0,
        ));
    }
    Some(())
}
