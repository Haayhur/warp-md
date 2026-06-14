pub(crate) fn known_standard_diacyl_tail_names() -> &'static [&'static str] {
    &[
        "CC", "DL", "DF", "DA", "DD", "PY", "PO", "SO", "PE", "PL", "SL", "PI", "PF", "PQ", "PA",
        "SA", "PD", "SD", "OL", "OD", "LF", "DO", "DP", "DS",
    ]
}

pub(crate) fn ltf_release_tail_bead_names(tail_group: &str) -> Option<[Option<&'static str>; 12]> {
    match tail_group {
        "DT" | "DJ" => Some(tail(&["C1A", "C2A"], &["C1B", "C2B"])),
        "DU" | "DM" => Some(tail(&["C1A", "C2A", "C3A"], &["C1B", "C2B", "C3B"])),
        "MP" | "MS" => Some(tail(&["C1A", "C2A", "C3A", "C4A"], &["C1B", "C2B", "C3B"])),
        "PM" | "SM" => Some(tail(&["C1A", "C2A", "C3A"], &["C1B", "C2B", "C3B", "C4B"])),
        "DP" | "DS" | "PS" => Some(tail(
            &["C1A", "C2A", "C3A", "C4A"],
            &["C1B", "C2B", "C3B", "C4B"],
        )),
        "DK" | "DB" => Some(tail(
            &["C1A", "C2A", "C3A", "C4A", "C5A"],
            &["C1B", "C2B", "C3B", "C4B", "C5B"],
        )),
        "DX" => Some(tail(
            &["C1A", "C2A", "C3A", "C4A", "C5A"],
            &["C1B", "C2B", "C3B", "C4B", "C5B", "C6B"],
        )),
        "DC" => Some(tail(
            &["C1A", "C2A", "C3A", "C4A", "C5A", "C6A"],
            &["C1B", "C2B", "C3B", "C4B", "C5B", "C6B"],
        )),
        "DR" => Some(tail(&["C1A", "D2A", "C3A"], &["C1B", "D2B", "C3B"])),
        "DY" | "DV" => Some(tail(
            &["C1A", "C2A", "D3A", "C4A"],
            &["C1B", "C2B", "D3B", "C4B"],
        )),
        "YO" => Some(tail(
            &["C1A", "D2A", "C3A", "C4A"],
            &["C1B", "C2B", "D3B", "C4B"],
        )),
        "OE" => Some(tail(
            &["C1A", "C2A", "D3A", "C4A", "C5A"],
            &["C1B", "D2B", "C3B", "C4B"],
        )),
        "DG" | "DE" => Some(tail(
            &["C1A", "C2A", "D3A", "C4A", "C5A"],
            &["C1B", "C2B", "D3B", "C4B", "C5B"],
        )),
        "DN" => Some(tail(
            &["C1A", "C2A", "C3A", "D4A", "C5A", "C6A"],
            &["C1B", "C2B", "C3B", "D4B", "C5B", "C6B"],
        )),
        _ => None,
    }
}

pub(crate) fn known_ltf_release_tail_names() -> &'static [&'static str] {
    &[
        "DT", "DJ", "DU", "DM", "MP", "MS", "PM", "SM", "DK", "DB", "DX", "DC", "DR", "DY", "DV",
        "YO", "OE", "DG", "DE", "DN",
    ]
}

fn tail(sn1: &[&'static str], sn2: &[&'static str]) -> [Option<&'static str>; 12] {
    let mut beads = [None; 12];
    for (idx, bead) in sn1.iter().take(6).enumerate() {
        beads[idx] = Some(*bead);
    }
    for (idx, bead) in sn2.iter().take(6).enumerate() {
        beads[6 + idx] = Some(*bead);
    }
    beads
}
