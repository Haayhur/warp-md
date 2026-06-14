use crate::build_lipids::{tail_bead_names, LinkerKind};
use crate::build_lipids_tail_tables::{
    known_ltf_release_tail_names, known_standard_diacyl_tail_names, ltf_release_tail_bead_names,
};

pub(crate) fn diacyl_bead_names(head_group: &str, tail_group: &str) -> [Option<&'static str>; 20] {
    diacyl_bead_names_with_linker(head_group, tail_group, LinkerKind::Glycerol)
}

pub(crate) fn diacyl_bead_names_with_linker(
    head_group: &str,
    tail_group: &str,
    linker: LinkerKind,
) -> [Option<&'static str>; 20] {
    let tail = match tail_group {
        "PO" | "SO" => [
            Some("C1A"),
            Some("D2A"),
            Some("C3A"),
            Some("C4A"),
            None,
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "CC" => [
            Some("C1A"),
            Some("C2A"),
            Some("C3A"),
            None,
            None,
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            None,
            None,
            None,
        ],
        "DO" => [
            Some("C1A"),
            Some("D2A"),
            Some("C3A"),
            Some("C4A"),
            None,
            None,
            Some("C1B"),
            Some("D2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "DL" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("C4A"),
            None,
            None,
            Some("C1B"),
            Some("D2B"),
            Some("D3B"),
            Some("C4B"),
            None,
            None,
        ],
        "DF" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            None,
            None,
            Some("C1B"),
            Some("D2B"),
            Some("D3B"),
            Some("D4B"),
            None,
            None,
        ],
        "DA" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            Some("C5A"),
            None,
            Some("C1B"),
            Some("D2B"),
            Some("D3B"),
            Some("D4B"),
            Some("C5B"),
            None,
        ],
        "DD" => [
            Some("D1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            Some("D5A"),
            None,
            Some("D1B"),
            Some("D2B"),
            Some("D3B"),
            Some("D4B"),
            Some("D5B"),
            None,
        ],
        "PY" => [
            Some("C1A"),
            Some("C2A"),
            Some("D3A"),
            Some("C4A"),
            None,
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "PE" => [
            Some("C1A"),
            Some("C2A"),
            Some("D3A"),
            Some("C4A"),
            Some("C5A"),
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "PL" | "SL" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("C4A"),
            None,
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "PI" => [
            Some("C1A"),
            Some("C2A"),
            Some("D3A"),
            Some("D4A"),
            Some("C5A"),
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "PF" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            None,
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "PQ" | "PA" | "SA" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            Some("C5A"),
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "PD" | "SD" => [
            Some("D1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            Some("D5A"),
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "OL" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("C4A"),
            None,
            None,
            Some("C1B"),
            Some("D2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "OD" => [
            Some("D1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            Some("D5A"),
            None,
            Some("C1B"),
            Some("D2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "LF" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            None,
            None,
            Some("C1B"),
            Some("D2B"),
            Some("D3B"),
            Some("C4B"),
            None,
            None,
        ],
        _ => ltf_release_tail_bead_names(tail_group)
            .or_else(|| generated_tail_bead_names(tail_group))
            .unwrap_or([
                Some("C1A"),
                Some("C2A"),
                Some("C3A"),
                Some("C4A"),
                None,
                None,
                Some("C1B"),
                Some("C2B"),
                Some("C3B"),
                Some("C4B"),
                None,
                None,
            ]),
    };
    diacyl_bead_names_from_tail(head_group, tail, linker)
}

pub(crate) fn diacyl_bead_names_from_tail(
    head_group: &str,
    tail: [Option<&'static str>; 12],
    linker: LinkerKind,
) -> [Option<&'static str>; 20] {
    let head = match head_group {
        "PA" => [None, None, None, None, None, Some("PO4")],
        "PC" => [None, None, None, Some("NC3"), None, Some("PO4")],
        "PE" => [None, None, None, Some("NH3"), None, Some("PO4")],
        "PG" => [None, None, None, Some("GL0"), None, Some("PO4")],
        "PS" => [None, None, None, Some("CNO"), None, Some("PO4")],
        "DG" => [None, None, None, None, None, Some("OH")],
        "PI" => [
            Some("C1"),
            Some("C2"),
            Some("C3"),
            Some("C4"),
            None,
            Some("PO4"),
        ],
        _ => [None, None, None, None, None, Some("PO4")],
    };
    let linker_beads = linker_bead_names(linker);
    [
        head[0],
        head[1],
        head[2],
        head[3],
        head[4],
        head[5],
        linker_beads[0],
        linker_beads[1],
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

pub(crate) fn linker_bead_names(linker: LinkerKind) -> [Option<&'static str>; 2] {
    match linker {
        LinkerKind::Glycerol => [Some("GL1"), Some("GL2")],
        LinkerKind::Ether => [Some("ET1"), Some("ET2")],
        LinkerKind::Plasmalogen => [Some("GL1"), Some("PL2")],
    }
}

pub(crate) fn plasmalogen_tail_bead_names(tail_group: &str) -> [Option<&'static str>; 12] {
    match tail_group {
        "O" => [
            Some("C1A"),
            Some("D2A"),
            Some("C3A"),
            Some("C4A"),
            None,
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "A" => [
            Some("C1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            Some("C5A"),
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        "D" => [
            Some("D1A"),
            Some("D2A"),
            Some("D3A"),
            Some("D4A"),
            Some("D5A"),
            None,
            Some("C1B"),
            Some("C2B"),
            Some("C3B"),
            Some("C4B"),
            None,
            None,
        ],
        _ => tail_bead_names("PO"),
    }
}

pub(crate) fn parse_linked_diacyl_name(
    name: &str,
) -> Option<(&str, &'static str, &str, LinkerKind)> {
    let (base_name, linker) = split_linker_suffix(name);
    parse_diacyl_name(base_name).and_then(|(head_group, tail_group)| {
        if matches!(linker, LinkerKind::Ether) && head_group == "DG" {
            None
        } else {
            Some((base_name, head_group, tail_group, linker))
        }
    })
}

pub(crate) fn split_linker_suffix(name: &str) -> (&str, LinkerKind) {
    if let Some(base_name) = name.strip_suffix(".GL") {
        (base_name, LinkerKind::Glycerol)
    } else if let Some(base_name) = name.strip_suffix(".ET") {
        (base_name, LinkerKind::Ether)
    } else {
        (name, LinkerKind::Glycerol)
    }
}

pub(crate) fn parse_diacyl_name(name: &str) -> Option<(&'static str, &str)> {
    for head_group in known_diacyl_heads() {
        if let Some(tail_group) = name.strip_suffix(head_group) {
            if known_standard_diacyl_tail_names().contains(&tail_group)
                || ltf_release_tail_bead_names(tail_group).is_some()
                || generated_tail_bead_names(tail_group).is_some()
            {
                return Some((head_group, tail_group));
            }
        }
    }
    None
}

pub(crate) fn parse_plasmalogen_name(name: &str) -> Option<(&'static str, &'static str)> {
    for tail_group in known_plasmalogen_tails() {
        if let Some(head_name) = name
            .strip_prefix(tail_group)
            .and_then(|name| name.strip_prefix("PL"))
        {
            for head_group in known_plasmalogen_heads() {
                if head_name == *head_group {
                    return Some((head_group, tail_group));
                }
            }
        }
    }
    None
}

pub(crate) fn known_diacyl_heads() -> &'static [&'static str] {
    &[
        "PA", "PC", "PE", "PG", "PS", "DG", "PI", "P1", "P2", "P3", "P4", "P5", "P6", "P7",
    ]
}

pub(crate) fn expanded_diacyl_tail_names() -> Vec<String> {
    let mut tails = known_standard_diacyl_tail_names()
        .iter()
        .map(|tail| (*tail).to_string())
        .collect::<std::collections::BTreeSet<_>>();
    for sn1 in coby_tail_code_names() {
        for sn2 in coby_tail_code_names() {
            let name = if sn1 == sn2 {
                format!("D{sn1}")
            } else {
                format!("{sn2}{sn1}")
            };
            tails.insert(name);
        }
    }
    for tail in known_ltf_release_tail_names() {
        tails.insert((*tail).to_string());
    }
    tails.into_iter().collect()
}
pub(crate) fn generated_tail_bead_names(tail_group: &str) -> Option<[Option<&'static str>; 12]> {
    let mut chars = tail_group.chars();
    let first = chars.next()?;
    let second = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    let (sn1, sn2) = if first == 'D' {
        (coby_tail_code(second)?, coby_tail_code(second)?)
    } else {
        (coby_tail_code(second)?, coby_tail_code(first)?)
    };
    let mut tail = [None; 12];
    fill_tail_chain(&mut tail, 0, sn1, 'A');
    fill_tail_chain(&mut tail, 6, sn2, 'B');
    Some(tail)
}
pub(crate) fn generated_sphingomyelin_tail_bead_names(
    tail_group: &str,
) -> Option<[Option<&'static str>; 12]> {
    let mut chars = tail_group.chars();
    let first = chars.next()?;
    let second = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    let (sn1, sn2) = if first == 'D' {
        let code = coby_sphingomyelin_sn1_code(second)?;
        (code, coby_tail_code(second)?)
    } else {
        (coby_sphingomyelin_sn1_code(second)?, coby_tail_code(first)?)
    };
    let mut tail = [None; 12];
    fill_tail_chain(&mut tail, 0, sn1, 'A');
    fill_tail_chain(&mut tail, 6, sn2, 'B');
    Some(tail)
}
fn fill_tail_chain(
    tail: &mut [Option<&'static str>; 12],
    offset: usize,
    code: &'static str,
    chain: char,
) {
    for (idx, bead_code) in code.chars().take(6).enumerate() {
        tail[offset + idx] = tail_bead_name(bead_code, idx + 1, chain);
    }
}
fn tail_bead_name(code: char, idx: usize, chain: char) -> Option<&'static str> {
    match (code, idx, chain) {
        ('C', 1, 'A') => Some("C1A"),
        ('C', 2, 'A') => Some("C2A"),
        ('C', 3, 'A') => Some("C3A"),
        ('C', 4, 'A') => Some("C4A"),
        ('C', 5, 'A') => Some("C5A"),
        ('C', 6, 'A') => Some("C6A"),
        ('D', 1, 'A') => Some("D1A"),
        ('D', 2, 'A') => Some("D2A"),
        ('D', 3, 'A') => Some("D3A"),
        ('D', 4, 'A') => Some("D4A"),
        ('D', 5, 'A') => Some("D5A"),
        ('D', 6, 'A') => Some("D6A"),
        ('T', 1, 'A') => Some("T1A"),
        ('T', 2, 'A') => Some("T2A"),
        ('T', 3, 'A') => Some("T3A"),
        ('T', 4, 'A') => Some("T4A"),
        ('T', 5, 'A') => Some("T5A"),
        ('T', 6, 'A') => Some("T6A"),
        ('C', 1, 'B') => Some("C1B"),
        ('C', 2, 'B') => Some("C2B"),
        ('C', 3, 'B') => Some("C3B"),
        ('C', 4, 'B') => Some("C4B"),
        ('C', 5, 'B') => Some("C5B"),
        ('C', 6, 'B') => Some("C6B"),
        ('D', 1, 'B') => Some("D1B"),
        ('D', 2, 'B') => Some("D2B"),
        ('D', 3, 'B') => Some("D3B"),
        ('D', 4, 'B') => Some("D4B"),
        ('D', 5, 'B') => Some("D5B"),
        ('D', 6, 'B') => Some("D6B"),
        ('T', 1, 'B') => Some("T1B"),
        ('T', 2, 'B') => Some("T2B"),
        ('T', 3, 'B') => Some("T3B"),
        ('T', 4, 'B') => Some("T4B"),
        ('T', 5, 'B') => Some("T5B"),
        ('T', 6, 'B') => Some("T6B"),
        _ => None,
    }
}
pub(crate) fn coby_tail_code_names() -> &'static [char] {
    &[
        'C', 'T', 'L', 'M', 'P', 'B', 'X', 'Y', 'O', 'V', 'G', 'N', 'I', 'F', 'E', 'Q', 'A', 'U',
        'R', 'J',
    ]
}
fn coby_tail_code(name: char) -> Option<&'static str> {
    match name {
        'C' => Some("C"),
        'T' => Some("CC"),
        'L' | 'M' => Some("CCC"),
        'P' => Some("CCCC"),
        'B' => Some("CCCCC"),
        'X' => Some("CCCCCC"),
        'Y' => Some("CDC"),
        'O' => Some("CDCC"),
        'V' => Some("CCDC"),
        'G' => Some("CCDCC"),
        'N' => Some("CCCDCC"),
        'I' => Some("CDDC"),
        'F' => Some("CDDD"),
        'E' => Some("CCDDC"),
        'Q' => Some("CDDDC"),
        'A' => Some("DDDDC"),
        'U' => Some("DDDDD"),
        'R' => Some("DDDDDD"),
        'J' => Some("TCCC"),
        _ => None,
    }
}
fn coby_sphingomyelin_sn1_code(name: char) -> Option<&'static str> {
    match name {
        'P' => Some("TCC"),
        'B' => Some("TCCC"),
        'X' => Some("TCCCC"),
        _ => None,
    }
}
pub(crate) fn known_plasmalogen_heads() -> &'static [&'static str] {
    &["C", "E"]
}
pub(crate) fn known_plasmalogen_tails() -> &'static [&'static str] {
    &["O", "A", "D"]
}
