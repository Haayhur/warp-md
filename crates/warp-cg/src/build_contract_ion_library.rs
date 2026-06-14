use super::*;

pub(super) fn lookup_ion_library(name: &str) -> Option<IonLibraryEntry> {
    let trimmed = name.trim();
    match trimmed {
        "Na" => return Some(positive_ion_entry("NA", "Na+", 1)),
        "Cl" => return Some(negative_ion_entry("CL", "Cl-", -1)),
        _ => {}
    }
    let raw = name.trim().to_ascii_uppercase();
    match raw.as_str() {
        "NA+" => return Some(positive_ion_entry("NA", "Na+", 1)),
        "NC3+" | "NC3" => return Some(positive_ion_entry("NC3+", "NC3", 1)),
        "CA+" => return Some(positive_ion_entry("CA+", "CA+", 1)),
        "CL-" => return Some(negative_ion_entry("CL", "Cl-", -1)),
        _ => {}
    }
    match normalize_library_name(name).as_str() {
        "SOD" => Some(positive_ion_entry("SOD", "SOD", 1)),
        "NA" => Some(positive_ion_entry("NA", "NA", 1)),
        "NAW" => Some(positive_ion_entry("NaW", "NaW", 1)),
        "K" => Some(positive_ion_entry("K", "K", 1)),
        "MG" => Some(positive_ion_entry("MG", "MG", 2)),
        "TMA" => Some(positive_ion_entry("TMA", "TMA", 1)),
        "CA" => Some(positive_ion_entry("CA", "CA", 2)),
        "CLA" => Some(negative_ion_entry("CLA", "CLA", -1)),
        "CL" => Some(negative_ion_entry("CL", "CL", -1)),
        "CLW" => Some(negative_ion_entry("ClW", "ClW", -1)),
        "BR" => Some(negative_ion_entry("BR", "BR", -1)),
        "IOD" | "ID" => Some(negative_ion_entry("IOD", "ID", -1)),
        "ACE" => Some(negative_ion_entry("ACE", "CL", -1)),
        "BF4" => Some(negative_ion_entry("BF4", "BF4", -1)),
        "PF6" => Some(negative_ion_entry("PF6", "PF6", -1)),
        "SCN" => Some(negative_ion_entry("SCN", "SCN", -1)),
        "CLO4" | "CLO" => Some(negative_ion_entry("CLO4", "CLO", -1)),
        "NO3" => Some(negative_ion_entry("NO3", "NO3", -1)),
        _ => None,
    }
}

pub(super) fn known_cation_library_names() -> Vec<&'static str> {
    vec![
        "SOD", "NA", "Na", "NA+", "NaW", "K", "MG", "TMA", "CA", "NC3+", "CA+",
    ]
}

pub(super) fn known_anion_library_names() -> Vec<&'static str> {
    vec![
        "CLA", "CL", "Cl", "CL-", "ClW", "BR", "IOD", "ACE", "BF4", "PF6", "SCN", "CLO4", "NO3",
    ]
}

fn positive_ion_entry(
    name: &'static str,
    atom_name: &'static str,
    charge_e: i32,
) -> IonLibraryEntry {
    IonLibraryEntry {
        name,
        atom_name,
        charge_e,
        default_charge: default_cation_charge(),
    }
}

fn negative_ion_entry(
    name: &'static str,
    atom_name: &'static str,
    charge_e: i32,
) -> IonLibraryEntry {
    IonLibraryEntry {
        name,
        atom_name,
        charge_e,
        default_charge: default_anion_charge(),
    }
}
