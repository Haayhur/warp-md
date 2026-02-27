pub fn normalize_element(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut chars = trimmed.chars().filter(|c| c.is_ascii_alphabetic());
    let first = chars.next()?;
    let second = chars.next();
    let mut out = String::new();
    out.push(first.to_ascii_uppercase());
    if let Some(c) = second {
        out.push(c.to_ascii_lowercase());
    }
    Some(out)
}

pub fn infer_element_from_atom_name(atom_name: &str) -> Option<String> {
    let cleaned: String = atom_name
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect();
    if cleaned.is_empty() {
        return None;
    }
    let mut chars = cleaned.chars();
    let first = chars.next().unwrap();
    let second = chars.next();
    let mut out = String::new();
    out.push(first.to_ascii_uppercase());
    if let Some(c) = second {
        out.push(c.to_ascii_lowercase());
    }
    Some(out)
}

pub fn mass_for_element(element: &str) -> f32 {
    match element {
        "H" => 1.008,
        "He" => 4.0026,
        "C" => 12.011,
        "N" => 14.007,
        "O" => 15.999,
        "F" => 18.998,
        "P" => 30.974,
        "S" => 32.06,
        "Cl" => 35.45,
        "Br" => 79.904,
        "I" => 126.904,
        "Na" => 22.99,
        "K" => 39.098,
        "Mg" => 24.305,
        "Ca" => 40.078,
        "Zn" => 65.38,
        "Fe" => 55.845,
        _ => 0.0,
    }
}

pub fn infer_element_from_mass(mass: f32) -> Option<String> {
    let table = [
        ("H", 1.008),
        ("C", 12.011),
        ("N", 14.007),
        ("O", 15.999),
        ("P", 30.974),
        ("S", 32.06),
        ("Cl", 35.45),
        ("Na", 22.99),
        ("K", 39.098),
        ("Ca", 40.078),
        ("Mg", 24.305),
    ];
    let mut best = None;
    for (elem, ref_mass) in table {
        let diff = (mass - ref_mass).abs();
        if diff < 1.0 {
            if best.map(|(_, bdiff)| diff < bdiff).unwrap_or(true) {
                best = Some((elem, diff));
            }
        }
    }
    best.map(|(elem, _)| elem.to_string())
}
