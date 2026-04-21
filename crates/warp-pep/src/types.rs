//! Shared peptide residue type definitions.

/// Three-letter residue name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResName {
    GLY,
    ALA,
    SER,
    CYS,
    VAL,
    ILE,
    LEU,
    THR,
    ARG,
    LYS,
    ASP,
    GLU,
    ASN,
    GLN,
    MET,
    HIS,
    PRO,
    PHE,
    TYR,
    TRP,
    /// Acetyl N-terminal cap (not a standard amino acid).
    ACE,
    /// N-methylamide C-terminal cap (not a standard amino acid).
    NME,
}

impl ResName {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::GLY => "GLY",
            Self::ALA => "ALA",
            Self::SER => "SER",
            Self::CYS => "CYS",
            Self::VAL => "VAL",
            Self::ILE => "ILE",
            Self::LEU => "LEU",
            Self::THR => "THR",
            Self::ARG => "ARG",
            Self::LYS => "LYS",
            Self::ASP => "ASP",
            Self::GLU => "GLU",
            Self::ASN => "ASN",
            Self::GLN => "GLN",
            Self::MET => "MET",
            Self::HIS => "HIS",
            Self::PRO => "PRO",
            Self::PHE => "PHE",
            Self::TYR => "TYR",
            Self::TRP => "TRP",
            Self::ACE => "ACE",
            Self::NME => "NME",
        }
    }

    pub fn from_one_letter(c: char) -> Option<Self> {
        match c {
            'G' => Some(Self::GLY),
            'A' => Some(Self::ALA),
            'S' => Some(Self::SER),
            'C' => Some(Self::CYS),
            'V' => Some(Self::VAL),
            'I' => Some(Self::ILE),
            'L' => Some(Self::LEU),
            'T' => Some(Self::THR),
            'R' => Some(Self::ARG),
            'K' => Some(Self::LYS),
            'D' => Some(Self::ASP),
            'E' => Some(Self::GLU),
            'N' => Some(Self::ASN),
            'Q' => Some(Self::GLN),
            'M' => Some(Self::MET),
            'H' => Some(Self::HIS),
            'P' => Some(Self::PRO),
            'F' => Some(Self::PHE),
            'Y' => Some(Self::TYR),
            'W' => Some(Self::TRP),
            _ => None,
        }
    }

    pub fn to_one_letter(self) -> char {
        match self {
            Self::GLY => 'G',
            Self::ALA => 'A',
            Self::SER => 'S',
            Self::CYS => 'C',
            Self::VAL => 'V',
            Self::ILE => 'I',
            Self::LEU => 'L',
            Self::THR => 'T',
            Self::ARG => 'R',
            Self::LYS => 'K',
            Self::ASP => 'D',
            Self::GLU => 'E',
            Self::ASN => 'N',
            Self::GLN => 'Q',
            Self::MET => 'M',
            Self::HIS => 'H',
            Self::PRO => 'P',
            Self::PHE => 'F',
            Self::TYR => 'Y',
            Self::TRP => 'W',
            Self::ACE | Self::NME => 'X',
        }
    }

    pub fn from_three_letter(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "GLY" => Some(Self::GLY),
            "ALA" => Some(Self::ALA),
            "SER" => Some(Self::SER),
            "CYS" => Some(Self::CYS),
            "VAL" => Some(Self::VAL),
            "ILE" => Some(Self::ILE),
            "LEU" => Some(Self::LEU),
            "THR" => Some(Self::THR),
            "ARG" => Some(Self::ARG),
            "LYS" => Some(Self::LYS),
            "ASP" => Some(Self::ASP),
            "GLU" => Some(Self::GLU),
            "ASN" => Some(Self::ASN),
            "GLN" => Some(Self::GLN),
            "MET" => Some(Self::MET),
            "HIS" => Some(Self::HIS),
            "PRO" => Some(Self::PRO),
            "PHE" => Some(Self::PHE),
            "TYR" => Some(Self::TYR),
            "TRP" => Some(Self::TRP),
            "ACE" => Some(Self::ACE),
            "NME" => Some(Self::NME),
            "CYX" => Some(Self::CYS),
            "HID" | "HIE" | "HIP" => Some(Self::HIS),
            "ASH" => Some(Self::ASP),
            "GLH" => Some(Self::GLU),
            "LYN" => Some(Self::LYS),
            _ => None,
        }
    }

    /// Returns true if this is a terminal cap (ACE or NME), not a standard amino acid.
    pub fn is_cap(self) -> bool {
        matches!(self, Self::ACE | Self::NME)
    }
}

/// Amber force field residue name variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmberVariant {
    /// Disulfide-bonded cysteine (no SG hydrogen).
    CYX,
    /// Histidine protonated at Nδ only.
    HID,
    /// Histidine protonated at Nε only (most common).
    HIE,
    /// Histidine doubly protonated (charged +1).
    HIP,
    /// Protonated aspartate.
    ASH,
    /// Protonated glutamate.
    GLH,
    /// Neutral (deprotonated) lysine.
    LYN,
}

impl AmberVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CYX => "CYX",
            Self::HID => "HID",
            Self::HIE => "HIE",
            Self::HIP => "HIP",
            Self::ASH => "ASH",
            Self::GLH => "GLH",
            Self::LYN => "LYN",
        }
    }

    /// Canonical residue this variant belongs to.
    pub fn canonical(self) -> ResName {
        match self {
            Self::CYX => ResName::CYS,
            Self::HID | Self::HIE | Self::HIP => ResName::HIS,
            Self::ASH => ResName::ASP,
            Self::GLH => ResName::GLU,
            Self::LYN => ResName::LYS,
        }
    }
}

/// Parse an Amber-convention residue name into canonical type + optional variant.
pub fn parse_amber_name(s: &str) -> Option<(ResName, Option<AmberVariant>)> {
    match s.to_uppercase().as_str() {
        "CYX" => Some((ResName::CYS, Some(AmberVariant::CYX))),
        "HID" => Some((ResName::HIS, Some(AmberVariant::HID))),
        "HIE" => Some((ResName::HIS, Some(AmberVariant::HIE))),
        "HIP" => Some((ResName::HIS, Some(AmberVariant::HIP))),
        "ASH" => Some((ResName::ASP, Some(AmberVariant::ASH))),
        "GLH" => Some((ResName::GLU, Some(AmberVariant::GLH))),
        "LYN" => Some((ResName::LYS, Some(AmberVariant::LYN))),
        _ => ResName::from_three_letter(s).map(|r| (r, None)),
    }
}
