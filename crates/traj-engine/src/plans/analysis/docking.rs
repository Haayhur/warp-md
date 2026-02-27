use std::collections::HashMap;

use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};

const INTERACTION_HYDROGEN_BOND: u8 = 1;
const INTERACTION_HYDROPHOBIC: u8 = 2;
const INTERACTION_CLOSE_CONTACT: u8 = 3;
const INTERACTION_CLASH: u8 = 4;
const INTERACTION_SALT_BRIDGE: u8 = 5;
const INTERACTION_HALOGEN_BOND: u8 = 6;
const INTERACTION_METAL_COORDINATION: u8 = 7;
const INTERACTION_CATION_PI: u8 = 8;
const INTERACTION_PI_PI_STACKING: u8 = 9;

#[derive(Clone, Copy, Debug, Default)]
struct AtomFlags {
    donor: bool,
    acceptor: bool,
    hydrophobic: bool,
    halogen: bool,
    metal: bool,
    positive: bool,
    negative: bool,
    aromatic: bool,
    hydrogen: bool,
    resid: i32,
    chain_id: u32,
}

pub struct DockingAnalysisPlan {
    receptor: Selection,
    ligand: Selection,
    close_contact_cutoff: f64,
    hydrophobic_cutoff: f64,
    hydrogen_bond_cutoff: f64,
    clash_cutoff: f64,
    salt_bridge_cutoff: f64,
    halogen_bond_cutoff: f64,
    metal_coordination_cutoff: f64,
    cation_pi_cutoff: f64,
    pi_pi_cutoff: f64,
    hbond_min_angle_deg: f64,
    donor_hydrogen_cutoff: f64,
    allow_missing_hydrogen: bool,
    length_scale: f64,
    max_events_per_frame: usize,
    atom_flags: Vec<AtomFlags>,
    hydrogens_by_residue: HashMap<(u32, i32), Vec<usize>>,
    rows: Vec<f32>,
    frame_cursor: usize,
}

impl DockingAnalysisPlan {
    pub fn new(
        receptor: Selection,
        ligand: Selection,
        close_contact_cutoff: f64,
        hydrophobic_cutoff: f64,
        hydrogen_bond_cutoff: f64,
        clash_cutoff: f64,
    ) -> TrajResult<Self> {
        if receptor.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "docking receptor selection must be non-empty".into(),
            ));
        }
        if ligand.indices.is_empty() {
            return Err(TrajError::Mismatch(
                "docking ligand selection must be non-empty".into(),
            ));
        }
        if close_contact_cutoff <= 0.0 || !close_contact_cutoff.is_finite() {
            return Err(TrajError::Parse(
                "docking close_contact_cutoff must be finite and > 0".into(),
            ));
        }
        if hydrophobic_cutoff <= 0.0 || !hydrophobic_cutoff.is_finite() {
            return Err(TrajError::Parse(
                "docking hydrophobic_cutoff must be finite and > 0".into(),
            ));
        }
        if hydrogen_bond_cutoff <= 0.0 || !hydrogen_bond_cutoff.is_finite() {
            return Err(TrajError::Parse(
                "docking hydrogen_bond_cutoff must be finite and > 0".into(),
            ));
        }
        if clash_cutoff < 0.0 || !clash_cutoff.is_finite() {
            return Err(TrajError::Parse(
                "docking clash_cutoff must be finite and >= 0".into(),
            ));
        }
        if hydrophobic_cutoff > close_contact_cutoff {
            return Err(TrajError::Parse(
                "docking hydrophobic_cutoff must be <= close_contact_cutoff".into(),
            ));
        }
        if hydrogen_bond_cutoff > close_contact_cutoff {
            return Err(TrajError::Parse(
                "docking hydrogen_bond_cutoff must be <= close_contact_cutoff".into(),
            ));
        }
        if clash_cutoff > close_contact_cutoff {
            return Err(TrajError::Parse(
                "docking clash_cutoff must be <= close_contact_cutoff".into(),
            ));
        }
        Ok(Self {
            receptor,
            ligand,
            close_contact_cutoff,
            hydrophobic_cutoff,
            hydrogen_bond_cutoff,
            clash_cutoff,
            salt_bridge_cutoff: 5.5,
            halogen_bond_cutoff: 5.5,
            metal_coordination_cutoff: 3.5,
            cation_pi_cutoff: 6.0,
            pi_pi_cutoff: 7.5,
            hbond_min_angle_deg: 120.0,
            donor_hydrogen_cutoff: 1.25,
            allow_missing_hydrogen: true,
            length_scale: 1.0,
            max_events_per_frame: 20_000,
            atom_flags: Vec::new(),
            hydrogens_by_residue: HashMap::new(),
            rows: Vec::new(),
            frame_cursor: 0,
        })
    }

    pub fn with_max_events_per_frame(mut self, value: usize) -> Self {
        self.max_events_per_frame = value;
        self
    }

    pub fn with_length_scale(mut self, value: f64) -> Self {
        self.length_scale = value;
        self
    }

    pub fn with_salt_bridge_cutoff(mut self, value: f64) -> Self {
        self.salt_bridge_cutoff = value;
        self
    }

    pub fn with_halogen_bond_cutoff(mut self, value: f64) -> Self {
        self.halogen_bond_cutoff = value;
        self
    }

    pub fn with_metal_coordination_cutoff(mut self, value: f64) -> Self {
        self.metal_coordination_cutoff = value;
        self
    }

    pub fn with_cation_pi_cutoff(mut self, value: f64) -> Self {
        self.cation_pi_cutoff = value;
        self
    }

    pub fn with_pi_pi_cutoff(mut self, value: f64) -> Self {
        self.pi_pi_cutoff = value;
        self
    }

    pub fn with_hbond_min_angle_deg(mut self, value: f64) -> Self {
        self.hbond_min_angle_deg = value;
        self
    }

    pub fn with_donor_hydrogen_cutoff(mut self, value: f64) -> Self {
        self.donor_hydrogen_cutoff = value;
        self
    }

    pub fn with_allow_missing_hydrogen(mut self, value: bool) -> Self {
        self.allow_missing_hydrogen = value;
        self
    }
}

impl Plan for DockingAnalysisPlan {
    fn name(&self) -> &'static str {
        "docking_analysis"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if self.length_scale <= 0.0 || !self.length_scale.is_finite() {
            return Err(TrajError::Parse(
                "docking length_scale must be finite and > 0".into(),
            ));
        }
        if self.max_events_per_frame == 0 {
            return Err(TrajError::Parse(
                "docking max_events_per_frame must be >= 1".into(),
            ));
        }
        validate_positive_cutoff(self.salt_bridge_cutoff, "salt_bridge_cutoff")?;
        validate_positive_cutoff(self.halogen_bond_cutoff, "halogen_bond_cutoff")?;
        validate_positive_cutoff(self.metal_coordination_cutoff, "metal_coordination_cutoff")?;
        validate_positive_cutoff(self.cation_pi_cutoff, "cation_pi_cutoff")?;
        validate_positive_cutoff(self.pi_pi_cutoff, "pi_pi_cutoff")?;
        validate_positive_cutoff(self.donor_hydrogen_cutoff, "donor_hydrogen_cutoff")?;
        if !self.hbond_min_angle_deg.is_finite()
            || self.hbond_min_angle_deg <= 0.0
            || self.hbond_min_angle_deg > 180.0
        {
            return Err(TrajError::Parse(
                "docking hbond_min_angle_deg must be finite and in (0, 180]".into(),
            ));
        }
        let n_atoms = system.n_atoms();
        if self
            .receptor
            .indices
            .iter()
            .any(|&idx| idx as usize >= n_atoms)
            || self
                .ligand
                .indices
                .iter()
                .any(|&idx| idx as usize >= n_atoms)
        {
            return Err(TrajError::Mismatch(
                "docking selection index out of system bounds".into(),
            ));
        }
        self.atom_flags = build_atom_flags(system);
        self.hydrogens_by_residue.clear();
        for (idx, flags) in self.atom_flags.iter().copied().enumerate() {
            if flags.hydrogen {
                self.hydrogens_by_residue
                    .entry((flags.chain_id, flags.resid))
                    .or_default()
                    .push(idx);
            }
        }
        self.rows.clear();
        self.frame_cursor = 0;
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let close2 = self.close_contact_cutoff * self.close_contact_cutoff;
        let hydrophobic2 = self.hydrophobic_cutoff * self.hydrophobic_cutoff;
        let hbond2 = self.hydrogen_bond_cutoff * self.hydrogen_bond_cutoff;
        let clash2 = self.clash_cutoff * self.clash_cutoff;
        let salt2 = self.salt_bridge_cutoff * self.salt_bridge_cutoff;
        let halogen2 = self.halogen_bond_cutoff * self.halogen_bond_cutoff;
        let metal2 = self.metal_coordination_cutoff * self.metal_coordination_cutoff;
        let cation_pi2 = self.cation_pi_cutoff * self.cation_pi_cutoff;
        let pi_pi2 = self.pi_pi_cutoff * self.pi_pi_cutoff;
        let donor_h2 = self.donor_hydrogen_cutoff * self.donor_hydrogen_cutoff;
        let max2 = close2
            .max(hydrophobic2)
            .max(hbond2)
            .max(clash2)
            .max(salt2)
            .max(halogen2)
            .max(metal2)
            .max(cation_pi2)
            .max(pi_pi2);
        let n_atoms = chunk.n_atoms;
        for local_frame in 0..chunk.n_frames {
            let base = local_frame * n_atoms;
            let frame_index = self.frame_cursor + local_frame;
            let mut frame_events = 0usize;
            for &receptor_idx_u32 in self.receptor.indices.iter() {
                let receptor_idx = receptor_idx_u32 as usize;
                if receptor_idx >= n_atoms {
                    return Err(TrajError::Mismatch(
                        "docking receptor index out of trajectory bounds".into(),
                    ));
                }
                let receptor = chunk.coords[base + receptor_idx];
                for &ligand_idx_u32 in self.ligand.indices.iter() {
                    let ligand_idx = ligand_idx_u32 as usize;
                    if ligand_idx >= n_atoms {
                        return Err(TrajError::Mismatch(
                            "docking ligand index out of trajectory bounds".into(),
                        ));
                    }
                    let ligand = chunk.coords[base + ligand_idx];
                    let r2 = distance_sq(receptor, ligand, self.length_scale);
                    if r2 > max2 {
                        continue;
                    }
                    let receptor_flags = self
                        .atom_flags
                        .get(receptor_idx)
                        .copied()
                        .unwrap_or_default();
                    let ligand_flags = self.atom_flags.get(ligand_idx).copied().unwrap_or_default();
                    let hbond_score = if r2 <= hbond2 {
                        hydrogen_bond_score(
                            receptor_idx,
                            receptor_flags,
                            ligand_idx,
                            ligand_flags,
                            base,
                            &chunk.coords,
                            &self.atom_flags,
                            &self.hydrogens_by_residue,
                            donor_h2,
                            self.hbond_min_angle_deg,
                            self.length_scale,
                            self.allow_missing_hydrogen,
                        )
                    } else {
                        None
                    };
                    let code = classify_interaction(
                        receptor_flags,
                        ligand_flags,
                        r2,
                        close2,
                        hydrophobic2,
                        hbond2,
                        clash2,
                        salt2,
                        halogen2,
                        metal2,
                        cation_pi2,
                        pi_pi2,
                        hbond_score.is_some(),
                    );
                    if code == 0 {
                        continue;
                    }
                    frame_events += 1;
                    if frame_events > self.max_events_per_frame {
                        return Err(TrajError::Mismatch(
                            "docking event count exceeded max_events_per_frame".into(),
                        ));
                    }
                    let dist = r2.sqrt();
                    let strength = interaction_strength(
                        code,
                        dist,
                        self.close_contact_cutoff,
                        self.hydrophobic_cutoff,
                        self.hydrogen_bond_cutoff,
                        self.clash_cutoff,
                        self.salt_bridge_cutoff,
                        self.halogen_bond_cutoff,
                        self.metal_coordination_cutoff,
                        self.cation_pi_cutoff,
                        self.pi_pi_cutoff,
                        hbond_score,
                    );
                    self.rows.push(frame_index as f32);
                    self.rows.push(receptor_idx as f32);
                    self.rows.push(ligand_idx as f32);
                    self.rows.push(code as f32);
                    self.rows.push(dist as f32);
                    self.rows.push(strength as f32);
                }
            }
        }
        self.frame_cursor += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: self.rows.clone(),
            rows: self.rows.len() / 6,
            cols: 6,
        })
    }
}

fn validate_positive_cutoff(value: f64, name: &str) -> TrajResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return Err(TrajError::Parse(format!(
            "docking {name} must be finite and > 0"
        )));
    }
    Ok(())
}

fn build_atom_flags(system: &System) -> Vec<AtomFlags> {
    let mut flags = Vec::with_capacity(system.n_atoms());
    for atom_idx in 0..system.n_atoms() {
        let name_id = system.atoms.name_id.get(atom_idx).copied().unwrap_or(0);
        let element_id = system.atoms.element_id.get(atom_idx).copied().unwrap_or(0);
        let resname_id = system.atoms.resname_id.get(atom_idx).copied().unwrap_or(0);
        let name = system
            .interner
            .resolve(name_id)
            .unwrap_or("")
            .to_ascii_uppercase();
        let element = system
            .interner
            .resolve(element_id)
            .unwrap_or("")
            .to_ascii_uppercase();
        let resname = system
            .interner
            .resolve(resname_id)
            .unwrap_or("")
            .to_ascii_uppercase();
        let symbol = infer_element_symbol(&name, &element);
        let hydrogen = symbol == "H";
        let halogen = matches!(symbol.as_str(), "F" | "CL" | "BR" | "I");
        let metal = is_metal_symbol(&symbol);
        let donor = matches!(symbol.as_str(), "N" | "O" | "S");
        let acceptor = matches!(symbol.as_str(), "N" | "O" | "S");
        let hydrophobic = symbol == "C";
        let aromatic = is_aromatic(&resname, &name, &symbol);
        let positive = is_positive(&resname, &name, &symbol) || metal;
        let negative = is_negative(&resname, &name, &symbol);
        flags.push(AtomFlags {
            donor,
            acceptor,
            hydrophobic,
            halogen,
            metal,
            positive,
            negative,
            aromatic,
            hydrogen,
            resid: *system.atoms.resid.get(atom_idx).unwrap_or(&0),
            chain_id: *system.atoms.chain_id.get(atom_idx).unwrap_or(&0),
        });
    }
    flags
}

fn infer_element_symbol(name: &str, element: &str) -> String {
    let compact_element: String = element
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect();
    if !compact_element.is_empty() {
        let upper = compact_element.to_ascii_uppercase();
        if upper.starts_with("CL") {
            return "CL".to_string();
        }
        if upper.starts_with("BR") {
            return "BR".to_string();
        }
        let first = upper.chars().next().unwrap_or('C');
        return first.to_string();
    }
    let upper_name = name.to_ascii_uppercase();
    if upper_name.starts_with("CL") {
        return "CL".to_string();
    }
    if upper_name.starts_with("BR") {
        return "BR".to_string();
    }
    let first = upper_name
        .chars()
        .find(|c| c.is_ascii_alphabetic())
        .unwrap_or('C');
    first.to_string()
}

fn is_metal_symbol(symbol: &str) -> bool {
    matches!(
        symbol,
        "LI" | "NA"
            | "K"
            | "RB"
            | "CS"
            | "MG"
            | "CA"
            | "SR"
            | "BA"
            | "ZN"
            | "MN"
            | "FE"
            | "CO"
            | "NI"
            | "CU"
            | "CD"
    )
}

fn is_aromatic(resname: &str, atom_name: &str, symbol: &str) -> bool {
    if !matches!(symbol, "C" | "N") {
        return false;
    }
    if matches!(
        resname,
        "PHE" | "TYR" | "TRP" | "HIS" | "HID" | "HIE" | "HIP"
    ) {
        return true;
    }
    starts_with_any(atom_name, &["CG", "CD", "CE", "CZ", "CH", "ND", "NE"])
}

fn is_positive(resname: &str, atom_name: &str, symbol: &str) -> bool {
    if matches!(resname, "LYS" | "LYN") && starts_with_any(atom_name, &["NZ"]) {
        return true;
    }
    if resname == "ARG" && starts_with_any(atom_name, &["NE", "NH"]) {
        return true;
    }
    if matches!(resname, "HIS" | "HID" | "HIE" | "HIP") && starts_with_any(atom_name, &["ND", "NE"])
    {
        return true;
    }
    matches!(symbol, "N") && atom_name.ends_with('+')
}

fn is_negative(resname: &str, atom_name: &str, symbol: &str) -> bool {
    if resname == "ASP" && starts_with_any(atom_name, &["OD"]) {
        return true;
    }
    if resname == "GLU" && starts_with_any(atom_name, &["OE"]) {
        return true;
    }
    if atom_name == "OXT" {
        return true;
    }
    matches!(symbol, "O") && atom_name.ends_with('-')
}

fn starts_with_any(value: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|prefix| value.starts_with(prefix))
}

fn distance_sq(a: [f32; 4], b: [f32; 4], length_scale: f64) -> f64 {
    let dx = (a[0] as f64 - b[0] as f64) * length_scale;
    let dy = (a[1] as f64 - b[1] as f64) * length_scale;
    let dz = (a[2] as f64 - b[2] as f64) * length_scale;
    dx * dx + dy * dy + dz * dz
}

#[allow(clippy::too_many_arguments)]
fn hydrogen_bond_score(
    receptor_idx: usize,
    receptor_flags: AtomFlags,
    ligand_idx: usize,
    ligand_flags: AtomFlags,
    base: usize,
    coords: &[[f32; 4]],
    atom_flags: &[AtomFlags],
    hydrogens_by_residue: &HashMap<(u32, i32), Vec<usize>>,
    donor_h_cutoff2: f64,
    hbond_min_angle_deg: f64,
    length_scale: f64,
    allow_missing_hydrogen: bool,
) -> Option<f64> {
    if receptor_flags.donor && ligand_flags.acceptor {
        return donor_acceptor_hbond_score(
            receptor_idx,
            ligand_idx,
            base,
            coords,
            atom_flags,
            hydrogens_by_residue,
            donor_h_cutoff2,
            hbond_min_angle_deg,
            length_scale,
            allow_missing_hydrogen,
        );
    }
    if ligand_flags.donor && receptor_flags.acceptor {
        return donor_acceptor_hbond_score(
            ligand_idx,
            receptor_idx,
            base,
            coords,
            atom_flags,
            hydrogens_by_residue,
            donor_h_cutoff2,
            hbond_min_angle_deg,
            length_scale,
            allow_missing_hydrogen,
        );
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn donor_acceptor_hbond_score(
    donor_idx: usize,
    acceptor_idx: usize,
    base: usize,
    coords: &[[f32; 4]],
    atom_flags: &[AtomFlags],
    hydrogens_by_residue: &HashMap<(u32, i32), Vec<usize>>,
    donor_h_cutoff2: f64,
    hbond_min_angle_deg: f64,
    length_scale: f64,
    allow_missing_hydrogen: bool,
) -> Option<f64> {
    let donor_flags = atom_flags.get(donor_idx).copied().unwrap_or_default();
    let donor = coords[base + donor_idx];
    let acceptor = coords[base + acceptor_idx];
    let key = (donor_flags.chain_id, donor_flags.resid);
    let mut found_h = false;
    let mut best_angle = f64::NEG_INFINITY;
    if let Some(hydrogen_indices) = hydrogens_by_residue.get(&key) {
        for &h_idx in hydrogen_indices.iter() {
            if h_idx == donor_idx || h_idx == acceptor_idx {
                continue;
            }
            let h = coords[base + h_idx];
            let dh2 = distance_sq(donor, h, length_scale);
            if dh2 > donor_h_cutoff2 {
                continue;
            }
            found_h = true;
            if let Some(angle) = angle_degrees(donor, h, acceptor, length_scale) {
                if angle > best_angle {
                    best_angle = angle;
                }
            }
        }
    }
    if found_h {
        if best_angle < hbond_min_angle_deg {
            return None;
        }
        let denom = (180.0 - hbond_min_angle_deg).max(1.0);
        return Some(((best_angle - hbond_min_angle_deg) / denom).clamp(0.0, 1.0));
    }
    if allow_missing_hydrogen {
        return Some(0.5);
    }
    None
}

fn angle_degrees(a: [f32; 4], vertex: [f32; 4], c: [f32; 4], length_scale: f64) -> Option<f64> {
    let v1 = [
        (a[0] as f64 - vertex[0] as f64) * length_scale,
        (a[1] as f64 - vertex[1] as f64) * length_scale,
        (a[2] as f64 - vertex[2] as f64) * length_scale,
    ];
    let v2 = [
        (c[0] as f64 - vertex[0] as f64) * length_scale,
        (c[1] as f64 - vertex[1] as f64) * length_scale,
        (c[2] as f64 - vertex[2] as f64) * length_scale,
    ];
    let n1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
    let n2 = (v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]).sqrt();
    if n1 == 0.0 || n2 == 0.0 {
        return None;
    }
    let cos = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2);
    let clamped = cos.clamp(-1.0, 1.0);
    Some(clamped.acos().to_degrees())
}

#[allow(clippy::too_many_arguments)]
fn classify_interaction(
    receptor: AtomFlags,
    ligand: AtomFlags,
    r2: f64,
    close2: f64,
    hydrophobic2: f64,
    hbond2: f64,
    clash2: f64,
    salt2: f64,
    halogen2: f64,
    metal2: f64,
    cation_pi2: f64,
    pi_pi2: f64,
    hbond_ok: bool,
) -> u8 {
    if clash2 > 0.0 && r2 <= clash2 {
        return INTERACTION_CLASH;
    }
    if is_metal_pair(receptor, ligand) && r2 <= metal2 {
        return INTERACTION_METAL_COORDINATION;
    }
    if is_salt_pair(receptor, ligand) && r2 <= salt2 {
        return INTERACTION_SALT_BRIDGE;
    }
    if is_halogen_pair(receptor, ligand) && r2 <= halogen2 {
        return INTERACTION_HALOGEN_BOND;
    }
    if hbond_ok && r2 <= hbond2 {
        return INTERACTION_HYDROGEN_BOND;
    }
    if is_cation_pi_pair(receptor, ligand) && r2 <= cation_pi2 {
        return INTERACTION_CATION_PI;
    }
    if receptor.aromatic && ligand.aromatic && r2 <= pi_pi2 {
        return INTERACTION_PI_PI_STACKING;
    }
    if receptor.hydrophobic && ligand.hydrophobic && r2 <= hydrophobic2 {
        return INTERACTION_HYDROPHOBIC;
    }
    if r2 <= close2 {
        return INTERACTION_CLOSE_CONTACT;
    }
    0
}

fn is_metal_pair(receptor: AtomFlags, ligand: AtomFlags) -> bool {
    (receptor.metal && (ligand.acceptor || ligand.donor))
        || (ligand.metal && (receptor.acceptor || receptor.donor))
}

fn is_salt_pair(receptor: AtomFlags, ligand: AtomFlags) -> bool {
    (receptor.positive && ligand.negative) || (ligand.positive && receptor.negative)
}

fn is_halogen_pair(receptor: AtomFlags, ligand: AtomFlags) -> bool {
    (receptor.halogen && ligand.acceptor) || (ligand.halogen && receptor.acceptor)
}

fn is_cation_pi_pair(receptor: AtomFlags, ligand: AtomFlags) -> bool {
    (receptor.positive && ligand.aromatic) || (ligand.positive && receptor.aromatic)
}

#[allow(clippy::too_many_arguments)]
fn interaction_strength(
    code: u8,
    dist: f64,
    close_cutoff: f64,
    hydrophobic_cutoff: f64,
    hbond_cutoff: f64,
    clash_cutoff: f64,
    salt_cutoff: f64,
    halogen_cutoff: f64,
    metal_cutoff: f64,
    cation_pi_cutoff: f64,
    pi_pi_cutoff: f64,
    hbond_score: Option<f64>,
) -> f64 {
    let raw = match code {
        INTERACTION_HYDROGEN_BOND => hbond_score.unwrap_or(1.0 - dist / hbond_cutoff),
        INTERACTION_HYDROPHOBIC => 1.0 - dist / hydrophobic_cutoff,
        INTERACTION_CLOSE_CONTACT => 1.0 - dist / close_cutoff,
        INTERACTION_CLASH => {
            if clash_cutoff <= 0.0 {
                0.0
            } else {
                1.0 - dist / clash_cutoff
            }
        }
        INTERACTION_SALT_BRIDGE => 1.0 - dist / salt_cutoff,
        INTERACTION_HALOGEN_BOND => 1.0 - dist / halogen_cutoff,
        INTERACTION_METAL_COORDINATION => 1.0 - dist / metal_cutoff,
        INTERACTION_CATION_PI => 1.0 - dist / cation_pi_cutoff,
        INTERACTION_PI_PI_STACKING => 1.0 - dist / pi_pi_cutoff,
        _ => 0.0,
    };
    raw.clamp(0.0, 1.0)
}

pub type DockingPlan = DockingAnalysisPlan;
