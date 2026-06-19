use std::collections::{HashMap, HashSet};

use traj_core::frame::{Box3, FrameChunk};
use traj_core::pbc_math::{apply_pbc, apply_pbc_triclinic, cell_and_inv_from_box};
use traj_core::selection::Selection;
use traj_core::system::System;

pub(crate) const DSSP_CODE_C: u8 = 0;
pub(crate) const DSSP_CODE_H: u8 = 1;
pub(crate) const DSSP_CODE_B: u8 = 2;
pub(crate) const DSSP_CODE_E: u8 = 3;
pub(crate) const DSSP_CODE_G: u8 = 4;
pub(crate) const DSSP_CODE_I: u8 = 5;
pub(crate) const DSSP_CODE_T: u8 = 6;
pub(crate) const DSSP_CODE_S: u8 = 7;

const HBOND_CUTOFF_KCAL: f64 = -0.5;
const HELIX_PHI: f64 = -55.0;
const HELIX_PSI: f64 = -45.0;
const HELIX_PPRMS2_CUTOFF: f64 = 2500.0;
const HELIX_D4_CUTOFF: f64 = 3.6;
const PEPTIDE_H_BOND_LENGTH: f64 = 1.0;

#[derive(Clone, Debug)]
pub(crate) struct BackboneResidue {
    pub resid: i32,
    pub chain_id: u32,
    pub segment_id: usize,
    pub n_idx: Option<usize>,
    pub h_idx: Option<usize>,
    pub ca_idx: Option<usize>,
    pub c_idx: Option<usize>,
    pub o_idx: Option<usize>,
    pub prev_index: Option<usize>,
    pub next_index: Option<usize>,
    pub is_proline: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BackboneModel {
    pub residues: Vec<BackboneResidue>,
    pub labels: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct BackboneFrame {
    pub n: Vec<Option<[f64; 3]>>,
    pub h: Vec<Option<[f64; 3]>>,
    pub ca: Vec<Option<[f64; 3]>>,
    pub c: Vec<Option<[f64; 3]>>,
    pub o: Vec<Option<[f64; 3]>>,
    pub phi: Vec<Option<f64>>,
    pub psi: Vec<Option<f64>>,
    pub pprms2: Vec<Option<f64>>,
    pub d3: Vec<Option<f64>>,
    pub d4: Vec<Option<f64>>,
    pub d5: Vec<Option<f64>>,
    pub hbond_energy: Vec<f64>,
    pub hbond_present: Vec<bool>,
    pub turn3: Vec<bool>,
    pub turn4: Vec<bool>,
    pub turn5: Vec<bool>,
    pub states: Vec<u8>,
}

#[derive(Clone, Debug)]
struct BoxTransform {
    orthorhombic: Option<[f64; 3]>,
    triclinic: Option<([[f64; 3]; 3], [[f64; 3]; 3])>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum BridgeOrientation {
    Antiparallel,
    Parallel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BridgePair {
    i: usize,
    j: usize,
    orientation: BridgeOrientation,
}

pub(crate) fn build_backbone_model(system: &System, selection: &Selection) -> BackboneModel {
    let n_atoms = system.n_atoms();
    if n_atoms == 0 {
        return BackboneModel::default();
    }
    let selected: HashSet<usize> = selection.indices.iter().map(|&idx| idx as usize).collect();
    let use_all = selected.is_empty();
    let atoms = &system.atoms;
    let mut residues = Vec::new();
    let mut labels = Vec::new();

    let mut start = 0usize;
    while start < n_atoms {
        let chain = atoms.chain_id[start];
        let resid = atoms.resid[start];
        let resname_id = atoms.resname_id[start];
        let mut end = start + 1;
        while end < n_atoms
            && atoms.chain_id[end] == chain
            && atoms.resid[end] == resid
            && atoms.resname_id[end] == resname_id
        {
            end += 1;
        }

        let mut include = use_all;
        if !include {
            for idx in start..end {
                if selected.contains(&idx) {
                    include = true;
                    break;
                }
            }
        }
        if include {
            let resname = system
                .interner
                .resolve(resname_id)
                .unwrap_or("RES")
                .to_ascii_uppercase();
            let label = format!("{resname}:{resid}");
            let mut residue = BackboneResidue {
                resid,
                chain_id: chain,
                segment_id: 0,
                n_idx: None,
                h_idx: None,
                ca_idx: None,
                c_idx: None,
                o_idx: None,
                prev_index: None,
                next_index: None,
                is_proline: resname == "PRO",
            };
            for idx in start..end {
                let atom_name = system
                    .interner
                    .resolve(atoms.name_id[idx])
                    .unwrap_or("")
                    .to_ascii_uppercase();
                match atom_name.as_str() {
                    "N" => residue.n_idx = Some(idx),
                    "CA" => residue.ca_idx = Some(idx),
                    "C" => residue.c_idx = Some(idx),
                    "O" | "OT1" => {
                        if residue.o_idx.is_none() {
                            residue.o_idx = Some(idx);
                        }
                    }
                    "H" | "HN" | "H1" | "HN1" | "HT1" => {
                        if residue.h_idx.is_none() {
                            residue.h_idx = Some(idx);
                        }
                    }
                    _ => {}
                }
            }
            labels.push(label);
            residues.push(residue);
        }
        start = end;
    }

    let mut segment_id = 0usize;
    for i in 0..residues.len() {
        if i == 0 {
            residues[i].segment_id = segment_id;
            continue;
        }
        let contiguous = residues[i].chain_id == residues[i - 1].chain_id
            && residues[i].resid == residues[i - 1].resid + 1;
        if !contiguous {
            segment_id += 1;
        } else {
            residues[i].prev_index = Some(i - 1);
            residues[i - 1].next_index = Some(i);
        }
        residues[i].segment_id = segment_id;
    }

    BackboneModel { residues, labels }
}

pub(crate) fn build_backbone_io_selection(model: &BackboneModel) -> Vec<u32> {
    let mut seen = HashSet::new();
    let mut indices = Vec::new();
    for residue in model.residues.iter() {
        for atom_idx in [
            residue.n_idx,
            residue.ca_idx,
            residue.c_idx,
            residue.o_idx,
            residue.h_idx,
        ]
        .into_iter()
        .flatten()
        {
            if seen.insert(atom_idx) {
                indices.push(atom_idx as u32);
            }
        }
    }
    indices
}

pub(crate) fn remap_backbone_model(model: &BackboneModel, io_selection: &[u32]) -> BackboneModel {
    let slot_by_atom: HashMap<usize, usize> = io_selection
        .iter()
        .enumerate()
        .map(|(slot, &atom)| (atom as usize, slot))
        .collect();
    let mut remapped = model.clone();
    for residue in remapped.residues.iter_mut() {
        residue.n_idx = remap_atom_index(residue.n_idx, &slot_by_atom);
        residue.h_idx = remap_atom_index(residue.h_idx, &slot_by_atom);
        residue.ca_idx = remap_atom_index(residue.ca_idx, &slot_by_atom);
        residue.c_idx = remap_atom_index(residue.c_idx, &slot_by_atom);
        residue.o_idx = remap_atom_index(residue.o_idx, &slot_by_atom);
    }
    remapped
}

fn remap_atom_index(
    atom_idx: Option<usize>,
    slot_by_atom: &HashMap<usize, usize>,
) -> Option<usize> {
    atom_idx.and_then(|idx| slot_by_atom.get(&idx).copied())
}

pub(crate) fn compute_backbone_frame(
    model: &BackboneModel,
    chunk: &FrameChunk,
    frame: usize,
) -> BackboneFrame {
    let n = model.residues.len();
    let mut out = BackboneFrame::default();
    reset_backbone_frame(&mut out, n);
    compute_backbone_frame_into(model, chunk, frame, &mut out);
    out
}

pub(crate) fn compute_backbone_frame_into(
    model: &BackboneModel,
    chunk: &FrameChunk,
    frame: usize,
    out: &mut BackboneFrame,
) {
    let n = model.residues.len();
    reset_backbone_frame(out, n);
    if n == 0 {
        return;
    }

    let transform = box_transform(chunk.box_.get(frame).copied().unwrap_or(Box3::None));
    let base = frame * chunk.n_atoms;

    for i in 0..n {
        let residue = &model.residues[i];
        if let Some(ca_idx) = residue.ca_idx {
            let ca_raw = point(chunk, base + ca_idx);
            let ca_unwrapped = if let Some(prev_idx) = residue.prev_index {
                match (
                    model.residues[prev_idx].ca_idx,
                    out.ca[prev_idx],
                    model.residues[prev_idx]
                        .ca_idx
                        .map(|idx| point(chunk, base + idx)),
                ) {
                    (Some(_), Some(prev_unwrapped), Some(prev_raw)) => {
                        let mut delta = sub(ca_raw, prev_raw);
                        apply_minimum_image(&mut delta, &transform);
                        add(prev_unwrapped, delta)
                    }
                    _ => ca_raw,
                }
            } else {
                ca_raw
            };
            out.ca[i] = Some(ca_unwrapped);
            out.n[i] = map_atom(
                residue.n_idx,
                residue.ca_idx,
                out.ca[i],
                chunk,
                base,
                &transform,
            );
            out.c[i] = map_atom(
                residue.c_idx,
                residue.ca_idx,
                out.ca[i],
                chunk,
                base,
                &transform,
            );
            out.o[i] = map_atom(
                residue.o_idx,
                residue.ca_idx,
                out.ca[i],
                chunk,
                base,
                &transform,
            );
            out.h[i] = map_atom(
                residue.h_idx,
                residue.ca_idx,
                out.ca[i],
                chunk,
                base,
                &transform,
            );
        } else {
            out.n[i] = residue.n_idx.map(|idx| point(chunk, base + idx));
            out.c[i] = residue.c_idx.map(|idx| point(chunk, base + idx));
            out.o[i] = residue.o_idx.map(|idx| point(chunk, base + idx));
            out.h[i] = residue.h_idx.map(|idx| point(chunk, base + idx));
        }
    }

    for i in 0..n {
        if out.h[i].is_none() && !model.residues[i].is_proline {
            if let (Some(prev_idx), Some(n_pos), Some(ca_pos), Some(c_prev_pos)) = (
                model.residues[i].prev_index,
                out.n[i],
                out.ca[i],
                model.residues[i].prev_index.and_then(|idx| {
                    if model.residues[idx].segment_id == model.residues[i].segment_id {
                        out.c[idx]
                    } else {
                        None
                    }
                }),
            ) {
                let _ = prev_idx;
                out.h[i] = synthesize_backbone_hydrogen(n_pos, ca_pos, c_prev_pos);
            }
        }
    }

    for i in 0..n {
        let residue = &model.residues[i];
        if let (Some(prev_idx), Some(n_pos), Some(ca_pos), Some(c_pos)) =
            (residue.prev_index, out.n[i], out.ca[i], out.c[i])
        {
            if model.residues[prev_idx].segment_id == residue.segment_id {
                if let Some(prev_c) = out.c[prev_idx] {
                    out.phi[i] = dihedral(prev_c, n_pos, ca_pos, c_pos);
                }
            }
        }
        if let (Some(next_idx), Some(n_pos), Some(ca_pos), Some(c_pos)) =
            (residue.next_index, out.n[i], out.ca[i], out.c[i])
        {
            if model.residues[next_idx].segment_id == residue.segment_id {
                if let Some(next_n) = out.n[next_idx] {
                    out.psi[i] = dihedral(n_pos, ca_pos, c_pos, next_n);
                }
            }
        }
        if let (Some(phi), Some(psi)) = (out.phi[i], out.psi[i]) {
            out.pprms2[i] = Some((phi - HELIX_PHI).powi(2) + (psi - HELIX_PSI).powi(2));
        }
        out.d3[i] = o_to_n_distance(model, &out, i, 3);
        out.d4[i] = o_to_n_distance(model, &out, i, 4);
        out.d5[i] = o_to_n_distance(model, &out, i, 5);
    }

    for i in 0..n {
        for j in 0..n {
            if i == j || i.abs_diff(j) < 2 {
                continue;
            }
            if let (Some(o_pos), Some(c_pos), Some(n_pos), Some(h_pos)) =
                (out.o[i], out.c[i], out.n[j], out.h[j])
            {
                let energy = hydrogen_bond_energy(o_pos, c_pos, n_pos, h_pos);
                let idx = pair_index(n, i, j);
                out.hbond_energy[idx] = energy;
                out.hbond_present[idx] = energy < HBOND_CUTOFF_KCAL;
            }
        }
    }

    for i in 0..n {
        out.turn3[i] = has_turn(model, &out, i, 3);
        out.turn4[i] = has_turn(model, &out, i, 4);
        out.turn5[i] = has_turn(model, &out, i, 5);
    }
    out.states = assign_dssp_states(model, &out);
}

fn reset_backbone_frame(out: &mut BackboneFrame, n: usize) {
    out.n.clear();
    out.n.resize(n, None);
    out.h.clear();
    out.h.resize(n, None);
    out.ca.clear();
    out.ca.resize(n, None);
    out.c.clear();
    out.c.resize(n, None);
    out.o.clear();
    out.o.resize(n, None);
    out.phi.clear();
    out.phi.resize(n, None);
    out.psi.clear();
    out.psi.resize(n, None);
    out.pprms2.clear();
    out.pprms2.resize(n, None);
    out.d3.clear();
    out.d3.resize(n, None);
    out.d4.clear();
    out.d4.resize(n, None);
    out.d5.clear();
    out.d5.resize(n, None);
    out.hbond_energy.clear();
    out.hbond_energy.resize(n * n, f64::NAN);
    out.hbond_present.clear();
    out.hbond_present.resize(n * n, false);
    out.turn3.clear();
    out.turn3.resize(n, false);
    out.turn4.clear();
    out.turn4.resize(n, false);
    out.turn5.clear();
    out.turn5.resize(n, false);
    out.states.clear();
    out.states.resize(n, DSSP_CODE_C);
}

#[cfg(test)]
pub(crate) fn collapse_dssp_code(code: u8) -> u8 {
    match code {
        DSSP_CODE_H | DSSP_CODE_G | DSSP_CODE_I => DSSP_CODE_H,
        DSSP_CODE_E | DSSP_CODE_B => DSSP_CODE_E,
        _ => DSSP_CODE_C,
    }
}

pub(crate) fn helix_flags(model: &BackboneModel, frame: &BackboneFrame) -> Vec<bool> {
    let n = model.residues.len();
    let mut flags = vec![false; n];
    for i in 0..n {
        let Some(pprms2) = frame.pprms2[i] else {
            continue;
        };
        if pprms2 >= HELIX_PPRMS2_CUTOFF {
            continue;
        }
        let d4_ok = frame.d4[i].map(|d| d < HELIX_D4_CUTOFF).unwrap_or(false);
        let contiguous_prev = if i > 0 {
            model.residues[i].segment_id == model.residues[i - 1].segment_id
                && model.residues[i].prev_index == Some(i - 1)
        } else {
            false
        };
        if d4_ok || (contiguous_prev && flags[i - 1]) {
            flags[i] = true;
        }
    }
    flags
}

pub(crate) fn longest_true_run(model: &BackboneModel, flags: &[bool]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut start = 0usize;
    while start < flags.len() {
        while start < flags.len() && !flags[start] {
            start += 1;
        }
        if start >= flags.len() {
            break;
        }
        let mut end = start;
        while end + 1 < flags.len()
            && flags[end + 1]
            && model.residues[end + 1].segment_id == model.residues[start].segment_id
            && model.residues[end + 1].prev_index == Some(end)
        {
            end += 1;
        }
        if best
            .map(|(best_start, best_end)| end - start > best_end - best_start)
            .unwrap_or(true)
        {
            best = Some((start, end));
        }
        start = end + 1;
    }
    best
}

pub(crate) fn average_phi_psi_weights(phi: f64, psi: f64) -> Option<f64> {
    const PPW: &[(f64, f64, f64)] = &[
        (-67.0, -44.0, 0.31),
        (-66.0, -41.0, 0.31),
        (-59.0, -44.0, 0.44),
        (-57.0, -47.0, 0.56),
        (-53.0, -52.0, 0.78),
        (-48.0, -57.0, 1.00),
        (-70.5, -35.8, 0.15),
        (-57.0, -79.0, 0.23),
        (-38.0, -78.0, 1.20),
        (-60.0, -30.0, 0.24),
        (-54.0, -28.0, 0.46),
        (-44.0, -33.0, 0.68),
    ];
    PPW.iter().find_map(|&(pphi, ppsi, weight)| {
        let distance2 = (phi - pphi).powi(2) + (psi - ppsi).powi(2);
        if distance2 < 64.0 {
            Some(weight)
        } else {
            None
        }
    })
}

pub(crate) fn jcaha_delta(phi: f64, psi: f64) -> f64 {
    1.4 * (psi + 138.0).to_radians().sin() - 4.1 * (2.0 * (psi + 138.0).to_radians()).cos()
        + 2.0 * (2.0 * (phi + 30.0).to_radians()).cos()
}

fn assign_dssp_states(model: &BackboneModel, frame: &BackboneFrame) -> Vec<u8> {
    let n = model.residues.len();
    let mut states = vec![DSSP_CODE_C; n];
    let turn3_span = turn_span(&frame.turn3, 3, model);
    let turn4_span = turn_span(&frame.turn4, 4, model);
    let turn5_span = turn_span(&frame.turn5, 5, model);

    for i in 0..n.saturating_sub(5) {
        if frame.turn5[i]
            && frame.turn5[i + 1]
            && same_segment(model, i, i + 1)
            && contiguous_chain(model, i + 1, i + 5)
        {
            for state in states.iter_mut().take(i + 6).skip(i + 1) {
                *state = DSSP_CODE_I;
            }
        }
    }
    for i in 0..n.saturating_sub(4) {
        if frame.turn4[i]
            && frame.turn4[i + 1]
            && same_segment(model, i, i + 1)
            && contiguous_chain(model, i + 1, i + 4)
        {
            for state in states.iter_mut().take(i + 5).skip(i + 1) {
                *state = DSSP_CODE_H;
            }
        }
    }
    for i in 0..n.saturating_sub(3) {
        if frame.turn3[i]
            && frame.turn3[i + 1]
            && same_segment(model, i, i + 1)
            && contiguous_chain(model, i + 1, i + 3)
        {
            for state in states.iter_mut().take(i + 4).skip(i + 1) {
                if *state == DSSP_CODE_C || *state == DSSP_CODE_T {
                    *state = DSSP_CODE_G;
                }
            }
        }
    }

    let bridges = bridge_pairs(model, frame);
    let mut bridge_count = vec![0usize; n];
    let mut ladder_state = vec![false; n];
    for bridge in bridges.iter() {
        bridge_count[bridge.i] += 1;
        bridge_count[bridge.j] += 1;
    }
    for bridge in bridges.iter() {
        let in_ladder = bridges.iter().any(|other| {
            bridge != other
                && bridge.orientation == other.orientation
                && ladder_neighbor(*bridge, *other)
        });
        if in_ladder {
            ladder_state[bridge.i] = true;
            ladder_state[bridge.j] = true;
        }
    }
    for i in 0..n {
        if states[i] == DSSP_CODE_C {
            if ladder_state[i] {
                states[i] = DSSP_CODE_E;
            } else if bridge_count[i] > 0 {
                states[i] = DSSP_CODE_B;
            }
        }
    }

    for i in 0..n {
        if states[i] == DSSP_CODE_C && (turn3_span[i] || turn4_span[i] || turn5_span[i]) {
            states[i] = DSSP_CODE_T;
        }
    }
    let bends = bend_flags(model, frame);
    for i in 0..n {
        if states[i] == DSSP_CODE_C && bends[i] {
            states[i] = DSSP_CODE_S;
        }
    }
    states
}

fn turn_span(turns: &[bool], span: usize, model: &BackboneModel) -> Vec<bool> {
    let mut out = vec![false; turns.len()];
    for i in 0..turns.len() {
        if !turns[i] || i + span >= turns.len() || !contiguous_chain(model, i, i + span) {
            continue;
        }
        for value in out.iter_mut().take(i + span + 1).skip(i + 1) {
            *value = true;
        }
    }
    out
}

fn bend_flags(model: &BackboneModel, frame: &BackboneFrame) -> Vec<bool> {
    let n = model.residues.len();
    let mut out = vec![false; n];
    for i in 2..n.saturating_sub(2) {
        if !(contiguous_chain(model, i - 2, i) && contiguous_chain(model, i, i + 2)) {
            continue;
        }
        let (Some(ca_prev2), Some(ca_i), Some(ca_next2)) =
            (frame.ca[i - 2], frame.ca[i], frame.ca[i + 2])
        else {
            continue;
        };
        let v1 = normalize(sub(ca_prev2, ca_i));
        let v2 = normalize(sub(ca_next2, ca_i));
        let angle = clamp(dot(v1, v2)).acos().to_degrees();
        if angle > 70.0 {
            out[i] = true;
        }
    }
    out
}

fn bridge_pairs(model: &BackboneModel, frame: &BackboneFrame) -> Vec<BridgePair> {
    let n = model.residues.len();
    let mut out = Vec::new();
    for i in 0..n {
        for j in (i + 2)..n {
            let anti = hb(frame, n, i as isize, j as isize) && hb(frame, n, j as isize, i as isize)
                || hb(frame, n, i as isize - 1, j as isize + 1)
                    && hb(frame, n, j as isize - 1, i as isize + 1);
            let parallel = hb(frame, n, i as isize - 1, j as isize)
                && hb(frame, n, j as isize, i as isize + 1)
                || hb(frame, n, j as isize - 1, i as isize)
                    && hb(frame, n, i as isize, j as isize + 1);
            if anti {
                out.push(BridgePair {
                    i,
                    j,
                    orientation: BridgeOrientation::Antiparallel,
                });
            }
            if parallel {
                out.push(BridgePair {
                    i,
                    j,
                    orientation: BridgeOrientation::Parallel,
                });
            }
        }
    }
    out.sort_by_key(|pair| (pair.i, pair.j, pair.orientation));
    out.dedup();
    out.retain(|pair| {
        model.residues[pair.i].ca_idx.is_some() && model.residues[pair.j].ca_idx.is_some()
    });
    out
}

fn ladder_neighbor(a: BridgePair, b: BridgePair) -> bool {
    let di = b.i as isize - a.i as isize;
    let dj = b.j as isize - a.j as isize;
    if di.abs() != 1 || dj.abs() != 1 {
        return false;
    }
    match a.orientation {
        BridgeOrientation::Antiparallel => di == dj,
        BridgeOrientation::Parallel => di == -dj,
    }
}

fn hb(frame: &BackboneFrame, n: usize, i: isize, j: isize) -> bool {
    if i < 0 || j < 0 {
        return false;
    }
    let (i, j) = (i as usize, j as usize);
    if i >= n || j >= n {
        return false;
    }
    frame.hbond_present[pair_index(n, i, j)]
}

fn has_turn(model: &BackboneModel, frame: &BackboneFrame, i: usize, span: usize) -> bool {
    if i + span >= model.residues.len() || !contiguous_chain(model, i, i + span) {
        return false;
    }
    frame.hbond_present[pair_index(model.residues.len(), i, i + span)]
}

fn o_to_n_distance(
    model: &BackboneModel,
    frame: &BackboneFrame,
    i: usize,
    span: usize,
) -> Option<f64> {
    if i + span >= model.residues.len() || !contiguous_chain(model, i, i + span) {
        return None;
    }
    match (frame.o[i], frame.n[i + span]) {
        (Some(o_pos), Some(n_pos)) => Some(distance(o_pos, n_pos)),
        _ => None,
    }
}

fn same_segment(model: &BackboneModel, i: usize, j: usize) -> bool {
    model.residues[i].segment_id == model.residues[j].segment_id
}

fn contiguous_chain(model: &BackboneModel, start: usize, end: usize) -> bool {
    if start > end || end >= model.residues.len() {
        return false;
    }
    for i in start + 1..=end {
        if model.residues[i].prev_index != Some(i - 1)
            || model.residues[i].segment_id != model.residues[start].segment_id
        {
            return false;
        }
    }
    true
}

fn hydrogen_bond_energy(o_pos: [f64; 3], c_pos: [f64; 3], n_pos: [f64; 3], h_pos: [f64; 3]) -> f64 {
    let r_on = distance(o_pos, n_pos);
    let r_ch = distance(c_pos, h_pos);
    let r_oh = distance(o_pos, h_pos);
    let r_cn = distance(c_pos, n_pos);
    if r_on <= 1e-6 || r_ch <= 1e-6 || r_oh <= 1e-6 || r_cn <= 1e-6 {
        return f64::INFINITY;
    }
    27.888 * (1.0 / r_on + 1.0 / r_ch - 1.0 / r_oh - 1.0 / r_cn)
}

fn synthesize_backbone_hydrogen(
    n_pos: [f64; 3],
    ca_pos: [f64; 3],
    c_prev_pos: [f64; 3],
) -> Option<[f64; 3]> {
    let dir1 = normalize(sub(n_pos, ca_pos));
    let dir2 = normalize(sub(n_pos, c_prev_pos));
    let direction = normalize(add(dir1, dir2));
    if norm(direction) <= 1e-6 {
        return None;
    }
    Some(add(n_pos, mul(direction, PEPTIDE_H_BOND_LENGTH)))
}

fn map_atom(
    atom_idx: Option<usize>,
    ref_idx: Option<usize>,
    ref_unwrapped: Option<[f64; 3]>,
    chunk: &FrameChunk,
    base: usize,
    transform: &BoxTransform,
) -> Option<[f64; 3]> {
    match (atom_idx, ref_idx, ref_unwrapped) {
        (Some(atom_idx), Some(ref_idx), Some(ref_unwrapped)) => {
            let raw = point(chunk, base + atom_idx);
            let ref_raw = point(chunk, base + ref_idx);
            let mut delta = sub(raw, ref_raw);
            apply_minimum_image(&mut delta, transform);
            Some(add(ref_unwrapped, delta))
        }
        (Some(atom_idx), _, _) => Some(point(chunk, base + atom_idx)),
        _ => None,
    }
}

fn pair_index(n: usize, i: usize, j: usize) -> usize {
    i * n + j
}

fn box_transform(box_: Box3) -> BoxTransform {
    match box_ {
        Box3::Orthorhombic { lx, ly, lz } => BoxTransform {
            orthorhombic: Some([lx as f64, ly as f64, lz as f64]),
            triclinic: None,
        },
        Box3::Triclinic { .. } => match cell_and_inv_from_box(box_) {
            Ok((cell, inv)) => BoxTransform {
                orthorhombic: None,
                triclinic: Some((cell, inv)),
            },
            Err(_) => BoxTransform {
                orthorhombic: None,
                triclinic: None,
            },
        },
        Box3::None => BoxTransform {
            orthorhombic: None,
            triclinic: None,
        },
    }
}

fn apply_minimum_image(delta: &mut [f64; 3], transform: &BoxTransform) {
    let [mut dx, mut dy, mut dz] = *delta;
    if let Some([lx, ly, lz]) = transform.orthorhombic {
        apply_pbc(&mut dx, &mut dy, &mut dz, lx, ly, lz);
    } else if let Some((cell, inv)) = transform.triclinic.as_ref() {
        apply_pbc_triclinic(&mut dx, &mut dy, &mut dz, cell, inv);
    }
    *delta = [dx, dy, dz];
}

fn point(chunk: &FrameChunk, idx: usize) -> [f64; 3] {
    let p = chunk.coords[idx];
    [p[0] as f64, p[1] as f64, p[2] as f64]
}

fn dihedral(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> Option<f64> {
    let b0 = sub(a, b);
    let b1 = sub(c, b);
    let b2 = sub(d, c);

    let norm_b1 = norm(b1);
    if norm_b1 <= 1e-12 {
        return None;
    }
    let b1n = mul(b1, 1.0 / norm_b1);
    let v = sub(b0, mul(b1n, dot(b0, b1n)));
    let w = sub(b2, mul(b1n, dot(b2, b1n)));
    let norm_v = norm(v);
    let norm_w = norm(w);
    if norm_v <= 1e-12 || norm_w <= 1e-12 {
        return None;
    }
    let vn = mul(v, 1.0 / norm_v);
    let wn = mul(w, 1.0 / norm_w);
    Some(dot(cross(b1n, vn), wn).atan2(dot(vn, wn)).to_degrees())
}

pub(crate) fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub(crate) fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(crate) fn mul(a: [f64; 3], scale: f64) -> [f64; 3] {
    [a[0] * scale, a[1] * scale, a[2] * scale]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub(crate) fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn normalize(a: [f64; 3]) -> [f64; 3] {
    let len = norm(a);
    if len <= 1e-12 {
        [0.0, 0.0, 0.0]
    } else {
        mul(a, 1.0 / len)
    }
}

pub(crate) fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm(sub(a, b))
}

pub(crate) fn clamp(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_model(n: usize) -> BackboneModel {
        BackboneModel {
            residues: (0..n)
                .map(|i| BackboneResidue {
                    resid: i as i32 + 1,
                    chain_id: 0,
                    segment_id: 0,
                    n_idx: None,
                    h_idx: None,
                    ca_idx: Some(i),
                    c_idx: None,
                    o_idx: None,
                    prev_index: if i > 0 { Some(i - 1) } else { None },
                    next_index: if i + 1 < n { Some(i + 1) } else { None },
                    is_proline: false,
                })
                .collect(),
            labels: (0..n).map(|i| format!("ALA:{}", i + 1)).collect(),
        }
    }

    #[test]
    fn collapse_codes_maps_8_state_to_3_state() {
        assert_eq!(collapse_dssp_code(DSSP_CODE_H), DSSP_CODE_H);
        assert_eq!(collapse_dssp_code(DSSP_CODE_G), DSSP_CODE_H);
        assert_eq!(collapse_dssp_code(DSSP_CODE_I), DSSP_CODE_H);
        assert_eq!(collapse_dssp_code(DSSP_CODE_B), DSSP_CODE_E);
        assert_eq!(collapse_dssp_code(DSSP_CODE_E), DSSP_CODE_E);
        assert_eq!(collapse_dssp_code(DSSP_CODE_T), DSSP_CODE_C);
        assert_eq!(collapse_dssp_code(DSSP_CODE_S), DSSP_CODE_C);
    }

    #[test]
    fn consecutive_turns_promote_helices() {
        let model = empty_model(10);
        let mut frame = BackboneFrame {
            n: vec![None; 10],
            h: vec![None; 10],
            ca: vec![Some([0.0, 0.0, 0.0]); 10],
            c: vec![None; 10],
            o: vec![None; 10],
            phi: vec![None; 10],
            psi: vec![None; 10],
            pprms2: vec![None; 10],
            d3: vec![None; 10],
            d4: vec![None; 10],
            d5: vec![None; 10],
            hbond_energy: vec![f64::NAN; 100],
            hbond_present: vec![false; 100],
            turn3: vec![false; 10],
            turn4: vec![false; 10],
            turn5: vec![false; 10],
            states: vec![DSSP_CODE_C; 10],
        };
        frame.turn4[1] = true;
        frame.turn4[2] = true;
        frame.turn3[6] = true;
        frame.turn3[7] = true;
        frame.turn5[0] = true;
        frame.turn5[1] = true;
        let states = assign_dssp_states(&model, &frame);
        assert_eq!(states[1], DSSP_CODE_I);
        assert_eq!(states[2], DSSP_CODE_H);
        assert_eq!(states[8], DSSP_CODE_G);
    }

    #[test]
    fn bridge_ladder_promotes_sheet_and_isolated_bridge() {
        let model = empty_model(6);
        let mut frame = BackboneFrame {
            n: vec![None; 6],
            h: vec![None; 6],
            ca: vec![Some([0.0, 0.0, 0.0]); 6],
            c: vec![None; 6],
            o: vec![None; 6],
            phi: vec![None; 6],
            psi: vec![None; 6],
            pprms2: vec![None; 6],
            d3: vec![None; 6],
            d4: vec![None; 6],
            d5: vec![None; 6],
            hbond_energy: vec![f64::NAN; 36],
            hbond_present: vec![false; 36],
            turn3: vec![false; 6],
            turn4: vec![false; 6],
            turn5: vec![false; 6],
            states: vec![DSSP_CODE_C; 6],
        };
        let n = 6;
        frame.hbond_present[pair_index(n, 0, 4)] = true;
        frame.hbond_present[pair_index(n, 4, 0)] = true;
        frame.hbond_present[pair_index(n, 1, 5)] = true;
        frame.hbond_present[pair_index(n, 5, 1)] = true;
        frame.hbond_present[pair_index(n, 2, 4)] = true;
        frame.hbond_present[pair_index(n, 4, 2)] = true;
        let states = assign_dssp_states(&model, &frame);
        assert_eq!(states[0], DSSP_CODE_E);
        assert_eq!(states[1], DSSP_CODE_E);
        assert_eq!(states[2], DSSP_CODE_B);
    }

    #[test]
    fn bend_detection_marks_s() {
        let model = empty_model(5);
        let frame = BackboneFrame {
            n: vec![None; 5],
            h: vec![None; 5],
            ca: vec![
                Some([-2.0, 0.0, 0.0]),
                Some([-1.0, 0.0, 0.0]),
                Some([0.0, 0.0, 0.0]),
                Some([0.0, 1.0, 0.0]),
                Some([0.0, 2.0, 0.0]),
            ],
            c: vec![None; 5],
            o: vec![None; 5],
            phi: vec![None; 5],
            psi: vec![None; 5],
            pprms2: vec![None; 5],
            d3: vec![None; 5],
            d4: vec![None; 5],
            d5: vec![None; 5],
            hbond_energy: vec![f64::NAN; 25],
            hbond_present: vec![false; 25],
            turn3: vec![false; 5],
            turn4: vec![false; 5],
            turn5: vec![false; 5],
            states: vec![DSSP_CODE_C; 5],
        };
        let states = assign_dssp_states(&model, &frame);
        assert_eq!(states[2], DSSP_CODE_S);
    }

    #[test]
    fn helix_flags_extend_from_hbond_seed() {
        let model = empty_model(6);
        let frame = BackboneFrame {
            n: vec![None; 6],
            h: vec![None; 6],
            ca: vec![None; 6],
            c: vec![None; 6],
            o: vec![None; 6],
            phi: vec![None; 6],
            psi: vec![None; 6],
            pprms2: vec![Some(10.0); 6],
            d3: vec![None; 6],
            d4: vec![Some(3.2), None, None, None, None, None],
            d5: vec![None; 6],
            hbond_energy: vec![f64::NAN; 36],
            hbond_present: vec![false; 36],
            turn3: vec![false; 6],
            turn4: vec![false; 6],
            turn5: vec![false; 6],
            states: vec![DSSP_CODE_C; 6],
        };
        let flags = helix_flags(&model, &frame);
        assert_eq!(flags, vec![true, true, true, true, true, true]);
        assert_eq!(longest_true_run(&model, &flags), Some((0, 5)));
    }
}
