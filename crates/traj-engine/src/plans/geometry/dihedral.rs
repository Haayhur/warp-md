use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use super::utils::*;
use crate::executor::{Device, Plan, PlanOutput, PlanRequirements};
use crate::plans::{PbcMode, ReferenceMode};

struct DihedralDef {
    sel_a: Selection,
    sel_b: Selection,
    sel_c: Selection,
    sel_d: Selection,
}

struct DihedralDefLocal {
    sel_a: Vec<u32>,
    sel_b: Vec<u32>,
    sel_c: Vec<u32>,
    sel_d: Vec<u32>,
}

pub struct MultiDihedralPlan {
    dihedrals: Vec<DihedralDef>,
    dihedrals_local: Vec<DihedralDefLocal>,
    preferred_selection: Vec<u32>,
    selected_masses: Vec<f32>,
    use_selected_input: bool,
    mass_weighted: bool,
    pbc: PbcMode,
    degrees: bool,
    range360: bool,
    results: Vec<f32>,
    frames: usize,
}

impl MultiDihedralPlan {
    pub fn new(
        dihedrals: Vec<(Selection, Selection, Selection, Selection)>,
        mass_weighted: bool,
        pbc: PbcMode,
        degrees: bool,
        range360: bool,
    ) -> Self {
        let defs = dihedrals
            .into_iter()
            .map(|(a, b, c, d)| DihedralDef {
                sel_a: a,
                sel_b: b,
                sel_c: c,
                sel_d: d,
            })
            .collect();
        Self {
            dihedrals: defs,
            dihedrals_local: Vec::new(),
            preferred_selection: Vec::new(),
            selected_masses: Vec::new(),
            use_selected_input: false,
            mass_weighted,
            pbc,
            degrees,
            range360,
            results: Vec::new(),
            frames: 0,
        }
    }

    fn rebuild_selected_path(&mut self, system: &System) {
        self.dihedrals_local.clear();
        self.preferred_selection.clear();
        self.selected_masses.clear();

        let mut global_to_local = std::collections::HashMap::<u32, u32>::new();
        let mut push_idx =
            |idx: u32, preferred_selection: &mut Vec<u32>, selected_masses: &mut Vec<f32>| {
                if global_to_local.contains_key(&idx) {
                    return;
                }
                let local = preferred_selection.len() as u32;
                global_to_local.insert(idx, local);
                preferred_selection.push(idx);
                selected_masses.push(system.atoms.mass[idx as usize]);
            };

        for def in self.dihedrals.iter() {
            for &idx in def
                .sel_a
                .indices
                .iter()
                .chain(def.sel_b.indices.iter())
                .chain(def.sel_c.indices.iter())
                .chain(def.sel_d.indices.iter())
            {
                push_idx(
                    idx,
                    &mut self.preferred_selection,
                    &mut self.selected_masses,
                );
            }
        }

        for def in self.dihedrals.iter() {
            self.dihedrals_local.push(DihedralDefLocal {
                sel_a: def
                    .sel_a
                    .indices
                    .iter()
                    .map(|idx| *global_to_local.get(idx).unwrap())
                    .collect(),
                sel_b: def
                    .sel_b
                    .indices
                    .iter()
                    .map(|idx| *global_to_local.get(idx).unwrap())
                    .collect(),
                sel_c: def
                    .sel_c
                    .indices
                    .iter()
                    .map(|idx| *global_to_local.get(idx).unwrap())
                    .collect(),
                sel_d: def
                    .sel_d
                    .indices
                    .iter()
                    .map(|idx| *global_to_local.get(idx).unwrap())
                    .collect(),
            });
        }
    }
}

impl Plan for MultiDihedralPlan {
    fn name(&self) -> &'static str {
        "multidihedral"
    }

    fn requirements(&self) -> PlanRequirements {
        PlanRequirements::new(matches!(self.pbc, PbcMode::Orthorhombic), false)
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.frames = 0;
        self.use_selected_input = true;
        self.rebuild_selected_path(system);
        Ok(())
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        if self.use_selected_input && !self.preferred_selection.is_empty() {
            Some(&self.preferred_selection)
        } else {
            None
        }
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.dihedrals.is_empty() {
            self.frames += chunk.n_frames;
            return Ok(());
        }
        let masses = &system.atoms.mass;
        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            for def in self.dihedrals.iter() {
                let a = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_a.indices,
                    masses,
                    self.mass_weighted,
                );
                let b = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_b.indices,
                    masses,
                    self.mass_weighted,
                );
                let c = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_c.indices,
                    masses,
                    self.mass_weighted,
                );
                let d = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_d.indices,
                    masses,
                    self.mass_weighted,
                );
                let mut b0x = a[0] - b[0];
                let mut b0y = a[1] - b[1];
                let mut b0z = a[2] - b[2];
                let mut b1x = c[0] - b[0];
                let mut b1y = c[1] - b[1];
                let mut b1z = c[2] - b[2];
                let mut b2x = d[0] - c[0];
                let mut b2y = d[1] - c[1];
                let mut b2z = d[2] - c[2];
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut b0x, &mut b0y, &mut b0z, lx, ly, lz);
                    apply_pbc(&mut b1x, &mut b1y, &mut b1z, lx, ly, lz);
                    apply_pbc(&mut b2x, &mut b2y, &mut b2z, lx, ly, lz);
                }
                let angle = dihedral_from_vectors(
                    [b0x, b0y, b0z],
                    [b1x, b1y, b1z],
                    [b2x, b2y, b2z],
                    self.degrees,
                    self.range360,
                );
                self.results.push(angle);
            }
        }
        self.frames += chunk.n_frames;
        Ok(())
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        _source_selection: &[u32],
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.dihedrals_local.is_empty() {
            self.frames += chunk.n_frames;
            return Ok(());
        }

        for frame in 0..chunk.n_frames {
            let (lx, ly, lz) = if matches!(self.pbc, PbcMode::Orthorhombic) {
                box_lengths(chunk, frame)?
            } else {
                (0.0, 0.0, 0.0)
            };
            for def in self.dihedrals_local.iter() {
                let a = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_a,
                    &self.selected_masses,
                    self.mass_weighted,
                );
                let b = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_b,
                    &self.selected_masses,
                    self.mass_weighted,
                );
                let c = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_c,
                    &self.selected_masses,
                    self.mass_weighted,
                );
                let d = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_d,
                    &self.selected_masses,
                    self.mass_weighted,
                );
                let mut b0x = a[0] - b[0];
                let mut b0y = a[1] - b[1];
                let mut b0z = a[2] - b[2];
                let mut b1x = c[0] - b[0];
                let mut b1y = c[1] - b[1];
                let mut b1z = c[2] - b[2];
                let mut b2x = d[0] - c[0];
                let mut b2y = d[1] - c[1];
                let mut b2z = d[2] - c[2];
                if matches!(self.pbc, PbcMode::Orthorhombic) {
                    apply_pbc(&mut b0x, &mut b0y, &mut b0z, lx, ly, lz);
                    apply_pbc(&mut b1x, &mut b1y, &mut b1z, lx, ly, lz);
                    apply_pbc(&mut b2x, &mut b2y, &mut b2z, lx, ly, lz);
                }
                let angle = dihedral_from_vectors(
                    [b0x, b0y, b0z],
                    [b1x, b1y, b1z],
                    [b2x, b2y, b2z],
                    self.degrees,
                    self.range360,
                );
                self.results.push(angle);
            }
        }

        self.frames += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Matrix {
            data: std::mem::take(&mut self.results),
            rows: self.frames,
            cols: self.dihedrals.len(),
        })
    }
}

pub struct PermuteDihedralsPlan {
    inner: MultiDihedralPlan,
}

impl PermuteDihedralsPlan {
    pub fn new(
        dihedrals: Vec<(Selection, Selection, Selection, Selection)>,
        mass_weighted: bool,
        pbc: PbcMode,
        degrees: bool,
        range360: bool,
    ) -> Self {
        Self {
            inner: MultiDihedralPlan::new(dihedrals, mass_weighted, pbc, degrees, range360),
        }
    }
}

impl Plan for PermuteDihedralsPlan {
    fn name(&self) -> &'static str {
        "permute_dihedrals"
    }

    fn requirements(&self) -> PlanRequirements {
        self.inner.requirements()
    }

    fn init(&mut self, system: &System, device: &Device) -> TrajResult<()> {
        self.inner.init(system, device)
    }

    fn preferred_selection(&self) -> Option<&[u32]> {
        self.inner.preferred_selection()
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner.process_chunk(chunk, system, device)
    }

    fn process_chunk_selected(
        &mut self,
        chunk: &FrameChunk,
        source_selection: &[u32],
        system: &System,
        device: &Device,
    ) -> TrajResult<()> {
        self.inner
            .process_chunk_selected(chunk, source_selection, system, device)
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        self.inner.finalize()
    }
}

pub struct DihedralRmsPlan {
    dihedrals: Vec<DihedralDef>,
    mass_weighted: bool,
    pbc: PbcMode,
    degrees: bool,
    range360: bool,
    reference_mode: ReferenceMode,
    reference: Option<Vec<f32>>,
    results: Vec<f32>,
}

impl DihedralRmsPlan {
    pub fn new(
        dihedrals: Vec<(Selection, Selection, Selection, Selection)>,
        reference_mode: ReferenceMode,
        mass_weighted: bool,
        pbc: PbcMode,
        degrees: bool,
        range360: bool,
    ) -> Self {
        let defs = dihedrals
            .into_iter()
            .map(|(a, b, c, d)| DihedralDef {
                sel_a: a,
                sel_b: b,
                sel_c: c,
                sel_d: d,
            })
            .collect();
        Self {
            dihedrals: defs,
            mass_weighted,
            pbc,
            degrees,
            range360,
            reference_mode,
            reference: None,
            results: Vec::new(),
        }
    }
}

impl Plan for DihedralRmsPlan {
    fn name(&self) -> &'static str {
        "dihedral_rms"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.results.clear();
        self.reference = match self.reference_mode {
            ReferenceMode::Topology => {
                let positions0 = system
                    .positions0
                    .as_ref()
                    .ok_or_else(|| TrajError::Mismatch("no topology reference coords".into()))?;
                let masses = &system.atoms.mass;
                let mut refs = Vec::with_capacity(self.dihedrals.len());
                for def in self.dihedrals.iter() {
                    let a = center_of_coords(
                        positions0,
                        &def.sel_a.indices,
                        masses,
                        self.mass_weighted,
                    );
                    let b = center_of_coords(
                        positions0,
                        &def.sel_b.indices,
                        masses,
                        self.mass_weighted,
                    );
                    let c = center_of_coords(
                        positions0,
                        &def.sel_c.indices,
                        masses,
                        self.mass_weighted,
                    );
                    let d = center_of_coords(
                        positions0,
                        &def.sel_d.indices,
                        masses,
                        self.mass_weighted,
                    );
                    let val =
                        dihedral_value(a, b, c, d, self.pbc, None, self.degrees, self.range360)?;
                    refs.push(val);
                }
                Some(refs)
            }
            ReferenceMode::Frame0 => None,
        };
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        if self.dihedrals.is_empty() {
            self.results
                .extend(std::iter::repeat(0.0).take(chunk.n_frames));
            return Ok(());
        }
        let masses = &system.atoms.mass;
        let need_ref =
            self.reference.is_none() && matches!(self.reference_mode, ReferenceMode::Frame0);
        if need_ref {
            if chunk.n_frames == 0 {
                return Ok(());
            }
            let mut refs = Vec::with_capacity(self.dihedrals.len());
            for def in self.dihedrals.iter() {
                let a =
                    center_of_selection(chunk, 0, &def.sel_a.indices, masses, self.mass_weighted);
                let b =
                    center_of_selection(chunk, 0, &def.sel_b.indices, masses, self.mass_weighted);
                let c =
                    center_of_selection(chunk, 0, &def.sel_c.indices, masses, self.mass_weighted);
                let d =
                    center_of_selection(chunk, 0, &def.sel_d.indices, masses, self.mass_weighted);
                let val = dihedral_value(
                    a,
                    b,
                    c,
                    d,
                    self.pbc,
                    Some((chunk, 0)),
                    self.degrees,
                    self.range360,
                )?;
                refs.push(val);
            }
            self.reference = Some(refs);
        }
        let reference = self
            .reference
            .as_ref()
            .ok_or_else(|| TrajError::Mismatch("reference not initialized".into()))?;

        for frame in 0..chunk.n_frames {
            let mut sum = 0.0f64;
            for (idx, def) in self.dihedrals.iter().enumerate() {
                let a = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_a.indices,
                    masses,
                    self.mass_weighted,
                );
                let b = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_b.indices,
                    masses,
                    self.mass_weighted,
                );
                let c = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_c.indices,
                    masses,
                    self.mass_weighted,
                );
                let d = center_of_selection(
                    chunk,
                    frame,
                    &def.sel_d.indices,
                    masses,
                    self.mass_weighted,
                );
                let val = dihedral_value(
                    a,
                    b,
                    c,
                    d,
                    self.pbc,
                    Some((chunk, frame)),
                    self.degrees,
                    self.range360,
                )?;
                let diff = angle_diff(val as f64, reference[idx] as f64, self.degrees);
                sum += diff * diff;
            }
            let rms = (sum / self.dihedrals.len() as f64).sqrt() as f32;
            self.results.push(rms);
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        Ok(PlanOutput::Series(std::mem::take(&mut self.results)))
    }
}
