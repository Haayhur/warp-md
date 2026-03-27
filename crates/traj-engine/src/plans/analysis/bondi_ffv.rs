use traj_core::error::{TrajError, TrajResult};
use traj_core::frame::{Box3, FrameChunk};
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::analysis::surf::resolve_radii;

const DEFAULT_BONDI_SCALE: f64 = 1.3;
const DEFAULT_NINSERT_PER_NM3: usize = 1000;
const DALTON_TO_G_CM3_PER_A3: f64 = 1.660_539_066_60;

pub struct BondiFfvPlan {
    selection: Selection,
    length_scale: f64,
    bondi_scale: f64,
    probe_radius: f64,
    ninsert_per_nm3: usize,
    seed: i64,
    radii_a: Vec<f64>,
    molar_mass_dalton: f64,
    time: Vec<f32>,
    data: Vec<f32>,
    frames_seen: usize,
}

impl BondiFfvPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            length_scale: 1.0,
            bondi_scale: DEFAULT_BONDI_SCALE,
            probe_radius: 0.0,
            ninsert_per_nm3: DEFAULT_NINSERT_PER_NM3,
            seed: 0,
            radii_a: Vec::new(),
            molar_mass_dalton: 0.0,
            time: Vec::new(),
            data: Vec::new(),
            frames_seen: 0,
        }
    }

    pub fn with_length_scale(mut self, scale: f64) -> Self {
        self.length_scale = scale;
        self
    }

    pub fn with_bondi_scale(mut self, scale: f64) -> Self {
        self.bondi_scale = scale;
        self
    }

    pub fn with_probe_radius(mut self, radius: f64) -> Self {
        self.probe_radius = radius.max(0.0);
        self
    }

    pub fn with_ninsert_per_nm3(mut self, ninsert: usize) -> Self {
        self.ninsert_per_nm3 = ninsert.max(1);
        self
    }

    pub fn with_seed(mut self, seed: i64) -> Self {
        self.seed = seed;
        self
    }

    pub fn bondi_scale(&self) -> f64 {
        self.bondi_scale
    }

    pub fn molar_mass_dalton(&self) -> f64 {
        self.molar_mass_dalton
    }

    pub fn probe_radius(&self) -> f64 {
        self.probe_radius
    }

    pub fn ninsert_per_nm3(&self) -> usize {
        self.ninsert_per_nm3
    }

    pub fn seed(&self) -> i64 {
        self.seed
    }
}

impl Plan for BondiFfvPlan {
    fn name(&self) -> &'static str {
        "bondi_ffv"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        if !(self.length_scale.is_finite() && self.length_scale > 0.0) {
            return Err(TrajError::Mismatch(
                "bondi_ffv requires a positive length_scale".into(),
            ));
        }
        if !(self.bondi_scale.is_finite() && self.bondi_scale > 0.0) {
            return Err(TrajError::Mismatch(
                "bondi_ffv requires a positive bondi_scale".into(),
            ));
        }
        if !(self.probe_radius.is_finite() && self.probe_radius >= 0.0) {
            return Err(TrajError::Mismatch(
                "bondi_ffv requires a non-negative probe_radius".into(),
            ));
        }
        if self.ninsert_per_nm3 == 0 {
            return Err(TrajError::Mismatch(
                "bondi_ffv requires ninsert_per_nm3 >= 1".into(),
            ));
        }

        self.time.clear();
        self.data.clear();
        self.frames_seen = 0;
        self.radii_a = resolve_radii(system, &self.selection.indices, None)?;
        self.molar_mass_dalton = self
            .selection
            .indices
            .iter()
            .map(|&idx| system.atoms.mass.get(idx as usize).copied().unwrap_or(0.0) as f64)
            .sum();
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_atoms = chunk.n_atoms;
        for frame in 0..chunk.n_frames {
            let [lx, ly, lz] =
                orthorhombic_box_lengths_a(chunk.box_.get(frame).copied(), self.length_scale)?;
            let total_volume_a3 = lx * ly * lz;
            let total_volume_nm3 = total_volume_a3 / 1000.0;
            let ninsert = ((self.ninsert_per_nm3 as f64) * total_volume_nm3)
                .round()
                .max(1.0) as usize;
            let mut rng = SplitMix64::new(seed_for_frame(self.seed, self.frames_seen + frame));
            let mut free = 0usize;
            let frame_offset = frame * n_atoms;

            'samples: for _ in 0..ninsert {
                let probe = [
                    rng.next_f64() * lx,
                    rng.next_f64() * ly,
                    rng.next_f64() * lz,
                ];
                for (sel_i, &idx) in self.selection.indices.iter().enumerate() {
                    let coord = chunk.coords[frame_offset + idx as usize];
                    let atom = [
                        coord[0] as f64 * self.length_scale,
                        coord[1] as f64 * self.length_scale,
                        coord[2] as f64 * self.length_scale,
                    ];
                    let cutoff = self.radii_a[sel_i] + self.probe_radius;
                    if orthorhombic_distance2(atom, probe, [lx, ly, lz]) < cutoff * cutoff {
                        continue 'samples;
                    }
                }
                free += 1;
            }

            let raw_free_volume_fraction = free as f64 / ninsert as f64;
            let raw_free_volume_a3 = total_volume_a3 * raw_free_volume_fraction;
            let vdw_volume_a3 = total_volume_a3 - raw_free_volume_a3;
            let fractional_free_volume = 1.0 - self.bondi_scale * (1.0 - raw_free_volume_fraction);
            let density_g_cm3 =
                self.molar_mass_dalton * DALTON_TO_G_CM3_PER_A3 / total_volume_a3.max(1.0e-12);

            self.time.push(
                chunk
                    .time_ps
                    .as_ref()
                    .and_then(|times| times.get(frame).copied())
                    .unwrap_or((self.frames_seen + frame) as f32),
            );
            self.data.extend_from_slice(&[
                total_volume_a3 as f32,
                vdw_volume_a3 as f32,
                raw_free_volume_a3 as f32,
                raw_free_volume_fraction as f32,
                fractional_free_volume as f32,
                density_g_cm3 as f32,
            ]);
        }
        self.frames_seen += chunk.n_frames;
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let rows = self.time.len();
        Ok(PlanOutput::TimeSeries {
            time: std::mem::take(&mut self.time),
            data: std::mem::take(&mut self.data),
            rows,
            cols: 6,
        })
    }
}

fn orthorhombic_box_lengths_a(box_: Option<Box3>, length_scale: f64) -> TrajResult<[f64; 3]> {
    match box_.unwrap_or(Box3::None) {
        Box3::Orthorhombic { lx, ly, lz } => Ok([
            lx as f64 * length_scale,
            ly as f64 * length_scale,
            lz as f64 * length_scale,
        ]),
        Box3::None => Err(TrajError::Mismatch(
            "bondi_ffv requires box metadata".into(),
        )),
        Box3::Triclinic { .. } => Err(TrajError::Unsupported(
            "bondi_ffv currently requires an orthorhombic box".into(),
        )),
    }
}

fn orthorhombic_distance2(a: [f64; 3], b: [f64; 3], box_lengths: [f64; 3]) -> f64 {
    let mut dx = (a[0] - b[0]).abs();
    let mut dy = (a[1] - b[1]).abs();
    let mut dz = (a[2] - b[2]).abs();
    dx = dx.min(box_lengths[0] - dx);
    dy = dy.min(box_lengths[1] - dy);
    dz = dz.min(box_lengths[2] - dz);
    dx * dx + dy * dy + dz * dz
}

fn seed_for_frame(seed: i64, frame: usize) -> u64 {
    (seed as u64) ^ ((frame as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 / ((1u64 << 53) as f64)
    }
}
