use traj_core::error::TrajResult;
use traj_core::frame::FrameChunk;
use traj_core::selection::Selection;
use traj_core::system::System;

use crate::executor::{Device, Plan, PlanOutput};
use crate::plans::analysis::secondary_structure::{
    build_backbone_model, compute_backbone_frame, BackboneModel,
};

pub const DSSP_OUTPUT_AVG_KEYS: [&str; 8] = [
    "none_avg",
    "extended_avg",
    "bridge_avg",
    "3-10_avg",
    "alpha_avg",
    "pi_avg",
    "turn_avg",
    "bend_avg",
];

pub struct DsspPlan {
    selection: Selection,
    model: BackboneModel,
    codes: Vec<u8>,
    frames: usize,
}

impl DsspPlan {
    pub fn new(selection: Selection) -> Self {
        Self {
            selection,
            model: BackboneModel::default(),
            codes: Vec::new(),
            frames: 0,
        }
    }

    pub fn labels(&self) -> &[String] {
        &self.model.labels
    }
}

pub fn dssp_internal_to_output_code(code: u8) -> u8 {
    match code {
        0 => 0,
        1 => 4, // H/alpha
        2 => 2, // B/bridge
        3 => 1, // E/extended
        4 => 3, // G/3-10
        5 => 5, // I/pi
        6 => 6, // T/turn
        7 => 7, // S/bend
        _ => 0,
    }
}

pub fn dssp_output_code_to_symbol(code: u8, simplified: bool) -> &'static str {
    if simplified {
        match code {
            1 | 2 => "E",
            3..=5 => "H",
            _ => "C",
        }
    } else {
        match code {
            1 => "b",
            2 => "B",
            3 => "G",
            4 => "H",
            5 => "I",
            6 => "T",
            7 => "S",
            _ => "0",
        }
    }
}

pub fn dssp_output_average_fractions(codes: &[u8], rows: usize, cols: usize) -> Vec<f32> {
    let mut fractions = vec![0.0f32; DSSP_OUTPUT_AVG_KEYS.len() * cols];
    if rows == 0 || cols == 0 {
        return fractions;
    }
    for row in 0..rows {
        let row_offset = row * cols;
        for col in 0..cols {
            let code = codes[row_offset + col].min(7) as usize;
            fractions[code * cols + col] += 1.0;
        }
    }
    let norm = rows as f32;
    for value in fractions.iter_mut() {
        *value /= norm;
    }
    fractions
}

impl Plan for DsspPlan {
    fn name(&self) -> &'static str {
        "dssp"
    }

    fn init(&mut self, system: &System, _device: &Device) -> TrajResult<()> {
        self.codes.clear();
        self.frames = 0;
        self.model = build_backbone_model(system, &self.selection);
        Ok(())
    }

    fn process_chunk(
        &mut self,
        chunk: &FrameChunk,
        _system: &System,
        _device: &Device,
    ) -> TrajResult<()> {
        let n_res = self.model.residues.len();
        if n_res == 0 {
            return Ok(());
        }
        for frame in 0..chunk.n_frames {
            let frame_data = compute_backbone_frame(&self.model, chunk, frame);
            self.codes.extend(frame_data.states.into_iter());
            self.frames += 1;
        }
        Ok(())
    }

    fn finalize(&mut self) -> TrajResult<PlanOutput> {
        let n_res = self.model.residues.len();
        if self.frames == 0 || n_res == 0 {
            return Ok(PlanOutput::Matrix {
                data: Vec::new(),
                rows: 0,
                cols: n_res,
            });
        }
        let data: Vec<f32> = self.codes.iter().map(|&value| value as f32).collect();
        Ok(PlanOutput::Matrix {
            data,
            rows: self.frames,
            cols: n_res,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dssp_internal_to_output_code, dssp_output_average_fractions, dssp_output_code_to_symbol,
    };

    #[test]
    fn dssp_maps_internal_codes_to_standard_output_order() {
        let internal = [0_u8, 1, 2, 3, 4, 5, 6, 7];
        let mapped: Vec<u8> = internal
            .iter()
            .copied()
            .map(dssp_internal_to_output_code)
            .collect();
        assert_eq!(mapped, vec![0, 4, 2, 1, 3, 5, 6, 7]);

        let full: Vec<&str> = mapped
            .iter()
            .copied()
            .map(|code| dssp_output_code_to_symbol(code, false))
            .collect();
        assert_eq!(full, vec!["0", "H", "B", "b", "G", "I", "T", "S"]);

        let simplified: Vec<&str> = mapped
            .iter()
            .copied()
            .map(|code| dssp_output_code_to_symbol(code, true))
            .collect();
        assert_eq!(simplified, vec!["C", "H", "E", "E", "H", "H", "C", "C"]);
    }

    #[test]
    fn dssp_average_fractions_are_per_residue_by_output_code() {
        let codes = vec![
            0_u8, 4, //
            4, 4, //
            1, 0, //
        ];
        let fractions = dssp_output_average_fractions(&codes, 3, 2);
        assert_eq!(fractions.len(), 16);
        assert!((fractions[0] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((fractions[1] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((fractions[2] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((fractions[8] - (1.0 / 3.0)).abs() < 1e-6);
        assert!((fractions[9] - (2.0 / 3.0)).abs() < 1e-6);
    }
}
