use super::*;
use traj_core::error::TrajResult;
use traj_core::frame::{Box3, FrameChunkBuilder};
use traj_core::interner::StringInterner;
use traj_core::system::{AtomTable, System};
use traj_io::TrajReader;

struct InMemoryTraj {
    n_atoms: usize,
    frames: Vec<Vec<[f32; 4]>>,
    cursor: usize,
}

struct InMemoryTrajWithBox {
    n_atoms: usize,
    frames: Vec<Vec<[f32; 4]>>,
    box_: Box3,
    cursor: usize,
}

impl InMemoryTraj {
    fn new(frames: Vec<Vec<[f32; 4]>>) -> Self {
        let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
        Self {
            n_atoms,
            frames,
            cursor: 0,
        }
    }
}

impl TrajReader for InMemoryTraj {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.frames.len())
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        out.reset(self.n_atoms, max_frames);
        let mut count = 0;
        while self.cursor < self.frames.len() && count < max_frames {
            let coords = out.start_frame(Box3::None, None);
            coords.copy_from_slice(&self.frames[self.cursor]);
            self.cursor += 1;
            count += 1;
        }
        Ok(count)
    }
}

impl InMemoryTrajWithBox {
    fn new(frames: Vec<Vec<[f32; 4]>>, box_: Box3) -> Self {
        let n_atoms = frames.first().map(|f| f.len()).unwrap_or(0);
        Self {
            n_atoms,
            frames,
            box_,
            cursor: 0,
        }
    }
}

impl TrajReader for InMemoryTrajWithBox {
    fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    fn n_frames_hint(&self) -> Option<usize> {
        Some(self.frames.len())
    }

    fn read_chunk(&mut self, max_frames: usize, out: &mut FrameChunkBuilder) -> TrajResult<usize> {
        out.reset(self.n_atoms, max_frames);
        let mut count = 0;
        while self.cursor < self.frames.len() && count < max_frames {
            let coords = out.start_frame(self.box_, None);
            coords.copy_from_slice(&self.frames[self.cursor]);
            self.cursor += 1;
            count += 1;
        }
        Ok(count)
    }
}

fn build_system() -> System {
    let mut interner = StringInterner::new();
    let name = interner.intern_upper("CA");
    let res = interner.intern_upper("ALA");
    let atoms = AtomTable {
        name_id: vec![name, name],
        resname_id: vec![res, res],
        resid: vec![1, 1],
        chain_id: vec![0, 0],
        element_id: vec![0, 0],
        mass: vec![1.0, 1.0],
    };
    let positions0 = Some(vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]);
    System::with_atoms(atoms, interner, positions0)
}

fn build_single_atom_system() -> System {
    let mut interner = StringInterner::new();
    let name = interner.intern_upper("CA");
    let res = interner.intern_upper("ALA");
    let atoms = AtomTable {
        name_id: vec![name],
        resname_id: vec![res],
        resid: vec![1],
        chain_id: vec![0],
        element_id: vec![0],
        mass: vec![1.0],
    };
    let positions0 = Some(vec![[0.0, 0.0, 0.0, 1.0]]);
    System::with_atoms(atoms, interner, positions0)
}

fn build_two_resid_system() -> System {
    let mut interner = StringInterner::new();
    let name_a = interner.intern_upper("DA");
    let name_b = interner.intern_upper("AC");
    let res = interner.intern_upper("SOL");
    let atoms = AtomTable {
        name_id: vec![name_a, name_b],
        resname_id: vec![res, res],
        resid: vec![1, 2],
        chain_id: vec![0, 0],
        element_id: vec![0, 0],
        mass: vec![1.0, 1.0],
    };
    let positions0 = Some(vec![[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]);
    System::with_atoms(atoms, interner, positions0)
}

fn build_plane_system() -> System {
    let mut interner = StringInterner::new();
    let name = interner.intern_upper("CA");
    let res = interner.intern_upper("ALA");
    let atoms = AtomTable {
        name_id: vec![name, name, name],
        resname_id: vec![res, res, res],
        resid: vec![1, 1, 1],
        chain_id: vec![0, 0, 0],
        element_id: vec![0, 0, 0],
        mass: vec![1.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
    ]);
    System::with_atoms(atoms, interner, positions0)
}

fn build_four_resid_system() -> System {
    let mut interner = StringInterner::new();
    let name = interner.intern_upper("CA");
    let res = interner.intern_upper("ALA");
    let atoms = AtomTable {
        name_id: vec![name, name, name, name],
        resname_id: vec![res, res, res, res],
        resid: vec![1, 2, 3, 4],
        chain_id: vec![0, 0, 0, 0],
        element_id: vec![0, 0, 0, 0],
        mass: vec![1.0, 1.0, 1.0, 1.0],
    };
    let positions0 = Some(vec![
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0, 1.0],
    ]);
    System::with_atoms(atoms, interner, positions0)
}

fn build_polymer_system(n_chains: usize, chain_len: usize) -> System {
    let mut interner = StringInterner::new();
    let name = interner.intern_upper("C");
    let res = interner.intern_upper("POL");
    let n_atoms = n_chains * chain_len;
    let mut name_id = Vec::with_capacity(n_atoms);
    let mut resname_id = Vec::with_capacity(n_atoms);
    let mut resid = Vec::with_capacity(n_atoms);
    let mut chain_id = Vec::with_capacity(n_atoms);
    let mut element_id = Vec::with_capacity(n_atoms);
    let mut mass = Vec::with_capacity(n_atoms);
    let mut positions0 = Vec::with_capacity(n_atoms);
    for chain in 0..n_chains {
        for i in 0..chain_len {
            name_id.push(name);
            resname_id.push(res);
            resid.push((i + 1) as i32);
            chain_id.push(chain as u32);
            element_id.push(0);
            mass.push(1.0);
            positions0.push([0.0, 0.0, 0.0, 1.0]);
        }
    }
    let atoms = AtomTable {
        name_id,
        resname_id,
        resid,
        chain_id,
        element_id,
        mass,
    };
    System::with_atoms(atoms, interner, Some(positions0))
}

fn linear_frame(n_chains: usize, chain_len: usize, spacing: f32, y_sep: f32) -> Vec<[f32; 4]> {
    let mut coords = Vec::with_capacity(n_chains * chain_len);
    for chain in 0..n_chains {
        let y = chain as f32 * y_sep;
        for i in 0..chain_len {
            let x = i as f32 * spacing;
            coords.push([x, y, 0.0, 1.0]);
        }
    }
    coords
}

fn right_angle_frame(n_chains: usize, y_sep: f32) -> Vec<[f32; 4]> {
    let mut coords = Vec::with_capacity(n_chains * 3);
    for chain in 0..n_chains {
        let y = chain as f32 * y_sep;
        coords.push([0.0, y, 0.0, 1.0]);
        coords.push([1.0, y, 0.0, 1.0]);
        coords.push([1.0, y + 1.0, 0.0, 1.0]);
    }
    coords
}

include!("part1.rs");
include!("part2.rs");
include!("part3.rs");
include!("part4.rs");
