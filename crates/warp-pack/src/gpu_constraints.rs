#[cfg(feature = "cuda")]
use crate::constraints::{ConstraintMode, ConstraintSpec, ShapeSpec};
#[cfg(feature = "cuda")]
use crate::pack::AtomRecord;
#[cfg(feature = "cuda")]
use traj_gpu::Float4;

#[cfg(feature = "cuda")]
pub(crate) struct ConstraintGpuData {
    pub(crate) types: Vec<u8>,
    pub(crate) modes: Vec<u8>,
    pub(crate) data0: Vec<Float4>,
    pub(crate) data1: Vec<Float4>,
    pub(crate) atom_offsets: Vec<i32>,
    pub(crate) atom_indices: Vec<i32>,
}

#[cfg(feature = "cuda")]
pub(crate) fn build_constraint_gpu_data(
    atoms: &[AtomRecord],
    mol_constraints: &[&[ConstraintSpec]],
) -> Option<ConstraintGpuData> {
    if atoms.is_empty() || mol_constraints.is_empty() {
        return None;
    }

    let mut types = Vec::new();
    let mut modes = Vec::new();
    let mut data0 = Vec::new();
    let mut data1 = Vec::new();
    let mut mol_ranges = Vec::with_capacity(mol_constraints.len());

    for constraints in mol_constraints {
        let start = types.len() as i32;
        for constraint in *constraints {
            let (shape_type, mode_code, d0, d1) = match constraint {
                ConstraintSpec { mode, shape } => {
                    let mut mode_code = match mode {
                        ConstraintMode::Inside => 0u8,
                        ConstraintMode::Outside => 1u8,
                        ConstraintMode::Above | ConstraintMode::Over => 2u8,
                        ConstraintMode::Below => 3u8,
                    };
                    match shape {
                        ShapeSpec::Box { min, max } => {
                            let d0 = Float4 {
                                x: min[0],
                                y: min[1],
                                z: min[2],
                                w: max[0],
                            };
                            let d1 = Float4 {
                                x: max[1],
                                y: max[2],
                                z: 0.0,
                                w: 0.0,
                            };
                            (0u8, mode_code, d0, d1)
                        }
                        ShapeSpec::Cube { center, side } => {
                            let half = *side * 0.5;
                            let min = [center[0] - half, center[1] - half, center[2] - half];
                            let max = [center[0] + half, center[1] + half, center[2] + half];
                            let d0 = Float4 {
                                x: min[0],
                                y: min[1],
                                z: min[2],
                                w: max[0],
                            };
                            let d1 = Float4 {
                                x: max[1],
                                y: max[2],
                                z: 0.0,
                                w: 0.0,
                            };
                            (0u8, mode_code, d0, d1)
                        }
                        ShapeSpec::Sphere { center, radius } => {
                            let d0 = Float4 {
                                x: center[0],
                                y: center[1],
                                z: center[2],
                                w: *radius,
                            };
                            let d1 = Float4::default();
                            (1u8, mode_code, d0, d1)
                        }
                        ShapeSpec::Ellipsoid { center, radii } => {
                            let d0 = Float4 {
                                x: center[0],
                                y: center[1],
                                z: center[2],
                                w: radii[0],
                            };
                            let d1 = Float4 {
                                x: radii[1],
                                y: radii[2],
                                z: 0.0,
                                w: 0.0,
                            };
                            (2u8, mode_code, d0, d1)
                        }
                        ShapeSpec::Cylinder {
                            base,
                            axis,
                            radius,
                            height,
                        } => {
                            let d0 = Float4 {
                                x: base[0],
                                y: base[1],
                                z: base[2],
                                w: *radius,
                            };
                            let d1 = Float4 {
                                x: axis[0],
                                y: axis[1],
                                z: axis[2],
                                w: *height,
                            };
                            (3u8, mode_code, d0, d1)
                        }
                        ShapeSpec::Plane { point, normal } => {
                            if mode_code == 0 {
                                mode_code = 2;
                            } else if mode_code == 1 {
                                mode_code = 3;
                            }
                            let d0 = Float4 {
                                x: point[0],
                                y: point[1],
                                z: point[2],
                                w: 0.0,
                            };
                            let d1 = Float4 {
                                x: normal[0],
                                y: normal[1],
                                z: normal[2],
                                w: 0.0,
                            };
                            (4u8, mode_code, d0, d1)
                        }
                        ShapeSpec::XyGauss {
                            center,
                            sigma,
                            z0,
                            amplitude,
                        } => {
                            if mode_code == 0 {
                                mode_code = 2;
                            } else if mode_code == 1 {
                                mode_code = 3;
                            }
                            let d0 = Float4 {
                                x: center[0],
                                y: center[1],
                                z: *z0,
                                w: *amplitude,
                            };
                            let d1 = Float4 {
                                x: sigma[0],
                                y: sigma[1],
                                z: 0.0,
                                w: 0.0,
                            };
                            (5u8, mode_code, d0, d1)
                        }
                    }
                }
            };
            types.push(shape_type);
            modes.push(mode_code);
            data0.push(d0);
            data1.push(d1);
        }
        let end = types.len() as i32;
        mol_ranges.push((start, end));
    }

    if types.is_empty() {
        return None;
    }

    let mut atom_offsets = vec![0i32; atoms.len() + 1];
    let mut atom_indices = Vec::new();
    let mut offset = 0i32;
    for (i, atom) in atoms.iter().enumerate() {
        atom_offsets[i] = offset;
        let mol_idx = atom.mol_id.max(1) as usize - 1;
        if mol_idx < mol_ranges.len() {
            let (start, end) = mol_ranges[mol_idx];
            for idx in start..end {
                atom_indices.push(idx);
                offset += 1;
            }
        }
    }
    atom_offsets[atoms.len()] = offset;

    Some(ConstraintGpuData {
        types,
        modes,
        data0,
        data1,
        atom_offsets,
        atom_indices,
    })
}
