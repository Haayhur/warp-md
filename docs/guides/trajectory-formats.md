---
description: High-performance trajectory readers supporting 10 standard simulation formats
icon: database
---

# Trajectory Formats

`warp-md` leverages `traj-io`, a native, zero-copy (where possible) trajectory parsing backend optimized for streaming and batch processing. We explicitly support 10 formats for reading and processing molecular dynamics simulations.

## Supported Formats

| Format | Extension | Notes |
| --- | --- | --- |
| **GROMACS XTC** | `.xtc` | Highly compressed, widely used standard. Natively decompressed using optimized routines. |
| **GROMACS TRR** | `.trr` | High-precision trajectory that can include velocities and forces. |
| **GROMACS TNG** | `.tng` | Next-generation GROMACS format with block compression and arbitrary data precision. |
| **NAMD/CHARMM DCD** | `.dcd` | Legacy standard format natively parsed without external C libraries. |
| **GROMACS GRO** | `.gro` | Multi-frame GRO files parsing. |
| **GROMOS96** | `.g96`, `.gromos96` | Multi-frame GROMOS96 trajectory support. |
| **H5MD** | `.h5md` | HDF5-based generic MD format; supports streaming subsets of frames effectively. |
| **PDB Trajectory** | `.pdb` | Multi-model PDB trajectories. |
| **XYZ Trajectory** | `.xyz` | Multi-frame simple XYZ formats. |
| **GROMACS CPT** | `.cpt` | Checkpoint reading for state extraction. |

## Performance and Engine Synergy

The trajectory engine is designed for high-throughput streaming. Instead of loading the entire trajectory into memory, formats are parsed in chunks (or mapped in via `mmap` when possible) and streamed frame-by-frame to `traj-engine`. 

This enables `warp-md` to process arbitrarily large trajectories (terabytes in scale) on minimal memory footprints.

## Seamless Extension Routing

Tools within the `warp-md` framework (such as `warp-cg`, `warp-pack`, or `warp-build`) infer the trajectory parsing logic automatically based on the file extension and the internal magic bytes. 

For complex multi-model structural files like `.pdb` or `.gro`, the parser intelligently yields frames to iterative plans automatically.
