use std::cmp::Ordering;
use std::path::Path;
use std::time::Instant;

use rand::Rng;

use crate::atom_params::AtomParams;
use crate::config::{PackConfig, StructureSpec};
use crate::constraints::{satisfies_constraints, ConstraintMode, ShapeSpec};
use crate::error::{PackError, PackResult};
use crate::geom::{center_of_geometry, Quaternion, Vec3};
use crate::io::{read_molecule, write_output};
use crate::pack::{AtomRecord, PackOutput};
use crate::pbc::PbcBox;
use crate::spatial_hash::{SpatialHash, SpatialHashParamsExt, SpatialHashV2};

pub(crate) struct MoleculeTemplate {
    pub(crate) atoms: Vec<AtomRecord>,
    pub(crate) bonds: Vec<(usize, usize)>,
    pub(crate) ter_after: Vec<usize>,
    pub(crate) radius: f32,
    pub(crate) dmax: f32,
    pub(crate) resid_min: i32,
    pub(crate) resid_count: i32,
}

pub(crate) struct TemplateEntry {
    pub(crate) template: MoleculeTemplate,
    pub(crate) res_mode: i32,
    pub(crate) connect: bool,
    pub(crate) maxmove: usize,
    pub(crate) atom_params: Vec<AtomParams>,
}

pub(crate) struct Placement {
    pub(crate) spec_index: usize,
    pub(crate) local_index: i32,
}

pub(crate) struct PlacementSeed {
    pub(crate) center: Vec3,
    pub(crate) rotation: Quaternion,
}

pub(crate) fn load_template(
    spec: &StructureSpec,
    mol_id: i32,
    format: Option<&str>,
    ignore_conect: bool,
    non_standard_conect: bool,
) -> PackResult<MoleculeTemplate> {
    let mut molecule = read_molecule(
        Path::new(&spec.path),
        format,
        ignore_conect,
        non_standard_conect,
        spec.topology.as_deref().map(Path::new),
    )?;
    if let Some(name) = &spec.name {
        for atom in molecule.atoms.iter_mut() {
            atom.resname = name.clone();
        }
    }
    if let Some(chain) = &spec.chain {
        let ch = chain.chars().next().unwrap_or('A');
        for atom in molecule.atoms.iter_mut() {
            atom.chain = ch;
        }
    }
    if let Some(segid) = &spec.segid {
        for atom in molecule.atoms.iter_mut() {
            atom.segid = segid.clone();
        }
    }
    if spec.center {
        let center = center_of_geometry(
            &molecule
                .atoms
                .iter()
                .map(|a| a.position)
                .collect::<Vec<_>>(),
        );
        for atom in molecule.atoms.iter_mut() {
            atom.position = atom.position.sub(center);
        }
    }
    if let Some(shift) = spec.translate {
        let delta = Vec3::from_array(shift);
        for atom in molecule.atoms.iter_mut() {
            atom.position = atom.position.add(delta);
        }
    }
    for atom in molecule.atoms.iter_mut() {
        atom.mol_id = mol_id;
    }
    let radius = molecule
        .atoms
        .iter()
        .map(|a| a.position.norm())
        .fold(0.0f32, |a, b| a.max(b));
    let mut dmax2 = 0.0f32;
    for i in 0..molecule.atoms.len() {
        for j in (i + 1)..molecule.atoms.len() {
            let diff = molecule.atoms[i].position.sub(molecule.atoms[j].position);
            dmax2 = dmax2.max(diff.dot(diff));
        }
    }
    let mut dmax = dmax2.sqrt();
    if dmax == 0.0 {
        dmax = 1.0;
    }
    let (resid_min, resid_count) = residue_stats(&molecule.atoms);
    Ok(MoleculeTemplate {
        atoms: molecule.atoms,
        bonds: molecule.bonds,
        ter_after: molecule.ter_after,
        radius,
        dmax,
        resid_min,
        resid_count,
    })
}

pub(crate) fn random_center<R: Rng + ?Sized>(
    rng: &mut R,
    box_size: [f32; 3],
    radius: f32,
) -> PackResult<Vec3> {
    for i in 0..3 {
        if box_size[i] <= 2.0 * radius {
            return Err(PackError::Invalid(
                "structure radius exceeds box size".into(),
            ));
        }
    }
    let x = rng.gen_range(radius..(box_size[0] - radius));
    let y = rng.gen_range(radius..(box_size[1] - radius));
    let z = rng.gen_range(radius..(box_size[2] - radius));
    Ok(Vec3::new(x, y, z))
}

pub(crate) fn random_center_for_structure<R: Rng + ?Sized>(
    rng: &mut R,
    spec: &StructureSpec,
    box_size: [f32; 3],
    box_origin: Vec3,
    radius: f32,
) -> PackResult<Vec3> {
    let mut lo = [
        box_origin.x + radius,
        box_origin.y + radius,
        box_origin.z + radius,
    ];
    let mut hi = [
        box_origin.x + box_size[0] - radius,
        box_origin.y + box_size[1] - radius,
        box_origin.z + box_size[2] - radius,
    ];
    let mut saw_inside_box = false;
    for constraint in &spec.constraints {
        match (&constraint.mode, &constraint.shape) {
            (ConstraintMode::Inside, ShapeSpec::Box { min, max }) => {
                saw_inside_box = true;
                for axis in 0..3 {
                    lo[axis] = lo[axis].max(min[axis]);
                    hi[axis] = hi[axis].min(max[axis]);
                }
            }
            _ => {
                let mut center = random_center(rng, box_size, radius)?;
                center = center.add(box_origin);
                return Ok(center);
            }
        }
    }
    if saw_inside_box && (0..3).all(|axis| hi[axis] > lo[axis]) {
        return Ok(Vec3::new(
            rng.gen_range(lo[0]..hi[0]),
            rng.gen_range(lo[1]..hi[1]),
            rng.gen_range(lo[2]..hi[2]),
        ));
    }
    let mut center = random_center(rng, box_size, radius)?;
    center = center.add(box_origin);
    Ok(center)
}

pub(crate) fn transform_atoms(
    template: &MoleculeTemplate,
    rotation: Quaternion,
    center: Vec3,
    mol_id: i32,
) -> Vec<AtomRecord> {
    template
        .atoms
        .iter()
        .map(|a| AtomRecord {
            record_kind: a.record_kind,
            name: a.name.clone(),
            element: a.element.clone(),
            resname: a.resname.clone(),
            resid: a.resid,
            chain: a.chain,
            segid: a.segid.clone(),
            charge: a.charge,
            position: rotation.rotate_vec(a.position).add(center),
            mol_id,
        })
        .collect()
}

pub(crate) fn transform_positions_into(
    template: &MoleculeTemplate,
    rotation: Quaternion,
    center: Vec3,
    out: &mut Vec<Vec3>,
) {
    out.clear();
    if out.capacity() < template.atoms.len() {
        out.reserve(template.atoms.len() - out.capacity());
    }
    for atom in &template.atoms {
        out.push(rotation.rotate_vec(atom.position).add(center));
    }
}

pub(crate) fn place_fixed(
    spec: &StructureSpec,
    template: &MoleculeTemplate,
    template_params: &[AtomParams],
    seeds: &[PlacementSeed],
    atoms: &mut Vec<AtomRecord>,
    positions: &mut Vec<Vec3>,
    params: &mut Vec<AtomParams>,
    hash: &mut SpatialHashV2,
    mol_id_start: i32,
    pbc: Option<PbcBox>,
    res_mode: i32,
    global_resid_counter: &mut i32,
    bonds: &mut Vec<(usize, usize)>,
    ter_after: &mut Vec<usize>,
    connect: bool,
    add_amber_ter: bool,
    amber_ter_preserve: bool,
    avoid_overlap: bool,
) -> PackResult<()> {
    for (i, seed) in seeds.iter().enumerate() {
        let center = seed.center;
        let mol_id = mol_id_start + i as i32;
        let mut candidate = transform_atoms(template, seed.rotation, center, mol_id);
        apply_chain(&mut candidate, spec, mol_id);
        let cand_positions: Vec<Vec3> = candidate.iter().map(|a| a.position).collect();
        if !satisfies_structure_constraints(&cand_positions, spec, pbc) {
            return Err(PackError::Placement(
                "fixed placement violates constraints".into(),
            ));
        }
        if avoid_overlap
            && hash.overlaps_params(&cand_positions, template_params, positions, params)
        {
            return Err(PackError::Placement("fixed placement overlaps".into()));
        }
        apply_resnumbers(
            &mut candidate,
            res_mode,
            template.resid_min,
            *global_resid_counter,
            (i + 1) as i32,
            mol_id,
        )?;
        *global_resid_counter += template.resid_count;
        let base = atoms.len();
        if connect {
            append_bonds(bonds, &template.bonds, base);
        }
        if amber_ter_preserve {
            append_ter_after(ter_after, &template.ter_after, base);
        }
        for (offset, atom) in candidate.into_iter().enumerate() {
            let idx = positions.len();
            positions.push(atom.position);
            params.push(template_params[offset]);
            if avoid_overlap {
                hash.insert(idx, atom.position);
            }
            atoms.push(atom);
        }
        if add_amber_ter {
            if let Some(last) = atoms.len().checked_sub(1) {
                ter_after.push(last);
            }
        }
    }
    Ok(())
}

pub(crate) fn build_order(cfg: &PackConfig, templates: &[TemplateEntry]) -> Vec<Placement> {
    let mut order = Vec::new();
    for (idx, spec) in cfg.structures.iter().enumerate() {
        if spec.fixed {
            continue;
        }
        for local_index in 1..=spec.count {
            order.push(Placement {
                spec_index: idx,
                local_index: local_index as i32,
            });
        }
    }
    order.sort_by(|a, b| {
        let ra = templates
            .get(a.spec_index)
            .map(|t| t.template.radius)
            .unwrap_or(0.0);
        let rb = templates
            .get(b.spec_index)
            .map(|t| t.template.radius)
            .unwrap_or(0.0);
        rb.partial_cmp(&ra)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.spec_index.cmp(&b.spec_index))
            .then_with(|| a.local_index.cmp(&b.local_index))
    });
    order
}

pub(crate) fn satisfies_structure_constraints(
    points: &[Vec3],
    spec: &StructureSpec,
    pbc: Option<PbcBox>,
) -> bool {
    if spec.constraints.is_empty() && spec.atom_constraints.is_empty() {
        return true;
    }
    if spec.atom_constraints.is_empty()
        && spec.constraints.iter().all(|c| {
            matches!(
                (&c.mode, &c.shape),
                (ConstraintMode::Inside, ShapeSpec::Box { .. })
            )
        })
    {
        for point in points.iter().copied() {
            let p = if let Some(pbc_box) = pbc {
                pbc_box.wrap(point)
            } else {
                point
            };
            for constraint in &spec.constraints {
                if let ShapeSpec::Box { min, max } = &constraint.shape {
                    if p.x < min[0]
                        || p.x > max[0]
                        || p.y < min[1]
                        || p.y > max[1]
                        || p.z < min[2]
                        || p.z > max[2]
                    {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    if !satisfies_constraints(points, &spec.constraints, pbc) {
        return false;
    }
    for atom_constraint in &spec.atom_constraints {
        for &idx1 in &atom_constraint.indices {
            if idx1 == 0 || idx1 > points.len() {
                return false;
            }
            let point = [points[idx1 - 1]];
            if !satisfies_constraints(
                &point,
                std::slice::from_ref(&atom_constraint.constraint),
                pbc,
            ) {
                return false;
            }
        }
    }
    true
}

pub(crate) fn validate_min_distance(atoms: &[AtomRecord], min_dist: f32) -> PackResult<()> {
    // Compute bounds from atom positions for SpatialHashV2
    let mut box_min = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut box_max = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for atom in atoms {
        let pos = atom.position;
        box_min.x = box_min.x.min(pos.x);
        box_min.y = box_min.y.min(pos.y);
        box_min.z = box_min.z.min(pos.z);
        box_max.x = box_max.x.max(pos.x);
        box_max.y = box_max.y.max(pos.y);
        box_max.z = box_max.z.max(pos.z);
    }
    // Add padding to avoid edge cases
    let pad = min_dist.max(1.0e-6);
    box_min = box_min.sub(Vec3::new(pad, pad, pad));
    box_max = box_max.add(Vec3::new(pad, pad, pad));

    let mut hash = SpatialHashV2::new(min_dist, box_min, box_max);
    let mut seen: Vec<Vec3> = Vec::with_capacity(atoms.len());
    let min2 = min_dist * min_dist;
    for atom in atoms {
        let pos = atom.position;
        let mut overlap = false;
        hash.for_each_neighbor(pos, |idx| {
            if atoms[idx].mol_id == atom.mol_id {
                return;
            }
            let d = pos.sub(seen[idx]);
            if d.dot(d) < min2 {
                overlap = true;
            }
        });
        if overlap {
            return Err(PackError::Placement(
                "check failed: overlaps detected".into(),
            ));
        }
        let idx = seen.len();
        seen.push(pos);
        hash.insert(idx, pos);
    }
    Ok(())
}

pub(crate) fn apply_resnumbers(
    atoms: &mut [AtomRecord],
    mode: i32,
    resid_min: i32,
    resid_base: i32,
    local_mol_index: i32,
    global_mol_index: i32,
) -> PackResult<()> {
    match mode {
        0 => {
            for atom in atoms.iter_mut() {
                atom.resid = local_mol_index;
            }
        }
        1 => {}
        2 => {
            for atom in atoms.iter_mut() {
                atom.resid = atom.resid - resid_min + resid_base;
            }
        }
        3 => {
            for atom in atoms.iter_mut() {
                atom.resid = global_mol_index;
            }
        }
        _ => {
            return Err(PackError::Invalid(
                "resnumbers must be 0, 1, 2, or 3".into(),
            ))
        }
    }
    Ok(())
}

pub(crate) fn apply_chain(atoms: &mut [AtomRecord], spec: &StructureSpec, mol_id: i32) {
    let chain_override = if let Some(chain) = &spec.chain {
        chain.chars().next()
    } else if spec.changechains {
        Some(chain_from_index(mol_id))
    } else {
        None
    };
    if let Some(ch) = chain_override {
        for atom in atoms.iter_mut() {
            atom.chain = ch;
        }
    }
}

fn chain_from_index(idx: i32) -> char {
    let symbols: Vec<char> = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890".chars().collect();
    let i = idx.max(1) as usize;
    if i <= symbols.len() {
        symbols[i - 1]
    } else {
        '#'
    }
}

pub(crate) fn append_bonds(
    out: &mut Vec<(usize, usize)>,
    template: &[(usize, usize)],
    offset: usize,
) {
    for &(a, b) in template {
        out.push((a + offset, b + offset));
    }
}

pub(crate) fn append_ter_after(out: &mut Vec<usize>, template: &[usize], offset: usize) {
    for &idx in template {
        out.push(idx + offset);
    }
}

pub(crate) fn maybe_write_snapshot(
    cfg: &PackConfig,
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    ter_after: &[usize],
    box_size: [f32; 3],
    last_write: &mut Instant,
) -> PackResult<()> {
    let Some(interval) = cfg.writeout else {
        return Ok(());
    };
    if interval <= 0.0 {
        return Ok(());
    }
    let Some(spec) = cfg.output.as_ref() else {
        return Ok(());
    };
    if last_write.elapsed().as_secs_f32() < interval {
        return Ok(());
    }
    let out = PackOutput {
        atoms: atoms.to_vec(),
        bonds: bonds.to_vec(),
        box_size,
        ter_after: ter_after.to_vec(),
    };
    let add_box_sides = cfg.add_box_sides || cfg.pbc;
    let box_fix = cfg.add_box_sides_fix.unwrap_or(0.0);
    let write_conect = !cfg.ignore_conect;
    write_output(
        &out,
        spec,
        add_box_sides,
        box_fix,
        write_conect,
        cfg.hexadecimal_indices,
    )?;
    *last_write = Instant::now();
    Ok(())
}

pub(crate) fn write_bad_snapshot(
    cfg: &PackConfig,
    atoms: &[AtomRecord],
    bonds: &[(usize, usize)],
    ter_after: &[usize],
    box_size: [f32; 3],
) -> PackResult<()> {
    let Some(spec) = cfg.output.as_ref() else {
        return Ok(());
    };
    let out = PackOutput {
        atoms: atoms.to_vec(),
        bonds: bonds.to_vec(),
        box_size,
        ter_after: ter_after.to_vec(),
    };
    let add_box_sides = cfg.add_box_sides || cfg.pbc;
    let box_fix = cfg.add_box_sides_fix.unwrap_or(0.0);
    let write_conect = !cfg.ignore_conect;
    write_output(
        &out,
        spec,
        add_box_sides,
        box_fix,
        write_conect,
        cfg.hexadecimal_indices,
    )?;
    Ok(())
}

fn residue_stats(atoms: &[AtomRecord]) -> (i32, i32) {
    let mut resids: Vec<i32> = atoms.iter().map(|a| a.resid).collect();
    if resids.is_empty() {
        return (1, 1);
    }
    resids.sort_unstable();
    resids.dedup();
    let resid_min = *resids.first().unwrap_or(&1);
    let resid_count = resids.len().max(1) as i32;
    (resid_min, resid_count)
}
