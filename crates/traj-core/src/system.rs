use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{TrajError, TrajResult};
use crate::interner::StringInterner;
use crate::selection::{compile_selection, Selection, SelectionContext};

#[derive(Debug, Default, Clone)]
pub struct AtomTable {
    pub name_id: Vec<u32>,
    pub resname_id: Vec<u32>,
    pub resid: Vec<i32>,
    pub chain_id: Vec<u32>,
    pub element_id: Vec<u32>,
    pub mass: Vec<f32>,
}

impl AtomTable {
    pub fn len(&self) -> usize {
        self.name_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.name_id.is_empty()
    }
}

#[derive(Debug, Default, Clone)]
pub struct System {
    pub atoms: AtomTable,
    pub interner: StringInterner,
    pub positions0: Option<Vec<[f32; 4]>>,
    pub bonds: Vec<(u32, u32)>,
    pub molecule_id: Vec<i32>,
    pub gb_radius: Vec<f32>,
    pub parse_radius: Vec<f32>,
    pub vdw_radius: Vec<f32>,
    selection_cache: HashMap<String, Selection>,
}

impl System {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_atoms(
        atoms: AtomTable,
        interner: StringInterner,
        positions0: Option<Vec<[f32; 4]>>,
    ) -> Self {
        Self {
            atoms,
            interner,
            positions0,
            bonds: Vec::new(),
            molecule_id: Vec::new(),
            gb_radius: Vec::new(),
            parse_radius: Vec::new(),
            vdw_radius: Vec::new(),
            selection_cache: HashMap::new(),
        }
    }

    pub fn with_topology_metadata(
        mut self,
        bonds: Vec<(usize, usize)>,
        molecule_id: Vec<i32>,
    ) -> TrajResult<Self> {
        self.set_topology_metadata(bonds, molecule_id)?;
        Ok(self)
    }

    pub fn set_topology_metadata(
        &mut self,
        bonds: Vec<(usize, usize)>,
        molecule_id: Vec<i32>,
    ) -> TrajResult<()> {
        self.bonds = normalize_bonds(self.n_atoms(), bonds)?;
        self.molecule_id = if molecule_id.is_empty() {
            molecule_ids_from_bonds(self.n_atoms(), &self.bonds)
        } else {
            if molecule_id.len() != self.n_atoms() {
                return Err(TrajError::Mismatch(
                    "molecule_id length does not match atom count".into(),
                ));
            }
            molecule_id
        };
        Ok(())
    }

    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
    }

    pub fn molecule_id_for_atom(&self, atom_idx: usize) -> Option<i32> {
        self.molecule_id.get(atom_idx).copied()
    }

    pub fn set_gb_radii(&mut self, radii: Vec<f32>) -> TrajResult<()> {
        validate_radii(self.n_atoms(), &radii, "gb_radius")?;
        self.gb_radius = radii;
        Ok(())
    }

    pub fn gb_radius_for_atom(&self, atom_idx: usize) -> Option<f32> {
        self.gb_radius.get(atom_idx).copied()
    }

    pub fn set_parse_radii(&mut self, radii: Vec<f32>) -> TrajResult<()> {
        validate_radii(self.n_atoms(), &radii, "parse_radius")?;
        self.parse_radius = radii;
        Ok(())
    }

    pub fn parse_radius_for_atom(&self, atom_idx: usize) -> Option<f32> {
        self.parse_radius.get(atom_idx).copied()
    }

    pub fn set_vdw_radii(&mut self, radii: Vec<f32>) -> TrajResult<()> {
        validate_radii(self.n_atoms(), &radii, "vdw_radius")?;
        self.vdw_radius = radii;
        Ok(())
    }

    pub fn vdw_radius_for_atom(&self, atom_idx: usize) -> Option<f32> {
        self.vdw_radius.get(atom_idx).copied()
    }

    pub fn bonded_neighbor_count(&self, atom_idx: usize, non_h_only: bool) -> Option<usize> {
        if self.bonds.is_empty() || atom_idx >= self.n_atoms() {
            return None;
        }
        let mut count = 0usize;
        for &(a, b) in self.bonds.iter() {
            let other = if a as usize == atom_idx {
                b as usize
            } else if b as usize == atom_idx {
                a as usize
            } else {
                continue;
            };
            if non_h_only && self.atom_symbol(other).eq_ignore_ascii_case("H") {
                continue;
            }
            count += 1;
        }
        Some(count)
    }

    pub fn atom_symbol(&self, atom_idx: usize) -> String {
        self.atoms
            .element_id
            .get(atom_idx)
            .and_then(|id| self.interner.resolve(*id))
            .unwrap_or("")
            .trim()
            .to_ascii_uppercase()
    }

    pub fn select(&mut self, expr: &str) -> TrajResult<Selection> {
        if let Some(sel) = self.selection_cache.get(expr) {
            return Ok(Selection {
                expr: sel.expr.clone(),
                indices: Arc::clone(&sel.indices),
            });
        }
        let compiled = compile_selection(expr, self)?;
        self.selection_cache
            .insert(expr.to_string(), compiled.clone());
        Ok(compiled)
    }

    pub fn selection_cache_len(&self) -> usize {
        self.selection_cache.len()
    }

    pub fn validate_positions0(&self) -> Result<(), TrajError> {
        if let Some(pos) = &self.positions0 {
            if pos.len() != self.n_atoms() {
                return Err(TrajError::Mismatch(
                    "positions0 length does not match atom count".into(),
                ));
            }
        }
        Ok(())
    }
}

fn validate_radii(n_atoms: usize, radii: &[f32], name: &str) -> TrajResult<()> {
    if radii.is_empty() {
        return Ok(());
    }
    if radii.len() != n_atoms {
        return Err(TrajError::Mismatch(format!(
            "{name} length does not match atom count"
        )));
    }
    if radii.iter().any(|value| !value.is_finite() || *value < 0.0) {
        return Err(TrajError::Mismatch(format!(
            "{name} values must be finite and non-negative"
        )));
    }
    Ok(())
}

fn normalize_bonds(n_atoms: usize, bonds: Vec<(usize, usize)>) -> TrajResult<Vec<(u32, u32)>> {
    let mut out = Vec::with_capacity(bonds.len());
    for (a, b) in bonds {
        if a >= n_atoms || b >= n_atoms {
            return Err(TrajError::Mismatch(
                "bond index outside system atom count".into(),
            ));
        }
        if a == b {
            continue;
        }
        let (i, j) = if a < b { (a, b) } else { (b, a) };
        out.push((i as u32, j as u32));
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn molecule_ids_from_bonds(n_atoms: usize, bonds: &[(u32, u32)]) -> Vec<i32> {
    if n_atoms == 0 || bonds.is_empty() {
        return Vec::new();
    }
    let mut parent: Vec<usize> = (0..n_atoms).collect();
    for &(a, b) in bonds {
        union(&mut parent, a as usize, b as usize);
    }
    let mut root_to_id = HashMap::<usize, i32>::new();
    let mut next = 1i32;
    let mut ids = Vec::with_capacity(n_atoms);
    for atom in 0..n_atoms {
        let root = find(&mut parent, atom);
        let id = *root_to_id.entry(root).or_insert_with(|| {
            let id = next;
            next += 1;
            id
        });
        ids.push(id);
    }
    ids
}

fn find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]);
    }
    parent[x]
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

impl SelectionContext for System {
    fn n_atoms(&self) -> usize {
        self.n_atoms()
    }

    fn atom_name_ids(&self) -> &[u32] {
        &self.atoms.name_id
    }

    fn residue_name_ids(&self) -> &[u32] {
        &self.atoms.resname_id
    }

    fn residue_numbers(&self) -> &[i32] {
        &self.atoms.resid
    }

    fn chain_ids(&self) -> &[u32] {
        &self.atoms.chain_id
    }

    fn element_ids(&self) -> &[u32] {
        &self.atoms.element_id
    }

    fn intern_upper(&mut self, value: &str) -> u32 {
        self.interner.intern_upper(value)
    }

    fn resolve(&self, id: u32) -> Option<&str> {
        self.interner.resolve(id)
    }
}
