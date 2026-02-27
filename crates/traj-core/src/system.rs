use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{TrajError, TrajResult};
use crate::interner::StringInterner;
use crate::selection::{compile_selection, Selection};

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
            selection_cache: HashMap::new(),
        }
    }

    pub fn n_atoms(&self) -> usize {
        self.atoms.len()
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
