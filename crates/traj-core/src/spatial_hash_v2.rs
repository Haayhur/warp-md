//! Optimized spatial hash based on Packmol's linked-cell algorithm
//!
//! Key optimizations from Packmol:
//! 1. Non-empty cell tracking - only iterate over cells that contain atoms
//! 2. Incremental updates - update only affected cells during moves
//! 3. 13-cell forward checking - avoid double-counting pairs
//! 4. Array-based structure - better cache locality than HashMap
//! 5. Hybrid storage - falls back to sparse HashMap for large grids

use crate::geom::Vec3;
use fxhash::FxHashMap;

/// Maximum number of cells before falling back to sparse storage
///
/// This threshold prevents excessive memory usage for large sparse domains.
/// At 3usize per cell (cell_first, cell_next, is_empty), 16M cells ≈ 48MB
/// which is a reasonable upper limit for dense allocation.
const MAX_DENSE_CELLS: usize = 16_000_000;

/// Optimized spatial hash with non-empty cell tracking
///
/// Based on Packmol's linked-cell algorithm from:
/// - cell_indexing.f90
/// - compute_data.f90
/// - computef.f90
///
/// Uses hybrid storage: dense array allocation for small grids,
/// sparse HashMap for large grids.
pub struct SpatialHashV2 {
    /// Cell size for spatial partitioning
    cell: f32,

    /// Storage mode - dense (array) or sparse (HashMap)
    storage: Storage,
}

/// Internal storage representation
enum Storage {
    /// Dense array-based storage (Packmol-style)
    /// Used for small grids where memory overhead is acceptable
    Dense {
        /// 3D grid of linked lists: maps cell (ix,iy,iz) to first atom index
        cell_first: Vec<usize>,
        /// Linked list of atoms within each cell
        atom_next: Vec<usize>,
        grid_dims: (i32, i32, i32),
        grid_offset: (i32, i32, i32),

        /// Linked list of non-empty cells
        non_empty_first: usize,
        cell_next: Vec<usize>,

        /// Track which cells are empty for O(1) checks
        is_empty: Vec<bool>,

        /// Total number of atoms
        n_atoms: usize,
    },

    /// Sparse HashMap-based storage
    /// Used for large grids to prevent excessive memory usage
    Sparse {
        /// Maps cell (ix,iy,iz) to first atom index
        cell_map: FxHashMap<(i32, i32, i32), usize>,
        /// Linked list of atoms within each cell
        atom_next: Vec<usize>,
        /// Total number of atoms
        n_atoms: usize,
    },
}

const END_OF_LIST: usize = usize::MAX;

impl SpatialHashV2 {
    /// Create a new spatial hash with the given cell size
    ///
    /// # Arguments
    /// * `cell` - Cell size (must be > 0)
    /// * `box_min` - Minimum box corner (for grid sizing)
    /// * `box_max` - Maximum box corner (for grid sizing)
    pub fn new(cell: f32, box_min: Vec3, box_max: Vec3) -> Self {
        let cell = cell.max(1.0e-6);

        // Calculate grid dimensions
        let grid_size = box_max.sub(box_min);
        let cells_for_axis = |span: f32| -> i32 {
            // Degenerate/inverted/non-finite extents still get one valid cell.
            if span.is_finite() && span > 0.0 {
                ((span / cell).ceil() as i32).max(1)
            } else {
                1
            }
        };
        let grid_dims = (
            cells_for_axis(grid_size.x),
            cells_for_axis(grid_size.y),
            cells_for_axis(grid_size.z),
        );

        // Calculate offset to handle negative coordinates
        let offset_for_axis = |min: f32| -> i32 {
            if min.is_finite() {
                (min / cell).floor() as i32
            } else {
                0
            }
        };
        let grid_offset = (
            offset_for_axis(box_min.x),
            offset_for_axis(box_min.y),
            offset_for_axis(box_min.z),
        );

        // Choose storage mode based on grid size
        let n_cells = (grid_dims.0 as usize)
            .checked_mul(grid_dims.1 as usize)
            .and_then(|n| n.checked_mul(grid_dims.2 as usize));

        match n_cells {
            Some(n) if n <= MAX_DENSE_CELLS => {
                // Use dense storage for small grids
                Self {
                    cell,
                    storage: Storage::Dense {
                        cell_first: vec![END_OF_LIST; n],
                        atom_next: Vec::new(),
                        grid_dims,
                        grid_offset,
                        non_empty_first: END_OF_LIST,
                        cell_next: vec![END_OF_LIST; n],
                        is_empty: vec![true; n],
                        n_atoms: 0,
                    },
                }
            }
            _ => {
                // Fall back to sparse storage for large grids
                Self {
                    cell,
                    storage: Storage::Sparse {
                        cell_map: FxHashMap::default(),
                        atom_next: Vec::new(),
                        n_atoms: 0,
                    },
                }
            }
        }
    }

    /// Convert position to cell coordinates
    #[inline]
    fn cell_coords_with_cell(cell: f32, p: Vec3) -> (i32, i32, i32) {
        (
            (p.x / cell).floor() as i32,
            (p.y / cell).floor() as i32,
            (p.z / cell).floor() as i32,
        )
    }

    /// Convert position to cell coordinates
    #[inline]
    fn cell_coords(&self, p: Vec3) -> (i32, i32, i32) {
        Self::cell_coords_with_cell(self.cell, p)
    }

    /// Convert position to flat cell index (dense mode only)
    #[inline]
    fn flat_cell_index_with_cell(
        cell: f32,
        p: Vec3,
        grid_offset: (i32, i32, i32),
        grid_dims: (i32, i32, i32),
    ) -> usize {
        let ix = ((p.x / cell).floor() as i32)
            .saturating_sub(grid_offset.0)
            .clamp(0, grid_dims.0 - 1);
        let iy = ((p.y / cell).floor() as i32)
            .saturating_sub(grid_offset.1)
            .clamp(0, grid_dims.1 - 1);
        let iz = ((p.z / cell).floor() as i32)
            .saturating_sub(grid_offset.2)
            .clamp(0, grid_dims.2 - 1);

        ((ix * grid_dims.1 + iy) * grid_dims.2 + iz) as usize
    }

    /// Convert position to flat cell index (dense mode only)
    #[inline]
    fn flat_cell_index(
        &self,
        p: Vec3,
        grid_offset: (i32, i32, i32),
        grid_dims: (i32, i32, i32),
    ) -> usize {
        Self::flat_cell_index_with_cell(self.cell, p, grid_offset, grid_dims)
    }

    /// Convert flat cell index back to 3D coordinates (dense mode only)
    #[inline]
    fn flat_to_3d(&self, idx: usize, grid_dims: (i32, i32, i32)) -> (i32, i32, i32) {
        let idx = idx as i32;
        let iz = idx % grid_dims.2;
        let iy = (idx / grid_dims.2) % grid_dims.1;
        let ix = idx / (grid_dims.1 * grid_dims.2);
        (ix, iy, iz)
    }

    /// Insert a single atom at the given position
    ///
    /// This is an incremental update - only the affected cell is modified
    pub fn insert(&mut self, atom_idx: usize, pos: Vec3) {
        let cell = self.cell;
        match &mut self.storage {
            Storage::Dense {
                cell_first,
                grid_dims,
                grid_offset,
                atom_next,
                non_empty_first,
                cell_next,
                is_empty,
                n_atoms,
            } => {
                // Ensure capacity for atom_next
                if atom_idx >= atom_next.len() {
                    atom_next.resize(atom_idx + 1, END_OF_LIST);
                }

                let cell_idx = Self::flat_cell_index_with_cell(cell, pos, *grid_offset, *grid_dims);

                // Add atom to cell's linked list
                let old_first = cell_first[cell_idx];
                atom_next[atom_idx] = old_first;
                cell_first[cell_idx] = atom_idx;

                // Update non-empty cell tracking (Packmol optimization)
                if is_empty[cell_idx] {
                    is_empty[cell_idx] = false;
                    cell_next[cell_idx] = *non_empty_first;
                    *non_empty_first = cell_idx;
                }

                *n_atoms = (*n_atoms).max(atom_idx + 1);
            }
            Storage::Sparse {
                cell_map,
                atom_next,
                n_atoms,
            } => {
                // Ensure capacity for atom_next
                if atom_idx >= atom_next.len() {
                    atom_next.resize(atom_idx + 1, END_OF_LIST);
                }

                let cell_key = Self::cell_coords_with_cell(cell, pos);

                // Add atom to cell's linked list
                let old_first = cell_map.entry(cell_key).or_insert(END_OF_LIST);
                atom_next[atom_idx] = *old_first;
                *old_first = atom_idx;

                *n_atoms = (*n_atoms).max(atom_idx + 1);
            }
        }
    }

    /// Remove an atom from its current cell
    ///
    /// This is an incremental update - only the affected cell is modified
    pub fn remove(&mut self, atom_idx: usize, pos: Vec3) {
        let cell = self.cell;
        match &mut self.storage {
            Storage::Dense {
                cell_first,
                grid_dims,
                grid_offset,
                atom_next,
                cell_next,
                non_empty_first,
                is_empty,
                ..
            } => {
                let cell_idx = Self::flat_cell_index_with_cell(cell, pos, *grid_offset, *grid_dims);

                let mut prev_idx = END_OF_LIST;
                let mut current = cell_first[cell_idx];

                while current != END_OF_LIST {
                    if current == atom_idx {
                        // Found the atom to remove
                        if prev_idx == END_OF_LIST {
                            // Atom was at head of list
                            cell_first[cell_idx] = atom_next[atom_idx];
                        } else {
                            // Atom was in middle of list
                            atom_next[prev_idx] = atom_next[atom_idx];
                        }

                        // Check if cell is now empty
                        if cell_first[cell_idx] == END_OF_LIST {
                            Self::mark_cell_empty_dense(
                                cell_idx,
                                cell_next,
                                non_empty_first,
                                is_empty,
                            );
                        }

                        return;
                    }
                    prev_idx = current;
                    current = atom_next[current];
                }
            }
            Storage::Sparse {
                cell_map,
                atom_next,
                ..
            } => {
                let cell_key = Self::cell_coords_with_cell(cell, pos);

                if let Some(head) = cell_map.get_mut(&cell_key) {
                    let mut prev = END_OF_LIST;
                    let mut current = *head;
                    while current != END_OF_LIST {
                        if current == atom_idx {
                            if prev == END_OF_LIST {
                                *head = atom_next[atom_idx];
                            } else {
                                atom_next[prev] = atom_next[atom_idx];
                            }
                            // Remove empty cell from map
                            if *head == END_OF_LIST {
                                cell_map.remove(&cell_key);
                            }
                            return;
                        }
                        prev = current;
                        current = atom_next[current];
                    }
                }
            }
        }
    }

    /// Update an atom's position (remove from old cell, insert into new)
    ///
    /// This is more efficient than rebuilding when only a few atoms move
    pub fn update(&mut self, atom_idx: usize, old_pos: Vec3, new_pos: Vec3) {
        let old_coords = self.cell_coords(old_pos);
        let new_coords = self.cell_coords(new_pos);

        if old_coords == new_coords {
            // Same cell - no change needed
            return;
        }

        // Remove from old cell
        self.remove(atom_idx, old_pos);
        // Insert into new cell
        self.insert(atom_idx, new_pos);
    }

    /// Mark a cell as empty and remove from non-empty list (dense mode only)
    fn mark_cell_empty_dense(
        cell_idx: usize,
        cell_next: &mut [usize],
        non_empty_first: &mut usize,
        is_empty: &mut [bool],
    ) {
        is_empty[cell_idx] = true;

        // Remove from non-empty cell linked list
        let mut prev = END_OF_LIST;
        let mut current = *non_empty_first;

        while current != END_OF_LIST {
            if current == cell_idx {
                if prev == END_OF_LIST {
                    *non_empty_first = cell_next[cell_idx];
                } else {
                    cell_next[prev] = cell_next[cell_idx];
                }
                cell_next[cell_idx] = END_OF_LIST;
                return;
            }
            prev = current;
            current = cell_next[current];
        }
    }

    /// Iterate over atoms in neighboring cells (27-cell stencil)
    pub fn for_each_neighbor<F>(&self, pos: Vec3, mut f: F)
    where
        F: FnMut(usize),
    {
        match &self.storage {
            Storage::Dense {
                cell_first,
                grid_dims,
                grid_offset,
                atom_next,
                is_empty,
                ..
            } => {
                let (center_ix, center_iy, center_iz) = self.flat_to_3d(
                    self.flat_cell_index(pos, *grid_offset, *grid_dims),
                    *grid_dims,
                );

                for dx in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dz in -1i32..=1 {
                            let ix = center_ix + dx;
                            let iy = center_iy + dy;
                            let iz = center_iz + dz;

                            // Check bounds
                            if ix < 0
                                || ix >= grid_dims.0
                                || iy < 0
                                || iy >= grid_dims.1
                                || iz < 0
                                || iz >= grid_dims.2
                            {
                                continue;
                            }

                            let cell_idx = ((ix * grid_dims.1 + iy) * grid_dims.2 + iz) as usize;

                            // Skip empty cells (Packmol optimization)
                            if is_empty[cell_idx] {
                                continue;
                            }

                            let mut atom_idx = cell_first[cell_idx];
                            while atom_idx != END_OF_LIST {
                                f(atom_idx);
                                atom_idx = atom_next[atom_idx];
                            }
                        }
                    }
                }
            }
            Storage::Sparse {
                cell_map,
                atom_next,
                ..
            } => {
                let (ix, iy, iz) = self.cell_coords(pos);
                for dx in -1i32..=1 {
                    for dy in -1i32..=1 {
                        for dz in -1i32..=1 {
                            let key = (ix + dx, iy + dy, iz + dz);
                            if let Some(&head) = cell_map.get(&key) {
                                let mut idx = head;
                                while idx != END_OF_LIST {
                                    f(idx);
                                    idx = atom_next[idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Iterate over atoms in forward 13 neighboring cells only
    ///
    /// This is Packmol's optimization to avoid double-counting pairs.
    /// Each pair (i,j) is checked exactly once when i < j.
    ///
    /// Forward neighbors are:
    /// - Current cell (0,0,0)
    /// - 3 face neighbors: (+1,0,0), (0,+1,0), (0,0,+1)
    /// - 6 edge neighbors: (+1,-1,0), (+1,0,-1), (0,+1,-1), (0,+1,+1), (+1,+1,0), (+1,0,+1)
    /// - 4 corner neighbors: (+1,-1,-1), (+1,-1,+1), (+1,+1,-1), (+1,+1,+1)
    pub fn for_each_forward_neighbor<F>(&self, pos: Vec3, mut f: F)
    where
        F: FnMut(usize),
    {
        // Forward neighbor offsets (14 cells total)
        const FORWARD_OFFSETS: [(i32, i32, i32); 14] = [
            (0, 0, 0),   // Current cell
            (1, 0, 0),   // +x face
            (0, 1, 0),   // +y face
            (0, 0, 1),   // +z face
            (1, -1, 0),  // +x-y edge
            (1, 0, -1),  // +x-z edge
            (0, 1, -1),  // +y-z edge
            (0, 1, 1),   // +y+z edge
            (1, 1, 0),   // +x+y edge
            (1, 0, 1),   // +x+z edge
            (1, -1, -1), // +x-y-z corner
            (1, -1, 1),  // +x-y+z corner
            (1, 1, -1),  // +x+y-z corner
            (1, 1, 1),   // +x+y+z corner
        ];

        match &self.storage {
            Storage::Dense {
                cell_first,
                grid_dims,
                grid_offset,
                atom_next,
                is_empty,
                ..
            } => {
                let (center_ix, center_iy, center_iz) = self.flat_to_3d(
                    self.flat_cell_index(pos, *grid_offset, *grid_dims),
                    *grid_dims,
                );

                for (dx, dy, dz) in FORWARD_OFFSETS {
                    let ix = center_ix + dx;
                    let iy = center_iy + dy;
                    let iz = center_iz + dz;

                    // Check bounds
                    if ix < 0
                        || ix >= grid_dims.0
                        || iy < 0
                        || iy >= grid_dims.1
                        || iz < 0
                        || iz >= grid_dims.2
                    {
                        continue;
                    }

                    let cell_idx = ((ix * grid_dims.1 + iy) * grid_dims.2 + iz) as usize;

                    // Skip empty cells
                    if is_empty[cell_idx] {
                        continue;
                    }

                    let mut atom_idx = cell_first[cell_idx];
                    while atom_idx != END_OF_LIST {
                        f(atom_idx);
                        atom_idx = atom_next[atom_idx];
                    }
                }
            }
            Storage::Sparse {
                cell_map,
                atom_next,
                ..
            } => {
                let (ix, iy, iz) = self.cell_coords(pos);
                for (dx, dy, dz) in FORWARD_OFFSETS {
                    let key = (ix + dx, iy + dy, iz + dz);
                    if let Some(&head) = cell_map.get(&key) {
                        let mut idx = head;
                        while idx != END_OF_LIST {
                            f(idx);
                            idx = atom_next[idx];
                        }
                    }
                }
            }
        }
    }

    /// Check for overlaps using 27-cell stencil
    pub fn overlaps_with<F, G>(
        &self,
        positions: &[Vec3],
        existing: &[Vec3],
        mut radius: F,
        mut existing_radius: G,
    ) -> bool
    where
        F: FnMut(usize) -> f32,
        G: FnMut(usize) -> f32,
    {
        match &self.storage {
            Storage::Dense {
                cell_first,
                grid_dims,
                grid_offset,
                atom_next,
                is_empty,
                ..
            } => {
                for (i, p) in positions.iter().enumerate() {
                    let r = radius(i);
                    let (center_ix, center_iy, center_iz) = self.flat_to_3d(
                        self.flat_cell_index(*p, *grid_offset, *grid_dims),
                        *grid_dims,
                    );

                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                let ix = center_ix + dx;
                                let iy = center_iy + dy;
                                let iz = center_iz + dz;

                                // Check bounds
                                if ix < 0
                                    || ix >= grid_dims.0
                                    || iy < 0
                                    || iy >= grid_dims.1
                                    || iz < 0
                                    || iz >= grid_dims.2
                                {
                                    continue;
                                }

                                let cell_idx =
                                    ((ix * grid_dims.1 + iy) * grid_dims.2 + iz) as usize;

                                // Skip empty cells (Packmol optimization)
                                if is_empty[cell_idx] {
                                    continue;
                                }

                                let mut atom_idx = cell_first[cell_idx];
                                while atom_idx != END_OF_LIST {
                                    let q = existing[atom_idx];
                                    let d = p.sub(q);
                                    let dist2 = d.dot(d);
                                    let r_sum = r + existing_radius(atom_idx);
                                    if dist2 < r_sum * r_sum {
                                        return true;
                                    }
                                    atom_idx = atom_next[atom_idx];
                                }
                            }
                        }
                    }
                }
                false
            }
            Storage::Sparse {
                cell_map,
                atom_next,
                ..
            } => {
                for (i, p) in positions.iter().enumerate() {
                    let r = radius(i);
                    let (ix, iy, iz) = self.cell_coords(*p);
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                let key = (ix + dx, iy + dy, iz + dz);
                                if let Some(&head) = cell_map.get(&key) {
                                    let mut idx = head;
                                    while idx != END_OF_LIST {
                                        let q = existing[idx];
                                        let d = p.sub(q);
                                        let dist2 = d.dot(d);
                                        let r_sum = r + existing_radius(idx);
                                        if dist2 < r_sum * r_sum {
                                            return true;
                                        }
                                        idx = atom_next[idx];
                                    }
                                }
                            }
                        }
                    }
                }
                false
            }
        }
    }

    /// Check for overlaps with short tolerance penalty
    pub fn overlaps_short_tol(
        &self,
        positions: &[Vec3],
        existing: &[Vec3],
        min_dist: f32,
        short_tol_dist: f32,
        short_tol_scale: f32,
    ) -> Option<f32> {
        if min_dist <= 0.0 || short_tol_dist <= 0.0 {
            return None;
        }

        let min2 = min_dist * min_dist;
        let short2 = short_tol_dist * short_tol_dist;
        let mut penalty = 0.0f32;

        match &self.storage {
            Storage::Dense {
                cell_first,
                grid_dims,
                grid_offset,
                atom_next,
                is_empty,
                ..
            } => {
                for p in positions {
                    let (center_ix, center_iy, center_iz) = self.flat_to_3d(
                        self.flat_cell_index(*p, *grid_offset, *grid_dims),
                        *grid_dims,
                    );

                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                let ix = center_ix + dx;
                                let iy = center_iy + dy;
                                let iz = center_iz + dz;

                                // Check bounds
                                if ix < 0
                                    || ix >= grid_dims.0
                                    || iy < 0
                                    || iy >= grid_dims.1
                                    || iz < 0
                                    || iz >= grid_dims.2
                                {
                                    continue;
                                }

                                let cell_idx =
                                    ((ix * grid_dims.1 + iy) * grid_dims.2 + iz) as usize;

                                // Skip empty cells (Packmol optimization)
                                if is_empty[cell_idx] {
                                    continue;
                                }

                                let mut atom_idx = cell_first[cell_idx];
                                while atom_idx != END_OF_LIST {
                                    let q = existing[atom_idx];
                                    let d = p.sub(q);
                                    let dist2 = d.dot(d);

                                    if dist2 < min2 {
                                        return Some(f32::INFINITY);
                                    }

                                    if dist2 < short2 {
                                        let diff = dist2 - short2;
                                        penalty += diff * diff * short_tol_scale;
                                    }

                                    atom_idx = atom_next[atom_idx];
                                }
                            }
                        }
                    }
                }
                Some(penalty)
            }
            Storage::Sparse {
                cell_map,
                atom_next,
                ..
            } => {
                for p in positions {
                    let (ix, iy, iz) = self.cell_coords(*p);
                    for dx in -1i32..=1 {
                        for dy in -1i32..=1 {
                            for dz in -1i32..=1 {
                                let key = (ix + dx, iy + dy, iz + dz);
                                if let Some(&head) = cell_map.get(&key) {
                                    let mut idx = head;
                                    while idx != END_OF_LIST {
                                        let q = existing[idx];
                                        let d = p.sub(q);
                                        let dist2 = d.dot(d);

                                        if dist2 < min2 {
                                            return Some(f32::INFINITY);
                                        }

                                        if dist2 < short2 {
                                            let diff = dist2 - short2;
                                            penalty += diff * diff * short_tol_scale;
                                        }

                                        idx = atom_next[idx];
                                    }
                                }
                            }
                        }
                    }
                }
                Some(penalty)
            }
        }
    }

    /// Clear all data (equivalent to creating a new instance)
    pub fn clear(&mut self) {
        match &mut self.storage {
            Storage::Dense {
                cell_first,
                atom_next,
                non_empty_first,
                cell_next,
                is_empty,
                n_atoms,
                ..
            } => {
                cell_first.fill(END_OF_LIST);
                atom_next.clear();
                *non_empty_first = END_OF_LIST;
                cell_next.fill(END_OF_LIST);
                is_empty.fill(true);
                *n_atoms = 0;
            }
            Storage::Sparse {
                cell_map,
                atom_next,
                n_atoms,
            } => {
                cell_map.clear();
                atom_next.clear();
                *n_atoms = 0;
            }
        }
    }

    /// Get statistics about the spatial hash
    pub fn stats(&self) -> SpatialHashStats {
        match &self.storage {
            Storage::Dense {
                cell_first,
                atom_next,
                grid_dims,
                non_empty_first,
                cell_next,
                n_atoms,
                ..
            } => {
                let mut n_non_empty = 0;
                let mut atoms_in_non_empty = 0;
                let mut max_atoms_per_cell = 0;

                let mut cell_idx = *non_empty_first;
                while cell_idx != END_OF_LIST {
                    n_non_empty += 1;
                    let mut count = 0;
                    let mut atom_idx = cell_first[cell_idx];
                    while atom_idx != END_OF_LIST {
                        count += 1;
                        atom_idx = atom_next[atom_idx];
                    }
                    atoms_in_non_empty += count;
                    max_atoms_per_cell = max_atoms_per_cell.max(count);
                    cell_idx = cell_next[cell_idx];
                }

                SpatialHashStats {
                    total_cells: (grid_dims.0 as usize)
                        .checked_mul(grid_dims.1 as usize)
                        .and_then(|n| n.checked_mul(grid_dims.2 as usize))
                        .expect("SpatialHashV2 grid has too many cells"),
                    non_empty_cells: n_non_empty,
                    total_atoms: *n_atoms,
                    avg_atoms_per_non_empty_cell: if n_non_empty > 0 {
                        atoms_in_non_empty as f32 / n_non_empty as f32
                    } else {
                        0.0
                    },
                    max_atoms_per_cell,
                }
            }
            Storage::Sparse {
                cell_map,
                atom_next,
                n_atoms,
            } => {
                let mut n_non_empty = 0;
                let mut atoms_in_non_empty = 0;
                let mut max_atoms_per_cell = 0;

                for (&_key, &head) in cell_map.iter() {
                    n_non_empty += 1;
                    let mut count = 0;
                    let mut idx = head;
                    while idx != END_OF_LIST {
                        count += 1;
                        idx = atom_next[idx];
                    }
                    atoms_in_non_empty += count;
                    max_atoms_per_cell = max_atoms_per_cell.max(count);
                }

                SpatialHashStats {
                    total_cells: cell_map.len(),
                    non_empty_cells: n_non_empty,
                    total_atoms: *n_atoms,
                    avg_atoms_per_non_empty_cell: if n_non_empty > 0 {
                        atoms_in_non_empty as f32 / n_non_empty as f32
                    } else {
                        0.0
                    },
                    max_atoms_per_cell,
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpatialHashStats {
    pub total_cells: usize,
    pub non_empty_cells: usize,
    pub total_atoms: usize,
    pub avg_atoms_per_non_empty_cell: f32,
    pub max_atoms_per_cell: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_insertion() {
        let mut hash =
            SpatialHashV2::new(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 10.0));

        hash.insert(0, Vec3::new(1.5, 1.5, 1.5));
        hash.insert(1, Vec3::new(1.6, 1.6, 1.6));

        let stats = hash.stats();
        assert_eq!(stats.total_atoms, 2);
        assert_eq!(stats.non_empty_cells, 1);
    }

    #[test]
    fn test_degenerate_extents_clamp_to_valid_dims() {
        let hash = SpatialHashV2::new(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 2.0, -3.0));

        // With the new implementation, we can't directly check grid_dims
        // but we can check that stats returns the correct cell count
        assert_eq!(hash.stats().total_cells, 2);
    }

    #[test]
    fn test_insert_and_query_with_zero_extent_box() {
        let mut hash = SpatialHashV2::new(1.0, Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0));
        hash.insert(0, Vec3::new(1.0, 1.0, 1.0));

        let mut seen = Vec::new();
        hash.for_each_neighbor(Vec3::new(1.0, 1.0, 1.0), |idx| seen.push(idx));

        assert_eq!(seen, vec![0]);
    }

    #[test]
    fn test_non_finite_box_min_is_sanitized() {
        let mut hash = SpatialHashV2::new(
            1.0,
            Vec3::new(f32::NAN, f32::INFINITY, f32::NEG_INFINITY),
            Vec3::new(5.0, 5.0, 5.0),
        );
        hash.insert(0, Vec3::new(1.0, 1.0, 1.0));

        let mut seen = Vec::new();
        hash.for_each_neighbor(Vec3::new(1.0, 1.0, 1.0), |idx| seen.push(idx));

        assert_eq!(seen, vec![0]);
    }

    #[test]
    fn test_huge_grid_uses_sparse_storage() {
        // This grid would exceed MAX_DENSE_CELLS, so it should fall back to sparse storage
        // without panicking
        let hash = SpatialHashV2::new(
            1.0e-6,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(f32::MAX, f32::MAX, f32::MAX),
        );

        // The hash should use sparse storage now
        let stats = hash.stats();
        assert_eq!(stats.total_cells, 0); // Empty grid initially
        assert_eq!(stats.total_atoms, 0);
    }

    #[test]
    fn test_sparse_storage_fallback() {
        // Create a grid that exceeds MAX_DENSE_CELLS
        // MAX_DENSE_CELLS = 16_000_000, so a 300x300x300 grid = 27M cells should trigger sparse mode
        let mut hash = SpatialHashV2::new(
            1.0,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(300.0, 300.0, 300.0),
        );

        // Insert some atoms
        hash.insert(0, Vec3::new(1.5, 1.5, 1.5));
        hash.insert(1, Vec3::new(100.0, 100.0, 100.0));

        let mut seen = Vec::new();
        hash.for_each_neighbor(Vec3::new(1.5, 1.5, 1.5), |idx| seen.push(idx));

        // Should find atom 0 in the same cell
        assert!(seen.contains(&0));

        let stats = hash.stats();
        assert_eq!(stats.total_atoms, 2);
    }

    #[test]
    fn test_clear_works_for_both_modes() {
        // Dense mode
        let mut dense_hash =
            SpatialHashV2::new(1.0, Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 10.0, 10.0));
        dense_hash.insert(0, Vec3::new(1.0, 1.0, 1.0));
        dense_hash.clear();
        assert_eq!(dense_hash.stats().total_atoms, 0);

        // Sparse mode
        let mut sparse_hash = SpatialHashV2::new(
            1.0,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(300.0, 300.0, 300.0),
        );
        sparse_hash.insert(0, Vec3::new(1.0, 1.0, 1.0));
        sparse_hash.clear();
        assert_eq!(sparse_hash.stats().total_atoms, 0);
    }
}
