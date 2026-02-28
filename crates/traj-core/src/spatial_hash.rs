use fxhash::FxHashMap;

use crate::geom::Vec3;

pub struct SpatialHash {
    cell: f32,
    map: FxHashMap<(i32, i32, i32), usize>,
    next: Vec<usize>,
}

impl SpatialHash {
    pub fn new(cell: f32) -> Self {
        Self::with_capacity(cell, 0)
    }

    pub fn with_capacity(cell: f32, expected_cells: usize) -> Self {
        Self {
            cell: cell.max(1.0e-6),
            map: FxHashMap::with_capacity_and_hasher(expected_cells.max(1), Default::default()),
            next: Vec::new(),
        }
    }

    fn cell_index(&self, p: Vec3) -> (i32, i32, i32) {
        (
            (p.x / self.cell).floor() as i32,
            (p.y / self.cell).floor() as i32,
            (p.z / self.cell).floor() as i32,
        )
    }

    pub fn insert(&mut self, idx: usize, pos: Vec3) {
        let key = self.cell_index(pos);
        if idx >= self.next.len() {
            self.next.resize(idx + 1, usize::MAX);
        }
        let head = self.map.entry(key).or_insert(usize::MAX);
        self.next[idx] = *head;
        *head = idx;
    }

    /// Remove an atom from its current cell.
    ///
    /// This is an incremental update - only the affected cell is modified.
    pub fn remove(&mut self, idx: usize, pos: Vec3) {
        let key = self.cell_index(pos);
        if let Some(head) = self.map.get_mut(&key) {
            let mut prev = usize::MAX;
            let mut current = *head;
            while current != usize::MAX {
                if current == idx {
                    if prev == usize::MAX {
                        *head = self.next[idx];
                    } else {
                        self.next[prev] = self.next[idx];
                    }
                    // Remove empty cell from map
                    if *head == usize::MAX {
                        self.map.remove(&key);
                    }
                    return;
                }
                prev = current;
                current = self.next[current];
            }
        }
    }

    /// Update an atom's position (remove from old cell, insert into new).
    ///
    /// This is more efficient than rebuilding when only a few atoms move.
    pub fn update(&mut self, idx: usize, old_pos: Vec3, new_pos: Vec3) {
        let old_key = self.cell_index(old_pos);
        let new_key = self.cell_index(new_pos);

        if old_key == new_key {
            // Same cell - no change needed
            return;
        }

        // Remove from old cell
        self.remove(idx, old_pos);
        // Insert into new cell
        self.insert(idx, new_pos);
    }

    pub fn for_each_neighbor<F>(&self, pos: Vec3, mut f: F)
    where
        F: FnMut(usize),
    {
        let (ix, iy, iz) = self.cell_index(pos);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let key = (ix + dx, iy + dy, iz + dz);
                    if let Some(&head) = self.map.get(&key) {
                        let mut idx = head;
                        while idx != usize::MAX {
                            f(idx);
                            idx = self.next[idx];
                        }
                    }
                }
            }
        }
    }

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
        for (i, p) in positions.iter().enumerate() {
            let (ix, iy, iz) = self.cell_index(*p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let key = (ix + dx, iy + dy, iz + dz);
                        if let Some(&head) = self.map.get(&key) {
                            let mut idx = head;
                            while idx != usize::MAX {
                                let q = existing[idx];
                                let d = p.sub(q);
                                let dist2 = d.dot(d);
                                let r = radius(i) + existing_radius(idx);
                                if dist2 < r * r {
                                    return true;
                                }
                                idx = self.next[idx];
                            }
                        }
                    }
                }
            }
        }
        false
    }

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
        for p in positions {
            let (ix, iy, iz) = self.cell_index(*p);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let key = (ix + dx, iy + dy, iz + dz);
                        if let Some(&head) = self.map.get(&key) {
                            let mut idx = head;
                            while idx != usize::MAX {
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
                                idx = self.next[idx];
                            }
                        }
                    }
                }
            }
        }
        Some(penalty)
    }
}
