use crate::backmap_geometry::*;
#[cfg(test)]
use nalgebra::Matrix3;
use nalgebra::Vector3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::fmt;

const EPSILON: f64 = 1.0e-12;
pub const BACKMAP_PLAN_SCHEMA_VERSION: &str = "warp-cg.backmap-plan.v1";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackmapError {
    EmptyTemplate(usize),
    AtomNameCountMismatch(usize),
    SourceAtomIndexCountMismatch(usize),
    DuplicateSourceAtomIndex(usize),
    EmptyMapping(usize),
    MappingCountMismatch(usize),
    EmptyAtomGroup(usize, usize),
    InvalidAtomIndex(usize, usize, usize),
    DuplicateMappedAtom(usize, usize),
    InvalidTargetBeadIndex(usize, usize),
    InvalidMappingWeight(usize, usize),
    InvalidMappingWeights(usize, usize),
    InvalidFudgeFactor,
    NonFiniteReferenceCoordinate(usize, usize),
    NonFiniteCgCoordinate(usize),
    InvalidLinkTemplate(usize, usize),
    InvalidLinkAtom(usize, usize),
    SelfLink(usize),
    AlignmentFailure(usize),
}

impl fmt::Display for BackmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BackmapError {}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackmapArtifact {
    pub schema_version: String,
    pub plan: BackmapPlan,
}

impl BackmapArtifact {
    pub fn new(plan: BackmapPlan) -> Self {
        Self {
            schema_version: BACKMAP_PLAN_SCHEMA_VERSION.to_string(),
            plan,
        }
    }

    pub fn from_json(text: &str) -> Result<Self, String> {
        let artifact: Self =
            serde_json::from_str(text).map_err(|err| format!("invalid backmap artifact: {err}"))?;
        if artifact.schema_version != BACKMAP_PLAN_SCHEMA_VERSION {
            return Err(format!(
                "unsupported backmap schema_version {}; expected {}",
                artifact.schema_version, BACKMAP_PLAN_SCHEMA_VERSION
            ));
        }
        Ok(artifact)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Link {
    pub from_template: usize,
    pub from_atom: usize,
    pub to_template: usize,
    pub to_atom: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_distance: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BackmapDiagnostics {
    pub mapped_bead_rmse: f64,
    pub mapped_bead_max_error: f64,
    pub link_bond_rmse: f64,
    pub link_bond_max_error: f64,
    pub link_count: usize,
    pub internal_bond_max_error: f64,
    pub chirality_inversion_count: usize,
    pub steric_clash_count: usize,
    pub minimum_nonbonded_distance: Option<f64>,
    pub finite: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BackmapFrame {
    pub coordinates: Vec<[f64; 3]>,
    pub diagnostics: BackmapDiagnostics,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BackmapAtomMetadata {
    pub name: String,
    pub element: String,
    pub residue_name: String,
    pub residue_id: i32,
    pub chain: String,
}

/// Forward atom-to-bead mapping retained for inverse reconstruction.
///
/// `atom_indices` and `weights` use the same local ordering as warp-cg
/// trajectory mapping. Omitting `weights` selects center-of-geometry mapping.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BeadSite {
    pub target_bead_index: usize,
    pub atom_indices: Vec<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights: Option<Vec<f64>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BondConstraint {
    pub atom_i: usize,
    pub atom_j: usize,
    pub target_distance: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ChiralConstraint {
    pub center: usize,
    pub neighbor_a: usize,
    pub neighbor_b: usize,
    pub neighbor_c: usize,
    pub reference_sign: i8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ResidueTemplate {
    pub name: String,
    pub atom_names: Vec<String>,
    #[serde(default)]
    pub elements: Vec<String>,
    #[serde(default)]
    pub residue_names: Vec<String>,
    #[serde(default)]
    pub residue_ids: Vec<i32>,
    #[serde(default)]
    pub chains: Vec<String>,
    #[serde(default)]
    pub source_atom_indices: Vec<usize>,
    pub reference_coords: Vec<[f64; 3]>,
    pub bead_sites: Vec<BeadSite>,
    #[serde(default)]
    pub bonds: Vec<BondConstraint>,
    #[serde(default)]
    pub chirality: Vec<ChiralConstraint>,
}

impl ResidueTemplate {
    pub fn from_atom_groups(
        name: impl Into<String>,
        atom_names: Vec<String>,
        reference_coords: Vec<[f64; 3]>,
        atom_groups: &[Vec<usize>],
        target_bead_indices: &[usize],
    ) -> Result<Self, BackmapError> {
        if atom_groups.len() != target_bead_indices.len() {
            return Err(BackmapError::MappingCountMismatch(0));
        }
        Ok(Self {
            name: name.into(),
            atom_names,
            elements: Vec::new(),
            residue_names: Vec::new(),
            residue_ids: Vec::new(),
            chains: Vec::new(),
            source_atom_indices: Vec::new(),
            reference_coords,
            bead_sites: atom_groups
                .iter()
                .zip(target_bead_indices)
                .map(|(atom_indices, &target_bead_index)| BeadSite {
                    target_bead_index,
                    atom_indices: atom_indices.clone(),
                    weights: None,
                })
                .collect(),
            bonds: Vec::new(),
            chirality: Vec::new(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BackmapPlan {
    pub templates: Vec<ResidueTemplate>,
    #[serde(default)]
    pub links: Vec<Link>,
    /// Contracts atom offsets around each mapped bead center without moving
    /// that center. Must be finite and in `(0, 1]`.
    pub fudge_factor: f64,
}

impl BackmapPlan {
    pub fn validate(&self, cg_coords: &[[f64; 3]]) -> Result<(), BackmapError> {
        if !self.fudge_factor.is_finite() || self.fudge_factor <= 0.0 || self.fudge_factor > 1.0 {
            return Err(BackmapError::InvalidFudgeFactor);
        }
        for (bead_idx, coord) in cg_coords.iter().enumerate() {
            if !finite_coord(*coord) {
                return Err(BackmapError::NonFiniteCgCoordinate(bead_idx));
            }
        }
        let uses_source_indices = self
            .templates
            .iter()
            .any(|template| !template.source_atom_indices.is_empty());
        let mut source_indices = BTreeSet::new();
        for (template_idx, template) in self.templates.iter().enumerate() {
            if template.reference_coords.is_empty() {
                return Err(BackmapError::EmptyTemplate(template_idx));
            }
            if template.atom_names.len() != template.reference_coords.len() {
                return Err(BackmapError::AtomNameCountMismatch(template_idx));
            }
            for metadata_len in [
                template.elements.len(),
                template.residue_names.len(),
                template.residue_ids.len(),
                template.chains.len(),
            ] {
                if metadata_len != 0 && metadata_len != template.reference_coords.len() {
                    return Err(BackmapError::AtomNameCountMismatch(template_idx));
                }
            }
            if uses_source_indices
                && template.source_atom_indices.len() != template.reference_coords.len()
            {
                return Err(BackmapError::SourceAtomIndexCountMismatch(template_idx));
            }
            for &source_idx in &template.source_atom_indices {
                if !source_indices.insert(source_idx) {
                    return Err(BackmapError::DuplicateSourceAtomIndex(source_idx));
                }
            }
            if template.bead_sites.is_empty() {
                return Err(BackmapError::EmptyMapping(template_idx));
            }
            for (atom_idx, coord) in template.reference_coords.iter().enumerate() {
                if !finite_coord(*coord) {
                    return Err(BackmapError::NonFiniteReferenceCoordinate(
                        template_idx,
                        atom_idx,
                    ));
                }
            }
            let mut mapped_atoms = BTreeSet::new();
            for (site_idx, site) in template.bead_sites.iter().enumerate() {
                if site.atom_indices.is_empty() {
                    return Err(BackmapError::EmptyAtomGroup(template_idx, site_idx));
                }
                if site.target_bead_index >= cg_coords.len() {
                    return Err(BackmapError::InvalidTargetBeadIndex(
                        template_idx,
                        site.target_bead_index,
                    ));
                }
                if let Some(weights) = &site.weights {
                    if weights.len() != site.atom_indices.len() {
                        return Err(BackmapError::InvalidMappingWeights(template_idx, site_idx));
                    }
                    if weights
                        .iter()
                        .any(|weight| !weight.is_finite() || *weight <= 0.0)
                    {
                        return Err(BackmapError::InvalidMappingWeight(template_idx, site_idx));
                    }
                }
                for &atom_idx in &site.atom_indices {
                    if atom_idx >= template.reference_coords.len() {
                        return Err(BackmapError::InvalidAtomIndex(
                            template_idx,
                            site_idx,
                            atom_idx,
                        ));
                    }
                    // Overlapping forward groups do not have a unique inverse.
                    if !mapped_atoms.insert(atom_idx) {
                        return Err(BackmapError::DuplicateMappedAtom(template_idx, atom_idx));
                    }
                }
            }
            for bond in &template.bonds {
                if bond.atom_i >= template.reference_coords.len()
                    || bond.atom_j >= template.reference_coords.len()
                    || !bond.target_distance.is_finite()
                    || bond.target_distance <= 0.0
                {
                    return Err(BackmapError::InvalidAtomIndex(
                        template_idx,
                        0,
                        bond.atom_i.max(bond.atom_j),
                    ));
                }
            }
            for constraint in &template.chirality {
                if [
                    constraint.center,
                    constraint.neighbor_a,
                    constraint.neighbor_b,
                    constraint.neighbor_c,
                ]
                .into_iter()
                .any(|atom_idx| atom_idx >= template.reference_coords.len())
                    || !matches!(constraint.reference_sign, -1 | 1)
                {
                    return Err(BackmapError::InvalidAtomIndex(
                        template_idx,
                        0,
                        constraint.center,
                    ));
                }
            }
        }
        for (link_idx, link) in self.links.iter().enumerate() {
            if link.from_template >= self.templates.len()
                || link.to_template >= self.templates.len()
            {
                return Err(BackmapError::InvalidLinkTemplate(
                    link.from_template,
                    link.to_template,
                ));
            }
            if link.from_template == link.to_template {
                return Err(BackmapError::SelfLink(link_idx));
            }
            if link.from_atom >= self.templates[link.from_template].reference_coords.len() {
                return Err(BackmapError::InvalidLinkAtom(
                    link.from_template,
                    link.from_atom,
                ));
            }
            if link.to_atom >= self.templates[link.to_template].reference_coords.len() {
                return Err(BackmapError::InvalidLinkAtom(
                    link.to_template,
                    link.to_atom,
                ));
            }
            if link
                .target_distance
                .is_some_and(|distance| !distance.is_finite() || distance <= 0.0)
            {
                return Err(BackmapError::InvalidLinkAtom(
                    link.from_template,
                    link.from_atom,
                ));
            }
        }
        Ok(())
    }

    /// Reconstruct one atomistic frame. Templates may form arbitrary
    /// disconnected, branched, cyclic, or crosslinked graphs.
    pub fn execute(&self, cg_coords: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, BackmapError> {
        Ok(self.execute_frame(cg_coords)?.coordinates)
    }

    pub fn execute_frame(&self, cg_coords: &[[f64; 3]]) -> Result<BackmapFrame, BackmapError> {
        self.validate(cg_coords)?;
        let order = self.placement_order();
        let mut placed: Vec<Option<Vec<[f64; 3]>>> = vec![None; self.templates.len()];

        for template_idx in order {
            let template = &self.templates[template_idx];
            let adjusted = contracted_reference_coords(template, self.fudge_factor);
            let source_centers = template
                .bead_sites
                .iter()
                .map(|site| weighted_center(&adjusted, site))
                .collect::<Vec<_>>();
            let target_centers = template
                .bead_sites
                .iter()
                .map(|site| vec3(cg_coords[site.target_bead_index]))
                .collect::<Vec<_>>();
            let source_anchor = mean(&source_centers);
            let target_anchor = mean(&target_centers);

            let mut sources = source_centers;
            let mut targets = target_centers;
            for link in self.links_for(template_idx) {
                let (local_atom, neighbor_idx, neighbor_atom) =
                    if link.from_template == template_idx {
                        (link.from_atom, link.to_template, link.to_atom)
                    } else {
                        (link.to_atom, link.from_template, link.from_atom)
                    };
                let source = vec3(adjusted[local_atom]);
                let neighbor_target = placed[neighbor_idx]
                    .as_ref()
                    .map(|coords| vec3(coords[neighbor_atom]))
                    .unwrap_or_else(|| self.template_target_anchor(neighbor_idx, cg_coords));
                let direction = neighbor_target - target_anchor;
                let source_radius = (source - source_anchor).norm();
                if direction.norm_squared() > EPSILON && source_radius > EPSILON {
                    let target_distance = self.link_target_distance(link);
                    sources.push(source);
                    targets.push(neighbor_target - direction.normalize() * target_distance);
                }
            }

            let rotation = kabsch_rotation(&sources, &targets, template_idx)?;
            let coords = adjusted
                .iter()
                .map(|&coord| {
                    let point = rotation * (vec3(coord) - source_anchor) + target_anchor;
                    [point.x, point.y, point.z]
                })
                .collect();
            placed[template_idx] = Some(coords);
        }

        if self
            .templates
            .iter()
            .any(|template| !template.source_atom_indices.is_empty())
        {
            let mut indexed = self
                .templates
                .iter()
                .enumerate()
                .flat_map(|(template_idx, template)| {
                    template.source_atom_indices.iter().copied().zip(
                        placed[template_idx]
                            .as_ref()
                            .expect("all templates placed")
                            .iter()
                            .copied(),
                    )
                })
                .collect::<Vec<_>>();
            indexed.sort_by_key(|(source_idx, _)| *source_idx);
            let coordinates = indexed
                .into_iter()
                .map(|(_, coord)| coord)
                .collect::<Vec<_>>();
            Ok(BackmapFrame {
                diagnostics: self.diagnostics(cg_coords, &placed),
                coordinates,
            })
        } else {
            let coordinates = placed
                .iter()
                .flatten()
                .flatten()
                .copied()
                .collect::<Vec<[f64; 3]>>();
            Ok(BackmapFrame {
                diagnostics: self.diagnostics(cg_coords, &placed),
                coordinates,
            })
        }
    }

    fn link_target_distance(&self, link: &Link) -> f64 {
        link.target_distance.unwrap_or_else(|| {
            (vec3(self.templates[link.from_template].reference_coords[link.from_atom])
                - vec3(self.templates[link.to_template].reference_coords[link.to_atom]))
            .norm()
        })
    }

    fn diagnostics(
        &self,
        cg_coords: &[[f64; 3]],
        placed: &[Option<Vec<[f64; 3]>>],
    ) -> BackmapDiagnostics {
        let mut bead_squared = 0.0;
        let mut bead_max: f64 = 0.0;
        let mut bead_count = 0usize;
        for (template_idx, template) in self.templates.iter().enumerate() {
            let coords = placed[template_idx].as_ref().expect("template placed");
            for site in &template.bead_sites {
                let error = (weighted_center(coords, site)
                    - vec3(cg_coords[site.target_bead_index]))
                .norm();
                bead_squared += error * error;
                bead_max = bead_max.max(error);
                bead_count += 1;
            }
        }
        let mut link_squared = 0.0;
        let mut link_max: f64 = 0.0;
        let mut internal_bond_max: f64 = 0.0;
        let mut chirality_inversions = 0usize;
        for (template_idx, template) in self.templates.iter().enumerate() {
            let coords = placed[template_idx].as_ref().expect("template placed");
            for bond in &template.bonds {
                let actual = (vec3(coords[bond.atom_i]) - vec3(coords[bond.atom_j])).norm();
                internal_bond_max = internal_bond_max.max((actual - bond.target_distance).abs());
            }
            for constraint in &template.chirality {
                let sign = signed_volume_sign(
                    coords[constraint.center],
                    coords[constraint.neighbor_a],
                    coords[constraint.neighbor_b],
                    coords[constraint.neighbor_c],
                );
                if sign != 0 && sign != constraint.reference_sign {
                    chirality_inversions += 1;
                }
            }
        }
        for link in &self.links {
            let from = vec3(
                placed[link.from_template]
                    .as_ref()
                    .expect("template placed")[link.from_atom],
            );
            let to =
                vec3(placed[link.to_template].as_ref().expect("template placed")[link.to_atom]);
            let error = ((from - to).norm() - self.link_target_distance(link)).abs();
            link_squared += error * error;
            link_max = link_max.max(error);
        }
        let (steric_clash_count, minimum_nonbonded_distance) = self.steric_diagnostics(placed, 0.6);
        BackmapDiagnostics {
            mapped_bead_rmse: if bead_count == 0 {
                0.0
            } else {
                (bead_squared / bead_count as f64).sqrt()
            },
            mapped_bead_max_error: bead_max,
            link_bond_rmse: if self.links.is_empty() {
                0.0
            } else {
                (link_squared / self.links.len() as f64).sqrt()
            },
            link_bond_max_error: link_max,
            link_count: self.links.len(),
            internal_bond_max_error: internal_bond_max,
            chirality_inversion_count: chirality_inversions,
            steric_clash_count,
            minimum_nonbonded_distance,
            finite: placed
                .into_iter()
                .flatten()
                .flatten()
                .flatten()
                .all(|value| value.is_finite()),
        }
    }

    fn steric_diagnostics(
        &self,
        placed: &[Option<Vec<[f64; 3]>>],
        cutoff: f64,
    ) -> (usize, Option<f64>) {
        let mut bonded = BTreeSet::new();
        for (template_idx, template) in self.templates.iter().enumerate() {
            for bond in &template.bonds {
                bonded.insert(normalized_atom_pair(
                    (template_idx, bond.atom_i),
                    (template_idx, bond.atom_j),
                ));
            }
        }
        for link in &self.links {
            bonded.insert(normalized_atom_pair(
                (link.from_template, link.from_atom),
                (link.to_template, link.to_atom),
            ));
        }
        let mut grid = HashMap::<(i64, i64, i64), Vec<((usize, usize), [f64; 3])>>::new();
        let mut clashes = 0usize;
        let mut minimum: Option<f64> = None;
        for (template_idx, coords) in placed.iter().enumerate() {
            for (atom_idx, &coord) in coords.as_ref().expect("template placed").iter().enumerate() {
                let cell = (
                    (coord[0] / cutoff).floor() as i64,
                    (coord[1] / cutoff).floor() as i64,
                    (coord[2] / cutoff).floor() as i64,
                );
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            if let Some(candidates) =
                                grid.get(&(cell.0 + dx, cell.1 + dy, cell.2 + dz))
                            {
                                for &(other_id, other_coord) in candidates {
                                    if bonded.contains(&normalized_atom_pair(
                                        (template_idx, atom_idx),
                                        other_id,
                                    )) {
                                        continue;
                                    }
                                    let distance = (vec3(coord) - vec3(other_coord)).norm();
                                    minimum = Some(
                                        minimum.map_or(distance, |current| current.min(distance)),
                                    );
                                    if distance < cutoff {
                                        clashes += 1;
                                    }
                                }
                            }
                        }
                    }
                }
                grid.entry(cell)
                    .or_default()
                    .push(((template_idx, atom_idx), coord));
            }
        }
        (clashes, minimum)
    }

    pub fn atom_names_in_output_order(&self) -> Vec<String> {
        self.atom_metadata_in_output_order()
            .into_iter()
            .map(|atom| atom.name)
            .collect()
    }

    pub fn atom_metadata_in_output_order(&self) -> Vec<BackmapAtomMetadata> {
        let metadata = self
            .templates
            .iter()
            .flat_map(|template| {
                (0..template.atom_names.len()).map(|atom_idx| BackmapAtomMetadata {
                    name: template.atom_names[atom_idx].clone(),
                    element: template.elements.get(atom_idx).cloned().unwrap_or_default(),
                    residue_name: template
                        .residue_names
                        .get(atom_idx)
                        .cloned()
                        .unwrap_or_else(|| template.name.clone()),
                    residue_id: template.residue_ids.get(atom_idx).copied().unwrap_or(1),
                    chain: template
                        .chains
                        .get(atom_idx)
                        .cloned()
                        .unwrap_or_else(|| "A".to_string()),
                })
            })
            .collect::<Vec<_>>();
        if self
            .templates
            .iter()
            .any(|template| !template.source_atom_indices.is_empty())
        {
            let mut indexed = self
                .templates
                .iter()
                .scan(0usize, |offset, template| {
                    let start = *offset;
                    *offset += template.atom_names.len();
                    Some(
                        template
                            .source_atom_indices
                            .iter()
                            .copied()
                            .zip(metadata[start..*offset].iter().cloned())
                            .collect::<Vec<_>>(),
                    )
                })
                .flatten()
                .collect::<Vec<_>>();
            indexed.sort_by_key(|(source_idx, _)| *source_idx);
            indexed.into_iter().map(|(_, atom)| atom).collect()
        } else {
            metadata
        }
    }

    fn placement_order(&self) -> Vec<usize> {
        let mut adjacency = vec![BTreeSet::new(); self.templates.len()];
        for link in &self.links {
            adjacency[link.from_template].insert(link.to_template);
            adjacency[link.to_template].insert(link.from_template);
        }
        let mut visited = vec![false; self.templates.len()];
        let mut order = Vec::with_capacity(self.templates.len());
        for root in 0..self.templates.len() {
            if visited[root] {
                continue;
            }
            visited[root] = true;
            let mut queue = VecDeque::from([root]);
            while let Some(node) = queue.pop_front() {
                order.push(node);
                for &neighbor in &adjacency[node] {
                    if !visited[neighbor] {
                        visited[neighbor] = true;
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        order
    }

    fn links_for(&self, template_idx: usize) -> impl Iterator<Item = &Link> {
        self.links.iter().filter(move |link| {
            link.from_template == template_idx || link.to_template == template_idx
        })
    }

    fn template_target_anchor(&self, template_idx: usize, cg_coords: &[[f64; 3]]) -> Vector3<f64> {
        mean(
            &self.templates[template_idx]
                .bead_sites
                .iter()
                .map(|site| vec3(cg_coords[site.target_bead_index]))
                .collect::<Vec<_>>(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map_template(template: &ResidueTemplate, coords: &[[f64; 3]]) -> Vec<[f64; 3]> {
        template
            .bead_sites
            .iter()
            .map(|site| {
                let center = weighted_center(coords, site);
                [center.x, center.y, center.z]
            })
            .collect()
    }

    fn two_bead_template(offset: usize) -> ResidueTemplate {
        ResidueTemplate {
            name: "TEST".into(),
            atom_names: ["A", "B", "C", "D"].map(str::to_string).to_vec(),
            elements: Vec::new(),
            residue_names: Vec::new(),
            residue_ids: Vec::new(),
            chains: Vec::new(),
            source_atom_indices: Vec::new(),
            reference_coords: vec![
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, -1.0, 0.0],
                [3.0, 1.0, 0.0],
            ],
            bead_sites: vec![
                BeadSite {
                    target_bead_index: offset,
                    atom_indices: vec![0, 1],
                    weights: None,
                },
                BeadSite {
                    target_bead_index: offset + 1,
                    atom_indices: vec![2, 3],
                    weights: None,
                },
            ],
            bonds: Vec::new(),
            chirality: Vec::new(),
        }
    }

    #[test]
    fn round_trip_preserves_forward_bead_centers() {
        let template = two_bead_template(0);
        let angle = std::f64::consts::FRAC_PI_2;
        let rotation = Matrix3::new(
            angle.cos(),
            -angle.sin(),
            0.0,
            angle.sin(),
            angle.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );
        let translation = Vector3::new(5.0, -2.0, 3.0);
        let transformed = template
            .reference_coords
            .iter()
            .map(|&coord| {
                let point = rotation * vec3(coord) + translation;
                [point.x, point.y, point.z]
            })
            .collect::<Vec<_>>();
        let cg = map_template(&template, &transformed);
        let plan = BackmapPlan {
            templates: vec![template.clone()],
            links: vec![],
            fudge_factor: 1.0,
        };
        let rebuilt = plan.execute(&cg).unwrap();
        let remapped = map_template(&template, &rebuilt);
        for (actual, expected) in remapped.iter().zip(cg) {
            assert!((vec3(*actual) - vec3(expected)).norm() < 1.0e-10);
        }
    }

    #[test]
    fn local_fudge_preserves_each_bead_center() {
        let template = two_bead_template(0);
        let cg = map_template(&template, &template.reference_coords);
        let plan = BackmapPlan {
            templates: vec![template.clone()],
            links: vec![],
            fudge_factor: 0.4,
        };
        let rebuilt = plan.execute(&cg).unwrap();
        let remapped = map_template(&template, &rebuilt);
        for (actual, expected) in remapped.iter().zip(cg) {
            assert!((vec3(*actual) - vec3(expected)).norm() < 1.0e-10);
        }
    }

    #[test]
    fn cyclic_template_graph_is_supported() {
        let templates = (0..3)
            .map(|idx| ResidueTemplate {
                name: format!("R{idx}"),
                atom_names: vec!["A".into(), "B".into()],
                elements: Vec::new(),
                residue_names: Vec::new(),
                residue_ids: Vec::new(),
                chains: Vec::new(),
                source_atom_indices: Vec::new(),
                reference_coords: vec![[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
                bead_sites: vec![BeadSite {
                    target_bead_index: idx,
                    atom_indices: vec![0, 1],
                    weights: None,
                }],
                bonds: Vec::new(),
                chirality: Vec::new(),
            })
            .collect();
        let plan = BackmapPlan {
            templates,
            links: vec![
                Link {
                    from_template: 0,
                    from_atom: 1,
                    to_template: 1,
                    to_atom: 0,
                    target_distance: Some(1.0),
                },
                Link {
                    from_template: 1,
                    from_atom: 1,
                    to_template: 2,
                    to_atom: 0,
                    target_distance: Some(1.0),
                },
                Link {
                    from_template: 2,
                    from_atom: 1,
                    to_template: 0,
                    to_atom: 0,
                    target_distance: Some(1.0),
                },
            ],
            fudge_factor: 1.0,
        };
        let coords = plan
            .execute(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 2.0, 0.0]])
            .unwrap();
        assert_eq!(coords.len(), 6);
        assert!(coords.iter().flatten().all(|value| value.is_finite()));
    }

    #[test]
    fn rejects_non_invertible_overlapping_mapping() {
        let mut template = two_bead_template(0);
        template.bead_sites[1].atom_indices.push(1);
        let plan = BackmapPlan {
            templates: vec![template],
            links: vec![],
            fudge_factor: 1.0,
        };
        assert_eq!(
            plan.execute(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            Err(BackmapError::DuplicateMappedAtom(0, 1))
        );
    }

    #[test]
    fn rejects_invalid_numeric_contract() {
        let template = two_bead_template(0);
        let plan = BackmapPlan {
            templates: vec![template],
            links: vec![],
            fudge_factor: f64::NAN,
        };
        assert_eq!(
            plan.execute(&[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
            Err(BackmapError::InvalidFudgeFactor)
        );
    }

    #[test]
    fn output_is_restored_to_source_atom_order() {
        let template = |name: &str, source_idx: usize, bead_idx: usize| ResidueTemplate {
            name: name.into(),
            atom_names: vec![name.into()],
            elements: Vec::new(),
            residue_names: Vec::new(),
            residue_ids: Vec::new(),
            chains: Vec::new(),
            source_atom_indices: vec![source_idx],
            reference_coords: vec![[0.0, 0.0, 0.0]],
            bead_sites: vec![BeadSite {
                target_bead_index: bead_idx,
                atom_indices: vec![0],
                weights: None,
            }],
            bonds: Vec::new(),
            chirality: Vec::new(),
        };
        let plan = BackmapPlan {
            templates: vec![template("second", 1, 0), template("first", 0, 1)],
            links: vec![],
            fudge_factor: 1.0,
        };
        let rebuilt = plan.execute(&[[20.0, 0.0, 0.0], [10.0, 0.0, 0.0]]).unwrap();
        assert_eq!(rebuilt, vec![[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]);
        assert_eq!(plan.atom_names_in_output_order(), vec!["first", "second"]);
    }

    #[test]
    fn chirality_is_preserved_and_reported() {
        let plan = BackmapPlan {
            templates: vec![ResidueTemplate {
                name: "CHI".into(),
                atom_names: vec!["C".into(), "A".into(), "B".into(), "D".into()],
                elements: vec!["C".into(); 4],
                residue_names: vec!["CHI".into(); 4],
                residue_ids: vec![1; 4],
                chains: vec!["A".into(); 4],
                source_atom_indices: vec![0, 1, 2, 3],
                reference_coords: vec![
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                bead_sites: vec![BeadSite {
                    target_bead_index: 0,
                    atom_indices: vec![0, 1, 2, 3],
                    weights: None,
                }],
                bonds: vec![],
                chirality: vec![ChiralConstraint {
                    center: 0,
                    neighbor_a: 1,
                    neighbor_b: 2,
                    neighbor_c: 3,
                    reference_sign: 1,
                }],
            }],
            links: vec![],
            fudge_factor: 1.0,
        };
        let frame = plan.execute_frame(&[[5.0, 6.0, 7.0]]).unwrap();
        assert_eq!(frame.diagnostics.chirality_inversion_count, 0);
    }
}
