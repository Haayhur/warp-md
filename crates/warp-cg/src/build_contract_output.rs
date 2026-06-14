use super::build_contract_output_files::prepare_output_path;
use super::*;
use traj_core::geometry::Vec3;
use warp_structure::{io::write_output, AtomRecord, AtomRecordKind, OutputSpec, PackOutput};

pub(super) fn write_coordinates_and_topology(
    request: &BuildRequest,
    result: &BuildResult,
    beads: &[EmittedBead],
) -> Result<()> {
    let out = pack_output_from_beads(&result.box_meta, beads);
    if let Some(path) = &request.outputs.coordinates {
        write_build_coordinates(&out, path, None, &request.outputs)?;
    }
    if let Some(path) = &request.outputs.gro {
        write_build_coordinates(&out, path, Some("gro"), &request.outputs)?;
    }
    if let Some(path) = &request.outputs.pdb {
        write_build_coordinates(&out, path, Some("pdb"), &request.outputs)?;
    }
    if let Some(path) = &request.outputs.cif {
        write_build_coordinates(&out, path, Some("cif"), &request.outputs)?;
    }
    if let Some(path) = &request.outputs.topology {
        prepare_output_path(
            path,
            request.outputs.overwrite,
            request.outputs.backup_existing,
        )?;
        write_topology(path, request, result)?;
    }
    if let Some(path) = &request.outputs.log {
        write_build_log(path, request, result)?;
    }
    if let Some(path) = &request.outputs.snapshot {
        write_build_snapshot(path, request, result)?;
    }
    Ok(())
}

fn pack_output_from_beads(box_meta: &BuildBoxSummary, beads: &[EmittedBead]) -> PackOutput {
    let atoms = beads
        .iter()
        .enumerate()
        .map(|(idx, bead)| AtomRecord {
            record_kind: AtomRecordKind::Atom,
            name: bead.atom_name.clone(),
            element: "X".to_string(),
            resname: bead.residue_name.clone(),
            resid: bead.residue_id,
            chain: 'A',
            segid: String::new(),
            charge: bead.charge_e,
            position: Vec3::new(
                bead.position_angstrom[0],
                bead.position_angstrom[1],
                bead.position_angstrom[2],
            ),
            mol_id: idx as i32 + 1,
            pdb_metadata: None,
        })
        .collect();
    PackOutput {
        atoms,
        bonds: Vec::new(),
        box_size: box_meta.box_size_angstrom,
        box_vectors: Some(box_meta.box_vectors_angstrom),
        ter_after: Vec::new(),
    }
}

fn write_build_coordinates(
    out: &PackOutput,
    path: &str,
    format: Option<&str>,
    outputs: &BuildOutputs,
) -> Result<()> {
    prepare_output_path(path, outputs.overwrite, outputs.backup_existing)?;
    let spec = OutputSpec {
        path: path.to_string(),
        format: format.unwrap_or("").to_string(),
        scale: None,
    };
    write_output(out, &spec, true, 0.0, false, false)
        .map(|_| ())
        .map_err(|err| anyhow!(err.to_string()))
}

fn write_build_log(path: &str, request: &BuildRequest, result: &BuildResult) -> Result<()> {
    prepare_output_path(
        path,
        request.outputs.overwrite,
        request.outputs.backup_existing,
    )?;
    let text = format!(
        "warp-cg build\nschema_version: {}\nrun_id: {}\nstatus: {}\nmode: {}\nelapsed_ms: {}\n\n[ box ]\nbox_type: {}\npbc: {}\nbox_size_angstrom: {:?}\nunit_cell_angstrom: {:?}\n\n[ summary ]\nmembrane_count: {}\nleaflet_count: {}\nbead_count: {}\nprotein_count: {}\nsolute_count: {}\nlipid_counts: {:?}\ninserted_counts: {:?}\nsolvent_counts: {:?}\n\n[ charge ]\nnet_charge_before_neutralization_e: {:?}\nsolvent_charge_e: {:?}\nbaseline_ion_charge_e: {:?}\nneutralization_input_charge_e: {:?}\nneutralization: {}\nresidual_charge_e: {:?}\ncharge_sources: {:?}\n\n[ placement ]\nalgorithm: {}\nmode: {}\nrandom_seed: {:?}\ninserted_flood: candidate_count={} grid_squeeze_pass_count={} squeezed_candidate_count={} min_spacing_angstrom={:?} kick_attempt_count={} kicked_inserted_count={}\nleaflet_metric_count: {}\nsolvent: {}\ndiagnostics: bead_count={} distinct_residue_count={} excluded_bead_count={} pbc_axes={:?} min_inter_residue_distance_angstrom={:?} min_exclusion_margin_angstrom={:?} exclusion_violation_count={}\n\n[ artifacts ]\ncoordinates: {:?}\ngro: {:?}\npdb: {:?}\ncif: {:?}\ntopology: {:?}\nlog: {:?}\nsnapshot: {:?}\nmanifest: {}\noverwrite: {}\nbackup_existing: {}\n\n[ warnings ]\ncount: {}\n",
        result.schema_version,
        request.run_id.as_deref().unwrap_or(""),
        result.status,
        result.mode,
        result.elapsed_ms,
        result.box_meta.box_type,
        result.box_meta.pbc,
        result.box_meta.box_size_angstrom,
        result.box_meta.unit_cell_angstrom,
        result.summary.membrane_count,
        result.summary.leaflet_count,
        result.summary.bead_count,
        result.summary.protein_count,
        result.summary.solute_count,
        result.summary.lipid_counts,
        result.summary.inserted_counts,
        result.summary.solvent_counts,
        result.charge.net_charge_before_neutralization_e,
        result.charge.solvent_charge_e,
        result.charge.baseline_ion_charge_e,
        result.charge.neutralization_input_charge_e,
        result.charge.neutralization.enabled,
        result.charge.neutralization.residual_charge_e,
        result.charge.charge_sources,
        result.placement.algorithm,
        result.placement.mode,
        result.placement.random_seed,
        result.placement.inserted_flood.candidate_count,
        result.placement.inserted_flood.grid_squeeze_pass_count,
        result.placement.inserted_flood.squeezed_candidate_count,
        result.placement.inserted_flood.min_spacing_angstrom,
        result.placement.inserted_flood.kick_attempt_count,
        result.placement.inserted_flood.kicked_inserted_count,
        result.placement.leaflet_metrics.len(),
        result.placement.solvent.is_some(),
        result.placement.diagnostics.bead_count,
        result.placement.diagnostics.distinct_residue_count,
        result.placement.diagnostics.excluded_bead_count,
        result.placement.diagnostics.pbc_axes,
        result
            .placement
            .diagnostics
            .min_inter_residue_distance_angstrom,
        result.placement.diagnostics.min_exclusion_margin_angstrom,
        result.placement.diagnostics.exclusion_violation_count,
        result.artifacts.coordinates,
        result.artifacts.gro,
        result.artifacts.pdb,
        result.artifacts.cif,
        result.artifacts.topology,
        result.artifacts.log,
        result.artifacts.snapshot,
        result.artifacts.manifest,
        result.artifacts.output_policy.overwrite,
        result.artifacts.output_policy.backup_existing,
        result.warnings.len()
    );
    std::fs::write(path, text)?;
    Ok(())
}

fn write_build_snapshot(path: &str, request: &BuildRequest, result: &BuildResult) -> Result<()> {
    prepare_output_path(
        path,
        request.outputs.overwrite,
        request.outputs.backup_existing,
    )?;
    let snapshot = json!({
        "schema_version": "warp-cg.build.snapshot.v1",
        "request_schema_version": BUILD_SCHEMA_VERSION,
        "request": request,
        "result": result
    });
    std::fs::write(path, serde_json::to_string_pretty(&snapshot)?)?;
    Ok(())
}

fn write_topology(path: &str, request: &BuildRequest, result: &BuildResult) -> Result<()> {
    let mut text = String::new();
    text.push_str("; Generated by warp-cg build\n\n");
    for block in topology_molecule_blocks(request, result)? {
        text.push_str(&block);
        text.push('\n');
    }
    text.push_str("[ system ]\nwarp-cg membrane\n\n[ molecules ]\n");
    for (lipid, count) in &result.summary.lipid_counts {
        text.push_str(&format!("{lipid:<16} {count}\n"));
    }
    for (name, count) in &result.summary.inserted_counts {
        if *count > 0 {
            text.push_str(&format!("{name:<16} {count}\n"));
        }
    }
    for (name, count) in &result.summary.solvent_counts {
        if *count > 0 {
            text.push_str(&format!("{name:<16} {count}\n"));
        }
    }
    std::fs::write(path, text)?;
    Ok(())
}

fn topology_molecule_blocks(request: &BuildRequest, result: &BuildResult) -> Result<Vec<String>> {
    let mut blocks = Vec::new();
    let mut seen = BTreeSet::new();
    for membrane in &request.membranes {
        for leaflet in &membrane.leaflets {
            for lipid in &leaflet.composition {
                if seen.contains(&lipid.lipid) {
                    continue;
                }
                if result
                    .summary
                    .lipid_counts
                    .get(&lipid.lipid)
                    .copied()
                    .unwrap_or(0)
                    == 0
                {
                    continue;
                }
                let template = lookup_lipid_template(&lipid.lipid, &request.system.force_field);
                let beads = resolved_lipid_topology_beads(lipid, template.as_ref())?;
                if seen.insert(lipid.lipid.clone()) {
                    blocks.push(lipid_topology_block_from_beads(&lipid.lipid, &beads));
                }
            }
        }
    }
    for (name, count) in &result.summary.lipid_counts {
        if *count == 0 || seen.contains(name) {
            continue;
        }
        if let Some(template) = lookup_lipid_template(name, &request.system.force_field) {
            if seen.insert(template.name.clone()) {
                blocks.push(lipid_template_topology_block(&template));
            }
        }
    }
    for component in request.proteins.iter().chain(request.solutes.iter()) {
        if let Some(definition_path) = &component.definition {
            if seen.insert(component.name.clone()) {
                let definition = load_molecule_definition(definition_path)?;
                let beads = molecule_definition_beads(&definition)?;
                blocks.push(molecule_definition_topology_block(
                    &component.name,
                    &beads,
                    &definition.bonds,
                    &definition.angles,
                    &definition.dihedrals,
                ));
            }
        }
        if component.charge_topology.is_some() || !component.charge_topologies.is_empty() {
            let molecule_types = inserted_molecule_types(component);
            let topology_paths = inserted_charge_topology_paths(component, molecule_types.len())?;
            for (molecule_type, topology_path) in molecule_types.iter().zip(topology_paths.iter()) {
                if !seen.insert(molecule_type.clone()) {
                    continue;
                }
                let topology = warp_common::charge::read_gromacs_molecule_topology(
                    Path::new(topology_path),
                    molecule_type,
                )
                .map_err(|err| anyhow!(err))?;
                blocks.push(gromacs_molecule_topology_block(&topology));
            }
        }
        if component.definition.is_none()
            && component.charge_topology.is_none()
            && component.charge_topologies.is_empty()
            && component.coordinates.is_none()
            && component.beads.is_empty()
        {
            if let Some(template) = lookup_solute_template(&component.name) {
                if seen.insert(template.name.to_string()) {
                    blocks.push(solute_template_topology_block(
                        template.name,
                        template.beads,
                    ));
                }
            } else if let Some(library) = lookup_solvent_library(&component.name) {
                if seen.insert(library.name.clone()) {
                    blocks.push(solvent_library_topology_block(&library));
                }
            }
        }
    }
    for (name, count) in &result.summary.solvent_counts {
        if *count == 0 || seen.contains(name) {
            continue;
        }
        if let Some(library) = lookup_solvent_library(name) {
            if seen.insert(library.name.clone()) {
                blocks.push(solvent_library_topology_block(&library));
            }
            continue;
        }
        if let Some(ion) = lookup_ion_library(name) {
            if seen.insert(ion.name.to_string()) {
                blocks.push(ion_library_topology_block(&ion));
            }
        }
    }
    Ok(blocks)
}

fn resolved_lipid_topology_beads(
    lipid: &LipidComponent,
    template: Option<&crate::build_lipids::LipidTemplate>,
) -> Result<Vec<BuildBeadTemplate>> {
    let beads = normalized_lipid_beads(lipid, template)?;
    let template_charge = template
        .map(|template| template.net_charge_e)
        .unwrap_or(0.0);
    let has_template = template.is_some();
    let (_, _, beads) = resolve_lipid_charge(lipid, template_charge, has_template, beads)?;
    Ok(beads)
}

fn lipid_template_topology_block(template: &crate::build_lipids::LipidTemplate) -> String {
    lipid_topology_block_from_beads(&template.name, &template.beads)
}

fn lipid_topology_block_from_beads(
    molecule_name: &str,
    beads: &[impl LipidTopologyBead],
) -> String {
    let beads = beads
        .iter()
        .map(|bead| BuildBeadTemplate {
            name: bead.topology_name(),
            offset_angstrom: bead.topology_offset(),
            charge_e: bead.topology_charge(),
        })
        .collect::<Vec<_>>();
    molecule_definition_topology_block(molecule_name, &beads, &[], &[], &[])
}

trait LipidTopologyBead {
    fn topology_name(&self) -> String;
    fn topology_offset(&self) -> [f32; 3];
    fn topology_charge(&self) -> f32;
}

impl LipidTopologyBead for crate::build_lipids::TemplateBead {
    fn topology_name(&self) -> String {
        self.name.clone()
    }

    fn topology_offset(&self) -> [f32; 3] {
        self.offset_angstrom
    }

    fn topology_charge(&self) -> f32 {
        self.charge_e
    }
}

impl LipidTopologyBead for BuildBeadTemplate {
    fn topology_name(&self) -> String {
        self.name.clone()
    }

    fn topology_offset(&self) -> [f32; 3] {
        self.offset_angstrom
    }

    fn topology_charge(&self) -> f32 {
        self.charge_e
    }
}

fn solute_template_topology_block(
    molecule_name: &str,
    beads: &[crate::build_solutes::SoluteTemplateBead],
) -> String {
    let bead_templates = beads
        .iter()
        .map(|bead| BuildBeadTemplate {
            name: bead.name.to_string(),
            offset_angstrom: bead.offset_angstrom,
            charge_e: bead.charge_e,
        })
        .collect::<Vec<_>>();
    let bonds = lookup_solute_template_bonds(molecule_name)
        .iter()
        .map(|bond| MoleculeDefinitionBond {
            bead_indices: bond.bead_indices,
            length_nm: Some(bond.length_nm),
            force_kj_mol_nm2: Some(bond.force_kj_mol_nm2),
        })
        .collect::<Vec<_>>();
    molecule_definition_topology_block(molecule_name, &bead_templates, &bonds, &[], &[])
}

pub(super) fn solvent_library_topology_block(library: &SolventLibraryEntry) -> String {
    let bead_templates = library
        .beads
        .iter()
        .map(|bead| BuildBeadTemplate {
            name: bead.atom_name.clone(),
            offset_angstrom: bead.offset_angstrom,
            charge_e: bead.charge_e,
        })
        .collect::<Vec<_>>();
    let mut bonds = lookup_solute_template_bonds(&library.name)
        .iter()
        .map(|bond| MoleculeDefinitionBond {
            bead_indices: bond.bead_indices,
            length_nm: Some(bond.length_nm),
            force_kj_mol_nm2: Some(bond.force_kj_mol_nm2),
        })
        .collect::<Vec<_>>();
    if bonds.is_empty() {
        bonds = standard_solvent_bonds(&library.name);
    }
    let angles = standard_solvent_angles(&library.name);
    molecule_definition_topology_block(&library.name, &bead_templates, &bonds, &angles, &[])
}

fn ion_library_topology_block(ion: &IonLibraryEntry) -> String {
    let beads = [BuildBeadTemplate {
        name: ion.atom_name.to_string(),
        offset_angstrom: [0.0, 0.0, 0.0],
        charge_e: ion.charge_e as f32,
    }];
    molecule_definition_topology_block(ion.name, &beads, &[], &[], &[])
}

fn molecule_definition_topology_block(
    molecule_name: &str,
    beads: &[BuildBeadTemplate],
    bonds: &[MoleculeDefinitionBond],
    angles: &[MoleculeDefinitionAngle],
    dihedrals: &[MoleculeDefinitionDihedral],
) -> String {
    let mut text = String::new();
    text.push_str("[ moleculetype ]\n");
    text.push_str("; name  nrexcl\n");
    text.push_str(&format!("{:<16} 1\n\n", topology_token(molecule_name)));
    text.push_str("[ atoms ]\n");
    text.push_str("; nr  type  resnr  residue  atom  cgnr  charge\n");
    for (idx, bead) in beads.iter().enumerate() {
        let atom = topology_token(&bead.name);
        text.push_str(&format!(
            "{:>5} {:<6} {:>5} {:<8} {:<6} {:>5} {:>8.3}\n",
            idx + 1,
            atom,
            1,
            topology_token(molecule_name),
            atom,
            idx + 1,
            bead.charge_e
        ));
    }
    if !bonds.is_empty() {
        text.push_str("\n[ bonds ]\n");
        text.push_str("; i  j  funct  length(nm)  force\n");
        for bond in bonds {
            text.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>10.5} {:>10.3}\n",
                bond.bead_indices[0] + 1,
                bond.bead_indices[1] + 1,
                1,
                bond.length_nm.unwrap_or(0.47),
                bond.force_kj_mol_nm2.unwrap_or(1250.0)
            ));
        }
    }
    if !angles.is_empty() {
        text.push_str("\n[ angles ]\n");
        text.push_str("; i  j  k  funct  angle(deg)  force\n");
        for angle in angles {
            text.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>10.3} {:>10.3}\n",
                angle.bead_indices[0] + 1,
                angle.bead_indices[1] + 1,
                angle.bead_indices[2] + 1,
                2,
                angle.angle_degrees.unwrap_or(180.0),
                angle.force_kj_mol_rad2.unwrap_or(25.0)
            ));
        }
    }
    if !dihedrals.is_empty() {
        text.push_str("\n[ dihedrals ]\n");
        text.push_str("; i  j  k  l  funct  phase(deg)  force  mult\n");
        for dihedral in dihedrals {
            text.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>5} {:>10.3} {:>10.3} {:>5}\n",
                dihedral.bead_indices[0] + 1,
                dihedral.bead_indices[1] + 1,
                dihedral.bead_indices[2] + 1,
                dihedral.bead_indices[3] + 1,
                1,
                dihedral.phase_degrees.unwrap_or(0.0),
                dihedral.force_kj_mol.unwrap_or(0.0),
                dihedral.multiplicity.unwrap_or(1)
            ));
        }
    }
    text
}

fn gromacs_molecule_topology_block(topology: &GromacsMoleculeTopology) -> String {
    let molecule_name = topology_token(&topology.molecule_type);
    let mut text = String::new();
    text.push_str("[ moleculetype ]\n");
    text.push_str("; name  nrexcl\n");
    text.push_str(&format!("{molecule_name:<16} 1\n\n"));
    text.push_str("[ atoms ]\n");
    text.push_str("; nr  type  resnr  residue  atom  cgnr  charge\n");
    for atom in &topology.atoms {
        text.push_str(&format!(
            "{:>5} {:<6} {:>5} {:<8} {:<6} {:>5} {:>8.3}\n",
            atom.nr,
            topology_token(&atom.atom_type),
            atom.residue_nr,
            topology_token(&atom.residue_name),
            topology_token(&atom.atom_name),
            atom.charge_group,
            atom.charge_e
        ));
    }
    if !topology.bonds.is_empty() {
        text.push_str("\n[ bonds ]\n");
        text.push_str("; i  j  funct  length(nm)  force\n");
        for bond in &topology.bonds {
            text.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>10.5} {:>10.3}\n",
                bond.atom_i,
                bond.atom_j,
                bond.function,
                bond.length_nm.unwrap_or(0.47),
                bond.force_kj_mol_nm2.unwrap_or(1250.0)
            ));
        }
    }
    if !topology.angles.is_empty() {
        text.push_str("\n[ angles ]\n");
        text.push_str("; i  j  k  funct  angle(deg)  force\n");
        for angle in &topology.angles {
            text.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>10.3} {:>10.3}\n",
                angle.atom_i,
                angle.atom_j,
                angle.atom_k,
                angle.function,
                angle.angle_degrees.unwrap_or(180.0),
                angle.force_kj_mol_rad2.unwrap_or(25.0)
            ));
        }
    }
    if !topology.dihedrals.is_empty() {
        text.push_str("\n[ dihedrals ]\n");
        text.push_str("; i  j  k  l  funct  phase(deg)  force  mult\n");
        for dihedral in &topology.dihedrals {
            text.push_str(&format!(
                "{:>5} {:>5} {:>5} {:>5} {:>5} {:>10.3} {:>10.3} {:>5}\n",
                dihedral.atom_i,
                dihedral.atom_j,
                dihedral.atom_k,
                dihedral.atom_l,
                dihedral.function,
                dihedral.phase_degrees.unwrap_or(0.0),
                dihedral.force_kj_mol.unwrap_or(0.0),
                dihedral.multiplicity.unwrap_or(1)
            ));
        }
    }
    text
}

fn topology_token(name: &str) -> String {
    let token = name
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_' || *ch == '-')
        .take(16)
        .collect::<String>();
    if token.is_empty() {
        "MOL".to_string()
    } else {
        token
    }
}
