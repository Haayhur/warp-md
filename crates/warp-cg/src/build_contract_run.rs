use super::*;

pub fn validate_request_json(text: &str) -> (i32, Value) {
    match parse_request(text).and_then(validate_request) {
        Ok(request) => (
            0,
            json!({
                "schema_version": BUILD_SCHEMA_VERSION,
                "status": "ok",
                "valid": true,
                "normalized_request": request,
                "errors": [],
                "warnings": []
            }),
        ),
        Err(err) => (
            2,
            json!({
                "schema_version": BUILD_SCHEMA_VERSION,
                "status": "error",
                "valid": false,
                "errors": [{
                    "code": "E_BUILD_REQUEST",
                    "path": "",
                    "message": err.to_string(),
                    "severity": "error"
                }],
                "warnings": []
            }),
        ),
    }
}

pub fn run_request_json(text: &str, stream_ndjson: bool) -> (i32, Value) {
    let started = Instant::now();
    match parse_request(text).and_then(validate_request) {
        Ok(request) => {
            if stream_ndjson {
                eprintln!(
                    "{}",
                    serde_json::to_string(&BuildEvent::BuildStarted {
                        schema_version: BUILD_SCHEMA_VERSION.to_string(),
                        run_id: request.run_id.clone()
                    })
                    .unwrap_or_default()
                );
            }
            match run_request(request, started) {
                Ok(result) => {
                    if stream_ndjson {
                        eprintln!(
                            "{}",
                            serde_json::to_string(&BuildEvent::ChargeResolved {
                                schema_version: BUILD_SCHEMA_VERSION.to_string(),
                                net_charge_e: result.charge.net_charge_before_neutralization_e
                            })
                            .unwrap_or_default()
                        );
                        eprintln!(
                            "{}",
                            serde_json::to_string(&BuildEvent::BuildComplete {
                                schema_version: BUILD_SCHEMA_VERSION.to_string(),
                                status: result.status.clone()
                            })
                            .unwrap_or_default()
                        );
                    }
                    (0, serde_json::to_value(result).unwrap_or_else(|err| {
                        json!({"schema_version": BUILD_SCHEMA_VERSION, "status": "error", "error": err.to_string()})
                    }))
                }
                Err(err) => error_result(err),
            }
        }
        Err(err) => error_result(err),
    }
}

fn error_result(err: anyhow::Error) -> (i32, Value) {
    (
        2,
        json!({
            "schema_version": BUILD_SCHEMA_VERSION,
            "status": "error",
            "error": {
                "code": "E_BUILD_REQUEST",
                "path": "",
                "message": err.to_string(),
                "severity": "error"
            }
        }),
    )
}

fn parse_request(text: &str) -> Result<BuildRequest> {
    expand_stacked_membranes(serde_json::from_str(text)?)
}

pub fn run_request(request: BuildRequest, started: Instant) -> Result<BuildResult> {
    let mut lipid_counts = BTreeMap::new();
    let mut inserted_counts = BTreeMap::new();
    let mut component_charges = Vec::new();
    let mut shared_components = Vec::new();
    let mut charge_sources = Vec::new();
    let mut emitted_beads = Vec::new();
    let mut solvent_counts = BTreeMap::new();
    let mut leaflet_metrics = Vec::new();
    let mut inserted_flood = InsertedFloodPlacementSummary::default();
    let mut next_residue_id = 1i32;
    let protein_exclusions = protein_component_exclusions(&request.proteins)?;

    for membrane in &request.membranes {
        for leaflet in &membrane.leaflets {
            let leaflet_exclusions =
                combined_leaflet_exclusions(leaflet, protein_exclusions.as_slice());
            let mut leaflet_exclusions = leaflet_exclusions;
            leaflet_exclusions.extend(boundary_protein_exclusions_for_leaflet(
                membrane,
                leaflet,
                &request.proteins,
            )?);
            let (resolved_lipids, area_summary) = resolve_leaflet_lipid_counts(
                &request.system,
                membrane,
                leaflet,
                &request.proteins,
            )?;
            let (leaflet_beads, metrics, geometry) = layout_leaflet_beads(
                &request.system,
                membrane,
                leaflet,
                &request.proteins,
                &leaflet_exclusions,
                &resolved_lipids,
                &mut next_residue_id,
            )?;
            leaflet_metrics.push(LeafletPlacementSummary {
                membrane: membrane.name.clone(),
                leaflet: leaflet.name.clone(),
                lipid_count: resolved_lipids.iter().map(|lipid| lipid.count).sum(),
                exclusion_count: leaflet_exclusions.len(),
                area: area_summary,
                metrics,
                geometry,
            });
            emitted_beads.extend(leaflet_beads);
            for lipid in resolved_lipids {
                *lipid_counts.entry(lipid.name.clone()).or_insert(0) += lipid.count;
                let total = lipid.charge_e * lipid.count as f32;
                component_charges.push(ComponentChargeSummary {
                    name: format!("membrane:{}:{}:{}", membrane.name, leaflet.name, lipid.name),
                    count: lipid.count,
                    per_instance_net_charge_e: Some(lipid.charge_e),
                    per_instance_bead_charge_sum_e: Some(lipid.bead_charge_sum_e()),
                    charge_balance_delta_e: Some(lipid.bead_charge_sum_e() - lipid.charge_e),
                    total_charge_e: Some(total),
                    source: lipid.charge_source.clone(),
                });
                shared_components.push(ComponentCharge {
                    name: lipid.name.clone(),
                    count: lipid.count,
                    per_instance_net_charge_e: Some(lipid.charge_e),
                });
                charge_sources.push(lipid.charge_source.clone());
                charge_sources.push(lipid.template_source.clone());
            }
        }
    }

    for component in &request.proteins {
        let inserted = emit_inserted_component(
            component,
            InsertedKind::Protein,
            &request.system,
            &emitted_beads,
            &mut inserted_flood,
            &mut next_residue_id,
        )?;
        emitted_beads.extend(inserted);
        append_inserted_counts(component, &mut inserted_counts);
        append_inserted_charge(
            component,
            &mut component_charges,
            &mut shared_components,
            &mut charge_sources,
        )?;
    }
    for component in &request.solutes {
        let inserted = emit_inserted_component(
            component,
            InsertedKind::Solute,
            &request.system,
            &emitted_beads,
            &mut inserted_flood,
            &mut next_residue_id,
        )?;
        emitted_beads.extend(inserted);
        append_inserted_counts(component, &mut inserted_counts);
        append_inserted_charge(
            component,
            &mut component_charges,
            &mut shared_components,
            &mut charge_sources,
        )?;
    }

    charge_sources.sort();
    charge_sources.dedup();
    let net_charge = warp_common::charge::sum_component_charges(&shared_components);
    let mut neutralization = resolve_neutralization(&request.environment.ions, net_charge);
    let mut solvent_charge_e = None;
    let mut baseline_ion_charge_e = None;
    let mut neutralization_input_charge_e = None;
    let solvent_placement = if request.environment.solvent.enabled {
        let solvent = emit_solvent_and_ions(
            &request,
            net_charge,
            &mut neutralization,
            &mut emitted_beads,
            &mut next_residue_id,
        )?;
        solvent_charge_e = Some(solvent.solvent_charge_e);
        baseline_ion_charge_e = Some(solvent.baseline_ion_charge_e);
        neutralization_input_charge_e = Some(solvent.neutralization_input_charge_e);
        for (name, count) in &solvent.counts {
            solvent_counts.insert(name.clone(), *count);
        }
        Some(solvent.summary)
    } else {
        None
    };
    let leaflet_count = request
        .membranes
        .iter()
        .map(|membrane| membrane.leaflets.len())
        .sum();
    let box_meta = resolved_box_metadata(&request.system)?;
    let placement_diagnostics = placement_diagnostics(&request.system, &emitted_beads);

    let result = BuildResult {
        schema_version: BUILD_SCHEMA_VERSION.to_string(),
        status: "ok".to_string(),
        run_id: request.run_id.clone(),
        mode: request.mode.clone(),
        box_meta,
        summary: BuildSummary {
            membrane_count: request.membranes.len(),
            leaflet_count,
            lipid_counts,
            inserted_counts,
            bead_count: emitted_beads.len(),
            solvent_counts,
            protein_count: request
                .proteins
                .iter()
                .map(|component| component.count)
                .sum(),
            solute_count: request
                .solutes
                .iter()
                .map(|component| component.count)
                .sum(),
        },
        charge: ChargeBuildSummary {
            component_charges,
            net_charge_before_neutralization_e: net_charge,
            solvent_charge_e,
            baseline_ion_charge_e,
            neutralization_input_charge_e,
            neutralization,
            charge_sources,
        },
        placement: PlacementBuildSummary {
            algorithm: placement_algorithm_name(&request.system),
            mode: request.system.placement.mode.clone(),
            candidate_source: request.system.placement.candidate_source.clone(),
            random_seed: request.system.placement.random_seed,
            inserted_flood,
            leaflet_metrics,
            solvent: solvent_placement,
            diagnostics: placement_diagnostics,
        },
        artifacts: BuildArtifacts {
            coordinates: request.outputs.coordinates.clone(),
            gro: request.outputs.gro.clone(),
            pdb: request.outputs.pdb.clone(),
            cif: request.outputs.cif.clone(),
            topology: request.outputs.topology.clone(),
            log: request.outputs.log.clone(),
            snapshot: request.outputs.snapshot.clone(),
            manifest: request.outputs.manifest.clone(),
            output_policy: BuildOutputPolicy {
                overwrite: request.outputs.overwrite,
                backup_existing: request.outputs.backup_existing,
            },
        },
        warnings: Vec::new(),
        elapsed_ms: started.elapsed().as_millis(),
    };

    write_coordinates_and_topology(&request, &result, &emitted_beads)?;
    write_manifest(&result)?;
    Ok(result)
}
