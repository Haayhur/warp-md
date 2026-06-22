"""Fallback agent contract adjunct snapshot.

Generated from Rust contract metadata. Do not edit by hand.
"""

ERROR_CODES = ('E_CONFIG_LOAD',
 'E_CONFIG_VALIDATION',
 'E_CONFIG_VERSION',
 'E_CONFIG_MISSING_FIELD',
 'E_ANALYSIS_UNKNOWN',
 'E_ANALYSIS_SPEC',
 'E_SELECTION_EMPTY',
 'E_SELECTION_INVALID',
 'E_SYSTEM_LOAD',
 'E_TRAJECTORY_LOAD',
 'E_TRAJECTORY_EOF',
 'E_RUNTIME_EXEC',
 'E_OUTPUT_DIR',
 'E_OUTPUT_WRITE',
 'E_DEVICE_UNAVAILABLE',
 'E_INPUT_MISSING',
 'E_UNSUPPORTED_FORMAT',
 'E_TOPOLOGY_TRAJECTORY_MISMATCH',
 'E_TOPOLOGY_ATOM_MISSING',
 'E_NO_FRAMES',
 'E_EXTERNAL_TABLE_LOAD',
 'E_EXTERNAL_TABLE_COLUMN',
 'E_PLOT_RENDER',
 'E_BUNDLE_PARTIAL',
 'E_INTERNAL')

BOX_REQUIRED_ANALYSES = ('bondi_ffv',
 'conductivity',
 'density',
 'dielectric',
 'free_volume',
 'gist',
 'rdf',
 'structure_factor',
 'volmap',
 'water_count',
 'watershell')

VELOCITY_REQUIRED_ANALYSES = ('equipartition',)

ANALYSIS_BUNDLES = {'standard_md_report': {'description': 'General MD report: structure, transport, density, and '
                                       'external state/energy series.',
                        'analyses': ['rg', 'rmsd', 'rdf', 'msd', 'density'],
                        'external_tables': ['energy_table', 'state_table']},
 'protein_md_report': {'description': 'Protein trajectory report: fold stability, secondary '
                                      'structure, hydrogen bonds, contacts.',
                       'analyses': ['rg', 'rmsd', 'dssp', 'hbond', 'native_contacts']},
 'solvent_ion_report': {'description': 'Solvent and ion report: RDF, diffusion, electrostatics, '
                                       'and hydration structure.',
                        'analyses': ['rdf',
                                     'msd',
                                     'conductivity',
                                     'dielectric',
                                     'water_count',
                                     'watershell']},
 'polymer_report': {'description': 'Polymer report: size, chain geometry, persistence, and free '
                                   'volume.',
                    'analyses': ['rg',
                                 'chain_rg',
                                 'end_to_end',
                                 'contour_length',
                                 'persistence_length',
                                 'bondi_ffv']}}
