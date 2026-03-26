# MD Initial Structures with OpenMM + PACKMOL (Beginner Guide)

This file tracks the notebook:
`initial_structure_openmm_packmol.ipynb`

Theme: learn by building one small system at a time, then visualize immediately.

## What changed
1. Beginner-first flow: build cell followed by a py3Dmol cell.
2. Removed custom PDB/PDBx writer wrappers.
3. Uses OpenMM built-ins directly:
   - `app.PDBFile.writeFile(...)`
   - `app.PDBxFile.writeFile(...)`
4. Official PACKMOL archive is used for initial structure templates.
5. Notebook now writes readable `.inp` text in cells (not hidden external-only runs).
6. Final PACKMOL artifacts are exported as PDB.

## Setup behavior
The notebook setup cell:
1. Creates `outputs/`
2. Installs OpenMM/py3Dmol/PACKMOL in Colab if needed
3. Downloads and extracts:
   - `https://m3g.github.io/packmol/examples/examples.tar.gz`
   - to `packmol_examples/examples/`

## Learning path in notebook
1. Build one TIP3P water from scratch in OpenMM.
2. Solvate + ionize (`Modeller.addSolvent`) in a 2.2 nm box.
3. DIY grid packing for intuition.
4. PACKMOL water box from notebook-written input text.
5. Official-template examples with notebook `.inp` text:
   - water + urea mixture
   - hormone at interface
   - 4246-atom protein spherical solvation
6. Fun build: `HELLO` spelled with water molecules.
7. Optional peptide packing with `warp-pep` + `warp-pack`.

## Official templates usage policy
The tutorial uses official files as templates:
- `water.pdb`, `urea.pdb`, `protein.pdb`, `CLA.pdb`, `SOD.pdb`
- `water.xyz`, `chlor.xyz`, `t3.xyz`

For each major official case, the notebook:
1. Prints the official `.inp`
2. Writes a beginner-editable notebook `.inp`
3. Runs PACKMOL
4. Visualizes in the next cell

## PDB export note
- Mixture and solvprotein cases run directly to PDB.
- Interface case uses official XYZ templates, converts those templates to PDB first, then runs PACKMOL in PDB mode.
- Final packed interface artifact is directly produced as:
  - `outputs/08_official_interface.pdb`

## py3Dmol style pattern used
The visualization cells follow this pattern:

```python
view = py3Dmol.view()
view.addModel(open("some_file.pdb", "r").read(), "pdb")
view.setBackgroundColor("white")
view.setStyle({"chain": "A"}, {"cartoon": {"color": "purple"}})
view.zoomTo()
view.show()
```

## Main outputs
- `outputs/01_single_tip3p.pdb`
- `outputs/01_single_tip3p.cif`
- `outputs/02_waterbox_salt.pdb`
- `outputs/03_waterbox_grid.pdb`
- `outputs/04_waterbox_packmol.pdb`
- `outputs/07_official_mixture.pdb`
- `outputs/08_official_interface.pdb`
- `outputs/09_official_solvprotein.pdb`
- `outputs/10_hello_water.pdb`
- `outputs/11_pep_alpha_pack.pdb` (optional)
- `outputs/12_two_peptides_pack.pdb` (optional)

## Quick troubleshooting
- Missing `packmol`: install and rerun setup.
- Missing `warp-pep`/`warp-pack`: optional section will skip.
- Remember units: PACKMOL uses Angstrom, OpenMM uses nanometer internally.
