---
description: Your agent's way of pointing at atoms
icon: crosshairs
---

# Atom Selections

How does your agent tell warp-md which atoms to analyze? With selections — a simple query language that's surprisingly powerful.

---

## Basic Syntax

Selections are like SQL for molecules. Match atom properties with intuitive predicates:

```python
from warp_md import System

system = System.from_pdb("protein.pdb")

# Select by atom name
ca_atoms = system.select("name CA")  # All alpha carbons

# Select by residue name
water = system.select("resname SOL")  # All water molecules

# Select by residue ID (supports ranges!)
first_10 = system.select("resid 1-10")

# Select by chain
chain_a = system.select("chain A")
```

---

## The Selection Menu

| Predicate | What It Does | Examples |
|-----------|--------------|----------|
| `name` | Match atom name | `name CA`, `name OW` |
| `resname` | Match residue name | `resname ALA`, `resname SOL` |
| `resid` | Match residue ID (ranges OK) | `resid 10`, `resid 10-50` |
| `chain` | Match chain identifier | `chain A`, `chain B` |
| `protein` | All protein atoms | `protein` |
| `backbone` | Protein backbone (N, CA, C, O) | `backbone` |

---

## Boolean Logic: Get Precise

Combine predicates with `and`, `or`, `not`, and parentheses. Your agent can express exactly what it needs:

```python
# AND: atoms matching both conditions
backbone_chain_a = system.select("backbone and chain A")

# OR: atoms matching either condition
ala_or_gly = system.select("resname ALA or resname GLY")

# NOT: atoms NOT matching condition
non_water = system.select("not resname SOL")

# Complex selections with parentheses
complex_sel = system.select("(resname ALA or resname GLY) and backbone")
```

---

## Common Patterns (Copy-Paste Ready)

{% tabs %}
{% tab title="Protein Analysis" %}
```python
# All protein atoms
protein = system.select("protein")

# Backbone only
backbone = system.select("backbone")

# Alpha carbons (classic for Rg/RMSD)
ca = system.select("name CA")

# Active site residues
active_site = system.select("resid 100-150 and chain A")
```
{% endtab %}

{% tab title="Solvent & Ions" %}
```python
# Water oxygens (GROMACS naming)
water_o = system.select("resname SOL and name OW")

# Alternative water naming (CHARMM/NAMD)
water_o_alt = system.select("resname WAT and name O")

# Ions
sodium = system.select("resname NA")
chloride = system.select("resname CL")
```
{% endtab %}

{% tab title="Ionic Liquids" %}
```python
# Cations and anions
cations = system.select("resname BMIM")
anions = system.select("resname BF4")

# All ionic species
ions = system.select("resname BMIM or resname BF4")
```
{% endtab %}
{% endtabs %}

---

## Using Selections in Analyses

Selections plug directly into analysis plans:

```python
from warp_md import RgPlan, RmsdPlan

# Create selection once
backbone = system.select("backbone")

# Reuse it everywhere
rg = RgPlan(backbone, mass_weighted=False)
rmsd = RmsdPlan(backbone, reference="topology", align=True)
```

---

## Pro Tips

{% hint style="success" %}
**Performance**: Create selections once and reuse them. Don't re-select in a loop — that's inefficient and your agent knows better.
{% endhint %}

{% hint style="warning" %}
**Case matters**: `name CA` ≠ `name ca`. Atom names are case-sensitive.
{% endhint %}

{% hint style="info" %}
**Ranges are inclusive**: `resid 10-50` includes both 10 and 50.
{% endhint %}

---

<a href="basic-analyses.md" class="button primary">Next: Basic Analyses →</a>
