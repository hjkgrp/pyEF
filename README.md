# pyEF

A Python package for calculating electric fields, electrostatic potentials (ESP), and electrostatic interactions from quantum mechanical calculations.

For more details on the theory behind this package and best practices for using it in your research, please see our open source paper:

Melissa T. Manetsch, David W. Kastner, Yuriy Román-Leshkov, Heather J. Kulik; pyEF: A Python Framework for QM and QM/MM Atom-Wise Electric Field Analysis. J. Chem. Theory Comput. 26 May 2026; 22 (10): 4982–4994. https://doi.org/10.1021/acs.jctc.6c00065

(we also request that you cite the above paper if you find this package helpful in your research!)

Note: This package is under active development as of 2026, if you encounter any issues during use or have any features you would like to see incorporated, please do not hesitate to raise an issue or reach out to the developers at manets12@mit.edu. 

## Table of Contents
1. **Installation**
2. **Configuration Options (kwargs)**
3. **Basic Usage**
4. **PDB-Only Mode (No Multiwfn Required)**
5. **ESP Profile (2-D ESP Map)**

## 1. Installation

```bash
git clone git@github.com:hjkgrp/pyEF.git
cd pyEF
conda env create -f environment.yml
conda activate pyef
pip install -e .
```

### Installing Multiwfn

PyEF requires [Multiwfn](http://sobereva.com/multiwfn/) for charge partitioning and wavefunction analysis. 

NOTE: *currently Multiwfn is NOT supported on MACOSX, so we recommend using a linux or windows system.*

**Download and compile:**
```bash
# Download from http://sobereva.com/multiwfn/
wget http://sobereva.com/multiwfn/misc/Multiwfn_3.8_bin_Linux_noGUI.zip
unzip Multiwfn_3.8_bin_Linux_noGUI.zip 
cd Multiwfn_3.8_bin_Linux_noGUI

# Give the Multiwfn executable permission to run
chmod +x Multiwfn_noGUI
```

**Add to your PATH (add to ~/.bashrc):**
```bash
export PATH=/path/to/Multiwfn_3.8_bin_Linux_noGUI:$PATH
```

Or use the full path when running PyEF (e.g., `multiwfn_path: /path/to/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_noGUI`).

---

## 2. Configuration Options (kwargs)

When creating an `Electrostatics` object, you can pass keyword arguments to customize the calculation. All are optional.

### ECP (Effective Core Potential) Support

If your QM calculation used ECPs, set `hasECP=True` and specify the ECP family:

```python
es = Electrostatics(molden_paths, xyz_paths, hasECP=True, ECP="lacvps")
```

**Supported `ECP` values:**

| ECP Value | Description |
|-----------|-------------|
| `"lacvps"` | (default) Hybrid: all-electron for Z ≤ 18 (up to Ar), LANL2DZ for heavier elements |
| `"lacvp"` | Hybrid: all-electron for Z ≤ 10 (up to Ne), LANL2DZ for heavier elements |
| `"lanl2dz"` | LANL2DZ — widely used ECP for transition metals |
| `"def2"` | def2-type ECPs (e.g., def2-SVP, def2-TZVP families) |
| `"crenbl"` | CRENBL shape-consistent relativistic ECPs |
| `"stuttgart_rsc"` | Stuttgart Relativistic Small Core ECPs |

### All Configuration kwargs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `hasECP` | bool | `False` | Whether the QM calculation used ECPs |
| `ECP` | str | `"lacvps"` | ECP basis set family (see above). Only used when `hasECP=True` |
| `dielectric` | float | `1` | Dielectric constant (1=vacuum, 4≈protein, 78.5≈water) |
| `dielectric_scale` | float | `1` | Scaling factor for point charges in dielectric treatment |
| `changeDielectBoundBool` | bool | `False` | Use dielectric=1 for bonded atoms |
| `includePtChgs` | bool | `False` | Include QM/MM point charges in calculations |
| `rerun` | bool | `False` | Force recalculation of charges (ignore cache) |
| `maxIHirshBasis` | int | `12000` | Max basis functions for Hirshfeld-I analysis |
| `maxIHirshFuzzyBasis` | int | `6000` | Max basis functions for fuzzy Hirshfeld-I analysis |
| `excludeAtomfromEcalc` | list | `[]` | Atom indices to exclude from E-field calculations |
| `skip_missing_files` | bool | `False` | Skip jobs with missing files instead of raising errors |
| `visualize_ef` | bool | `False` | Create PDB files with atom-wise E-field data |
| `visualize_charges` | bool | `False` | Create PDB files with partial charges |
| `visualize_per_bond` | bool | `False` | Create separate PDB files per bond |

---

## 3. Basic Usage

### Electric Field Calculation

**CLI:**
```bash
# Create jobs.csv
echo "ef, /path/to/structure.molden, /path/to/structure.xyz, (25, 26)" > jobs.csv

# Create config.yaml
cat > config.yaml << EOF
input: jobs.csv
dielectric: 1
multiwfn_path: /path/to/multiwfn
charge_types:
  - Hirshfeld_I
EOF

# Run
pyef -c config.yaml
```

**Python API:**
```python
import numpy as np
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz']

# Create an Electrostatics object for the structures
es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

# Bonds (0-indexed) to project the E-field onto: one in structure 1, two in structure 2
ef_bonds = [[(0, 10)], [(20, 21), (25, 26)]]

# Substrate atoms in structure 1 and structure 2
sub_indices = [np.arange(0, 10), np.arange(10, 25)]

# Exclude the substrate from the sources: only the environment's field on the bond is measured
es.setExcludeAtomFromCalc(sub_indices)

# charge_types can be a list or a single value
df = es.getEfield(charge_types='Hirshfeld_I', Efielddata_filename='output', multiwfn_path='/path/to/multiwfn',
                  input_bond_indices=ef_bonds)
```

### Electrostatic Potential (ESP) Calculation

**CLI:**
```bash
# Create jobs.csv with metal atom index
echo "esp, /path/to/structure.molden, /path/to/structure.xyz, 30" > jobs.csv

# Run with same config.yaml
pyef -c config.yaml
```

**Python API:**
```python
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz']

# One atom index per structure (0-indexed): atom 30 in structure 1, atom 0 in structure 2
metal_indices = [30, 0]

es = Electrostatics(molden_paths, xyz_paths, esp_atom_idx=metal_indices, dielectric=1.0)

# ESP at those atoms for every structure, saved to espHirshfeldI.csv
df = es.getESP(charge_types=['Hirshfeld_I'], ESPdata_filename='espHirshfeldI', multiwfn_path='/path/to/multiwfn')
```

### Electrostatic Stabilization Calculation

**CLI:**
```bash
# Create jobs.csv
echo "estab, /path/to/structure.molden, /path/to/structure.xyz" > jobs.csv

# Create config.yaml with substrate indices
cat > config.yaml << EOF
input: jobs.csv
dielectric: 1
multiwfn_path: /path/to/multiwfn
charge_types:
  - Hirshfeld_I
multipole_order: 2
substrate_idxs: [1, 2, 3, 4, 5]
EOF

# Run
pyef -c config.yaml
```

**Python API:**
```python
import numpy as np
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz']

es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

# Substrate atoms in structure 1 and structure 2
sub_indices = [np.arange(0, 10), np.arange(10, 25)]

# multipole_order=2 needs Hirshfeld, Hirshfeld_I or Becke; other schemes fall back to 1
df = es.getElectrostatic_stabilization(charge_type='Hirshfeld', multiwfn_path='/path/to/multiwfn',
                                       substrate_idxs=sub_indices, multipole_order=2)
```

### Partial Charge Calculation

**Python API:**
```python
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure.molden']
xyz_paths = ['/path/to/structure.xyz']

es = Electrostatics(molden_paths, xyz_paths)

# Get multiple charge types with PDB output for visualization
df = es.getCharges(charge_types=['RESP', 'Hirshfeld_I'], multiwfn_path='/path/to/multiwfn',
                   output_filename='my_charges', write_pdb=True)
```

**Output:** Returns a DataFrame with columns: `Job`, `Charge_Type`, `Atom_Index`, `Element`, `x`, `y`, `z`, `Charge`, `Molden_Path`, `XYZ_Path`.

**Available charge types:**

| Charge Type | Description | Notes |
|-------------|-------------|-------|
| `Hirshfeld_I` | Iterative Hirshfeld | Recommended for most systems |
| `Hirshfeld` | Standard Hirshfeld partitioning | Fast, good for most systems |
| `RESP` | Restrained ESP fitting | Standard for force field development |
| `CHELPG` | ESP fitting (Breneman) | Good for molecular mechanics |
| `MK` | Merz-Kollmann ESP fitting | Alternative ESP method |
| `CM5` | Charge Model 5 | Good balance of accuracy/speed |
| `ADCH` | Atomic dipole corrected Hirshfeld | Recommended by Multiwfn |
| `Mulliken` | Mulliken population | Fast but basis-set dependent |
| `Lowdin` | Löwdin population | Orthogonalized basis |
| `Voronoi` | Voronoi deformation density | Space partitioning method |
| `SCPA` | Ros & Schuit modified Mulliken | Modified Mulliken scheme |
| `Becke` | Becke partitioning with dipole correction | Real-space integration |
| `EEM` | Electronegativity equalization | ⚠️ Requires bonded atoms (fails for ionic systems) |
| `PEOE` | Gasteiger charges | ⚠️ Missing parameters for some elements (e.g., Na, transition metals) |

**PDB visualization:** When `write_pdb=True`, creates PDB files with charges in the B-factor column. Visualize in PyMOL with:
```
spectrum b, blue_white_red, minimum=-1, maximum=1
```

---

## 4. PDB-Only Mode (No Multiwfn Required)

Skip the QM workflow entirely by placing your partial charges in the **B-factor column** of a PDB file. No `.molden`, `.xyz`, or Multiwfn needed. Monopole charges only.

**Electric Field:**
```python
import numpy as np
from pyef.analysis import Electrostatics

es = Electrostatics(pdb_charge_paths=['job1/charges.pdb', 'job2/charges.pdb'], dielectric=1.0)

sub_indices = [np.arange(0, 20), np.arange(0, 20)]
es.setExcludeAtomFromCalc(sub_indices)

df = es.getEfield(input_bond_indices=[[(25, 26)], [(25, 26)]], Efielddata_filename='ef_pdb')
```

**ESP:**
```python
es = Electrostatics(pdb_charge_paths=['job1/charges.pdb', 'job2/charges.pdb'],
                    esp_atom_idx=[30, 30], dielectric=4.0)

df = es.getESP(ESPdata_filename='esp_pdb')
```

**Electrostatic Stabilization:**
```python
es = Electrostatics(pdb_charge_paths=['job1/charges.pdb', 'job2/charges.pdb'], dielectric=1.0)

df = es.getElectrostatic_stabilization(substrate_idxs=[np.arange(0, 10), np.arange(0, 10)],
                                       name_dataStorage='estab_pdb')
```

`pdb_charge_paths` can also be passed directly to any of the three calculation functions on a standard (molden/xyz) object to override charges for that call only.

---

## 5. ESP Profile (2-D ESP Map)

Maps the ESP on a plane through three atoms and plots it. P1 → P2 is the vertical axis and P3 sets the plane; a range such as `2-34` stands for the centroid of those atoms (0-indexed). Several jobs are treated as frames of the same system and averaged into one map. Writes `<name>.svg`, `<name>.png` and `<name>.npz`.

**CLI:**
```bash
# Create jobs.csv: plane through atoms 1 and 0 and the centroid of atoms 2-34
echo "espprofile, /path/to/structure.molden, /path/to/structure.xyz, (1, 0, 2-34)" > jobs.csv

# Run with same config.yaml
pyef -c config.yaml
```

**Python API:**
```python
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure.molden']
xyz_paths = ['/path/to/structure.xyz']

es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

# Optional, QM/MM: include the point-charge file in each job's folder
# es.includePtChgs('pointcharges.txt')

# Plane through atoms 1 and 0 and the centroid of atoms 2-34; writes esp_profile.svg/.png/.npz
res = es.getESPProfile(charge_types=['Hirshfeld_I'], profile_filename='esp_profile',
                       multiwfn_path='/path/to/multiwfn', plane_points=(1, 0, '2-34'))
```

**Defaults:** the ESP of the cage alone (a guest is left out), colour limits taken from the cage cavity so the metals and walls saturate, and the atoms in the plane drawn as white sticks. Common options, as keywords to `getESPProfile` or as `profile_<option>` in `config.yaml`:

| Option | Example | Effect |
|--------|---------|--------|
| `esp_range` | `(-10, 10)` | Fixed colour scale in V |
| `conditions` | `{'Cage': 'cage', 'All': 'all'}` | Which charges to sum; `All` includes the guest |
| `cut` | `'horizontal'` | Plane perpendicular to P1 → P2 instead of through P3 |
| `mask_radius` | `1.6` | Blank cells within this distance (Å) of an atom |

If the `Charges<type>.txt` files already exist, `pyef espprofile --template > esp.yaml` then `pyef espprofile -c esp.yaml` makes the map without Multiwfn.

---

**Authors:** Melissa Manetsch and David W. Kastner
