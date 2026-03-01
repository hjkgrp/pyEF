# pyEF

A Python package for calculating electric fields, electrostatic potentials (ESP), and electrostatic interactions from quantum mechanical calculations.

Note: This package is under active development as of 2026, if you encounter any issues during use or have any features you would like to see incorporated, please do not hesitate to raise an issue or reach out to the developers at manets12@mit.edu. 

## Table of Contents
1. **Installation**
2. **Configuration Options (kwargs)**
3. **Basic Usage**
4. **PDB-Only Mode (No Multiwfn Required)**

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

#give executable permission to the Multiwfn executable file
chmod +x Multiwfn_noGUI
```

**Add to your PATH (add to ~/.bashrc):**
```bash
export PATH=/path/to/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_noGUI:$PATH
```

Or use the full path when running PyEF (e.g., `multiwfn_path: /path/to/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_noGUI`).

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
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz' ]

#create an electrostatics object associated with the file structure
es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

#Record the indices (0-index) for the bonds to project Efield across. Here I have one bond: (0, 10) in structure 1 and two bonds: (20, 21) and (25, 26) in structure 2
ef_bond = [[(0, 10)], [(20, 21), (25, 26)]

#record indices of the substrate containing bonds of interest in structure 1, and structure 2
sub_indices = [ np.arange(0,10), np.arange(10,25)]

#exclude substrates from Efield calc (here we are only probing impact of environmnet on bond, NOT intramolecular polarization)
es.setExcludeAtomFromCalc(sub_indices)

#charge_types can be a list or single value like here
df = es.getEfield(charge_types='Hirshfeld_I', Efielddata_filename='output', multiwfn_path='/path/to/multiwfn',
                  input_bond_indices=sub_indices)
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
xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz' ]

# One atom index per file (0-indexed)
metal_indices = [30, 0]  # atom 30 for structure1, atom 0 for structure2

#name for csv data file that will record ESP data
fn='espHirshfeldI'

#pathway to the installation of the multiwfn install
multiwfn_install = '/path/to/multiwfn'

#create an electrostatics object
es = Electrostatics(molden_paths, xyz_paths, esp_atom_idx=metal_indices, dielectric=1.0)

#run electrostatic potential calculations on  all files in list
df = es.getESP(charge_types=['Hirshfeld_I'], ESPdata_filename=fn, multiwfn_path=multiwfn_install)
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
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz' ]

es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

#record indices of the substrate containins bonds of interest in structure 1, and structure 2
sub_indices = [ np.arange(0,10), np.arange(10,25)]

#only Hirshfeld, Hirshfeld_I, and Becke are compatible with multipole_order=2. Otherwise will default to multipole_order=1
df = es.getElectrostatic_stabilization(charge_types='Hirshfeld', multiwfn_path='/path/to/multiwfn', substrate_idxs=sub_indices, multipole_order=2)
```

### Partial Charge Calculation

**Python API:**
```python
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure.molden']
xyz_paths = ['/path/to/structure.xyz']

es = Electrostatics(molden_paths, xyz_paths)

# Get multiple charge types with PDB output for visualization
df = es.getCharges(charge_types =['RESP', 'Hirshfeld_I'], multiwfn_path='/path/to/multiwfn',
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
| `PEOE` | Gasteiger charges | ⚠️ Missing parameters for some elements (e.g., Na, transition metals)

**PDB visualization:** When `write_pdb=True`, creates PDB files with charges in the B-factor column. Visualize in PyMOL with:
```
spectrum b, blue_white_red, minimum=-1, maximum=1
```

---

## 4. PDB-Only Mode (No Multiwfn Required)

Skip the QM workflow entirely by placing your partial charges in the **B-factor column** of a PDB file. No `.molden`, `.xyz`, or Multiwfn needed. Monopole charges only.

**Electric Field:**
```python
from pyef.analysis import Electrostatics
import numpy as np

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

**Authors:** Melissa Manetsch and David W. Kastner
