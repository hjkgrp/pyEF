# pyEF

A Python package for calculating electric fields, electrostatic potentials (ESP), and electrostatic interactions from quantum mechanical calculations.

Note: This package is under active development as of 2026, if you encounter any issues during use or have any features you would like to see incorporated, please do not hesitate to raise an issue or reach out to the developers at manets12@mit.edu. 

## Table of Contents
1. **Installation**
2. **Basic Usage**

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

## 2. Basic Usage

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

molden_paths = ['/path/to/structure.molden']
xyz_paths = ['/path/to/structure.xyz']

es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)
df = es.getEfield('Hirshfeld_I', 'output', '/path/to/multiwfn',
                  input_bond_indices=[(25, 26)])
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

molden_paths = ['/path/to/structure.molden']
xyz_paths = ['/path/to/structure.xyz']
metal_indices = [30]  # 0-indexed

es = Electrostatics(molden_paths, xyz_paths, lst_of_tmcm_idx=metal_indices, dielectric=1.0)
df = es.getESP(['Hirshfeld_I'], 'output', 'multiwfn', '/path/to/multiwfn')
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

molden_paths = ['/path/to/structure.molden']
xyz_paths = ['/path/to/structure.xyz']

es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)
df = es.getElectrostatic_stabilization('/path/to/multiwfn',
                                        substrate_idxs=[1, 2, 3, 4, 5],
                                        multipole_order=2)
```

### Partial Charge Calculation

**Python API:**
```python
from pyef.analysis import Electrostatics

molden_paths = ['/path/to/structure.molden']
xyz_paths = ['/path/to/structure.xyz']

es = Electrostatics(molden_paths, xyz_paths)

# Get RESP charges for all atoms
df = es.getCharges('RESP', '/path/to/multiwfn')

# Get multiple charge types with PDB output for visualization
df = es.getCharges(['RESP', 'Hirshfeld_I'], '/path/to/multiwfn',
                   output_filename='my_charges', write_pdb=True)
```

**Output:** Returns a DataFrame with columns: `Job`, `Charge_Type`, `Atom_Index`, `Element`, `x`, `y`, `z`, `Charge`, `Molden_Path`, `XYZ_Path`.

**Available charge types:** `RESP`, `Hirshfeld`, `Hirshfeld_I`, `CHELPG`, `Mulliken`, `Lowdin`, `SCPA`, `Becke`, `ADCH`, `Voronoi`, `MK`, `AIM`, `CM5`, `EEM`, `PEOE`

**PDB visualization:** When `write_pdb=True`, creates PDB files with charges in the B-factor column. Visualize in PyMOL with:
```
spectrum b, blue_white_red, minimum=-1, maximum=1
```

---

**Authors:** Melissa Manetsch and David W. Kastner
