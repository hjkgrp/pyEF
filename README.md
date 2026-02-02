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

molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz' ]

#create an electrostatics object associated with the file structure
es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

#Record the indices (0-index) for the bonds to project Efield across. Here I have one bond: (0, 10) in structure 1 and two bonds: (20, 21) and (25, 26) in structure 2
ef_bond = [[(0, 10)], [(20, 21), (25, 26)]

#record indices of the substrate containins bonds of interest in structure 1, and structure 2
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

**Authors:** Melissa Manetsch and David W. Kastner
