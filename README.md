# pyEF

A Python package for calculating electric fields, electrostatic potentials (ESP), and electrostatic interactions from quantum mechanical calculations.

For more details on the theory behind this pacakge, best practices for using it in your reserach, please see our open source paper:

Melissa T. Manetsch, David W. Kastner, Yuriy Román-Leshkov, Heather J. Kulik; pyEF: A Python Framework for QM and QM/MM Atom-Wise Electric Field Analysis. J. Chem. Theory Comput. 26 May 2026; 22 (10): 4982–4994. https://doi.org/10.1021/acs.jctc.6c00065

(we also request that you cite the above paper if you find this package helpful in your research!)

Note: This package is under active development as of 2026, if you encounter any issues during use or have any features you would like to see incorporated, please do not hesitate to raise an issue or reach out to the developers at manets12@mit.edu. 

## Table of Contents
1. **Installation**
2. **Configuration Options (kwargs)**
3. **Basic Usage**
4. **PDB-Only Mode (No Multiwfn Required)**
5. **ESP Profile on a Plane Through a Cage**

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

`pdb_charge_paths` can also be passed directly to any of the three calculation functions on a standard (molden/xyz) object to override charges for that call onl## 5. ESP Profile on a Plane Through a Cage

Map the electrostatic potential on a plane through a cage (or any molecule)
and plot it. Give one frame for a single-frame map, or many frames (e.g. QM/MM
snapshots) to average them. Each frame gets its own plane before averaging, so
the map follows the molecule.

It reads the files pyEF already makes: `Charges<type>.txt` (`sym x y z q`) and,
optionally, a QM/MM `pointcharges.txt` (2 header lines, then `q x y z`). If the
charges already exist, Multiwfn is not needed.

**Automatic: one input file, only the frames are required.**
```bash
pyef espprofile --template > esp.yaml     # commented starter file (also demo/esp_profile.yaml)
pyef espprofile -c esp.yaml               # or: python -m pyef.esp_profile -c esp.yaml
```
```yaml
charges: frames/*/ChargesCM5.txt    # one file, a list, or a glob; 1 file = single frame, >1 = averaged
pointcharges: pointcharges.txt      # optional QM/MM charges, looked up next to each charge file
cut: vertical                       # or horizontal
esp_range: cavity                   # colour limits from the cage cavity (default); or [vmin, vmax] in V
```
The first frame's geometry is used to find:
- **the metals:** the two d/f-block atoms;
- **the ligands:** fragments bonded to a metal;
- **the cage:** the metals plus their ligands;
- **the guest:** everything else in the QM region.

The run prints what it found, so you can check it. From this it sets:
- **`plane: auto`:** the metal–metal axis plus the first ligand's centroid.
- **`cut: vertical`:** the plane containing the M–M axis, with the horizontal axis toward ligand 1.
- **`cut: horizontal`:** the plane perpendicular to the M–M axis through its midpoint, i.e. the equatorial slice. `offset: 2.0` slides either plane 2 Å along its normal.
- **`extent: auto`:** a grid that holds the whole cage plus 2 Å (`extent_pad`).
- **Default conditions:** `Cage` and `Cage_Env` (+ all point charges). The guest is not a source and is not drawn; add `All: all` to include it.
- **No masking by default:** the colour limits come from the cage cavity (`esp_range: cavity`), so the metals, the walls and the nuclei saturate instead of stretching the scale. Charges closer than `r_min` (0.5 Å) to a grid point count as `r_min` away, so each nucleus saturates rather than leaving a hole.
- **Plot:** the structure within 1.5 Å of the plane is drawn as white sticks, metals as white circles.

**ESP:** `ESP (V) = 14.3996 / dielectric * Σ q/r`. Pairs closer than `r_min`
(0.5 Å) are skipped.

**Manual control:** set any of these yourself:
- **`plane: [1, 0, 2-34]`:** P1, P2, P3 as 0-based atom indices, or groups whose centroid is used. The axis runs P1 → P2, and the horizontal axis points toward P3. This works for any molecule, not only cages.
- **`origin`:** `centroid`, `midpoint`, an atom index or `[x, y, z]`.
- **`extent`, `spacing`:** a number, or `[z, x]`.
- **`conditions`**, written as tokens: `cage`, `guest`, `metals`, `qm`, `mm`, `all`, `qm:a-b`, `mm:a-b`, `mm:a-` (inclusive, 0-based). Listing the same charges twice in one condition is an error. Example: `Cage_Anion: cage,mm:0-275`, when the first 276 MM charges are the counterions.
- **Masks:**
  - Off by default. `mask_radius`: e.g. 1.6 Å around the mask atoms (default: the cage, so a guest does not blank the cavity), or `vdw` for Bondi radii × `vdw_scale`, or per group, e.g. `{metals: 2.0, '2-133': 1.6}`.
  - `mask_atoms`, `hull_atoms`, `mask_frac`: a cell is blanked when it's flagged in at least `mask_frac` of the frames.
  - Masks are stored next to the map and never applied to the data.
- **No charges yet:** use `molden: frames/*/*.molden`, `charge_type` and `multiwfn_path` in place of `charges:`. Multiwfn then generates any missing `Charges<type>.txt`.

**Python API (to add to your own script):**
```python
from pyef import esp_profile as ep

files = ['f1/ChargesCM5.txt', 'f2/ChargesCM5.txt']                 # or one file
res = ep.compute_esp_profile(files, ptchg_files=['f1/pointcharges.txt', 'f2/pointcharges.txt'],
                             cut='horizontal')                      # plane, extent, conditions: auto
ep.save_esp_profile('esp_profile.npz', res, name='mysystem')
ep.plot_esp_profile(res, out='esp_profile')                         # esp_profile.svg / .png
cavity = ep.region_mean(res, 'Cage', z_lim=4.0, x_lim=4.0)          # mean over unmasked |z|,|x| <= 4 A

# explicit plane through three atoms / centroids, with your own source sets
res = ep.compute_esp_profile(files, plane_points=(1, 0, '2-34'), origin='midpoint',
                             conditions={'CageOnly': 'qm:0-133', 'Cage_Solv': 'qm:0-133,mm'})
```
From an `Electrostatics` object, the charges are generated if they are missing:
```python
es = Electrostatics(molden_paths, xyz_paths)
es.includePtChgs('pointcharges.txt')                                # optional, QM/MM
res = es.getESPProfile(charge_types=['Hirshfeld_I'], profile_filename='esp_profile',
                       multiwfn_path=multiwfn_path, plane_points=(1, 0, '2-41'),
                       formats=('svg', 'png'))                # esp_profile.svg/.png/.npz
```
In a `pyef -c config.yaml` jobs file, use `espprofile, job.molden, job.xyz` for
the automatic plane, or add `(1, 0, 2-34)` to give it yourself. Jobs that share
a plane are averaged. Set the options with `profile_cut`, `profile_offset`,
`profile_extent`, `profile_spacing`, `profile_conditions`, `profile_origin`,
`profile_mask_radius`, `profile_mask_atoms`, `profile_hull_atoms`,
`profile_r_min` and `profile_esp_range`.

**Output:** `<output>.npz` holds:
- `<name>_mean_<cond>` and `<name>_std_<cond>`: V, unmasked.
- `<name>_mask`, `<name>_z_vals` / `_x_vals`.
- `<name>_cut`, `<name>_plane_spec`.
- `<name>_cage_atoms` / `_guest_atoms`.
- `<name>_mean_dist`: \|P2 − P1\|.
- the frame-averaged in-plane coordinates of the three points and of every QM atom.

Load it with `ep.load_esp_profile`, and blank the masked cells with `ep.masked(res, cond)`.

`.

---

**Authors:** Melissa Manetsch and David W. Kastner
