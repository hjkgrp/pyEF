Getting Started
===============

**Jump to:** `Installation`_ · `Configuration kwargs`_ · `getCharges()`_ · `getEfield()`_ · `getESP()`_ · `getElectrostatic_stabilization()`_

.. _Configuration kwargs: `Initialization Configuration (kwargs)`_
.. _getCharges(): `Computing Partial Charges`_
.. _getEfield(): `Computing Electric Fields`_
.. _getESP(): `Computing Electrostatic Potentials`_
.. _getElectrostatic_stabilization(): `Computing Electrostatic Stabilization`_

Installation
------------

.. code-block:: bash

   git clone git@github.com:hjkgrp/pyEF.git
   cd pyEF
   conda env create -f environment.yml
   conda activate pyef
   pip install -e .

Requirements
~~~~~~~~~~~~

- Python 3.8+
- `Multiwfn <http://sobereva.com/multiwfn/>`_ (for charge calculations)

Installing Multiwfn
~~~~~~~~~~~~~~~~~~~

PyEF requires `Multiwfn <http://sobereva.com/multiwfn/>`_ for charge partitioning and wavefunction analysis.

.. note::
   Currently Multiwfn is NOT supported on macOS. We recommend using a Linux or Windows system.

**Download and compile:**

.. code-block:: bash

   # Download from http://sobereva.com/multiwfn/
   wget http://sobereva.com/multiwfn/misc/Multiwfn_3.8_bin_Linux_noGUI.zip
   unzip Multiwfn_3.8_bin_Linux_noGUI.zip
   cd Multiwfn_3.8_bin_Linux_noGUI

   # Give executable permission to the Multiwfn executable file
   chmod +x Multiwfn_noGUI

**Add to your PATH (add to ~/.bashrc):**

.. code-block:: bash

   export PATH=/path/to/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_noGUI:$PATH

Or use the full path when running PyEF (e.g., ``multiwfn_path: /path/to/Multiwfn_3.8_bin_Linux_noGUI/Multiwfn_noGUI``).

Core Concepts
-------------

PyEF requires two input files per structure:

- **Molden file** (``.molden``): Contains wavefunction data from QM calculations
- **XYZ file** (``.xyz``): Contains atomic coordinates

All atom indices in PyEF are **0-indexed**.

Supported Charge Schemes
~~~~~~~~~~~~~~~~~~~~~~~~

**Recommended:**

- **Hirshfeld_I**: Iterative Hirshfeld - most accurate for most systems
- **ADCH**: Atomic dipole corrected Hirshfeld - recommended by Multiwfn
- **CM5**: Charge Model 5 - good balance of accuracy and speed

**ESP-based methods:**

- **RESP**: Restrained ESP fitting - standard for force field development
- **CHELPG**: ESP fitting (Breneman) - good for molecular mechanics
- **MK**: Merz-Kollmann ESP fitting - alternative ESP method

**Population analysis:**

- **Hirshfeld**: Standard Hirshfeld partitioning - fast, good for most systems
- **Mulliken**: Mulliken population - fast but basis-set dependent
- **Lowdin**: Löwdin population - uses orthogonalized basis
- **Voronoi**: Voronoi deformation density - space partitioning method
- **SCPA**: Ros & Schuit modified Mulliken scheme
- **Becke**: Becke partitioning with atomic dipole correction

**Methods with limitations:**

- **EEM**: Electronegativity equalization method

  - ⚠️ **Limitation**: Requires all atoms to be bonded. Fails for ionic systems
    (e.g., systems with Na+, K+, or other isolated ions)

- **PEOE**: Gasteiger (PEOE) charges

  - ⚠️ **Limitation**: Missing parameters for some elements including Na, K,
    and most transition metals. Will fail if your system contains unsupported elements

.. _initialization-kwargs:

Initialization Configuration (kwargs)
--------------------------------------

When creating an ``Electrostatics`` object, you can pass keyword arguments to configure the
calculation behavior. All kwargs are optional.

**ECP (Effective Core Potential) support:**

If your QM calculation used ECPs, you must tell pyEF so it can fix molden file artifacts:

.. code-block:: python

   # For a TeraChem calculation with LACVPS (default ECP)
   es = Electrostatics(molden_paths, xyz_paths, hasECP=True, ECP="lacvps")

   # For a calculation with Stuttgart RSC ECPs
   es = Electrostatics(molden_paths, xyz_paths, hasECP=True, ECP="stuttgart_rsc")

Supported ``ECP`` values:

- ``"lacvps"`` (default) -- Hybrid: all-electron for Z |le| 18, LANL2DZ for heavier elements
- ``"lacvp"`` -- Hybrid: all-electron for Z |le| 10, LANL2DZ for heavier elements
- ``"lanl2dz"`` -- LANL2DZ ECP, widely used for transition metals
- ``"def2"`` -- def2-type ECPs (e.g., def2-SVP, def2-TZVP families)
- ``"crenbl"`` -- CRENBL shape-consistent relativistic ECPs
- ``"stuttgart_rsc"`` -- Stuttgart Relativistic Small Core ECPs

.. |le| unicode:: U+2264

**Dielectric environment:**

.. code-block:: python

   # Protein interior (dielectric ~ 4)
   es = Electrostatics(molden_paths, xyz_paths, dielectric=4.0)

   # Water solvent (dielectric ~ 78.5)
   es = Electrostatics(molden_paths, xyz_paths, dielectric=78.5)

   # Use dielectric=1 for bonded atoms (special boundary treatment)
   es = Electrostatics(molden_paths, xyz_paths, dielectric=4.0, changeDielectBoundBool=True)

**QM/MM point charges:**

.. code-block:: python

   # New API: per-job point charge paths
   es = Electrostatics(molden_paths, xyz_paths,
                        ptchg_paths=['/path/to/ptchg1.txt', '/path/to/ptchg2.txt'],
                        includePtChgs=True)

**Computation options:**

.. code-block:: python

   # Force recalculation (ignore cached charges)
   es = Electrostatics(molden_paths, xyz_paths, rerun=True)

   # Skip missing files in batch processing
   es = Electrostatics(molden_paths, xyz_paths, skip_missing_files=True)

   # Increase basis function limit for very large systems
   es = Electrostatics(molden_paths, xyz_paths, maxIHirshBasis=20000)

**Visualization:**

.. code-block:: python

   # Enable PDB output for E-fields and charges
   es = Electrostatics(molden_paths, xyz_paths,
                        visualize_ef=True, visualize_charges=True)

For the complete list of all kwargs and their defaults, see the :ref:`API Reference <configuration-kwargs>`.

Computing Partial Charges
-------------------------

Compute partial charges for all atoms using specified charge partitioning scheme(s).
Charges can optionally be written to PDB files for visualization.

Python API
~~~~~~~~~~

.. code-block:: python

   from pyef.analysis import Electrostatics

   molden_paths = ['/path/to/structure.molden']
   xyz_paths = ['/path/to/structure.xyz']

   es = Electrostatics(molden_paths, xyz_paths)

   # Get multiple charge types with PDB output for visualization
   df = es.getCharges(
       charge_types=['RESP', 'Hirshfeld_I'],
       multiwfn_path='/path/to/multiwfn',
       output_filename='my_charges',
       write_pdb=True
   )

**Output:** Returns a DataFrame with columns: ``Job``, ``Charge_Type``, ``Atom_Index``,
``Element``, ``x``, ``y``, ``z``, ``Charge``, ``Molden_Path``, ``XYZ_Path``.

**PDB Visualization:** When ``write_pdb=True``, creates PDB files with charges in the
B-factor column. Visualize in PyMOL with:

.. code-block:: text

   spectrum b, blue_white_red, minimum=-1, maximum=1

Computing Electric Fields
-------------------------

Calculate electric fields at specific bonds.

Python API
~~~~~~~~~~

.. code-block:: python

   from pyef.analysis import Electrostatics
   import numpy as np

   molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
   xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz']

   # Create an electrostatics object
   es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

   # Record the indices (0-indexed) for the bonds to project E-field across
   # Here: one bond (0, 10) in structure 1 and two bonds (20, 21), (25, 26) in structure 2
   ef_bond = [[(0, 10)], [(20, 21), (25, 26)]]

   # Record indices of the substrate containing bonds of interest
   sub_indices = [np.arange(0, 10), np.arange(10, 25)]

   # Exclude substrates from E-field calc (only probe environment impact, not intramolecular)
   es.setExcludeAtomFromCalc(sub_indices)

   # Calculate E-field (charge_types can be a list or single value)
   df = es.getEfield(
       charge_types='Hirshfeld_I',
       Efielddata_filename='output',
       multiwfn_path='/path/to/multiwfn',
       input_bond_indices=ef_bond
   )

**Key parameters:**

- ``charge_types``: Charge scheme ('Hirshfeld_I', 'CHELPG', etc.) - can be string or list
- ``input_bond_indices``: List of (atom1, atom2) tuples defining bonds per structure
- ``multipole_bool``: Use multipole expansion (default: False)
- ``dielectric``: Dielectric constant (1=vacuum, 4=protein, 78.5=water)

CLI
~~~

Create a jobs file (``jobs.csv``):

.. code-block:: text

   ef, /path/to/optim.molden, /path/to/optim.xyz, (25, 26), (25, 27)

Create a config file (``config.yaml``):

.. code-block:: yaml

   input: jobs.csv
   multiwfn_path: /path/to/multiwfn
   charge_types:
     - Hirshfeld_I
   dielectric: 1

Run:

.. code-block:: bash

   pyef -c config.yaml

Computing Electrostatic Potentials
----------------------------------

Calculate ESP at metal centers or specific atomic sites.

Python API
~~~~~~~~~~

.. code-block:: python

   from pyef.analysis import Electrostatics

   molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
   xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz']

   # One atom index per file (0-indexed)
   metal_indices = [30, 0]  # atom 30 for structure1, atom 0 for structure2

   # Create an electrostatics object with esp_atom_idx
   es = Electrostatics(molden_paths, xyz_paths, esp_atom_idx=metal_indices, dielectric=1.0)

   # Run electrostatic potential calculations
   df = es.getESP(
       charge_types=['Hirshfeld_I'],
       ESPdata_filename='esp_results',
       multiwfn_path='/path/to/multiwfn'
   )

**Key parameters:**

- ``esp_atom_idx``: Atom indices where ESP is calculated (set at initialization, 0-indexed)
- ``charge_types``: Charge scheme
- ``use_multipole``: Use multipole expansion (default: False)
- ``dielectric``: Dielectric constant

CLI
~~~

Create a jobs file:

.. code-block:: text

   esp, /path/to/optim.molden, /path/to/optim.xyz, 30

Run with config:

.. code-block:: bash

   pyef -c config.yaml

Computing Electrostatic Stabilization
-------------------------------------

Calculate electrostatic stabilization energy between substrate and environment.

CLI
~~~

Create a jobs file:

.. code-block:: text

   estab, /path/to/structure.molden, /path/to/structure.xyz

Create a config file (``config.yaml``):

.. code-block:: yaml

   input: jobs.csv
   dielectric: 1
   multiwfn_path: /path/to/multiwfn
   charge_types:
     - Hirshfeld_I
   multipole_order: 2
   substrate_idxs: [1, 2, 3, 4, 5]

Run:

.. code-block:: bash

   pyef -c config.yaml

Python API
~~~~~~~~~~

.. code-block:: python

   from pyef.analysis import Electrostatics
   import numpy as np

   molden_paths = ['/path/to/structure1.molden', '/path/to/structure2.molden']
   xyz_paths = ['/path/to/structure1.xyz', '/path/to/structure2.xyz']

   es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

   # Record indices of the substrate in structure 1 and structure 2
   sub_indices = [np.arange(0, 10), np.arange(10, 25)]

   # Only Hirshfeld, Hirshfeld_I, and Becke are compatible with multipole_order=2
   # Otherwise will default to multipole_order=1
   df = es.getElectrostatic_stabilization(
       charge_types='Hirshfeld',
       multiwfn_path='/path/to/multiwfn',
       substrate_idxs=sub_indices,
       multipole_order=2
   )

**Key parameters:**

- ``substrate_idxs``: List of atom indices for the substrate (one per structure)
- ``charge_types``: Charge scheme (Hirshfeld, Hirshfeld_I, or Becke for multipole_order=2)
- ``multipole_order``: 1 for monopole, 2 for dipole corrections
- ``dielectric``: Dielectric constant

Dielectric Constants
--------------------

Common values:

- **1.0**: Vacuum
- **2-4**: Protein interior
- **20-40**: Protein-solvent interface
- **78.5**: Water

Output Files
------------

Results are saved as CSV files with the specified filename prefix:

- ``*_Efielddata.csv``: Electric field results
- ``*_ESPdata.csv``: Electrostatic potential results

Package Structure
-----------------

.. code-block:: text

   pyef/
   ├── analysis.py      # Main Electrostatics class
   ├── cli.py           # Command-line interface
   ├── geometry.py      # Geometry utilities
   ├── utility.py       # Helper functions
   └── multiwfn_interface.py  # Multiwfn integration
