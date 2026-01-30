Getting Started
===============

Installation
------------

.. code-block:: bash

   git clone git@github.com:davidkastner/pyEF.git
   cd pyEF
   ./install.sh

This creates a conda environment with all dependencies and installs PyEF.

Requirements
~~~~~~~~~~~~

- Python 3.8+
- `Multiwfn <http://sobereva.com/multiwfn/>`_ (for charge calculations)

Core Concepts
-------------

PyEF requires two input files per structure:

- **Molden file** (``.molden``): Contains wavefunction data from QM calculations
- **XYZ file** (``.xyz``): Contains atomic coordinates

All atom indices in PyEF are **0-indexed**.

Supported Charge Schemes
~~~~~~~~~~~~~~~~~~~~~~~~

- **Hirshfeld_I** (recommended): Iterative Hirshfeld, most accurate
- **Hirshfeld**: Standard Hirshfeld partitioning
- **CHELPG**: Fitted to electrostatic potential
- **Mulliken**: Fast but basis-set dependent
- **Other**: Becke, Lowdin, ADCH, MK, AIM, CM5, RESP, PEOE

Computing Partial Charges
-------------------------

Partial charges are computed automatically when calculating electric fields or ESP.
The charges are obtained via Multiwfn using your specified charge scheme.

.. code-block:: python

   from pyef.analysis import Electrostatics

   es = Electrostatics(['structure.molden'], ['structure.xyz'])

   # Charges are computed as part of E-field or ESP calculations
   # and stored internally in the Electrostatics object

Computing Electric Fields
-------------------------

Calculate electric fields at specific bonds.

Python API
~~~~~~~~~~

.. code-block:: python

   from pyef.analysis import Electrostatics

   # Initialize with molden and xyz files
   es = Electrostatics(
       molden_paths=['job1/optim.molden', 'job2/optim.molden'],
       xyz_paths=['job1/optim.xyz', 'job2/optim.xyz'],
       dielectric=4.0  # optional: protein dielectric
   )

   # Calculate E-field at bonds (atom indices are 0-indexed)
   df = es.getEfield(
       charge_types='Hirshfeld_I',
       Efielddata_filename='efield_results',
       multiwfn_path='/path/to/multiwfn',
       input_bond_indices=[(25, 26), (25, 27)]  # bonds to analyze
   )

**Key parameters:**

- ``charge_types``: Charge scheme ('Hirshfeld_I', 'CHELPG', etc.)
- ``input_bond_indices``: List of (atom1, atom2) tuples defining bonds
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

   # Initialize with metal center index (0-indexed)
   es = Electrostatics(
       molden_paths=['optim.molden'],
       xyz_paths=['optim.xyz'],
       lst_of_tmcm_idx=[30]  # metal atom index
   )

   # Calculate ESP
   df = es.getESP(
       charge_types='Hirshfeld_I',
       ESPdata_filename='esp_results',
       multiwfn_path='/path/to/multiwfn'
   )

**Key parameters:**

- ``lst_of_tmcm_idx``: Metal atom indices where ESP is calculated (set at initialization)
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
