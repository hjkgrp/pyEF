API Reference
=============

This page documents the main classes and functions for computing electric fields,
electrostatic potentials, and partial charges.

.. _quick-reference:

Quick Reference
---------------

.. list-table::
   :widths: 20 35 25 20
   :header-rows: 1

   * - Function
     - Key Inputs
     - Output
     - Requirements
   * - :ref:`getEfield() <getefield-electric-field>`
     - ``charge_types``, ``input_bond_indices``, ``multiwfn_path``
     - DataFrame with E-field (V/Å)
     - Molden + XYZ files
   * - :ref:`getESP() <getesp-electrostatic-potential>`
     - ``charge_types``, ``multiwfn_path``
     - DataFrame with ESP (V)
     - ``esp_atom_idx`` at init
   * - :ref:`getCharges() <getcharges-partial-charges>`
     - ``charge_types``, ``multiwfn_path``, ``write_pdb``
     - DataFrame with charges
     - Molden + XYZ files
   * - :ref:`getElectrostatic_stabilization() <getelectrostatic-stabilization-electrostatic-stabilization>`
     - ``substrate_idxs``, ``charge_type``, ``multipole_order``
     - DataFrame with energies (kcal/mol)
     - Molden + XYZ files

Electrostatics Class
--------------------

The ``Electrostatics`` class in ``pyef.analysis`` is the main interface for all calculations.

Initialization
~~~~~~~~~~~~~~

.. code-block:: python

   from pyef.analysis import Electrostatics

   es = Electrostatics(
       molden_paths,      # List of .molden file paths
       xyz_paths,         # List of .xyz file paths
       esp_atom_idx=None, # Atom indices for ESP (0-indexed)
       ptchg_paths=None   # Point charge files for QM/MM
   )

**Required parameters:**

- ``molden_paths`` (list of str): Paths to molden files, one per structure
- ``xyz_paths`` (list of str): Paths to XYZ files, one per structure

**Optional parameters:**

- ``esp_atom_idx`` (list of int): Atom indices for ESP calculations (0-indexed)
- ``ptchg_paths`` (list): Point charge file paths for QM/MM calculations

.. _getefield-electric-field:

getEfield() - Electric Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = es.getEfield(
       charge_types,           # 'Hirshfeld_I', 'CHELPG', etc.
       Efielddata_filename,    # Output filename prefix
       multiwfn_path,          # Path to Multiwfn executable
       input_bond_indices=[],  # [(atom1, atom2), ...]
       multipole_bool=False,   # Use multipole expansion
       dielectric=1            # Dielectric constant
   )

**Parameters:**

- ``charge_types`` (str or list): Charge scheme(s) to use
- ``Efielddata_filename`` (str): Output CSV filename prefix
- ``multiwfn_path`` (str): Path to Multiwfn executable
- ``input_bond_indices`` (list of tuples): Bond pairs as (atom1, atom2)
- ``multipole_bool`` (bool): True for multipole expansion, False for monopole
- ``dielectric`` (float): Dielectric constant

**Returns:** DataFrame with electric field results

.. _getesp-electrostatic-potential:

getESP() - Electrostatic Potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = es.getESP(
       charge_types,        # 'Hirshfeld_I', 'CHELPG', etc.
       ESPdata_filename,    # Output filename prefix
       multiwfn_path,       # Path to Multiwfn executable
       use_multipole=False, # Use multipole expansion
       dielectric=1         # Dielectric constant
   )

**Parameters:**

- ``charge_types`` (str or list): Charge scheme(s) to use
- ``ESPdata_filename`` (str): Output CSV filename prefix
- ``multiwfn_path`` (str): Path to Multiwfn executable
- ``use_multipole`` (bool): True for multipole expansion
- ``dielectric`` (float): Dielectric constant

**Returns:** DataFrame with ESP results

**Note:** Requires ``esp_atom_idx`` to be set during initialization.

.. _getcharges-partial-charges:

getCharges() - Partial Charges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = es.getCharges(
       charge_types='Hirshfeld_I',  # Charge scheme(s)
       multiwfn_path='/path/to/multiwfn',
       output_filename='charges',   # Output filename prefix
       write_pdb=False              # Write PDB files with charges
   )

**Parameters:**

- ``charge_types`` (str or list): Charge scheme(s) to use
- ``multiwfn_path`` (str): Path to Multiwfn executable
- ``output_filename`` (str): Output CSV filename prefix
- ``write_pdb`` (bool): If True, write PDB files with charges in B-factor column

**Returns:** DataFrame with columns: Job, Charge_Type, Atom_Index, Element, x, y, z, Charge, Molden_Path, XYZ_Path

.. _getelectrostatic-stabilization-electrostatic-stabilization:

getElectrostatic_stabilization() - Electrostatic Stabilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = es.getElectrostatic_stabilization(
       multiwfn_path='/path/to/multiwfn',
       substrate_idxs=[],           # Substrate atom indices per structure
       charge_type='Hirshfeld_I',   # Charge scheme
       multipole_order=2,           # 1 for monopole, 2 for dipole
       dielectric=1                 # Dielectric constant
   )

**Parameters:**

- ``multiwfn_path`` (str): Path to Multiwfn executable
- ``substrate_idxs`` (list): List of atom indices for substrate per structure
- ``charge_type`` (str): Charge scheme (Hirshfeld, Hirshfeld_I, or Becke for multipole_order=2)
- ``multipole_order`` (int): 1 for monopole, 2 for dipole corrections
- ``dielectric`` (float): Dielectric constant

**Returns:** DataFrame with electrostatic stabilization energies

Charge Schemes
--------------

**Available schemes:**

+---------------+------------------------------------------+--------------------------------------------------+
| Charge Type   | Description                              | Notes                                            |
+===============+==========================================+==================================================+
| Hirshfeld_I   | Iterative Hirshfeld                      | Recommended for most systems                     |
+---------------+------------------------------------------+--------------------------------------------------+
| Hirshfeld     | Standard Hirshfeld partitioning          | Fast, good for most systems                      |
+---------------+------------------------------------------+--------------------------------------------------+
| RESP          | Restrained ESP fitting                   | Standard for force field development             |
+---------------+------------------------------------------+--------------------------------------------------+
| CHELPG        | ESP fitting (Breneman)                   | Good for molecular mechanics                     |
+---------------+------------------------------------------+--------------------------------------------------+
| MK            | Merz-Kollmann ESP fitting                | Alternative ESP method                           |
+---------------+------------------------------------------+--------------------------------------------------+
| CM5           | Charge Model 5                           | Good balance of accuracy/speed                   |
+---------------+------------------------------------------+--------------------------------------------------+
| ADCH          | Atomic dipole corrected Hirshfeld        | Recommended by Multiwfn                          |
+---------------+------------------------------------------+--------------------------------------------------+
| Mulliken      | Mulliken population                      | Fast but basis-set dependent                     |
+---------------+------------------------------------------+--------------------------------------------------+
| Lowdin        | Löwdin population                        | Orthogonalized basis                             |
+---------------+------------------------------------------+--------------------------------------------------+
| Voronoi       | Voronoi deformation density              | Space partitioning method                        |
+---------------+------------------------------------------+--------------------------------------------------+
| SCPA          | Ros & Schuit modified Mulliken           | Modified Mulliken scheme                         |
+---------------+------------------------------------------+--------------------------------------------------+
| Becke         | Becke partitioning with dipole corr.     | Real-space integration                           |
+---------------+------------------------------------------+--------------------------------------------------+
| EEM           | Electronegativity equalization           | ⚠️ Requires bonded atoms (fails for ionic)       |
+---------------+------------------------------------------+--------------------------------------------------+
| PEOE          | Gasteiger charges                        | ⚠️ Missing parameters for Na, transition metals  |
+---------------+------------------------------------------+--------------------------------------------------+

**Multipole-capable schemes** (for ``multipole_bool=True`` or ``use_multipole=True``):

- Hirshfeld, Hirshfeld_I, Becke

Module Reference
----------------

.. automodule:: pyef.analysis
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyef.cli
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyef.utility
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: pyef.multiwfn_interface
   :members:
   :undoc-members:
   :show-inheritance:
