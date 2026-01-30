API Reference
=============

This page documents the main classes and functions for computing electric fields,
electrostatic potentials, and partial charges.

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
       lst_of_tmcm_idx=None,  # Metal atom indices for ESP (0-indexed)
       dielectric=1,      # Dielectric constant
       ptchg_paths=None   # Point charge files for QM/MM
   )

**Required parameters:**

- ``molden_paths`` (list of str): Paths to molden files, one per structure
- ``xyz_paths`` (list of str): Paths to XYZ files, one per structure

**Optional parameters:**

- ``lst_of_tmcm_idx`` (list of int): Metal atom indices for ESP calculations
- ``dielectric`` (float): Dielectric constant (1=vacuum, 4=protein, 78.5=water)
- ``ptchg_paths`` (list): Point charge file paths for QM/MM calculations

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

**Note:** Requires ``lst_of_tmcm_idx`` to be set during initialization.

Charge Schemes
--------------

**Available schemes:**

- Hirshfeld, Hirshfeld_I, Voronoi, Mulliken, Lowdin
- SCPA, Becke, ADCH, CHELPG, MK, AIM
- CM5, EEM, RESP, PEOE

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
