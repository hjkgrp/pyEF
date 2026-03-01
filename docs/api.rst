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
       ptchg_paths=None,  # Point charge files for QM/MM
       **kwargs           # Configuration options (see below)
   )

**Required parameters:**

- ``molden_paths`` (list of str): Paths to molden files, one per structure
- ``xyz_paths`` (list of str): Paths to XYZ files, one per structure

**Optional parameters:**

- ``esp_atom_idx`` (list of int): Atom indices for ESP calculations (0-indexed)
- ``ptchg_paths`` (list): Point charge file paths for QM/MM calculations

.. _configuration-kwargs:

Configuration kwargs
~~~~~~~~~~~~~~~~~~~~

The following keyword arguments can be passed to the ``Electrostatics`` constructor to
configure the calculation behavior. All are optional and have sensible defaults.

**ECP (Effective Core Potential) Options:**

.. list-table::
   :widths: 15 10 15 60
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``hasECP``
     - bool
     - ``False``
     - Set to ``True`` if your QM calculation used Effective Core Potentials. When enabled, pyEF will
       automatically reformat molden files to fix ECP-related artifacts that make them incompatible with Multiwfn.
   * - ``ECP``
     - str
     - ``"lacvps"``
     - The ECP basis set family used in the QM calculation. Only relevant when ``hasECP=True``.
       See :ref:`supported ECPs <supported-ecps>` below.

.. _supported-ecps:

**Supported ECP basis set families:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - ECP Value
     - Description
   * - ``"lacvps"``
     - LACVPS (default). Hybrid ECP: all-electron for elements with Z |le| 18 (up to Ar),
       LANL2DZ ECP for heavier elements. Common for TeraChem calculations with transition metals.
   * - ``"lacvp"``
     - LACVP. Hybrid ECP: all-electron for elements with Z |le| 10 (up to Ne),
       LANL2DZ ECP for heavier elements.
   * - ``"lanl2dz"``
     - LANL2DZ (Los Alamos National Laboratory 2-Double-Zeta). Widely used ECP for transition metals.
   * - ``"def2"``
     - def2-type ECPs (e.g., def2-SVP, def2-TZVP). Stuttgart/Cologne group ECPs used in the Ahlrichs basis set family.
   * - ``"crenbl"``
     - CRENBL (Christiansen, Ross, Ermler, Nash, Bursten, Large-core). Shape-consistent relativistic ECPs.
   * - ``"stuttgart_rsc"``
     - Stuttgart RSC (Relativistic Small Core). Energy-consistent ECPs from the Stuttgart/Cologne group.

.. |le| unicode:: U+2264

**Example with ECP:**

.. code-block:: python

   # For a TeraChem calculation that used LACVPS basis set
   es = Electrostatics(molden_paths, xyz_paths, hasECP=True, ECP="lacvps")

   # For a calculation that used def2-type ECPs
   es = Electrostatics(molden_paths, xyz_paths, hasECP=True, ECP="def2")

**Dielectric and Environment Options:**

.. list-table::
   :widths: 25 10 10 55
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``dielectric``
     - float
     - ``1``
     - Dielectric constant for the environment. Common values: 1.0 (vacuum), 2-4 (protein interior),
       20-40 (protein-solvent interface), 78.5 (water). Can also be overridden per method call.
   * - ``dielectric_scale``
     - float
     - ``1``
     - Scaling factor applied to point charges in the dielectric treatment. Can also be overridden per method call.
   * - ``changeDielectBoundBool``
     - bool
     - ``False``
     - When ``True``, uses dielectric=1 for atoms that are directly bonded, applying the dielectric
       scaling only to non-bonded interactions. Useful for modeling distinct dielectric boundaries.
   * - ``excludeAtomfromEcalc``
     - list
     - ``[]``
     - List of atom indices to exclude from E-field calculations. Typically used to exclude substrate
       atoms when probing only the environment's contribution. Can also be set via ``setExcludeAtomFromCalc()``.

**QM/MM Point Charge Options:**

.. list-table::
   :widths: 20 10 10 60
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``includePtChgs``
     - bool
     - ``False``
     - Include QMMM point charges in ESP/E-field calculations. Point charge files must be provided
       via ``ptchg_paths`` or ``ptChgfp``.
   * - ``ptChgfp``
     - str
     - ``''``
     - Legacy API: a single point charge filename applied to all jobs. Prefer using ``ptchg_paths`` instead.

**Computation Options:**

.. list-table::
   :widths: 25 10 10 55
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``rerun``
     - bool
     - ``False``
     - Force recalculation of charges even if cached results exist. By default, pyEF caches Multiwfn
       charge results and reuses them on subsequent runs.
   * - ``maxIHirshBasis``
     - int
     - ``12000``
     - Maximum number of basis functions allowed for Hirshfeld-I analysis. Increase for very large systems.
   * - ``maxIHirshFuzzyBasis``
     - int
     - ``6000``
     - Maximum number of basis functions allowed for fuzzy Hirshfeld-I analysis.
   * - ``skip_missing_files``
     - bool
     - ``False``
     - When ``True``, skip jobs where required input files are missing instead of raising an error.
       Useful for batch processing where some calculations may have failed.

**Visualization Options:**

.. list-table::
   :widths: 20 10 10 60
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``visualize_ef``
     - bool
     - ``False``
     - Create PDB files with atom-wise electric field contributions in the B-factor column.
   * - ``visualize_charges``
     - bool
     - ``False``
     - Create PDB files with partial charges in the B-factor column.
   * - ``visualize_per_bond``
     - bool
     - ``False``
     - Create separate PDB files for each bond in E-field calculations.

**Legacy/Compatibility Options:**

.. list-table::
   :widths: 20 10 25 45
   :header-rows: 1

   * - Parameter
     - Type
     - Default
     - Description
   * - ``molden_filename``
     - str
     - ``'final_optim.molden'``
     - Molden filename for backward compatibility with older directory-based workflows.
   * - ``xyzfilename``
     - str
     - ``'final_optim.xyz'``
     - XYZ filename for backward compatibility with older directory-based workflows.

.. _getefield-electric-field:

getEfield() - Electric Field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = es.getEfield(
       charge_types='Hirshfeld_I',   # Charge scheme(s)
       Efielddata_filename='ef',     # Output filename prefix
       multiwfn_path=None,           # Path to Multiwfn executable
       input_bond_indices=[],        # [(atom1, atom2), ...] per structure
       auto_find_bonds=False,        # Auto-detect bonds to adjacent atoms
       multipole_bool=False,         # Use multipole expansion
       save_atomwise_decomposition=False,  # Save per-atom contributions
       visualize=None,               # Create PDB visualization files
       dielectric=1,                 # Dielectric constant
       dielectric_scale=1,           # Dielectric scaling for point charges
       includePtChgs=None,           # Override point charge setting
       ptchg_paths=None,             # Override point charge file paths
       pdb_charge_paths=None         # Use charges from PDB B-factor column
   )

**Parameters:**

- ``charge_types`` (str or list): Charge scheme(s) to use (first entry used if list)
- ``Efielddata_filename`` (str): Output CSV filename prefix (default: ``'ef'``)
- ``multiwfn_path`` (str): Path to Multiwfn executable
- ``input_bond_indices`` (list of tuples): Bond pairs as ``(atom1, atom2)`` per structure
- ``auto_find_bonds`` (bool): Automatically find bonds to adjacent atoms (default: ``False``)
- ``multipole_bool`` (bool): ``True`` for multipole expansion, ``False`` for monopole (default: ``False``)
- ``save_atomwise_decomposition`` (bool): Save per-atom E-field breakdown (default: ``False``)
- ``visualize`` (bool or None): Create PDB files with E-field data in B-factor column
- ``dielectric`` (float): Dielectric constant (default: ``1``)
- ``dielectric_scale`` (float): Scaling factor for point charges (default: ``1``)
- ``includePtChgs`` (bool or None): Override the class-level point charge inclusion setting
- ``ptchg_paths`` (list or None): Override point charge file paths
- ``pdb_charge_paths`` (str or list): Read charges from PDB B-factor column instead of computing them (monopole only)

**Returns:** DataFrame with electric field results. If ``save_atomwise_decomposition=True``,
returns a tuple ``(total_df, atomwise_df)``.

.. _getesp-electrostatic-potential:

getESP() - Electrostatic Potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = es.getESP(
       charge_types='CM5',          # Charge scheme(s)
       ESPdata_filename='ESP',      # Output filename prefix
       multiwfn_path='None',        # Path to Multiwfn executable
       use_multipole=False,         # Use multipole expansion
       include_decay=False,         # Include distance-sorted ESP decay
       include_coord_shells=False,  # Include coordination shell ESP
       visualize=None,              # Create PDB visualization files
       dielectric=1,                # Dielectric constant
       dielectric_scale=1,          # Dielectric scaling for point charges
       includePtChgs=None,          # Override point charge setting
       ptchg_paths=None             # Override point charge file paths
   )

**Parameters:**

- ``charge_types`` (str or list): Charge scheme(s) to use
- ``ESPdata_filename`` (str): Output CSV filename prefix (default: ``'ESP'``)
- ``multiwfn_path`` (str): Path to Multiwfn executable
- ``use_multipole`` (bool): Use multipole expansion instead of monopole (default: ``False``)
- ``include_decay`` (bool): Include distance-sorted ESP decay analysis (default: ``False``)
- ``include_coord_shells`` (bool): Include coordination shell ESP breakdown (default: ``False``)
- ``visualize`` (bool or None): Create PDB files with ESP in B-factor column
- ``dielectric`` (float): Dielectric constant (default: ``1``)
- ``dielectric_scale`` (float): Scaling factor for point charges (default: ``1``)
- ``includePtChgs`` (bool or None): Override the class-level point charge inclusion setting
- ``ptchg_paths`` (list or None): Override point charge file paths

**Returns:** DataFrame with ESP results

**Note:** Requires ``esp_atom_idx`` to be set during initialization.

.. _getcharges-partial-charges:

getCharges() - Partial Charges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = es.getCharges(
       charge_types='Hirshfeld_I',        # Charge scheme(s)
       multiwfn_path=None,                # Path to Multiwfn executable
       output_filename='charges',         # Output filename prefix
       write_pdb=False,                   # Write PDB files with charges
       pdb_bfactor=True                   # Store charges in B-factor column
   )

**Parameters:**

- ``charge_types`` (str or list): Charge scheme(s) to use
- ``multiwfn_path`` (str): Path to Multiwfn executable
- ``output_filename`` (str): Output CSV filename prefix (default: ``'charges'``)
- ``write_pdb`` (bool): If ``True``, write PDB files with charges in B-factor column (default: ``False``)
- ``pdb_bfactor`` (bool): Store charges in B-factor column of PDB (default: ``True``)

**Returns:** DataFrame with columns: ``Job``, ``Charge_Type``, ``Atom_Index``, ``Element``, ``x``, ``y``, ``z``, ``Charge``, ``Molden_Path``, ``XYZ_Path``

.. _getelectrostatic-stabilization-electrostatic-stabilization:

getElectrostatic_stabilization() - Electrostatic Stabilization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   df = es.getElectrostatic_stabilization(
       multiwfn_path=None,                   # Path to Multiwfn executable
       substrate_idxs=[],                    # Substrate atom indices per structure
       charge_type='Hirshfeld_I',            # Charge scheme
       name_dataStorage='estatic',           # Output filename prefix
       env_idxs=None,                        # Environment atom indices (default: all non-substrate)
       save_atomwise_decomposition=False,    # Save per-atom contributions
       visualize=None,                       # Create PDB visualization files
       multipole_order=2,                    # 1=monopole, 2=dipole, 3=quadrupole
       substrate_multipole_order=None,       # Override multipole order for substrate
       env_multipole_order=None,             # Override multipole order for environment
       dielectric=1,                         # Dielectric constant
       dielectric_scale=1,                   # Dielectric scaling for point charges
       includePtChgs=None,                   # Override point charge setting
       ptchg_paths=None                      # Override point charge file paths
   )

**Parameters:**

- ``multiwfn_path`` (str): Path to Multiwfn executable
- ``substrate_idxs`` (list): List of atom indices for the substrate, one list per structure
- ``charge_type`` (str): Charge scheme (default: ``'Hirshfeld_I'``). Must be Hirshfeld, Hirshfeld_I, or Becke for ``multipole_order`` >= 2
- ``name_dataStorage`` (str): Output filename prefix (default: ``'estatic'``)
- ``env_idxs`` (list or None): Environment atom indices. If ``None``, uses all non-substrate atoms
- ``save_atomwise_decomposition`` (bool): Save per-atom energy breakdown (default: ``False``)
- ``visualize`` (bool or None): Create PDB files with stabilization data
- ``multipole_order`` (int): Multipole expansion order (default: ``2``):

  - ``1``: Monopole only (charge-charge interactions)
  - ``2``: Monopole + dipole (charge-charge, charge-dipole, dipole-dipole)
  - ``3``: Monopole + dipole + quadrupole (all terms up to quadrupole-quadrupole)

- ``substrate_multipole_order`` (int or None): Override ``multipole_order`` for the substrate only
- ``env_multipole_order`` (int or None): Override ``multipole_order`` for the environment only
- ``dielectric`` (float): Dielectric constant (default: ``1``)
- ``dielectric_scale`` (float): Scaling factor for point charges (default: ``1``)
- ``includePtChgs`` (bool or None): Include point charges as part of the environment
- ``ptchg_paths`` (list or None): Override point charge file paths

**Returns:** DataFrame with electrostatic stabilization energies (kcal/mol).
If ``save_atomwise_decomposition=True``, returns a tuple ``(total_df, atomwise_df)``.

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
