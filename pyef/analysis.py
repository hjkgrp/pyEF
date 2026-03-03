"""Electrostatics Analysis Module for Quantum Chemistry Calculations.

This module provides the Electrostatics class for computing electrostatic
properties (ESP, electric fields, partial charges) from quantum chemistry
calculations. It interfaces with Multiwfn for charge analysis and supports
various charge partitioning schemes (Hirshfeld, Hirshfeld-I, CHELPG, etc.).

Key Features:
    - Multiple charge partitioning schemes
    - ESP and electric field calculations
    - Multipole moment analysis
    - QMMM point charge integration
    - Dielectric environment effects
    - ECP (Effective Core Potential) support

Dependencies:
    - Multiwfn (external program)
    - NumPy, pandas, scipy
    - OpenBabel, BioPandas
"""

import re
import os
import mmap
import glob
import shutil
import logging
import traceback
import subprocess
import math
import time
import warnings

import numpy as np
import pandas as pd
import scipy.linalg as la
from scipy.optimize import LinearConstraint, minimize
from collections import deque
from importlib import resources
from distutils.dir_util import copy_tree

#import openbabel
#from biopandas.pdb import PandasPdb

from .geometry import Geometry, Visualize
from .utility import MoldenObject, MultiwfnInterface
from . import utility as constants  # Backward compatibility alias
from .validation import (
    validate_charge_type,
    validate_numeric_range,
    VALID_CHARGE_TYPES,
    filter_charge_types_for_multipole,
    filter_charge_types_for_monopole
)


class Electrostatics:
    """Compute electrostatic properties from quantum chemistry calculations.

    This class processes series of quantum chemistry calculations to extract
    electrostatic properties including ESP, electric fields, and partial charges
    using various charge partitioning schemes via Multiwfn.

    Attributes
    ----------
    molden_paths : list of str
        Paths to .molden files, one per job (absolute or relative).
    xyz_paths : list of str
        Paths to .xyz files, one per job (absolute or relative).
    esp_atom_idx : list of int
        Atom indices for ESP evaluation (typically metal centers).
    config : dict
        Configuration dictionary containing:
            - hasECP (bool): Whether ECP was used in calculation
            - includePtChgs (bool): Include point charges in ESP calculation
            - ptChgfp (str): Path to point charge file
            - molden_filename (str): Name of molden file (compatibility)
            - xyzfilename (str): Name of xyz coordinate file (compatibility)
            - rerun (bool): Force recalculation of charges
            - maxIHirshBasis (int): Max basis functions for Hirshfeld-I
            - maxIHirshFuzzyBasis (int): Max basis for fuzzy Hirshfeld-I
            - ECP (str): ECP basis set family
            - dielectric (float): Dielectric constant
            - dielectric_scale (float): Scaling factor for dielectric
            - excludeAtomfromEcalc (list): Atoms to exclude from E-field calc
            - changeDielectBoundBool (bool): Use dielectric=1 for bonded atoms
            - visualize_ef (bool): Create PDB files for atom-wise E-fields
            - visualize_charges (bool): Create PDB files for partial charges
            - visualize_per_bond (bool): Create separate PDB files for each bond
            - skip_missing_files (bool): Skip calculations if required files missing (default: False)
    dict_of_calcs : dict
        Mapping of charge scheme names to Multiwfn command codes.
    dict_of_multipole : dict
        Mapping of multipole schemes to Multiwfn command sequences.
    ptchgdf : pd.DataFrame or None
        DataFrame containing point charge data for QMMM calculations.
    periodic_table : dict
        Mapping of element symbols to atomic numbers.
    amassdict : dict
        Atomic properties (mass, atomic number, covalent radius, valence).

    Examples
    --------
    >>> molden_paths = ['calc1/scr/optim.molden', 'calc2/scr/optim.molden']
    >>> xyz_paths = ['calc1/scr/optim.xyz', 'calc2/scr/optim.xyz']
    >>> es = Electrostatics(molden_paths, xyz_paths, dielectric=2.0)
    >>>
    >>> # For ESP calculations (metal indices required)
    >>> es_with_metals = Electrostatics(molden_paths, xyz_paths, esp_atom_idx=[0, 0], dielectric=2.0)
    >>> data = es_with_metals.getESP(['Hirshfeld', 'Hirshfeld_I'], 'esp_results', ...)
    """

    def __init__(self, molden_paths=None, xyz_paths=None, esp_atom_idx=None,
                 ptchg_paths=None, pdb_charge_paths=None, **kwargs):
        """Initialize Electrostatics analysis object.

        Parameters
        ----------
        molden_paths : list of str or None
            List of .molden file paths for each job (absolute or relative).
            Example: ['/path/to/job1/optim.molden', '/path/to/job2/optim.molden']
            Not required when pdb_charge_paths is provided (pdb_only mode).
        xyz_paths : list of str or None
            List of .xyz file paths for each job (absolute or relative).
            Example: ['/path/to/job1/optim.xyz', '/path/to/job2/optim.xyz']
            Not required when pdb_charge_paths is provided (pdb_only mode).
        esp_atom_idx : list of int, optional
            List of atom indices for ESP calculation (0-indexed).
            Only required when running ESP analysis. Not needed for E-field
            or electrostatic stabilization if bond indices are provided.
        ptchg_paths : list of str or None, optional
            List of point charge file paths, one per job (absolute or relative).
            Use None for jobs without point charges.
            Example: ['/path/to/job1/ptchg.txt', None, '/path/to/job3/ptchg.txt']
            If not provided, can use includePtChgs() or ptChgfp config for compatibility.
        pdb_charge_paths : list of str or None, optional
            List of PDB file paths whose B-factor column contains partial charges,
            one per job. When provided, molden_paths and xyz_paths are not required
            and Multiwfn is never called (pdb_only mode). Coordinates and charges
            are read entirely from these PDB files.
            Example: ['/path/to/job1/charges.pdb', '/path/to/job2/charges.pdb']
        **kwargs : dict, optional
            Configuration options to override defaults. See class docstring
            for available configuration keys.

        Notes
        -----
        ECP options: "stuttgart_rsc", "def2", "crenbl", "lanl2dz", "lacvps"

        pdb_only mode: Pass pdb_charge_paths and omit (or pass None for) molden_paths
        and xyz_paths. All three calculation functions (getESP, getEfield,
        getElectrostatic_stabilization) will use the PDB charges directly.
        """
        # Backward compatibility: accept old keyword lst_of_tmcm_idx
        if 'lst_of_tmcm_idx' in kwargs:
            warnings.warn(
                "lst_of_tmcm_idx is deprecated, use esp_atom_idx instead",
                DeprecationWarning,
                stacklevel=2
            )
            if esp_atom_idx is not None:
                raise ValueError(
                    "Cannot specify both esp_atom_idx and lst_of_tmcm_idx. "
                    "Use esp_atom_idx (lst_of_tmcm_idx is deprecated)."
                )
            esp_atom_idx = kwargs.pop('lst_of_tmcm_idx')

        # Detect pdb_only mode: pdb_charge_paths provided without molden/xyz
        _pdb_only = pdb_charge_paths is not None and molden_paths is None and xyz_paths is None

        # Validate inputs (skip for pdb_only mode)
        if not _pdb_only:
            # Type check: molden_paths and xyz_paths must be lists
            if not isinstance(molden_paths, list):
                raise TypeError(f"""
{'='*60}
ERROR: Invalid type for molden_paths
{'='*60}
Expected: list of strings
Got: {type(molden_paths).__name__}

Correct usage:
  estat = Electrostatics(
      molden_paths=['job1.molden', 'job2.molden'],
      xyz_paths=['job1.xyz', 'job2.xyz']
  )

For pdb_only mode (no Multiwfn required):
  estat = Electrostatics(
      pdb_charge_paths=['job1.pdb', 'job2.pdb'],
      esp_atom_idx=[25, 30]
  )

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
- README.md: Section 3 (Detailed Tutorials)
{'='*60}
""")

            if not isinstance(xyz_paths, list):
                raise TypeError(f"""
{'='*60}
ERROR: Invalid type for xyz_paths
{'='*60}
Expected: list of strings
Got: {type(xyz_paths).__name__}

Correct usage:
  estat = Electrostatics(
      molden_paths=['job1.molden', 'job2.molden'],
      xyz_paths=['job1.xyz', 'job2.xyz']
  )

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
- README.md: Section 3 (Detailed Tutorials)
{'='*60}
""")

            # Length check
            if len(molden_paths) != len(xyz_paths):
                raise ValueError(f"""
{'='*60}
ERROR: Mismatched list lengths
{'='*60}
molden_paths has {len(molden_paths)} items
xyz_paths has {len(xyz_paths)} items

Both lists must have the same length (one molden and one xyz file per job).

Correct usage:
  estat = Electrostatics(
      molden_paths=['job1.molden', 'job2.molden'],  # 2 files
      xyz_paths=['job1.xyz', 'job2.xyz']            # 2 files
  )

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
{'='*60}
""")

        # Type check: esp_atom_idx must be list of integers if provided
        if esp_atom_idx is not None:
            # AUTO-CORRECT: Convert numpy arrays or other array-like objects to list
            if hasattr(esp_atom_idx, '__iter__') and not isinstance(esp_atom_idx, (str, dict)):
                # This catches numpy arrays, tuples, sets, etc. but not strings or dicts
                try:
                    esp_atom_idx = list(esp_atom_idx)
                    # Successfully converted - continue with validation below
                except (TypeError, ValueError):
                    pass  # If conversion fails, fall through to error handling

            if not isinstance(esp_atom_idx, list):
                # ENHANCED: Detect previous API usage pattern
                # Previous API: Electrostatics(folders, atoms, folder_to_file_path, molden_filename, xyzfilename)
                # Current API: Electrostatics(molden_paths, xyz_paths, esp_atom_idx)

                if isinstance(esp_atom_idx, str):
                    # Likely previous API: third argument was folder_to_file_path (string)
                    hint_msg = """

HINT: It looks like you may be using the previous API format.
The API has been updated. Here's how to migrate your code:

PREVIOUS API:
  estat = Electrostatics(
      list_of_folders,           # List of folder paths
      list_of_atoms,             # List of atom indices
      folder_to_file_path,       # String path
      molden_filename='optim.molden',
      xyzfilename='optim.xyz'
  )

CURRENT API:
  estat = Electrostatics(
      molden_paths,              # List of COMPLETE paths to .molden files
      xyz_paths,                 # List of COMPLETE paths to .xyz files
      esp_atom_idx=[25, 30]   # List of metal atom indices (optional)
  )

Or if you have folders, construct paths like this:
  folders = ['/path/to/job1', '/path/to/job2']
  molden_paths = [f'{folder}/optim.molden' for folder in folders]
  xyz_paths = [f'{folder}/optim.xyz' for folder in folders]
  metal_indices = [25, 30]  # Your atom indices

  estat = Electrostatics(molden_paths, xyz_paths, metal_indices)
"""
                else:
                    hint_msg = ""

                raise TypeError(f"""
{'='*60}
ERROR: Invalid type for esp_atom_idx
{'='*60}
Expected: list of integers
Got: {type(esp_atom_idx).__name__}

Correct usage:
  estat = Electrostatics(
      molden_paths=['job1.molden', 'job2.molden'],
      xyz_paths=['job1.xyz', 'job2.xyz'],
      esp_atom_idx=[25, 30]  # Metal atom indices (0-based)
  )

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py{hint_msg}
{'='*60}
""")

            # Check all elements are integers
            non_integers = [idx for idx in esp_atom_idx if not isinstance(idx, (int, np.integer))]
            if non_integers:
                raise TypeError(f"""
{'='*60}
ERROR: Non-integer values in esp_atom_idx
{'='*60}
Expected: all elements must be integers
Found non-integer values: {non_integers}

Correct usage:
  esp_atom_idx: [25, 30]  # Integers only

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
{'='*60}
""")

        # Initialize configuration with defaults
        self.config = {
            'hasECP': False,              # Use Effective Core Potentials
            'includePtChgs': False,       # Include QMMM point charges
            'ptChgfp': '',                # Point charge file path
            'molden_filename': 'final_optim.molden',  # Molden file name (compatibility)
            'xyzfilename': 'final_optim.xyz',         # XYZ file name (compatibility)
            'rerun': False,               # Force recalculation
            'maxIHirshBasis': 12000,      # Hirshfeld-I basis limit
            'maxIHirshFuzzyBasis': 6000,  # Fuzzy Hirshfeld-I basis limit
            'ECP': "lacvps",              # ECP basis set family
            'dielectric': 1,              # Dielectric constant
            'dielectric_scale': 1,        # Dielectric scaling factor
            'excludeAtomfromEcalc': [],   # Atoms to exclude from E-field
            'changeDielectBoundBool': False,  # Special dielectric for bonds
            'visualize_ef': False,    # Create PDB files for atom-wise E-fields
            'visualize_charges': False,   # Create PDB files for partial charges
            'visualize_per_bond': False,  # Create separate PDB files for each bond
            'skip_missing_files': False   # Skip calculations if required files are missing (don't raise errors)
        }
        # Override defaults with user-provided kwargs
        self.config.update(kwargs)

        # Validate config parameters from kwargs
        if 'dielectric' in kwargs:
            validate_numeric_range(kwargs['dielectric'], 'dielectric', min_val=0.001, context="Electrostatics initialization")

        if 'dielectric_scale' in kwargs:
            validate_numeric_range(kwargs['dielectric_scale'], 'dielectric_scale', min_val=0.001, context="Electrostatics initialization")

        if 'multipole_order' in kwargs:
            validate_numeric_range(kwargs['multipole_order'], 'multipole_order', allowed_values=[1, 2, 3], context="Electrostatics initialization")

        if 'charge_types' in kwargs:
            charge_types = kwargs['charge_types']
            if isinstance(charge_types, str):
                charge_types = [charge_types]
            elif not isinstance(charge_types, list):
                raise TypeError(f"""
{'='*60}
ERROR: Invalid type for charge_types parameter
{'='*60}
Expected: string or list of strings
Got: {type(charge_types).__name__}

Correct usage:
  estat = Electrostatics(
      molden_paths=['job1.molden'],
      xyz_paths=['job1.xyz'],
      charge_types=['Hirshfeld_I']  # List of charge types
  )

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
{'='*60}
""")
            for charge_type in charge_types:
                validate_charge_type(charge_type, context="Electrostatics initialization")

        # Store file paths directly
        if _pdb_only:
            # pdb_only mode: molden/xyz are not used; store None sentinels
            n_jobs = len(pdb_charge_paths)
            self.molden_paths = [None] * n_jobs
            self.xyz_paths = [None] * n_jobs
        else:
            self.molden_paths = [os.path.abspath(p) for p in molden_paths]
            self.xyz_paths = [os.path.abspath(p) for p in xyz_paths]
        self.esp_atom_idx = esp_atom_idx if esp_atom_idx is not None else []

        # Handle point charge paths with backward compatibility
        # Support 3 modes:
        # 1. New API: ptchg_paths=['path1.txt', None, 'path3.txt']
        # 2. Old API via kwargs: ptChgfp='filename.txt' (applied to all jobs)
        # 3. Old API via includePtChgs() method: called after init
        if ptchg_paths is not None:
            # New API: validate and store list
            if not isinstance(ptchg_paths, list):
                raise TypeError(f"""
{'='*60}
ERROR: Invalid type for ptchg_paths
{'='*60}
Expected: list of strings or None values
Got: {type(ptchg_paths).__name__}

Correct usage:
  estat = Electrostatics(
      molden_paths=['job1.molden', 'job2.molden'],
      xyz_paths=['job1.xyz', 'job2.xyz'],
      ptchg_paths=['job1_ptchg.txt', None]  # None for jobs without point charges
  )

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
{'='*60}
""")

            # Validate length matches
            if len(ptchg_paths) != len(molden_paths):
                raise ValueError(f"""
{'='*60}
ERROR: Length mismatch for ptchg_paths
{'='*60}
Expected length: {len(molden_paths)} (same as molden_paths)
Got length: {len(ptchg_paths)}

All path lists must have the same length. Use None for jobs without point charges.

Correct usage:
  estat = Electrostatics(
      molden_paths=['job1.molden', 'job2.molden'],
      xyz_paths=['job1.xyz', 'job2.xyz'],
      ptchg_paths=['job1_ptchg.txt', None]  # 2 items, matching molden_paths
  )
{'='*60}
""")

            # Convert to absolute paths, preserving None values
            self.ptchg_paths = [
                os.path.abspath(p) if p is not None else None
                for p in ptchg_paths
            ]

            # Set config flag if any non-None paths exist
            if any(p is not None for p in self.ptchg_paths):
                self.config['includePtChgs'] = True

        elif 'ptChgfp' in kwargs and kwargs['ptChgfp']:
            # Old API via kwargs: single filename for all jobs
            self.ptchg_paths = None  # Signal to use filename mode
            # ptChgfp already set in config update above
        else:
            # No point charges specified
            self.ptchg_paths = None

        # Store pdb_only mode flag and PDB paths
        self.pdb_only = _pdb_only
        if _pdb_only:
            self.pdb_charge_paths = [os.path.abspath(p) for p in pdb_charge_paths]
            # Working directories come from PDB file locations
            self.lst_of_folders = [os.path.dirname(p) for p in self.pdb_charge_paths]
        else:
            self.pdb_charge_paths = None
            # Create list of working directories (one per job) from the molden file paths
            self.lst_of_folders = [os.path.dirname(p) for p in self.molden_paths]
        self.folder_to_file_path = ''  # No longer needed, but kept for compatibility

        # Initialize Multiwfn interface with current config
        self.multiwfn = MultiwfnInterface(config=self.config)

        # Periodic table: element symbol -> atomic number
        # Source: molsimplify, http://www.webelements.com/
        # Note: Full capitalization included for compatibility with various
        # software that write .xyz files in all caps
        # Use centralized constants from constants module
        self.periodic_table = constants.PERIODIC_TABLE
        self.amassdict = constants.ATOMIC_MASS_DICT

        # Prepare data files (validate file existence)
        if self.pdb_only:
            self.prepData_pdb()
        else:
            self.prepData()
            # Fix molden files if ECP was used in calculations
            if self.config['hasECP']:
                self.fix_allECPmolden()

    # =========================================================================
    # Validation Methods
    # =========================================================================

    def _validate_atom_indices(self, indices, xyz_path, context=""):
        """
        Validate that atom indices are within structure bounds.

        Parameters
        ----------
        indices : int, list of int, or list of tuples
            Atom index or list of atom indices to validate (0-indexed)
        xyz_path : str
            Path to XYZ file to get atom count
        context : str, optional
            Description of where indices are used (for error message)

        Raises
        ------
        ValueError
            If any index is out of bounds
        """
        from .validation import get_atom_count_from_xyz, validate_atom_indices

        # Get atom count from XYZ file
        atom_count = get_atom_count_from_xyz(xyz_path)

        # Convert to flat list of indices
        flat_indices = []
        if isinstance(indices, int):
            flat_indices = [indices]
        elif isinstance(indices, (list, tuple)):
            for item in indices:
                if isinstance(item, (int, np.integer)):
                    flat_indices.append(int(item))
                elif isinstance(item, (list, tuple)):
                    # Handle bond tuples like [(1, 2), (3, 4)]
                    for sub_item in item:
                        if isinstance(sub_item, (int, np.integer)):
                            flat_indices.append(int(sub_item))

        # Validate all indices
        if flat_indices:
            validate_atom_indices(flat_indices, atom_count, context=context, xyz_path=xyz_path)

    # =========================================================================
    # Multiwfn Interface Methods
    # =========================================================================

    def run_multiwfn(self, command, input_commands, output_file=None,
                     description="Multiwfn calculation", capture_output=True):
        """Centralized Multiwfn runner with proper error handling and output display.

        Parameters
        ----------
        command : str
            The shell command to run Multiwfn (e.g., "multiwfn file.molden")
        input_commands : list of str
            List of commands to send to Multiwfn's stdin
        output_file : str, optional
            Path to redirect stdout to a file
        description : str, optional
            Description of the calculation for error messages
        capture_output : bool, optional
            If True, capture and display output. If False, let it stream to terminal

        Returns
        -------
        tuple
            (stdout, stderr, returncode) from the Multiwfn process

        Raises
        ------
        RuntimeError
            If Multiwfn fails with non-zero exit code
        """
        import sys

        # Prepare the full command with output redirection if specified
        full_command = command
        if output_file:
            full_command = f"{command} > {output_file}"

        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {full_command}")
        print(f"{'='*60}\n")

        try:
            # Run Multiwfn process
            if capture_output:
                proc = subprocess.Popen(
                    full_command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    text=True,
                    bufsize=1
                )

                # Send commands to stdin
                input_str = "\n".join(input_commands)
                stdout, stderr = proc.communicate(input=input_str)

                # Display output in real-time style
                if stdout:
                    print("MULTIWFN OUTPUT:")
                    print("-" * 60)
                    print(stdout)
                    print("-" * 60)

                if stderr:
                    print("MULTIWFN STDERR:")
                    print("-" * 60)
                    print(stderr, file=sys.stderr)
                    print("-" * 60)

            else:
                # Let output stream directly to terminal
                proc = subprocess.Popen(
                    full_command,
                    stdin=subprocess.PIPE,
                    shell=True,
                    text=True
                )
                input_str = "\n".join(input_commands)
                proc.communicate(input=input_str)
                stdout, stderr = "", ""

            returncode = proc.returncode

            # Check for errors
            if returncode != 0:
                error_msg = f"""
{'='*60}
ERROR: {description} FAILED
{'='*60}
Command: {full_command}
Return code: {returncode}

STDOUT:
{stdout if stdout else '(empty)'}

STDERR:
{stderr if stderr else '(empty)'}

Input commands sent:
{chr(10).join(input_commands)}
{'='*60}
"""
                raise RuntimeError(error_msg)

            print(f"\n✓ {description} completed successfully\n")
            return stdout, stderr, returncode

        except Exception as e:
            error_msg = f"""
{'='*60}
EXCEPTION during {description}
{'='*60}
Command: {full_command}
Error: {str(e)}

Traceback:
{traceback.format_exc()}

Input commands that were to be sent:
{chr(10).join(input_commands)}
{'='*60}
"""
            print(error_msg, file=sys.stderr)
            raise RuntimeError(error_msg) from e

    # =========================================================================
    # Configuration Methods
    # =========================================================================

    def updateCalcSettings(self, key, value):
        """Update calculation settings dictionary.

        Parameters
        ----------
        key : str
            Setting key to update.
        value : any
            New value for the setting.
        """
        self.dict_settings[key] = value

    def setExcludeAtomFromCalc(self, lstExcludeAtoms):
        """Set atoms to exclude from electric field calculations.

        Parameters
        ----------
        lstExcludeAtoms : list of int
            Atom indices (0-indexed) to exclude from E-field calculations.
        """
        self.config['excludeAtomfromEcalc'] = lstExcludeAtoms

    def includePtChgs(self, name_ptch_file):
        """Include point charges in ESP calculation (for QMMM).

        Parameters
        ----------
        name_ptch_file : str or list
            If str, uses a single filename (e.g., 'pointcharges.txt') that is
            searched for in each job directory.
            If list, uses per-job file paths, one per job; use None for jobs
            without point charges.

        Notes
        -----
        Sets includePtChgs config to True and stores filename/paths.
        Point charges are typically from MM region in QM/MM calculations.

        For filename mode (string), the file is automatically located in each job's working directory.
        For list mode, paths may be absolute or relative to the current working directory.

        Examples
        --------
        # Single filename (applies per job directory)
        estat.includePtChgs('pointcharges.txt')

        # Per-job paths
        estat.includePtChgs(['/path/job1/ptchg.txt', None, '/path/job3/ptchg.txt'])
        """
        if isinstance(name_ptch_file, str):
            # Filename mode: single filename for all jobs
            self.config['ptChgfp'] = name_ptch_file
            self.config['includePtChgs'] = True
            self.ptchg_paths = None  # Signal to use filename mode
            print(f'Point charges to be included via {name_ptch_file}')
        elif isinstance(name_ptch_file, list):
            # List mode: per-job paths
            if len(name_ptch_file) != len(self.molden_paths):
                raise ValueError(f"""
{'='*60}
ERROR: Length mismatch for point charge paths
{'='*60}
Expected length: {len(self.molden_paths)} (same as molden_paths)
Got length: {len(name_ptch_file)}

All path lists must have the same length. Use None for jobs without point charges.

Correct usage:
  estat.includePtChgs(['job1_ptchg.txt', None, 'job3_ptchg.txt'])
{'='*60}
""")

            self.ptchg_paths = [
                os.path.abspath(p) if p is not None else None
                for p in name_ptch_file
            ]
            self.config['includePtChgs'] = True
            non_none_count = sum(1 for p in self.ptchg_paths if p is not None)
            print(f'Point charges to be included for {non_none_count}/{len(self.ptchg_paths)} jobs')
        else:
            raise TypeError(f"name_ptch_file must be str or list, got {type(name_ptch_file)}")

    def _get_ptchg_path_for_job(self, job_index, folder_path, override_path=None):
        """Get point charge file path for a specific job.

        Handles multiple modes with proper precedence:
        1. Function-level override (highest priority)
        2. Per-job ptchg_paths list
        3. Filename-based ptChgfp setting

        Parameters
        ----------
        job_index : int
            Index of the current job (0-based)
        folder_path : str
            Absolute path to job folder (for filename-based mode)
        override_path : str or None, optional
            Function-level override path (highest priority)

        Returns
        -------
        str or None
            Absolute path to point charge file, or None if not available
        """
        # Priority 1: Function-level override
        if override_path is not None:
            if override_path == '':  # Empty string means explicitly disable
                return None
            return os.path.abspath(override_path)

        # Priority 2: Per-job ptchg_paths list
        if self.ptchg_paths is not None:
            if job_index >= len(self.ptchg_paths):
                raise IndexError(f"Job index {job_index} out of range for ptchg_paths (length {len(self.ptchg_paths)})")
            return self.ptchg_paths[job_index]  # May be None

        # Priority 3: Filename-based ptChgfp setting
        if self.config.get('ptChgfp', ''):
            ptchg_filename = self.config['ptChgfp']
            return os.path.join(folder_path, ptchg_filename)

        return None

    def set_dielec_scale(self, dielec):
        """Set dielectric scaling factor for point charges.

        Parameters
        ----------
        dielec : float
            Scaling factor for dielectric in QMMM calculations.
        """
        self.config['dielectric_scale'] = dielec

    def initialize_excludeAtomsFromEfieldCalc(self, atom_to_exclude):
        """Initialize atoms to exclude from electric field calculation.

        Parameters
        ----------
        atom_to_exclude : list of int
            List of atom indices (0-indexed) to exclude.
        """
        self.config['excludeAtomfromEcalc'] = atom_to_exclude

    def minDielecBonds(self, bool_bonds):
        """Set dielectric to 1 for bonded atoms in ESP calculation.

        Parameters
        ----------
        bool_bonds : bool
            If True, use dielectric=1 for atoms bonded to ESP center.
            Helps avoid over-estimating screening from bound atoms.
        """
        self.config['changeDielectBoundBool'] = bool_bonds

    def runlowmemory(self):
        """Enable low-memory mode for large systems (>300 atoms).

        Notes
        -----
        Switches Hirshfeld-I calculation to slower but lower-memory algorithm.
        Useful for large systems that would otherwise exceed memory limits.
        """
        self.multiwfn.dict_of_multipole = {
            'Hirshfeld': ['3', '2'],
            'Hirshfeld_I': ['4', '-2', '1', '2'],
            'Becke': ['1', '2']
        }
        self.multiwfn.dict_of_calcs = {
            'Hirshfeld': '1', 'Voronoi': '2', 'Mulliken': '5', 'Lowdin': '6',
            'SCPA': '7', 'Becke': '10', 'ADCH': '11', 'CHELPG': '12',
            'MK': '13', 'AIM': '14', 'Hirshfeld_I': ['15', '-2'],
            'CM5': '16', 'EEM': '17', 'RESP': '18', 'PEOE': '19'
        }

    def changeDielectric(self, dlc):
        """Change dielectric constant of the environment.

        Parameters
        ----------
        dlc : float
            Dielectric constant of solvent/environment.

        Notes
        -----
        Used to model solvation effects in ESP calculations.
        Default is 1 (vacuum). Common values: water=80, protein=4.
        """
        self.config['dielectric'] = dlc

    def set_molden_filename(self, new_name):
        """Set the expected molden filename.

        Parameters
        ----------
        new_name : str
            Name of the molden file to read.
        """
        self.config['molden_filename'] = new_name
        self.multiwfn.config['molden_filename'] = new_name

    def set_xyzfilename(self, new_name):
        """Set the expected XYZ coordinate filename.

        Parameters
        ----------
        new_name : str
            Name of the XYZ file to read.
        """
        self.config['xyzfilename'] = new_name
        self.multiwfn.config['xyzfilename'] = new_name

    def rePrep(self):
        """Re-run data preparation (file extraction and validation).

        Notes
        -----
        Useful if files have been updated or regenerated.
        """
        self.prepData()

    # =========================================================================
    # File Preparation and Utility Methods
    # =========================================================================

    def fix_allECPmolden(self):
        """Fix molden files that used Effective Core Potentials.

        Notes
        -----
        ECPs can cause artifacts in molden files that make them incompatible
        with Multiwfn. This method reformats the molden files to fix these issues.
        Processes all molden/xyz file pairs.
        """
        owd = os.getcwd()
        basis_set_fam = self.config['ECP']
        print('   > Re-formatting .molden files to fix ECP artifacts')

        for molden_path, xyz_path in zip(self.molden_paths, self.xyz_paths):
            os.chdir(owd)
            molden = MoldenObject(xyz_path, molden_path)
            molden.fix_ECPmolden(owd, basis_set_fam)

        os.chdir(owd)

    def prepData(self):
        """Validate that all specified .molden and .xyz files exist.

        Checks that all file paths provided during initialization are valid
        and accessible. Removes invalid entries from processing lists.

        Raises
        ------
        FileNotFoundError
            If any of the specified .molden or .xyz files cannot be found.
        """
        print('   > Validating file paths')

        valid_indices = []
        for idx, (molden_path, xyz_path) in enumerate(zip(self.molden_paths, self.xyz_paths)):
            try:
                # Check if molden file exists
                if not os.path.exists(molden_path):
                    print(f'ERROR: Molden file not found: {molden_path}')
                    continue

                # Check if xyz file exists
                if not os.path.exists(xyz_path):
                    print(f'ERROR: XYZ file not found: {xyz_path}')
                    continue

                # Both files exist
                print(f'      ✓ Job {idx+1}: Found .molden and .xyz files')
                print(f'         - Molden: {molden_path}')
                print(f'         - XYZ:    {xyz_path}')
                valid_indices.append(idx)

            except Exception as e:
                print(f'ERROR: Exception while checking files for job {idx+1}: {e}')
                continue

        # Update file lists to only include valid entries
        if len(valid_indices) == 0:
            raise FileNotFoundError("No valid .molden and .xyz file pairs found!")

        if len(valid_indices) < len(self.molden_paths):
            print(f'\nWARNING: Only {len(valid_indices)}/{len(self.molden_paths)} jobs have valid files.')
            print('         Invalid jobs will be removed from processing.')
            self.molden_paths = [self.molden_paths[i] for i in valid_indices]
            self.xyz_paths = [self.xyz_paths[i] for i in valid_indices]
            self.lst_of_folders = [self.lst_of_folders[i] for i in valid_indices]
            if self.esp_atom_idx:
                self.esp_atom_idx = [self.esp_atom_idx[i] for i in valid_indices]

        print(f'\n   > Validation complete: {len(valid_indices)} valid job(s) ready for processing')

    def prepData_pdb(self):
        """Validate PDB charge files for pdb_only mode.

        Checks that all PDB files provided during initialization exist.
        Removes invalid entries from processing lists.

        Raises
        ------
        FileNotFoundError
            If no valid PDB charge files are found.
        """
        print('   > Validating PDB charge file paths (pdb_only mode)')

        valid_indices = []
        for idx, pdb_path in enumerate(self.pdb_charge_paths):
            if not os.path.exists(pdb_path):
                print(f'ERROR: PDB charge file not found: {pdb_path}')
                continue
            print(f'      ✓ Job {idx+1}: Found PDB charge file')
            print(f'         - PDB: {pdb_path}')
            valid_indices.append(idx)

        if len(valid_indices) == 0:
            raise FileNotFoundError("No valid PDB charge files found!")

        if len(valid_indices) < len(self.pdb_charge_paths):
            print(f'\nWARNING: Only {len(valid_indices)}/{len(self.pdb_charge_paths)} jobs have valid PDB files.')
            print('         Invalid jobs will be removed from processing.')
            self.pdb_charge_paths = [self.pdb_charge_paths[i] for i in valid_indices]
            self.molden_paths = [self.molden_paths[i] for i in valid_indices]
            self.xyz_paths = [self.xyz_paths[i] for i in valid_indices]
            self.lst_of_folders = [self.lst_of_folders[i] for i in valid_indices]
            if self.esp_atom_idx:
                self.esp_atom_idx = [self.esp_atom_idx[i] for i in valid_indices]

        print(f'\n   > Validation complete: {len(valid_indices)} valid pdb_only job(s) ready for processing')

    # =========================================================================
    # Multipole and Charge Parsing Methods
    # =========================================================================

    @staticmethod
    def getmultipoles(multipole_name):
        """Parse multipole moments from Multiwfn output file.

        Parameters
        ----------
        multipole_name : str
            Path to multipole moment output file from Multiwfn.

        Returns
        -------
        list of dict
            List of dictionaries containing multipole data for each atom:
            - 'Index': atom index (str)
            - 'Element': element symbol (str)
            - 'Atom_Charge': atomic charge (float)
            - 'Dipole_Moment': dipole moment vector [x, y, z] (list of float)
            - 'Quadrupole_Moment': 3x3 quadrupole tensor (np.ndarray)

        Notes
        -----
        Parses Multiwfn output format with sections separated by asterisks.
        Extracts monopole (charge), dipole, and quadrupole moments for each atom.
        """
        # Read the text file
        with open(multipole_name, 'r') as file:
            text = file.read()
            # Split the text into sections based on the delimiter
            sections = re.split(r'\n\s*[*]+\s*', text)

        # Define a pattern to extract the index, element name, and atomic dipole and quadrupole moment values
        #pattern = r'Atomic multipole moments of\s+(\d+)\s*\(([^)]+)\s*\).*?X=\s*([-+]?\d+\.\d+)\s+Y=\s*([-+]?\d+\.\d+)\s+Z=\s*([-+]?\d+\.\d+).*?XX=\s*([-+]?\d+\.\d+)\s+XY=\s*([-+]?\d+\.\d+)\s+XZ=\s*([-+]?\d+\.\d+).*?YX=\s*([-+]?\d+\.\d+)\s+YY=\s*([-+]?\d+\.\d+)\s+YZ=\s*([-+]?\d+\.\d+).*?ZX=\s*([-+]?\d+\.\d+)\s+ZY=\s*([-+]?\d+\.\d+)\s+ZZ=\s*([-+]?\d+\.\d+)'
        pattern = (r'Atomic multipole moments of\s+(\d+)\s*\(([^)]+)\s*\).*?'
    r'Atomic monopole moment \(electron\):\s*([-+]?\d+\.\d+)\s+Atomic charge:\s*([-+]?\d+\.\d+).*?'
    r'X=\s*([-+]?\d+\.\d+)\s+Y=\s*([-+]?\d+\.\d+)\s+Z=\s*([-+]?\d+\.\d+).*?'
    r'XX=\s*([-+]?\d+\.\d+)\s+XY=\s*([-+]?\d+\.\d+)\s+XZ=\s*([-+]?\d+\.\d+).*?'
    r'YX=\s*([-+]?\d+\.\d+)\s+YY=\s*([-+]?\d+\.\d+)\s+YZ=\s*([-+]?\d+\.\d+).*?'
    r'ZX=\s*([-+]?\d+\.\d+)\s+ZY=\s*([-+]?\d+\.\d+)\s+ZZ=\s*([-+]?\d+\.\d+)'
    )
        atomicDicts = []

        # Iterate through sections and extract the information
        for section in sections:
            match = re.search(pattern, section, re.DOTALL)
            if match:
                index = match.group(1)
                element = match.group(2)
                monopole_moment = float(match.group(3))
                atomic_charge = float(match.group(4))
                x_value = match.group(5)
                y_value = match.group(6)
                z_value = match.group(7)
                dipole_moment = [float(x_value), float(y_value), float(z_value)]

                xx_value = match.group(8)
                xy_value = match.group(9)
                xz_value = match.group(10)
                yx_value = match.group(11)
                yy_value = match.group(12)
                yz_value = match.group(13)
                zx_value = match.group(14)
                zy_value = match.group(15)
                zz_value = match.group(16)
                # Create a 3x3 matrix for the quadrupole moment
                quadrupole_moment = np.array([
                [float(xx_value), float(xy_value), float(xz_value)],
                [float(yx_value), float(yy_value), float(yz_value)],
                [float(zx_value), float(zy_value), float(zz_value)]
                ])

                atomDict = {"Index": index, "Element": element, "Atom_Charge": atomic_charge, 'Dipole_Moment': dipole_moment, 'Quadrupole_Moment': quadrupole_moment}
                atomicDicts.append(atomDict)
        return atomicDicts

    def getPtChgs(self, filename_pt):
        """Load point charges from file (for QMMM calculations).

        Parameters
        ----------
        filename_pt : str
            Path to point charge file (typically from Amber/CHARMM).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['Atom', 'charge', 'x', 'y', 'z']
            All atoms labeled as 'pnt' (point charge).

        Notes
        -----
        File format: whitespace-delimited with 2 header lines to skip.
        """
        chg_df = pd.read_table(
            filename_pt, skiprows=2, delim_whitespace=True,
            names=['charge', 'x', 'y', 'z']
        )
        atm_name = ['pnt']
        atoms = atm_name * len(chg_df['charge'])
        chg_df['Atom'] = atoms
        return chg_df

    def getPdbCharges(self, pdb_path):
        """Load point charges from a PDB file using the B-factor column.

        Parameters
        ----------
        pdb_path : str
            Path to PDB file containing charges in the B-factor column.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ['Atom', 'charge', 'x', 'y', 'z']

        Notes
        -----
        Reads ATOM/HETATM records and parses fixed-width PDB columns.
        The B-factor column is interpreted as the partial charge (e).
        """
        rows = []
        with open(pdb_path, "r") as handle:
            for line_num, line in enumerate(handle, start=1):
                record = line[0:6].strip()
                if record not in ("ATOM", "HETATM"):
                    continue
                if len(line) < 66:
                    raise ValueError(
                        f"PDB line too short to parse coordinates/B-factor at line {line_num}: {line.rstrip()}"
                    )
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid PDB coordinate at line {line_num}: {line.rstrip()}"
                    ) from exc
                try:
                    charge = float(line[60:66].strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid PDB B-factor charge at line {line_num}: {line.rstrip()}"
                    ) from exc
                atom = line[76:78].strip()
                if not atom:
                    atom = line[12:16].strip()
                rows.append({"Atom": atom, "x": x, "y": y, "z": z, "charge": charge})

        if not rows:
            raise ValueError(f"No ATOM/HETATM records found in PDB: {pdb_path}")

        return pd.DataFrame(rows)

    @staticmethod
    def mapcount(filename):
        """Rapidly count lines in a file using memory mapping.

        Parameters
        ----------
        filename : str
            Path to file to count.

        Returns
        -------
        int
            Number of lines in the file.

        Notes
        -----
        Uses mmap for efficient line counting in large files.
        """
        f = open(filename, "r+")
        buf = mmap.mmap(f.fileno(), 0)
        lines = 0
        readline = buf.readline

        while readline():
            lines += 1
        return lines

    # =========================================================================
    # Electrostatic Potential (ESP) Calculation Methods
    # =========================================================================

    def monopole_esp(self, path_to_xyz, espatom_idx, charge_range, charge_file, ptchg_path=None):
        """Calculate electrostatic potential at specified atom.

        Parameters
        ----------
        path_to_xyz : str
            Path to XYZ coordinate file.
        espatom_idx : int
            Atom index (0-indexed) where ESP will be calculated.
        charge_range : list of int
            Indices of atoms to include in ESP calculation.
        charge_file : str
            Path to partial charge file from Multiwfn.
        ptchg_path : str or None, optional
            Path to point charge file. If None, no point charges are included.

        Returns
        -------
        list
            [ESP (float), atom_symbol (str)]
            ESP in Volts at the specified atom.

        Notes
        -----
        - Uses Coulomb's law: V = k * q / r with dielectric screening
        - Point charges from QMMM included if ptchg_path is provided
        - Bonded atoms treated with dielectric=1 if config['changeDielectBoundBool'] is True
        - Units: Volts (N·m/C = J/C)
        """
        dielectric = self.config['dielectric']
        df = pd.read_csv(charge_file, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])

        # Convert each column to list for quicker indexing
        atoms = list(df['Atom'])
        charges = list(df['charge'])
        xs = list(df['x'])
        ys = list(df['y'])
        zs = list(df['z'])

        #For QMMM calculation, include point charges in ESP calculation
        if ptchg_path is not None:
            df_ptchg = self.getPtChgs(ptchg_path)
            xs = xs + list(df_ptchg['x'])
            ys = ys + list(df_ptchg['y'])
            zs = zs + list(df_ptchg['z'])
            charges = charges + list(df_ptchg['charge'])
            atoms = atoms + list(df_ptchg['Atom'])
            #change charge range to include these new partial charges!
            charge_range = range(0, len(xs))
        # Pick the index of the atom at which the esp should be calculated
        idx_atom = espatom_idx

        # Determine position and charge of the target atom
        xo = xs[idx_atom]
        yo = ys[idx_atom]
        zo = zs[idx_atom]
        total_esp = 0

        bound_atoms = []
        #create list of bound atoms, these are treated with a different dielectric
        if self.config['changeDielectBoundBool']:
            bound_atoms = self.getBondedAtoms(path_to_xyz, idx_atom)
 
        for idx in charge_range:
            if idx == idx_atom:
                continue
            elif idx in bound_atoms:
                #now account for bound atoms
                 r = (((xs[idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((ys[idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((zs[idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                 total_esp = total_esp + (charges[idx]/r)
            else:
                # Calculate esp and convert to units (A to m)
                r = (((xs[idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((ys[idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((zs[idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                total_esp = total_esp + (1/dielectric)*(charges[idx]/r)

        final_esp = constants.COULOMB_CONSTANT*total_esp*constants.ELEMENTARY_CHARGE  #N*m^2/(C^2)*(C/m) = N*m/C = J/C = Volt
        return [final_esp, df['Atom'][idx_atom]]

    def monopole_esp_df(self, df_charges, espatom_idx, charge_range):
        """Calculate ESP at a specified atom using charges from a DataFrame.

        This is the pdb_only equivalent of monopole_esp(): it reads directly
        from a DataFrame (e.g. produced by getPdbCharges) instead of a file.
        The changeDielectBoundBool feature is not supported in this mode because
        bond detection requires an XYZ file.

        Parameters
        ----------
        df_charges : pd.DataFrame
            DataFrame with columns ['Atom', 'x', 'y', 'z', 'charge'].
            Typically returned by getPdbCharges().
        espatom_idx : int
            Atom index (0-indexed) at which the ESP is calculated.
        charge_range : list of int
            Indices of atoms to include in the ESP sum.

        Returns
        -------
        list
            [ESP (float in Volts), atom_symbol (str)]
        """
        dielectric = self.config['dielectric']
        df = df_charges.reset_index(drop=True)
        atoms = list(df['Atom'])
        charges = list(df['charge'])
        xs = list(df['x'])
        ys = list(df['y'])
        zs = list(df['z'])

        idx_atom = espatom_idx
        xo, yo, zo = xs[idx_atom], ys[idx_atom], zs[idx_atom]
        total_esp = 0

        for idx in charge_range:
            if idx == idx_atom:
                continue
            r = (((xs[idx] - xo) * constants.ANGSTROM_TO_M)**2 +
                 ((ys[idx] - yo) * constants.ANGSTROM_TO_M)**2 +
                 ((zs[idx] - zo) * constants.ANGSTROM_TO_M)**2) ** 0.5
            total_esp += (1 / dielectric) * (charges[idx] / r)

        final_esp = constants.COULOMB_CONSTANT * total_esp * constants.ELEMENTARY_CHARGE
        return [final_esp, atoms[idx_atom]]

    def calc_firstTermE(espatom_idx, charge_range, charge_file):
        '''
        Input: espatom_idx: integer of atom index
        charge_range: list of integers of atom indices
        charge_file: string of charge filename
        Output: list of E-field vector and atomic symbol
        '''

        # E in units of V/(ansgrom) = N*m/(C*Angstrom)
        df = pd.read_csv(charge_file, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])

        # Convert each column to list for quicker indexing
        charges = df['charge']
        xs = df['x']
        ys = df['y']
        zs = df['z']

        # Pick the index of the atom at which the esp should be calculated
        idx_atom = espatom_idx

        # Determine position and charge of the target atom
        xo = xs[idx_atom]
        yo = ys[idx_atom]
        zo = zs[idx_atom]
        position_vec = [xo, yo, zo]
        Ex = 0
        Ey = 0
        Ez = 0

        for idx in charge_range:
            if idx == idx_atom:
                continue
            else:
                # Calculate Efield and convert to units (A to m); Calc E-field stenth in kJ/mol*e*m
                r = (((xs[idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((ys[idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((zs[idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                Ex = Ex - constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx]/r)*(1/(xs[idx] - xo))
                Ey = Ey - constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx]/r)*(1/(ys[idx] - yo))
                Ez = Ez - constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx]/r)*(1/(zs[idx] - zo))

        E_vec = [Ex, Ey, Ez]
        return [E_vec, position_vec, df['Atom'][idx_atom]]



    def monopole_atomicEfield(self, espatom_idx, charge_range, charge_file, df_ptchg=None):
        '''
        Input: espatom_idx: integer of atom index
        charge_range: list of integers of atom indices
        charge_file: string of charge filename
        df_ptchg: optional pre-loaded point charge dataframe
        Output: list of E-field vector and atomic symbol
        '''

        inv_eps = 1/(self.config['dielectric'])
        # E in units of V/(ansgrom) = N*m/(C*Angstrom)
        df = pd.read_csv(charge_file, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])

        # Convert each column to list for quicker indexing
        charges = df['charge']
        xs = df['x']
        ys = df['y']
        zs = df['z']

        # Pick the index of the atom at which the efield should be calculated
        idx_atom = espatom_idx

        # Determine position and charge of the target atom
        xo = xs[idx_atom]
        yo = ys[idx_atom]
        zo = zs[idx_atom]
        position_vec = constants.ANGSTROM_TO_M*np.array([xo, yo, zo])
        Ex = 0
        Ey = 0
        Ez = 0 
        atom_wise_additions = []
        for idx in range(0, len(xs)):
            if idx == idx_atom:
                atom_wise_additions.append([0,0,0])
                continue
            elif idx not in charge_range:
                atom_wise_additions.append([0,0,0])
                continue
            else:
                # Calculate esp and convert to units (A to m); Calc E-field stenth in kJ/mol*e*m
                r = (((xs[idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((ys[idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((zs[idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                Ex_contrib = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx])*((xs[idx] - xo)*constants.ANGSTROM_TO_M)/(r**3)
                Ey_contrib = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx])*((ys[idx] - yo)*constants.ANGSTROM_TO_M)/(r**3)
                Ez_contrib = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx])*((zs[idx] - zo)*constants.ANGSTROM_TO_M)/(r**3)
                Ex = Ex + Ex_contrib
                Ey = Ey + Ey_contrib
                Ez = Ez + Ez_contrib
                atom_wise_additions.append([Ex_contrib, Ey_contrib,Ez_contrib ])


        E_vec = [Ex, Ey, Ez]
        QM_charges = charges
                #For QMMM calculation, include point charges in E field calc
        if self.config['includePtChgs']:
            qm_charges = charges
            QM_coords = [xs, ys, zs]
            #check if self.ptchf is population
            if df_ptchg is None:
                # If point charges are enabled but not passed, try to load them
                # Note: This may fail if ptChgfp is not a full path
                df_ptchg = self.getPtChgs(self.config['ptChgfp'])
            MM_xs = list(df_ptchg['x'])
            MM_ys = list(df_ptchg['y'])
            MM_zs = list(df_ptchg['z'])
            init_MM_charges = list(df_ptchg['charge'])

            #make new MM_charges by scaling!
            mm_coords = np.column_stack((np.array(MM_xs), np.array(MM_ys), np.array(MM_zs)))
            mm_charges = np.array(init_MM_charges)
            qm_charges = np.array(QM_charges)
            #MM_charges = Electrostatics.correct_mm_charges(QM_coords, qm_charges, mm_coords, mm_charges)

            MM_charges = mm_charges*(1/(math.sqrt(self.config['dielectric_scale'])))
            #change charge range to include these new partial charges!
            charge_range = range(0, len(MM_xs))
            for chg_idx in charge_range:
                r = (((MM_xs[chg_idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((MM_ys[chg_idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((MM_zs[chg_idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                dist_vec = constants.ANGSTROM_TO_M*np.array([(MM_xs[chg_idx] - xo), (MM_ys[chg_idx] - yo), (MM_zs[chg_idx] - zo)])
                E_to_add = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*MM_charges[chg_idx]*(1/(r**3))*dist_vec
                E_vec[0] += E_to_add[0]
                E_vec[1] += E_to_add[1]
                E_vec[2] += E_to_add[2]
                # Append point charge contribution to atom-wise list
                atom_wise_additions.append(E_to_add)

        return [constants.VM_TO_VA*np.array(E_vec), position_vec, df['Atom'][idx_atom], constants.VM_TO_VA*np.array(atom_wise_additions)]

    def pointcharge_atomicEfield(self, espatom_idx, charge_range, df_charges):
        """
        Calculate E-field using point charges from a DataFrame only.

        Parameters
        ----------
        espatom_idx : int
            Atom index (0-indexed) where the E-field is calculated.
        charge_range : list or range
            Atom indices to include in the E-field calculation.
        df_charges : pd.DataFrame
            DataFrame with columns: ['Atom', 'x', 'y', 'z', 'charge'].

        Returns
        -------
        list
            [E_vec, position_vec, atom_symbol, atom_wise_contributions]
            E_vec is in V/Angstrom.
        """
        inv_eps = 1/(self.config['dielectric'])

        charges = df_charges['charge']
        xs = df_charges['x']
        ys = df_charges['y']
        zs = df_charges['z']

        idx_atom = espatom_idx

        xo = xs[idx_atom]
        yo = ys[idx_atom]
        zo = zs[idx_atom]
        position_vec = constants.ANGSTROM_TO_M*np.array([xo, yo, zo])
        Ex = 0
        Ey = 0
        Ez = 0
        atom_wise_additions = []

        for idx in range(0, len(xs)):
            if idx == idx_atom:
                atom_wise_additions.append([0, 0, 0])
                continue
            if idx not in charge_range:
                atom_wise_additions.append([0, 0, 0])
                continue

            r = (((xs[idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((ys[idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((zs[idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
            Ex_contrib = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx])*((xs[idx] - xo)*constants.ANGSTROM_TO_M)/(r**3)
            Ey_contrib = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx])*((ys[idx] - yo)*constants.ANGSTROM_TO_M)/(r**3)
            Ez_contrib = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(charges[idx])*((zs[idx] - zo)*constants.ANGSTROM_TO_M)/(r**3)
            Ex = Ex + Ex_contrib
            Ey = Ey + Ey_contrib
            Ez = Ez + Ez_contrib
            atom_wise_additions.append([Ex_contrib, Ey_contrib, Ez_contrib])

        E_vec = [Ex, Ey, Ez]
        return [constants.VM_TO_VA*np.array(E_vec), position_vec, df_charges['Atom'][idx_atom], constants.VM_TO_VA*np.array(atom_wise_additions)]

    #Helper functions for resp-based adjustement of MD point charges

    def compute_esp(q_coords, q_vals, probe_coords):
        """
        Compute ESP at probe_coords due to point charges at q_coords.
        """
        diff = probe_coords[:, np.newaxis, :] - q_coords[np.newaxis, :, :] 
        dists = np.linalg.norm(diff, axis=2)  # shape: (P, Q)
        mask = dists > 1e-8 
        # Avoid division by zero by masking
        inv_dists = np.zeros_like(dists)
        inv_dists[mask] = 1.0 / dists[mask]
        
        esp = np.dot(inv_dists, q_vals) 
        return esp
    def compute_esp_from_qm(qm_coords, qm_charges, mm_coords):
        '''
        Compute the ESP at MM atoms due to QM charges.
        '''
        return Electrostatics.compute_esp(qm_coords, qm_charges, mm_coords)


    def multipole_atomicEfield(self, idx_atom, charge_range, xyz_file, atom_multipole_file, df_ptchg=None):
        '''

        Input: idx_atom: integer of atom index
        charge_range: list of integers of atom indices
        xyz_file: string of xyz filename
        atom_multipole_file: string of multipole filename
        df_ptchg: optional pre-loaded point charge dataframe
        Output: list of E-field vector and atomic symbol, Efields are reported in units of Volts/Angstrom
        '''
        df = Geometry(xyz_file).getGeomInfo()
        #df = pd.read_csv(charge_file, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])

        # Convert each column to list for quicker indexing
        xs = df['X']
        ys = df['Y']
        zs = df['Z']

        
        # Following derivation of E-field strength 
        Monopole_E = np.array([0, 0, 0])

        # Unit conversion
        # Bohr to meters (atomic units)
        b_to_A = 0.529177
        Ex = 0
        Ey = 0
        Ez = 0

        xo = xs[idx_atom]
        yo = ys[idx_atom]
        zo = zs[idx_atom]

        position_vec = constants.ANGSTROM_TO_M*np.array([xo, yo, zo])

        inv_eps = 1/self.config['dielectric']

        if self.config['includePtChgs'] and df_ptchg is None:
            # If point charges are enabled but not passed, try to load them
            # Note: This may fail if ptChgfp is not a full path
            df_ptchg = self.getPtChgs(self.config['ptChgfp'])

        #load multipole moments from processed outputs
        lst_multipole_dict = MultiwfnInterface.getmultipoles(atom_multipole_file)

        #make a list for each term in multipole expansion
        # Create mapping: atom_index (0-indexed) -> multipole_dict
        # Note: Multiwfn uses 1-indexed atoms, so we subtract 1 for 0-indexed Python arrays
        # However, if the file is already 0-indexed, the Index field will be "0", "1", etc.
        # We'll use enumerate to ensure proper mapping regardless
        multipole_dict_by_idx = {idx: atom for idx, atom in enumerate(lst_multipole_dict)}

        # Create charge list in proper order
        QM_charges = [atom['Atom_Charge'] for atom in lst_multipole_dict]
        QM_coords = np.column_stack((np.array(xs), np.array(ys), np.array(zs)))
        #This ensure that atoms we would like to exclude from Efield calculation are excluded from every term in the multipole expansion
        # Use actual atom indices that exist in both geometry and multipole data
        multipole_chg_range = [i for i in range(len(lst_multipole_dict)) if i in charge_range]
        multipole_Efield_contribution = []
        for idx in multipole_chg_range:
            #If the atom idx is outside of the charge range then skip
            atom_dict = multipole_dict_by_idx[idx]
            if idx == idx_atom:
                # Append zero contribution for self (maintains consistent array size)
                multipole_Efield_contribution.append([0.0, 0.0, 0.0])
                continue
            else:
                # Units of dipole_vec:
                dipole_vec = constants.BOHR_TO_M*np.array(atom_dict["Dipole_Moment"])
                #convention of vector pointing towards atom of interest, so positive charges exert positive Efield
                dist_vec = constants.ANGSTROM_TO_M*np.array([(xs[idx] - xo), (ys[idx] - yo), (zs[idx] - zo)])
                quadrupole_arr = constants.BOHR_TO_M*constants.BOHR_TO_M*atom_dict['Quadrupole_Moment']

                # Calculate esp and convert to units (A to m); Calc E-field stenth in kJ/mol*e*m
                r = (((xs[idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((ys[idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((zs[idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                r_squared = r**2
                Monopole_E = Monopole_E - inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*atom_dict["Atom_Charge"]*(1/(r**3))*dist_vec  #neg for sign convention so Efield points toward neg charge

                # Calculate full multipole contribution using proper tensor contractions
                # Monopole term: -(k*e/ε_r) * q_j * r_ij / r³
                E_monopole_atomic = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*atom_dict["Atom_Charge"]*(1/(r**3))*dist_vec

                # Dipole term: (k*e/ε_r) * [3(μ_j·r_ij)*r_ij - r²*μ_j] / r⁵
                mu_dot_r = np.dot(dipole_vec, dist_vec)
                E_dipole_atomic = inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(1/(r**5))*(3*mu_dot_r*dist_vec - r_squared*dipole_vec)

                # Quadrupole term: (k*e/ε_r) * [-5*(r_ij^T Q_j r_ij)*r_ij + 2*r²*(Q_j·r_ij)] / r⁷
                Qr = np.dot(quadrupole_arr, dist_vec)  # Matrix-vector product: Q·r
                rTQr = np.dot(dist_vec, Qr)  # Scalar: r^T Q r = r·(Q·r)
                E_quad_atomic = inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*(1/r**7)*(-5*rTQr*dist_vec + 2*r_squared*Qr)

                # Total atomic contribution (all three multipole terms)
                E_atomic_contrib_vec = E_monopole_atomic + E_dipole_atomic + E_quad_atomic

                E_x_atomic_contrib = E_atomic_contrib_vec[0]
                E_y_atomic_contrib = E_atomic_contrib_vec[1]
                E_z_atomic_contrib = E_atomic_contrib_vec[2]

                Ex = Ex + E_x_atomic_contrib
                Ey = Ey + E_y_atomic_contrib
                Ez = Ez + E_z_atomic_contrib
                # Append individual atomic contribution (not cumulative)
                multipole_Efield_contribution.append([E_x_atomic_contrib, E_y_atomic_contrib, E_z_atomic_contrib])

        E_vec = [Ex, Ey, Ez]


        #For QMMM calculation, include point charges in E field calc
        if self.config['includePtChgs']:
            MM_xs = list(df_ptchg['x'])
            MM_ys = list(df_ptchg['y'])
            MM_zs = list(df_ptchg['z'])
            init_MM_charges = list(df_ptchg['charge'])
 
            #make new MM_charges by scaling!
            mm_coords = np.column_stack((np.array(MM_xs), np.array(MM_ys), np.array(MM_zs)))
            mm_charges = np.array(init_MM_charges)
            qm_charges = np.array(QM_charges)
            #MM_charges = Electrostatics.correct_mm_charges(QM_coords, qm_charges, mm_coords, mm_charges)

            MM_charges = mm_charges*(1/(math.sqrt(self.config['dielectric_scale'])))
            #change charge range to include these new partial charges!
            charge_range = range(0, len(MM_xs))
            for chg_idx in charge_range:
                r = (((MM_xs[chg_idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((MM_ys[chg_idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((MM_zs[chg_idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                dist_vec = constants.ANGSTROM_TO_M*np.array([(MM_xs[chg_idx] - xo), (MM_ys[chg_idx] - yo), (MM_zs[chg_idx] - zo)])
                E_to_add = -inv_eps*constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*MM_charges[chg_idx]*(1/(r**3))*dist_vec
                Monopole_E = Monopole_E +  E_to_add
                E_vec[0] += E_to_add[0]
                E_vec[1] += E_to_add[1]
                E_vec[2] += E_to_add[2]
                multipole_Efield_contribution.append(E_to_add)
            #Add contributions to Monopole E from point charges to total E
        return [constants.VM_TO_VA*np.array(E_vec), position_vec, df['Atom'][idx_atom], constants.VM_TO_VA*np.array(Monopole_E), constants.VM_TO_VA*np.array(multipole_Efield_contribution)]

    def multipole_esp(self, xyzfilepath, atom_multipole_file, charge_range, idx_atom, ptchg_path=None):
        '''

        Input: idx_atom: integer of atom index
        charge_range: list of integers of atom indices
        xyfilepath: string of xyz filename
        atom_multipole_file: string of multipole filename
        ptchg_path: optional path to point charge file
        Output: list of ESP and atomic symbol Units of ESP is Volts!
        '''
        df = Geometry(xyzfilepath).getGeomInfo()
        #df = pd.read_csv(charge_file, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])

        # Convert each column to list for quicker indexing
        atoms = list(df['Atom'])
        xs = list(df['X'])
        ys = list(df['Y'])
        zs = list(df['Z'])

        # Determine position and charge of the target atom
        xo = xs[idx_atom]
        yo = ys[idx_atom]
        zo = zs[idx_atom]

         # Unit conversion
        #KJ_J = 10**-3
        #faraday = 23.06   #kcal/(mol*V)
        one_mol = 6.02*(10**23)
        #cal_J = 4.184
        dielectric = self.config['dielectric']
        lst_multipole_dict = MultiwfnInterface.getmultipoles(atom_multipole_file)

        # For QMMM calculation, include point charges in ESP calculation
        # Point charges are treated as monopoles only (no dipoles/quadrupoles)
        if ptchg_path is not None:
            df_ptchg = self.getPtChgs(ptchg_path)
            # Store original number of QM atoms
            n_qm_atoms = len(xs)
            # Extend geometry lists with point charges
            xs = xs + list(df_ptchg['x'])
            ys = ys + list(df_ptchg['y'])
            zs = zs + list(df_ptchg['z'])
            atoms = atoms + list(df_ptchg['Atom'])
            # Add monopole-only entries to multipole dict for point charges
            for i, (charge_val, x, y, z) in enumerate(zip(df_ptchg['charge'], df_ptchg['x'], df_ptchg['y'], df_ptchg['z'])):
                ptchg_dict = {
                    'Atom_Charge': charge_val,
                    'X': x,
                    'Y': y,
                    'Z': z
                }
                lst_multipole_dict.append(ptchg_dict)
            # Extend charge_range to include point charges
            charge_range = range(0, len(xs))

        #create list of bound atoms, these are treated with a different dielectric
        bound_atoms = Geometry(xyzfilepath).getBondedAtoms(idx_atom)
        total_esp = 0
        for idx in charge_range:
            atom_dict = lst_multipole_dict[idx]
            if idx == idx_atom:
                continue

            # r_ij = r_i - r_j (target - source), in meters
            dist_vec = np.array([
                (xo - xs[idx]) * constants.ANGSTROM_TO_M,
                (yo - ys[idx]) * constants.ANGSTROM_TO_M,
                (zo - zs[idx]) * constants.ANGSTROM_TO_M
            ])
            r = np.linalg.norm(dist_vec)
            inv_eps = 1.0 if idx in bound_atoms else 1.0 / dielectric

            # Monopole term: q / R
            V_atomic = atom_dict["Atom_Charge"] / r

            # Dipole term: -μ · r_ij / R³
            if 'Dipole_Moment' in atom_dict:
                dipole_vec = constants.BOHR_TO_M * np.array(atom_dict["Dipole_Moment"])
                V_atomic -= np.dot(dipole_vec, dist_vec) / r**3

            # Quadrupole term: (1/2) * r_ij^T Θ r_ij / R^5
            if 'Quadrupole_Moment' in atom_dict:
                quadrupole_arr = constants.BOHR_TO_M**2 * np.array(atom_dict['Quadrupole_Moment'])
                rTQr = np.dot(dist_vec, np.dot(quadrupole_arr, dist_vec))
                V_atomic += 0.5 * rTQr / r**5

            total_esp += inv_eps * V_atomic

        final_esp = constants.COULOMB_CONSTANT * total_esp * constants.ELEMENTARY_CHARGE  #Units: N*m^2/(C^2)*(C/m) = N*m/C = J/C = Volt
        return [final_esp, df['Atom'][idx_atom] ]
 

 # Bond_indices is a list of tuples where each tuple contains the zero-indexed values of location of the atoms of interest
    def calc_bond_dipoles(self, bond_indices, xyz_filepath, atom_multipole_file, bool_multipole, charge_df=None):
        """Calculate bond dipole moments for given bond indices.

        For each bond between atoms A and B, calculates the bond dipole moment vector
        relative to the geometric center (midpoint) of the bond. This is origin-independent
        for both neutral and charged bonds:
        μ_bond = q_A * (r_A - r_center) + q_B * (r_B - r_center)
        which simplifies to: μ_bond = (q_A - q_B) * (r_A - r_B) / 2

        Parameters
        ----------
        bond_indices : list of tuples
            List of (atomA_idx, atomB_idx) tuples specifying bonds
        xyz_filepath : str
            Path to XYZ structure file
        atom_multipole_file : str
            Path to multipole/charge file
        bool_multipole : bool
            If True, use multipole file; if False, use monopole charges
        charge_df : pd.DataFrame or None
            Optional DataFrame with monopole charges (columns: Atom, x, y, z, charge).
            If provided, overrides charge file input.

        Returns
        -------
        list
            List of bond dipole information dictionaries containing:
            - 'bond_dipole_vec': [x, y, z] dipole vector in Debye
            - 'bond_dipole_mag': magnitude of dipole in Debye
            - 'bond_indices': (atomA_idx, atomB_idx)
            - 'charges': (charge_A, charge_B)
        """
        bond_dipoles = []

        # Get geometry information
        # In pdb_only mode xyz_filepath is None; use coordinates already in charge_df.
        if xyz_filepath is None and charge_df is not None:
            df_geom = charge_df[['x', 'y', 'z']].rename(
                columns={'x': 'X', 'y': 'Y', 'z': 'Z'}
            ).reset_index(drop=True)
        else:
            df_geom = Geometry(xyz_filepath).getGeomInfo()

        # Get charge information
        if charge_df is not None:
            charge_dict = {idx: charge for idx, charge in enumerate(charge_df['charge'])}
        elif bool_multipole:
            multipole_data = MultiwfnInterface.getmultipoles(atom_multipole_file)
            # Use enumerate for proper 0-indexed mapping, regardless of Index field value
            charge_dict = {idx: atom['Atom_Charge'] for idx, atom in enumerate(multipole_data)}
        else:
            df_charges = pd.read_csv(atom_multipole_file, sep='\s+', names=["Atom", 'x', 'y', 'z', "charge"])
            charge_dict = {idx: charge for idx, charge in enumerate(df_charges['charge'])}

        for atomA_idx, atomB_idx in bond_indices:
            # Get positions (in Angstrom from XYZ file)
            posA = np.array([df_geom.loc[atomA_idx, 'X'],
                           df_geom.loc[atomA_idx, 'Y'],
                           df_geom.loc[atomA_idx, 'Z']])
            posB = np.array([df_geom.loc[atomB_idx, 'X'],
                           df_geom.loc[atomB_idx, 'Y'],
                           df_geom.loc[atomB_idx, 'Z']])

            # Get charges (in elementary charge units)
            chargeA = charge_dict[atomA_idx]
            chargeB = charge_dict[atomB_idx]

            # Calculate bond dipole relative to geometric center (midpoint)
            # This makes the dipole origin-independent for both neutral and charged bonds
            # and consistent with line integral formalism
            r_center = (posA + posB) / 2
            dipole_vec_eA = chargeA * (posA - r_center) + chargeB * (posB - r_center)  # in e*Angstrom
            # Simplified: dipole_vec = (chargeA - chargeB) * (posA - posB) / 2

            # Convert to Debye (1 e*Angstrom = 4.80320427 Debye)
            convert_to_Debye = 4.80320427
            dipole_vec = convert_to_Debye * dipole_vec_eA  # in Debye
            dipole_mag = np.linalg.norm(dipole_vec)  # in Debye

            bond_dipole_info = {
                'bond_dipole_vec': dipole_vec.tolist(),
                'bond_dipole_mag': dipole_mag,
                'bond_indices': (atomA_idx, atomB_idx),
                'charges': (chargeA, chargeB)
            }
            bond_dipoles.append(bond_dipole_info)

        return bond_dipoles

    # Bond_indices is a list of tuples where each tuple contains the zero-indexed values of location of the atoms of interest
    def bondEfield(self, bond_indices, xyz_filepath, atom_multipole_file, all_lines, bool_multipole, df_ptchg=None):
        '''
        Input: bond_indices: list of tuples of atom indices
        xyz_filepath: string of xyz filename
        atom_multipole_file: string of multipole filename
        all_lines: list of strings of lines in xyz file
        df_ptchg: optional pre-loaded point charge dataframe
        Output: list of E-field vector and atomic symbol in units of V/Angstrom
        '''
        bonded_atoms = []
        E_projected = []
        E_monopole_proj = []
        bonded_positions = []
        # Determine the Efield vector at point of central metal stom
        bond_lens = []
        E_proj_atomwise_list = []  # List to store per-bond atom-wise contributions

        if bool_multipole:
            for atomidxA, atomidxB in bond_indices:
                [A_bonded_E, A_bonded_position, A_bonded_atom, A_monopole_E_bonded, A_atom_wise_E]  =  self.multipole_atomicEfield(atomidxA, all_lines, xyz_filepath, atom_multipole_file, df_ptchg=df_ptchg)
                [B_bonded_E, B_bonded_position, B_bonded_atom, B_monopole_E_bonded, B_atom_wise_E]  =  self.multipole_atomicEfield(atomidxB, all_lines, xyz_filepath, atom_multipole_file, df_ptchg=df_ptchg)
                bond_vec_unnorm = np.subtract(np.array(A_bonded_position), np.array(B_bonded_position))
                bond_len = np.linalg.norm(bond_vec_unnorm)
                bond_vec = bond_vec_unnorm/(bond_len)
                # Initialized a bond_dipole_vec as the (bond_vec_unnorm )*(sum of the partial charges).. can just use dipole!
                # Compute E-field projected along this bond!
                E_proj = (1/2)*np.dot((np.array(A_bonded_E) + np.array(B_bonded_E)), bond_vec)
                E_proj_monopole = (1/2)*np.dot((np.array(A_monopole_E_bonded) + np.array(B_monopole_E_bonded)), bond_vec)
                E_projected.append(E_proj)
                E_monopole_proj.append(E_proj_monopole)
                bonded_atoms.append((A_bonded_atom, B_bonded_atom))
                bonded_positions.append((A_bonded_position, B_bonded_position))
                bond_lens.append(bond_len)
                E_proj_atomwise = (1/2)*(np.array(A_atom_wise_E) + np.array(B_atom_wise_E))@ bond_vec.T

                E_proj_atomwise_list.append(E_proj_atomwise)

        else:
            for atomidxA, atomidxB in bond_indices:
                [A_bonded_E, A_bonded_position, A_bonded_atom,  A_atom_wise_E]  =  self.monopole_atomicEfield(atomidxA, all_lines, atom_multipole_file, df_ptchg=df_ptchg)
                [B_bonded_E, B_bonded_position, B_bonded_atom, B_atom_wise_E]  =  self.monopole_atomicEfield(atomidxB, all_lines, atom_multipole_file, df_ptchg=df_ptchg)

                bond_vec_unnorm = np.subtract(np.array(A_bonded_position), np.array(B_bonded_position))
                bond_len = np.linalg.norm(bond_vec_unnorm)
                bond_vec = bond_vec_unnorm/(bond_len)
                # Initialized a bond_dipole_vec as the (bond_vec_unnorm )*(sum of the partial charges).. can just use dipole!
                # Compute E-field projected along this bond!
                E_proj = (1/2)*np.dot((np.array(A_bonded_E) + np.array(B_bonded_E)), bond_vec)
                E_projected.append(E_proj)
                bonded_atoms.append((A_bonded_atom, B_bonded_atom))
                bonded_positions.append((A_bonded_position, B_bonded_position))
                bond_lens.append(bond_len)

                E_proj_atomwise = (1/2)*((np.array(A_atom_wise_E) + np.array(B_atom_wise_E))@ bond_vec.T)

                E_proj_atomwise_list.append(E_proj_atomwise)

        # For backwards compatibility, return the last bond's atomwise data as the 5th element
        # and return the full list as the 6th element
        E_proj_atomwise = E_proj_atomwise_list[-1] if E_proj_atomwise_list else None
        return [E_projected, bonded_atoms, bond_indices, bond_lens, E_proj_atomwise, E_proj_atomwise_list]

    def bondEfield_pointcharges(self, bond_indices, df_charges, all_lines):
        """
        Compute bond-projected E-fields using point charges only.

        Parameters
        ----------
        bond_indices : list of tuples
            List of (atomA_idx, atomB_idx) tuples specifying bonds.
        df_charges : pd.DataFrame
            DataFrame with columns: ['Atom', 'x', 'y', 'z', 'charge'].
        all_lines : list
            Atom indices to include in the E-field calculation.

        Returns
        -------
        list
            [E_projected, bonded_atoms, bond_indices, bond_lens, E_proj_atomwise, E_proj_atomwise_list]
        """
        bonded_atoms = []
        E_projected = []
        bonded_positions = []
        bond_lens = []
        E_proj_atomwise_list = []

        for atomidxA, atomidxB in bond_indices:
            [A_bonded_E, A_bonded_position, A_bonded_atom, A_atom_wise_E] = self.pointcharge_atomicEfield(
                atomidxA, all_lines, df_charges
            )
            [B_bonded_E, B_bonded_position, B_bonded_atom, B_atom_wise_E] = self.pointcharge_atomicEfield(
                atomidxB, all_lines, df_charges
            )

            bond_vec_unnorm = np.subtract(np.array(A_bonded_position), np.array(B_bonded_position))
            bond_len = np.linalg.norm(bond_vec_unnorm)
            bond_vec = bond_vec_unnorm/(bond_len)

            E_proj = (1/2)*np.dot((np.array(A_bonded_E) + np.array(B_bonded_E)), bond_vec)
            E_projected.append(E_proj)
            bonded_atoms.append((A_bonded_atom, B_bonded_atom))
            bonded_positions.append((A_bonded_position, B_bonded_position))
            bond_lens.append(bond_len)

            E_proj_atomwise = (1/2)*((np.array(A_atom_wise_E) + np.array(B_atom_wise_E)) @ bond_vec.T)
            E_proj_atomwise_list.append(E_proj_atomwise)

        E_proj_atomwise = E_proj_atomwise_list[-1] if E_proj_atomwise_list else None
        return [E_projected, bonded_atoms, bond_indices, bond_lens, E_proj_atomwise, E_proj_atomwise_list]


    def esp_coord_shell(self, metal_idx, charge_file, path_to_xyz, n_shells=1):
        """Calculate ESP accounting for electrostatic contributions from n coordination shells.

        This unified method replaces esp_first_coord() and esp_second_coord() by accepting
        a parameter to specify how many coordination shells to include.

        Parameters
        ----------
        metal_idx : int
            Atom index of the central metal
        charge_file : str
            Path to charge filename
        path_to_xyz : str
            Path to xyz structure file
        n_shells : int, optional
            Number of coordination shells to include (default=1)
            n_shells=1: first coordination shell only
            n_shells=2: first and second coordination shells

        Returns
        -------
        float
            ESP value for the specified coordination shell(s)
        """
        print(f'The index of the metal atom is: {metal_idx}')

        # Build list of atoms to include based on n_shells
        atoms_to_include = []

        if n_shells >= 1:
            # First coordination shell: atoms directly bonded to metal
            first_shell_atoms = self.getBondedAtoms(path_to_xyz, metal_idx)
            atoms_to_include.extend(first_shell_atoms)

        if n_shells >= 2:
            # Second coordination shell: atoms bonded to first shell atoms
            for first_shell_atom in first_shell_atoms:
                second_shell_atoms = self.getBondedAtoms(path_to_xyz, first_shell_atom)
                atoms_to_include.extend(second_shell_atoms)

        if n_shells >= 3:
            # Third and higher coordination shells can be added here if needed
            raise NotImplementedError(f"Coordination shells beyond n=2 are not yet implemented (requested n={n_shells})")

        # Remove duplicates while preserving order
        unique_atoms = list(dict.fromkeys(atoms_to_include))

        # Calculate ESP
        [coord_shell_ESP, atom_type] = self.monopole_esp(path_to_xyz, metal_idx, unique_atoms, charge_file)
        return coord_shell_ESP


    def charge_atom(filename, atom_idx):
        '''Compute charges and return total charge and partial charge of atom
        Input: filename: string of charge filename
        atom_idx: integer of atom index
        Output: list of total charge and partial charge of atom'''
        df = pd.read_csv(filename, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])
        atoms = df['Atom']
        charges = df['charge']
        xs = df['x']
        ys = df['y']
        zs = df['z']
        total_charge = np.sum(df['charge'])
        partial_charge_atom = charges[atom_idx]
        return [total_charge, partial_charge_atom]

    #This just gets the charge if the charge is in monopole setup
    def charge_atoms(self, chg_filename, xyzfilename):
        '''Input: filename: string of charge filename or multipole filename... if the xyzfilename is included must be multipole! '''
        if len(xyzfilename) > 1:
            dict_multipole = MultiwfnInterface.getmultipoles(chg_filename)  #Index, Element, Atom_Charge, Dipole_Moment....
            df_multipole = pd.DataFrame(dict_multipole)
            df_geom = Geometry(xyzfilename).getGeomInfo()
            merged_df = pd.merge(df_geom, df_multipole, left_index=True, right_index=True)
            merged_df['charge'] = merged_df['Atom_Charge']
            merged_df['x'] = merged_df['X']
            merged_df['y'] = merged_df['Y']
            merged_df['z'] = merged_df['Z']
            df = merged_df[["Atom",'x', 'y', 'z', "charge"]]
        else:
            df = pd.read_csv(chg_filename, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])
        return df


    def getAtomInfo(filename, atom_idx):
        '''
        Input: filename: string of charge filename
        atom_idx: integer of atom index
        Output: list of total charge and partial charge of atom'''
        df = pd.read_csv(filename, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])
        atoms = df['Atom']
        charges = df['charge']
        xs = df['x']
        ys = df['y']
        zs = df['z']
        total_charge = np.sum(df['charge'])
        partial_charge_atom = charges[atom_idx]
        atom_type = atoms[atom_idx]
        return [atom_type, partial_charge_atom]


    def getAtomsInfo(filename, atom_indices):
        ''' Input: filename: string of charge filename
        atom_indices: list of integers of atom indices
        Output: list of total charge and partial charge of atom'''
        df = pd.read_csv(filename, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])
        atoms = df['Atom']
        charges = df['charge']
        xs = df['x']
        ys = df['y']
        zs = df['z']
        total_charge = np.sum(df['charge'])
        res_charge = 0
        for atm_idx in atom_indices:
            print(f'Atom index: {atm_idx} and charge: {charges[atm_idx]}')
            res_charge = res_charge + charges[atm_idx]
        return [res_charge]


    def esp_bydistance(self, path_to_xyz, espatom_idx,  charge_file):
        '''
        Input: path_to_xyz: string of xyz filename
        espatom_idx: integer of atom index
        charge_file: string of charge filename
        Output: list of ESP and atomic symbol
        '''
        dielectric = self.config['dielectric']
        df = pd.read_csv(charge_file, sep='\s+', names=["Atom",'x', 'y', 'z', "charge"])

        # Unit conversion
        KJ_J = 10**-3
        faraday = 23.06   #kcal/(mol*V)
        one_mol = 6.02*(10**23)
        cal_J = 4.184

        # Convert each column to list for quicker indexing
        atoms = list(df['Atom'])
        charges = list(df['charge'])
        xs = list(df['x'])
        ys = list(df['y'])
        zs = list(df['z'])

        # Pick the index of the atom at which the esp should be calculated
        idx_atom = espatom_idx

        # Determine position and charge of the target atom
        xo = xs[idx_atom]
        yo = ys[idx_atom]
        zo = zs[idx_atom]
        chargeo = charges[idx_atom]

        #For QMMM calculation, include point charges in ESP calculation
        if self.config['includePtChgs']:
            ptchg_filename = self.config['ptChgfp']
            if not ptchg_filename or ptchg_filename.strip() == '':
                raise ValueError(
                    "Point charge file path is not set. Please call set_ptChgfile() "
                    "with the filename (e.g., 'ptchg.dat') before using includePtChgs=True"
                )
            init_file_path = path_to_xyz[0:-len(self.folder_to_file_path + self.config['xyzfilename'])]
            full_ptchg_fp = init_file_path + ptchg_filename
            df_ptchg = self.getPtChgs(full_ptchg_fp)
            xs = xs + list(df_ptchg['x'])
            ys = ys + list(df_ptchg['y'])
            zs = zs + list(df_ptchg['z'])
            charges = charges + list(df_ptchg['charge'])
            atoms = atoms + list(df_ptchg['Atom'])

        total_esp = 0
        # Create an ordering of the atoms based on distance from the central atom
        total_atoms = len(xs)
        distances = []
        esps = []
        bound_atoms = []
        if self.config['changeDielectBoundBool']:
            bound_atoms = self.getBondedAtoms(path_to_xyz, idx_atom)
        for idx in range(0, total_atoms):
            if idx == idx_atom:
                continue
            elif idx in bound_atoms:
                r = (((xs[idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((ys[idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((zs[idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                esps.append(constants.COULOMB_CONSTANT*constants.ELEMENTARY_CHARGE*charges[idx]/r) #units of N*m/C =Volt
                distances.append(r)
            else:
                r = (((xs[idx] - xo)*constants.ANGSTROM_TO_M)**2 + ((ys[idx] - yo)*constants.ANGSTROM_TO_M)**2 + ((zs[idx] - zo)*constants.ANGSTROM_TO_M)**2)**(0.5)
                distances.append(r)
                esps.append(constants.COULOMB_CONSTANT*(1/dielectric)*constants.ELEMENTARY_CHARGE*charges[idx]/r) #untis of N*m/C = Volt
        # Now we sort the distance list, and use sorted indices to sort the
        atm_lst = list(atoms)
        atm_lst.pop(idx_atom)

        chg_lst = list(charges)
        chg_lst.pop(idx_atom)
        chg_arr = np.array(chg_lst)
        
        dist_arr = np.array(distances)
        init_sorted_idx = np.argsort(dist_arr)
        lst_sorted_idx = list(init_sorted_idx)
        lst_sorted_idx.append(len(lst_sorted_idx))
        lst_sorted_idx.remove(idx_atom)
        sorted_idx = np.array(lst_sorted_idx)

        esp_arr = np.array(esps)
        sorted_esps = esp_arr.take(init_sorted_idx)
        cumulative_esps = np.cumsum(sorted_esps)

        sorted_dist = dist_arr.take(init_sorted_idx)
        sorted_partial_charges = chg_arr.take(init_sorted_idx)
        sorted_atomTypes = [atm_lst[i] for i in init_sorted_idx]
       
        return [sorted_dist, sorted_esps, cumulative_esps, sorted_idx, sorted_partial_charges, sorted_atomTypes]
  

        # Current API uses molden_paths/xyz_paths; folders are derived from those paths.

    # ============================================================================
    # CONSOLIDATED ESP AND E-FIELD METHODS
    # These unified methods replace the previous separate implementations
    # ============================================================================

    def getESP(self, charge_types='CM5', ESPdata_filename='ESP', multiwfn_path='None',
               use_multipole=False, include_decay=False, include_coord_shells=False,
               visualize=None, dielectric=1, dielectric_scale=1, includePtChgs=None, ptchg_paths=None,
               pdb_charge_paths=None):
        """Unified ESP calculation method supporting multiple modes.

        This consolidated method replaces getESPData(), getESPMultipole(), and getESPDecay()
        to eliminate code duplication while preserving all functionality.

        Parameters
        ----------
        charge_types : list of str or str
            Charge partitioning scheme(s). Options: 'Hirshfeld', 'Voronoi', 'Mulliken',
            'Lowdin', 'SCPA', 'Becke', 'ADCH', 'CHELPG', 'MK', 'AIM', 'Hirshfeld_I',
            'CM5', 'EEM', 'RESP', 'PEOE'. Can be a single string (for multipole mode)
            or list of strings.
        ESPdata_filename : str
            Output CSV filename (without extension).
        multiwfn_path : str
            Path to Multiwfn executable.
            Ignored when pdb_charge_paths is provided.
        use_multipole : bool, optional
            If True, use multipole expansion; if False, use monopole charges (default: False).
        include_decay : bool, optional
            If True, include distance-sorted ESP decay analysis (default: False).
        include_coord_shells : bool, optional
            If True, include first and second coordination shell ESP (default: False).
        visualize : bool or None, optional
            If True, create PDB file with ESP values in B-factor column.
            None uses config defaults (default: None).
        dielectric : float, optional
            Dielectric constant (default: 1).
        dielectric_scale : float, optional
            Scaling factor for point charges. Point charge contributions are scaled by
            1/sqrt(dielectric_scale). Typically used in QM/MM calculations (default: 1).
        includePtChgs : bool or None, optional
            Override object-level includePtChgs setting for this calculation.
            If None, uses object-level setting (default: None).
        ptchg_paths : list of str or None, optional
            Override point charge file paths for this calculation.
            List of paths (one per job), with None for jobs without point charges.
            If provided, takes precedence over object-level settings (default: None).
        pdb_charge_paths : str or list of str or None, optional
            If provided, use partial charges from PDB B-factor columns only (no Multiwfn).
            Can be a single PDB path or a list of paths (one per job), with None to skip a job.
            Overrides the object-level pdb_charge_paths if the object was created in pdb_only mode.
            This mode supports monopole charges only (use_multipole must be False).

        Returns
        -------
        pd.DataFrame
            DataFrame with ESP results saved to CSV.

        Output Files
        ------------
        **CSV file:** Auto-generated with format:
        - Monopole: `esp_{charge_type}.csv`
        - Multipole: `esp_multipole_{charge_type}.csv`
        - Examples: `esp_Hirshfeld_I.csv`, `esp_multipole_Hirshfeld_I.csv`
        - Contains ESP values at metal centers for all jobs
        - Columns: job name, charge type, ESP values, coordinates, etc.

        **PDB file (if visualize=True):** `esp_{charge_type}_{structure}_atom{idx}.pdb`
        - One file per job and metal atom
        - B-factor column contains ESP values at each atom
        - Example: `esp_Hirshfeld_I_complex_atom25.pdb`

        Note: If you provide a custom filename (not 'output', 'esp', etc.),
        your filename will be used instead of automatic naming.

        Examples
        --------
        >>> # Monopole mode (replaces getESPData)
        >>> es.getESP(['Hirshfeld', 'Hirshfeld_I'], 'esp_mono', ...)

        >>> # Multipole mode (replaces getESPMultipole)
        >>> es.getESP('Hirshfeld_I', 'esp_multi', ..., use_multipole=True)

        >>> # Decay analysis mode (replaces getESPDecay)
        >>> es.getESP(['Hirshfeld'], 'esp_decay', ..., include_decay=True, include_coord_shells=True)

        >>> # pdb_only mode (no Multiwfn required)
        >>> es.getESP(pdb_charge_paths=['job1.pdb', 'job2.pdb'])
        """
        # Resolve pdb_charge_paths: function-level arg takes priority, else use object-level
        if pdb_charge_paths is None and self.pdb_only:
            pdb_charge_paths = self.pdb_charge_paths
        use_pdb_charges = pdb_charge_paths is not None

        if use_pdb_charges and use_multipole:
            raise ValueError("pdb_only mode supports monopole charges only. Set use_multipole=False.")

        # Warn if multiwfn_path is not provided (skip warning in pdb_only mode)
        if not use_pdb_charges and (multiwfn_path is None or multiwfn_path == 'None'):
            warnings.warn(
                "multiwfn_path is not set. Multiwfn is required for charge partitioning. "
                "Consider using pyef.utility.get_openbabel_charges() as an alternative for "
                "simpler charge models (Gasteiger, MMFF94, etc.). "
                "Usage: from pyef.utility import get_openbabel_charges",
                UserWarning,
                stacklevel=2
            )

        # Ensure charge_types is a list
        if isinstance(charge_types, str):
            charge_types = [charge_types]

        # Filter charge types based on multipole/monopole mode
        if use_multipole:
            charge_types = filter_charge_types_for_multipole(charge_types, context="getESP")
        else:
            charge_types = filter_charge_types_for_monopole(charge_types, context="getESP")

        # Return empty DataFrame if no valid charge types remain
        if not charge_types:
            return pd.DataFrame()

        # Validate that metal indices are provided for ESP calculation
        if not self.esp_atom_idx:
            raise ValueError(f"""
{'='*60}
ERROR: ESP calculation requires metal atom indices
{'='*60}
Metal atom indices (esp_atom_idx) were not provided during
initialization of the Electrostatics object.

Correct usage:
  estat = Electrostatics(
      molden_paths=['job1.molden', 'job2.molden'],
      xyz_paths=['job1.xyz', 'job2.xyz'],
      esp_atom_idx=[25, 30],  # Metal indices (0-indexed)
      dielectric=4.0
  )
  esp_df = estat.getESP(['Hirshfeld_I'], 'output', multiwfn_path)

Note: Atom indices are 0-based (first atom = index 0)

For complete examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
- README.md: Section 3.2 (ESP Calculation)
{'='*60}
""")

        self.config['dielectric'] = dielectric
        self.config['dielectric_scale'] = dielectric_scale
        metal_idxs = self.esp_atom_idx
        list_of_file = self.lst_of_folders
        final_structure_file = self.config['xyzfilename']
        # Handle xyzfilename being a list (take first element)
        if isinstance(final_structure_file, list):
            final_structure_file = final_structure_file[0]

        # Handle visualization settings
        if visualize is None:
            viz_esp = self.config.get('visualize_esp', False)
        else:
            viz_esp = visualize

        owd = os.getcwd()
        allspeciesdict = []
        counter = 0

        for f in list_of_file:
            # Convert f to string if it's not already (handles int folder names)
            f = str(f)
            print(f'-----------------{f}------------------')
            atom_idx = metal_idxs[counter]
            results_dict = {}
            results_dict['Name'] = f
            # Use stored absolute path instead of constructing it
            file_path_xyz = self.xyz_paths[counter]

            # pdb_only: resolve and load PDB charges for this job
            df_pdb_charges = None
            if use_pdb_charges:
                if isinstance(pdb_charge_paths, list):
                    pdb_charge_path = pdb_charge_paths[counter] if counter < len(pdb_charge_paths) else None
                else:
                    pdb_charge_path = pdb_charge_paths
                if pdb_charge_path is None:
                    print(f"Warning: No PDB charge file for job {counter}, skipping")
                    counter += 1
                    continue
                pdb_charge_path = os.path.abspath(pdb_charge_path)
                if not os.path.exists(pdb_charge_path):
                    raise FileNotFoundError(f"PDB charge file not found: {pdb_charge_path}")
                df_pdb_charges = self.getPdbCharges(pdb_charge_path)
                n_atoms = len(df_pdb_charges)
                all_lines = [x for x in range(n_atoms) if x not in self.config['excludeAtomfromEcalc']]

            counter += 1

            # Calculate total lines for monopole calculations (standard mode)
            if not use_pdb_charges and (not use_multipole or include_decay):
                total_lines = Electrostatics.mapcount(file_path_xyz)
                init_all_lines = range(0, total_lines - 2)
                all_lines = [x for x in init_all_lines if x not in self.config['excludeAtomfromEcalc']]

            for charge_type in charge_types:
                print(f'Partial Charge Scheme: {charge_type}')
                try:
                    # Use directory path directly (f is already absolute path from lst_of_folders)
                    file_path_multipole = f"{f}/Multipole{charge_type}.txt"
                    file_path_monopole = f"{f}/Charges{charge_type}.txt"

                    if use_pdb_charges:
                        # pdb_only: skip partitionCharge entirely
                        pass
                    else:
                        # Check if required files exist when skip_missing_files is enabled
                        if self.config.get('skip_missing_files', False):
                            required_file = file_path_multipole if use_multipole else file_path_monopole
                            if not os.path.exists(required_file):
                                print(f"""
{'='*60}
WARNING: Required charge file not found - skipping this calculation
{'='*60}
Expected file: {required_file}
Job folder: {f}
Charge type: {charge_type}

Skipping to next charge type (skip_missing_files=True)
{'='*60}
""")
                                continue

                        # Use centralized partitionCharge() to get/compute charges
                        # Extract actual filenames from paths
                        molden_filename = os.path.basename(self.molden_paths[counter-1])
                        xyz_filename = os.path.basename(self.xyz_paths[counter-1])
                        comp_cost = self.multiwfn.partitionCharge(
                            multipole_bool=use_multipole,
                            f=f,
                            multiwfn_path=multiwfn_path,
                            charge_type=charge_type,
                            owd=owd,
                            molden_filename=molden_filename,
                            xyz_filename=xyz_filename
                        )

                        if comp_cost == -1:
                            print(f"WARNING: Charge calculation failed for {charge_type} in {f}")
                            continue
                    path_to_xyz = file_path_xyz

                    # Get point charge path for this job
                    ptchg_path = None
                    should_include_ptchg = includePtChgs if includePtChgs is not None else self.config.get('includePtChgs', False)
                    if should_include_ptchg:
                        if ptchg_paths is not None:
                            # Function-level override
                            if not isinstance(ptchg_paths, list):
                                raise TypeError("ptchg_paths must be a list of paths or None values")
                            if counter - 1 >= len(ptchg_paths):
                                raise IndexError(f"Job index {counter-1} out of range for ptchg_paths (length {len(ptchg_paths)})")
                            ptchg_path = ptchg_paths[counter - 1]
                            if ptchg_path is not None:
                                ptchg_path = os.path.abspath(ptchg_path)
                        else:
                            # Use helper method (handles list and filename modes)
                            ptchg_path = self._get_ptchg_path_for_job(counter - 1, f, override_path=None)

                        if ptchg_path is not None and not os.path.exists(ptchg_path):
                            if self.config.get('skip_missing_files', False):
                                print(f"""
{'='*60}
WARNING: Point charge file not found - skipping this calculation
{'='*60}
Expected path: {ptchg_path}
Job index: {counter - 1}
Job folder: {f}
Charge type: {charge_type}

Skipping to next calculation (skip_missing_files=True)
{'='*60}
""")
                                continue
                            else:
                                raise FileNotFoundError(f"""
{'='*60}
ERROR: Point charge file not found
{'='*60}
Expected path: {ptchg_path}
Job index: {counter - 1}
Job folder: {f}

Please verify the file exists or set the path to None for jobs without point charges.
{'='*60}
""")

                    # Calculate ESP using appropriate method
                    if use_pdb_charges:
                        # pdb_only: compute ESP directly from the PDB charge DataFrame
                        [ESP_all, atom_type] = self.monopole_esp_df(
                            df_pdb_charges, atom_idx, all_lines
                        )
                        df_pdb_reset = df_pdb_charges.reset_index(drop=True)
                        total_charge = df_pdb_reset['charge'].sum()
                        partial_charge_atom = df_pdb_reset['charge'].iloc[atom_idx]
                        results_dict['Atoms'] = atom_type
                        results_dict['Total Charge'] = total_charge
                        results_dict[f'Partial Charge pdb'] = partial_charge_atom
                        results_dict[f'ESP pdb'] = ESP_all
                    elif use_multipole:
                        [final_esp, atom_name] = self.multipole_esp(
                            file_path_xyz, file_path_multipole, all_lines, atom_idx, ptchg_path=ptchg_path
                        )
                        results_dict['Total ESP'] = final_esp
                        results_dict['Atom'] = atom_name
                    else:
                        [ESP_all, atom_type] = self.monopole_esp(
                            file_path_xyz, atom_idx, all_lines, file_path_monopole, ptchg_path=ptchg_path
                        )
                        [total_charge, partial_charge_atom] = Electrostatics.charge_atom(
                            file_path_monopole, atom_idx
                        )
                        results_dict['Atoms'] = atom_type
                        results_dict['Total Charge'] = total_charge
                        results_dict[f'Partial Charge {charge_type}'] = partial_charge_atom
                        results_dict[f'ESP {charge_type}'] = ESP_all

                    # Add decay analysis if requested
                    if include_decay:
                        try:
                            [sorted_distances, sorted_esps, cum_esps, sorted_cum_idx,
                             sorted_cum_chg, sorted_atomTypes] = self.esp_bydistance(
                                path_to_xyz, atom_idx, file_path_monopole
                            )
                            results_dict['Sorted Distances'] = sorted_distances
                            results_dict[f'Sorted ESP {charge_type}'] = sorted_esps
                            results_dict[f'Cumulative ESP {charge_type}'] = cum_esps
                            results_dict[f'Dist Sorted Idxs {charge_type}'] = sorted_cum_idx
                            results_dict[f'Dist Sorted Partial Charges {charge_type}'] = sorted_cum_chg
                            results_dict[f'Dist Sorted Atom Types {charge_type}'] = sorted_atomTypes
                        except Exception as e:
                            print(f'Warning: Decay analysis failed for {charge_type}: {e}')

                    # Add coordination shell analysis if requested
                    if include_coord_shells:
                        try:
                            ESP_fcoord = self.esp_first_coord(atom_idx, file_path_monopole, path_to_xyz)
                            ESP_scoord = self.esp_second_coord(atom_idx, file_path_monopole, path_to_xyz)
                            results_dict[f'{charge_type} ESP First Coor Shell (kcal/mol)'] = ESP_fcoord
                            results_dict[f'{charge_type} ESP Second Coor Shell (kcal/mol)'] = ESP_scoord
                        except Exception as e:
                            print(f'Warning: Coordination shell analysis failed for {charge_type}: {e}')

                    # Visualization: ESP values
                    if viz_esp:
                        try:
                            from .geometry import Visualize
                            import numpy as np

                            # Get total number of atoms
                            total_atoms = Electrostatics.mapcount(file_path_xyz) - 2

                            # Create b_factor array with ESP value at metal atom
                            b_factors = np.zeros(total_atoms)

                            # Get ESP value based on mode
                            if use_multipole:
                                esp_value = final_esp if 'final_esp' in locals() else 0.0
                            else:
                                esp_value = ESP_all if 'ESP_all' in locals() else 0.0

                            b_factors[atom_idx] = esp_value

                            # Create PDB file
                            structure_name = os.path.basename(f.rstrip("/"))
                            path_to_pol = file_path_multipole if use_multipole else file_path_monopole
                            pdbName = f'esp_{charge_type}_{structure_name}_atom{atom_idx}.pdb'
                            Visualize(file_path_xyz).makePDB(path_to_pol, b_factors, pdbName)
                            print(f"    Created ESP visualization PDB: {pdbName}")
                        except Exception as e:
                            print(f'Warning: ESP visualization failed for {charge_type}: {e}')

                except Exception as e:
                    logging.exception(f'Exception during ESP calculation for {charge_type}')
                    continue

            allspeciesdict.append(results_dict)

        os.chdir(owd)
        df = pd.DataFrame(allspeciesdict)

        # Generate automatic CSV filename
        # Format: esp_{charge_type}.csv or esp_multipole_{charge_type}.csv
        if len(charge_types) == 1:
            charge_type_str = charge_types[0]
        else:
            charge_type_str = "_".join(charge_types)

        prefix = "multipole_" if use_multipole else ""
        auto_filename = f"esp_{prefix}{charge_type_str}.csv"

        # Use automatic filename if user provided default/generic name, otherwise use their name
        if ESPdata_filename in ['output', 'esp', 'ESPdata', 'esp_data']:
            output_filename = auto_filename
        else:
            output_filename = f"{ESPdata_filename}.csv"

        df.to_csv(output_filename)
        print(f"Saved ESP results to: {output_filename}")
        return df

    def getEfield(self, charge_types='Hirshfeld_I', Efielddata_filename='ef', multiwfn_path=None,
                  multipole_bool=False, input_bond_indices=[], auto_find_bonds=False,
                  save_atomwise_decomposition=False, visualize=None, dielectric=1,
                  dielectric_scale=1, includePtChgs=None, ptchg_paths=None,
                  pdb_charge_paths=None):
        """Unified E-field calculation method supporting multiple modes.

        This consolidated method replaces getEFieldMultipole(), getEfield_acrossBond(),
        and getEfield_decomposable() to eliminate code duplication while preserving all functionality.

        Parameters
        ----------
        charge_types : str or list of str
            Charge partitioning scheme. If a list is provided, only the first entry is used.
            Options: 'Hirshfeld', 'Becke', 'Hirshfeld_I', etc.
            When pdb_charge_paths is provided, this value is used for labeling only.
        Efielddata_filename : str
            Output CSV filename (without extension).
        multiwfn_path : str
            Path to Multiwfn executable.
        multipole_bool : bool, optional
            If True, use multipole expansion; if False, use monopole charges (default: False).
        input_bond_indices : list, optional
            Bond indices as list of tuples [(atomA, atomB), ...].
            If empty, bonds are auto-found using metal indices (esp_atom_idx).
            If provided, bonds are used as-is unless auto_find_bonds=True.
        auto_find_bonds : bool, optional
            If True, automatically find bonds to adjacent atoms using getBondedAtoms()
            and ignore input_bond_indices (default: False).
        save_atomwise_decomposition : bool, optional
            If True, save atom-wise E-field decomposition to CSV (default: False).
        visualize : bool or None, optional
            Override config visualization settings. None uses config defaults (default: None).
        dielectric : float, optional
            Dielectric constant (default: 1).
        dielectric_scale : float, optional
            Scaling factor for point charges. Point charge contributions are scaled by
            1/sqrt(dielectric_scale). Typically used in QM/MM calculations (default: 1).
        includePtChgs : bool or None, optional
            Override object-level includePtChgs setting for this calculation.
            If None, uses object-level setting (default: None).
        ptchg_paths : list of str or None, optional
            Override point charge file paths for this calculation.
            List of paths (one per job), with None for jobs without point charges.
            If provided, takes precedence over object-level settings (default: None).
        pdb_charge_paths : str or list of str or None, optional
            If provided, use partial charges from PDB B-factor columns only (no Multiwfn).
            Can be a single PDB path or a list of paths (one per job), with None to skip a job.
            This mode supports monopole charges only (multipole_bool must be False).

        Returns
        -------
        pd.DataFrame or tuple of pd.DataFrame
            If save_atomwise_decomposition=False: returns DataFrame with E-field results
            If save_atomwise_decomposition=True: returns (total_df, atomwise_df) tuple

        Output Files
        ------------
        **Main CSV file:** Auto-generated with format:
        - Monopole: `ef_{charge_type}.csv`
        - Multipole: `ef_multipole_{charge_type}.csv`
        - Examples: `ef_Hirshfeld_I.csv`, `ef_multipole_Hirshfeld_I.csv`
        - Contains E-field magnitudes and projections for all jobs and bonds
        - Columns: job name, bond atoms, E-field magnitude (MV/cm), projections, etc.

        **Atomwise CSV (if save_atomwise_decomposition=True):**
        - Format: `ef_{charge_type}_atomwise.csv` or `ef_multipole_{charge_type}_atomwise.csv`
        - Example: `ef_Hirshfeld_I_atomwise.csv`
        - Per-atom E-field contributions for each bond
        - Shows which atoms contribute most to the E-field
        - Columns: job, bond, atom index, contribution magnitude, etc.

        **PDB files (if visualize=True):**
        - Per bond: `ef_{charge_type}_{structure}_bond{i}-{j}.pdb`
        - Combined: `ef_{charge_type}_{structure}_combined.pdb`
        - B-factor contains E-field contribution from each atom
        - Examples: `ef_Hirshfeld_I_complex_bond25-26.pdb`

        Note: If you provide a custom filename (not 'output', 'efield', etc.),
        your filename will be used instead of automatic naming.

        Examples
        --------
        >>> # Basic multipole mode with specified bonds
        >>> es.getEfield('Hirshfeld_I', 'efield', ..., input_bond_indices=[(0,1), (0,2)])

        >>> # Auto-find bonds to adjacent atoms
        >>> es.getEfield('Hirshfeld_I', 'efield', ..., auto_find_bonds=True)

        >>> # Monopole mode with atom-wise decomposition
        >>> es.getEfield('Hirshfeld', 'efield', ..., multipole_bool=False, save_atomwise_decomposition=True)
        """
        # Resolve pdb_charge_paths: function-level arg takes priority, else use object-level
        if pdb_charge_paths is None and self.pdb_only:
            pdb_charge_paths = self.pdb_charge_paths
        use_pdb_charges = pdb_charge_paths is not None

        if use_pdb_charges and multipole_bool:
            raise ValueError("PDB charge mode supports monopole charges only. Set multipole_bool=False.")

        # Ensure charge_types is a string (E-field works with single scheme)
        if isinstance(charge_types, list):
            if len(charge_types) > 1:
                print(f"Warning: Multiple charge types provided. Using first: {charge_types[0]}")
            charge_type = charge_types[0]
        else:
            charge_type = charge_types

        # Filter charge type based on multipole/monopole mode (skip for PDB charge mode)
        if not use_pdb_charges:
            if multipole_bool:
                filtered_types = filter_charge_types_for_multipole(charge_type, context="getEfield")
                if not filtered_types:
                    # Return empty DataFrame if charge type not supported
                    return pd.DataFrame()
                charge_type = filtered_types[0]
            else:
                filtered_types = filter_charge_types_for_monopole(charge_type, context="getEfield")
                if not filtered_types:
                    # Return empty DataFrame if charge type not supported
                    return pd.DataFrame()
                charge_type = filtered_types[0]

        self.config['dielectric'] = dielectric
        self.config['dielectric_scale'] = dielectric_scale
        metal_idxs = self.esp_atom_idx
        list_of_file = self.lst_of_folders
        final_structure_file = self.config['xyzfilename']
        # Handle xyzfilename being a list (take first element)
        if isinstance(final_structure_file, list):
            final_structure_file = final_structure_file[0]

        # Handle visualization settings
        if visualize is None:
            viz_efield = self.config.get('visualize_ef', False)
            viz_charges = self.config.get('visualize_charges', False)
            viz_per_bond = self.config.get('visualize_per_bond', False)
        else:
            viz_efield = visualize
            viz_charges = visualize
            viz_per_bond = False

        owd = os.getcwd()
        allspeciesdict = []
        all_atomwise_data = []  # Collect atomwise E-field decomposition
        counter = 0

        for f in list_of_file:
            try:
                # Convert f to string if it's not already (handles int folder names)
                f = str(f)
                results_dict = {}
                # Use stored absolute path instead of constructing it
                file_path_xyz = self.xyz_paths[counter]

                # In pdb_only mode xyz_paths[counter] is None; defer total_lines/all_lines
                # until after the PDB charge file has been loaded (see below).
                if not use_pdb_charges:
                    total_lines = Electrostatics.mapcount(file_path_xyz)
                    init_all_lines = range(0, total_lines - 2)
                else:
                    total_lines = None   # set after loading PDB charges
                    init_all_lines = None

                # Handle excludeAtomfromEcalc - support both flat list and nested list
                exclude_atoms = self.config.get('excludeAtomfromEcalc', [])
                if exclude_atoms and isinstance(exclude_atoms[0], (list, tuple, np.ndarray)):
                    # Nested structure: different exclusions per folder (list of lists/arrays)
                    exclude_list = list(exclude_atoms[counter])
                else:
                    # Flat list/array: same exclusions for all folders
                    exclude_list = list(exclude_atoms) if hasattr(exclude_atoms, '__iter__') else []

                if not use_pdb_charges:
                    all_lines = [x for x in init_all_lines if x not in exclude_list]
                else:
                    all_lines = None   # set after loading PDB charges

                # Determine bond indices
                if auto_find_bonds or (not input_bond_indices):
                    # Auto-find bonded atoms (requires metal indices)
                    if not self.esp_atom_idx:
                        raise ValueError(f"""
{'='*60}
ERROR: Electric field calculation requires bond or metal indices
{'='*60}
You need to either:
1. Provide metal indices (esp_atom_idx) for auto-finding bonds, OR
2. Specify bond indices explicitly via input_bond_indices parameter

Option 1 - Auto-find bonds:
  estat = Electrostatics(
      molden_paths=['job1.molden'],
      xyz_paths=['job1.xyz'],
      esp_atom_idx=[25]  # Metal atom index
  )
  ef_df = estat.getEfield('Hirshfeld_I', 'output', multiwfn_path,
                          auto_find_bonds=True)

Option 2 - Specify bond indices:
  estat = Electrostatics(
      molden_paths=['job1.molden'],
      xyz_paths=['job1.xyz']
  )
  ef_df = estat.getEfield('Hirshfeld_I', 'output', multiwfn_path,
                          input_bond_indices=[[(25, 26), (25, 27)]])

Note: Atom indices are 0-based (first atom = index 0)

For complete examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
- README.md: Section 3.1 (Electric Field Calculation)
{'='*60}
""")
                    atom_idx = metal_idxs[counter]
                    bonded_atoms_list = self.getBondedAtoms(file_path_xyz, atom_idx)
                    bond_indices_to_use = [(atom_idx, bonded_idx) for bonded_idx in bonded_atoms_list]
                    print(f"Auto-found {len(bond_indices_to_use)} bonds for atom {atom_idx}: {bond_indices_to_use}")
                elif counter < len(input_bond_indices):
                    bond_indices_to_use = input_bond_indices[counter] if isinstance(input_bond_indices[counter], list) else [input_bond_indices[counter]]
                else:
                    bond_indices_to_use = []

                if not bond_indices_to_use:
                    print(f"Warning: No bond indices for {f}, skipping")
                    counter += 1
                    continue

                # Partition charges unless using PDB charge mode
                if not use_pdb_charges:
                    # Partition charges - extract actual filenames from paths
                    molden_filename = os.path.basename(self.molden_paths[counter])
                    xyz_filename = os.path.basename(self.xyz_paths[counter])
                    comp_cost = self.multiwfn.partitionCharge(
                        multipole_bool, f, multiwfn_path, charge_type, owd,
                        molden_filename=molden_filename, xyz_filename=xyz_filename
                    )

                    if comp_cost == -1:
                        print(f"Warning: Charge calculation failed for {f}")
                        counter += 1
                        continue

                    # Use directory path directly (f is already absolute path from lst_of_folders)
                    file_path_multipole = f"{f}/Multipole{charge_type}.txt"
                    file_path_charges = f"{f}/Charges{charge_type}.txt"
                else:
                    comp_cost = 0
                    file_path_multipole = None
                    file_path_charges = None

                # Handle point charges with function-level overrides (skip in PDB charge mode)
                df_ptchg = None
                ptchg_path = None
                should_include_ptchg = includePtChgs if includePtChgs is not None else self.config.get('includePtChgs', False)

                if should_include_ptchg and not use_pdb_charges:
                    # Get path for this specific job
                    if ptchg_paths is not None:
                        # Function-level override
                        if not isinstance(ptchg_paths, list):
                            raise TypeError("ptchg_paths must be a list of paths or None values")
                        if counter >= len(ptchg_paths):
                            raise IndexError(f"Job index {counter} out of range for ptchg_paths (length {len(ptchg_paths)})")
                        ptchg_path = ptchg_paths[counter]
                        if ptchg_path is not None:
                            ptchg_path = os.path.abspath(ptchg_path)
                    else:
                        # Use helper method to get path (handles list and filename modes)
                        ptchg_path = self._get_ptchg_path_for_job(counter, f, override_path=None)

                    if ptchg_path is not None:
                        if not os.path.exists(ptchg_path):
                            if self.config.get('skip_missing_files', False):
                                print(f"""
{'='*60}
WARNING: Point charge file not found - skipping this calculation
{'='*60}
Expected path: {ptchg_path}
Job index: {counter}
Job folder: {f}
Charge type: {charge_type}

Skipping to next job (skip_missing_files=True)
{'='*60}
""")
                                counter += 1
                                continue
                            else:
                                raise FileNotFoundError(f"""
{'='*60}
ERROR: Point charge file not found
{'='*60}
Expected path: {ptchg_path}
Job index: {counter}
Job folder: {f}

Please verify the file exists or set the path to None for jobs without point charges.
{'='*60}
""")
                        df_ptchg = self.getPtChgs(ptchg_path)
                        print(f"  Loaded point charges from: {ptchg_path}")

                # Select appropriate file path
                path_to_pol = file_path_multipole if multipole_bool else file_path_charges
                temp_xyz = file_path_xyz if multipole_bool else ''
                if use_pdb_charges:
                    path_to_pol = file_path_xyz
                    temp_xyz = ''

                # Load PDB charges if requested
                df_pdb_charges = None
                if use_pdb_charges:
                    if isinstance(pdb_charge_paths, list):
                        if counter >= len(pdb_charge_paths):
                            raise IndexError(f"Job index {counter} out of range for pdb_charge_paths (length {len(pdb_charge_paths)})")
                        pdb_charge_path = pdb_charge_paths[counter]
                    else:
                        pdb_charge_path = pdb_charge_paths

                    if pdb_charge_path is None:
                        print(f"Warning: No PDB charge file for {f}, skipping")
                        counter += 1
                        continue

                    pdb_charge_path = os.path.abspath(pdb_charge_path)
                    if not os.path.exists(pdb_charge_path):
                        if self.config.get('skip_missing_files', False):
                            print(f"""
{'='*60}
WARNING: PDB charge file not found - skipping this calculation
{'='*60}
Expected path: {pdb_charge_path}
Job index: {counter}
Job folder: {f}

Skipping to next job (skip_missing_files=True)
{'='*60}
""")
                            counter += 1
                            continue
                        raise FileNotFoundError(f"""
{'='*60}
ERROR: PDB charge file not found
{'='*60}
Expected path: {pdb_charge_path}
Job index: {counter}
Job folder: {f}

Please verify the file exists or set the path to None for jobs without charges.
{'='*60}
""")

                    df_pdb_charges = self.getPdbCharges(pdb_charge_path)
                    # In pdb_only mode the atom count comes from the PDB itself
                    total_lines = len(df_pdb_charges) + 2
                    init_all_lines = range(0, len(df_pdb_charges))
                    all_lines = [x for x in init_all_lines if x not in exclude_list]

                # Calculate E-field with atom-wise decomposition
                if use_pdb_charges:
                    [proj_Efields, bondedAs, bonded_idx, bond_lens, E_proj_atomwise,
                     E_proj_atomwise_list] = self.bondEfield_pointcharges(
                        bond_indices_to_use, df_pdb_charges, all_lines
                    )
                else:
                    [proj_Efields, bondedAs, bonded_idx, bond_lens, E_proj_atomwise,
                     E_proj_atomwise_list] = self.bondEfield(
                        bond_indices_to_use, file_path_xyz, path_to_pol, all_lines,
                        multipole_bool, df_ptchg=df_ptchg
                    )

                # Calculate bond dipoles for each bond
                bond_dipoles = self.calc_bond_dipoles(
                    bond_indices_to_use, file_path_xyz, path_to_pol, multipole_bool, charge_df=df_pdb_charges
                )

                # Get charge data for visualization
                if not use_pdb_charges:
                    df = Electrostatics.charge_atoms(self, path_to_pol, temp_xyz)

                # Visualization: E-field atom-wise contributions
                if viz_efield and E_proj_atomwise is not None:
                    # Create full-length array including zeros for excluded atoms
                    total_atoms = total_lines - 2
                    # E_atomwise contains QM contributions followed by MM (point charge) contributions
                    n_qm_atoms = len(all_lines)

                    # Determine if we have point charges
                    has_ptchg = df_ptchg is not None and len(df_ptchg) > 0
                    n_ptchg = len(df_ptchg) if has_ptchg else 0

                    # Create combined structure file if point charges are present
                    if has_ptchg:
                        import tempfile
                        import warnings
                        # Suppress OpenBabel kekulization warnings for combined structures
                        warnings.filterwarnings('ignore', message='.*kekulize.*')
                        temp_combined_xyz = tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False)
                        combined_xyz_path = temp_combined_xyz.name
                        df_geom = Geometry(file_path_xyz).getGeomInfo()
                        n_total_combined = len(df_geom) + n_ptchg
                        temp_combined_xyz.write(f"{n_total_combined}\n")
                        temp_combined_xyz.write(f"Combined QM + Point Charges\n")
                        for idx, row in df_geom.iterrows():
                            temp_combined_xyz.write(f"{row['Atom']}  {row['X']:.6f}  {row['Y']:.6f}  {row['Z']:.6f}\n")
                        for idx, row in df_ptchg.iterrows():
                            temp_combined_xyz.write(f"X  {row['x']:.6f}  {row['y']:.6f}  {row['z']:.6f}\n")
                        temp_combined_xyz.close()
                        structure_file_for_viz = combined_xyz_path
                    else:
                        structure_file_for_viz = file_path_xyz

                    # Create context manager to suppress OpenBabel warnings
                    class SuppressStderr:
                        def __enter__(self):
                            import sys
                            import os
                            self.null_fd = os.open(os.devnull, os.O_RDWR)
                            self.save_fd = os.dup(2)
                            os.dup2(self.null_fd, 2)
                        def __exit__(self, *_):
                            import os
                            os.dup2(self.save_fd, 2)
                            os.close(self.null_fd)
                            os.close(self.save_fd)

                    # Always create separate PDB files for each bond when multiple bonds exist
                    if len(E_proj_atomwise_list) > 1:
                        # Multiple bonds: create one PDB per bond
                        for bond_idx, (E_atomwise, bond_pair) in enumerate(zip(E_proj_atomwise_list, bonded_idx)):
                            # Use all contributions (QM + point charges)
                            if has_ptchg:
                                E_atomwise_full = np.zeros(total_atoms + n_ptchg)
                                if multipole_bool:
                                    # Multipole mode: E_atomwise has only non-excluded atoms
                                    n_qm_contributions = len(all_lines)
                                    for i, atom_idx in enumerate(all_lines):
                                        E_atomwise_full[atom_idx] = E_atomwise[i]
                                    E_atomwise_full[total_atoms:] = E_atomwise[n_qm_contributions:]
                                else:
                                    # Monopole mode: E_atomwise already has all atoms (with zeros for excluded)
                                    E_atomwise_full[:total_atoms] = E_atomwise[:total_atoms]
                                    E_atomwise_full[total_atoms:] = E_atomwise[total_atoms:]
                            else:
                                E_atomwise_full = np.zeros(total_atoms)
                                if multipole_bool:
                                    # Multipole mode: Map contributions to original atom indices
                                    for i, atom_idx in enumerate(all_lines):
                                        E_atomwise_full[atom_idx] = E_atomwise[i]
                                else:
                                    # Monopole mode: Direct copy
                                    E_atomwise_full[:] = E_atomwise[:]

                            structure_name = os.path.basename(f.rstrip("/"))
                            pdbName = f'ef_{charge_type}_{structure_name}_bond{bond_pair[0]}-{bond_pair[1]}.pdb'
                            # Suppress OpenBabel warnings for combined structures with point charges
                            if has_ptchg:
                                with SuppressStderr():
                                    Visualize(structure_file_for_viz).makePDB(path_to_pol, E_atomwise_full, pdbName)
                            else:
                                Visualize(structure_file_for_viz).makePDB(path_to_pol, E_atomwise_full, pdbName)
                    elif len(E_proj_atomwise_list) == 1:
                        # Single bond: create one PDB with bond info in name
                        E_atomwise = E_proj_atomwise_list[0]
                        if has_ptchg:
                            E_atomwise_full = np.zeros(total_atoms + n_ptchg)
                            if multipole_bool:
                                # Multipole mode: E_atomwise has only non-excluded atoms
                                n_qm_contributions = len(all_lines)
                                for i, atom_idx in enumerate(all_lines):
                                    E_atomwise_full[atom_idx] = E_atomwise[i]
                                E_atomwise_full[total_atoms:] = E_atomwise[n_qm_contributions:]
                            else:
                                # Monopole mode: E_atomwise already has all atoms (with zeros for excluded)
                                E_atomwise_full[:total_atoms] = E_atomwise[:total_atoms]
                                E_atomwise_full[total_atoms:] = E_atomwise[total_atoms:]
                        else:
                            E_atomwise_full = np.zeros(total_atoms)
                            if multipole_bool:
                                # Multipole mode: Map contributions to original atom indices
                                for i, atom_idx in enumerate(all_lines):
                                    E_atomwise_full[atom_idx] = E_atomwise[i]
                            else:
                                # Monopole mode: Direct copy
                                E_atomwise_full[:] = E_atomwise[:]

                        bond_pair = bonded_idx[0]
                        structure_name = os.path.basename(f.rstrip("/"))
                        pdbName = f'ef_{charge_type}_{structure_name}_bond{bond_pair[0]}-{bond_pair[1]}.pdb'
                        # Suppress OpenBabel warnings for combined structures with point charges
                        if has_ptchg:
                            with SuppressStderr():
                                Visualize(structure_file_for_viz).makePDB(path_to_pol, E_atomwise_full, pdbName)
                        else:
                            Visualize(structure_file_for_viz).makePDB(path_to_pol, E_atomwise_full, pdbName)
                    else:
                        # Fallback: use the combined E_proj_atomwise
                        if has_ptchg:
                            E_proj_atomwise_full = np.zeros(total_atoms + n_ptchg)
                            if multipole_bool:
                                # Multipole mode: E_proj_atomwise has only non-excluded atoms
                                n_qm_contributions = len(all_lines)
                                for i, atom_idx in enumerate(all_lines):
                                    E_proj_atomwise_full[atom_idx] = E_proj_atomwise[i]
                                E_proj_atomwise_full[total_atoms:] = E_proj_atomwise[n_qm_contributions:]
                            else:
                                # Monopole mode: E_proj_atomwise already has all atoms (with zeros for excluded)
                                E_proj_atomwise_full[:total_atoms] = E_proj_atomwise[:total_atoms]
                                E_proj_atomwise_full[total_atoms:] = E_proj_atomwise[total_atoms:]
                        else:
                            E_proj_atomwise_full = np.zeros(total_atoms)
                            if multipole_bool:
                                # Multipole mode: Map contributions to original atom indices
                                for i, atom_idx in enumerate(all_lines):
                                    E_proj_atomwise_full[atom_idx] = E_proj_atomwise[i]
                            else:
                                # Monopole mode: Direct copy
                                E_proj_atomwise_full[:] = E_proj_atomwise[:]

                        structure_name = os.path.basename(f.rstrip("/"))
                        pdbName = f'ef_{charge_type}_{structure_name}_combined.pdb'
                        # Suppress OpenBabel warnings for combined structures with point charges
                        if has_ptchg:
                            with SuppressStderr():
                                Visualize(structure_file_for_viz).makePDB(path_to_pol, E_proj_atomwise_full, pdbName)
                        else:
                            Visualize(structure_file_for_viz).makePDB(path_to_pol, E_proj_atomwise_full, pdbName)

                    # Clean up temporary combined XYZ file if created
                    if has_ptchg:
                        import os as os_module
                        try:
                            os_module.unlink(combined_xyz_path)
                        except:
                            pass

                # Store results
                results_dict['Max Eproj'] = max(abs(np.array(proj_Efields)))
                results_dict['Projected_Efields V/Angstrom'] = proj_Efields
                results_dict['Bonded Atoms'] = bondedAs
                results_dict['Bonded Indices'] = bonded_idx
                results_dict['Bond Lengths'] = bond_lens
                results_dict['Bond Dipole Vectors (Debye)'] = [bd['bond_dipole_vec'] for bd in bond_dipoles]
                results_dict['Bond Dipole Magnitudes (Debye)'] = [bd['bond_dipole_mag'] for bd in bond_dipoles]
                results_dict['Bond Charges'] = [bd['charges'] for bd in bond_dipoles]
                results_dict['Comp Cost'] = comp_cost
                results_dict['Folder'] = f

                # Collect atomwise decomposition if requested
                if save_atomwise_decomposition and E_proj_atomwise_list:
                    structure_name = os.path.basename(f.rstrip("/"))
                    total_atoms = total_lines - 2

                    # Store atomwise data for each bond
                    for bond_idx, (E_atomwise, bond_pair) in enumerate(zip(E_proj_atomwise_list, bonded_idx)):
                        # E_atomwise contains contributions from each atom
                        for atom_idx in range(len(E_atomwise)):
                            atomwise_entry = {
                                'Structure': structure_name,
                                'Bond_Index': bond_idx,
                                'Bond_Atoms': f"{bond_pair[0]}-{bond_pair[1]}",
                                'Atom_Index': atom_idx,
                                'Efield_Contribution_V_per_A': E_atomwise[atom_idx],
                                'Charge_Type': charge_type
                            }
                            all_atomwise_data.append(atomwise_entry)

                allspeciesdict.append(results_dict)
                counter += 1

            except Exception as e:
                print(f"Error processing {f}: {e}")
                logging.exception(f'Exception during E-field calculation')
                counter += 1
                continue

        os.chdir(owd)
        df = pd.DataFrame(allspeciesdict)

        # Generate automatic CSV filename
        # Format: ef_{charge_type}.csv or ef_multipole_{charge_type}.csv
        if isinstance(charge_types, list) and len(charge_types) > 0:
            charge_type_str = charge_types[0]
        else:
            charge_type_str = charge_types

        prefix = "multipole_" if multipole_bool else ""
        auto_filename = f"ef_{prefix}{charge_type_str}"

        # Use automatic filename if user provided default/generic name
        if Efielddata_filename in ['output', 'efield', 'Efielddata', 'ef_data']:
            base_filename = auto_filename
        else:
            base_filename = Efielddata_filename

        main_csv = f"{base_filename}.csv"
        df.to_csv(main_csv)
        print(f"Saved E-field results to: {main_csv}")

        # Save atomwise decomposition if requested
        if save_atomwise_decomposition and all_atomwise_data:
            df_atomwise = pd.DataFrame(all_atomwise_data)
            atomwise_csv = f"{base_filename}_atomwise.csv"
            df_atomwise.to_csv(atomwise_csv, index=False)
            print(f"Saved atomwise E-field decomposition to: {atomwise_csv}")
            return df, df_atomwise

        return df

    def getCharges(self, charge_types='Hirshfeld_I', multiwfn_path=None, output_filename='charges',
                   write_pdb=False, pdb_bfactor=True):
        """Compute partial charges for all atoms using specified charge partitioning scheme(s).

        This method provides a standalone interface for computing partial charges
        without calculating E-fields or ESP. Charges can optionally be written to
        PDB files with the B-factor column containing the charge values.

        Parameters
        ----------
        charge_types : str or list of str
            Charge partitioning scheme(s) to use. Options include:
            'RESP', 'Hirshfeld', 'Hirshfeld_I', 'CHELPG', 'Mulliken', 'Lowdin',
            'SCPA', 'Becke', 'ADCH', 'Voronoi', 'MK', 'AIM', 'CM5', 'EEM', 'PEOE'
        multiwfn_path : str
            Path to the Multiwfn executable.
        output_filename : str, optional
            Base name for output files (default: 'charges').
            CSV will be saved as '{output_filename}.csv'.
        write_pdb : bool, optional
            If True, create PDB files for each structure (default: False).
        pdb_bfactor : bool, optional
            If True and write_pdb=True, store charges in B-factor column (default: True).
            This allows visualization of charges in molecular viewers like PyMOL.

        Returns
        -------
        pd.DataFrame
            DataFrame containing partial charges for all atoms in each structure.
            Columns include: 'Job', 'Charge_Type', 'Atom_Index', 'Element',
            'x', 'y', 'z', 'Charge'.

        Output Files
        ------------
        **CSV file:** `{output_filename}.csv`
            Contains all partial charges for all atoms and all jobs.

        **PDB files (if write_pdb=True):** `{structure}_{charge_type}_charges.pdb`
            One PDB file per job per charge type.
            B-factor column contains the partial charge values.

        Examples
        --------
        >>> es = Electrostatics(molden_paths, xyz_paths)
        >>>
        >>> # Get RESP charges for all atoms
        >>> df = es.getCharges('RESP', '/path/to/multiwfn')
        >>>
        >>> # Get multiple charge types with PDB output
        >>> df = es.getCharges(['RESP', 'Hirshfeld_I'], '/path/to/multiwfn',
        ...                    output_filename='my_charges', write_pdb=True)
        >>>
        >>> # Access charges for a specific atom
        >>> atom_5_charges = df[df['Atom_Index'] == 5]

        Notes
        -----
        - Charges are computed via Multiwfn and cached in 'Charges{type}.txt' files.
        - Subsequent calls with the same charge type will reuse cached results
          unless config['rerun'] is True.
        - PDB files with B-factor charges can be visualized in PyMOL using:
          spectrum b, blue_white_red, minimum=-1, maximum=1
        """
        # Ensure charge_types is a list
        if isinstance(charge_types, str):
            charge_types = [charge_types]

        # Filter charge types for monopole analysis
        charge_types = filter_charge_types_for_monopole(charge_types, context="getCharges")

        if not charge_types:
            raise ValueError("No valid charge types provided. Check that the charge types are supported.")

        owd = os.getcwd()
        all_results = []

        print(f"\n{'='*60}")
        print(f"Computing partial charges")
        print(f"Charge types: {charge_types}")
        print(f"Number of jobs: {len(self.lst_of_folders)}")
        print(f"{'='*60}\n")

        for counter, f in enumerate(self.lst_of_folders):
            # Convert f to string if it's not already (handles int folder names)
            f = str(f)
            job_name = os.path.basename(f.rstrip('/'))

            # Get actual filenames from stored paths
            molden_filename = os.path.basename(self.molden_paths[counter])
            xyz_filename = os.path.basename(self.xyz_paths[counter])
            file_path_xyz = self.xyz_paths[counter]

            print(f"\nProcessing job {counter + 1}/{len(self.lst_of_folders)}: {job_name}")

            for charge_type in charge_types:
                print(f"  Computing {charge_type} charges...")

                # Run charge partitioning via Multiwfn
                comp_cost = self.multiwfn.partitionCharge(
                    multipole_bool=False,
                    f=f,
                    multiwfn_path=multiwfn_path,
                    charge_type=charge_type,
                    owd=owd,
                    molden_filename=molden_filename,
                    xyz_filename=xyz_filename
                )

                if comp_cost == -1:
                    print(f"  ERROR: Failed to compute {charge_type} charges for {job_name}")
                    continue

                # Read the computed charges
                charge_file = f"{f}/Charges{charge_type}.txt"

                if not os.path.exists(charge_file):
                    print(f"  ERROR: Charge file not found: {charge_file}")
                    continue

                # Parse charge file (format: Element x y z charge)
                try:
                    df_charges = pd.read_csv(charge_file, sep=r'\s+',
                                            names=["Element", 'x', 'y', 'z', "Charge"])
                    df_charges['Atom_Index'] = range(len(df_charges))
                    df_charges['Job'] = job_name
                    df_charges['Charge_Type'] = charge_type
                    df_charges['Molden_Path'] = self.molden_paths[counter]
                    df_charges['XYZ_Path'] = file_path_xyz

                    # Reorder columns
                    df_charges = df_charges[['Job', 'Charge_Type', 'Atom_Index', 'Element',
                                            'x', 'y', 'z', 'Charge', 'Molden_Path', 'XYZ_Path']]

                    all_results.append(df_charges)

                    total_charge = df_charges['Charge'].sum()
                    print(f"    Loaded {len(df_charges)} atoms, total charge: {total_charge:.4f}")

                    # Create PDB with charges in B-factor column
                    if write_pdb and pdb_bfactor:
                        pdb_name = f"{f}/{job_name}_{charge_type}_charges.pdb"
                        charges_array = df_charges['Charge'].values

                        try:
                            visualize_charges_pdb(file_path_xyz, charges_array, pdb_name)
                            print(f"    Created PDB: {os.path.basename(pdb_name)}")
                        except Exception as e:
                            print(f"    Warning: Could not create PDB file: {e}")

                except Exception as e:
                    print(f"  ERROR reading charge file: {e}")
                    continue

        os.chdir(owd)

        if not all_results:
            print("\nNo charge results were computed successfully.")
            return pd.DataFrame()

        # Combine all results
        df = pd.concat(all_results, ignore_index=True)

        # Save to CSV
        csv_filename = f"{output_filename}.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n{'='*60}")
        print(f"Saved charges to: {csv_filename}")
        print(f"Total atoms: {len(df)}")
        print(f"{'='*60}\n")

        return df


    def get_residues(self, auto_gen, solute_solvent_dict):
        res_dict = {}
        if auto_gen:
            print(f'I am automatically determining the residues and then assigning to a dict')
        else:
            solute_idxs = solute_solvent_dict['Solute Indices']
            res_dict['Solute'] = solute_idxs
            solvent_idxs = solute_solvent_dict['Solvent Indices']
            num_atoms_solvent = solute_solvent_dict['Num Atoms in Solvent']
            num_solvent_res = int(len(solvent_idxs)/num_atoms_solvent)
            for solv_num in range(0, num_solvent_res):
                res_dict[f'Solvent_{solv_num}'] = solvent_idxs[solv_num*num_atoms_solvent:(solv_num+1)*(num_atoms_solvent)]
        return res_dict



    def partitionCharge(self, multipole_bool, f, multiwfn_path, charge_type, owd):
        '''
        Partition electron density using Multiwfn to generate partial charges or multipole moments.

        This is the centralized interface for all charge partitioning schemes (Hirshfeld, CHELPG, etc.).
        Checks if calculation was previously completed and reuses results unless rerun=True.

        Parameters:
        -----------
        multipole_bool : bool
            True to compute multipole moments, False for monopoles only
        f : str
            Complete path to folder containing the calculation
        multiwfn_path : str
            Path to Multiwfn executable
        charge_type : str
            Partitioning scheme (e.g., 'Hirshfeld', 'CHELPG', 'Hirshfeld_I')
        owd : str
            Original working directory

        Returns:
        --------
        comp_cost : float
            Computation time in seconds, 0 if calculation was previously completed (skipped),
            or -1 if calculation failed

        Notes:
        ------
        For Hirshfeld-I calculations, atmrad files are automatically loaded from
        pyef.resources.atmrad (bundled with the package).
        '''
        molden_filename = self.config['molden_filename']
        final_structure_file = self.config['xyzfilename']
        # Handle xyzfilename being a list (take first element)
        if isinstance(final_structure_file, list):
            final_structure_file = final_structure_file[0]
        comp_cost = 0  # Default to 0 for "already exists" case
        num_atoms = 0
        need_to_run_calculation = True
        os.chdir(owd)
        os.chdir(f)
        #subprocess.call(multiwfn_module, shell=True)
        file_path_multipole = f"{os.getcwd()}/Multipole{charge_type}.txt"
        file_path_monopole = f"{os.getcwd()}/Charges{charge_type}.txt"
        file_path_xyz = f"{os.getcwd()}/{final_structure_file}"
        #file_path_xyz = f"{os.getcwd()}/{f + folder_to_molden}{final_structure_file}"

        #check if previous calculations fully converged for desired multipole/charge scheme
        calc_type = "multipole" if multipole_bool else "monopole"
        if multipole_bool:
            if os.path.exists(file_path_multipole):
                with open(file_path_multipole, 'r') as file:
                    contents = file.read()
                    if "Calculation took up" in contents:
                        need_to_run_calculation = False
        else:
            if os.path.exists(file_path_monopole):
                need_to_run_calculation = False
                #Dynamically get path to package settings.ini file
                #path_to_ini_file = str(ini_path)


        #If you need to run the calculations, urn either the multipole or the monopole calculation!
        if need_to_run_calculation or self.config['rerun']:
            print(f'Running {charge_type} {calc_type} calculation for {os.path.basename(f)}')
            try: 
                start = time.time()
                if multipole_bool:
                    chg_prefix, _ = os.path.splitext(molden_filename)
                    #Dynamically get path to package settings.ini file
                    with resources.path('pyef.resources', 'settings.ini') as ini_path:
                        path_to_init_file = str(ini_path)
                        command = f"{multiwfn_path} {molden_filename} -set {path_to_init_file}"

                        multiwfn_commands = ['15', '-1'] + self.multiwfn.dict_of_multipole[charge_type] + ['0', 'q']
                        num_atoms = Electrostatics.mapcount(final_structure_file) - 2
                        if charge_type == 'Hirshfeld_I':
                            #get the number of basis functions
                            num_basis = MoldenObject(file_path_xyz, molden_filename).countBasis()

                            # Use bundled atmrad from package resources
                            with resources.path('pyef.resources', 'atmrad') as atmrad_resource:
                                atmrad_src = str(atmrad_resource)
                                copy_tree(atmrad_src, os.getcwd() + '/atmrad/')

                            print(f'Current num of basis is: {num_basis}')
                            print(f'The current max num is: {self.config["maxIHirshFuzzyBasis"]}')
                            if  num_basis > self.config['maxIHirshFuzzyBasis']:
                                print(f'Number of basis functions: {num_basis}')
                                multiwfn_commands = ['15', '-1'] + ['4', '-2', '1', '2'] + ['0', 'q']
                                print(f'I-Hirshfeld command should be low memory and slow to accomodate large system')

                        # Use centralized Multiwfn runner
                        self.multiwfn.run_multiwfn(
                            command=command,
                            input_commands=multiwfn_commands,
                            output_file=file_path_multipole,
                            description=f"Multipole {charge_type} calculation for {f}"
                        )

                else:
                    # Dynamically get path to package settings.ini file
                    with resources.path('pyef.resources', 'settings.ini') as ini_path:
                        path_to_init_file = str(ini_path)
                        command = f"{multiwfn_path} {molden_filename} -set {path_to_init_file}"
                        chg_prefix,  _ = os.path.splitext(molden_filename)
                        calc_command = self.multiwfn.dict_of_calcs[charge_type]
                        # Default command for most charge types (Hirshfeld, Voronoi, ADCH, CM5)
                        # These have sub-menu where option 1 = use built-in densities or output charges
                        commands = ['7', calc_command, '1', 'y', '0', 'q']

                        if charge_type == 'Mulliken':
                            # Mulliken has a sub-menu that needs extra '0' to exit back to population menu
                            commands = ['7', '5', '1', 'y', '0', '0', 'q']
                        elif charge_type == 'Becke':
                            # Becke has sub-menu: 0=Start calculation
                            commands = ['7', '10', '0', 'y', '0', 'q']
                        elif charge_type == 'SCPA':
                            # SCPA directly calculates, no sub-menu
                            commands = ['7', '7', 'y', '0', 'q']
                        elif charge_type == 'Lowdin':
                            # Lowdin asks for output path (empty=screen), then y/n for .chg file
                            commands = ['7', '6', '', 'y', '0', 'q']
                        elif charge_type == 'EEM':
                            # EEM needs 'g' to guess bonds from molden, then 0=start calculation
                            commands = ['7', '17', 'g', '0', 'y', '0', 'q']
                        elif charge_type == 'PEOE':
                            # PEOE directly calculates (may fail for some elements like Na)
                            commands = ['7', '19', 'y', '0', 'q']
                        elif charge_type == 'AIM':
                            # AIM is not available in menu 7 - needs basin analysis module
                            print(f"WARNING: AIM charges cannot be calculated in population analysis module.")
                            print(f"         Use basin analysis module (main menu 17) instead.")
                            raise ValueError("AIM charge calculation not supported in this module")
                        elif charge_type in ['CHELPG', 'RESP', 'MK']:
                            commands = ['7', calc_command, '1', 'y', '0', '0', 'q']
                        elif charge_type == 'Hirshfeld_I':
                            num_basis = MoldenObject(file_path_xyz, molden_filename).countBasis()
                            print(f'Number of basis functions: {num_basis}')
                            if num_basis > self.config['maxIHirshBasis']:
                                commands = ['7', '15', '-2', '1', '\n', 'y', '0', 'q']

                            # Use bundled atmrad from package resources
                            with resources.path('pyef.resources', 'atmrad') as atmrad_resource:
                                atmrad_src = str(atmrad_resource)
                                copy_tree(atmrad_src, os.getcwd() + '/atmrad/')
                            #if too many atoms will need to change calc to run with reasonable memory

                        # Use centralized Multiwfn runner
                        self.multiwfn.run_multiwfn(
                            command=command,
                            input_commands=commands,
                            description=f"Monopole {charge_type} calculation for {f}"
                        )
                    os.rename(f'{chg_prefix}.chg', file_path_monopole)
                end = time.time()
                comp_cost = end - start
            except Exception as e:
                print(f"\n{'='*60}")
                print(f"ERROR in partitionCharge for {f}")
                print(f"Charge type: {charge_type}, Multipole: {multipole_bool}")
                print(f"{'='*60}")
                print(f"Error: {str(e)}")
                print(f"\nFull traceback:")
                print(traceback.format_exc())
                print(f"{'='*60}\n")
                # Issue could be from lost memory or Multiwfn failure
                os.chdir(owd)
                return -1  # Return -1 to indicate failure instead of comp_cost
        else:
            result_file = os.path.basename(file_path_multipole if multipole_bool else file_path_monopole)
            print(f'Found existing {charge_type} {calc_type} result: {result_file} for {os.path.basename(f)}')
        os.chdir(owd)
        return comp_cost

    def get_residueDipoles(self, charge_type, multiwfn_path, multipole_bool, solute_indices = [], num_atoms_solvent=0, save_distance_dipoles=False):
        '''
        Function computes a partial charges of all atoms in one system... if the desired file already exists the just return it!!
        Inputs:
        ------- 
        charge_type: string, possible choices include: 'Hirshfeld', 'Voronoi', 'Mulliken',  'Lowdin', 'SCPA', 'Becke', 'ADCH', 'CHELPG', 'MK', 'AIM', 'Hirshfeld_I', 'CM5', 'EEM', 'RESP', 'PEOE'}
                    Also accepts multipole options including: 'Hirshfeld': ['3', '2'], 'Hirshfeld_I': ['4', '1', '2'],   'Becke': ['1', '2']
        '''
        lst_dicts = []
        need_to_run_calculation = True


        # Access Class Variables
        list_of_folders = self.lst_of_folders
        owd = os.getcwd() # Old working directory

        # Filter charge type based on multipole/monopole mode
        if multipole_bool:
            filtered_types = filter_charge_types_for_multipole(charge_type, context="get_residueDipoles")
            if not filtered_types:
                # Return empty DataFrame if charge type not supported
                return pd.DataFrame()
            charge_type = filtered_types[0]
        else:
            filtered_types = filter_charge_types_for_monopole(charge_type, context="get_residueDipoles")
            if not filtered_types:
                # Return empty DataFrame if charge type not supported
                return pd.DataFrame()
            charge_type = filtered_types[0]

        molden_filename = self.config['molden_filename']
        final_structure_file = self.config['xyzfilename']
        # Handle xyzfilename being a list (take first element)
        if isinstance(final_structure_file, list):
            final_structure_file = final_structure_file[0]
        frame_num_lst = []

        counter = 0
        for f in list_of_folders:
            try:
                # Convert f to string if it's not already (handles int folder names)
                f = str(f)
                # Extract actual filenames from paths
                molden_filename = os.path.basename(self.molden_paths[counter])
                xyz_filename = os.path.basename(self.xyz_paths[counter])
                comp_cost = self.multiwfn.partitionCharge(multipole_bool, f, multiwfn_path, charge_type, owd,
                                                         molden_filename=molden_filename, xyz_filename=xyz_filename)
                #If the calculation is not successful, continue
                if comp_cost == -1:
                    print(f"Warning: Charge calculation failed for {f}, skipping")
                    counter += 1
                    continue
                # Use directory path directly (f is already absolute path from lst_of_folders)
                file_path_multipole = f"{f}/Multipole{charge_type}.txt"
                file_path_monopole = f"{f}/Charges{charge_type}.txt"
                # Use stored absolute path instead of constructing it
                file_path_xyz = self.xyz_paths[counter]
                counter += 1
                    
                if multipole_bool:
                    xyz_fp =  file_path_xyz
                    chg_fp = file_path_multipole
                    df_charge_atoms = self.charge_atoms(chg_fp, xyz_fp)
                else:
                    chg_fp = file_path_monopole
                    xyz_fp = ''
                    df_charge_atoms = self.charge_atoms(chg_fp, xyz_fp)
                solute_solvent_dict = {}
                solute_solvent_dict['Solute Indices'] = solute_indices
                solute_solvent_dict['Num Atoms in Solvent'] = num_atoms_solvent
                total_atoms = Electrostatics.mapcount(file_path_xyz) -2
                all_indices = list(np.arange(0, total_atoms))
                solute_solvent_dict['Solvent Indices'] = [x for x in all_indices if x not in solute_indices]
                res_dict = self.get_residues(False, solute_solvent_dict)
                df_dip = self.getdipole_residues(res_dict, df_charge_atoms) 
                #re-order res by closest to the solute
                all_centroids = np.vstack(df_dip['centroid'].values)
                solute_name = 'Solute'
                ref_coord = df_dip.loc[df_dip['Label'] == solute_name, 'centroid']
                ref_coord_arr = ref_coord.values[0]
                # Compute distances to reference
                distances = np.linalg.norm(all_centroids - ref_coord_arr, axis=1)
                # Get sort order (indices of sorted distances)
                df_dip['Distance from solute'] = distances
                init_dipole = df_dip['Dipole']
            
                df_dip.sort_values(by='Distance from solute', ascending=True)
                if save_distance_dipoles:
                    df_dip.to_csv(f'{f}/chg{charge_type}_DistDependentDipoles.csv')
                sorted_dips = df_dip['Dipole']
                avg_dips = np.average(sorted_dips[1:]) if len(sorted_dips) > 1 else np.nan
            
                first_5A = df_dip.loc[(df_dip['Distance from solute'] < 5), 'Dipole'][1:]
                v5to7A = df_dip.loc[(df_dip['Distance from solute'] < 7) & (df_dip['Distance from solute'] > 5), 'Dipole']
                v7to11A = df_dip.loc[(df_dip['Distance from solute'] < 11) & (df_dip['Distance from solute'] > 7), 'Dipole']
                above11A = df_dip.loc[(df_dip['Distance from solute'] > 11), 'Dipole']

                #return a list of the solvents
                # Handle empty arrays by using np.nan as default
                avg_5A = np.average(first_5A) if len(first_5A) > 0 else np.nan
                avg_5to7A = np.average(v5to7A) if len(v5to7A) > 0 else np.nan
                avg_7to11A = np.average(v7to11A) if len(v7to11A) > 0 else np.nan
                avg_above11A = np.average(above11A) if len(above11A) > 0 else np.nan
                dipole_solute = sorted_dips[0] if len(sorted_dips) > 0 else np.nan

                dip_dict = {'DipoleSolute': dipole_solute, 'AvgDipSolv': avg_dips, 'Avg5Ang': avg_5A, 'Avg5to7Ang': avg_5to7A , 'Avg7to11Ang': avg_7to11A, 'Avgabove11Ang': avg_above11A, 'CompCost': comp_cost, 'Num5Avg':len(first_5A), 'Num5to7':len(v5to7A), 'Num7to11A': len(v7to11A), 'Numabove11A': len(above11A), 'Total_atoms': total_atoms, 'FrameNum': f }
                lst_dicts.append(dip_dict)
            except Exception as e:
                print(e)
                print(f'Error Traceback: {traceback.print_exc()}')
                continue

        all_file_df = pd.DataFrame(lst_dicts)
        all_file_df.to_csv('Dipoles.csv')
        return all_file_df






    def getpartialchgs(self, charge_types, lst_atom_idxs, partial_chg_filename, multiwfn_path):
        '''
        Function computes partial charges on a select set of atoms using the charge scheme specified in charge types. Note atom indices will be carried over between csvs
        
    Inputs:
        --
        charge_types: list(str)
            list of strings
        lst_of_atom_idxs: list(int)
            list of integers denoting atom indices (0 indexed!)
        partial_chg_filename: string
            Name of the output file name    
        multiwfn_path: string
            Path to the multiwfn executable

            Path to the atmrad executable
        
       Outputs:
        --
        df: Pandas Dataframe with partial charge info
        
        
        Notes
        -----
        Will Create a csv file entitled partial_chg_filename.csv with partial charge info
        '''
       # Access Class Variables
        list_of_file = self.lst_of_folders

        # Filter charge types for monopole analysis only
        charge_types = filter_charge_types_for_monopole(charge_types, context="getpartialchgs")

        owd = os.getcwd() # Old working directory
        allspeciesdict = []
        counter = 0  # Iterator to account for atomic indices of interest
        for f in list_of_file:
            # Convert f to string if it's not already (handles int folder names)
            f = str(f)
            print('-----------------' + f + '------------------')
            counter = counter + 1
            os.chdir(owd)
            os.chdir(f)
            # Dynamically get path to package settings.ini file
            with resources.path('pyef.resources', 'settings.ini') as ini_path:
                path_to_init_file = str(ini_path)
                command_A = f"{multiwfn_path} {self.config['molden_filename']} -set {path_to_init_file}"
                results_dir = os.getcwd() + '/'

                results_dict = {}
                results_dict['Name'] = f


                for key in charge_types:
                    print('Partial Charge Scheme:' + str(key))
                    try:
                        full_file_path = os.getcwd() +'/final_optim_' +key+'.txt'
                        path_to_xyz = os.getcwd() + '/' + self.config['xyzfilename']
                        if key == "Hirshfeld_I":
                            with resources.path('pyef.resources', 'atmrad') as atmrad_resource:
                                atmrad_src = str(atmrad_resource)
                                copy_tree(atmrad_src, results_dir + 'atmrad/')
                        try:
                            for atom_idx in lst_atom_idxs:
                                [atom_type, partial_charge_atom] = Electrostatics.getAtomInfo(full_file_path, atom_idx)
                                results_dict[f'{key} Charge {atom_idx} {atom_type}'] = partial_charge_atom
                        except Exception as e:
                            print('The Exception is: ' + str(e))
                            print(traceback.format_exc())
                            print('Error when trying to access electrostatic information: Attemtping to re-compute partial charges of type: ' + str(key))

                            # Re-run multiwfn computation of partial charge
                            proc = subprocess.Popen(command_A, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
                            calc_command = self.multiwfn.dict_of_calcs[key]
                            commands = ['7', calc_command, '1', 'y', '0', 'q'] # for atomic charge type corresponding to dict key
                            if key in ['CHELPG', 'RESP', 'MK']:
                                commands = ['7', calc_command, '1', 'y', '0', '0', 'q']
                            output = proc.communicate("\n".join(commands).encode())
                            new_name = 'final_optim_' +key+'.txt'
                            os.rename('final_optim.chg', new_name)


                            for atom_idx in lst_atom_idxs:
                                [atom_type, partial_charge_atom] = Electrostatics.getAtomInfo(full_file_path, atom_idx)
                                results_dict[f'{key} Charge {atom_idx} {atom_type}'] = partial_charge_atom

                    except Exception as e:
                        logging.exception('An Exception was thrown')
                        continue
            allspeciesdict.append(results_dict)
        os.chdir(owd)
        df = pd.DataFrame(allspeciesdict)
        df.to_csv(partial_chg_filename +'.csv')
        return df

    def compute_dipole(res_coords, res_chgs, nuc_chgs):
        '''
           res_coords in units of angstroms, res_chgs are in a.u. units
           returns dipole in units of Debye
        '''
        convert_Debye = 4.80320427

        # Center coordinates at geometric center (simple average)
        # This makes the dipole moment origin-independent and consistent with line integral formalism
        center = np.average(res_coords, axis=0)
        coords_centered = res_coords - center
        dipole_vector = convert_Debye*np.sum(coords_centered*res_chgs[:, np.newaxis], axis=0)  #shape of (3,)
        dipole_magnitude = np.linalg.norm(dipole_vector)
        return dipole_vector, dipole_magnitude


    def getdipole_residues(self, res_dict, df):
        ''' res_dict: dictionary with strings mapped to list of lists(int)   
          strings denote name of residues which are mapped to the associated atom indices in the lists(0 indexed!)
        '''
        dipole_vectors = []
        dipole_magnitudes = []
        res_labels = []
        centroid_lst = []

        arr_coords = df[['x', 'y', 'z']].to_numpy()
        atoms = df['Atom']
        arr_chgs = np.array(df['charge'])
        for res_name in res_dict.keys():
            res_indices = res_dict[res_name]

            # Skip empty residues (no atoms)
            if len(res_indices) == 0:
                continue

            res_labels.append(res_name)
            res_coords = arr_coords[res_indices]
            res_centroid = res_coords.mean(axis=0)
            res_chgs = arr_chgs[res_indices]
            atom_vals = [atoms[i] for i in res_indices]
            nuc_chgs = [self.amassdict[atom_val][1] for atom_val in atom_vals]
            dip_vec, dip_mag = Electrostatics.compute_dipole(res_coords, res_chgs, nuc_chgs)
            dipole_vectors.append(dip_vec)
            dipole_magnitudes.append(dip_mag)
            centroid_lst.append(res_centroid)
        
        df = pd.DataFrame({
           'Label': res_labels,
           'Dipole_vec': dipole_vectors,
           'Dipole' : dipole_magnitudes,
           'centroid' : centroid_lst
        })
        return df



    def getcharge_residues(self, charge_types, res_dict, partial_chg_filename, multiwfn_path, multipole_mode=True, polarization_scheme='Hirshfeld_I'):
        '''Function computes partial charges on a select set of atoms using the charge scheme specified in charge types. Note atom indices will be carried over between csvs

        Inputs:
        -------
        charge_types: list(str)
            list of strings
        res_dict: dictionary with strings mapped to list of lists(int)   
            strings denote name of residues which are mapped to the associated atom indices in the lists(0 indexed!)
        partial_chg_filename: string
            Name of the output file name
        multiwfn_path: string
            Path to the multiwfn executable

            Path to the atmrad executable
        multipole_mode: boolean
            If True, will use multipole mode to compute partial charges, if False will use the standard method

         Outputs:
        -------
        df: Pandas Dataframe with partial charge info
        
        Notes
        -----
        Will Create a csv file entitled partial_chg_filename.csv with partial charge info
        '''
       # Access Class Variables
        list_of_file = self.lst_of_folders

        owd = os.getcwd() # Old working directory
        allspeciesdict = []
        counter = 0  # Iterator to account for atomic indices of interest

        if multipole_mode:
            final_structure_file = self.config['xyzfilename']
            # Handle xyzfilename being a list (take first element)
            if isinstance(final_structure_file, list):
                final_structure_file = final_structure_file[0]
            polarization_file = "Multipole" + polarization_scheme + ".txt"
            molden_filename = self.config['molden_filename']

            file_idx = 0
            for f in list_of_file:
                # Convert f to string if it's not already (handles int folder names)
                f = str(f)
                print('-----------------' + f + '------------------')
                counter = counter + 1
                os.chdir(owd)
                os.chdir(f)
                results_dir = os.getcwd() + '/'
                # Check if the atomic polarizations have been computed
                path_to_pol = os.path.join(os.getcwd(), polarization_file)
                backup_path_to_pol = os.path.join(os.getcwd(), "final_optim_polarization.txt")
                #Initially we assume that we do need to run the calculation
                need_to_run_calculation = True
                if os.path.exists(path_to_pol):
                    with open(path_to_pol, 'r') as file:
                        contents = file.read()
                        if "Calculation took up" in contents:
                            print(f"   > Atomic Multipoles already calculated here: {f}!!")
                            need_to_run_calculation = False
                elif os.path.exists(backup_path_to_pol):    
                    path_to_pol = backup_path_to_pol
                    with open(path_to_pol, 'r') as file:
                        contents = file.read()
                        if "Calculation took up" in contents:
                            print(f"   > Atomic Multipoles already calculated here: {f}!!")
                            need_to_run_calculation = False
                if need_to_run_calculation:
                    # Dynamically get path to package settings.ini file
                    with resources.path('pyef.resources', 'settings.ini') as ini_path:
                        path_to_ini_file = str(ini_path)
                        Command_Polarization = f"{multiwfn_path} {molden_filename} -set {path_to_ini_file} > {polarization_file}"
                        command_A = f"{multiwfn_path} final_optim.molden -set {path_to_ini_file}"
                        print('Atomic Multipole Calculation initialized')
                        # Now Run the calculation for atomic dipole and quadrupole moment
                        with resources.path('pyef.resources', 'atmrad') as atmrad_resource:
                            atmrad_src = str(atmrad_resource)
                            copy_tree(atmrad_src, os.getcwd() + '/atmrad/')
                        print(f"   > Submitting Multiwfn job using: {Command_Polarization}")
                        proc = subprocess.Popen(Command_Polarization, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
                        polarization_commands = ['15', '-1'] + self.multiwfn.dict_of_multipole[polarization_scheme] + ['0', 'q']
                        proc.communicate("\n".join(polarization_commands).encode())
              


                results_dict = {}
                results_dict['Name'] = f


                lst_multipole_dict = MultiwfnInterface.getmultipoles(path_to_pol)
                for res_name in res_dict.keys():
                   all_res_idxs = res_dict[res_name]
                   res_idxs = all_res_idxs[file_idx]
                   res_chg = 0
                   for idx in res_idxs:
                       atom_dict = lst_multipole_dict[idx]
                       chg = atom_dict["Atom_Charge"]
                       res_chg += chg
                   results_dict[f'{res_name}_{polarization_scheme}_Charge'] = res_chg
                   print(f'Current charge is: {res_chg}')
                file_idx += 1
                allspeciesdict.append(results_dict)
         
        else:
            for f in list_of_file:
                # Convert f to string if it's not already (handles int folder names)
                f = str(f)
                print('-----------------' + f + '------------------')
                counter = counter + 1
                os.chdir(owd)
                os.chdir(f)
                #working in key mode!
                for key in charge_types:
                    try:
                        full_file_path = os.getcwd() +'/final_optim_' +key+'.txt'
                        if key == "Hirshfeld_I":
                            with resources.path('pyef.resources', 'atmrad') as atmrad_resource:
                                atmrad_src = str(atmrad_resource)
                                copy_tree(atmrad_src, results_dir + 'atmrad/')
                        try:
                            for res_name in res_dict.keys():
                                res_indices = res_dict[res_name]
                                [res_charge] = Electrostatics.getAtomsInfo(full_file_path, res_indices)
                                results_dict[f'{res_name} Charge'] = res_charge

                        except Exception as e:
                            print('The Exception is: ' + str(e))
                            print(traceback.format_exc())
                            print('Error when trying to access electrostatic information: Attemtping to re-compute partial charges of type: ' + str(key))

                            # Re-run multiwfn computation of partial charge
                            proc = subprocess.Popen(command_A, stdin=subprocess.PIPE, stdout=subprocess.PIPE, shell=True)
                            calc_command = self.multiwfn.dict_of_calcs[key]
                            commands = ['7', calc_command, '1', 'y', '0', 'q'] # for atomic charge type corresponding to dict key
                            if key in ['CHELPG', 'RESP', 'MK']:
                                commands = ['7', calc_command, '1', 'y', '0', '0', 'q']
                            output = proc.communicate("\n".join(commands).encode())
                            new_name = 'final_optim_' +key+'.txt'
                            os.rename('final_optim.chg', new_name)


                            for res_name in res_dict.keys():
                                all_res_idxs = res_dict[res_name]
                                res_indices= all_res_idxs[file_idx]
                                [res_charge] = Electrostatics.getAtomsInfo(full_file_path, res_indices)
                                results_dict[f'{res_name}_{key}_Charge'] = res_charge
                                file_idx += 1
                    except Exception as e:
                        logging.exception('An Exception was thrown')
                        continue 
                allspeciesdict.append(results_dict)
        os.chdir(owd)
        df = pd.DataFrame(allspeciesdict)
        df.to_csv(partial_chg_filename +'.csv')
        return df


    def compute_atomwise_dipole_contributions(self, atomwise_Efield, df_substrate, df_env, df_all_atoms, one_mol, filename):
        """
        Compute atom-wise energy contributions from environment E-fields interacting with substrate dipoles.

        The interaction energy is: E = -μ·E (negative dot product of dipole moment with electric field)

        Parameters:
        -----------
        atomwise_Efield : dict
            Dictionary with (substrate_idx, env_idx) as keys and E-field vector [Ex, Ey, Ez] (in V/Å) as values
        df_substrate : DataFrame
            DataFrame containing substrate atom information
        df_env : DataFrame
            DataFrame containing environment atom information
        df_all_atoms : DataFrame
            DataFrame containing all atom information (for element names)
        one_mol : float
            Avogadro's number
        filename : str
            Name of the current file being processed

        Returns:
        --------
        DataFrame with columns: FileName, Env_Atom_Index, Env_Element, Total_Energy_Contribution_kcal_mol,
                                Substrate_Atom_Index, Substrate_Element, Energy_Contribution_kcal_mol
        """
        atomwise_data = []

        # Get substrate dipole mapping (dipole moments are in Bohr from multipoles)
        substrate_dipoles = {}
        substrate_elements = {}
        for _, row in df_substrate.iterrows():
            substrate_dipoles[row['Index']] = np.array(row['Dipole_Moment'])  # Expected to be [μx, μy, μz] in Bohr
            if 'Element' in df_all_atoms.columns:
                substrate_elements[row['Index']] = df_all_atoms.loc[df_all_atoms['Index'] == row['Index'], 'Element'].iloc[0]
            else:
                substrate_elements[row['Index']] = 'Unknown'

        # Get environment element mapping
        env_elements = {}
        for _, row in df_env.iterrows():
            if 'Element' in df_all_atoms.columns:
                env_elements[row['Index']] = df_all_atoms.loc[df_all_atoms['Index'] == row['Index'], 'Element'].iloc[0]
            else:
                env_elements[row['Index']] = 'Unknown'

        # Compute energy contribution for each (substrate, environment) pair
        # Energy = -μ·E where μ is dipole moment and E is electric field
        for (sub_idx, env_idx), efield_contribution in atomwise_Efield.items():
            # efield_contribution is [Ex, Ey, Ez] in V/Å
            # substrate dipole is in Bohr, need to convert to appropriate units
            dipole_bohr = substrate_dipoles[sub_idx]

            # Convert dipole from Bohr to C·m
            dipole_Cm = constants.BOHR_TO_M * constants.ELEMENTARY_CHARGE * dipole_bohr

            # Convert E-field from V/Å to V/m
            efield_Vm = efield_contribution / constants.ANGSTROM_TO_M

            # Energy = -μ·E (in Joules)
            energy_contribution_J = -np.dot(dipole_Cm, efield_Vm)

            # Convert to kcal/mol
            energy_contribution_kcal_mol = energy_contribution_J * one_mol / 4184

            atomwise_data.append({
                'FileName': filename,
                'Env_Atom_Index': env_idx,
                'Env_Element': env_elements.get(env_idx, 'Unknown'),
                'Substrate_Atom_Index': sub_idx,
                'Substrate_Element': substrate_elements.get(sub_idx, 'Unknown'),
                'Energy_Contribution_kcal_mol': energy_contribution_kcal_mol,
                'Efield_x_V_per_A': efield_contribution[0],
                'Efield_y_V_per_A': efield_contribution[1],
                'Efield_z_V_per_A': efield_contribution[2]
            })

        df_atomwise = pd.DataFrame(atomwise_data)

        # Add total contribution from each environment atom (summed over all substrate atoms)
        env_totals = df_atomwise.groupby('Env_Atom_Index')['Energy_Contribution_kcal_mol'].sum().reset_index()
        env_totals.columns = ['Env_Atom_Index', 'Total_Energy_Contribution_kcal_mol']

        # Merge back to get total contributions
        df_atomwise = df_atomwise.merge(env_totals, on='Env_Atom_Index', how='left')

        # Sort by total contribution to identify most important atoms
        df_atomwise = df_atomwise.sort_values('Total_Energy_Contribution_kcal_mol', ascending=False)

        return df_atomwise

    def visualize_atomwise_contributions(self, df_atomwise, file_path_xyz, charge_file, filename, charge_type):
        """
        Create PDB visualization of atom-wise energy contributions.

        Parameters:
        -----------
        df_atomwise : DataFrame
            DataFrame with atom-wise contribution data from _compute_atomwise_contributions
        file_path_xyz : str
            Path to XYZ file
        charge_file : str
            Path to charge/multipole file
        filename : str
            Folder/file name for output
        charge_type : str
            Type of charge calculation used
        """
        # Get total number of atoms from XYZ file
        total_atoms = Electrostatics.mapcount(file_path_xyz) - 2

        # Initialize contribution array (all atoms start at 0)
        b_factor_contributions = np.zeros(total_atoms)

        # Get unique environment atoms and their total contributions
        env_contributions = df_atomwise.groupby('Env_Atom_Index').agg({
            'Total_Energy_Contribution_kcal_mol': 'first'
        }).reset_index()

        # Fill in the environment atom contributions
        for _, row in env_contributions.iterrows():
            atom_idx = int(row['Env_Atom_Index'])
            contribution = row['Total_Energy_Contribution_kcal_mol']
            b_factor_contributions[atom_idx] = contribution

        # Create PDB file with contributions in B-factor column
        pdb_name = f'estab_{charge_type}_{filename}_atomwise.pdb'
        Visualize(file_path_xyz).makePDB(charge_file, b_factor_contributions, pdb_name)
        print(f"Created visualization PDB: {pdb_name}")

    def compute_atomwise_contributions(self, atomwise_ESP, df_substrate, df_env, df_all_atoms, one_mol, filename):
        """
        Compute atom-wise energy contributions from environment atoms to substrate stabilization.

        Parameters:
        -----------
        atomwise_ESP : dict
            Dictionary with (substrate_idx, env_idx) as keys and ESP contribution (in Volts) as values
        df_substrate : DataFrame
            DataFrame containing substrate atom information
        df_env : DataFrame
            DataFrame containing environment atom information
        df_all_atoms : DataFrame
            DataFrame containing all atom information (for element names)
        one_mol : float
            Avogadro's number
        filename : str
            Name of the current file being processed

        Returns:
        --------
        DataFrame with columns: FileName, Env_Atom_Index, Env_Element, Total_Energy_Contribution_kcal_mol,
                                Substrate_Atom_Index, Substrate_Element, Energy_Contribution_kcal_mol
        """
        atomwise_data = []

        # Get substrate charge mapping
        substrate_charges = {}
        substrate_elements = {}
        for _, row in df_substrate.iterrows():
            substrate_charges[row['Index']] = row['Atom_Charge']
            if 'Element' in df_all_atoms.columns:
                substrate_elements[row['Index']] = df_all_atoms.loc[df_all_atoms['Index'] == row['Index'], 'Element'].iloc[0]
            else:
                substrate_elements[row['Index']] = 'Unknown'

        # Get environment element mapping
        env_elements = {}
        for _, row in df_env.iterrows():
            if 'Element' in df_all_atoms.columns:
                env_elements[row['Index']] = df_all_atoms.loc[df_all_atoms['Index'] == row['Index'], 'Element'].iloc[0]
            else:
                env_elements[row['Index']] = 'Unknown'

        # Compute energy contribution for each (substrate, environment) pair
        # Energy = ESP * charge_substrate (in atomic units, then convert to kcal/mol)
        for (sub_idx, env_idx), esp_contribution in atomwise_ESP.items():
            # ESP is in Volts, charge is in elementary charges
            # Energy = ESP (V) * charge (e) = ESP (J/C) * charge (e) * e (C/e) = Energy (J)
            # Convert to kcal/mol: multiply by Avogadro's number and divide by 4184 J/kcal
            energy_contribution_J = esp_contribution * substrate_charges[sub_idx] * constants.ELEMENTARY_CHARGE
            energy_contribution_kcal_mol = energy_contribution_J * one_mol / 4184

            atomwise_data.append({
                'FileName': filename,
                'Env_Atom_Index': env_idx,
                'Env_Element': env_elements.get(env_idx, 'Unknown'),
                'Substrate_Atom_Index': sub_idx,
                'Substrate_Element': substrate_elements.get(sub_idx, 'Unknown'),
                'Energy_Contribution_kcal_mol': energy_contribution_kcal_mol,
                'ESP_Contribution_V': esp_contribution
            })

        df_atomwise = pd.DataFrame(atomwise_data)

        # Add total contribution from each environment atom (summed over all substrate atoms)
        env_totals = df_atomwise.groupby('Env_Atom_Index')['Energy_Contribution_kcal_mol'].sum().reset_index()
        env_totals.columns = ['Env_Atom_Index', 'Total_Energy_Contribution_kcal_mol']

        # Merge back to get total contributions
        df_atomwise = df_atomwise.merge(env_totals, on='Env_Atom_Index', how='left')

        # Sort by total contribution to identify most important atoms
        df_atomwise = df_atomwise.sort_values('Total_Energy_Contribution_kcal_mol', ascending=False)

        return df_atomwise
    

    def build_interaction_tensor(self, r_i, r_j, multipole_order, dielectric):
        """
        Build the multipole interaction tensor T_ij using AMOEBA formalism.

        This tensor contains derivatives of 1/r and enables computation of all
        multipole-multipole interactions via: E = M_i^T · T_ij · M_j

        Parameters:
        -----------
        r_i, r_j : array-like
            Position vectors of atoms i and j in Angstroms
        multipole_order : int
            Maximum multipole order (1=monopole, 2=+dipole, 3=+quadrupole)
        dielectric : float
            Dielectric constant

        Returns:
        --------
        T : ndarray
            Interaction tensor with dimensions based on multipole_order
            - Order 1: 1×1 (monopole-monopole only)
            - Order 2: 4×4 (monopole + dipole)
            - Order 3: 9×9 (monopole + dipole + quadrupole, traceless)

        Notes:
        ------
        Tensor elements represent spatial derivatives of Coulomb potential:
        - T[0,0]: 1/r (monopole-monopole)
        - T[0,1:4]: ∂(1/r)/∂r_j (monopole-dipole)
        - T[1:4,1:4]: ∂²(1/r)/∂r_i∂r_j (dipole-dipole)
        - Higher orders follow similar pattern
        """
        # Convert to meters and compute distance
        r_vec = (r_j - r_i) * constants.ANGSTROM_TO_M
        r = np.linalg.norm(r_vec)
        r_hat = r_vec / r  # Unit vector from i to j

        # Coulomb prefactor with dielectric
        # Note: charges are dimensionless (in units of e), so we need e² for energy
        k = constants.COULOMB_CONSTANT * (constants.ELEMENTARY_CHARGE ** 2) / dielectric

        if multipole_order == 1:
            # Monopole-monopole only: T = k/r
            T = np.array([[k / r]])

        elif multipole_order == 2:
            # Monopole + dipole interactions (4×4 tensor)
            T = np.zeros((4, 4))

            # T[0,0]: monopole-monopole interaction (q_i × q_j / r)
            T[0, 0] = k / r

            # T[0, 1:4]: monopole-dipole interaction (∂/∂r_j of 1/r)
            # Derivative: ∂(1/r)/∂r_j = -r_hat/r²
            T[0, 1:4] = -k * r_hat / (r**2)

            # T[1:4, 0]: dipole-monopole interaction (∂/∂r_i of 1/r)
            # By symmetry with opposite sign: ∂(1/r)/∂r_i = +r_hat/r²
            T[1:4, 0] = k * r_hat / (r**2)

            # T[1:4, 1:4]: dipole-dipole interaction (∂²/∂r_i∂r_j of 1/r)
            # Formula: (3·r_hat⊗r_hat - I)/r³
            I = np.eye(3)
            T[1:4, 1:4] = k * (3 * np.outer(r_hat, r_hat) - I) / (r**3)

        elif multipole_order == 3:
            # Monopole + dipole + quadrupole interactions (9×9 tensor)
            T = np.zeros((9, 9))

            # Monopole-monopole (same as order 2)
            T[0, 0] = k / r

            # Monopole-dipole (same as order 2)
            T[0, 1:4] = -k * r_hat / (r**2)
            T[1:4, 0] = k * r_hat / (r**2)

            # Dipole-dipole (same as order 2)
            I = np.eye(3)
            T[1:4, 1:4] = k * (3 * np.outer(r_hat, r_hat) - I) / (r**3)

            # Monopole-quadrupole: T[0, 4:9] and T[4:9, 0]
            # Uses traceless quadrupole representation (5 independent components)
            # ∂²(1/r)/∂r_α∂r_β derivatives
            for idx, (alpha, beta) in enumerate([(0,0), (0,1), (0,2), (1,1), (1,2)]):
                # Monopole-quadrupole term
                delta_ab = 1.0 if alpha == beta else 0.0
                T_mq = k * (3*r_hat[alpha]*r_hat[beta] - delta_ab) / (r**3)
                T[0, 4+idx] = T_mq
                T[4+idx, 0] = T_mq

            # Dipole-quadrupole: T[1:4, 4:9] and T[4:9, 1:4]
            # Third derivatives: ∂³(1/r)/∂r_i∂r_α∂r_β
            for i in range(3):
                for idx, (alpha, beta) in enumerate([(0,0), (0,1), (0,2), (1,1), (1,2)]):
                    delta_ia = 1.0 if i == alpha else 0.0
                    delta_ib = 1.0 if i == beta else 0.0
                    delta_ab = 1.0 if alpha == beta else 0.0

                    T_dq = k * (-15*r_hat[i]*r_hat[alpha]*r_hat[beta]
                               + 3*(delta_ia*r_hat[beta] + delta_ib*r_hat[alpha] + delta_ab*r_hat[i])) / (r**4)
                    T[1+i, 4+idx] = T_dq
                    T[4+idx, 1+i] = T_dq

            # Quadrupole-quadrupole: T[4:9, 4:9]
            # Fourth derivatives: ∂⁴(1/r)/∂r_α∂r_β∂r_γ∂r_δ
            for idx1, (alpha, beta) in enumerate([(0,0), (0,1), (0,2), (1,1), (1,2)]):
                for idx2, (gamma, delta) in enumerate([(0,0), (0,1), (0,2), (1,1), (1,2)]):
                    delta_ab = 1.0 if alpha == beta else 0.0
                    delta_ag = 1.0 if alpha == gamma else 0.0
                    delta_ad = 1.0 if alpha == delta else 0.0
                    delta_bg = 1.0 if beta == gamma else 0.0
                    delta_bd = 1.0 if beta == delta else 0.0
                    delta_gd = 1.0 if gamma == delta else 0.0

                    T_qq = k * (105*r_hat[alpha]*r_hat[beta]*r_hat[gamma]*r_hat[delta]
                               - 15*(r_hat[alpha]*r_hat[beta]*delta_gd + r_hat[gamma]*r_hat[delta]*delta_ab
                                    + r_hat[alpha]*r_hat[gamma]*delta_bd + r_hat[alpha]*r_hat[delta]*delta_bg
                                    + r_hat[beta]*r_hat[gamma]*delta_ad + r_hat[beta]*r_hat[delta]*delta_ag)
                               + 3*(delta_ab*delta_gd + delta_ag*delta_bd + delta_ad*delta_bg)) / (r**5)
                    T[4+idx1, 4+idx2] = T_qq
        else:
            raise ValueError(f"multipole_order must be 1, 2, or 3. Got {multipole_order}")

        return T

    def build_multipole_vector(self, atom_data, multipole_order):
        """
        Build multipole moment vector M for an atom using AMOEBA formalism.

        Parameters:
        -----------
        atom_data : dict or pd.Series
            Atom data containing 'Atom_Charge', 'Dipole_Moment', 'Quadrupole_Moment'
        multipole_order : int
            Maximum multipole order (1=monopole, 2=+dipole, 3=+quadrupole)

        Returns:
        --------
        M : ndarray
            Multipole vector [q, μ_x, μ_y, μ_z, Q_xx, Q_xy, ...] in SI-based units
            - Order 1: [q] (1 element)
            - Order 2: [q, μ_x, μ_y, μ_z] (4 elements)
            - Order 3: [q, μ_x, μ_y, μ_z, Q_xx, Q_xy, Q_xz, Q_yy, Q_yz] (9 elements, traceless)

        Notes:
        ------
        - Charges are dimensionless (in units of elementary charge)
        - Dipoles are converted from Bohr to meters
        - Quadrupoles are converted from Bohr² to meters²
        - Uses traceless quadrupole representation (Q_zz = -Q_xx - Q_yy)
        """
        M = [atom_data['Atom_Charge']]  # Monopole (dimensionless, in units of e)

        if multipole_order >= 2:
            # Add dipole components (convert from atomic units to SI)
            # Multiwfn outputs dipole in Bohr (atomic units)
            # In a.u., dipole has units of e·a₀, so we only convert length
            dipole_bohr = np.array(atom_data['Dipole_Moment'])
            dipole_m = dipole_bohr * constants.BOHR_TO_M  # Now in units of e·meters
            M.extend(dipole_m)

        if multipole_order >= 3:
            # Add quadrupole components (convert from Bohr² to m²)
            # Use traceless representation: 5 independent components
            Q = atom_data['Quadrupole_Moment']
            Q_m2 = Q * (constants.BOHR_TO_M**2)

            # Extract 5 independent components (Q is symmetric and traceless)
            # Components: Q_xx, Q_xy, Q_xz, Q_yy, Q_yz
            # (Q_zz is determined by traceless condition: Q_zz = -Q_xx - Q_yy)
            M.extend([Q_m2[0, 0], Q_m2[0, 1], Q_m2[0, 2], Q_m2[1, 1], Q_m2[1, 2]])

        return np.array(M)

    def getElectrostatic_stabilization(self, multiwfn_path=None,
                                              substrate_idxs=[], charge_type='Hirshfeld_I',
                                              name_dataStorage='estatic', env_idxs=None,
                                              save_atomwise_decomposition=False, visualize=None,
                                              multipole_order=2, substrate_multipole_order=None,
                                              env_multipole_order=None, dielectric=1, dielectric_scale=1,
                                              includePtChgs=None, ptchg_paths=None,
                                              pdb_charge_paths=None):
        """
        Compute electrostatic stabilization using AMOEBA-style multipole tensor formalism.

        This function implements the complete multipole expansion for intermolecular
        electrostatic interactions using the tensor formalism:

            E_total = Σᵢ∈substrate Σⱼ∈environment M_i^T · T_ij · M_j

        where M is the multipole vector and T_ij is the interaction tensor containing
        all derivatives of the Coulomb potential. This naturally includes ALL
        multipole-multipole interaction terms:

        - Order 1 (monopole): charge-charge (q×q)
        - Order 2 (+ dipole): + charge-dipole (q×μ) + dipole-dipole (μ×μ)
        - Order 3 (+ quadrupole): + charge-quadrupole (q×Q) + dipole-quadrupole (μ×Q) + quadrupole-quadrupole (Q×Q)

        This is the formalism used in the AMOEBA polarizable force field and provides
        a complete treatment of distributed multipole interactions.

        Parameters:
        -----------
        multiwfn_path : str
            Path to Multiwfn executable
        substrate_idxs : list
            List of substrate atom indices (the molecule being stabilized)
        charge_type : str, optional
            Charge partitioning scheme (default: 'Hirshfeld_I')
        name_dataStorage : str, optional
            Output filename prefix (default: 'estaticFile_tensor')
        env_idxs : list or None, optional
            List of environment atom indices. If None, uses all non-substrate atoms (default: None)
        save_atomwise_decomposition : bool, optional
            If True, save atom-wise decomposition to CSV showing each substrate atom's contribution (default: False)
        visualize : bool or None, optional
            If True, create PDB files with energy contributions in B-factor column.
            None uses config defaults (default: None)
        multipole_order : int, optional
            Order of multipole expansion:
            - 1: monopole only (charge-charge)
            - 2: monopole + dipole (includes charge-charge, charge-dipole, dipole-dipole)
            - 3: monopole + dipole + quadrupole (includes all terms up to Q×Q)
            (default: 2)
        substrate_multipole_order : int or None, optional
            Multipole order for substrate atoms. If None, uses multipole_order.
            Set to 1 for QM/MM where substrate is MM (charges only). (default: None)
        env_multipole_order : int or None, optional
            Multipole order for environment atoms. If None, uses multipole_order.
            Set to 1 for QM/MM where environment is MM (charges only). (default: None)
        dielectric : float, optional
            Dielectric constant (default: 1).
        dielectric_scale : float, optional
            Scaling factor for point charges. Point charge contributions are scaled by
            1/sqrt(dielectric_scale). Typically used in QM/MM calculations (default: 1).
        includePtChgs : bool or None, optional
            Include point charges as environment atoms only.
            If None, uses object-level setting (default: None).
            Point charges are ONLY added to environment, never to substrate.
        ptchg_paths : list of str or None, optional
            Override point charge file paths for this calculation.
            List of paths (one per job), with None for jobs without point charges.
            Point charges are treated as monopole-only environment atoms.
            If provided, takes precedence over object-level settings (default: None).
        pdb_charge_paths : str or list of str or None, optional
            If provided, use partial charges from PDB B-factor columns only (no Multiwfn).
            Monopole-only (multipole_order is forced to 1). Coordinates are also read from
            the PDB, so no separate xyz file is needed.
            Overrides the object-level pdb_charge_paths when the object was created in pdb_only mode.

        Returns:
        --------
        pd.DataFrame or tuple of pd.DataFrame
            If save_atomwise_decomposition=False: returns DataFrame with total stabilization energies
            If save_atomwise_decomposition=True: returns (total_df, atomwise_df) tuple

        Output Files
        ------------
        **Main CSV file:** Auto-generated with format:
        - Monopole: `estab_{charge_type}.csv`
        - Multipole: `estab_multipole{N}_{charge_type}.csv` (N = multipole order)
        - Examples: `estab_Hirshfeld_I.csv`, `estab_multipole2_Hirshfeld_I.csv`
        - Contains total electrostatic stabilization energies (kcal/mol) for all jobs
        - Columns: job name, total energy, substrate/environment details, etc.

        **Atomwise CSV (if save_atomwise_decomposition=True):**
        - Format: `estab_{charge_type}_atomwise.csv` or `estab_multipole{N}_{charge_type}_atomwise.csv`
        - Examples: `estab_Hirshfeld_I_atomwise.csv`, `estab_multipole2_Hirshfeld_I_atomwise.csv`
        - Per-atom energy contributions showing which atoms contribute most
        - Columns: job, atom index, energy contribution, coordinates, etc.

        **PDB file (if visualize=True):** `estab_{charge_type}_{structure}_tensor_sub{N}_env{M}.pdb`
        - B-factor contains energy contribution from each environment atom to substrate
        - N = substrate multipole order, M = environment multipole order
        - Example: `estab_Hirshfeld_I_complex_tensor_sub2_env1.pdb`

        Note: If you provide a custom filename (not 'output', 'estatic', etc.),
        your filename will be used instead of automatic naming.

        Notes:
        ------
        **Comparison with other methods:**
        - `getElectrostatic_stabilization()`: Computes q·V (substrate charge in environment potential)
        - `get_Electrostatic_stabilization_dipole()`: Computes -μ·E (substrate dipole in environment field)
        - This function: Computes DIRECT pairwise multipole-multipole interactions

        **Advantages of tensor formalism:**
        - Complete: automatically includes all multipole-multipole terms
        - Symmetric: properly treats both substrate and environment multipoles
        - Efficient: single matrix multiplication per atom pair
        - Extensible: easy to add higher-order terms

        **Important:**
        - Requires Multiwfn multipole analysis for orders ≥ 2
        - **Automatic fallback**: If multipole files don't exist, automatically uses charges-only files
        - **QM/MM support**: Substrate and environment can have different multipole orders (e.g., QM substrate with multipoles, MM environment with charges only)
        - Results are the electrostatic STABILIZATION energy (environment → substrate)
        - Energy is in kcal/mol
        - Follows AMOEBA force field formalism (Ren & Ponder, J. Phys. Chem. B 2003)

        Examples:
        ---------
        >>> # Compute monopole + dipole + dipole-dipole interactions
        >>> df = estat.get_Electrostatic_stabilization_tensor(
        ...     multiwfn_path,
        ...     substrate_idxs=[[0, 1, 2]],  # 3-atom substrate
        ...     multipole_order=2  # Include dipole-dipole terms
        ... )

        >>> # Include quadrupole terms with atom-wise decomposition
        >>> df, df_atomwise = estat.get_Electrostatic_stabilization_tensor(
        ...     multiwfn_path,
        ...     substrate_idxs=[[0, 1, 2]],
        ...     multipole_order=3,
        ...     save_atomwise_decomposition=True
        ... )

        >>> # QM/MM: QM substrate (with dipoles) interacting with MM environment (charges only)
        >>> df = estat.get_Electrostatic_stabilization_tensor(
        ...     multiwfn_path,
        ...     substrate_idxs=[[0, 1, 2]],  # QM region
        ...     substrate_multipole_order=2,  # QM: charges + dipoles
        ...     env_multipole_order=1         # MM: charges only
        ... )
        """
        # Resolve pdb_charge_paths: function-level arg takes priority, else use object-level
        if pdb_charge_paths is None and self.pdb_only:
            pdb_charge_paths = self.pdb_charge_paths
        use_pdb_charges = pdb_charge_paths is not None

        # In pdb_only mode, only monopole interactions are supported
        if use_pdb_charges:
            multipole_order = 1
            substrate_multipole_order = 1
            env_multipole_order = 1

        # Validate required parameters
        if substrate_idxs is None or (isinstance(substrate_idxs, list) and len(substrate_idxs) == 0):
            raise ValueError(f"""
{'='*60}
ERROR: Electrostatic stabilization requires substrate indices
{'='*60}
substrate_idxs parameter is required but not provided.

Correct usage:
  estab_df = estat.getElectrostatic_stabilization(
      multiwfn_path='/path/to/multiwfn',
      substrate_idxs=[0, 1, 2, 3],  # Active site atoms
      env_idxs=[10, 11, 12],        # Environment atoms (optional)
      multipole_order=2
  )

Note: Atom indices are 0-based (first atom = index 0)

For complete examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
- README.md: Section 3.3 (Electrostatic Stabilization)
{'='*60}
""")

        # Validate charge type (skip in pdb_only mode — no Multiwfn scheme needed)
        if not use_pdb_charges:
            validate_charge_type(charge_type, context="getElectrostatic_stabilization")

        # Validate multipole orders
        validate_numeric_range(multipole_order, "multipole_order", allowed_values=[1, 2, 3],
                             context="getElectrostatic_stabilization")
        if substrate_multipole_order is not None:
            validate_numeric_range(substrate_multipole_order, "substrate_multipole_order",
                                 allowed_values=[1, 2, 3], context="getElectrostatic_stabilization")
        if env_multipole_order is not None:
            validate_numeric_range(env_multipole_order, "env_multipole_order",
                                 allowed_values=[1, 2, 3], context="getElectrostatic_stabilization")

        # Validate substrate and environment indices don't overlap
        if env_idxs is not None and substrate_idxs is not None:
            from .validation import check_index_overlap
            # Handle both flat lists and nested lists
            flat_substrate = substrate_idxs
            if substrate_idxs and isinstance(substrate_idxs[0], (list, tuple)):
                flat_substrate = [idx for sublist in substrate_idxs for idx in sublist]

            flat_env = env_idxs
            if env_idxs and isinstance(env_idxs[0], (list, tuple)):
                flat_env = [idx for sublist in env_idxs for idx in sublist]

            check_index_overlap(flat_substrate, flat_env, "substrate_idxs", "env_idxs",
                              context="electrostatic stabilization")

        self.config['dielectric'] = dielectric
        self.config['dielectric_scale'] = dielectric_scale
        dielectric = self.config['dielectric']
        list_of_file = self.lst_of_folders
        final_structure_file = self.config['xyzfilename']
        # Handle xyzfilename being a list (take first element)
        if isinstance(final_structure_file, list):
            final_structure_file = final_structure_file[0]

        # Handle visualization settings
        if visualize is None:
            viz_contributions = self.config.get('visualize_estabilization', False)
        else:
            viz_contributions = visualize

        owd = os.getcwd()
        allspeciesdict = []
        all_atomwise_data = []
        counter = 0
        one_mol = 6.02e23

        # Set default multipole orders for substrate and environment
        if substrate_multipole_order is None:
            substrate_multipole_order = multipole_order
        if env_multipole_order is None:
            env_multipole_order = multipole_order

        # Determine if we need multipole analysis (orders 2 and 3 need dipoles/quadrupoles)
        # We need multipoles if EITHER substrate or environment needs them
        need_multipoles = (substrate_multipole_order >= 2 or env_multipole_order >= 2)

        for f in list_of_file:
            # Convert f to string if it's not already (handles int folder names)
            f = str(f)
            substrate_idx = substrate_idxs[counter]
            results_dict = {}
            # Use stored absolute path instead of constructing it
            file_path_xyz = self.xyz_paths[counter]

            if use_pdb_charges:
                # pdb_only: load everything from the PDB file
                if isinstance(pdb_charge_paths, list):
                    pdb_charge_path = pdb_charge_paths[counter] if counter < len(pdb_charge_paths) else None
                else:
                    pdb_charge_path = pdb_charge_paths
                if pdb_charge_path is None:
                    print(f"Warning: No PDB charge file for job {counter}, skipping")
                    counter += 1
                    continue
                pdb_charge_path = os.path.abspath(pdb_charge_path)
                if not os.path.exists(pdb_charge_path):
                    raise FileNotFoundError(f"PDB charge file not found: {pdb_charge_path}")
                df_pdb = self.getPdbCharges(pdb_charge_path)
                n_atoms = len(df_pdb)
                init_all_lines = np.arange(0, n_atoms)

                # Build df_all (monopole-only) from PDB DataFrame
                df_all = df_pdb.rename(columns={'Atom': 'Element', 'charge': 'Atom_Charge'}).copy()
                df_all["Index"] = range(0, n_atoms)
                df_all['Dipole_Moment'] = [[0.0, 0.0, 0.0]] * n_atoms
                df_all['Quadrupole_Moment'] = [np.zeros((3, 3))] * n_atoms

                # Build df_geom from PDB coordinates (same column names as Geometry.getGeomInfo())
                df_geom = df_pdb.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'})[['Atom', 'X', 'Y', 'Z']].reset_index(drop=True)
                df_geom.index = range(len(df_geom))

            else:
                total_lines = Electrostatics.mapcount(file_path_xyz)
                init_all_lines = np.arange(0, total_lines - 2)

            if not env_idxs:
                env_idx = [x for x in init_all_lines if x not in substrate_idx]
            else:
                env_idx = env_idxs[counter]

            if use_pdb_charges:
                # Skip partitionCharge — df_all already built above
                pass
            else:
                # Partition charges (with multipole analysis if needed)
                # Extract actual filenames from paths
                molden_filename = os.path.basename(self.molden_paths[counter])
                xyz_filename = os.path.basename(self.xyz_paths[counter])
                comp_cost = self.multiwfn.partitionCharge(need_multipoles, f,
                                                multiwfn_path, charge_type, owd,
                                                molden_filename=molden_filename, xyz_filename=xyz_filename)

                if comp_cost == -1:
                    print(f"Warning: Charge calculation failed for {f}, skipping")
                    counter += 1
                    continue

                # Get geometry information
                geom = Geometry(file_path_xyz)
                df_geom = geom.getGeomInfo()

            if use_pdb_charges:
                # pdb_only: df_all and df_geom already built above; just build multipole_dict
                multipole_dict = {int(row['Index']): row for _, row in df_all.iterrows()}
                multipole_available = False
            else:
                # Load charge/multipole data with automatic fallback
                # Use directory path directly (f is already absolute path from lst_of_folders)
                multipole_name = f"{f}/Multipole{charge_type}.txt"
                monopole_name = f"{f}/Charges{charge_type}.txt"

                # Try to load multipole file first, fall back to charges if not available
                multipole_available = os.path.exists(multipole_name)

                if need_multipoles and not multipole_available:
                    print(f"Warning: Multipole file not found for {f}, falling back to charges-only")
                    print(f"  Expected path: {multipole_name}")
                    print(f"  Requested multipole_order={multipole_order}, but only charges available")
                    print(f"  Setting both substrate and environment to monopole-only")
                    substrate_multipole_order = 1
                    env_multipole_order = 1
                    need_multipoles = False

                if multipole_available and need_multipoles:
                    # Load multipole data
                    print(f"Loading multipole data from: {multipole_name}")
                    atomicDict = MultiwfnInterface.getmultipoles(multipole_name)
                    if not atomicDict:
                        print(f"Warning: Multipole file exists but parsing failed for {f}")
                        print(f"  File path: {multipole_name}")
                        print(f"  Falling back to charges-only mode")
                        substrate_multipole_order = 1
                        env_multipole_order = 1
                        need_multipoles = False
                        # Load charges instead
                        df_all = pd.read_csv(monopole_name, sep='\s+',
                                            names=["Element", 'x', 'y', 'z', "Atom_Charge"])
                        df_all["Index"] = range(0, len(df_all))
                        df_all['Dipole_Moment'] = [[0.0, 0.0, 0.0]] * len(df_all)
                        df_all['Quadrupole_Moment'] = [np.zeros((3, 3))] * len(df_all)
                        multipole_dict = {int(row['Index']): row for _, row in df_all.iterrows()}
                    else:
                        df_all = pd.DataFrame(atomicDict)
                        df_all['Index'] = df_all['Index'].astype(int) - 1
                        multipole_dict = {int(row['Index']): row for _, row in df_all.iterrows()}
                else:
                    # Load charges-only data
                    df_all = pd.read_csv(monopole_name, sep='\s+',
                                        names=["Element", 'x', 'y', 'z', "Atom_Charge"])
                    df_all["Index"] = range(0, len(df_all))
                    df_all['Dipole_Moment'] = [[0.0, 0.0, 0.0]] * len(df_all)
                    df_all['Quadrupole_Moment'] = [np.zeros((3, 3))] * len(df_all)
                    multipole_dict = {row['Index']: row for _, row in df_all.iterrows()}

            # Now handle substrate and environment separately for QM/MM support
            df_substrate = df_all[df_all["Index"].isin(substrate_idx)].copy()
            df_env = df_all[df_all["Index"].isin(env_idx)].copy()

            # If substrate is monopole-only but we loaded multipoles, zero out higher terms
            if substrate_multipole_order == 1 and multipole_available:
                df_substrate['Dipole_Moment'] = [[0.0, 0.0, 0.0]] * len(df_substrate)
                df_substrate['Quadrupole_Moment'] = [np.zeros((3, 3))] * len(df_substrate)

            # If environment is monopole-only but we loaded multipoles, zero out higher terms
            if env_multipole_order == 1 and multipole_available:
                df_env['Dipole_Moment'] = [[0.0, 0.0, 0.0]] * len(df_env)
                df_env['Quadrupole_Moment'] = [np.zeros((3, 3))] * len(df_env)

            # Handle point charges - add to environment only as monopoles
            ptchg_path = None
            should_include_ptchg = includePtChgs if includePtChgs is not None else self.config.get('includePtChgs', False)

            if should_include_ptchg:
                # Get path for this specific job
                if ptchg_paths is not None:
                    # Function-level override
                    if not isinstance(ptchg_paths, list):
                        raise TypeError("ptchg_paths must be a list of paths or None values")
                    if counter >= len(ptchg_paths):
                        raise IndexError(f"Job index {counter} out of range for ptchg_paths (length {len(ptchg_paths)})")
                    ptchg_path = ptchg_paths[counter]
                    if ptchg_path is not None:
                        ptchg_path = os.path.abspath(ptchg_path)
                else:
                    # Use helper method (handles list and filename modes)
                    ptchg_path = self._get_ptchg_path_for_job(counter, f, override_path=None)

                if ptchg_path is not None:
                    if not os.path.exists(ptchg_path):
                        if self.config.get('skip_missing_files', False):
                            print(f"""
{'='*60}
WARNING: Point charge file not found - skipping this calculation
{'='*60}
Expected path: {ptchg_path}
Job index: {counter}
Job folder: {f}

Skipping to next job (skip_missing_files=True)
{'='*60}
""")
                            counter += 1
                            continue
                        else:
                            raise FileNotFoundError(f"""
{'='*60}
ERROR: Point charge file not found
{'='*60}
Expected path: {ptchg_path}
Job index: {counter}
Job folder: {f}

Please verify the file exists or set the path to None for jobs without point charges.
{'='*60}
""")

                    df_ptchg = self.getPtChgs(ptchg_path)
                    print(f"  Loaded {len(df_ptchg)} point charges from: {ptchg_path}")

                    # Add point charges to environment DataFrame
                    # Point charges are monopole-only (no dipoles/quadrupoles)
                    ptchg_start_idx = len(df_all)  # Start index for point charges

                    # Create point charge entries with zero multipoles
                    ptchg_entries = []
                    for idx, row in df_ptchg.iterrows():
                        ptchg_entry = {
                            'Index': ptchg_start_idx + idx,
                            'Element': 'pnt',  # Point charge marker
                            'x': row['x'],
                            'y': row['y'],
                            'z': row['z'],
                            'Atom_Charge': row['charge'],
                            'Dipole_Moment': [0.0, 0.0, 0.0],
                            'Quadrupole_Moment': np.zeros((3, 3))
                        }
                        ptchg_entries.append(ptchg_entry)

                    # Append to df_env
                    df_ptchg_formatted = pd.DataFrame(ptchg_entries)
                    df_env = pd.concat([df_env, df_ptchg_formatted], ignore_index=True)

                    # Update environment indices to include point charges
                    ptchg_indices = list(range(ptchg_start_idx, ptchg_start_idx + len(df_ptchg)))
                    env_idx = list(env_idx) + ptchg_indices

                    print(f"  Added point charges to environment (indices {ptchg_start_idx}-{ptchg_start_idx + len(df_ptchg) - 1})")

            # Compute multipole tensor interactions
            total_energy = 0.0
            atomwise_contributions = []
            missing_env_atoms = set()  # Track missing environment atoms

            # Loop over all substrate-environment atom pairs
            for sub_idx in substrate_idx:
                sub_atom_energy = 0.0

                # Get substrate atom data using dictionary lookup for multipoles or direct dataframe for monopoles
                if need_multipoles:
                    if sub_idx not in multipole_dict:
                        print(f"Warning: Substrate atom {sub_idx} not found in multipole data, skipping")
                        continue
                    sub_row = multipole_dict[sub_idx]
                else:
                    sub_row = df_substrate[df_substrate['Index'] == sub_idx].iloc[0]

                # Get substrate atom position
                r_i = np.array([df_geom['X'][sub_idx],
                               df_geom['Y'][sub_idx],
                               df_geom['Z'][sub_idx]])

                # Build substrate multipole vector with substrate-specific order
                M_i = self.build_multipole_vector(sub_row, substrate_multipole_order)

                for env_idx_atom in env_idx:
                    # Get environment atom data using dictionary lookup for multipoles or direct dataframe for monopoles
                    if need_multipoles:
                        if env_idx_atom not in multipole_dict:
                            missing_env_atoms.add(env_idx_atom)
                            continue
                        env_row = multipole_dict[env_idx_atom]
                    else:
                        env_row = df_env[df_env['Index'] == env_idx_atom].iloc[0]

                    # Get environment atom position
                    r_j = np.array([df_geom['X'][env_idx_atom],
                                   df_geom['Y'][env_idx_atom],
                                   df_geom['Z'][env_idx_atom]])

                    # Build environment multipole vector with environment-specific order
                    M_j = self.build_multipole_vector(env_row, env_multipole_order)

                    # Build interaction tensor with max order (tensor must accommodate both)
                    tensor_order = max(substrate_multipole_order, env_multipole_order)
                    T_ij = self.build_interaction_tensor(r_i, r_j, tensor_order, dielectric)

                    # Pad multipole vectors to match tensor dimensions if needed
                    # This handles QM/MM cases where substrate and environment have different orders
                    if len(M_i) < len(M_j):
                        M_i_padded = np.pad(M_i, (0, len(M_j) - len(M_i)))
                        M_j_padded = M_j
                    elif len(M_j) < len(M_i):
                        M_i_padded = M_i
                        M_j_padded = np.pad(M_j, (0, len(M_i) - len(M_j)))
                    else:
                        M_i_padded = M_i
                        M_j_padded = M_j

                    # Compute pairwise interaction: E = M_i^T · T_ij · M_j
                    E_pair_J = M_i_padded.T @ T_ij @ M_j_padded  # Energy in Joules
                    E_pair_kcal_mol = E_pair_J * one_mol / 4184.0  # Convert to kcal/mol

                    sub_atom_energy += E_pair_kcal_mol
                    total_energy += E_pair_kcal_mol

                # Store atom-wise decomposition if requested
                if save_atomwise_decomposition:
                    atom_data = {
                        'FileName': f,
                        'Atom_Index': sub_idx,
                        'Atom_Symbol': df_geom.loc[df_geom.index == sub_idx, 'Atom'].iloc[0],
                        'Charge': sub_row['Atom_Charge'],
                        'Energy_Contribution_kcal_mol': sub_atom_energy
                    }

                    # Add multipole information if substrate has multipoles
                    if substrate_multipole_order >= 2:
                        dipole = np.array(sub_row['Dipole_Moment'])
                        atom_data.update({
                            'Dipole_x_Bohr': dipole[0],
                            'Dipole_y_Bohr': dipole[1],
                            'Dipole_z_Bohr': dipole[2],
                            'Dipole_magnitude_Bohr': np.linalg.norm(dipole)
                        })

                    if substrate_multipole_order >= 3:
                        Q = sub_row['Quadrupole_Moment']
                        atom_data.update({
                            'Quadrupole_xx_Bohr2': Q[0, 0],
                            'Quadrupole_yy_Bohr2': Q[1, 1],
                            'Quadrupole_zz_Bohr2': Q[2, 2],
                            'Quadrupole_xy_Bohr2': Q[0, 1]
                        })

                    atomwise_contributions.append(atom_data)

            # Print summary of missing environment atoms if any
            if missing_env_atoms:
                min_idx = min(missing_env_atoms)
                max_idx = max(missing_env_atoms)
                print(f"Warning: {len(missing_env_atoms)} environment atoms not found in multipole data (indices {min_idx}-{max_idx}), skipped")

            # Store results for this file
            results_dict['Total_Energy_kcal_mol'] = total_energy
            results_dict['Substrate_Multipole_Order'] = substrate_multipole_order
            results_dict['Environment_Multipole_Order'] = env_multipole_order
            results_dict['Num_Substrate_Atoms'] = len(substrate_idx)
            results_dict['Num_Environment_Atoms'] = len(env_idx)
            results_dict['FileName'] = f
            allspeciesdict.append(results_dict)

            if save_atomwise_decomposition:
                all_atomwise_data.extend(atomwise_contributions)

                # Visualize atom-wise contributions if requested
                if viz_contributions:
                    df_atomwise_temp = pd.DataFrame(atomwise_contributions)
                    # Create PDB visualization with substrate atom contributions in B-factor
                    try:
                        total_atoms = len(df_geom)
                        b_factors = np.zeros(total_atoms)

                        # Fill in substrate atom contributions
                        for _, row in df_atomwise_temp.iterrows():
                            atom_idx = int(row['Atom_Index'])
                            if 0 <= atom_idx < total_atoms:
                                b_factors[atom_idx] = row['Energy_Contribution_kcal_mol']

                        structure_name = os.path.basename(f.rstrip("/"))

                        # Create PDB file
                        pdb_name = f'estab_{charge_type}_{structure_name}_tensor_sub{substrate_multipole_order}_env{env_multipole_order}.pdb'
                        if use_pdb_charges:
                            print(f"    Visualization not supported in pdb_only mode; skipping PDB creation.")
                        elif multipole_available and (substrate_multipole_order >= 2 or env_multipole_order >= 2):
                            Visualize(file_path_xyz).makePDB(multipole_name, b_factors, pdb_name)
                        else:
                            Visualize(file_path_xyz).makePDB(monopole_name, b_factors, pdb_name)
                        print(f"    Created visualization PDB: {pdb_name}")
                    except Exception as e:
                        print(f"    Warning: Could not create visualization PDB: {e}")

            print(f"File {f}: Total energy (substrate order {substrate_multipole_order}, env order {env_multipole_order}): {total_energy:.4f} kcal/mol")

            os.chdir(owd)
            counter += 1

        # Create output DataFrames
        df = pd.DataFrame(allspeciesdict)

        # Generate automatic CSV filename
        # Format: estab_{charge_type}.csv or estab_multipole{N}_{charge_type}.csv
        if multipole_order > 1:
            prefix = f"multipole{multipole_order}_"
        else:
            prefix = ""

        auto_filename = f"estab_{prefix}{charge_type}"

        # Use automatic filename if user provided default/generic name
        if name_dataStorage in ['output', 'estatic', 'estab', 'estab_data']:
            base_filename = auto_filename
        else:
            base_filename = name_dataStorage

        main_csv = f"{base_filename}.csv"
        df.to_csv(main_csv, index=False)
        print(f"Saved electrostatic stabilization results to: {main_csv}")

        if save_atomwise_decomposition:
            df_atomwise = pd.DataFrame(all_atomwise_data)
            atomwise_csv = f"{base_filename}_atomwise.csv"
            df_atomwise.to_csv(atomwise_csv, index=False)
            print(f"Saved atomwise decomposition to: {atomwise_csv}")
            return df, df_atomwise

        return df
    


def visualize_charges_pdb(xyz_file, charges, pdb_name):
    """Create a PDB file with charges visualized in the b-factor column.

    Args:
        xyz_file (str): Path to the XYZ structure file
        charges (array-like): Array of atomic charges to visualize
        pdb_name (str): Output PDB filename

    Returns:
        None: Writes PDB file to disk
    """
    import openbabel
    from biopandas.pdb import PandasPdb
    import warnings
    import sys

    # Suppress Open Babel warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    stderr_backup = sys.stderr
    sys.stderr = open(os.devnull, 'w')

    try:
        # Convert XYZ to PDB using OpenBabel
        obConversion = openbabel.OBConversion()
        obConversion.SetInAndOutFormats("xyz", "pdb")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, xyz_file)
        mol.ConnectTheDots()
        mol.PerceiveBondOrders()
        mol.FindRingAtomsAndBonds()
        obConversion.WriteFile(mol, pdb_name)

        # Load PDB and assign charges to b-factor column
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_name)

        # Verify lengths match
        if len(charges) != len(ppdb.df['HETATM']):
            print(f"WARNING: Length mismatch in visualize_charges_pdb!")
            print(f"  charges length: {len(charges)}")
            print(f"  PDB DataFrame length: {len(ppdb.df['HETATM'])}")
            # Resize charges if needed
            if len(charges) < len(ppdb.df['HETATM']):
                charges_padded = np.zeros(len(ppdb.df['HETATM']))
                charges_padded[:len(charges)] = charges
                charges = charges_padded
            else:
                charges = charges[:len(ppdb.df['HETATM'])]

        # Sort and assign charges to b-factor
        ppdb.df['HETATM'] = ppdb.df['HETATM'].sort_values('atom_number').reset_index(drop=True)
        ppdb.df['HETATM']['b_factor'] = charges
        ppdb.to_pdb(path=pdb_name, records=['HETATM'], gz=False, append_newline=True)

    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = stderr_backup
