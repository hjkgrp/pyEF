"""Utility module for pyEF package.

This module contains all utility functions, classes, and constants used throughout
the pyEF package for processing quantum chemistry calculations and electric field analysis.

Physical Constants
------------------
All constants are in SI units unless otherwise specified.

Atomic Data
-----------
- PERIODIC_TABLE: Element symbols mapped to atomic numbers
- ATOMIC_MASS_DICT: Element properties (mass, atomic number, covalent radius, valence)
- VDW_RADII_DICT: Van der Waals radii for elements

Sources:
    - Physical constants: NIST, CODATA
    - Atomic masses: molsimplify, http://www.webelements.com/
    - VDW radii: https://periodictable.com/Properties/A/VanDerWaalsRadius.v.html
"""

import numpy as np
import re
import os
import mmap
import glob
import shutil
import logging
import traceback
import subprocess
import warnings
import numpy as np
import pandas as pd
import scipy.linalg as la
from importlib import resources
from distutils.dir_util import copy_tree
#import openbabel
#from biopandas.pdb import PandasPdb
import math
import time
import basis_set_exchange as bse
import json
import shlex

# ============================================================================
# Physical Constants
# ============================================================================

# Coulomb's constant (N*m^2/C^2) aka (kg*m^3/(s^4*A^2))
COULOMB_CONSTANT = 8.987551e9

# Unit conversions
ANGSTROM_TO_M = 1e-10           # Angstrom to meters
BOHR_TO_M = 5.291772109e-11     # Bohr to meters
BOHR_TO_ANGSTROM = 0.529177210903  # Bohr to Angstrom

# Elementary charge (Coulombs)
ELEMENTARY_CHARGE = 1.6023e-19

# Electric field unit conversion
VM_TO_VA = 1e-10  # Volts/meter to Volts/Angstrom

# ============================================================================
# Periodic Table
# ============================================================================

PERIODIC_TABLE = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44,
    "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62,
    "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
    "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86,
    "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90, "Pa": 91, "U": 92, "Np": 93, "Pu": 94,
    "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102,
    "Lr": 103,
    "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117,
    "Og": 118
}

# ============================================================================
# Atomic Properties Dictionary
# ============================================================================

# Format: 'Symbol': (atomic_mass_amu, atomic_number, covalent_radius_Å, valence_electrons)
# Source: molsimplify, http://www.webelements.com/
# Updated values: Pd covalent radius (1.39 -> 1.38), 2008 values (1/18/24)

ATOMIC_MASS_DICT = {
    'X': (1.0, 0, 0.77, 0),
    'H': (1.0079, 1, 0.37, 1),
    'D': (2.0141, 1, 0.37, 1),
    'He': (4.002602, 2, 0.46, 2),
    'Li': (6.94, 3, 1.33, 1),
    'Be': (9.0121831, 4, 1.02, 2),
    'B': (10.83, 5, 0.85, 3),
    'C': (12.0107, 6, 0.77, 4),
    'N': (14.0067, 7, 0.75, 5),
    'O': (15.9994, 8, 0.73, 6),
    'F': (18.9984, 9, 0.71, 7),
    'Ne': (20.1797, 10, 0.67, 8),
    'Na': (22.99, 11, 1.55, 1),
    'Mg': (24.30, 12, 1.39, 2),
    'Al': (26.98, 13, 1.26, 3),
    'Si': (28.08, 14, 1.16, 4),
    'P': (30.9738, 15, 1.06, 5),
    'S': (32.065, 16, 1.02, 6),
    'Cl': (35.453, 17, 0.99, 7),
    'Ar': (39.948, 18, 0.96, 8),
    'K': (39.10, 19, 1.96, 1),
    'Ca': (40.08, 20, 1.71, 2),
    'Sc': (44.96, 21, 1.7, 3),
    'Ti': (47.867, 22, 1.36, 4),
    'V': (50.94, 23, 1.34, 5),
    'Cr': (51.9961, 24, 1.27, 6),
    'Mn': (54.938, 25, 1.39, 7),
    'Fe': (55.84526, 26, 1.32, 8),
    'Co': (58.9332, 27, 1.26, 9),
    'Ni': (58.4934, 28, 1.24, 10),
    'Cu': (63.546, 29, 1.38, 11),
    'Zn': (65.39, 30, 1.31, 12),
    'Ga': (69.72, 31, 1.26, 3),
    'Ge': (72.63, 32, 1.22, 4),
    'As': (74.92, 33, 1.21, 5),
    'Se': (78.96, 34, 1.16, 6),
    'Br': (79.904, 35, 1.14, 7),
    'Kr': (83.798, 36, 1.17, 8),
    'Rb': (85.47, 37, 2.10, 1),
    'Sr': (87.62, 38, 1.85, 2),
    'Y': (88.91, 39, 1.63, 3),
    'Zr': (91.22, 40, 1.48, 4),
    'Nb': (92.91, 41, 1.47, 5),
    'Mo': (95.96, 42, 1.45, 6),
    'Tc': (98.9, 43, 1.56, 7),
    'Ru': (101.1, 44, 1.25, 8),
    'Rh': (102.9, 45, 1.35, 9),
    'Pd': (106.4, 46, 1.38, 10),
    'Ag': (107.9, 47, 1.45, 11),
    'Cd': (112.4, 48, 1.48, 12),
    'In': (111.818, 49, 1.42, 3),
    'Sn': (118.710, 50, 1.41, 4),
    'Sb': (121.760, 51, 1.40, 5),
    'Te': (127.60, 52, 1.99, 6),
    'I': (126.90447, 53, 1.40, 7),
    'Xe': (131.293, 54, 1.31, 8),
    'Cs': (132.9055, 55, 2.44, 1),
    'Ba': (137.327, 56, 1.96, 2),
    'La': (138.9, 57, 1.69, 3),
    'Ce': (140.116, 58, 1.63, 4),
    'Pr': (140.90766, 59, 1.76, 5),
    'Nd': (144.242, 60, 1.74, 6),
    'Pm': (145, 61, 1.73, 7),
    'Sm': (150.36, 62, 1.72, 8),
    'Eu': (151.964, 63, 1.68, 9),
    'Gd': (157.25, 64, 1.69, 10),
    'Tb': (158.92535, 65, 1.68, 11),
    'Dy': (162.500, 66, 1.67, 12),
    'Ho': (164.93033, 67, 1.66, 13),
    'Er': (167.259, 68, 1.65, 14),
    'Tm': (168.93422, 69, 1.64, 15),
    'Yb': (173.045, 70, 1.70, 16),
    'Lu': (174.9668, 71, 1.62, 3),
    'Hf': (178.5, 72, 1.50, 8),
    'Ta': (180.9, 73, 1.38, 5),
    'W': (183.8, 74, 1.46, 6),
    'Re': (186.2, 75, 1.59, 7),
    'Os': (190.2, 76, 1.28, 8),
    'Ir': (192.2, 77, 1.37, 9),
    'Pt': (195.1, 78, 1.36, 10),
    'Au': (197.0, 79, 1.44, 11),
    'Hg': (200.6, 80, 1.49, 2),
    'Tl': (204.38, 81, 1.44, 3),
    'Pb': (207.2, 82, 1.47, 4),
    'Bi': (208.9804, 83, 1.51, 5),
    'Po': (208.98, 84, 1.90, 6),
    'At': (209.99, 85, 2.00, 7),
    'Rn': (222.6, 86, 142, 4),
    'Fr': (223.02, 87, 3.48, 8),
    'Ra': (226.03, 88, 2.01, 2),
    'Ac': (277, 89, 1.86, 3),
    'Th': (232.0377, 90, 1.75, 4),
    'Pa': (231.04, 91, 2.00, 5),
    'U': (238.02891, 92, 1.70, 6),
    'Np': (237.05, 93, 1.90, 7),
    'Pu': (244.06, 94, 1.75, 8),
    'Am': (243.06, 95, 1.80, 9),
    'Cm': (247.07, 96, 1.69, 10),
    'Bk': (247.07, 97, 1.68, 11),
    'Cf': (251.08, 98, 1.68, 12)
}

# ============================================================================
# Van der Waals Radii Dictionary
# ============================================================================

# Van der Waals radii in Angstroms
# Source: https://periodictable.com/Properties/A/VanDerWaalsRadius.v.html
# Note: Some entries have tuple values (legacy compatibility) while others are floats

VDW_RADII_DICT = {
    'X': (1.0, 0, 0.77, 0),
    'H': 1.2,
    'D': (2.0141, 1, 0.37, 1),
    'He': 1.4,
    'Li': (6.94, 3, 1.33, 1),
    'Be': (9.0121831, 4, 1.02, 2),
    'B': (10.83, 5, 0.85, 3),
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'Ne': (20.1797, 10, 0.67, 8),
    'Na': (22.99, 11, 1.55, 1),
    'Mg': (24.30, 12, 1.39, 2),
    'Al': (26.98, 13, 1.26, 3),
    'Si': (28.08, 14, 1.16, 4),
    'P': 1.75,
    'S': 1.8,
    'Cl': 1.75,
    'Ar': (39.948, 18, 0.96, 8),
    'K': (39.10, 19, 1.96, 1),
    'Ca': (40.08, 20, 1.71, 2),
    'Sc': (44.96, 21, 1.7, 3),
    'Ti': (47.867, 22, 1.36, 4),
    'V': (50.94, 23, 1.34, 5),
    'Cr': (51.9961, 24, 1.27, 6),
    'Mn': (54.938, 25, 1.39, 7),
    'Fe': (55.84526, 26, 1.32, 8),
    'Co': (58.9332, 27, 1.26, 9),
    'Ni': (58.4934, 28, 1.24, 10),
    'Cu': (63.546, 29, 1.38, 11),
    'Zn': (65.39, 30, 1.31, 12),
    'Ga': (69.72, 31, 1.26, 3),
    'Ge': (72.63, 32, 1.22, 4),
    'As': (74.92, 33, 1.21, 5),
    'Se': (78.96, 34, 1.16, 6),
    'Br': (79.904, 35, 1.14, 7),
    'Kr': (83.798, 36, 1.17, 8),
    'Rb': (85.47, 37, 2.10, 1),
    'Sr': (87.62, 38, 1.85, 2),
    'Y': (88.91, 39, 1.63, 3),
    'Zr': (91.22, 40, 1.48, 4),
    'Nb': (92.91, 41, 1.47, 5),
    'Mo': (95.96, 42, 1.45, 6),
    'Tc': (98.9, 43, 1.56, 7),
    'Ru': (101.1, 44, 1.25, 8),
    'Rh': (102.9, 45, 1.35, 9),
    'Pd': (106.4, 46, 1.38, 10),
    'Ag': (107.9, 47, 1.45, 11),
    'Cd': (112.4, 48, 1.48, 12),
    'In': (111.818, 49, 1.42, 3),
    'Sn': (118.710, 50, 1.41, 4),
    'Sb': (121.760, 51, 1.40, 5),
    'Te': (127.60, 52, 1.99, 6),
    'I': (126.90447, 53, 1.40, 7),
    'Xe': (131.293, 54, 1.31, 8),
    'Cs': (132.9055, 55, 2.44, 1),
    'Ba': (137.327, 56, 1.96, 2),
    'La': (138.9, 57, 1.69, 3),
    'Ce': (140.116, 58, 1.63, 4),
    'Pr': (140.90766, 59, 1.76, 5),
    'Nd': (144.242, 60, 1.74, 6),
    'Pm': (145, 61, 1.73, 7),
    'Sm': (150.36, 62, 1.72, 8),
    'Eu': (151.964, 63, 1.68, 9),
    'Gd': (157.25, 64, 1.69, 10),
    'Tb': (158.92535, 65, 1.68, 11),
    'Dy': (162.500, 66, 1.67, 12),
    'Ho': (164.93033, 67, 1.66, 13),
    'Er': (167.259, 68, 1.65, 14),
    'Tm': (168.93422, 69, 1.64, 15),
    'Yb': (173.045, 70, 1.70, 16),
    'Lu': (174.9668, 71, 1.62, 3),
    'Hf': (178.5, 72, 1.50, 8),
    'Ta': (180.9, 73, 1.38, 5),
    'W': (183.8, 74, 1.46, 6),
    'Re': (186.2, 75, 1.59, 7),
    'Os': (190.2, 76, 1.28, 8),
    'Ir': (192.2, 77, 1.37, 9),
    'Pt': (195.1, 78, 1.36, 10),
    'Au': (197.0, 79, 1.44, 11),
    'Hg': (200.6, 80, 1.49, 2),
    'Tl': (204.38, 81, 1.44, 3),
    'Pb': (207.2, 82, 1.47, 4),
    'Bi': (208.9804, 83, 1.51, 5),
    'Po': (208.98, 84, 1.90, 6),
    'At': (209.99, 85, 2.00, 7),
    'Rn': (222.6, 86, 142, 4),
    'Fr': (223.02, 87, 3.48, 8),
    'Ra': (226.03, 88, 2.01, 2),
    'Ac': (277, 89, 1.86, 3),
    'Th': (232.0377, 90, 1.75, 4),
    'Pa': (231.04, 91, 2.00, 5),
    'U': (238.02891, 92, 1.70, 6),
    'Np': (237.05, 93, 1.90, 7),
    'Pu': (244.06, 94, 1.75, 8),
    'Am': (243.06, 95, 1.80, 9),
    'Cm': (247.07, 96, 1.69, 10),
    'Bk': (247.07, 97, 1.68, 11),
    'Cf': (251.08, 98, 1.68, 12)
}

# ============================================================================
# Utility Classes
# ============================================================================

class MoldenObject:
    def __init__(self, xyzfile, moldenfile):
        self.xyzfile = xyzfile
        self.moldenFile = moldenfile
        # Use constants from this module
        self.periodic_table = PERIODIC_TABLE
        self.amassdict = ATOMIC_MASS_DICT



    

    def countBasis(self, spherical=True):
        degeneracy = {
        's': 1,
        'p': 3,
        'd': 5 if spherical else 6,
        'f': 7 if spherical else 10,
        'g': 9 if spherical else 15}
        nbf = 0
        in_gto = False
        with open(self.moldenFile) as f:
            for line in f:
                if line.strip().startswith('[GTO]'):
                    in_gto = True
                    continue
                if in_gto:
                    if line.strip().startswith('['):  # end of GTO block
                        break
                    parts = line.split()
                    if parts and parts[0].lower() in degeneracy:
                        nbf += degeneracy[parts[0].lower()]
        return nbf


    def build_core_map(self, family):
        """
        Build a {symbol: core_electrons} dictionary for a given ECP family
        """
        mapping = {}
        for element, Z in self.periodic_table.items():
            try:
                data_str = bse.get_basis(family, elements=[Z], fmt="json")
                # Parse JSON string into dict
                data = json.loads(data_str)
                if str(Z) in data["elements"]:
                    print(f'This is Z: {Z}')
                    mapping[element] = data["elements"][str(Z)]['ecp_electrons']
                else:
                    mapping[element] = 0  # all-electron if no ECP
            except Exception as e:
                print("An error occurred:", e)
                traceback.print_exc()
                mapping[element] = 0  # element not defined
        return mapping

    #utility functions to build properly fix molden format in the presence of ECPs
    def build_hybrid_core_map(self, name, cutoff_Z, heavy_family, base_maps):
        """
        Build core-electron map for hybrid family
        Elements with Z <= cutoff_Z ?~F~R core=0
        Elements with Z > cutoff_Z ?~F~R use heavy_family core mapping
        """
        hybrid_map = {}
        heavy_map = base_maps[heavy_family]
        for element, Z in self.periodic_table.items():
            if Z <= cutoff_Z:
                hybrid_map[element] = 0
            else:
                hybrid_map[element] = heavy_map.get(element, 0)
        return hybrid_map


    def fix_ECPmolden(self, owd, ECP):
        """Fix ECP artifacts in TeraChem Molden files for Multiwfn compatibility.

        Performs three corrections:
        1. Replaces full atomic numbers with valence electron counts for ECP atoms
        2. Adds [Pseudo] section listing ECP atoms (required by Multiwfn)
        3. Removes any incorrectly added [5d]/[7f]/[9g] tags (TeraChem writes
           Cartesian basis functions which is the Molden default)
        """
        from .geometry import Geometry  # Import here to avoid circular dependency
        FAMILIES = ["lanl2dz", "stuttgart_rsc", "def2", "crenbl"]
        # Hybrid families with cutoff rules
        HYBRID_FAMILIES = {
            "lacvps": {
                "cutoff_Z": 18,          # Z <= 18 → all-electron
                "heavy_family": "lanl2dz"},
            "lacvp": {
                "cutoff_Z": 10,          # Z <= 10 → all-electron
                "heavy_family": "lanl2dz"
                }
        }

        #check if the ECP json already exists... if not you will need to generate it using the funcitonality above
        json_path = owd + '/' + ECP + "ecp_core_maps.json"

        # Check if the file exists
        if os.path.exists(json_path) and os.path.isfile(json_path):
               pass
        #if cannot find the json, will need to look for it:
        else:
            #check if the ECP is part of a hybrid family (in which there is a cutoff in atom numbers below which we will NOT apply ECP)
            if ECP in HYBRID_FAMILIES.keys():
                hname = ECP
                rules = HYBRID_FAMILIES[hname]

                base_maps = {}
                for fam in FAMILIES:
                    print(f"Building core map for {fam}...")
                    base_maps[fam] = self.build_core_map(fam)
                hybrid_maps = {}
                hybrid_maps[hname] = self.build_hybrid_core_map(
                    hname,
                    cutoff_Z=rules["cutoff_Z"],
                    heavy_family=rules["heavy_family"],
                    base_maps=base_maps
                    )
                with open(json_path, "w") as f:
                    json.dump(hybrid_maps, f, indent=2)
            else:
                base_maps = {}
                print(f"Building core map for {ECP}...")
                base_maps[ECP] = self.build_core_map(ECP)
                with open(json_path, "w") as f:
                    json.dump(base_maps, f, indent=2)

        #Now the json should be establish

        with open(json_path, "r") as f:
            print(f'        >loading {ECP} parameters')
            ecp_dict = json.load(f)[ECP]
            change_dict = {}
            #get set of atoms in molecule

            df = Geometry(self.xyzfile).getGeomInfo()
            elems_molden = set(list(df['Atom']))
            for elem in elems_molden:
                if ecp_dict[elem] > 0:
                    #this means we need to change it!
                    Z = self.periodic_table[elem]
                    new_Z = Z - ecp_dict[elem]
                    change_dict[elem] = new_Z

        with open(self.moldenFile, 'r') as file:
            lines = file.readlines()

        # Check if fixes have already been applied (idempotency)
        full_content = ''.join(lines)
        has_pseudo_already = '[pseudo]' in full_content.lower()

        # --- Step 1: Fix atomic numbers in [Atoms] section ---
        # Also collect ECP atom entries for the [Pseudo] section
        pseudo_entries = []  # list of (symbol, atom_index, new_Z)
        in_atoms = False
        gto_line_idx = None
        mo_line_idx = None

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.lower().startswith('[atoms]'):
                in_atoms = True
                continue
            if stripped.lower().startswith('[pseudo]'):
                in_atoms = False
                continue
            if stripped.lower().startswith('[gto]'):
                in_atoms = False
                gto_line_idx = i
                continue
            if stripped.lower().startswith('[mo]'):
                mo_line_idx = i
                continue

            if in_atoms and stripped:
                parts = line.split()
                if len(parts) >= 6:
                    sym = parts[0]
                    if sym in change_dict:
                        new_Z = change_dict[sym]
                        atom_idx = parts[1]
                        current_Z = int(parts[2])
                        # Only replace if not already corrected
                        if current_Z != new_Z:
                            pattern = re.compile(rf'({re.escape(sym)}\s+{re.escape(atom_idx)}\s+)(\d+)')
                            lines[i] = pattern.sub(rf'\g<1>{new_Z}', line)
                        pseudo_entries.append((sym, atom_idx, new_Z))

        for change_elem, new_Z in change_dict.items():
            print(f'            Molden file at {self.moldenFile} changed: {change_elem} non-core electrons set to {new_Z}')

        # --- Step 2: Add [Pseudo] section before [GTO] ---
        if pseudo_entries and gto_line_idx is not None and not has_pseudo_already:
            pseudo_lines = ['[Pseudo]\n']
            for sym, atom_idx, new_Z in pseudo_entries:
                pseudo_lines.append(f'{sym}   {atom_idx}  {new_Z}\n')
            lines[gto_line_idx:gto_line_idx] = pseudo_lines
            # Adjust mo_line_idx since we inserted lines before it
            if mo_line_idx is not None:
                mo_line_idx += len(pseudo_lines)

        # --- Step 3: Remove any incorrectly added [5d]/[7f]/[9g] tags ---
        # TeraChem writes Cartesian basis functions in Molden format (6d/10f/15g),
        # which is the Molden default. These tags should NOT be present.
        lines = [l for l in lines if l.strip().lower() not in ('[5d]', '[7f]', '[9g]',
                                                                '[5d7f]', '[5d10f]')]

        with open(self.moldenFile, 'w') as file:
            file.writelines(lines)


# ============================================================================
# Multiwfn Interface Classes
# ============================================================================

class MultiwfnInterface:
    """
    Interface class for Multiwfn charge partitioning and multipole calculations.

    This class encapsulates all interactions with the Multiwfn external program,
    providing methods for charge partitioning, multipole moment calculations,
    and parsing of Multiwfn output files.

    Attributes
    ----------
    dict_of_calcs : dict
        Mapping of charge scheme names to Multiwfn command codes
    dict_of_multipole : dict
        Mapping of multipole schemes to Multiwfn command sequences
    config : dict
        Configuration dictionary with Multiwfn settings
    """

    # Multiwfn command codes for charge calculation methods
    CHARGE_SCHEMES = {
        'Hirshfeld': '1', 'Voronoi': '2', 'Mulliken': '5',
        'Lowdin': '6', 'SCPA': '7', 'Becke': '10', 'ADCH': '11',
        'CHELPG': '12', 'MK': '13', 'AIM': '14', 'Hirshfeld_I': '15',
        'CM5': '16', 'EEM': '17', 'RESP': '18', 'PEOE': '19'
    }

    # Multiwfn command codes for multipole moment calculations
    MULTIPOLE_SCHEMES = {
        'Hirshfeld': ['3', '2'],
        'Hirshfeld_I': ['4', '1', '2'],
        'Becke': ['1', '2']
    }

    def __init__(self, config=None):
        """
        Initialize the Multiwfn interface.

        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with keys:
            - molden_filename: Name of molden file
            - xyzfilename: Name of XYZ geometry file
            - rerun: Whether to rerun existing calculations
            - hasECP: Whether calculation used ECP
            - maxIHirshFuzzyBasis: Max basis functions for Hirshfeld-I fuzzy
            - maxIHirshBasis: Max basis functions for Hirshfeld-I
        """
        self.config = config if config is not None else {
            'molden_filename': 'final_optim.molden',
            'xyzfilename': 'final_optim.xyz',
            'rerun': False,
            'hasECP': False,
            'maxIHirshFuzzyBasis': 1500,
            'maxIHirshBasis': 2000
        }

        self.dict_of_calcs = self.CHARGE_SCHEMES
        self.dict_of_multipole = self.MULTIPOLE_SCHEMES

    def run_multiwfn(self, command, input_commands, output_file=None,
                     description="Multiwfn calculation", capture_output=True):
        """
        Centralized Multiwfn runner with proper error handling and output display.

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
        # Prepare the full command with output redirection if specified
        # Prepend ulimit and KMP_STACKSIZE to prevent segfaults with large systems
        command = f"ulimit -s unlimited; export KMP_STACKSIZE=200000000; {command}"
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
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=True,
                    text=True
                )

                # Send input commands
                input_str = '\n'.join(input_commands) + '\n'
                stdout, stderr = proc.communicate(input=input_str)

                # Save to output file if specified
                if output_file:
                    with open(output_file, 'w') as f:
                        f.write(stdout)

                # Display output
                if stdout:
                    print("STDOUT:")
                    print(stdout[:500])  # Print first 500 chars
                    if len(stdout) > 500:
                        print(f"... (truncated {len(stdout) - 500} characters)")

                if stderr:
                    print("\nSTDERR:")
                    print(stderr)

                returncode = proc.returncode
            else:
                # Let output stream to terminal
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    shell=True,
                    text=True
                )
                input_str = '\n'.join(input_commands) + '\n'
                proc.communicate(input=input_str)
                stdout, stderr, returncode = "", "", proc.returncode

            if returncode != 0:
                raise RuntimeError(
                    f"{description} failed with exit code {returncode}\n"
                    f"Command: {full_command}"
                )

            print(f"\n{'='*60}")
            print(f"Completed: {description}")
            print(f"{'='*60}\n")

            return stdout, stderr, returncode

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"ERROR running Multiwfn")
            print(f"Description: {description}")
            print(f"Command: {full_command}")
            print(f"{'='*60}")
            print(f"Error: {str(e)}")
            print(f"{'='*60}\n")
            raise

    def partitionCharge(self, multipole_bool, f,
                       multiwfn_path, charge_type, owd,
                       molden_filename=None, xyz_filename=None):
        """
        Partition electron density using Multiwfn to generate partial charges or multipole moments.

        This is the centralized interface for all charge partitioning schemes (Hirshfeld, CHELPG, etc.).
        Checks if calculation was previously completed and reuses results unless rerun=True.

        Parameters
        ----------
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
        molden_filename : str, optional
            Name of the molden file (not full path, just filename).
            If None, uses self.config['molden_filename']
        xyz_filename : str, optional
            Name of the xyz file (not full path, just filename).
            If None, uses self.config['xyzfilename']

        Returns
        -------
        float
            Computation time in seconds, 0 if calculation was previously completed (skipped),
            or -1 if calculation failed

        Notes
        -----
        For Hirshfeld-I calculations, atmrad files are automatically loaded from
        pyef.resources.atmrad (bundled with the package).
        """
        # Handle f being a list (take first element)
        if isinstance(f, list):
            f = f[0]

        # Use provided filenames or fall back to config defaults
        if molden_filename is None:
            molden_filename = self.config['molden_filename']
        if xyz_filename is None:
            final_structure_file = self.config['xyzfilename']
            # Handle xyzfilename being a list (take first element)
            if isinstance(final_structure_file, list):
                final_structure_file = final_structure_file[0]
        else:
            final_structure_file = xyz_filename
        comp_cost = 0  # Default to 0 for "already exists" case
        num_atoms = 0
        need_to_run_calculation = True

        os.chdir(owd)
        os.chdir(f)

        file_path_multipole = f"{os.getcwd()}/Multipole{charge_type}.txt"
        file_path_monopole = f"{os.getcwd()}/Charges{charge_type}.txt"
        file_path_xyz = f"{os.getcwd()}/{final_structure_file}"

        # Check if previous calculations fully converged for desired multipole/charge scheme
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

        # If you need to run the calculations, run either the multipole or the monopole calculation!
        if need_to_run_calculation or self.config['rerun']:
            print(f'Running {charge_type} {calc_type} calculation for {os.path.basename(f)}')
            try:
                start = time.time()
                if multipole_bool:
                    chg_prefix, _ = os.path.splitext(molden_filename)
                    # Dynamically get path to package settings.ini file
                    with resources.path('pyef.resources', 'settings.ini') as ini_path:
                        path_to_init_file = str(ini_path)
                        command = f"{multiwfn_path} {shlex.quote(molden_filename)} -set {path_to_init_file}"

                        multiwfn_commands = ['15', '-1'] + self.dict_of_multipole[charge_type] + ['0', 'q']
                        num_atoms = self.count_atoms(final_structure_file)

                        if charge_type == 'Hirshfeld_I':
                            # Get the number of basis functions
                            molden_full_path = f"{os.getcwd()}/{molden_filename}"
                            num_basis = MoldenObject(file_path_xyz, molden_full_path).countBasis()

                            # Use bundled atmrad from package resources
                            with resources.path('pyef.resources', 'atmrad') as atmrad_resource:
                                atmrad_src = str(atmrad_resource)
                                copy_tree(atmrad_src, os.getcwd() + '/atmrad/')

                            print(f'Current num of basis is: {num_basis}')
                            print(f'The current max num is: {self.config["maxIHirshFuzzyBasis"]}')
                            if num_basis > self.config['maxIHirshFuzzyBasis']:
                                print(f'Number of basis functions: {num_basis}')
                                multiwfn_commands = ['15', '-1'] + ['4', '-2', '1', '2'] + ['0', 'q']
                                print(f'I-Hirshfeld command should be low memory and slow to accommodate large system')

                        # Use centralized Multiwfn runner
                        self.run_multiwfn(
                            command=command,
                            input_commands=multiwfn_commands,
                            output_file=file_path_multipole,
                            description=f"Multipole {charge_type} calculation for {f}"
                        )

                else:
                    # Dynamically get path to package settings.ini file
                    with resources.path('pyef.resources', 'settings.ini') as ini_path:
                        path_to_init_file = str(ini_path)
                        command = f"{multiwfn_path} {shlex.quote(molden_filename)} -set {path_to_init_file}"
                        chg_prefix, _ = os.path.splitext(molden_filename)
                        calc_command = self.dict_of_calcs[charge_type]
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

                        # Use centralized Multiwfn runner
                        self.run_multiwfn(
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
                return -1  # Return -1 to indicate failure
        else:
            result_file = os.path.basename(file_path_multipole if multipole_bool else file_path_monopole)
            print(f'Found existing {charge_type} {calc_type} result: {result_file} for {os.path.basename(f)}')

        os.chdir(owd)
        return comp_cost

    @staticmethod
    def count_atoms(xyz_filename):
        """Count number of atoms in XYZ file (mapcount replacement)."""
        with open(xyz_filename, 'r') as f:
            for i, line in enumerate(f):
                pass
        return i + 1  # Total lines including first two header lines

    @staticmethod
    def getmultipoles(multipole_name):
        """
        Parse multipole moments from Multiwfn output file.

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

                atomDict = {
                    "Index": index,
                    "Element": element,
                    "Atom_Charge": atomic_charge,
                    'Dipole_Moment': dipole_moment,
                    'Quadrupole_Moment': quadrupole_moment
                }
                atomicDicts.append(atomDict)

        return atomicDicts


# ============================================================================
# OpenBabel Charge Models
# ============================================================================

def get_openbabel_charges(xyz_path, charge_model='gasteiger', output_path=None):
    """Calculate partial atomic charges using OpenBabel charge models.

    This provides an alternative to Multiwfn for simpler charge models when
    quantum chemistry-derived charges are not needed.

    Parameters
    ----------
    xyz_path : str
        Path to XYZ coordinate file (or other format OpenBabel can read).
    charge_model : str, optional
        Charge model to use. Options:
        - 'gasteiger': Gasteiger-Marsili electronegativity equalization (default)
        - 'mmff94': MMFF94 partial charges
        - 'eem': Electronegativity Equalization Method
        - 'qeq': Charge Equilibration method
        - 'qtpie': QTPIE charges
        - 'none': No charges (all zeros)
    output_path : str or None, optional
        If provided, write charges to file in Multiwfn-compatible format.
        If None, only return the DataFrame (default: None).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['Atom', 'x', 'y', 'z', 'charge']
        Compatible with pyEF charge file format.

    Raises
    ------
    ImportError
        If OpenBabel (openbabel) is not installed.
    ValueError
        If an unsupported charge model is specified.
    FileNotFoundError
        If the input XYZ file does not exist.

    Examples
    --------
    >>> # Get Gasteiger charges
    >>> charges_df = get_openbabel_charges('molecule.xyz')

    >>> # Get MMFF94 charges and save to file
    >>> charges_df = get_openbabel_charges('molecule.xyz', 'mmff94', 'charges.txt')

    >>> # Use with Electrostatics for ESP calculation
    >>> charges_df = get_openbabel_charges('molecule.xyz', 'gasteiger', 'ChargesGasteiger.txt')

    Notes
    -----
    Supported charge models and their typical use cases:
    - gasteiger: Fast, reasonable for organic molecules. Based on electronegativity.
    - mmff94: Force field charges, good for drug-like molecules.
    - eem: Similar to gasteiger but with different parameterization.
    - qeq: Good for extended systems, metals.

    The output format matches Multiwfn's charge file format:
    Atom  x  y  z  charge
    """
    try:
        import openbabel
    except ImportError:
        raise ImportError(
            "OpenBabel is required for get_openbabel_charges(). "
            "Install with: conda install -c conda-forge openbabel"
        )

    # Validate charge model
    valid_models = ['gasteiger', 'mmff94', 'eem', 'qeq', 'qtpie', 'none']
    charge_model_lower = charge_model.lower()
    if charge_model_lower not in valid_models:
        raise ValueError(
            f"Unsupported charge model: '{charge_model}'. "
            f"Valid options: {', '.join(valid_models)}"
        )

    # Check input file exists
    if not os.path.exists(xyz_path):
        raise FileNotFoundError(f"Input file not found: {xyz_path}")

    # Determine input format from extension
    ext = os.path.splitext(xyz_path)[1].lower()
    format_map = {
        '.xyz': 'xyz',
        '.pdb': 'pdb',
        '.mol': 'mol',
        '.mol2': 'mol2',
        '.sdf': 'sdf',
    }
    input_format = format_map.get(ext, 'xyz')

    # Read molecule with OpenBabel
    obConversion = openbabel.OBConversion()
    obConversion.SetInFormat(input_format)
    mol = openbabel.OBMol()

    if not obConversion.ReadFile(mol, xyz_path):
        raise ValueError(f"Failed to read file: {xyz_path}")

    # Assign charges using the specified model
    if charge_model_lower != 'none':
        charge_model_obj = openbabel.OBChargeModel.FindType(charge_model_lower)
        if charge_model_obj is None:
            raise ValueError(
                f"OpenBabel could not find charge model: '{charge_model}'. "
                f"This model may not be available in your OpenBabel installation."
            )
        success = charge_model_obj.ComputeCharges(mol)
        if not success:
            warnings.warn(
                f"Charge calculation with '{charge_model}' may have failed. "
                "Charges may be zero or inaccurate.",
                UserWarning
            )

    # Extract atom data
    atoms = []
    coords_x = []
    coords_y = []
    coords_z = []
    charges = []

    for atom in openbabel.OBMolAtomIter(mol):
        # Get element symbol
        atomic_num = atom.GetAtomicNum()
        element = openbabel.GetSymbol(atomic_num)

        atoms.append(element)
        coords_x.append(atom.GetX())
        coords_y.append(atom.GetY())
        coords_z.append(atom.GetZ())
        charges.append(atom.GetPartialCharge())

    # Create DataFrame
    df = pd.DataFrame({
        'Atom': atoms,
        'x': coords_x,
        'y': coords_y,
        'z': coords_z,
        'charge': charges
    })

    # Write to file if output_path is specified
    if output_path is not None:
        df.to_csv(output_path, sep=' ', index=False, header=False)

    return df


def list_openbabel_charge_models():
    """List available OpenBabel charge models.

    Returns
    -------
    list of str
        List of available charge model names.

    Notes
    -----
    Common models include:
    - gasteiger: Gasteiger-Marsili charges (fast, electronegativity-based)
    - mmff94: MMFF94 force field charges
    - eem: Electronegativity Equalization Method
    - qeq: Charge Equilibration
    - qtpie: QTPIE method
    """
    return ['gasteiger', 'mmff94', 'eem', 'qeq', 'qtpie', 'none']
