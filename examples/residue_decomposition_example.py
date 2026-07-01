#!/usr/bin/env python3
"""
Example: residue-level decomposition of E-field and ESP with pyEF.

This is a self-contained, runnable example - it writes a small synthetic PDB
file (so you don't need real QM output to try it), then shows how to call
Electrostatics.getEfield_byResidue() and Electrostatics.getESP_byResidue().

Run with the conda environment that has pyEF installed:

    conda run -n pyef python examples/residue_decomposition_example.py

To adapt this to real data, see the "Using your own data" section at the
bottom of this file.
"""

import os
import tempfile

from pyef.analysis import Electrostatics


# ---------------------------------------------------------------------------
# 1. Build (or point to) a PDB file with partial charges in the B-factor
#    column. Residue names (columns 18-20 of each ATOM/HETATM line) are what
#    getEfield_byResidue()/getESP_byResidue() group atoms by.
#
#    In a real workflow this PDB comes from your own pipeline (e.g. an MD
#    frame with QM-derived or force-field partial charges written into the
#    B-factor column) - see "Using your own data" below.
# ---------------------------------------------------------------------------

PDB_TEXT = """\
ATOM      1  PD  PD1 A   1       0.000   0.000   0.000  1.00  1.200           Pd
ATOM      2  PD  PD2 A   2       8.000   0.000   0.000  1.00  1.200           Pd
ATOM      3  C   LIG B   3       2.000   1.000   0.000  1.00 -0.150           C
ATOM      4  N   LIG B   3       3.500   1.000   0.000  1.00 -0.300           N
ATOM      5  C   LIG B   3       5.500   1.000   0.000  1.00 -0.150           C
ATOM      6  B   BAR C   4      -4.000   3.000   0.000  1.00 -1.000           B
ATOM      7  F   BAR C   4      -5.500   3.000   0.000  1.00 -0.350           F
ATOM      8  O   MOL D   5      -2.000  -2.000   1.000  1.00 -0.800           O
ATOM      9  H   MOL D   5      -2.700  -1.500   1.500  1.00  0.400           H
ATOM     10  H   MOL D   5      -1.500  -2.700   1.500  1.00  0.400           H
ATOM     11  C3  GST E   6       4.000   0.200   0.000  1.00  0.450           C
ATOM     12  O1  GST E   6       4.000  -1.200   0.000  1.00 -0.450           O
END
"""

tmpdir = tempfile.mkdtemp(prefix="pyef_residue_example_")
pdb_path = os.path.join(tmpdir, "frame_001.pdb")
with open(pdb_path, "w") as fh:
    fh.write(PDB_TEXT)

print(f"Wrote example PDB to: {pdb_path}\n")


# ---------------------------------------------------------------------------
# 2. Define residue groups: group label -> list of PDB residue names
#    (the ResName field, matched verbatim/case-sensitively).
# ---------------------------------------------------------------------------

RESIDUE_GROUPS = {
    "Metals":  ["PD1", "PD2"],
    "Linkers": ["LIG"],
    "BARF":    ["BAR"],
    "Solvent": ["MOL"],
    # "GST" (the guest, atoms 10-11, 0-indexed) is deliberately left out of
    # every group and instead fully excluded below via exclude_atoms, so it
    # doesn't contribute to its own field/potential.
}

GUEST_ATOM_IDXS = [10, 11]  # 0-indexed: C3 and O1 of the guest (GST)


# ---------------------------------------------------------------------------
# 3. Build the Electrostatics object in pdb_only mode (charges come straight
#    from the PDB B-factor column - no Multiwfn/molden/xyz needed).
# ---------------------------------------------------------------------------

es = Electrostatics(pdb_charge_paths=[pdb_path])


# ---------------------------------------------------------------------------
# 4. E-field decomposition across the guest's C3=O1 bond (atom indices 10, 11)
# ---------------------------------------------------------------------------

df_ef = es.getEfield_byResidue(
    residue_groups=RESIDUE_GROUPS,
    input_bond_indices=[[(10, 11)]],   # one list of (atomA, atomB) per job/frame
    exclude_atoms=GUEST_ATOM_IDXS,
    output_filename=os.path.join(tmpdir, "ef_byresidue"),
    dielectric=1,
)

print("E-field decomposition across the C3=O1 bond (MV/cm):")
print(df_ef[["Group", "N_atoms", "Efield_MV_per_cm"]].to_string(index=False))
print(f"  Sum across groups: {df_ef['Efield_MV_per_cm'].sum():+.2f} MV/cm\n")


# ---------------------------------------------------------------------------
# 5. ESP decomposition evaluated at the guest's carbonyl carbon (atom index 10)
# ---------------------------------------------------------------------------

df_esp = es.getESP_byResidue(
    residue_groups=RESIDUE_GROUPS,
    esp_atom_idx=[10],
    exclude_atoms=GUEST_ATOM_IDXS,
    output_filename=os.path.join(tmpdir, "esp_byresidue"),
    dielectric=1,
)

print("ESP decomposition at the carbonyl carbon (Volts):")
print(df_esp[["Group", "N_atoms", "ESP_V"]].to_string(index=False))
print(f"  Sum across groups: {df_esp['ESP_V'].sum():+.4f} V\n")

print(f"CSV outputs and the example PDB are in: {tmpdir}")


# ---------------------------------------------------------------------------
# Using your own data
# ---------------------------------------------------------------------------
# 1. Replace pdb_path with your real PDB file(s) (one per frame/structure),
#    each with partial charges written into the B-factor column.
# 2. Set RESIDUE_GROUPS to the actual residue names (PDB ResName field) in
#    your system, e.g. {'Pd1': ['PD1'], 'Solvent': ['MOL']}.
# 3. For multiple frames, pass a list of paths to pdb_charge_paths=[...] and
#    a matching per-frame list to input_bond_indices / esp_atom_idx /
#    exclude_atoms (or a single flat exclude_atoms list if it's the same
#    atom indices on every frame).
# 4. Any residue name not listed in RESIDUE_GROUPS falls into an 'Other'
#    bucket (with a printed warning) instead of being silently dropped -
#    a good way to catch a forgotten residue the first time you run this
#    on a new system.
