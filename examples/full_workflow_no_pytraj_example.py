#!/usr/bin/env python3
"""
Full residue-decomposition workflow on real data, with zero pytraj dependency.

Uses two files that already exist in the Pd2L4 project (no synthetic/fabricated
molecules, no OpenBabel charge guessing):

    THF/pTolinCageA_BARF_THFsolv.pdb     - real cage+BARF+PZQ+THF structure
    THF/pTolinCageA_BARF_THFsolv.prmtop  - the matching AMBER topology, which
                                            also stores the force-field partial
                                            charges pyEF needs

step1_prep_bound_frames.py normally gets these same charges via pytraj
(`np.array(top.charge)`) because it also needs pytraj to walk the MD
trajectory and pick out frames. This script only needs a *static* structure,
so it skips pytraj entirely and reads the charges directly out of the
prmtop's plain-text "%FLAG CHARGE" section instead - AMBER topology files are
plain ASCII, so this needs nothing beyond Python's standard library.

Workflow:
  1. Parse partial charges straight out of the .prmtop file (no pytraj).
  2. Pair them positionally with the atoms in the real .pdb structure file
     (same atom order in both, since they're the topology/coordinate pair
     for the same system) to build a pyEF pdb_only charge-PDB.
  3. Run getEfield_byResidue() across both PZQ C=O bonds and
     getESP_byResidue() at the C3 carbonyl carbon, using the exact
     RESIDUE_GROUPS from step2_efield_analysis.py.
  4. Verify the per-group sums exactly reproduce the non-decomposed
     getEfield()/getESP() result.

Run with:  conda run -n pyef python examples/full_workflow_no_pytraj_example.py

(This script hardcodes paths into the Pd2L4 project since it's meant to
demonstrate the feature on real project data. For a fully portable example
with no external files, see residue_decomposition_example.py in this folder.)
"""

import os
import time

import numpy as np

from pyef.analysis import Electrostatics

PROJECT_DIR = '/home/gridsan/mmanetsch/Pd2L4/pToluquinone_SolventDependence'
PDB_PATH = os.path.join(PROJECT_DIR, 'THF', 'pTolinCageA_BARF_THFsolv.pdb')
PRMTOP_PATH = os.path.join(PROJECT_DIR, 'THF', 'pTolinCageA_BARF_THFsolv.prmtop')

OUT_DIR = os.path.join(PROJECT_DIR, 'efield_analysis', '_full_workflow_no_pytraj_example')

# AMBER stores charges scaled by this factor; dividing converts to elementary
# charge units (e). See e.g. AmberTools' prmtop format documentation.
AMBER_CHARGE_SCALE = 18.2223

RESIDUE_GROUPS = {
    'Pd1':     ['PD1'],
    'Pd2':     ['PD2'],
    'Linkers': ['LA1', 'LB1', 'LC1', 'LD1'],
    'BARF':    ['BFV', 'BFW', 'BFX', 'BFY'],
    'Solvent': ['MOL'],
}


def parse_prmtop_charges(prmtop_path):
    """Read the %FLAG CHARGE section of an AMBER prmtop (pure text, no pytraj).

    Returns one float per atom, in elementary charge units, in atom order.
    """
    with open(prmtop_path) as fh:
        lines = fh.read().splitlines()

    start = None
    for i, line in enumerate(lines):
        if line.startswith('%FLAG CHARGE'):
            start = i + 2   # skip the %FLAG and %FORMAT lines
            break
    if start is None:
        raise ValueError(f"No %FLAG CHARGE section found in {prmtop_path}")

    raw_values = []
    for line in lines[start:]:
        if line.startswith('%FLAG'):
            break
        # %FORMAT(5E16.8): five fixed-width 16-character fields per line
        for i in range(0, len(line), 16):
            chunk = line[i:i + 16].strip()
            if chunk:
                raw_values.append(float(chunk))

    return [v / AMBER_CHARGE_SCALE for v in raw_values]


def read_pdb_atoms(pdb_path):
    """Read (element, atom_name, resname, (x, y, z)) for every ATOM/HETATM line."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            aname = line[12:16].strip()
            resname = line[17:20].strip()
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            elem = ''.join(c for c in aname if c.isalpha())[:2] or 'X'
            atoms.append((elem, aname, resname, (x, y, z)))
    return atoms


def write_charge_pdb(atoms, charges, path):
    with open(path, 'w') as fh:
        for serial, ((elem, aname, resname, (x, y, z)), chg) in enumerate(zip(atoms, charges), start=1):
            fh.write(
                f"ATOM  {serial:5d} {aname:<4s} {resname:3s}  {1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{chg:6.3f}          {elem:>2s}\n"
            )
        fh.write("END\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    charges = parse_prmtop_charges(PRMTOP_PATH)
    atoms = read_pdb_atoms(PDB_PATH)
    assert len(charges) == len(atoms), (
        f"Charge count ({len(charges)}) doesn't match atom count ({len(atoms)}) - "
        "the .pdb and .prmtop must be the matching topology/coordinate pair."
    )
    print(f"Loaded {len(atoms)} atoms with real AMBER charges (parsed from prmtop, no pytraj)")

    charge_pdb_path = os.path.join(OUT_DIR, 'THF_real_charges.pdb')
    write_charge_pdb(atoms, charges, charge_pdb_path)
    print(f"Wrote charge-PDB: {charge_pdb_path}\n")

    # Locate PZQ (the guest) and its two C=O bonds by atom name, rather than
    # hardcoding indices, so this still works if the topology ever changes.
    pzq_idxs = [i for i, (_, _, resname, _) in enumerate(atoms) if resname == 'PZQ']
    pzq_names = {atoms[i][1]: i for i in pzq_idxs}
    c3, o1 = pzq_names['C3'], pzq_names['O1']
    c6, o2 = pzq_names['C6'], pzq_names['O2']
    print(f"PZQ guest: {len(pzq_idxs)} atoms, bonds C3=O1 ({c3},{o1}) and C6=O2 ({c6},{o2})\n")

    # -----------------------------------------------------------------
    # Residue-decomposed E-field across both PZQ C=O bonds
    # -----------------------------------------------------------------
    es = Electrostatics(pdb_charge_paths=[charge_pdb_path])
    t0 = time.time()
    df_ef = es.getEfield_byResidue(
        residue_groups=RESIDUE_GROUPS,
        input_bond_indices=[[(c3, o1), (c6, o2)]],
        exclude_atoms=pzq_idxs,
        output_filename=os.path.join(OUT_DIR, 'ef_byresidue'),
    )
    print(f"getEfield_byResidue on {len(atoms)} atoms took {time.time() - t0:.1f}s")
    print(df_ef[['Bond', 'Group', 'N_atoms', 'Efield_MV_per_cm']].to_string(index=False))

    es_full = Electrostatics(pdb_charge_paths=[charge_pdb_path])
    es_full.config['excludeAtomfromEcalc'] = pzq_idxs
    df_ef_full = es_full.getEfield(
        pdb_charge_paths=[charge_pdb_path], input_bond_indices=[[(c3, o1), (c6, o2)]],
        Efielddata_filename=os.path.join(OUT_DIR, 'ef_full'),
    )
    full_efields = df_ef_full['Projected_Efields V/Angstrom'].iloc[0]
    for bond_pos, (a, b) in enumerate([(c3, o1), (c6, o2)]):
        bond_str = f"{a}-{b}"
        decomposed_sum = df_ef.loc[df_ef['Bond'] == bond_str, 'Efield_V_per_A'].sum()
        assert np.isclose(decomposed_sum, full_efields[bond_pos]), (
            f"Mismatch for {bond_str}: decomposed={decomposed_sum}, full={full_efields[bond_pos]}"
        )
    print("[OK] residue-group sums match getEfield() exactly\n")

    # -----------------------------------------------------------------
    # Residue-decomposed ESP at the C3 carbonyl carbon
    # -----------------------------------------------------------------
    es_esp = Electrostatics(pdb_charge_paths=[charge_pdb_path], esp_atom_idx=[c3])
    df_esp = es_esp.getESP_byResidue(
        residue_groups=RESIDUE_GROUPS,
        exclude_atoms=pzq_idxs,
        output_filename=os.path.join(OUT_DIR, 'esp_byresidue'),
    )
    print(df_esp[['Group', 'N_atoms', 'ESP_V']].to_string(index=False))

    es_esp_full = Electrostatics(pdb_charge_paths=[charge_pdb_path], esp_atom_idx=[c3])
    es_esp_full.config['excludeAtomfromEcalc'] = pzq_idxs
    df_esp_full = es_esp_full.getESP(
        pdb_charge_paths=[charge_pdb_path], ESPdata_filename=os.path.join(OUT_DIR, 'esp_full'),
    )
    esp_col = [c for c in df_esp_full.columns if c.startswith('ESP')][0]
    full_esp = df_esp_full[esp_col].iloc[0]
    decomposed_esp_sum = df_esp['ESP_V'].sum()
    assert np.isclose(decomposed_esp_sum, full_esp), (
        f"Mismatch: decomposed={decomposed_esp_sum}, full={full_esp}"
    )
    print("[OK] residue-group sums match getESP() exactly\n")

    print(f"All outputs written to: {OUT_DIR}")


if __name__ == '__main__':
    main()
