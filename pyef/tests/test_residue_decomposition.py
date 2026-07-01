"""Unit tests for residue-level decomposition of E-field and ESP (pdb_only mode)"""

from pathlib import Path

import numpy as np
import pytest

from pyef.analysis import Electrostatics


PDB_TEXT = """\
ATOM      1  PD  PD1 A   1       0.000   0.000   0.000  1.00  2.000           P
ATOM      2  O   LIG A   2       1.500   0.000   0.000  1.00 -0.500           O
ATOM      3  C   LIG A   2       2.700   0.000   0.000  1.00  0.300           C
ATOM      4  O   MOL A   3      -2.000   1.000   0.000  1.00 -0.800           O
ATOM      5  H   MOL A   3      -2.500   1.800   0.000  1.00  0.400           H
ATOM      6  X   GST A   4       0.000   2.000   0.000  1.00  5.000           X
END
"""


@pytest.fixture
def pdb_path(tmp_path):
    path = tmp_path / "frame.pdb"
    path.write_text(PDB_TEXT)
    return str(path)


@pytest.fixture
def out_dir(pdb_path):
    return Path(pdb_path).parent


class TestGetPdbChargesResName:
    """getPdbCharges() must expose residue names so atoms can be grouped"""

    def test_resname_column_present_and_ordered(self, pdb_path):
        es = Electrostatics(pdb_charge_paths=[pdb_path])
        df = es.getPdbCharges(pdb_path)

        assert "ResName" in df.columns
        assert list(df["ResName"]) == ["PD1", "LIG", "LIG", "MOL", "MOL", "GST"]
        # Row position must equal the 0-indexed atom index used elsewhere in pyEF
        assert df["charge"].iloc[0] == pytest.approx(2.0)
        assert df["charge"].iloc[5] == pytest.approx(5.0)


class TestGetEfieldByResidue:
    """getEfield_byResidue() should decompose the same field getEfield() reports"""

    def test_groups_sum_to_full_field(self, pdb_path, out_dir):
        es = Electrostatics(pdb_charge_paths=[pdb_path])

        df_byres = es.getEfield_byResidue(
            residue_groups={"Metal": ["PD1"], "Solvent": ["MOL"], "Guest": ["GST"]},
            input_bond_indices=[[(1, 2)]],
            output_filename=str(out_dir / "ef_byres"),
        )
        df_full = es.getEfield(
            pdb_charge_paths=[pdb_path],
            input_bond_indices=[[(1, 2)]],
            Efielddata_filename=str(out_dir / "ef_full"),
        )

        full_value = df_full["Projected_Efields V/Angstrom"].iloc[0][0]
        assert df_byres["Efield_V_per_A"].sum() == pytest.approx(full_value)
        # MV/cm column must be the V/Angstrom column scaled by 100
        assert np.allclose(df_byres["Efield_MV_per_cm"], df_byres["Efield_V_per_A"] * 100.0)

    def test_unmapped_residue_goes_to_other(self, pdb_path, out_dir):
        es = Electrostatics(pdb_charge_paths=[pdb_path])
        df = es.getEfield_byResidue(
            residue_groups={"Metal": ["PD1"], "Solvent": ["MOL"]},
            input_bond_indices=[[(1, 2)]],
            output_filename=str(out_dir / "ef_other"),
        )
        groups = set(df["Group"])
        assert groups == {"Metal", "Solvent", "Other"}
        # LIG (2 atoms) and GST (1 atom) are unmapped -> 3 atoms total fall under 'Other'
        other_row = df[df["Group"] == "Other"].iloc[0]
        assert other_row["N_atoms"] == 3

    def test_exclude_atoms_removes_contribution(self, pdb_path, out_dir):
        es = Electrostatics(pdb_charge_paths=[pdb_path])
        df = es.getEfield_byResidue(
            residue_groups={"Metal": ["PD1"], "Solvent": ["MOL"], "Guest": ["GST"]},
            input_bond_indices=[[(1, 2)]],
            exclude_atoms=[5],
            output_filename=str(out_dir / "ef_excl"),
        )
        guest_row = df[df["Group"] == "Guest"].iloc[0]
        assert guest_row["N_atoms"] == 0
        assert guest_row["Efield_V_per_A"] == 0.0

    def test_requires_pdb_charge_paths(self, pdb_path):
        # Simulate a non-pdb_only instance (no Multiwfn-derived charges available)
        es = Electrostatics(pdb_charge_paths=[pdb_path])
        es.pdb_only = False
        es.pdb_charge_paths = None
        with pytest.raises(ValueError, match="pdb_charge_paths"):
            es.getEfield_byResidue(
                residue_groups={"Metal": ["PD1"]},
                input_bond_indices=[[(0, 1)]],
            )

    def test_requires_nonempty_residue_groups(self, pdb_path):
        es = Electrostatics(pdb_charge_paths=[pdb_path])
        with pytest.raises(ValueError, match="residue_groups"):
            es.getEfield_byResidue(residue_groups={}, input_bond_indices=[[(1, 2)]])


class TestGetESPByResidue:
    """getESP_byResidue() should decompose the same ESP getESP() reports"""

    def test_groups_sum_to_full_esp(self, pdb_path, out_dir):
        es = Electrostatics(pdb_charge_paths=[pdb_path], esp_atom_idx=[1])

        df_byres = es.getESP_byResidue(
            residue_groups={"Metal": ["PD1"], "Solvent": ["MOL"], "Guest": ["GST"]},
            output_filename=str(out_dir / "esp_byres"),
        )
        df_full = es.getESP(
            pdb_charge_paths=[pdb_path],
            ESPdata_filename=str(out_dir / "esp_full"),
        )

        esp_col = [c for c in df_full.columns if c.startswith("ESP")][0]
        assert df_byres["ESP_V"].sum() == pytest.approx(df_full[esp_col].iloc[0])

    def test_requires_esp_atom_idx(self, pdb_path):
        es = Electrostatics(pdb_charge_paths=[pdb_path])
        with pytest.raises(ValueError, match="esp_atom_idx"):
            es.getESP_byResidue(residue_groups={"Metal": ["PD1"]})
