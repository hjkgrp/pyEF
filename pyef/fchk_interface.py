"""
Parser for formatted checkpoint (.fchk) files.

Supports:
  - Gaussian fchk: geometry + normal modes + frequencies
  - Q-Chem fchk:   geometry only (no Vib-Modes section)

Auto-detection: if 'Vib-Modes' section is present → Gaussian format.

All coordinates are returned in Angstroms (fchk stores Bohr).
Normal mode eigenvectors are returned as (n_modes, n_atoms, 3) arrays,
already mass-weighted and normalized as printed by the QM code.
"""

import numpy as np

BOHR_TO_ANG = 0.529177210903


def _read_section(lines, key):
    """
    Return (header_line, flat_list_of_values) for the first section whose
    header starts with `key`.  Returns (None, None) if not found.
    """
    for i, line in enumerate(lines):
        if line.startswith(key):
            header = line.rstrip()
            # scalar value on the same line?
            parts = header.split()
            if parts[-2] in ('I', 'R', 'C', 'L') and parts[-1] not in ('N=',):
                # scalar: value is last token (no continuation block)
                try:
                    return header, [parts[-1]]
                except Exception:
                    return header, []
            # array: read until next header-like line
            values = []
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                # header lines start with a letter and contain type codes
                if (nxt and nxt[0].isalpha() and
                        any(f' {t} ' in nxt for t in (' I ', ' R ', ' C ', ' L '))):
                    break
                values.extend(nxt.split())
                j += 1
            return header, values
    return None, None


def _parse_int_array(values):
    return [int(v) for v in values]


def _parse_float_array(values):
    return [float(v.replace('D', 'E').replace('d', 'e')) for v in values]


def detect_format(fchk_path):
    """Return 'gaussian' if Vib-Modes section present, else 'qchem'."""
    with open(fchk_path) as f:
        for line in f:
            if line.startswith('Vib-Modes'):
                return 'gaussian'
    return 'qchem'


def parse_geometry(fchk_path):
    """
    Parse reference geometry from fchk (Gaussian or Q-Chem).

    Returns
    -------
    atomic_numbers : np.ndarray  shape (n_atoms,)  int
    coords         : np.ndarray  shape (n_atoms, 3) float, Angstroms
    """
    with open(fchk_path) as f:
        lines = f.readlines()

    _, anum_vals = _read_section(lines, 'Atomic numbers')
    _, coord_vals = _read_section(lines, 'Current cartesian coordinates')

    if anum_vals is None or coord_vals is None:
        raise ValueError(f"Could not find geometry sections in {fchk_path}")

    atomic_numbers = np.array(_parse_int_array(anum_vals), dtype=int)
    coords_bohr    = np.array(_parse_float_array(coord_vals), dtype=float)
    n_atoms        = len(atomic_numbers)

    if len(coords_bohr) != 3 * n_atoms:
        raise ValueError(
            f"Coordinate count mismatch: expected {3*n_atoms}, got {len(coords_bohr)}"
        )

    coords = coords_bohr.reshape(n_atoms, 3) * BOHR_TO_ANG
    return atomic_numbers, coords


def parse_normal_modes(fchk_path):
    """
    Parse normal mode frequencies and eigenvectors from a **Gaussian** fchk.

    Returns
    -------
    frequencies : np.ndarray  shape (n_modes,)        cm⁻¹
    modes       : np.ndarray  shape (n_modes, n_atoms, 3)
                  Each row is one normalised eigenvector in Cartesian space.

    Raises
    ------
    ValueError  if the file is Q-Chem format (no Vib-Modes section).
    """
    if detect_format(fchk_path) != 'gaussian':
        raise ValueError(
            f"{fchk_path} appears to be Q-Chem format — no Vib-Modes section.\n"
            "Pass mode vectors directly via the `mode_vectors` argument."
        )

    with open(fchk_path) as f:
        lines = f.readlines()

    _, natom_vals = _read_section(lines, 'Number of atoms')
    n_atoms = int(natom_vals[0])

    _, freq_vals = _read_section(lines, 'Vib-Freq')
    if freq_vals is None:
        raise ValueError("No 'Vib-Freq' section found in fchk.")
    frequencies = np.array(_parse_float_array(freq_vals), dtype=float)
    n_modes = len(frequencies)

    _, mode_vals = _read_section(lines, 'Vib-Modes')
    if mode_vals is None:
        raise ValueError("No 'Vib-Modes' section found in fchk.")
    mode_flat = np.array(_parse_float_array(mode_vals), dtype=float)

    # Gaussian stores modes column-major: all atoms for mode 0, then mode 1, …
    # Total elements = n_modes * 3 * n_atoms
    if len(mode_flat) != n_modes * 3 * n_atoms:
        raise ValueError(
            f"Vib-Modes size mismatch: expected {n_modes * 3 * n_atoms}, "
            f"got {len(mode_flat)}"
        )
    modes = mode_flat.reshape(n_modes, n_atoms, 3)
    return frequencies, modes


def get_mode(fchk_path, mode_index, mode_vectors=None):
    """
    Retrieve a single normal mode vector.

    Parameters
    ----------
    fchk_path    : str   path to .fchk file
    mode_index   : int   1-based mode number (as printed by Q-Chem / Gaussian)
    mode_vectors : np.ndarray or None
        If provided, shape (n_modes, n_atoms, 3) or (n_atoms, 3).
        Required for Q-Chem fchk; ignored for Gaussian fchk.

    Returns
    -------
    freq   : float or None   frequency in cm⁻¹ (None for Q-Chem)
    vector : np.ndarray      shape (n_atoms, 3)  normalised displacement
    """
    fmt = detect_format(fchk_path)

    if fmt == 'gaussian':
        frequencies, modes = parse_normal_modes(fchk_path)
        idx = mode_index - 1   # convert to 0-based
        if idx < 0 or idx >= len(frequencies):
            raise IndexError(
                f"mode_index {mode_index} out of range (1–{len(frequencies)})"
            )
        return float(frequencies[idx]), modes[idx]

    else:  # Q-Chem: no Vib-Modes in fchk
        if mode_vectors is None:
            raise ValueError(
                "Q-Chem fchk has no normal mode data. "
                "Pass mode_vectors as a (n_modes, n_atoms, 3) array."
            )
        mv = np.asarray(mode_vectors, dtype=float)
        if mv.ndim == 2:
            # Single mode passed directly
            return None, mv
        idx = mode_index - 1
        if idx < 0 or idx >= len(mv):
            raise IndexError(
                f"mode_index {mode_index} out of range (1–{len(mv)})"
            )
        return None, mv[idx]
