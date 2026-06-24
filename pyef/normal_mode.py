"""
Normal mode electric field projection utilities.

Provides Kabsch alignment and projection of per-atom electric field vectors
onto a normal mode eigenvector.

The projected field is:

    F_mode = Σ_i  F_i · (R @ L_i)

where:
    F_i  : electric field vector at solute atom i  (V/Å, shape (3,))
    L_i  : QM normal mode displacement for atom i  (shape (3,))
    R    : 3×3 Kabsch rotation mapping the QM reference frame to the
           current molecular frame

This is the standard Stark-tuning projection used in vibrational Stark
effect spectroscopy.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Heavy-atom element sets (for auto-selecting alignment atoms)
# ---------------------------------------------------------------------------
_HEAVY_ATOM_NUMBERS = set(range(3, 119))   # everything except H (1) and He (2)


def get_heavy_atom_indices(atomic_numbers):
    """Return 0-based indices of non-hydrogen atoms."""
    return np.array([i for i, z in enumerate(atomic_numbers) if z > 1], dtype=int)


# ---------------------------------------------------------------------------
# Kabsch rotation
# ---------------------------------------------------------------------------

def kabsch_rotation(P, Q):
    """
    Find the rotation matrix R that best superposes P onto Q (both centred).

    Parameters
    ----------
    P, Q : np.ndarray  shape (N, 3)  — must already be mean-centred

    Returns
    -------
    R : np.ndarray  shape (3, 3)
        Rotation such that  Q_approx = P @ R.T  (i.e. R @ p_i ≈ q_i).
    """
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    # Correct for improper rotation (reflection)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return R


def align_to_reference(current_coords, ref_coords, align_indices=None):
    """
    Compute the Kabsch rotation R that maps the QM reference geometry
    onto the current (MD/QM cluster) geometry.

    Parameters
    ----------
    current_coords : np.ndarray  shape (n_atoms, 3)   current frame coords
    ref_coords     : np.ndarray  shape (n_atoms, 3)   QM reference coords (Å)
    align_indices  : array-like or None
        Atom indices used for the fit (default: all atoms).

    Returns
    -------
    R : np.ndarray  shape (3, 3)
        Rotation mapping QM frame → current frame.
    """
    if align_indices is None:
        align_indices = np.arange(len(ref_coords))
    align_indices = np.asarray(align_indices)

    P = ref_coords[align_indices]
    Q = current_coords[align_indices]

    P_c = P - P.mean(axis=0)
    Q_c = Q - Q.mean(axis=0)

    return kabsch_rotation(P_c, Q_c)


# ---------------------------------------------------------------------------
# Normal-mode projection
# ---------------------------------------------------------------------------

def project_efield_onto_mode(per_atom_efields, mode_vector, rotation):
    """
    Project per-atom electric field vectors onto a (rotated) normal mode.

    Parameters
    ----------
    per_atom_efields : np.ndarray  shape (n_solute, 3)
        Electric field at each solute atom from the environment (V/Å).
    mode_vector      : np.ndarray  shape (n_solute, 3)
        Normal mode displacement eigenvector in the QM reference frame.
    rotation         : np.ndarray  shape (3, 3)
        Kabsch rotation R mapping QM reference frame → current frame.

    Returns
    -------
    F_mode : float   projected field in V/Å
    """
    # Rotate each mode displacement vector into the current molecular frame
    L_rotated = mode_vector @ rotation.T   # (n_solute, 3)
    # Dot product summed over all solute atoms
    return float(np.sum(per_atom_efields * L_rotated))


# ---------------------------------------------------------------------------
# Per-atom electric field from point charges (monopole Coulomb)
# ---------------------------------------------------------------------------

_K_E = 8.987551e9    # N·m²/C²
_C_E = 1.6023e-19    # C
_A2M = 1e-10         # Å → m


def _minimum_image(r, box):
    """Minimum image convention; r (...,3), box (3,)."""
    return r - box * np.round(r / box)


def per_atom_efield_from_charges(solute_coords, env_coords, env_charges,
                                  box=None):
    """
    Compute the Coulomb electric field at each solute atom from environment.

    Parameters
    ----------
    solute_coords : np.ndarray  shape (n_solute, 3)   Å
    env_coords    : np.ndarray  shape (n_env, 3)       Å
    env_charges   : np.ndarray  shape (n_env,)         elementary charge units
    box           : np.ndarray  shape (3,) or None
        Periodic box lengths in Å.  None → no PBC.

    Returns
    -------
    efields : np.ndarray  shape (n_solute, 3)   V/Å
    """
    efields = np.zeros_like(solute_coords)
    for i, pos in enumerate(solute_coords):
        r = pos[np.newaxis, :] - env_coords           # (n_env, 3) in Å
        if box is not None:
            r = _minimum_image(r, box)
        r_m  = r * _A2M                               # Å → m
        d_m  = np.linalg.norm(r_m, axis=1, keepdims=True)
        mask = d_m[:, 0] > 0
        E = np.sum(
            _K_E * _C_E * env_charges[mask, np.newaxis]
            * r_m[mask] / d_m[mask] ** 3,
            axis=0
        ) * _A2M                                      # V/m → V/Å
        efields[i] = E
    return efields
