"""
ESP profiles on a plane defined by three atoms (or atom-group centroids).

Self-contained: turns pyEF's per-job charge files into frame-averaged ESP
maps on the plane through three points P1, P2, P3, and plots them.  Needs
only NumPy and SciPy (convex-hull mask); matplotlib is imported only when
plotting.

Frame, recomputed for every job (frame) before averaging:

* vertical (local z)   = (P2 - P1) / |P2 - P1|
* normal (local y)     = (P2 - P1) x (P3 - P1), normalized
* horizontal (local x) = normal x vertical -- the part of P3 - P1 perpendicular
  to the vertical axis, so it points toward P3

Each P is one atom index or a group of indices (its centroid is used).  The
origin is ``'centroid'`` (of P1, P2, P3; default), ``'midpoint'`` (of P1 and
P2), an atom index, or explicit lab xyz; it is projected into the plane.

    ESP(r) [V] = 14.399645 / dielectric * sum_j q_j / |r - r_j|

Source-probe pairs closer than ``r_min`` (0.5 A) are dropped.

Inputs are pyEF's existing files, one job per frame:

* ``Charges<type>.txt`` -- ``sym x y z q`` per QM atom (Multiwfn output)
* ``pointcharges.txt``  -- 2 header lines, then ``q x y z`` per MM charge

Conditions are named source sets written as tokens: ``qm``, ``mm``, ``all``,
``qm:0-133``, ``mm:0-275``, ``mm:276-`` (ranges inclusive, 0-indexed; an open
end runs to the last charge).  The published counterion/solvent ladder is::

    {'CageOnly': 'qm:0-133',
     'Cage_Anion': 'qm:0-133,mm:0-275',
     'Cage_Anion_Solv': 'qm:0-133,mm'}

Masks are stored next to the maps, never applied to them: a nucleus-radius
mask around chosen QM atoms plus an optional convex hull over chosen atoms.

Output npz keys are ``<name>_<field>`` (``<name>_mean_<cond>``,
``<name>_std_<cond>``, ``<name>_z_vals``, ``<name>_x_vals``, ``<name>_mask``,
``<name>_mean_dist``, ...), the ESP replot-bundle layout.

Command line (no Multiwfn needed when the charge files exist)::

    python -m pyef.esp_profile --charges f*/ChargesCM5.txt \\
        --ptchg f*/pointcharges.txt --plane 1 0 2-34 --origin midpoint \\
        --condition CageOnly=qm:0-133 --condition Cage_Anion=qm:0-133,mm:0-275
"""

import argparse
import os
import re
import warnings
from dataclasses import dataclass

import numpy as np

# Coulomb constant in V*A/e (CODATA 2018: e / (4 pi eps0) / 1 A).
COULOMB_EV_ANGSTROM = 14.399645

# Probe-chunk size, in (probe x source) elements: big blocks keep the
# probe @ source.T product BLAS-efficient.  1e8 elements = 800 MB per block.
CHUNK_ELEMS = 100_000_000

SOURCES = ('qm', 'mm')


# --- index specs --------------------------------------------------------------

def parse_index_spec(spec):
    """Atom indices from ``5``, ``'5'``, ``'2-34'``, ``'2-10+12'`` or a list.

    Ranges are inclusive.  Returns a list of ints.
    """
    if isinstance(spec, (int, np.integer)):
        return [int(spec)]
    if not isinstance(spec, str):
        out = []
        for s in spec:
            out.extend(parse_index_spec(s))
        return out
    out = []
    for part in spec.replace(' ', '').split('+'):
        m = re.fullmatch(r'(\d+)(?:-(\d+))?', part)
        if not m:
            raise ValueError(f'bad atom index spec {spec!r}; use e.g. 5, 2-34, 2-10+12')
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise ValueError(f'empty range {part!r} in {spec!r}')
        out.extend(range(lo, hi + 1))
    return out


def parse_condition(tokens):
    """Parse a condition into ``[(source, start, stop_or_None), ...]``.

    ``tokens`` is a comma/space separated string (``'qm:0-133,mm:276-'``) or
    a list of token strings.  ``stop`` is inclusive; ``None`` means open-ended.
    ``all`` expands to ``qm`` + ``mm``.  Overlapping tokens within one
    condition would double-count charges and raise ``ValueError``.
    """
    if isinstance(tokens, str):
        tokens = [t for t in re.split(r'[,\s]+', tokens.strip()) if t]
    out = []
    for tok in tokens:
        t = tok.strip().lower()
        if t == 'all':
            out += [('qm', 0, None), ('mm', 0, None)]
            continue
        m = re.fullmatch(r'(qm|mm)(?::(\d+)(?:-(\d*))?)?', t)
        if not m:
            raise ValueError(f'bad condition token {tok!r}; use qm, mm, all, '
                             'qm:0-133, mm:276-')
        src, lo, hi = m.groups()
        if lo is None:
            out.append((src, 0, None))
        elif hi is None:
            out.append((src, int(lo), int(lo)))     # single index
        elif hi == '':
            out.append((src, int(lo), None))
        else:
            if int(hi) < int(lo):
                raise ValueError(f'empty range in condition token {tok!r}')
            out.append((src, int(lo), int(hi)))
    for src in SOURCES:
        rs = sorted((a, np.inf if b is None else b) for s, a, b in out if s == src)
        for (a0, b0), (a1, _) in zip(rs, rs[1:]):
            if a1 <= b0:
                raise ValueError(f'condition {tokens!r} lists {src} charges more '
                                 'than once')
    if not out:
        raise ValueError('empty condition')
    return out


def _segment_conditions(conditions):
    """Split all condition ranges into disjoint segments.

    The ESP is linear in its sources, so each segment is evaluated once per
    frame and every condition is the sum of its segments.

    Returns ``(segments, seg_conditions)``: ``segments`` maps a stable name
    to ``(source, start, stop_exclusive_or_None)`` and ``seg_conditions`` maps
    each condition to a tuple of segment names.
    """
    parsed = {c: parse_condition(t) for c, t in conditions.items()}
    cuts = {s: {0} for s in SOURCES}
    for toks in parsed.values():
        for s, a, b in toks:
            cuts[s].add(a)
            cuts[s].add(np.inf if b is None else b + 1)
    segments = {}
    bounds = {}
    for s in SOURCES:
        cs = sorted(cuts[s]) + ([] if np.inf in cuts[s] else [np.inf])
        bounds[s] = list(zip(cs, cs[1:]))
    seg_conditions = {}
    for c, toks in parsed.items():
        names = []
        for s, a, b in toks:
            hi = np.inf if b is None else b + 1
            for lo_, hi_ in bounds[s]:
                if lo_ >= a and hi_ <= hi:
                    name = f'{s}:{int(lo_)}-' + ('' if hi_ == np.inf else str(int(hi_) - 1))
                    segments[name] = (s, int(lo_), None if hi_ == np.inf else int(hi_))
                    names.append(name)
        seg_conditions[c] = tuple(names)
    return segments, seg_conditions


def _closed_stops(conditions):
    """Largest closed (inclusive) stop per source, to check against each frame."""
    top = {s: -1 for s in SOURCES}
    for t in conditions.values():
        for s, a, b in parse_condition(t):
            top[s] = max(top[s], a if b is None else b)
    return top


def condition_name(tokens):
    """A key-safe default name for an unnamed condition, e.g. 'qm0-133_mm'."""
    if not isinstance(tokens, str):
        tokens = ','.join(tokens)
    return re.sub(r'[^A-Za-z0-9-]+', '_', tokens.replace(':', '')).strip('_')


# --- file readers ---------------------------------------------------------------

def read_charges(path):
    """Read a pyEF ``Charges<type>.txt`` (``sym x y z q``).

    Returns ``(symbols, xyz (N, 3) A, q (N,) e)``.
    """
    syms, rows = [], []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            syms.append(parts[0])
            rows.append([float(v) for v in parts[1:5]])
    if not rows:
        raise ValueError(f'no charges read from {path}')
    a = np.array(rows)
    return syms, a[:, :3], a[:, 3]


def read_ptchg(path):
    """Read a point-charge file (2 header lines, then ``q x y z``).

    Same format as :meth:`pyef.analysis.Electrostatics.getPtChgs`.
    Returns ``(xyz (M, 3) A, q (M,) e)``.
    """
    a = np.loadtxt(path, skiprows=2, usecols=(0, 1, 2, 3), ndmin=2)
    return a[:, 1:4], a[:, 0]


# --- ESP kernel and masks -----------------------------------------------------------

def _chunk_for(n_src):
    return max(32, int(CHUNK_ELEMS // max(1, n_src)))


def esp_at(probes, src_xyz, src_q, r_min=0.5, k=COULOMB_EV_ANGSTROM):
    """ESP (V) at ``probes`` (N, 3) from point charges, chunked over probes.

    Source-probe pairs closer than ``r_min`` (A, > 0) are dropped; ``k`` is
    the Coulomb constant in V*A/e (divide by a dielectric to screen).
    """
    probes = np.asarray(probes, dtype=float).reshape(-1, 3)
    src_xyz = np.asarray(src_xyz, dtype=float).reshape(-1, 3)
    src_q = np.asarray(src_q, dtype=float).ravel()
    if r_min <= 0:
        raise ValueError('r_min must be > 0')
    out = np.zeros(len(probes))
    if len(src_xyz) == 0:
        return out
    s2 = (src_xyz ** 2).sum(axis=1)
    chunk = _chunk_for(len(src_xyz))
    for i in range(0, len(probes), chunk):
        p = probes[i:i + chunk]
        # Build r^2 in place: as one expression this allocates several more
        # blocks the size of r2 and the kernel becomes memory-bound.
        r2 = p @ src_xyz.T
        r2 *= -2.0
        r2 += (p ** 2).sum(axis=1)[:, None]
        r2 += s2[None, :]
        np.maximum(r2, 0.0, out=r2)
        np.sqrt(r2, out=r2)                     # r
        with np.errstate(divide='ignore'):
            np.reciprocal(r2, out=r2)           # 1/r, inf where r == 0
        # A probe exactly on a source gives 1/r = inf, and inf * False = NaN
        # would poison the whole sum.  Clamp to a finite value that still
        # fails the r_min test so the pair is dropped as intended.
        np.minimum(r2, 2.0 / r_min, out=r2)
        np.multiply(r2, r2 <= 1.0 / r_min, out=r2)
        out[i:i + chunk] = k * (r2 @ src_q)
    return out


def min_dist(probes, src_xyz):
    """Distance (A) from each probe to the nearest source point."""
    probes = np.asarray(probes, dtype=float).reshape(-1, 3)
    src_xyz = np.asarray(src_xyz, dtype=float).reshape(-1, 3)
    out = np.full(len(probes), np.inf)
    if len(src_xyz) == 0:
        return out
    s2 = (src_xyz ** 2).sum(axis=1)
    chunk = _chunk_for(len(src_xyz))
    for i in range(0, len(probes), chunk):
        p = probes[i:i + chunk]
        r2 = (p ** 2).sum(axis=1)[:, None] - 2.0 * (p @ src_xyz.T) + s2[None, :]
        out[i:i + chunk] = np.sqrt(np.maximum(r2.min(axis=1), 0.0))
    return out


def hull_mask(probes, hull_xyz):
    """True where a probe lies OUTSIDE the convex hull of ``hull_xyz``."""
    from scipy.spatial import Delaunay
    return Delaunay(np.asarray(hull_xyz, dtype=float)).find_simplex(
        np.asarray(probes, dtype=float).reshape(-1, 3)) < 0


# --- plane ----------------------------------------------------------------------

@dataclass
class PlaneFrame:
    """Orthonormal frame of the profile plane.

    ``rotation`` has the local x (horizontal), y (normal), z (vertical) unit
    vectors as its COLUMNS: ``lab = origin + local @ rotation.T`` and
    ``local = (lab - origin) @ rotation``.  ``axis_length`` is |P2 - P1|.
    """
    origin: np.ndarray
    rotation: np.ndarray
    axis_length: float

    @property
    def x_hat(self):
        return self.rotation[:, 0]

    @property
    def y_hat(self):
        return self.rotation[:, 1]

    @property
    def z_hat(self):
        return self.rotation[:, 2]

    def to_local(self, xyz):
        return (np.asarray(xyz, dtype=float) - self.origin) @ self.rotation

    def to_lab(self, local):
        return self.origin + np.asarray(local, dtype=float) @ self.rotation.T


def plane_basis(p1, p2, p3, origin=None, min_perp=1e-3):
    """Orthonormal frame of the plane through ``p1``, ``p2``, ``p3``.

    local z = vertical (P1 -> P2), local y = plane normal, local x =
    horizontal toward P3.  ``origin`` (lab xyz) defaults to the centroid of
    the three points and is projected into the plane.
    """
    p1, p2, p3 = (np.asarray(p, dtype=float) for p in (p1, p2, p3))
    v = p2 - p1
    length = float(np.linalg.norm(v))
    if length == 0.0:
        raise ValueError('P1 and P2 coincide')
    z_hat = v / length
    n = np.cross(v, p3 - p1)
    nn = np.linalg.norm(n)
    if nn < min_perp * length * max(np.linalg.norm(p3 - p1), 1e-12):
        raise ValueError('P1, P2, P3 are (nearly) collinear; the plane is undefined')
    y_hat = n / nn
    x_hat = np.cross(y_hat, z_hat)
    o = (p1 + p2 + p3) / 3.0 if origin is None else np.asarray(origin, dtype=float)
    o = o - np.dot(o - p1, y_hat) * y_hat
    return PlaneFrame(o, np.column_stack([x_hat, y_hat, z_hat]), length)


def plane_frame(xyz, plane_points, origin='centroid'):
    """:func:`plane_basis` from QM coordinates and index specs.

    ``plane_points`` is three index specs (int, ``'2-34'``, or a list) into
    ``xyz``.  ``origin`` is ``'centroid'``, ``'midpoint'``, an atom index, or a
    length-3 lab xyz.  Returns ``(frame, points (3, 3))``.
    """
    if len(plane_points) != 3:
        raise ValueError(f'need exactly three plane points, got {len(plane_points)}')
    xyz = np.asarray(xyz, dtype=float)
    pts = []
    for spec in plane_points:
        idx = parse_index_spec(spec)
        if max(idx) >= len(xyz):
            raise IndexError(f'plane point index {max(idx)} out of range for '
                             f'{len(xyz)} QM atoms')
        pts.append(xyz[idx].mean(axis=0))
    p1, p2, p3 = pts
    if isinstance(origin, str):
        if origin == 'centroid':
            o = None
        elif origin == 'midpoint':
            o = 0.5 * (p1 + p2)
        else:
            raise ValueError(f"origin must be 'centroid', 'midpoint', an atom "
                             f"index or xyz, got {origin!r}")
    elif np.ndim(origin) == 0:
        o = xyz[int(origin)]
    else:
        o = np.asarray(origin, dtype=float).reshape(3)
    return plane_basis(p1, p2, p3, origin=o), np.array(pts)


def grid_shape(extent, spacing):
    """``(z_half, x_half, nz, nx)`` from a half-extent and spacing.

    ``extent`` is one number or ``(z_half, x_half)``, A; ``spacing`` is one
    number or ``(dz, dx)``, A.
    """
    zh, xh = np.broadcast_to(np.asarray(extent, dtype=float), (2,))
    dz, dx = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
    if min(zh, xh, dz, dx) <= 0:
        raise ValueError('extent and spacing must be > 0')
    return float(zh), float(xh), int(round(2 * zh / dz)) + 1, int(round(2 * xh / dx)) + 1


def plane_grid(z_half, x_half, nz, nx):
    """Probe points of the local y = 0 plane.

    Returns ``(z_vals, x_vals, local)`` where ``local`` is (nz*nx, 3) in
    local coordinates, ordered so ``values.reshape(nz, nx)`` puts z on axis 0.
    """
    z_vals = np.linspace(-z_half, z_half, nz)
    x_vals = np.linspace(-x_half, x_half, nx)
    ZZ, XX = np.meshgrid(z_vals, x_vals, indexing='ij')
    local = np.column_stack([XX.ravel(), np.zeros(ZZ.size), ZZ.ravel()])
    return z_vals, x_vals, local


def esp_on_grid(frame, src_xyz, src_q, extent=(7.5, 6.0), spacing=0.1,
                r_min=0.5, dielectric=1.0):
    """ESP (V) on the frame's plane from one set of point charges.

    Returns ``(z_vals, x_vals, esp (nz, nx))``.
    """
    zh, xh, nz, nx = grid_shape(extent, spacing)
    z_vals, x_vals, local = plane_grid(zh, xh, nz, nx)
    v = esp_at(frame.to_lab(local), src_xyz, src_q, r_min=r_min,
               k=COULOMB_EV_ANGSTROM / dielectric)
    return z_vals, x_vals, v.reshape(nz, nx)


# --- driver ---------------------------------------------------------------------

def _default_conditions(have_mm):
    return {'qm': 'qm', 'all': 'all'} if have_mm else {'qm': 'qm'}


def _normalize_conditions(conditions, have_mm):
    if conditions is None:
        return _default_conditions(have_mm)
    if isinstance(conditions, str):
        conditions = [conditions]
    if isinstance(conditions, dict):
        return {str(k): v for k, v in conditions.items()}
    out = {}
    for c in conditions:
        if isinstance(c, str) and '=' in c:
            k, v = c.split('=', 1)
            out[k.strip()] = v.strip()
        else:
            out[condition_name(c)] = c
    return out


def compute_esp_profile(charge_files, plane_points, ptchg_files=None,
                        conditions=None, origin='centroid', extent=(7.5, 6.0),
                        spacing=0.1, r_min=0.5, dielectric=1.0, mm_scale=1.0,
                        mask_radius=1.6, mask_atoms=None, hull_atoms=None,
                        mask_frac=0.3):
    """Frame-averaged ESP maps on the plane through three atoms/centroids.

    Parameters
    ----------
    charge_files : list of str
        One ``Charges<type>.txt`` per frame.
    plane_points : sequence of 3 index specs
        P1, P2, P3 as QM atom indices or groups (``1``, ``'2-34'``, ``[2, 3]``).
    ptchg_files : list of str or None
        One point-charge file per frame (``None`` entries allowed), aligned
        with ``charge_files``.
    conditions : dict, list or None
        ``{name: tokens}``, or a list of ``'name=tokens'`` / bare token
        strings.  Default ``qm`` (and ``all`` if point charges are given).
    origin : 'centroid', 'midpoint', atom index, or lab xyz.
    extent : float or (z_half, x_half)
        Half-extent of the map in A.
    spacing : float or (dz, dx), A.
    r_min : float
        Pairs closer than this (A) are dropped.
    dielectric : float
        Uniform dielectric; the ESP is divided by it.
    mm_scale : float
        Factor applied to every MM charge (pyEF uses
        ``1/sqrt(dielectric_scale)``).
    mask_radius : float, dict or None
        Cells within this distance (A) of a ``mask_atoms`` atom are flagged.
        A dict ``{atom_spec: radius}`` gives each atom group its own radius,
        e.g. ``{'0-1': 2.0, '2-133': 1.6}`` (``mask_atoms`` is then ignored).
    mask_atoms : index spec or None
        QM atoms for a float ``mask_radius``; default all QM atoms.
    hull_atoms : index spec or None
        Cells outside the convex hull of these QM atoms are flagged.
    mask_frac : float
        A cell is masked if flagged in at least this fraction of frames.

    Returns
    -------
    dict with ``mean_<cond>``, ``std_<cond>`` (nz, nx) in V, unmasked;
    ``mask`` (nz, nx) bool; ``z_vals``, ``x_vals``; ``mean_dist`` /
    ``std_dist`` (|P2 - P1|, A); ``plane_points_local`` (3, 3) and
    ``atoms_local`` (N_qm, 3) frame-averaged local coordinates; ``symbols``,
    ``frames``, ``n_frames``, ``conditions``, ``condition_tokens``,
    ``dielectric``, ``r_min``.
    """
    charge_files = list(charge_files)
    if not charge_files:
        raise ValueError('no charge files given')
    if ptchg_files is None:
        ptchg_files = [None] * len(charge_files)
    ptchg_files = list(ptchg_files)
    if len(ptchg_files) != len(charge_files):
        raise ValueError(f'{len(charge_files)} charge files but '
                         f'{len(ptchg_files)} point-charge files')
    have_mm = any(p is not None for p in ptchg_files)
    conditions = _normalize_conditions(conditions, have_mm)
    segments, seg_conditions = _segment_conditions(conditions)
    top = _closed_stops(conditions)

    zh, xh, nz, nx = grid_shape(extent, spacing)
    z_vals, x_vals, local = plane_grid(zh, xh, nz, nx)
    k = COULOMB_EV_ANGSTROM / dielectric
    if isinstance(mask_radius, dict):
        mask_sets = [(parse_index_spec(a), float(r)) for a, r in mask_radius.items()]
    elif mask_radius:
        mask_sets = [(None if mask_atoms is None else parse_index_spec(mask_atoms),
                      float(mask_radius))]
    else:
        mask_sets = []
    hull_idx = None if hull_atoms is None else parse_index_spec(hull_atoms)

    maps = {c: [] for c in conditions}
    excluded = np.zeros(len(local))
    dists, ref_local = [], []
    symbols, n_qm = None, None
    for cf, pf in zip(charge_files, ptchg_files):
        syms, qm_xyz, qm_q = read_charges(cf)
        if n_qm is None:
            symbols, n_qm = syms, len(syms)
        elif len(syms) != n_qm:
            raise ValueError(f'{cf}: {len(syms)} QM atoms, expected {n_qm} '
                             '(all frames must be the same system)')
        if pf is not None:
            mm_xyz, mm_q = read_ptchg(pf)
            mm_q = mm_q * mm_scale
        else:
            mm_xyz, mm_q = np.zeros((0, 3)), np.zeros(0)
        n = {'qm': n_qm, 'mm': len(mm_q)}
        for s in SOURCES:
            if top[s] >= n[s]:
                raise IndexError(f'{cf}: a condition uses {s} index {top[s]} but '
                                 f'this frame has {n[s]} {s.upper()} charges')
        src = {'qm': (qm_xyz, qm_q), 'mm': (mm_xyz, mm_q)}

        frame, pts = plane_frame(qm_xyz, plane_points, origin)
        probes = frame.to_lab(local)

        excl = np.zeros(len(local), dtype=bool)
        for idx, r in mask_sets:
            excl |= min_dist(probes, qm_xyz if idx is None else qm_xyz[idx]) < r
        if hull_idx is not None:
            excl |= hull_mask(local, frame.to_local(qm_xyz[hull_idx]))
        excluded += excl

        seg_v = {g: esp_at(probes, src[s][0][a:b], src[s][1][a:b], r_min=r_min, k=k)
                 for g, (s, a, b) in segments.items()}
        for c, gs in seg_conditions.items():
            maps[c].append(sum((seg_v[g] for g in gs), np.zeros(len(local))).reshape(nz, nx))

        dists.append(frame.axis_length)
        ref_local.append(frame.to_local(np.vstack([pts, qm_xyz])))

    n_frames = len(charge_files)
    ref = np.mean(ref_local, axis=0)
    res = {'n_frames': n_frames, 'z_vals': z_vals, 'x_vals': x_vals,
           'mask': (excluded / n_frames >= mask_frac).reshape(nz, nx),
           'mean_dist': float(np.mean(dists)), 'std_dist': float(np.std(dists)),
           'plane_points_local': ref[:3], 'atoms_local': ref[3:],
           'symbols': np.array(symbols),
           'frames': np.array([os.path.abspath(f) for f in charge_files]),
           'condition_tokens': np.array(
               [f'{c}={t if isinstance(t, str) else ",".join(t)}'
                for c, t in conditions.items()]),
           'conditions': list(conditions),
           'dielectric': float(dielectric), 'r_min': float(r_min)}
    for c in conditions:
        arr = np.stack(maps[c])
        res[f'mean_{c}'] = arr.mean(axis=0)
        res[f'std_{c}'] = arr.std(axis=0)
    return res


def masked(result, cond):
    """Mean map of ``cond`` with masked cells set to NaN."""
    m = np.asarray(result[f'mean_{cond}'], dtype=float).copy()
    m[np.asarray(result['mask'], dtype=bool)] = np.nan
    return m


def region_mean(result, cond, z_lim=4.0, x_lim=4.0, stat='mean'):
    """Average of a map over unmasked cells with |z| <= z_lim, |x| <= x_lim.

    ``stat='std'`` averages the per-cell frame-to-frame standard deviation.
    """
    ZZ, XX = np.meshgrid(result['z_vals'], result['x_vals'], indexing='ij')
    core = (~np.asarray(result['mask'], dtype=bool)) \
        & (np.abs(XX) <= x_lim) & (np.abs(ZZ) <= z_lim)
    return float(np.asarray(result[f'{stat}_{cond}'])[core].mean())


# --- I/O ------------------------------------------------------------------------

def save_esp_profile(path, result, name='profile', merge=True):
    """Write ``result`` to ``path`` with keys ``<name>_<field>``.

    With ``merge=True`` other systems already in ``path`` are kept, so one
    npz can hold several profiles; a system of the same name is replaced.
    Returns the sorted list of system names in the file.
    """
    data = {f'{name}_{k}': np.asarray(v) for k, v in result.items()
            if k != 'conditions'}           # re-derived from condition_tokens
    names = {name}
    if merge and os.path.exists(path):
        with np.load(path) as old:
            old_names = [str(x) for x in old['__names__']] if '__names__' in old.files else []
            for n in old_names:
                if n == name:
                    continue
                names.add(n)
                for key in old.files:
                    if key.startswith(f'{n}_') and _owner(key, old_names) == n:
                        data[key] = old[key]
    data['__names__'] = np.array(sorted(names))
    np.savez_compressed(path, **data)
    return sorted(names)


def load_esp_profile(path, name=None):
    """Read one profile back.  With ``name=None`` the file must hold exactly
    one system."""
    with np.load(path) as d:
        names = ([str(x) for x in d['__names__']] if '__names__' in d.files
                 else sorted({k.split('_')[0] for k in d.files}))
        if name is None:
            if len(names) != 1:
                raise ValueError(f'{path} holds {names}; pass name=')
            name = names[0]
        r = {}
        for key in d.files:
            if _owner(key, names) == name:
                v = d[key]
                r[key[len(name) + 1:]] = v.item() if v.ndim == 0 else v
    if not r:
        raise KeyError(f'no system {name!r} in {path}')
    if 'condition_tokens' in r:
        r['conditions'] = [str(t).split('=', 1)[0] for t in r['condition_tokens']]
    else:
        r['conditions'] = [k[5:] for k in r if k.startswith('mean_') and k != 'mean_dist']
    return r


def _owner(key, names):
    # longest matching prefix, so 'Cage_A' wins over 'Cage'
    best = None
    for n in names:
        if key.startswith(f'{n}_') and (best is None or len(n) > len(best)):
            best = n
    return best


# --- plotting -------------------------------------------------------------------

def _scale_for(maps, pct=99.5):
    """``(vmin, vmax, cmap)``: percentile-clipped; diverging only if the data
    straddle zero, else sequential so the whole range carries information."""
    vals = np.concatenate([np.asarray(m)[np.isfinite(m)].ravel() for m in maps])
    if vals.size == 0:
        raise ValueError('no finite values to plot (is every cell masked?)')
    lo, hi = np.percentile(vals, 100 - pct), np.percentile(vals, pct)
    if lo < 0.0 < hi:
        v = max(abs(lo), abs(hi))
        return -v, v, 'RdBu_r'
    if hi <= 0.0:
        return lo, 0.0, 'Blues_r'
    return 0.0, hi, 'YlOrRd'


def _contour_levels(vmin, vmax, step):
    """Iso-potential levels on an absolute grid of ``step`` volts, so one line
    means the same potential difference in every panel; 11 proportional
    levels if that would give fewer than three lines."""
    if step and step > 0:
        lev = np.arange(np.ceil(vmin / step - 1e-9) * step, vmax + 1e-9 * step, step)
        if len(lev) >= 3:
            return lev
    return np.linspace(vmin, vmax, 11)


def plot_esp_profile(result, conditions=None, out=None, fixed=None,
                     point_labels=('P1', 'P2', 'P3'), atom_slab=None,
                     ylabel='distance along P1→P2 axis (Å)',
                     xlabel='distance in plane (Å)', level_step=1.0,
                     panel_size=3.5, exts=('svg', 'png'), dpi=300):
    """One row of panels, one per condition, on a shared colour scale.

    Masked cells are blank.  Gray contours every ``level_step`` V, a white
    ESP = 0 line where a panel crosses zero, and each panel's mean over
    unmasked cells in the corner.  P1 and P2 are gold circles, P3 a green
    triangle (if inside the map); ``atom_slab`` (A) also draws the QM atoms
    within that distance of the plane.  ``fixed=(vmin, vmax, cmap)`` pins the
    colour scale.  With ``out`` set, saves ``out.<ext>`` and closes the
    figure.  Returns ``(fig, axes, (vmin, vmax, cmap))``.
    """
    import matplotlib.pyplot as plt

    conds = list(conditions) if conditions else list(result['conditions'])
    maps = [masked(result, c) for c in conds]
    z_vals, x_vals = np.asarray(result['z_vals']), np.asarray(result['x_vals'])
    vmin, vmax, cmap = tuple(fixed) if fixed is not None else _scale_for(maps)
    pts = np.asarray(result['plane_points_local'])
    near = None
    if atom_slab:
        a = np.asarray(result['atoms_local'])
        near = a[np.abs(a[:, 1]) <= atom_slab]

    with plt.rc_context({'font.weight': 'bold', 'axes.labelweight': 'bold',
                         'axes.titleweight': 'bold', 'axes.linewidth': 2,
                         'xtick.direction': 'in', 'ytick.direction': 'in',
                         'svg.fonttype': 'none'}):
        fig, axes = plt.subplots(1, len(conds), squeeze=False, sharex=True, sharey=True,
                                 figsize=(panel_size * len(conds) + 1.5, panel_size * 1.25))
        axes = axes[0]
        for j, (ax, c, m) in enumerate(zip(axes, conds, maps)):
            # rasterize the mesh only: a vector mesh makes a huge SVG
            im = ax.pcolormesh(x_vals, z_vals, m, cmap=cmap, vmin=vmin, vmax=vmax,
                               shading='auto', rasterized=True)
            ax.contour(x_vals, z_vals, np.where(np.isfinite(m), m, 0.0),
                       levels=_contour_levels(vmin, vmax, level_step),
                       colors='k', linewidths=0.3, alpha=0.35)
            if np.isfinite(m).any():
                if np.nanmin(m) < 0.0 < np.nanmax(m):
                    ax.contour(x_vals, z_vals, np.where(np.isfinite(m), m, 1e3),
                               levels=[0.0], colors='white', linewidths=1.6, zorder=4)
                ax.annotate(f'{np.nanmean(m):+.2f} V', (0.97, 0.03),
                            xycoords='axes fraction', ha='right', va='bottom',
                            fontsize=9, fontweight='bold', color='white',
                            bbox=dict(boxstyle='round,pad=0.25', fc='black',
                                      ec='none', alpha=0.55))
            if near is not None:
                ax.plot(near[:, 0], near[:, 2], '.', color='0.2', ms=4, alpha=0.6,
                        zorder=5)
            for i, p in enumerate(pts):
                ax.plot(p[0], p[2], 'o' if i < 2 else '^', ms=9, zorder=6,
                        color='gold' if i < 2 else 'limegreen',
                        markeredgecolor='black', markeredgewidth=1.5,
                        label=point_labels[i])
            ax.axvline(0, color='0.55', lw=0.6, ls='--', alpha=0.6)
            ax.axhline(0, color='0.55', lw=0.6, ls='--', alpha=0.6)
            ax.set_xlim(x_vals[0], x_vals[-1])
            ax.set_ylim(z_vals[0], z_vals[-1])
            ax.set_aspect('equal')
            ax.set_title(c, fontsize=11)
            ax.set_xlabel(xlabel, fontsize=11)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=11)
        cbar = fig.colorbar(im, ax=list(axes), fraction=0.02, pad=0.02,
                            extend='both' if fixed is not None else 'neither')
        cbar.set_label('ESP (V)', fontsize=11, fontweight='bold')
        if out:
            for ext in exts:
                fig.savefig(f'{out}.{ext}', bbox_inches='tight', dpi=dpi)
            plt.close(fig)
    return fig, axes, (vmin, vmax, cmap)


# --- command line ---------------------------------------------------------------

def _pair_ptchg(charges, ptchg):
    if not ptchg:
        return None
    if len(ptchg) != len(charges):
        raise SystemExit(f'error: {len(charges)} --charges but {len(ptchg)} --ptchg files')
    off = [(c, p) for c, p in zip(charges, ptchg)
           if os.path.dirname(os.path.abspath(c)) != os.path.dirname(os.path.abspath(p))]
    if off:
        warnings.warn(f'{len(off)} charge/point-charge pairs are in different '
                      f'directories, e.g. {off[0]}; pairing is by order')
    return ptchg


def _mask_radius_arg(v):
    if ':' not in v:
        return float(v)
    try:
        return {a.strip(): float(r) for a, r in (item.split(':') for item in v.split(','))}
    except ValueError:
        raise argparse.ArgumentTypeError(f'bad --mask-radius {v!r}; use 1.6 or 0-1:2.0,2-133:1.6')


def _origin_arg(vals):
    if len(vals) == 1:
        v = vals[0]
        return int(v) if v.isdigit() else v
    if len(vals) == 3:
        return [float(v) for v in vals]
    raise SystemExit('error: --origin takes centroid, midpoint, an atom index, or x y z')


TEMPLATE = """\
# ESP profile on the plane through three atoms -- pyEF input file.
# Run:  python -m pyef.esp_profile -c this_file.yaml     (or: pyef espprofile -c this_file.yaml)
# Paths are relative to this file.  One frame -> that frame's map;
# several frames -> the average, each frame in its own three-atom plane.

# --- frames: give EITHER charges OR molden -------------------------------------
charges: frames/*/ChargesCM5.txt        # one file, a list, or a glob (sorted f2 < f10)
# molden: frames/*/*.molden             # instead of charges: Multiwfn makes any missing
# xyz: frames/*/*.xyz                   #   Charges<charge_type>.txt (xyz default: molden stem + .xyz)
# charge_type: CM5
# multiwfn_path: /path/to/Multiwfn

pointcharges: pointcharges.txt          # optional MM charges (QM/MM): a bare file name is looked
                                        #   up next to each frame's charges; or a list / glob

# --- the plane --------------------------------------------------------------------
plane: [1, 0, 2-34]                     # P1, P2, P3: atom index (0-based) or group (2-34, 2-10+12)
                                        #   vertical axis P1 -> P2, horizontal toward P3
origin: midpoint                        # centroid (default) | midpoint (of P1, P2) | atom index | [x, y, z]

# --- what to sum (optional; default qm, plus all if pointcharges are given) ---------
conditions:
  CageOnly: qm:0-133                    # tokens: qm, mm, all, qm:a-b, mm:a-b, mm:a-  (inclusive)
  Cage_Anion: qm:0-133,mm:0-275
  Cage_Anion_Solv: qm:0-133,mm

# --- grid and physics (optional) ----------------------------------------------------
extent: [7.5, 6.0]                      # half-extent (A): one value or [z_half, x_half]
spacing: 0.1                            # A: one value or [dz, dx]
dielectric: 1
dielectric_scale: 1                     # MM charges scaled by 1/sqrt(dielectric_scale), as in pyEF
r_min: 0.5                              # drop source-probe pairs closer than this (A)

# --- masks (stored next to the map, blank in the plot) ---------------------------------
mask_radius: 1.6                        # A around mask_atoms, or per group: {'0-1': 2.0, '2-133': 1.6}
# mask_atoms: 0-133                     # default: all QM atoms
# hull_atoms: 0-133                     # also blank cells outside the convex hull of these atoms
mask_frac: 0.3                          # blank a cell flagged in >= this fraction of frames

# --- output -----------------------------------------------------------------------------
output: esp_profile                     # writes esp_profile.npz (+ .svg/.png)
# name: mysystem                        # system name inside the npz (default: output stem)
plot: true
# point_labels: [Pd2, Pd1, LA1]
# ylabel: distance along Pd-Pd axis (Å)
# xlabel: distance in plane (Å)
# fixed_scale: [-2, 9, viridis]         # pin the colour scale: vmin, vmax, colormap
# atom_slab: 1.0                        # also draw atoms within this distance (A) of the plane
"""

CONFIG_KEYS = {
    'charges', 'molden', 'xyz', 'charge_type', 'multiwfn_path', 'pointcharges',
    'plane', 'origin', 'conditions', 'extent', 'spacing', 'dielectric',
    'dielectric_scale', 'r_min', 'mask_radius', 'mask_atoms', 'hull_atoms',
    'mask_frac', 'output', 'name', 'plot', 'point_labels', 'ylabel', 'xlabel',
    'fixed_scale', 'atom_slab'}


def _natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', path)]


def _resolve_files(spec, base, what):
    """One path, a list of paths, or glob patterns -> sorted absolute paths."""
    import glob
    items = [spec] if isinstance(spec, str) else list(spec)
    out = []
    for item in items:
        pat = os.path.join(base, os.path.expanduser(str(item)))
        if glob.has_magic(pat):
            hits = sorted(glob.glob(pat), key=_natural_key)
            if not hits:
                raise FileNotFoundError(f'{what}: no files match {item!r}')
            out += hits
        else:
            if not os.path.exists(pat):
                raise FileNotFoundError(f'{what}: {pat} not found')
            out.append(pat)
    return [os.path.abspath(f) for f in out]


def _resolve_ptchg(spec, frame_files, base):
    """Point-charge files aligned with ``frame_files``.  A bare file name
    (no directory, no wildcard) is looked up in each frame's directory."""
    if spec is None:
        return None
    if isinstance(spec, str) and os.sep not in spec and not any(c in spec for c in '*?['):
        out = [os.path.join(os.path.dirname(f), spec) for f in frame_files]
        missing = [f for f in out if not os.path.exists(f)]
        if missing:
            raise FileNotFoundError(f'pointcharges: {len(missing)} frame(s) have no '
                                    f'{spec}, e.g. {missing[0]}')
        return out
    out = _resolve_files(spec, base, 'pointcharges')
    if len(out) != len(frame_files):
        raise ValueError(f'{len(frame_files)} frames but {len(out)} point-charge files')
    return _pair_ptchg(frame_files, out)


def _charges_from_molden(cfg, base):
    """Make sure Charges<type>.txt exists next to every molden; returns them."""
    from pyef.analysis import Electrostatics
    molden = _resolve_files(cfg['molden'], base, 'molden')
    if 'xyz' in cfg:
        xyz = _resolve_files(cfg['xyz'], base, 'xyz')
        if len(xyz) != len(molden):
            raise ValueError(f'{len(molden)} molden but {len(xyz)} xyz files')
    else:
        xyz = [os.path.splitext(m)[0] + '.xyz' for m in molden]
    ctype = cfg.get('charge_type', 'CM5')
    charges = [os.path.join(os.path.dirname(m), f'Charges{ctype}.txt') for m in molden]
    todo = [i for i, c in enumerate(charges) if not os.path.exists(c)]
    if todo:
        mw = cfg.get('multiwfn_path')
        if not mw:
            raise ValueError(f'{len(todo)} frame(s) have no Charges{ctype}.txt and '
                             'multiwfn_path is not set')
        print(f'Generating {ctype} charges for {len(todo)} frame(s) with Multiwfn')
        es = Electrostatics([molden[i] for i in todo], [xyz[i] for i in todo])
        owd = os.getcwd()
        for i in todo:
            rc = es.multiwfn.partitionCharge(
                multipole_bool=False, f=os.path.dirname(molden[i]), multiwfn_path=mw,
                charge_type=ctype, owd=owd, molden_filename=os.path.basename(molden[i]),
                xyz_filename=os.path.basename(xyz[i]))
            if rc == -1 or not os.path.exists(charges[i]):
                raise RuntimeError(f'Multiwfn failed for {molden[i]}')
    return charges


def run_config(path):
    """Run an ESP profile from one YAML input file (see ``--template``)."""
    import yaml
    with open(path) as fh:
        cfg = yaml.safe_load(fh) or {}
    unknown = set(cfg) - CONFIG_KEYS
    if unknown:
        raise ValueError(f'{path}: unknown key(s) {sorted(unknown)}; valid keys: '
                         f'{sorted(CONFIG_KEYS)}')
    if ('charges' in cfg) == ('molden' in cfg):
        raise ValueError(f'{path}: give exactly one of "charges" or "molden"')
    if 'plane' not in cfg:
        raise ValueError(f'{path}: "plane: [P1, P2, P3]" is required')
    base = os.path.dirname(os.path.abspath(path))

    charges = (_resolve_files(cfg['charges'], base, 'charges') if 'charges' in cfg
               else _charges_from_molden(cfg, base))
    ptchg = _resolve_ptchg(cfg.get('pointcharges'), charges, base)
    n = len(charges)
    print(f'{n} frame{"s" if n > 1 else ""}: '
          + ('averaging, each frame in its own plane' if n > 1 else 'single-frame map'))

    origin = cfg.get('origin', 'centroid')
    if isinstance(origin, str) and origin.isdigit():
        origin = int(origin)
    res = compute_esp_profile(
        charges, cfg['plane'], ptchg_files=ptchg, conditions=cfg.get('conditions'),
        origin=origin, extent=cfg.get('extent', (7.5, 6.0)),
        spacing=cfg.get('spacing', 0.1), r_min=cfg.get('r_min', 0.5),
        dielectric=cfg.get('dielectric', 1.0),
        mm_scale=1.0 / np.sqrt(cfg.get('dielectric_scale', 1.0)),
        mask_radius=cfg.get('mask_radius', 1.6) or None,
        mask_atoms=cfg.get('mask_atoms'), hull_atoms=cfg.get('hull_atoms'),
        mask_frac=cfg.get('mask_frac', 0.3))

    out = os.path.join(base, cfg.get('output', os.path.splitext(os.path.basename(path))[0]))
    name = str(cfg.get('name', os.path.basename(out)))
    save_esp_profile(f'{out}.npz', res, name=name)
    print(f'-> {out}.npz  (system {name!r})')
    for c in res['conditions']:
        print(f'  {c:20s} mean over unmasked cells {np.nanmean(masked(res, c)):+8.3f} V')
    if cfg.get('plot', True):
        import matplotlib
        matplotlib.use('Agg')
        kw = {k: cfg[k] for k in ('ylabel', 'xlabel', 'atom_slab') if k in cfg}
        if 'point_labels' in cfg:
            kw['point_labels'] = tuple(cfg['point_labels'])
        if 'fixed_scale' in cfg:
            kw['fixed'] = tuple(cfg['fixed_scale'])
        plot_esp_profile(res, out=out, **kw)
        print(f'-> {out}.svg, {out}.png')
    return res


def build_parser():
    p = argparse.ArgumentParser(
        prog='python -m pyef.esp_profile',
        description='ESP map on the plane through three atoms (or atom-group '
                    'centroids) from pyEF charge files: one frame, or the average '
                    'of many.  Easiest: python -m pyef.esp_profile --template > '
                    'esp.yaml, edit it, then python -m pyef.esp_profile -c esp.yaml')
    p.add_argument('-c', '--config', help='YAML input file (all settings in one place)')
    p.add_argument('--template', action='store_true',
                   help='print a commented input file and exit')
    p.add_argument('--charges', nargs='+',
                   help='Charges<type>.txt, one per frame')
    p.add_argument('--ptchg', nargs='+', default=None,
                   help='point-charge files, one per frame, same order as --charges')
    p.add_argument('--plane', nargs=3, metavar=('P1', 'P2', 'P3'),
                   help='QM atom index or group (2-34, 2-10+12) for each point; '
                        'vertical axis P1->P2, horizontal toward P3')
    p.add_argument('--origin', nargs='+', default=['centroid'],
                   help='centroid (default), midpoint, an atom index, or x y z')
    p.add_argument('--condition', action='append', default=None,
                   help="NAME=TOKENS, e.g. CageOnly=qm:0-133 or 'Cage_Anion=qm:0-133,mm:0-275'; "
                        'repeatable. Default: qm (and all with --ptchg)')
    p.add_argument('--extent', nargs='+', type=float, default=[7.5, 6.0],
                   help='half-extent in A: one value or Z_HALF X_HALF (default 7.5 6.0)')
    p.add_argument('--spacing', nargs='+', type=float, default=[0.1],
                   help='grid spacing in A: one value or DZ DX (default 0.1)')
    p.add_argument('--dielectric', type=float, default=1.0)
    p.add_argument('--mm-scale', type=float, default=1.0,
                   help='factor on every MM charge (default 1)')
    p.add_argument('--r-min', type=float, default=0.5,
                   help='drop source-probe pairs closer than this, A (default 0.5)')
    p.add_argument('--mask-radius', type=_mask_radius_arg, default=1.6,
                   help='mask cells within this distance of mask atoms, A (0 disables), '
                        'or per-group radii ATOMS:R,..., e.g. 0-1:2.0,2-133:1.6')
    p.add_argument('--mask-atoms', default=None,
                   help='QM atoms for the radius mask (default: all QM atoms)')
    p.add_argument('--hull', default=None,
                   help='mask cells outside the convex hull of these QM atoms')
    p.add_argument('--mask-frac', type=float, default=0.3)
    p.add_argument('--name', default='profile', help='system name in the npz')
    p.add_argument('--out', default='esp_profile',
                   help='output stem: writes STEM.npz and STEM.svg/.png')
    p.add_argument('--no-plot', action='store_true')
    return p


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if a.template:
        print(TEMPLATE, end='')
        return None
    if a.config:
        return run_config(a.config)
    if not a.charges or not a.plane:
        parser.error('give -c CONFIG.yaml, or --charges and --plane')
    for opt, v in (('--extent', a.extent), ('--spacing', a.spacing)):
        if len(v) not in (1, 2):
            raise SystemExit(f'error: {opt} takes one or two values')
    res = compute_esp_profile(
        a.charges, a.plane, ptchg_files=_pair_ptchg(a.charges, a.ptchg),
        conditions=a.condition, origin=_origin_arg(a.origin),
        extent=a.extent, spacing=a.spacing, r_min=a.r_min,
        dielectric=a.dielectric, mm_scale=a.mm_scale,
        mask_radius=a.mask_radius or None, mask_atoms=a.mask_atoms,
        hull_atoms=a.hull, mask_frac=a.mask_frac)
    save_esp_profile(f'{a.out}.npz', res, name=a.name)
    print(f'{res["n_frames"]} frames -> {a.out}.npz ({a.name})')
    for c in res['conditions']:
        print(f'  {c:20s} mean over unmasked cells {np.nanmean(masked(res, c)):+8.3f} V')
    if not a.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        plot_esp_profile(res, out=a.out)
        print(f'plot -> {a.out}.svg, {a.out}.png')
    return res


if __name__ == '__main__':
    main()
