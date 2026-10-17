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
    """Highest index each source must have, to check against each frame.  An
    open range ``a-`` only needs ``a`` charges (it may be empty)."""
    top = {s: -1 for s in SOURCES}
    for t in conditions.values():
        for s, a, b in parse_condition(t):
            top[s] = max(top[s], a - 1 if b is None else b)
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


def esp_at(probes, src_xyz, src_q, r_min=0.5, k=COULOMB_EV_ANGSTROM, core='drop'):
    """ESP (V) at ``probes`` (N, 3) from point charges, chunked over probes.

    Source-probe pairs closer than ``r_min`` (A, > 0) are dropped
    (``core='drop'``), or counted as if ``r_min`` apart (``core='clamp'``,
    which keeps the map continuous: a nucleus saturates instead of leaving a
    hole).  ``k`` is the Coulomb constant in V*A/e (divide by a dielectric to
    screen).
    """
    if core not in ('drop', 'clamp'):
        raise ValueError(f"core must be 'drop' or 'clamp', got {core!r}")
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
        if core == 'clamp':
            np.minimum(r2, 1.0 / r_min, out=r2)
        else:
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


# Bondi (1964) van der Waals radii, A; B and Al from Mantina et al. (2009).
# Elements not listed fall back to DEFAULT_VDW_RADIUS (with a warning); set
# your own with ``vdw_radii={'Fe': 2.0}``.
VDW_RADII = {
    'H': 1.20, 'He': 1.40, 'Li': 1.82, 'B': 1.92, 'C': 1.70, 'N': 1.55,
    'O': 1.52, 'F': 1.47, 'Ne': 1.54, 'Na': 2.27, 'Mg': 1.73, 'Al': 1.84,
    'Si': 2.10, 'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Ar': 1.88, 'K': 2.75,
    'Ni': 1.63, 'Cu': 1.40, 'Zn': 1.39, 'Ga': 1.87, 'As': 1.85, 'Se': 1.90,
    'Br': 1.85, 'Kr': 2.02, 'Pd': 1.63, 'Ag': 1.72, 'Cd': 1.58, 'In': 1.93,
    'Sn': 2.17, 'Te': 2.06, 'I': 1.98, 'Xe': 2.16, 'Pt': 1.75, 'Au': 1.66,
    'Hg': 1.55, 'Tl': 1.96, 'Pb': 2.02, 'U': 1.86}
DEFAULT_VDW_RADIUS = 2.0


def vdw_radii_for(symbols, scale=1.0, overrides=None):
    """Per-atom van der Waals radius (A) times ``scale``.

    ``overrides`` (``{'Fe': 2.0}``) takes precedence over :data:`VDW_RADII`;
    unknown elements get :data:`DEFAULT_VDW_RADIUS` and a warning.
    """
    table = dict(VDW_RADII)
    table.update({str(k).capitalize(): float(v) for k, v in (overrides or {}).items()})
    out, unknown = [], set()
    for sym in symbols:
        el = re.match(r'[A-Za-z]{1,2}', str(sym))
        el = el.group(0).capitalize() if el else str(sym)
        if el not in table:
            unknown.add(el)
        out.append(table.get(el, DEFAULT_VDW_RADIUS))
    if unknown:
        warnings.warn(f'no van der Waals radius for {sorted(unknown)}; using '
                      f'{DEFAULT_VDW_RADIUS} A (set vdw_radii to override)')
    return scale * np.array(out)


# Cordero et al. (2008) covalent radii, A (low-spin for Mn, Fe, Co), for
# working out which atoms are bonded.  Unlisted elements use 1.5 A.
COVALENT_RADII = {
    'H': 0.31, 'Li': 1.28, 'Be': 0.96, 'B': 0.84, 'C': 0.76, 'N': 0.71,
    'O': 0.66, 'F': 0.57, 'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11,
    'P': 1.07, 'S': 1.05, 'Cl': 1.02, 'K': 2.03, 'Ca': 1.76, 'Sc': 1.70,
    'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39, 'Fe': 1.32, 'Co': 1.26,
    'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22, 'Ga': 1.22, 'Ge': 1.20, 'As': 1.19,
    'Se': 1.20, 'Br': 1.20, 'Y': 1.90, 'Zr': 1.75, 'Nb': 1.64, 'Mo': 1.54,
    'Tc': 1.47, 'Ru': 1.46, 'Rh': 1.42, 'Pd': 1.39, 'Ag': 1.45, 'Cd': 1.44,
    'In': 1.42, 'Sn': 1.39, 'Sb': 1.39, 'Te': 1.38, 'I': 1.39, 'La': 2.07,
    'Hf': 1.75, 'Ta': 1.70, 'W': 1.62, 'Re': 1.51, 'Os': 1.44, 'Ir': 1.41,
    'Pt': 1.36, 'Au': 1.36, 'Hg': 1.32, 'Pb': 1.46, 'Bi': 1.48}
# d- and f-block elements: the nodes of a metal-organic cage
CAGE_METALS = set(
    'Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd La Ce Pr Nd '
    'Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg'.split())
BOND_TOL = 1.2              # bonded if d < BOND_TOL * (r_cov,i + r_cov,j)
METAL_BOND_TOL = 1.3        # looser for metal-ligand (coordination) bonds


def _element(sym):
    m = re.match(r'[A-Za-z]{1,2}', str(sym))
    return m.group(0).capitalize() if m else str(sym)


def bond_pairs(symbols, xyz):
    """Bonded atom pairs ``(i, j, metal)`` by covalent radii: closer than
    ``BOND_TOL`` (``METAL_BOND_TOL`` when either atom is a metal) times the sum
    of the radii.  ``metal`` flags the bonds that involve a metal."""
    from scipy.spatial import cKDTree
    xyz = np.asarray(xyz, dtype=float)
    els = [_element(s) for s in symbols]
    rad = np.array([COVALENT_RADII.get(e, 1.5) for e in els])
    is_metal = np.array([e in CAGE_METALS for e in els])
    pairs = cKDTree(xyz).query_pairs(METAL_BOND_TOL * 2 * rad.max(), output_type='ndarray')
    if not len(pairs):
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0, dtype=bool)
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(xyz[i] - xyz[j], axis=1)
    either = is_metal[i] | is_metal[j]
    bonded = d < np.where(either, METAL_BOND_TOL, BOND_TOL) * (rad[i] + rad[j])
    return i[bonded], j[bonded], either[bonded]


def detect_system(symbols, xyz):
    """Find the metals, ligands, cage and guest in one QM geometry.

    Atoms are bonded if closer than ``BOND_TOL`` (``METAL_BOND_TOL`` for a
    metal) times the sum of their covalent radii.  Metals are the d/f-block
    atoms; a ligand is a metal-free fragment bonded to a metal; the cage is
    the metals plus their ligands; the guest is every other QM atom.

    Returns a dict with ``metals``, ``ligands`` (lists of indices, ordered by
    their lowest index), ``cage`` and ``guest`` (sorted index lists).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    els = [_element(s) for s in symbols]
    n = len(els)
    is_metal = np.array([e in CAGE_METALS for e in els])
    i, j, either = bond_pairs(symbols, xyz)
    # fragments of the metal-free bond graph
    org = ~either
    g = coo_matrix((np.ones(org.sum()), (i[org], j[org])), shape=(n, n))
    _, label = connected_components(g, directed=False)
    metals = [int(k) for k in np.flatnonzero(is_metal)]
    touching = set()
    for a, b in zip(i[either], j[either]):
        for m, x in ((a, b), (b, a)):
            if is_metal[m] and not is_metal[x]:
                touching.add(label[x])
    ligands = sorted(([int(k) for k in np.flatnonzero(label == L)] for L in touching),
                     key=min)
    cage = sorted(set(metals).union(*ligands)) if ligands or metals else []
    guest = sorted(set(range(n)) - set(cage))
    return {'metals': metals, 'ligands': ligands, 'cage': cage, 'guest': guest}


def index_spec_str(idx):
    """Compact spec for an index list: [0, 1, 2, 5] -> '0-2+5'."""
    idx = sorted(int(i) for i in idx)
    runs, start = [], None
    for a, b in zip(idx, idx[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            runs.append(f'{start}-{a}' if a > start else f'{start}')
            start = None
    return '+'.join(runs)


def describe_system(system, symbols):
    lines = []
    m = system['metals']
    lines.append(f'  metals: {len(m)}  ' + ', '.join(f'{_element(symbols[k])}{k}' for k in m))
    sizes = sorted({len(L) for L in system['ligands']})
    lines.append(f'  ligands: {len(system["ligands"])} (atoms each: '
                 f'{", ".join(map(str, sizes)) or "-"})  '
                 + '  '.join(index_spec_str(L) for L in system['ligands']))
    lines.append(f'  cage: {len(system["cage"])} atoms  {index_spec_str(system["cage"]) or "-"}')
    lines.append(f'  guest / rest of QM: {len(system["guest"])} atoms  '
                 f'{index_spec_str(system["guest"]) or "-"}')
    return '\n'.join(lines)


def _named_sets(system):
    return {'cage': system['cage'], 'guest': system['guest'], 'metals': system['metals']}


def _atom_indices(spec, n, named=None):
    """Index spec -> list.  'all' / 'qm' / None mean every QM atom; 'cage',
    'guest' and 'metals' come from :func:`detect_system`."""
    if spec is None or (isinstance(spec, str) and spec.strip().lower() in ('all', 'qm')):
        return list(range(n))
    if isinstance(spec, str) and spec.strip().lower() in ('cage', 'guest', 'metals'):
        idx = (named or {}).get(spec.strip().lower(), [])
        if not idx:
            raise ValueError(f'no {spec.strip().lower()} atoms were detected; give indices instead')
        return list(idx)
    idx = parse_index_spec(spec)
    if max(idx) >= n:
        raise IndexError(f'atom index {max(idx)} out of range for {n} QM atoms')
    return idx


def auto_extent(charge_files, make_frame, atoms, pad=2.0, spacing=0.1):
    """Half-extent ``(z_half, x_half)`` that holds every ``atoms`` atom (index
    list) of every frame, projected into that frame's plane (built by
    ``make_frame(xyz)``), plus ``pad`` A, rounded up to whole grid steps."""
    dz, dx = np.broadcast_to(np.asarray(spacing, dtype=float), (2,))
    zmax = xmax = 0.0
    for cf in charge_files:
        _, xyz, _ = read_charges(cf)
        frame, _ = make_frame(xyz)
        loc = frame.to_local(xyz[atoms])
        zmax = max(zmax, np.abs(loc[:, 2]).max())
        xmax = max(xmax, np.abs(loc[:, 0]).max())
    return (float(np.ceil((zmax + pad) / dz) * dz),
            float(np.ceil((xmax + pad) / dx) * dx))


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


def plane_frame(xyz, plane_points, origin=None, cut='vertical', offset=0.0):
    """Plane frame from QM coordinates and index specs.

    ``plane_points`` is three index specs (int, ``'2-34'``, or a list) into
    ``xyz``.  ``cut='vertical'`` (default): the plane through P1, P2, P3
    (:func:`plane_basis`).  ``cut='horizontal'``: the plane perpendicular to
    the P1 -> P2 axis, with the axis at the map centre and the horizontal
    axis toward P3.  ``origin`` is ``'centroid'``, ``'midpoint'``, an atom
    index, or a length-3 lab xyz (default: midpoint for a horizontal cut,
    centroid otherwise); ``offset`` (A) slides the plane along its normal.
    Returns ``(frame, points (3, 3))``.
    """
    if cut not in ('vertical', 'horizontal'):
        raise ValueError(f"cut must be 'vertical' or 'horizontal', got {cut!r}")
    if origin is None:
        origin = 'midpoint' if cut == 'horizontal' else 'centroid'

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
    if cut == 'vertical':
        frame = plane_basis(p1, p2, p3, origin=o)
    else:
        v = p2 - p1
        length = float(np.linalg.norm(v))
        if length == 0.0:
            raise ValueError('P1 and P2 coincide')
        a = v / length
        if o is None:
            o = (p1 + p2 + p3) / 3.0
        c = p1 + np.dot(o - p1, a) * a          # the axis point at the origin's height
        w = (p3 - c) - np.dot(p3 - c, a) * a
        if np.linalg.norm(w) < 1e-3 * max(np.linalg.norm(p3 - c), 1e-12):
            raise ValueError('P3 lies on the P1-P2 axis; the horizontal direction is undefined')
        x_hat = w / np.linalg.norm(w)
        frame = PlaneFrame(c, np.column_stack([x_hat, a, np.cross(x_hat, a)]), length)
    if offset:
        frame.origin = frame.origin + float(offset) * frame.y_hat
    return frame, np.array(pts)


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


def compute_esp_profile(charge_files, plane_points='auto', ptchg_files=None,
                        conditions=None, origin=None, extent='auto',
                        spacing=0.1, r_min=0.5, core='clamp', dielectric=1.0, mm_scale=1.0,
                        mask_radius=None, mask_atoms=None, hull_atoms=None,
                        mask_frac=0.3, vdw_scale=1.0, vdw_radii=None,
                        extent_pad=2.0, extent_atoms=None, cut='vertical',
                        offset=0.0, verbose=True):
    """Frame-averaged ESP maps on the plane through three atoms/centroids.

    Parameters
    ----------
    charge_files : list of str
        One ``Charges<type>.txt`` per frame.
    plane_points : sequence of 3 index specs, or 'auto'
        P1, P2, P3 as QM atom indices or groups (``1``, ``'2-34'``, ``[2, 3]``).
        ``'auto'`` (default): the two cage metals and the first ligand's
        centroid, from :func:`detect_system` (needs exactly two d/f-block
        metals).
    ptchg_files : list of str or None
        One point-charge file per frame (``None`` entries allowed), aligned
        with ``charge_files``.
    conditions : dict, list or None
        ``{name: tokens}``, or a list of ``'name=tokens'`` / bare token
        strings.  Tokens: ``qm``, ``mm``, ``all``, ``qm:a-b``, ``mm:a-``, and
        the detected sets ``cage``, ``guest``, ``metals``.  Default: ``Cage``
        and ``Cage_Env`` (with point charges) when a cage is detected -- the
        guest is not a source -- else ``qm`` (and ``all``).
    origin : 'centroid', 'midpoint', atom index, lab xyz, or None
        None: midpoint for an auto plane or a horizontal cut, else centroid.
    cut : 'vertical' or 'horizontal'
        Vertical: the plane through P1, P2, P3.  Horizontal: perpendicular to
        P1 -> P2 through the origin (e.g. a cage's equatorial plane).
    offset : float
        Slide the plane this far (A) along its normal.
    extent : float, (z_half, x_half) or 'auto'
        Half-extent of the map in A.  ``'auto'`` fits every ``extent_atoms``
        atom (default: ``hull_atoms`` if given, else all QM atoms) of every
        frame, projected into its plane, plus ``extent_pad`` A -- the whole
        cage for any cage size.
    spacing : float or (dz, dx), A.
    r_min : float
        Source-probe distances below this (A) are treated as ``r_min``
        (``core='clamp'``, default: the map stays continuous and a nucleus
        saturates) or the pair is dropped (``core='drop'``, the older
        behaviour, which leaves a pale hole of radius ``r_min`` at each atom).
    dielectric : float
        Uniform dielectric; the ESP is divided by it.
    mm_scale : float
        Factor applied to every MM charge (pyEF uses
        ``1/sqrt(dielectric_scale)``).
    mask_radius : float, dict, 'vdw' or None
        Cells within this distance (A) of a ``mask_atoms`` atom are flagged
        (default None: no mask; the plot's ``esp_range`` clips the nuclei instead).
        A dict ``{atom_spec: radius}`` gives each atom group its own radius,
        e.g. ``{'0-1': 2.0, '2-133': 1.6}`` (``mask_atoms`` is then ignored).
        ``'vdw'`` uses each atom's van der Waals radius times ``vdw_scale``
        (:data:`VDW_RADII`, overridable per element with ``vdw_radii``).
    mask_atoms : index spec, 'all', 'cage' or None
        QM atoms for a float or 'vdw' ``mask_radius``; default the detected
        cage (so a guest does not blank the cavity), else all QM atoms.
    hull_atoms : index spec, 'all' or None
        Cells outside the convex hull of these QM atoms are flagged
        (``'all'``: every QM atom, e.g. a cage with nothing else in the QM region).
    mask_frac : float
        A cell is masked if flagged in at least this fraction of frames.
    vdw_scale, vdw_radii
        See ``mask_radius='vdw'``.
    extent_pad, extent_atoms
        See ``extent='auto'``.

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
    say = print if verbose else (lambda *a, **k: None)

    # work out the system from the first frame
    syms0, xyz0, _ = read_charges(charge_files[0])
    n0 = len(syms0)
    try:
        system = detect_system(syms0, xyz0)
    except Exception as e:                        # odd input: named sets unavailable
        warnings.warn(f'could not detect the cage ({e}); use explicit indices')
        system = {'metals': [], 'ligands': [], 'cage': [], 'guest': list(range(n0))}
    named = _named_sets(system)
    say('detected (frame 1):\n' + describe_system(system, syms0))
    auto_plane = plane_points is None or (isinstance(plane_points, str)
                                          and plane_points.strip().lower() == 'auto')
    if auto_plane:
        if len(system['metals']) != 2 or not system['ligands']:
            raise ValueError(
                f'plane auto needs exactly two cage metals with ligands; found '
                f'{len(system["metals"])} metal(s), {len(system["ligands"])} ligand(s). '
                'Give plane_points=(P1, P2, P3) instead.')
        m1, m2 = system['metals']
        plane_points = (m2, m1, system['ligands'][0])
        say(f'plane auto: P1 = {_element(syms0[m2])}{m2}, P2 = {_element(syms0[m1])}{m1}, '
            f'P3 = centroid of ligand {index_spec_str(system["ligands"][0])}')
    if origin is None:
        origin = 'midpoint' if (auto_plane or cut == 'horizontal') else 'centroid'
    plane_points = [_atom_indices(pp, n0, named) if isinstance(pp, str) else pp
                    for pp in plane_points]

    if conditions is None and system['cage']:
        conditions = {'Cage': 'cage'}           # the guest is left out by default
        if have_mm:
            conditions['Cage_Env'] = 'cage,mm'
    conditions = _normalize_conditions(conditions, have_mm)
    condition_tokens = {c: t for c, t in conditions.items()}
    conditions = {c: _expand_named(t, named) for c, t in conditions.items()}
    segments, seg_conditions = _segment_conditions(conditions)
    top = _closed_stops(conditions)

    def make_frame(xyz):
        return plane_frame(xyz, plane_points, origin, cut, offset)

    if mask_atoms is None and not isinstance(mask_radius, dict) and system['cage']:
        mask_atoms = 'cage'
    hull_idx = None if hull_atoms is None else _atom_indices(hull_atoms, n0, named)
    # the cavity: inside the hull of the cage and clear of its atoms (2.0 A
    # around a metal, 1.6 A around the rest).  Never masked; it only feeds the
    # plot's dashed outline and esp_range='cavity'.
    cav_hull = hull_idx if hull_idx is not None else (system['cage'] or None)
    cav_metals = [i for i in system['metals'] if cav_hull and i in cav_hull]
    cav_rest = [i for i in (cav_hull or []) if i not in cav_metals]
    if isinstance(extent, str):
        if extent.strip().lower() != 'auto':
            raise ValueError(f"extent must be a number, a pair or 'auto', got {extent!r}")
        ext_atoms = (_atom_indices(extent_atoms, n0, named) if extent_atoms is not None
                     else hull_idx if hull_idx is not None
                     else system['cage'] or list(range(n0)))
        extent = auto_extent(charge_files, make_frame, ext_atoms, extent_pad, spacing)
        say(f'extent auto: z +/-{extent[0]:g} A, x +/-{extent[1]:g} A')
    zh, xh, nz, nx = grid_shape(extent, spacing)
    z_vals, x_vals, local = plane_grid(zh, xh, nz, nx)
    k = COULOMB_EV_ANGSTROM / dielectric
    mask_sets = None                          # built from the first frame

    maps = {c: [] for c in conditions}
    excluded = np.zeros(len(local))
    outside_cavity = np.zeros(len(local))
    dists, ref_local = [], []
    symbols, n_qm = None, None
    for cf, pf in zip(charge_files, ptchg_files):
        syms, qm_xyz, qm_q = read_charges(cf)
        if n_qm is None:
            symbols, n_qm = syms, len(syms)
            mask_sets = _mask_sets(mask_radius, mask_atoms, syms, vdw_scale, vdw_radii, named)
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

        frame, pts = make_frame(qm_xyz)
        probes = frame.to_lab(local)

        excl = np.zeros(len(local), dtype=bool)
        for idx, r in mask_sets:
            excl |= min_dist(probes, qm_xyz[idx]) < r
        if hull_idx is not None:
            excl |= hull_mask(local, frame.to_local(qm_xyz[hull_idx]))
        excluded += excl
        if cav_hull:
            try:
                out = hull_mask(local, frame.to_local(qm_xyz[cav_hull]))
            except Exception:                 # too few / coplanar atoms: no cavity
                cav_hull = None
            else:
                for idx, r in ((cav_metals, 2.0), (cav_rest, 1.6)):
                    if idx:
                        out |= min_dist(probes, qm_xyz[idx]) < r
                outside_cavity += out

        seg_v = {g: esp_at(probes, src[s][0][a:b], src[s][1][a:b], r_min=r_min, k=k,
                           core=core)
                 for g, (s, a, b) in segments.items()}
        for c, gs in seg_conditions.items():
            maps[c].append(sum((seg_v[g] for g in gs), np.zeros(len(local))).reshape(nz, nx))

        dists.append(frame.axis_length)
        ref_local.append(frame.to_local(np.vstack([pts, qm_xyz])))

    n_frames = len(charge_files)
    ref = np.mean(ref_local, axis=0)
    # the frame closest to the mean: a real geometry to draw (the mean itself
    # is not one -- a guest that tumbles averages to a collapsed blob)
    medoid = int(np.argmin([np.sum((r - ref) ** 2) for r in ref_local]))
    res = {'n_frames': n_frames, 'z_vals': z_vals, 'x_vals': x_vals,
           'mask': (excluded / n_frames >= mask_frac).reshape(nz, nx),
           'cavity': (outside_cavity / n_frames < mask_frac).reshape(nz, nx)
           if cav_hull else np.zeros((nz, nx), dtype=bool),
           'mean_dist': float(np.mean(dists)), 'std_dist': float(np.std(dists)),
           'plane_points_local': ref[:3], 'atoms_local': ref[3:],
           'structure_local': ref_local[medoid][3:], 'structure_frame': medoid,
           'symbols': np.array(symbols),
           'frames': np.array([os.path.abspath(f) for f in charge_files]),
           'condition_tokens': np.array(
               [f'{c}={t if isinstance(t, str) else ",".join(t)}'
                for c, t in condition_tokens.items()]),
           'cut': cut, 'offset': float(offset),
           'plane_spec': np.array([index_spec_str(parse_index_spec(pp)) for pp in plane_points]),
           'cage_atoms': np.array(system['cage'], dtype=int),
           'guest_atoms': np.array(system['guest'], dtype=int),
           'metal_atoms': np.array(system['metals'], dtype=int),
           'conditions': list(conditions),
           'dielectric': float(dielectric), 'r_min': float(r_min),
           'extent': np.array([zh, xh])}
    for c in conditions:
        arr = np.stack(maps[c])
        res[f'mean_{c}'] = arr.mean(axis=0)
        res[f'std_{c}'] = arr.std(axis=0)
    return res


def _mask_sets(mask_radius, mask_atoms, symbols, vdw_scale=1.0, vdw_radii=None,
               named=None):
    """``[(atom indices, radius), ...]`` for the radius mask; atoms sharing
    a radius are grouped so each group costs one distance pass."""
    n = len(symbols)
    if isinstance(mask_radius, dict):
        return [(_atom_indices(a, n, named), float(r)) for a, r in mask_radius.items()]
    if isinstance(mask_radius, str):
        if mask_radius.strip().lower() != 'vdw':
            raise ValueError(f"mask_radius must be a number, a dict or 'vdw', got {mask_radius!r}")
        idx = _atom_indices(mask_atoms, n, named)
        radii = vdw_radii_for([symbols[i] for i in idx], vdw_scale, vdw_radii)
        return [([i for i, r in zip(idx, radii) if r == u], float(u))
                for u in np.unique(radii)]
    if mask_radius:
        return [(_atom_indices(mask_atoms, n, named), float(mask_radius))]
    return []


def _expand_named(tokens, named):
    """Replace 'cage' / 'guest' / 'metals' tokens by their qm index runs."""
    if not isinstance(tokens, str):
        tokens = ','.join(tokens)
    out = []
    for tok in (t for t in re.split(r'[,\s]+', tokens.strip()) if t):
        key = tok.lower()
        if key in ('cage', 'guest', 'metals'):
            idx = named.get(key, [])
            if not idx:
                raise ValueError(f'condition token {tok!r}: no {key} atoms were detected')
            out += [f'qm:{r}' for r in index_spec_str(idx).split('+')]
        else:
            out.append(tok)
    return ','.join(out)


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


def _draw_structure(ax, result, slab=1.5, color='white', lw=1.5, skip=()):
    """Heavy-atom bonds within ``slab`` A of the plane as sticks, with a dark
    edge so they read on both the deep-red walls and a pale cavity; metals in
    the slab as circles.  Coordinates are local: x across, y out of plane, z up."""
    import matplotlib.patheffects as pe
    xyz = np.asarray(result.get('structure_local', result.get('atoms_local')), dtype=float)
    syms = [_element(s) for s in result['symbols']]
    heavy = np.array([e != 'H' for e in syms])
    near = np.abs(xyz[:, 1]) <= slab
    near[np.asarray(skip, dtype=int)] = False
    edge = [pe.withStroke(linewidth=lw + 1.2, foreground='0.15', alpha=0.85)]
    i, j, _ = bond_pairs(syms, xyz)
    for a, b in zip(i, j):
        if heavy[a] and heavy[b] and near[a] and near[b]:
            ax.plot(xyz[[a, b], 0], xyz[[a, b], 2], '-', lw=lw, color=color,
                    solid_capstyle='round', zorder=8, path_effects=edge)
    for k, e in enumerate(syms):
        if e in CAGE_METALS and near[k]:
            ax.plot(xyz[k, 0], xyz[k, 2], 'o', color=color, ms=9, zorder=10,
                    markeredgecolor='black', markeredgewidth=1.5)


def _cavity_scale(result, maps, cavity, clear=1.6, pct=99.0):
    """``(vmin, vmax, cmap)`` from the cavity cells at least ``clear`` A from
    every atom (a guest's nuclei would otherwise set the limits).  Not anchored
    at zero: a cavity that is all +5 to +10 V gets that range, sequential."""
    sel = cavity.copy()
    atoms = result.get('atoms_local')
    if atoms is not None and len(atoms):
        ZZ, XX = np.meshgrid(result['z_vals'], result['x_vals'], indexing='ij')
        cells = np.column_stack([XX.ravel(), np.zeros(XX.size), ZZ.ravel()])
        far = (min_dist(cells, np.asarray(atoms, dtype=float)) >= clear).reshape(XX.shape)
        if (sel & far).any():
            sel &= far
    vals = np.concatenate([np.asarray(m)[sel & np.isfinite(m)] for m in maps])
    lo, hi = np.percentile(vals, 100 - pct), np.percentile(vals, pct)
    if lo < 0.0 < hi:
        v = max(-lo, hi)
        return -v, v, 'RdBu_r'
    return lo, hi, 'Blues_r' if hi <= 0.0 else 'YlOrRd'


def _contour_levels(vmin, vmax, step):
    """Iso-potential levels on an absolute grid of ``step`` volts, so one line
    means the same potential difference in every panel; 11 proportional
    levels if that would give fewer than three lines."""
    if step and step > 0:
        lev = np.arange(np.ceil(vmin / step - 1e-9) * step, vmax + 1e-9 * step, step)
        if len(lev) >= 3:
            return lev
    return np.linspace(vmin, vmax, 11)


def plot_esp_profile(result, conditions=None, out=None, fixed=None, esp_range='cavity',
                     show_points=False, cavity_outline=False, structure=True,
                     structure_slab=1.5, structure_color='white', structure_guest=False,
                     annotate_mean=False,
                     point_labels=('P1', 'P2', 'P3'), atom_slab=None,
                     ylabel=None, xlabel=None, level_step=1.0,
                     panel_size=3.5, exts=('svg', 'png'), dpi=300):
    """One row of panels, one per condition, on a shared colour scale.

    Masked cells are blank.  Gray contours every ``level_step`` V, a white
    ESP = 0 line where a panel crosses zero, and each panel's mean over
    unmasked cells in the corner.  P1 and P2 are gold circles, P3 a green
    triangle, each only if it lies within 0.5 A of the plane (in a horizontal
    cut the P1-P2 axis is the map centre); ``atom_slab`` (A) also draws the QM atoms
    within that distance of the plane.  ``fixed=(vmin, vmax, cmap)`` pins the
    colour scale.  ``esp_range=(vmin, vmax)`` does the
    same with the colour map chosen for you (diverging if the range spans zero,
    else sequential); values outside it saturate.  ``esp_range='cavity'`` (default): the
    limits come from the cavity alone (1-99th percentile of the cells at least
    1.6 A from every atom), so the metals, the cage walls and the guest's
    nuclei saturate instead of washing the cavity out.  The range is not
    anchored at zero.  Panels with very different ranges (e.g. a guest-only
    condition) are better plotted separately: the scale is shared.
    ``esp_range=None``, or 'cavity' when no cage was detected: automatic,
    percentile-clipped over the whole map.

    The cavity (inside the cage's convex hull, clear of its atoms) is drawn as
    a dashed outline (``cavity_outline``).  ``annotate_mean`` writes the mean
    ESP in each panel's corner: over the cavity when there is one, else over
    the unmasked map.
    ``show_points=True`` marks P1/P2/P3.

    ``structure=True`` draws the molecule over the map as white sticks with a
    dark edge: every bond between heavy atoms within ``structure_slab`` A of
    the plane, and metals in that slab as white circles.  With several frames
    the geometry is the frame closest to the average (``structure_frame``),
    since the average itself is not a molecule.  The guest is not drawn
    unless ``structure_guest=True``.  With ``out`` set, saves ``out.<ext>`` and closes the
    figure.  Returns ``(fig, axes, (vmin, vmax, cmap))``.
    """
    import matplotlib.pyplot as plt

    conds = list(conditions) if conditions else list(result['conditions'])
    maps = [masked(result, c) for c in conds]
    z_vals, x_vals = np.asarray(result['z_vals']), np.asarray(result['x_vals'])
    cavity = np.asarray(result.get('cavity', np.zeros(np.shape(maps[0]), dtype=bool)), dtype=bool)
    if isinstance(esp_range, str):
        if esp_range != 'cavity':
            raise ValueError(f"esp_range must be (vmin, vmax), 'cavity' or None, got {esp_range!r}")
        if cavity.any():
            fixed = fixed or _cavity_scale(result, maps, cavity)
    elif fixed is None and esp_range is not None:
        lo, hi = sorted(float(v) for v in esp_range)
        fixed = (lo, hi, 'RdBu_r' if lo < 0.0 < hi else ('Blues_r' if hi <= 0.0 else 'YlOrRd'))
    vmin, vmax, cmap = tuple(fixed) if fixed is not None else _scale_for(maps)
    pts = np.asarray(result['plane_points_local'])
    horizontal = str(result.get('cut', 'vertical')) == 'horizontal'
    if ylabel is None:
        ylabel = ('in plane, ⊥ to P1→P2 axis (Å)' if horizontal
                  else 'distance along P1→P2 axis (Å)')
    if xlabel is None:
        xlabel = 'in plane, toward P3 (Å)'
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
            if cavity_outline and cavity.any():
                ax.contour(x_vals, z_vals, cavity.astype(float), levels=[0.5],
                           colors='k', linewidths=1.1, linestyles='--', alpha=0.75,
                           zorder=4)
            if np.isfinite(m).any():
                if np.nanmin(m) < 0.0 < np.nanmax(m):
                    ax.contour(x_vals, z_vals, np.where(np.isfinite(m), m, 1e3),
                               levels=[0.0], colors='white', linewidths=1.6, zorder=4)
                shown = m[cavity] if cavity.any() else m
                if annotate_mean:
                    ax.annotate(f'{np.nanmean(shown):+.2f} V', (0.97, 0.03),
                                xycoords='axes fraction', ha='right', va='bottom',
                                fontsize=9, fontweight='bold', color='white',
                                bbox=dict(boxstyle='round,pad=0.25', fc='black',
                                          ec='none', alpha=0.55))
            if near is not None:
                ax.plot(near[:, 0], near[:, 2], '.', color='0.2', ms=4, alpha=0.6,
                        zorder=5)
            for i, p in enumerate(pts if show_points else []):
                if abs(p[1]) > 0.5:
                    continue
                ax.plot(p[0], p[2], 'o' if i < 2 else '^', ms=9, zorder=6,
                        color='gold' if i < 2 else 'limegreen',
                        markeredgecolor='black', markeredgewidth=1.5,
                        label=point_labels[i])
            if structure:
                _draw_structure(ax, result, structure_slab, structure_color,
                                skip=() if structure_guest else result.get('guest_atoms', ()))
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
    if v.strip().lower() == 'vdw':
        return 'vdw'
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
# ESP profile on a plane through a cage -- pyEF input file.
# Run:  pyef espprofile -c this_file.yaml     (or: python -m pyef.esp_profile -c this_file.yaml)
# Paths are relative to this file.  One frame -> that frame's map;
# several frames -> the average, each frame in its own plane.
# Only `charges` (or `molden`) is required: everything else below is the default
# or a common choice.  The run prints the metals, ligands, cage and guest it
# detected -- check them.

# --- frames: give EITHER charges OR molden -------------------------------------
charges: frames/*/ChargesCM5.txt        # one file, a list, or a glob (sorted f2 < f10)
# molden: frames/*/*.molden             # instead of charges: Multiwfn makes any missing
# xyz: frames/*/*.xyz                   #   Charges<charge_type>.txt (xyz default: molden stem + .xyz)
# charge_type: CM5
# multiwfn_path: /path/to/Multiwfn
# pointcharges: pointcharges.txt        # QM/MM charges: a bare file name is looked up next to
                                        #   each frame's charges; or a list / glob

# --- the plane --------------------------------------------------------------------
plane: auto                             # auto: P1, P2 = the two cage metals, P3 = first ligand's
                                        #   centroid.  Or [P1, P2, P3]: atom index (0-based) or
                                        #   group (2-34, 2-10+12); axis P1 -> P2, toward P3
cut: vertical                           # vertical: plane through P1, P2, P3 (contains the M-M axis)
                                        # horizontal: perpendicular to P1 -> P2 (equatorial slice)
# offset: 0.0                           # slide the plane along its normal (A), e.g. a horizontal
                                        #   cut 2 A above the midpoint
# origin: midpoint                      # default: midpoint for auto/horizontal, else centroid;
                                        #   or an atom index, or [x, y, z]

# --- what to sum (optional) ---------------------------------------------------------
# Default with a detected cage: Cage = cage, Cage_Env = cage + mm.  The guest is left
# out of the sources (and the drawn structure); add e.g. All: all to include it.
# Tokens: cage, guest, metals, qm, mm, all, qm:a-b, mm:a-b, mm:a-  (ranges inclusive, 0-based)
# conditions:
#   Cage: cage
#   Cage_Anion: cage,mm:0-275           # e.g. if the first 276 MM charges are the counterions
#   Cage_Anion_Solv: cage,mm

# --- grid and physics (optional) ----------------------------------------------------
extent: auto                            # fits the whole cage (+ extent_pad); or a half-extent (A):
                                        #   one value or [z_half, x_half]
extent_pad: 2.0
spacing: 0.1                            # A: one value or [dz, dx]
dielectric: 1
dielectric_scale: 1                     # MM charges scaled by 1/sqrt(dielectric_scale), as in pyEF
r_min: 0.5                              # drop source-probe pairs closer than this (A)

# --- masks (optional; off by default -- esp_range clips the spikes at the nuclei instead) ---
# mask_radius: 1.6                      # blank cells within this distance (A) of the cage atoms; or vdw
                                        #   (each atom's van der Waals radius x vdw_scale); or per group
                                        #   {'metals': 2.0, '2-133': 1.6}
# vdw_scale: 1.0
# vdw_radii: {Fe: 2.0}                  # add/override radii for elements not in the table
# mask_atoms: cage                      # default: the detected cage (a guest does not blank the cavity)
# hull_atoms: cage                      # blank cells outside the cage's convex hull
mask_frac: 0.3                          # blank a cell flagged in >= this fraction of frames

# --- output -----------------------------------------------------------------------------
output: esp_profile                     # writes esp_profile.npz (+ .svg/.png)
# name: mysystem                        # system name inside the npz (default: output stem)
plot: true
# point_labels: [M2, M1, L1]
# ylabel: distance along M-M axis (Å)
# xlabel: x⊥ (Å)
esp_range: cavity                       # colour limits from the cage cavity only, so the metals and
                                        #   walls saturate; or [vmin, vmax] in V; null: whole-map automatic
# fixed_scale: [-2, 9, viridis]         # or pin vmin, vmax and the colormap yourself
# atom_slab: 1.0                        # also draw atoms within this distance (A) of the plane
"""

CONFIG_KEYS = {
    'charges', 'molden', 'xyz', 'charge_type', 'multiwfn_path', 'pointcharges',
    'plane', 'origin', 'conditions', 'extent', 'spacing', 'dielectric',
    'dielectric_scale', 'r_min', 'mask_radius', 'mask_atoms', 'hull_atoms',
    'mask_frac', 'output', 'name', 'plot', 'point_labels', 'ylabel', 'xlabel',
    'fixed_scale', 'atom_slab', 'vdw_scale', 'vdw_radii', 'extent_pad',
    'extent_atoms', 'cut', 'offset', 'esp_range'}


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
    base = os.path.dirname(os.path.abspath(path))

    charges = (_resolve_files(cfg['charges'], base, 'charges') if 'charges' in cfg
               else _charges_from_molden(cfg, base))
    ptchg = _resolve_ptchg(cfg.get('pointcharges'), charges, base)
    n = len(charges)
    print(f'{n} frame{"s" if n > 1 else ""}: '
          + ('averaging, each frame in its own plane' if n > 1 else 'single-frame map'))

    origin = cfg.get('origin')
    if isinstance(origin, str) and origin.isdigit():
        origin = int(origin)
    res = compute_esp_profile(
        charges, cfg.get('plane', 'auto'), ptchg_files=ptchg,
        conditions=cfg.get('conditions'), origin=origin,
        cut=str(cfg.get('cut', 'vertical')).lower(), offset=cfg.get('offset', 0.0),
        extent=cfg.get('extent', 'auto'),
        spacing=cfg.get('spacing', 0.1), r_min=cfg.get('r_min', 0.5),
        dielectric=cfg.get('dielectric', 1.0),
        mm_scale=1.0 / np.sqrt(cfg.get('dielectric_scale', 1.0)),
        mask_radius=cfg.get('mask_radius') or None,
        mask_atoms=cfg.get('mask_atoms'), hull_atoms=cfg.get('hull_atoms'),
        mask_frac=cfg.get('mask_frac', 0.3), vdw_scale=cfg.get('vdw_scale', 1.0),
        vdw_radii=cfg.get('vdw_radii'), extent_pad=cfg.get('extent_pad', 2.0),
        extent_atoms=cfg.get('extent_atoms'))

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
        if 'esp_range' in cfg:
            kw['esp_range'] = cfg['esp_range']
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
    p.add_argument('--plane', nargs='+', default=['auto'], metavar='P',
                   help='auto (default: the two cage metals + first ligand), or P1 P2 P3 as '
                        'QM atom index or group (2-34, 2-10+12, cage); axis P1->P2, toward P3')
    p.add_argument('--cut', choices=['vertical', 'horizontal'], default='vertical',
                   help='vertical: plane through P1, P2, P3; horizontal: perpendicular to '
                        'P1->P2 (default vertical)')
    p.add_argument('--offset', type=float, default=0.0,
                   help='slide the plane along its normal, A (default 0)')
    p.add_argument('--origin', nargs='+', default=None,
                   help='centroid, midpoint, an atom index, or x y z (default: midpoint for '
                        'auto/horizontal, else centroid)')
    p.add_argument('--condition', action='append', default=None,
                   help="NAME=TOKENS, e.g. CageOnly=qm:0-133 or 'Cage_Anion=qm:0-133,mm:0-275'; "
                        'repeatable. Default: qm (and all with --ptchg)')
    p.add_argument('--extent', nargs='+', default=['auto'],
                   help="half-extent in A: one value, Z_HALF X_HALF, or 'auto' to fit the "
                        'whole cage (default auto)')
    p.add_argument('--extent-pad', type=float, default=2.0,
                   help='A beyond the outermost atom for --extent auto (default 2)')
    p.add_argument('--spacing', nargs='+', type=float, default=[0.1],
                   help='grid spacing in A: one value or DZ DX (default 0.1)')
    p.add_argument('--dielectric', type=float, default=1.0)
    p.add_argument('--mm-scale', type=float, default=1.0,
                   help='factor on every MM charge (default 1)')
    p.add_argument('--r-min', type=float, default=0.5,
                   help='drop source-probe pairs closer than this, A (default 0.5)')
    p.add_argument('--mask-radius', type=_mask_radius_arg, default=None,
                   help="mask cells within this distance of mask atoms, A (default: no mask), "
                        "'vdw' for each atom's van der Waals radius, or per-group radii "
                        'ATOMS:R,..., e.g. 0-1:2.0,2-133:1.6')
    p.add_argument('--vdw-scale', type=float, default=1.0,
                   help='factor on the van der Waals radii for --mask-radius vdw')
    p.add_argument('--mask-atoms', default=None,
                   help='QM atoms for the radius mask (default: the detected cage, else all)')
    p.add_argument('--hull', default=None,
                   help='mask cells outside the convex hull of these QM atoms')
    p.add_argument('--mask-frac', type=float, default=0.3)
    p.add_argument('--name', default='profile', help='system name in the npz')
    p.add_argument('--out', default='esp_profile',
                   help='output stem: writes STEM.npz and STEM.svg/.png')
    p.add_argument('--esp-range', nargs='+', default=['cavity'],
                   help="colour scale: 'cavity' (default, limits from the cage cavity), "
                        "'auto', or VMIN VMAX in V; values outside saturate")
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
    if not a.charges:
        parser.error('give -c CONFIG.yaml, or --charges')
    if a.plane == ['auto']:
        plane = 'auto'
    elif len(a.plane) == 3:
        plane = a.plane
    else:
        parser.error('--plane takes auto or three points P1 P2 P3')
    for opt, v in (('--extent', a.extent), ('--spacing', a.spacing)):
        if len(v) not in (1, 2):
            raise SystemExit(f'error: {opt} takes one or two values')
    extent = 'auto' if a.extent == ['auto'] else [float(v) for v in a.extent]
    res = compute_esp_profile(
        a.charges, plane, ptchg_files=_pair_ptchg(a.charges, a.ptchg),
        conditions=a.condition, cut=a.cut, offset=a.offset,
        origin=None if a.origin is None else _origin_arg(a.origin),
        extent=extent, spacing=a.spacing, r_min=a.r_min,
        dielectric=a.dielectric, mm_scale=a.mm_scale,
        mask_radius=a.mask_radius or None, mask_atoms=a.mask_atoms,
        hull_atoms=a.hull, mask_frac=a.mask_frac, vdw_scale=a.vdw_scale,
        extent_pad=a.extent_pad)
    save_esp_profile(f'{a.out}.npz', res, name=a.name)
    print(f'{res["n_frames"]} frames -> {a.out}.npz ({a.name})')
    for c in res['conditions']:
        print(f'  {c:20s} mean over unmasked cells {np.nanmean(masked(res, c)):+8.3f} V')
    if not a.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        rng = a.esp_range
        if len(rng) == 2:
            rng = [float(v) for v in rng]
        elif rng == ['auto']:
            rng = None
        elif rng != ['cavity']:
            parser.error("--esp-range takes 'cavity', 'auto' or VMIN VMAX")
        else:
            rng = 'cavity'
        plot_esp_profile(res, out=a.out, esp_range=rng)
        print(f'plot -> {a.out}.svg, {a.out}.png')
    return res


if __name__ == '__main__':
    main()
