"""Unit tests for three-atom-plane ESP profiles (pyef.esp_profile)"""

import os

import numpy as np
import pytest

from pyef import esp_profile as ep
from pyef.esp_profile import COULOMB_EV_ANGSTROM as K

FIXTURE = os.path.join(os.path.dirname(__file__), 'fixtures', 'sample_1',
                       'ChargesHirshfeld.txt')


@pytest.fixture
def rng():
    return np.random.default_rng(1)


def write_charges(path, syms, xyz, q):
    with open(path, 'w') as fh:
        for s, (x, y, z), c in zip(syms, xyz, q):
            fh.write(f'{s} {x:.17g} {y:.17g} {z:.17g} {c:.17g}\n')
    return str(path)


def write_ptchg(path, xyz, q):
    with open(path, 'w') as fh:
        fh.write(f'{len(q)}\n\n')
        for (x, y, z), c in zip(xyz, q):
            fh.write(f'{c:.17g} {x:.17g} {y:.17g} {z:.17g}\n')
    return str(path)


# --- physics --------------------------------------------------------------------

def test_single_charge_matches_analytic():
    # plane z = 0 spanned by three atoms; one +1 charge 2 A above the origin
    frame = ep.plane_basis([0, -1.0, 0], [0, 1.0, 0], [1.0, 0, 0],
                           origin=[0, 0, 0])
    z, x, esp = ep.esp_on_grid(frame, [[0, 0, 2.0]], [1.0], extent=3.0,
                               spacing=0.5, dielectric=2.0)
    lab = frame.to_lab(ep.plane_grid(3.0, 3.0, len(z), len(x))[2])
    r = np.linalg.norm(lab - [0, 0, 2.0], axis=1)
    np.testing.assert_allclose(esp.ravel(), K / 2.0 / r, rtol=1e-12)
    assert esp[len(z) // 2, len(x) // 2] == pytest.approx(K / 4.0)


def test_r_min_drops_close_pairs_without_nan():
    frame = ep.plane_basis([0, 0, -1.0], [0, 0, 1.0], [1.0, 0, 0], origin=[0, 0, 0])
    _, _, esp = ep.esp_on_grid(frame, [[0, 0, 0]], [1.0], extent=1.0, spacing=0.1,
                               r_min=0.5)
    assert np.all(np.isfinite(esp))
    assert esp[10, 10] == 0.0                     # probe on the charge
    assert esp[10, 15] == pytest.approx(K / 0.5)  # exactly r_min is kept


# --- plane basis ------------------------------------------------------------------

def test_basis_orthonormal_and_points_toward_p3(rng):
    for _ in range(20):
        p1, p2, p3 = rng.normal(scale=5, size=(3, 3))
        f = ep.plane_basis(p1, p2, p3)
        R = f.rotation
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(f.z_hat, (p2 - p1) / np.linalg.norm(p2 - p1))
        loc = f.to_local(np.array([p1, p2, p3]))
        np.testing.assert_allclose(loc[:, 1], 0.0, atol=1e-10)   # all in plane
        assert loc[2, 0] > loc[0, 0]                             # P3 at +x
        np.testing.assert_allclose(f.origin, (p1 + p2 + p3) / 3, atol=1e-12)
        assert f.axis_length == pytest.approx(np.linalg.norm(p2 - p1))


def test_basis_collinear_raises():
    with pytest.raises(ValueError):
        ep.plane_basis([0, 0, 0], [0, 0, 1.0], [0, 0, 2.0])


def test_origin_options_and_projection(rng):
    xyz = rng.normal(scale=4, size=(12, 3))
    plane = (1, 0, '2-6')
    p1, p2, p3 = xyz[1], xyz[0], xyz[2:7].mean(axis=0)
    f, pts = ep.plane_frame(xyz, plane, 'midpoint')
    np.testing.assert_allclose(pts, [p1, p2, p3])
    np.testing.assert_allclose(f.origin, (p1 + p2) / 2, atol=1e-12)
    # an off-plane atom origin is projected into the plane
    f, _ = ep.plane_frame(xyz, plane, 9)
    assert abs(f.to_local(xyz[9])[1]) == pytest.approx(
        abs(np.dot(xyz[9] - p1, f.y_hat)))
    assert abs(np.dot(f.origin - p1, f.y_hat)) < 1e-10
    f, _ = ep.plane_frame(xyz, plane, [1.0, 2.0, 3.0])
    assert abs(np.dot(f.origin - p1, f.y_hat)) < 1e-10
    with pytest.raises(ValueError):
        ep.plane_frame(xyz, plane, 'nowhere')
    with pytest.raises(IndexError):
        ep.plane_frame(xyz, (0, 1, 40), 'centroid')


def test_paper_cage_frame(rng):
    # P1 = Pd2 (1), P2 = Pd1 (0), P3 = LA1 centroid, origin = midpoint must give
    # the cage frame of the published maps: z = Pd1 - Pd2, x toward LA1 (its
    # perpendicular component), y = z x x
    xyz = rng.normal(scale=4, size=(40, 3))
    f, _ = ep.plane_frame(xyz, (1, 0, '2-34'), 'midpoint')
    origin = 0.5 * (xyz[0] + xyz[1])
    z_hat = (xyz[0] - xyz[1]) / np.linalg.norm(xyz[0] - xyz[1])
    v = xyz[2:35].mean(axis=0) - origin
    v -= np.dot(v, z_hat) * z_hat
    x_hat = v / np.linalg.norm(v)
    R = np.column_stack([x_hat, np.cross(z_hat, x_hat), z_hat])
    np.testing.assert_allclose(f.rotation, R, atol=1e-12)
    np.testing.assert_allclose(f.origin, origin, atol=1e-12)


# --- tokens -----------------------------------------------------------------------

def test_condition_tokens():
    assert ep.parse_condition('qm') == [('qm', 0, None)]
    assert ep.parse_condition('all') == [('qm', 0, None), ('mm', 0, None)]
    assert ep.parse_condition('qm:0-133, mm:0-275') == [('qm', 0, 133), ('mm', 0, 275)]
    assert ep.parse_condition(['QM:0-133', 'mm:276-']) == [('qm', 0, 133), ('mm', 276, None)]
    assert ep.parse_condition('mm:5') == [('mm', 5, 5)]
    for bad in ('qm:0-133,qm:100-', 'all,mm', 'xx', 'qm:5-2', '', 'mm:0-10,mm:10'):
        with pytest.raises(ValueError):
            ep.parse_condition(bad)
    assert ep.parse_index_spec('2-4+7') == [2, 3, 4, 7]
    assert ep.parse_index_spec([1, '3-4']) == [1, 3, 4]
    with pytest.raises(ValueError):
        ep.parse_index_spec('2-a')


def test_segments_are_disjoint_and_cover_conditions():
    conds = {'CageOnly': 'qm:0-133', 'Cage_Anion': 'qm:0-133,mm:0-275',
             'Cage_Anion_Solv': 'qm:0-133,mm', 'All': 'all'}
    segs, sc = ep._segment_conditions(conds)
    assert sc['CageOnly'] == ('qm:0-133',)
    assert sc['Cage_Anion_Solv'] == ('qm:0-133', 'mm:0-275', 'mm:276-')
    assert sc['All'] == ('qm:0-133', 'qm:134-', 'mm:0-275', 'mm:276-')
    assert segs['mm:276-'] == ('mm', 276, None)
    assert segs['qm:0-133'] == ('qm', 0, 134)


# --- end to end ---------------------------------------------------------------------

def _toy_frame(rng, n_qm=20, n_mm=30):
    syms = ['Pd', 'Pd'] + ['C'] * (n_qm - 2)
    xyz = np.vstack([[0, 0, 3.0], [0, 0, -3.0], rng.normal(scale=4, size=(n_qm - 2, 3))])
    xyz[2:6] += [5.0, 0, 0]
    q = rng.normal(size=n_qm)
    mm_xyz = rng.normal(scale=8, size=(n_mm, 3))
    mm_q = rng.normal(size=n_mm)
    return syms, xyz, q, mm_xyz, mm_q


def test_conditions_are_sums_of_their_sources(tmp_path, rng):
    syms, xyz, q, mm_xyz, mm_q = _toy_frame(rng)
    cf = write_charges(tmp_path / 'c.txt', syms, xyz, q)
    pf = write_ptchg(tmp_path / 'p.txt', mm_xyz, mm_q)
    conds = {'cage': 'qm:0-9', 'cage_anion': 'qm:0-9,mm:0-4', 'every': 'all'}
    res = ep.compute_esp_profile([cf], (1, 0, '2-5'), [pf], conds,
                                 origin='midpoint', extent=(4, 3), spacing=0.5,
                                 dielectric=4.0, mm_scale=0.5, mask_radius=None)
    f, _ = ep.plane_frame(xyz, (1, 0, '2-5'), 'midpoint')
    probes = f.to_lab(ep.plane_grid(4, 3, 17, 13)[2])
    def ref(sx, sq):
        return ep.esp_at(probes, sx, sq, k=K / 4.0, core='clamp').reshape(17, 13)
    np.testing.assert_allclose(res['mean_cage'], ref(xyz[:10], q[:10]), rtol=1e-10)
    np.testing.assert_allclose(
        res['mean_cage_anion'],
        ref(np.vstack([xyz[:10], mm_xyz[:5]]), np.concatenate([q[:10], 0.5 * mm_q[:5]])),
        rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        res['mean_every'],
        ref(np.vstack([xyz, mm_xyz]), np.concatenate([q, 0.5 * mm_q])),
        rtol=1e-10, atol=1e-12)
    assert not res['mask'].any()
    with pytest.raises(IndexError):
        ep.compute_esp_profile([cf], (1, 0, 2), [pf], {'x': 'mm:0-99'})


def test_rigid_motion_leaves_map_unchanged(tmp_path, rng):
    from scipy.spatial.transform import Rotation
    syms, xyz, q, mm_xyz, mm_q = _toy_frame(rng)
    R = Rotation.random(random_state=3).as_matrix()
    t = np.array([10.0, -4.0, 7.5])
    files = []
    for name, f in (('a', lambda p: p), ('b', lambda p: p @ R.T + t)):
        files.append((write_charges(tmp_path / f'{name}c.txt', syms, f(xyz), q),
                      write_ptchg(tmp_path / f'{name}p.txt', f(mm_xyz), mm_q)))
    kw = dict(conditions={'cage': 'qm', 'all': 'all'}, origin=7, extent=(5, 4),
              spacing=0.25, hull_atoms='0-19')
    a = ep.compute_esp_profile([files[0][0]], (1, 0, '2-5'), [files[0][1]], **kw)
    b = ep.compute_esp_profile([files[1][0]], (1, 0, '2-5'), [files[1][1]], **kw)
    for c in ('cage', 'all'):
        np.testing.assert_allclose(a[f'mean_{c}'], b[f'mean_{c}'], rtol=1e-8, atol=1e-8)
    np.testing.assert_array_equal(a['mask'], b['mask'])
    np.testing.assert_allclose(a['atoms_local'], b['atoms_local'], atol=1e-9)


def test_frames_are_averaged_in_their_own_plane(tmp_path, rng):
    from scipy.spatial.transform import Rotation
    syms, xyz, q, _, _ = _toy_frame(rng)
    R = Rotation.random(random_state=5).as_matrix()
    one = write_charges(tmp_path / 'one.txt', syms, xyz, q)
    two = write_charges(tmp_path / 'two.txt', syms, xyz @ R.T, 2 * q)
    res = ep.compute_esp_profile([one, two], (1, 0, '2-5'), extent=3, spacing=0.5,
                                 mask_radius=1.0, conditions='qm')
    single = ep.compute_esp_profile([one], (1, 0, '2-5'), extent=3, spacing=0.5,
                                    conditions='qm')
    np.testing.assert_allclose(res['mean_qm'], 1.5 * single['mean_qm'], rtol=1e-8)
    np.testing.assert_allclose(res['std_qm'], 0.5 * np.abs(single['mean_qm']),
                               rtol=1e-8, atol=1e-10)
    assert res['n_frames'] == 2
    assert res['mask'].any()


def test_fixture_smoke_and_roundtrip(tmp_path):
    mpl = pytest.importorskip('matplotlib')
    mpl.use('Agg')
    syms, xyz, q = ep.read_charges(FIXTURE)
    assert len(syms) == 25
    res = ep.compute_esp_profile([FIXTURE], (0, 1, '2-5'), extent=4.0, spacing=0.2,
                                 mask_radius=0.8, hull_atoms='0-24')
    m = ep.masked(res, 'qm')
    assert m.shape == (41, 41)
    assert np.isfinite(m).any() and np.isnan(m).any()
    np.testing.assert_allclose(res['plane_points_local'][:, 1], 0.0, atol=1e-10)

    p = tmp_path / 'prof.npz'
    ep.save_esp_profile(p, res, name='sample_1')
    back = ep.load_esp_profile(p)
    np.testing.assert_array_equal(back['mean_qm'], res['mean_qm'])
    np.testing.assert_array_equal(back['mask'], res['mask'])
    assert back['conditions'] == ['qm']
    with np.load(p) as d:                       # ESP bundle key layout
        assert {'sample_1_mean_qm', 'sample_1_std_qm', 'sample_1_x_vals',
                'sample_1_z_vals', 'sample_1_mask', 'sample_1_atoms_local'} <= set(d.files)

    ep.plot_esp_profile(back, out=str(tmp_path / 'prof'), exts=('png',), atom_slab=1.0)
    assert (tmp_path / 'prof.png').stat().st_size > 0


def test_command_line(tmp_path, rng):
    pytest.importorskip('matplotlib')
    syms, xyz, q, mm_xyz, mm_q = _toy_frame(rng)
    d = tmp_path / 'f0'
    d.mkdir()
    cf = write_charges(d / 'ChargesCM5.txt', syms, xyz, q)
    pf = write_ptchg(d / 'pointcharges.txt', mm_xyz, mm_q)
    out = str(tmp_path / 'cli')
    res = ep.main(['--charges', cf, '--ptchg', pf, '--plane', '1', '0', '2-5',
                   '--origin', 'midpoint', '--condition', 'Cage=qm:0-9',
                   '--condition', 'Cage_MM=qm:0-9,mm', '--extent', '3', '2',
                   '--spacing', '0.5', '--name', 'toy', '--out', out])
    assert res['mean_Cage'].shape == (13, 9)
    assert os.path.exists(out + '.npz') and os.path.exists(out + '.png')
    assert ep.load_esp_profile(out + '.npz', 'toy')['conditions'] == ['Cage', 'Cage_MM']


# --- job batch file -------------------------------------------------------------------

@pytest.mark.parametrize('module', ['pyef.manage', 'pyef.cli'])
def test_job_file_three_tuple(tmp_path, module):
    import importlib
    parse = importlib.import_module(module).parse_job_batch_file
    jobs = tmp_path / 'jobs.csv'
    jobs.write_text('espprofile, a.molden, a.xyz, (1, 0, 2-34)\n'
                    'ef, b.molden, b.xyz, (25, 26), (25, 27)\n'
                    'espprofile, c.molden, c.xyz, (1, 0, 2-10+12)\n')
    at, _, _, metals, bonds, planes = parse(str(jobs), return_plane_points=True)
    assert at == ['espprofile', 'ef', 'espprofile']
    assert planes == [(1, 0, '2-34'), None, (1, 0, '2-10+12')]
    assert bonds == [[], [(25, 26), (25, 27)], []]
    assert metals == [None, 25, None]
    assert len(parse(str(jobs))) == 5          # old 5-tuple return is unchanged
    jobs.write_text('ef, a.molden, a.xyz, (1, 0, 2, 3)\n')
    with pytest.raises(ValueError):
        parse(str(jobs))


# --- one-file input (YAML config) ------------------------------------------------------

def _frame_dirs(tmp_path, rng, n):
    for i in range(n):
        syms, xyz, q, mm_xyz, mm_q = _toy_frame(rng)
        d = tmp_path / 'frames' / f'f{i * 5}'
        d.mkdir(parents=True)
        write_charges(d / 'ChargesCM5.txt', syms, xyz, q)
        write_ptchg(d / 'pointcharges.txt', mm_xyz, mm_q)


def test_config_single_frame_and_average(tmp_path, rng):
    pytest.importorskip('matplotlib')
    _frame_dirs(tmp_path, rng, 3)
    common = ('pointcharges: pointcharges.txt\nplane: [1, 0, 2-5]\norigin: midpoint\n'
              'conditions: {Cage: "qm:0-9", Total: all}\nextent: 3\nspacing: 0.5\n'
              'mask_radius: {"0-1": 1.0, "2-19": 0.5}\n')
    one = tmp_path / 'one.yaml'
    one.write_text('charges: frames/f5/ChargesCM5.txt\noutput: single\n' + common)
    many = tmp_path / 'many.yaml'
    many.write_text('charges: frames/*/ChargesCM5.txt\noutput: avg\nplot: false\n' + common)
    r1 = ep.run_config(str(one))
    rn = ep.run_config(str(many))
    assert r1['n_frames'] == 1 and np.all(r1['std_Cage'] == 0)
    assert rn['n_frames'] == 3
    # natural sort: f0, f5, f10
    assert [os.path.basename(os.path.dirname(f)) for f in rn['frames']] == ['f0', 'f5', 'f10']
    assert (tmp_path / 'single.png').exists() and not (tmp_path / 'avg.png').exists()
    # the average is the mean of the single-frame maps
    singles = []
    for f in ('f0', 'f5', 'f10'):
        c = tmp_path / f'{f}.yaml'
        c.write_text(f'charges: frames/{f}/ChargesCM5.txt\noutput: {f}\nplot: false\n' + common)
        singles.append(ep.run_config(str(c))['mean_Total'])
    np.testing.assert_allclose(rn['mean_Total'], np.mean(singles, axis=0), rtol=1e-10)
    np.testing.assert_allclose(r1['mean_Cage'], ep.load_esp_profile(tmp_path / 'single.npz')['mean_Cage'])


def test_config_errors(tmp_path, rng):
    _frame_dirs(tmp_path, rng, 1)
    bad = tmp_path / 'bad.yaml'
    bad.write_text('charges: frames/*/ChargesCM5.txt\nplane: [1, 0, 2]\nplnae: 1\n')
    with pytest.raises(ValueError, match='unknown key'):
        ep.run_config(str(bad))
    bad.write_text('charges: nothing/*/ChargesCM5.txt\nplane: [1, 0, 2]\n')
    with pytest.raises(FileNotFoundError):
        ep.run_config(str(bad))
    bad.write_text('charges: frames/*/ChargesCM5.txt\npointcharges: nope.txt\nplane: [1, 0, 2]\n')
    with pytest.raises(FileNotFoundError):
        ep.run_config(str(bad))


def test_template_is_a_valid_config(tmp_path, rng, capsys):
    pytest.importorskip('matplotlib')
    import yaml
    ep.main(['--template'])
    cfg = yaml.safe_load(capsys.readouterr().out)
    assert set(cfg) <= ep.CONFIG_KEYS
    # point the template at toy frames; the toy system has 20 QM / 30 MM charges
    _frame_dirs(tmp_path, rng, 2)
    cfg.update(conditions={'CageOnly': 'qm:0-9', 'Cage_Anion': 'qm:0-9,mm:0-4'},
               extent=3, spacing=0.5, plane=[1, 0, '2-5'], output='tmpl',
               pointcharges='pointcharges.txt', hull_atoms='all')
    (tmp_path / 'esp.yaml').write_text(yaml.safe_dump(cfg))
    res = ep.main(['-c', str(tmp_path / 'esp.yaml')])
    assert res['n_frames'] == 2 and (tmp_path / 'tmpl.npz').exists()


# --- new-cage defaults: auto extent, vdW masks, 'all' -----------------------------------

def test_auto_extent_holds_every_atom(tmp_path, rng):
    _frame_dirs(tmp_path, rng, 3)
    files = sorted(str(p) for p in tmp_path.glob('frames/*/ChargesCM5.txt'))
    res = ep.compute_esp_profile(files, (1, 0, '2-5'), extent='auto', extent_pad=1.5,
                                 extent_atoms='all', spacing=0.25, origin='midpoint',
                                 mask_radius=None)
    zh, xh = res['extent']
    assert zh == res['z_vals'][-1] and xh == res['x_vals'][-1]
    for f in files:
        _, xyz, _ = ep.read_charges(f)
        frame, _ = ep.plane_frame(xyz, (1, 0, '2-5'), 'midpoint')
        loc = frame.to_local(xyz)
        assert np.abs(loc[:, 2]).max() + 1.5 <= zh + 1e-9
        assert np.abs(loc[:, 0]).max() + 1.5 <= xh + 1e-9
    # tight: shrinking by one grid step would cut an atom's pad
    assert zh - 0.25 < max(np.abs(ep.plane_frame(ep.read_charges(f)[1], (1, 0, '2-5'),
                                                  'midpoint')[0].to_local(
                                                      ep.read_charges(f)[1])[:, 2]).max()
                           for f in files) + 1.5


def test_vdw_mask_uses_element_radii():
    syms = ['Pd', 'C', 'H', 'N']
    r = ep.vdw_radii_for(syms, scale=1.2)
    np.testing.assert_allclose(r, 1.2 * np.array([1.63, 1.70, 1.20, 1.55]))
    with pytest.warns(UserWarning, match='Fe'):
        assert ep.vdw_radii_for(['Fe'])[0] == ep.DEFAULT_VDW_RADIUS
    assert ep.vdw_radii_for(['FE'], overrides={'Fe': 2.3})[0] == 2.3
    sets = ep._mask_sets('vdw', 'all', syms)
    assert sorted(((tuple(i), r) for i, r in sets), key=lambda t: t[1]) == [((2,), 1.2), ((3,), 1.55),
                                                     ((0,), 1.63), ((1,), 1.7)]


def test_vdw_mask_on_grid(tmp_path):
    # one Pd and one H on the plane; masked cells are exactly those inside a vdW sphere
    syms = ['Pd', 'H', 'C']
    xyz = np.array([[0, 0, 2.0], [0, 0, -2.0], [3.0, 0, 0]])
    cf = write_charges(tmp_path / 'c.txt', syms, xyz, [1.0, 0.5, 0.0])
    res = ep.compute_esp_profile([cf], (1, 0, 2), mask_radius='vdw', mask_atoms='0-1',
                                 extent=4, spacing=0.1, origin='midpoint')
    ZZ, XX = np.meshgrid(res['z_vals'], res['x_vals'], indexing='ij')
    inside = (np.hypot(ZZ - 2, XX) < 1.63) | (np.hypot(ZZ + 2, XX) < 1.20)
    np.testing.assert_array_equal(res['mask'], inside)


def test_hull_all_equals_explicit_range(tmp_path, rng):
    _frame_dirs(tmp_path, rng, 1)
    f = [str(next(tmp_path.glob('frames/*/ChargesCM5.txt')))]
    kw = dict(extent=4, spacing=0.5, mask_radius=None)
    a = ep.compute_esp_profile(f, (1, 0, 2), hull_atoms='all', **kw)
    b = ep.compute_esp_profile(f, (1, 0, 2), hull_atoms='0-19', **kw)
    np.testing.assert_array_equal(a['mask'], b['mask'])
    assert a['mask'].any()


# --- automatic system detection and cuts ---------------------------------------------------

def _mini_cage():
    """Two Pd on z, two N-C-C-C-N ligands bridging them (at +y and +x), and a
    two-atom guest at the equator bonded to nothing."""
    syms, xyz = ['Pd', 'Pd'], [[0, 0, 3.0], [0, 0, -3.0]]
    for d in ([0, 2.0], [2.0, 0]):
        for el, z in zip('NCCCN', (3.0, 1.5, 0.0, -1.5, -3.0)):
            syms.append(el)
            xyz.append([d[0], d[1], z])
    syms += ['C', 'C']
    xyz += [[-1.2, -1.2, 0.0], [-2.2, -1.2, 0.0]]
    return syms, np.array(xyz)


def test_detect_system_finds_cage_and_guest():
    syms, xyz = _mini_cage()
    sysd = ep.detect_system(syms, xyz)
    assert sysd['metals'] == [0, 1]
    assert sysd['ligands'] == [list(range(2, 7)), list(range(7, 12))]
    assert sysd['cage'] == list(range(12)) and sysd['guest'] == [12, 13]
    assert ep.index_spec_str([0, 1, 2, 5, 7, 8]) == '0-2+5+7-8'


def test_auto_plane_vertical_and_horizontal(tmp_path):
    syms, xyz = _mini_cage()
    cf = write_charges(tmp_path / 'c.txt', syms, xyz, np.linspace(-1, 1, len(syms)))
    v = ep.compute_esp_profile([cf], spacing=0.5, mask_radius=None)
    assert list(v['plane_spec']) == ['1', '0', '2-6']          # Pd1 -> Pd0, ligand 1
    assert v['conditions'] == ['Cage']                           # guest present: left out by default
    np.testing.assert_allclose(v['plane_points_local'],
                               [[0, 0, -3], [0, 0, 3], [2, 0, 0]], atol=1e-12)
    h = ep.compute_esp_profile([cf], cut='horizontal', spacing=0.5, mask_radius=None)
    # horizontal: Pd-Pd axis is the normal, through the centre; ligand 1 toward +x
    np.testing.assert_allclose(h['plane_points_local'],
                               [[0, -3, 0], [0, 3, 0], [2, 0, 0]], atol=1e-12)
    f, _ = ep.plane_frame(xyz, (1, 0, '2-6'), 'midpoint', cut='horizontal', offset=1.0)
    R = f.rotation
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(f.y_hat, [0, 0, 1], atol=1e-12)   # normal = axis P1 -> P2
    np.testing.assert_allclose(f.x_hat, [0, 1, 0], atol=1e-12)   # toward ligand 1
    np.testing.assert_allclose(f.origin, [0, 0, 1.0], atol=1e-12)  # offset along normal
    # horizontal-cut ESP is the plain sum over the in-plane probe points
    probes = f.to_lab(ep.plane_grid(2, 2, 5, 5)[2])
    _, _, m = ep.esp_on_grid(f, xyz, np.linspace(-1, 1, len(syms)), extent=2, spacing=1.0)
    np.testing.assert_allclose(m.ravel(), ep.esp_at(probes, xyz, np.linspace(-1, 1, len(syms))))


def test_named_tokens_and_default_masks(tmp_path):
    syms, xyz = _mini_cage()
    q = np.linspace(-1, 1, len(syms))
    cf = write_charges(tmp_path / 'c.txt', syms, xyz, q)
    r = ep.compute_esp_profile([cf], conditions={'C': 'cage', 'G': 'guest', 'A': 'all',
                                                 'M': 'metals'},
                               extent=3, spacing=0.5, mask_radius=0.9)
    np.testing.assert_allclose(r['mean_C'] + r['mean_G'], r['mean_A'], atol=1e-12)
    # the default radius mask covers the cage only: in the equatorial cut the
    # guest atom lies in the plane and does not blank its cell; mask_atoms='all' does
    kw = dict(cut='horizontal', extent=3, spacing=0.1, mask_radius=0.9)
    default = ep.compute_esp_profile([cf], **kw)
    every = ep.compute_esp_profile([cf], mask_atoms='all', **kw)
    g = ep.plane_frame(xyz, (1, 0, '2-6'), 'midpoint', cut='horizontal')[0].to_local(xyz[12])
    assert abs(g[1]) < 1e-12
    cell = (np.argmin(np.abs(default['z_vals'] - g[2])), np.argmin(np.abs(default['x_vals'] - g[0])))
    assert not default['mask'][cell] and every['mask'][cell]
    with pytest.raises(ValueError, match='plane auto'):
        ep.compute_esp_profile([write_charges(tmp_path / 'n.txt', ['C', 'C', 'C'],
                                              np.eye(3) * 3, [0.1, 0.2, -0.3])])
