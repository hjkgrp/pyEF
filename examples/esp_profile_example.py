"""ESP map on a 2D plane defined by three atoms, from an Electrostatics object.

Same pattern as getESP / getEfield: make the object from the molden + xyz
files, then call getESPProfile.  It writes the map as .svg and/or .png
(plus an .npz with the numbers).  Every job is treated as one frame of the
same system and the maps are averaged; for a single structure pass one file.
Missing Charges<type>.txt files are generated with Multiwfn.

    python esp_profile_example.py
"""
from pyef.analysis import Electrostatics
from pyef import esp_profile as ep

molden_paths = ['/path/to/frame1/final.molden', '/path/to/frame2/final.molden']
xyz_paths = ['/path/to/frame1/final.xyz', '/path/to/frame2/final.xyz']
multiwfn_install = '/path/to/Multiwfn'

# create an electrostatics object associated with the file structure
es = Electrostatics(molden_paths, xyz_paths, dielectric=1.0)

# optional, QM/MM: point-charge file in each job's folder
# es.includePtChgs('pointcharges.txt')

# The plane: three atoms (0-indexed).  Vertical axis P1 -> P2, horizontal axis
# toward P3; a range ('2-41') stands for the centroid of those atoms (e.g. a
# linker).  'auto' picks the two cage metals and the first ligand.
plane = (1, 0, '2-41')

# What to sum.  Default (conditions=None): Cage, plus Cage_Env with point
# charges -- the guest is left out of the sources and the drawing.  Tokens:
# cage, guest, metals (detected automatically), qm, mm, all, qm:a-b, mm:a-
conditions = {'Cage': 'cage',
              'Cage_Env': 'cage,mm'}     # add 'All': 'all' to include the guest

fn = 'esp_profile'
res = es.getESPProfile(charge_types=['Hirshfeld_I'], profile_filename=fn,
                       multiwfn_path=multiwfn_install,
                       plane_points=plane, conditions=conditions,
                       origin='midpoint',      # 'centroid', 'midpoint' or an atom index
                       formats=('svg', 'png'),  # or just 'png'
                       esp_range='cavity')      # default: colour limits from the cavity, so the
                                                # metals and walls saturate; or e.g. (-10, 10) V
# Defaults: nothing masked, the structure in the plane drawn as white sticks.
# To blank cells near atoms or outside the cage add e.g. mask_radius=1.6
# and/or hull_atoms='cage'; plot_kw={'structure': False} drops the sticks.

# To map the other linker, point P3 at its atoms (placeholder range below) and change the filename
# res2 = es.getESPProfile(charge_types=['Hirshfeld_I'], profile_filename='esp_profile_linker2',
#                         multiwfn_path=multiwfn_install, plane_points=(1, 0, '42-81'),
#                         conditions=conditions, origin='midpoint')

print(f'{res["n_frames"]} frame(s) -> {fn}.svg, {fn}.png, {fn}.npz')
for c in res['conditions']:
    print(f'  {c:10s} cavity mean (|z|, |x| <= 4 A): {ep.region_mean(res, c):+.3f} V')
