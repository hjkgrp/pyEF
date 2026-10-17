"""ESP map on a 2D plane defined by three atoms -- everything in one script.

Needs, for every frame, the charge file pyEF writes (Charges<type>.txt) and,
optionally, the QM/MM point-charge file in the same folder.  One frame gives
that frame's map; several frames are averaged, each in its own plane.

    python esp_profile_example.py                       # uses the settings below
    python esp_profile_example.py 'other/*/ChargesCM5.txt'
"""
import glob
import os
import sys

from pyef import esp_profile as ep

# --- 1. frames --------------------------------------------------------------------
# one file, or a pattern matching one file per frame
CHARGES = sys.argv[1] if len(sys.argv) > 1 else 'frames/*/ChargesCM5.txt'
PTCHG_NAME = 'pointcharges.txt'      # QM/MM charges next to each charge file; None if none

# --- 2. the plane: three atoms (0-based rows of the charge file) -------------------
# vertical axis P1 -> P2, horizontal axis toward P3; a range ('2-41') means
# the centroid of those atoms
PLANE = (1, 0, '2-41')
ORIGIN = 'midpoint'                  # 'centroid' (of P1, P2, P3), 'midpoint' (of P1, P2) or an atom index

# --- 3. what to sum -------------------------------------------------------------------
# tokens: cage, guest, metals (detected automatically), qm, mm, all, qm:a-b, mm:a-
CONDITIONS = {'Cage': 'cage',
              'Cage_Env': 'cage,mm',
              'All': 'all'}

OUTPUT = 'esp_profile'               # writes esp_profile.png / .svg (+ .npz with the numbers)

# --- run ------------------------------------------------------------------------------
charge_files = sorted(glob.glob(os.path.expanduser(CHARGES)))
if not charge_files:
    sys.exit(f'no charge files match {CHARGES}')
ptchg_files = None
if PTCHG_NAME:
    ptchg_files = [os.path.join(os.path.dirname(f), PTCHG_NAME) for f in charge_files]

res = ep.compute_esp_profile(
    charge_files,
    plane_points=PLANE,
    ptchg_files=ptchg_files,
    conditions=CONDITIONS,
    origin=ORIGIN,
    extent='auto',                   # grid big enough for the whole cage
    spacing=0.1,                     # grid step, Angstrom
    mask_radius=1.6,                 # blank cells within 1.6 A of a cage atom
    hull_atoms='cage',               # and outside the cage (remove to map outside too)
)

ep.plot_esp_profile(res, out=OUTPUT)
ep.save_esp_profile(f'{OUTPUT}.npz', res, name=OUTPUT)

print(f'{res["n_frames"]} frame(s) -> {OUTPUT}.png, {OUTPUT}.svg, {OUTPUT}.npz')
for c in res['conditions']:
    print(f'  {c:10s} cavity mean (|z|, |x| <= 4 A): {ep.region_mean(res, c):+.3f} V')
