"""Functions for managing files needed for pyEF"""

def parse_job_batch_file(file_path, return_plane_points=False):
    """
    Parse a CSV file and extract specific columns as lists and tuples.
    The input file allows Python style comments on any line.

    Parameters
    ----------
    file_path : str
        The file path of the CSV file to be parsed.

    Returns
    -------
    analysis_types : list of str
        List of analysis types ('ef', 'esp', 'estab', or combinations like 'ef+esp').
    molden_paths : list of str
        List of paths to .molden files.
    xyz_paths : list of str
        List of paths to .xyz files.
    metal_indices : list of int or None
        List containing metal atom indices.
        None if metal indices are not provided.
    bond_indices : list of tuples or None
        List of bond tuples for each job.
        None if bond indices are not provided.
    plane_points : list of tuple or None
        Only returned with ``return_plane_points=True``: the ``(P1, P2, P3)``
        plane of each job (None where absent), for ``espprofile``.

    Notes
    -----
    New format (required):
    - analysis_type, path_to_molden, path_to_xyz, [bond_tuples or metal_index]
    - Examples:
      - ef, /path/to/optim.molden, /path/to/optim.xyz, (25, 26), (25, 27)
      - esp, /path/to/optim.molden, /path/to/optim.xyz, 30
      - estab, /path/to/optim.molden, /path/to/optim.xyz
      - ef+esp, /path/to/optim.molden, /path/to/optim.xyz, 35
      - espprofile, /path/to/optim.molden, /path/to/optim.xyz, (1, 0, 2-34)
    """
    import re

    analysis_types = []
    molden_paths = []
    xyz_paths = []
    metal_indices = []
    bond_indices = []
    plane_points = []

    # Valid analysis type keywords
    valid_analysis = {'ef', 'esp', 'estab', 'espprofile'}

    with open(file_path, 'r') as file:
        for line_num, line in enumerate(file, 1):
            # Skip empty lines and comments that start with '#'
            if line.strip() == '' or line.strip().startswith('#'):
                continue

            # Remove comments from the line
            line = line.split('#')[0].strip()

            # Skip the line if it's empty after removing the comment
            if line == '':
                continue

            # Split line into columns (initially by comma)
            columns = [col.strip() for col in line.split(',')]

            # Validate minimum number of columns
            if len(columns) < 3:
                raise ValueError(
                    f"Line {line_num}: Invalid format. Expected at least 3 columns: "
                    f"analysis_type, path_to_molden, path_to_xyz [, atom_indices]. "
                    f"Got {len(columns)} columns."
                )

            # Extract analysis type (column 1)
            analysis_type = columns[0].lower().strip()
            analysis_parts = set(analysis_type.replace('+', ' ').split())
            if not analysis_parts.issubset(valid_analysis) or not analysis_parts:
                raise ValueError(
                    f"Line {line_num}: Invalid analysis type '{columns[0]}'. "
                    f"Must be one of: {', '.join(valid_analysis)} or combinations like 'ef+esp'."
                )
            analysis_types.append(columns[0])  # Keep original case

            # Extract molden path (column 2)
            molden_path = columns[1].strip()
            if not molden_path:
                raise ValueError(f"Line {line_num}: Missing molden file path in column 2.")
            molden_paths.append(molden_path)

            # Extract xyz path (column 3)
            xyz_path = columns[2].strip()
            if not xyz_path:
                raise ValueError(f"Line {line_num}: Missing xyz file path in column 3.")
            xyz_paths.append(xyz_path)

            # Parse remaining columns for metal indices and/or bond tuples
            remainder = ','.join(columns[3:])
            has_tuples = '(' in remainder and ')' in remainder

            if has_tuples:
                # Parse tuple format for bonds
                job_bonds = []
                metal_idx = None

                job_bonds, job_planes = parse_index_tuples(remainder, line_num)
                if len(job_planes) > 1:
                    raise ValueError(f"Line {line_num}: only one (a, b, c) plane per job is supported.")
                if job_bonds:
                    # Set metal_idx to first atom of first bond (for compatibility)
                    metal_idx = job_bonds[0][0]

                metal_indices.append(metal_idx)
                bond_indices.append(job_bonds)
                plane_points.append(job_planes[0] if job_planes else None)

            elif len(columns) > 3 and columns[3]:
                # Single metal index
                try:
                    metal_index = int(columns[3])
                    metal_indices.append(metal_index)

                    # Check if there's a bonded atom index
                    if len(columns) > 4 and columns[4]:
                        bonded_atom_index = int(columns[4])
                        bond_indices.append([(metal_index, bonded_atom_index)])
                    else:
                        bond_indices.append([])
                    plane_points.append(None)
                except ValueError:
                    raise ValueError(
                        f"Line {line_num}: Invalid atom index '{columns[3]}'. "
                        f"Expected integer or tuple format like (1, 2)."
                    )
            else:
                # No additional data
                metal_indices.append(None)
                bond_indices.append([])
                plane_points.append(None)

    # Return None for metal/bond indices if all are empty/None
    if metal_indices and all(m is None for m in metal_indices):
        metal_indices = None
    if bond_indices and all(not b for b in bond_indices):
        bond_indices = None

    if return_plane_points:
        return analysis_types, molden_paths, xyz_paths, metal_indices, bond_indices, plane_points
    return analysis_types, molden_paths, xyz_paths, metal_indices, bond_indices


def parse_index_tuples(remainder, line_num=None):
    """Split the parenthesized groups of a job line into bonds and planes.

    ``(a, b)`` is a bond (two integer indices).  ``(a, b, c)`` is a plane for
    ``espprofile``: each entry is an index, an inclusive range ``2-34``, or
    ``2-10+12`` (a group, whose centroid is used).  Entries that are plain
    integers are returned as ints, the rest as strings.

    Returns ``(bonds, planes)``, lists of tuples.
    """
    import re

    where = f"Line {line_num}: " if line_num is not None else ""
    bonds, planes = [], []
    for group in re.findall(r'\(([^()]*)\)', remainder):
        entries = [e.strip() for e in group.split(',')]
        for e in entries:
            if not re.fullmatch(r'\d+(-\d+)?(\+\d+(-\d+)?)*', e):
                raise ValueError(
                    f"{where}Invalid atom index '{e}' in '({group})'. "
                    f"Use integers, ranges like 2-34, or groups like 2-10+12."
                )
        if len(entries) == 2:
            if not all(e.isdigit() for e in entries):
                raise ValueError(f"{where}Bond '({group})' must be two integer atom indices.")
            bonds.append((int(entries[0]), int(entries[1])))
        elif len(entries) == 3:
            planes.append(tuple(int(e) if e.isdigit() else e for e in entries))
        else:
            raise ValueError(
                f"{where}'({group})' has {len(entries)} entries; use (a, b) for a "
                f"bond or (a, b, c) for an espprofile plane."
            )
    return bonds, planes
