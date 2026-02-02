"""Command-line interface (CLI) entry point"""

import os
import yaml
import click
from .validation import (
    check_path_exists,
    validate_charge_type,
    validate_numeric_range,
    check_index_overlap,
    VALID_CHARGE_TYPES
)

def welcome():
    """Print first to welcome the user while it waits to load the modules"""
    print("\n")
    print("             ╔═══════════════════════╗")
    print("             ║        ┌──────┐       ║")
    print("             ║        │  ┌───┘       ║")
    print("             ║        │  └───┐       ║")
    print("             ║        │  ┌───┘       ║")
    print("             ║        │  ├┬┬┬┐       ║")
    print("             ║        └──┴┴┴┴┘       ║")
    print("             ║                       ║")
    print("             ║    WELCOME TO PYEF    ║")
    print("             ╚═══════════╗╔══════════╝")
    print("                 ╔═══════╝╚═══════╗                 ")
    print("                 ║ THE KULIK LAB  ║                 ")
    print("                 ╚═══════╗╔═══════╝                 ")
    print("  ╔══════════════════════╝╚══════════════════════╗  ")
    print("  ║  Code: github.com/davidkastner/pyef          ║  ")
    print("  ║  Docs: pyef.readthedocs.io                   ║  ")
    print("  ║     - Usage: pyef -c pyef.in                 ║  ")
    print("  ║  Example: github.com/davidkastner/pyef/demo  ║  ")
    print("  ╚══════════════════════════════════════════════╝  \n")

# Welcome even if no flags
welcome()

# Read in the configuration yaml file
def read_config(config_file):
    with open(config_file, 'r') as file:
        return yaml.safe_load(file)

def parse_job_batch_file(file_path):
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

    Notes
    -----
    New format (required):
    - analysis_type, path_to_molden, path_to_xyz, [bond_tuples or metal_index]
    - Examples:
      - ef, /path/to/optim.molden, /path/to/optim.xyz, (25, 26), (25, 27)
      - esp, /path/to/optim.molden, /path/to/optim.xyz, 30
      - estab, /path/to/optim.molden, /path/to/optim.xyz
      - ef+esp, /path/to/optim.molden, /path/to/optim.xyz, 35
    """
    import re

    analysis_types = []
    molden_paths = []
    xyz_paths = []
    metal_indices = []
    bond_indices = []

    # Valid analysis type keywords
    valid_analysis = {'ef', 'esp', 'estab'}

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
                raise ValueError(f"""
{'='*60}
ERROR: Invalid CSV format on line {line_num}
{'='*60}
Expected at least 3 columns, got {len(columns)}
Format: analysis_type, path_to_molden, path_to_xyz, [atom_indices]

Current line: {line}

Examples of correct format:
  ef, /path/to/job.molden, /path/to/job.xyz, (25, 26)
  esp, /path/to/job.molden, /path/to/job.xyz, 30
  estab, /path/to/job.molden, /path/to/job.xyz
  ef+esp, /path/to/job.molden, /path/to/job.xyz, (10, 11)

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml (lines 67-74)
- README.md: Section 2.2 (Job Batch File Format)
{'='*60}
""")

            # Extract analysis type (column 1)
            analysis_type = columns[0].lower().strip()
            analysis_parts = set(analysis_type.replace('+', ' ').split())
            if not analysis_parts.issubset(valid_analysis) or not analysis_parts:
                raise ValueError(f"""
{'='*60}
ERROR: Invalid analysis type on line {line_num}
{'='*60}
Specified: '{columns[0]}'
Valid options: {', '.join(sorted(valid_analysis))}

You can also combine analyses with '+':
  ef+esp    # Run both electric field and ESP
  ef+estab  # Run electric field and stabilization

Examples:
  ef, /path/to/job.molden, /path/to/job.xyz, (25, 26)
  esp, /path/to/job.molden, /path/to/job.xyz, 30
  estab, /path/to/job.molden, /path/to/job.xyz

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 2.2 (Job Batch File Format)
{'='*60}
""")
            analysis_types.append(columns[0])  # Keep original case

            # Extract molden path (column 2)
            molden_path = columns[1].strip()
            if not molden_path:
                raise ValueError(f"""
{'='*60}
ERROR: Missing molden file path on line {line_num}
{'='*60}
Column 2 is empty or missing.

Format: analysis_type, path_to_molden, path_to_xyz, [atom_indices]

Example:
  ef, /path/to/job/optim.molden, /path/to/job/optim.xyz, 25

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 2.2 (Job Batch File Format)
{'='*60}
""")
            molden_paths.append(molden_path)

            # Extract xyz path (column 3)
            xyz_path = columns[2].strip()
            if not xyz_path:
                raise ValueError(f"""
{'='*60}
ERROR: Missing XYZ file path on line {line_num}
{'='*60}
Column 3 is empty or missing.

Format: analysis_type, path_to_molden, path_to_xyz, [atom_indices]

Example:
  ef, /path/to/job/optim.molden, /path/to/job/optim.xyz, 25

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 2.2 (Job Batch File Format)
{'='*60}
""")
            xyz_paths.append(xyz_path)

            # Parse remaining columns for metal indices and/or bond tuples
            remainder = ','.join(columns[3:])
            has_tuples = '(' in remainder and ')' in remainder

            if has_tuples:
                # Parse tuple format for bonds
                job_bonds = []
                metal_idx = None

                tuple_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                matches = re.findall(tuple_pattern, remainder)

                for match in matches:
                    atom1, atom2 = int(match[0]), int(match[1])
                    job_bonds.append((atom1, atom2))
                    # Set metal_idx to first atom of first bond (for compatibility)
                    if metal_idx is None:
                        metal_idx = atom1

                metal_indices.append(metal_idx)
                bond_indices.append(job_bonds)

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
                except ValueError:
                    raise ValueError(f"""
{'='*60}
ERROR: Invalid atom index format on line {line_num}
{'='*60}
Specified: '{columns[3]}'
Expected: integer or tuple format

Valid formats:
  - Single atom index: 25
  - Bond tuple: (25, 26)
  - Multiple bonds: (25, 26), (25, 27)

Examples:
  ef, /path/to/job.molden, /path/to/job.xyz, (25, 26)
  esp, /path/to/job.molden, /path/to/job.xyz, 30

Note: Atom indices are 0-based (first atom = 0)

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 2.2 (Job Batch File Format)
{'='*60}
""")
            else:
                # No additional data
                metal_indices.append(None)
                bond_indices.append([])

    # Return None for metal/bond indices if all are empty/None
    if metal_indices and all(m is None for m in metal_indices):
        metal_indices = None
    if bond_indices and all(not b for b in bond_indices):
        bond_indices = None

    return analysis_types, molden_paths, xyz_paths, metal_indices, bond_indices

def read_file_lines(file_path):
    """Reads in auxiliary files containing job information"""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

def run_analysis(job_name, molden_paths, xyz_paths, analysis_types, metal_indices, bond_indices, dielectric,
         geom_flag,
         multiwfn_path,
         charge_types=['Hirshfeld_I'], multipole_bool=True, use_multipole=False,
         save_atomwise_decomposition=False, substrate_idxs=None, env_idxs=None, multipole_order=2,
         include_ptchgs=False, ptchg_file='', dielectric_scale=1.0, use_ecp=False, exclude_atoms=[]):
    """
    Main function for running the pyEF workflow.

    Parameters
    ----------
    job_name : str
        Name for output files
    molden_paths : list of str
        List of paths to .molden files
    xyz_paths : list of str
        List of paths to .xyz files
    analysis_types : list of str
        List of analysis types per job ('ef', 'esp', 'estab', or combinations)
    metal_indices : list of int or None
        List of metal atom indices (only required for ESP analysis or auto-finding bonds)
    bond_indices : list of tuples
        List of bond pairs (required for E-field if metal_indices not provided)
    dielectric : float
        Dielectric constant
    geom_flag : bool
        Run geometry checks
    multiwfn_path : str
        Path to Multiwfn executable
    charge_types : list of str, optional
        Charge partitioning schemes (default: ['Hirshfeld_I'])
    multipole_bool : bool, optional
        Use multipole expansion for E-field (default: True)
    use_multipole : bool, optional
        Use multipole expansion for ESP (default: False)
    save_atomwise_decomposition : bool, optional
        Save atom-wise decomposition to CSV (default: False)
    substrate_idxs : list or None, optional
        Substrate atom indices for estab calculation (default: None)
    env_idxs : list or None, optional
        Environment atom indices for estab calculation (default: None)
    multipole_order : int, optional
        Multipole expansion order (default: 2)
    include_ptchgs : bool, optional
        Include MM point charges from QM/MM calculations (default: False)
    ptchg_file : str, optional
        Filename of point charge file (default: '')
        File is looked for in the same directory as each job's molden file.
        Example: 'pointcharges.txt' will look for this file in each job directory.
    dielectric_scale : float, optional
        Dielectric scaling factor for MM charges (default: 1.0)
    use_ecp : bool, optional
        Use effective core potential (ECP) correction for molden files (default: False)
    exclude_atoms : list of int or list of lists, optional
        Atom indices to exclude from E-field calculations (default: [])
        Can be either:
        - Flat list [0, 1, 2] for same exclusions across all jobs
        - Nested list [[0, 1], [2, 3], []] for different exclusions per job

    Notes
    -----
    - ESP calculation for ~400 atom system takes about 15 minutes on one node
    - E-field calculation for ~400 atom system takes ~1 hour on one node
    - If jobs end abruptly, delete the polarization file before restarting
    """
    from pyef.analysis import Electrostatics
    from collections import defaultdict

    # Validate inputs
    if len(molden_paths) != len(xyz_paths) or len(molden_paths) != len(analysis_types):
        raise ValueError("molden_paths, xyz_paths, and analysis_types must have the same length")

    # Determine if using per-job analysis types (always True in new format)
    use_per_job_analysis = True

    # Group jobs by analysis type for efficient batch processing
    analysis_groups = defaultdict(list)  # {analysis_type: [job_indices, ...]}

    for idx, analysis_type in enumerate(analysis_types):
        if analysis_type:
            # Handle combined analysis types (e.g., 'ef+esp')
            for atype in analysis_type.lower().replace('+', ' ').split():
                analysis_groups[atype].append(idx)

    # Process each analysis type
    for analysis_type, job_indices in analysis_groups.items():
        # Filter data for this analysis type
        filtered_molden = [molden_paths[i] for i in job_indices]
        filtered_xyz = [xyz_paths[i] for i in job_indices]
        filtered_metals = [metal_indices[i] if metal_indices else None for i in job_indices]
        filtered_bonds = [bond_indices[i] if bond_indices else [] for i in job_indices]

        print(f"\n{'='*60}")
        print(f"Running {analysis_type.upper()} analysis on {len(filtered_molden)} jobs")
        print(f"{'='*60}\n")

        # Initialize Electrostatics Object for this group
        incage_bool = False
        if any(filtered_metals):
            dataObject = Electrostatics(filtered_molden, filtered_xyz,
                                      esp_atom_idx=filtered_metals,
                                      incage_bool=incage_bool, dielectric=dielectric)
        else:
            dataObject = Electrostatics(filtered_molden, filtered_xyz,
                                      incage_bool=incage_bool, dielectric=dielectric)

        # Configure QM/MM if requested
        if include_ptchgs and ptchg_file:
            dataObject.includePtChgs(ptchg_file)
            dataObject.set_dielectric_scale(dielectric_scale)

        # Set excluded atoms if specified
        if exclude_atoms:
            dataObject.setExcludeAtomFromCalc(exclude_atoms)

        # Prepare data
        dataObject.prepData()
        if use_ecp:
            dataObject.fix_allECPmolden()

        # Run geometry check if requested
        if geom_flag:
            print("Warning: geometry checks are not implemented yet; skipping.")

        # Run the appropriate analysis
        if analysis_type == 'esp':
            # ESP can use multiple charge types, so we'll use the first one for the filename
            chg_type = charge_types[0] if isinstance(charge_types, list) else charge_types
            ESPdata_filename = f'{analysis_type}_{job_name}_{chg_type}'
            dataObject.getESP(charge_types, ESPdata_filename,
                           multiwfn_path, use_multipole=use_multipole,
                           dielectric=dielectric)

        elif analysis_type == 'ef':
            chg_type = charge_types[0] if isinstance(charge_types, list) else charge_types
            Efielddata_filename = f'{analysis_type}_{job_name}_{chg_type}'
            dataObject.getEfield(chg_type,
                               Efielddata_filename, multiwfn_path,
                               multipole_bool=multipole_bool, input_bond_indices=filtered_bonds,
                               save_atomwise_decomposition=save_atomwise_decomposition, dielectric=dielectric)

        elif analysis_type == 'estab':
            if substrate_idxs is None:
                print(f"Warning: substrate_idxs not specified for {analysis_type}. Skipping...")
            else:
                chg_type = charge_types[0] if isinstance(charge_types, list) else charge_types
                estab_filename = f'{analysis_type}_{job_name}_{chg_type}'
                dataObject.getElectrostatic_stabilization(
                    multiwfn_path,
                    substrate_idxs=substrate_idxs,
                    charge_type=chg_type,
                    name_dataStorage=estab_filename,
                    env_idxs=env_idxs,
                    save_atomwise_decomposition=save_atomwise_decomposition,
                    multipole_order=multipole_order
                )

@click.group(invoke_without_command=True)
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to the configuration YAML file")
@click.pass_context
def cli(ctx, config):
    """CLI entry point"""
    if ctx.invoked_subcommand is None and config:
        ctx.invoke(run, config=config)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@click.command()
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to the configuration YAML file")
def run(config):
    config_data = read_config(config)
    input = config_data.get('input', '')
    dielectric = config_data.get('dielectric', 1)
    run_ef = config_data.get('ef', False)
    run_esp = config_data.get('esp', False)
    run_estab = config_data.get('estab', False)
    run_geometry_check = config_data.get('geometry_check', False)
    multiwfn_path = config_data.get('multiwfn_path', False)
    # Additional configuration options
    charge_types = config_data.get('charge_types', ['Hirshfeld_I'])
    multipole_bool = config_data.get('multipole', True)
    use_multipole = config_data.get('use_multipole', False)
    save_atomwise_decomposition = config_data.get('save_atomwise_decomposition', False)
    substrate_idxs = config_data.get('substrate_idxs', None)
    env_idxs = config_data.get('env_idxs', None)
    multipole_order = config_data.get('multipole_order', 2)

    # QM/MM options
    include_ptchgs = config_data.get('include_ptchgs', False)
    ptchg_file = config_data.get('ptchg_file', '')
    dielectric_scale = config_data.get('dielectric_scale', 1.0)

    # ECP (Effective Core Potential) option
    use_ecp = config_data.get('use_ecp', False)

    # Exclude atoms from E-field calculation
    exclude_atoms = config_data.get('exclude_atoms', [])

    # ===== VALIDATE CONFIG PARAMETERS =====

    # Validate multiwfn_path exists and is executable
    if not multiwfn_path:
        raise ValueError(f"""
{'='*60}
ERROR: Missing multiwfn_path in configuration
{'='*60}
The 'multiwfn_path' parameter is required but not specified.

Multiwfn is required for all pyEF calculations.

Example in config.yaml:
  multiwfn_path: /usr/local/bin/Multiwfn

For installation instructions, see:
- README.md: Section 4 (Installation)
- https://pyef.readthedocs.io/
{'='*60}
""")
    check_path_exists(multiwfn_path, path_type="file", context="multiwfn_path")

    # Validate input file exists
    if not input:
        raise ValueError(f"""
{'='*60}
ERROR: Missing input file in configuration
{'='*60}
The 'input' parameter is required but not specified.

Example in config.yaml:
  input: jobs.csv

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 2 (Quick Start Guide)
{'='*60}
""")
    check_path_exists(input, path_type="file", context="input file")

    # Validate dielectric constant
    validate_numeric_range(dielectric, "dielectric", min_val=0.001, context="config file")

    # Validate multipole_order
    validate_numeric_range(multipole_order, "multipole_order", allowed_values=[1, 2, 3], context="config file")

    # Validate dielectric_scale
    validate_numeric_range(dielectric_scale, "dielectric_scale", min_val=0.001, context="config file")

    # Validate charge types
    if not isinstance(charge_types, list):
        raise ValueError(f"""
{'='*60}
ERROR: Invalid charge_types format in config
{'='*60}
Expected: list of charge type strings
Got: {type(charge_types).__name__}

Example in config.yaml:
  charge_types: ['Hirshfeld_I']
  # or
  charge_types: ['Hirshfeld', 'Becke']

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")

    for charge_type in charge_types:
        validate_charge_type(charge_type, context="config file")

    # Validate substrate_idxs and env_idxs don't overlap (for estab calculations)
    if substrate_idxs is not None and env_idxs is not None:
        if not isinstance(substrate_idxs, list):
            raise ValueError(f"""
{'='*60}
ERROR: Invalid substrate_idxs format
{'='*60}
Expected: list of integers
Got: {type(substrate_idxs).__name__}

Example in config.yaml:
  substrate_idxs: [0, 1, 2, 3]

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")

        if not isinstance(env_idxs, list):
            raise ValueError(f"""
{'='*60}
ERROR: Invalid env_idxs format
{'='*60}
Expected: list of integers
Got: {type(env_idxs).__name__}

Example in config.yaml:
  env_idxs: [10, 11, 12, 13]

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")

        check_index_overlap(substrate_idxs, env_idxs, "substrate_idxs", "env_idxs", "electrostatic stabilization")

    # Validate QM/MM options
    if include_ptchgs and not ptchg_file:
        raise ValueError(f"""
{'='*60}
ERROR: Missing ptchg_file for QM/MM calculation
{'='*60}
include_ptchgs is set to true, but ptchg_file is not specified.

Example in config.yaml:
  include_ptchgs: true
  ptchg_file: 'pointcharges.txt'

Note: The file should exist in the same directory as each job's
molden file.

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 3.4 (QM/MM Calculations)
{'='*60}
""")

    # ===== END VALIDATION =====

    # Parse job batch file
    analysis_types, molden_paths, xyz_paths, metal_indices, bond_indices = parse_job_batch_file(input)
    job_name = os.path.splitext(input)[0]

    # New format always has analysis types specified per-job
    if not analysis_types:
        click.echo("Error: No valid jobs found in the input file.")
        click.echo("Expected format: analysis_type, path_to_molden, path_to_xyz [, atom_indices]")
        click.echo("Examples:")
        click.echo("  ef, /path/to/optim.molden, /path/to/optim.xyz, (25, 26)")
        click.echo("  esp, /path/to/optim.molden, /path/to/optim.xyz, 30")
        click.echo("  estab, /path/to/optim.molden, /path/to/optim.xyz")
        return

    # Run per-job analyses (new behavior - each job specifies its own analysis)
    run_analysis(
        job_name, molden_paths, xyz_paths, analysis_types, metal_indices, bond_indices, dielectric,
        run_geometry_check,
        multiwfn_path,
        charge_types, multipole_bool, use_multipole,
        save_atomwise_decomposition, substrate_idxs, env_idxs, multipole_order,
        include_ptchgs, ptchg_file, dielectric_scale, use_ecp, exclude_atoms
    )

cli.add_command(run)

if __name__ == '__main__':
    cli()
