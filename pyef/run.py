import os
import sys
import pyef
import argparse
from pyef.analysis import Electrostatics

def main(job_name, molden_paths, xyz_paths, analysis_types, metal_indices, bond_indices, dielectric,
         geom_flag,
         multiwfn_path,
         charge_types=['Hirshfeld_I'], multipole_bool=True, use_multipole=False,
         decompose_atomwise=False, substrate_idxs=None, env_idxs=None, multipole_order=2,
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
        Geometry checks are currently not implemented; flag is accepted but skipped
    multiwfn_path : str
        Path to Multiwfn executable
    charge_types : list of str, optional
        Charge partitioning schemes (default: ['Hirshfeld_I'])
    multipole_bool : bool, optional
        Use multipole expansion for E-field (default: True)
    use_multipole : bool, optional
        Use multipole expansion for ESP (default: False)
    decompose_atomwise : bool, optional
        Compute atom-wise decomposition (default: False)
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

    # Validate inputs
    if len(molden_paths) != len(xyz_paths) or len(molden_paths) != len(analysis_types):
        raise ValueError("molden_paths, xyz_paths, and analysis_types must have the same length")

    # Determine if using per-job analysis types (always True in new format)
    use_per_job_analysis = True

    # Group jobs by analysis type for efficient batch processing
    from collections import defaultdict
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
                           dielectric=dielectric)

        elif analysis_type == 'ef':
            chg_type = charge_types[0] if isinstance(charge_types, list) else charge_types
            Efielddata_filename = f'{analysis_type}_{job_name}_{chg_type}'
            dataObject.getEfield(chg_type,
                               multipole_bool=multipole_bool, input_bond_indices=filtered_bonds,
                               decompose_atomwise=decompose_atomwise, dielectric=dielectric)

        elif analysis_type == 'estab':
            if substrate_idxs is None:
                print(f"Warning: substrate_idxs not specified for {analysis_type}. Skipping...")
            else:
                chg_type = charge_types[0] if isinstance(charge_types, list) else charge_types
                estab_filename = f'{analysis_type}_{job_name}_{chg_type}'
                dataObject.getElectrostatic_stabilization(
                    substrate_idxs=substrate_idxs,
                    charge_type=chg_type,
                    name_dataStorage=estab_filename,
                    env_idxs=env_idxs,
                    decompose_atomwise=decompose_atomwise,
                    multipole_order=multipole_order
                )

def read_file_lines(file_path):
    """Reads in auxiliary files containing job information"""
    with open(file_path, 'r') as file:
        return [line.strip() for line in file.readlines()]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyEF Analysis Pipeline")
    parser.add_argument("--geom", action="store_true", help="Perform a geometry check")
    parser.add_argument("--esp", action="store_true", help="Perform electrostatic potential (ESP) analysis")
    parser.add_argument("--ef", action="store_true", help="Perform electric field analysis")
    parser.add_argument("--estab", action="store_true", help="Perform electrostatic stabilization analysis")
    parser.add_argument("--jobs_file", required=True, help="Path to file containing job paths")
    parser.add_argument("--metals_file", required=True, help="Path to file containing metal indices")
    parser.add_argument("--job_name", required=True, help="Name for the job/output files")
    parser.add_argument("--bonds_file", required=True, help="Path to file containing bond indices")
    parser.add_argument("--multiwfn_path", required=True, help="Path to multiwfn executable")
    parser.add_argument("--dielectric", type=float, default=1.0, help="Dielectric constant (default: 1.0)")
    parser.add_argument("--charge_types", nargs='+', default=['Hirshfeld_I'], help="Charge partitioning schemes (default: Hirshfeld_I)")
    parser.add_argument("--multipole", action="store_true", default=True, help="Use multipole expansion for E-field")
    parser.add_argument("--use_multipole", action="store_true", help="Use multipole expansion for ESP")
    parser.add_argument("--decompose_atomwise", action="store_true", help="Compute atom-wise decomposition")
    parser.add_argument("--substrate_idxs", type=str, help="Comma-separated substrate atom indices for estab (e.g., 1,2,3)")
    parser.add_argument("--env_idxs", type=str, help="Comma-separated environment atom indices for estab")
    parser.add_argument("--multipole_order", type=int, default=2, help="Multipole expansion order (default: 2)")

    args = parser.parse_args()
    geom_flag = args.geom
    esp_flag = args.esp
    ef_flag = args.ef
    estab_flag = args.estab
    jobs = read_file_lines(args.jobs_file)
    metal_indices = [int(idx) for idx in read_file_lines(args.metals_file)]
    bond_indices = [int(idx) for idx in read_file_lines(args.bonds_file)]

    # Parse substrate and environment indices if provided
    substrate_idxs = [int(x) for x in args.substrate_idxs.split(',')] if args.substrate_idxs else None
    env_idxs = [int(x) for x in args.env_idxs.split(',')] if args.env_idxs else None

    main(args.job_name, jobs, metal_indices, bond_indices, args.dielectric,
         geom_flag, esp_flag, ef_flag, estab_flag,
         args.charge_types, args.multipole, args.use_multipole,
         args.decompose_atomwise, substrate_idxs, env_idxs, args.multipole_order)
