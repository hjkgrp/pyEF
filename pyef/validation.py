"""
Validation utilities for pyEF.

This module provides centralized validation functions for input parameters,
file paths, and data formats. All validation functions raise ValueError with
helpful error messages that include examples and references to documentation.
"""

import os
from typing import Union, List, Optional, Any


# Valid charge partitioning schemes supported by pyEF
VALID_CHARGE_TYPES = {
    'Hirshfeld', 'Hirshfeld_I', 'Voronoi', 'Mulliken', 'Lowdin',
    'SCPA', 'Becke', 'ADCH', 'CHELPG', 'MK', 'AIM',
    'CM5', 'EEM', 'RESP', 'PEOE'
}

# Valid charge types for multipole analysis (only these work with Multiwfn multipole mode)
VALID_MULTIPOLE_CHARGE_TYPES = {
    'Hirshfeld', 'Hirshfeld_I', 'Becke'
}


def get_atom_count_from_xyz(xyz_path: str) -> int:
    """
    Parse the number of atoms from an XYZ file.

    Parameters
    ----------
    xyz_path : str
        Path to the XYZ file

    Returns
    -------
    int
        Number of atoms in the structure

    Raises
    ------
    ValueError
        If the file doesn't exist or has invalid format
    """
    if not os.path.exists(xyz_path):
        raise ValueError(f"""
{'='*60}
ERROR: XYZ file not found
{'='*60}
Path: {xyz_path}

The file does not exist. Please check the path.

For help with file paths, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 2.2 (Input File Formats)
{'='*60}
""")

    try:
        with open(xyz_path, 'r') as f:
            first_line = f.readline().strip()
            atom_count = int(first_line)
            return atom_count
    except (ValueError, IOError) as e:
        raise ValueError(f"""
{'='*60}
ERROR: Invalid XYZ file format
{'='*60}
File: {xyz_path}
Issue: {str(e)}

XYZ format requires:
- First line: number of atoms (integer)
- Second line: comment
- Following lines: element x y z

Example XYZ file:
  3
  Water molecule
  O  0.000  0.000  0.000
  H  0.757  0.586  0.000
  H -0.757  0.586  0.000

For more information, see:
- README.md: Section 2.2 (Input File Formats)
{'='*60}
""")


def validate_charge_type(charge_type: str, context: str = "") -> None:
    """
    Validate that a charge type is in the set of supported schemes.

    Parameters
    ----------
    charge_type : str
        The charge partitioning scheme to validate
    context : str, optional
        Additional context for the error message

    Raises
    ------
    ValueError
        If the charge type is not valid
    """
    if charge_type not in VALID_CHARGE_TYPES:
        context_str = f" in {context}" if context else ""

        # Find similar charge types (case-insensitive match)
        similar = [ct for ct in VALID_CHARGE_TYPES if ct.lower() == charge_type.lower()]
        suggestion = f"\nDid you mean: '{similar[0]}'? (Note: charge types are case-sensitive)" if similar else ""

        raise ValueError(f"""
{'='*60}
ERROR: Invalid charge partitioning scheme{context_str}
{'='*60}
Specified: '{charge_type}'

Valid charge types (case-sensitive):
  - Hirshfeld       (fast, good for most systems)
  - Hirshfeld_I     (most accurate, slower)
  - CHELPG          (for ESP fitting)
  - Becke           (fast, less accurate)
  - Mulliken, Lowdin, SCPA, ADCH, Voronoi
  - MK, AIM, CM5, EEM, RESP, PEOE{suggestion}

Example usage in config.yaml:
  charge_types: ['Hirshfeld_I']

Example usage in Python:
  esp_df = estat.getESP(['Hirshfeld_I'], 'output', multiwfn_path)

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 2.3 (Key Parameters)
{'='*60}
""")


def filter_charge_types_for_multipole(charge_types: Union[str, List[str]],
                                       context: str = "") -> List[str]:
    """
    Filter charge types to only those that work with multipole analysis.

    Only Hirshfeld, Hirshfeld_I, and Becke are supported for multipole analysis.
    This function validates the input, warns about unsupported types, and returns
    only the valid ones.

    Parameters
    ----------
    charge_types : str or list of str
        Charge partitioning scheme(s) to validate
    context : str, optional
        Additional context for warning/error messages

    Returns
    -------
    list of str
        Filtered list containing only valid multipole charge types.
        Returns empty list if no valid types are found.

    Warnings
    --------
    Prints warnings for any charge types that don't support multipole analysis
    """
    # Convert to list if string
    if isinstance(charge_types, str):
        charge_types = [charge_types]

    context_str = f" in {context}" if context else ""

    # First validate that all charge types are generally valid
    for charge_type in charge_types:
        validate_charge_type(charge_type, context=context)

    # Filter for multipole-compatible types
    valid_for_multipole = []
    invalid_for_multipole = []

    for charge_type in charge_types:
        if charge_type in VALID_MULTIPOLE_CHARGE_TYPES:
            valid_for_multipole.append(charge_type)
        else:
            invalid_for_multipole.append(charge_type)

    # Warn about invalid types
    if invalid_for_multipole:
        print(f"""
{'='*60}
WARNING: Invalid charge type(s) for multipole analysis{context_str}
{'='*60}
The following charge types do NOT support multipole analysis:
  {', '.join(invalid_for_multipole)}

Only these charge types work with multipole analysis:
  Hirshfeld, Hirshfeld_I, Becke

These charge types will be SKIPPED for multipole calculations.
""")

        if valid_for_multipole:
            print(f"Valid charge types that will be used: {', '.join(valid_for_multipole)}")
            print(f"{'='*60}\n")

    # If no valid types remain, just return empty list (caller will handle)
    if not valid_for_multipole:
        print(f"No valid charge types remaining for multipole analysis{context_str}. Skipping.")
        print(f"{'='*60}\n")

    return valid_for_multipole


def filter_charge_types_for_monopole(charge_types: Union[str, List[str]],
                                      context: str = "") -> List[str]:
    """
    Filter charge types to only those that work with monopole analysis.

    All charge types in VALID_CHARGE_TYPES are supported for monopole analysis.
    This function validates the input and returns only valid types.

    Parameters
    ----------
    charge_types : str or list of str
        Charge partitioning scheme(s) to validate
    context : str, optional
        Additional context for warning/error messages

    Returns
    -------
    list of str
        Filtered list containing only valid monopole charge types.
        Returns empty list if no valid types are found.

    Warnings
    --------
    Prints warnings for any invalid charge types
    """
    # Convert to list if string
    if isinstance(charge_types, str):
        charge_types = [charge_types]

    context_str = f" in {context}" if context else ""

    # Filter for valid monopole types
    valid_for_monopole = []
    invalid_for_monopole = []

    for charge_type in charge_types:
        if charge_type in VALID_CHARGE_TYPES:
            valid_for_monopole.append(charge_type)
        else:
            invalid_for_monopole.append(charge_type)

    # Warn about invalid types
    if invalid_for_monopole:
        print(f"""
{'='*60}
WARNING: Invalid charge type(s) for monopole analysis{context_str}
{'='*60}
The following charge types are NOT valid:
  {', '.join(invalid_for_monopole)}

Valid charge types for monopole analysis:
  Hirshfeld, Hirshfeld_I, Voronoi, Mulliken, Lowdin,
  SCPA, Becke, ADCH, CHELPG, MK, AIM, CM5, EEM, RESP, PEOE

These charge types will be SKIPPED.
""")

        if valid_for_monopole:
            print(f"Valid charge types that will be used: {', '.join(valid_for_monopole)}")
            print(f"{'='*60}\n")

    # If no valid types remain, just return empty list (caller will handle)
    if not valid_for_monopole:
        print(f"No valid charge types remaining for monopole analysis{context_str}. Skipping.")
        print(f"{'='*60}\n")

    return valid_for_monopole


def check_path_exists(path: str, path_type: str = "file", context: str = "") -> None:
    """
    Check if a file or directory path exists.

    Parameters
    ----------
    path : str
        Path to check
    path_type : str
        Either "file" or "directory"
    context : str, optional
        Additional context for the error message

    Raises
    ------
    ValueError
        If the path does not exist
    """
    if not os.path.exists(path):
        context_str = f" ({context})" if context else ""

        if path_type == "file":
            raise ValueError(f"""
{'='*60}
ERROR: File not found{context_str}
{'='*60}
Path: {path}

The file does not exist. Please check:
1. The path is correct
2. The file exists in the specified location
3. You have read permissions for the file

For help with file paths, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
- README.md: Section 2 (Quick Start Guide)
{'='*60}
""")
        else:
            raise ValueError(f"""
{'='*60}
ERROR: Directory not found{context_str}
{'='*60}
Path: {path}

The directory does not exist. Please check:
1. The path is correct
2. The directory exists
3. You have read permissions

For help, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")

    # Additional check for executability if it's the Multiwfn path
    if context.lower() == "multiwfn_path" and path_type == "file":
        if not os.access(path, os.X_OK):
            raise ValueError(f"""
{'='*60}
ERROR: Multiwfn executable is not executable
{'='*60}
Path: {path}

The file exists but is not executable. To fix:
  chmod +x {path}

Multiwfn is required for all pyEF calculations.

For installation instructions, see:
- README.md: Section 4 (Installation)
- https://pyef.readthedocs.io/
{'='*60}
""")


def validate_numeric_range(
    value: Any,
    name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    allowed_values: Optional[List] = None,
    context: str = ""
) -> None:
    """
    Validate that a numeric parameter is within acceptable range.

    Parameters
    ----------
    value : Any
        The value to validate
    name : str
        Parameter name for error messages
    min_val : float, optional
        Minimum allowed value (inclusive)
    max_val : float, optional
        Maximum allowed value (inclusive)
    allowed_values : list, optional
        List of specific allowed values
    context : str, optional
        Additional context for error message

    Raises
    ------
    ValueError
        If the value is not numeric or out of range
    """
    context_str = f" in {context}" if context else ""

    # Type check
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"""
{'='*60}
ERROR: Invalid type for {name}{context_str}
{'='*60}
Expected: numeric value (int or float)
Got: {type(value).__name__} = {value}

Example:
  {name}: 1.0
  # or
  {name}: 4

For more examples, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")

    # Check against allowed values
    if allowed_values is not None:
        if isinstance(value, int):
            if value not in allowed_values:
                raise ValueError(f"""
{'='*60}
ERROR: Invalid value for {name}{context_str}
{'='*60}
Value: {value}
Allowed values: {allowed_values}

Example:
  {name}: {allowed_values[0]}

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")
        else:
            # For float values, check if close to any allowed value
            if not any(abs(numeric_value - av) < 1e-10 for av in allowed_values):
                raise ValueError(f"""
{'='*60}
ERROR: Invalid value for {name}{context_str}
{'='*60}
Value: {value}
Allowed values: {allowed_values}

Example:
  {name}: {allowed_values[0]}

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")
        return

    # Range validation
    if min_val is not None and numeric_value < min_val:
        raise ValueError(f"""
{'='*60}
ERROR: {name} value too small{context_str}
{'='*60}
Value: {value}
Minimum allowed: {min_val}

Example:
  {name}: {min_val if min_val >= 1 else 1.0}

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")

    if max_val is not None and numeric_value > max_val:
        raise ValueError(f"""
{'='*60}
ERROR: {name} value too large{context_str}
{'='*60}
Value: {value}
Maximum allowed: {max_val}

Example:
  {name}: {max_val if max_val <= 10 else 10.0}

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/example_config.yaml
{'='*60}
""")


def validate_atom_indices(
    indices: Union[int, List[int]],
    atom_count: int,
    context: str = "",
    xyz_path: str = ""
) -> None:
    """
    Validate that atom indices are within structure bounds.

    Parameters
    ----------
    indices : int or list of int
        Atom index or list of atom indices to validate (0-indexed)
    atom_count : int
        Total number of atoms in the structure
    context : str, optional
        Description of where indices are used (for error message)
    xyz_path : str, optional
        Path to XYZ file for reference in error message

    Raises
    ------
    ValueError
        If any index is out of bounds
    """
    # Convert single index to list
    if isinstance(indices, int):
        indices = [indices]

    # Find invalid indices
    invalid = [idx for idx in indices if idx >= atom_count or idx < 0]

    if invalid:
        context_str = f" in {context}" if context else ""
        xyz_str = f"\nStructure: {xyz_path}" if xyz_path else ""

        raise ValueError(f"""
{'='*60}
ERROR: Atom index out of bounds{context_str}
{'='*60}{xyz_str}
Total atoms: {atom_count} (valid indices: 0 to {atom_count-1})
Invalid indices: {invalid}

Note: pyEF uses 0-based indexing (first atom = index 0)

To find the correct atom index:
1. Open the XYZ file in a text editor
2. Count atoms starting from 0 (first atom = index 0)
3. The Nth atom has index N-1

Example for a 50-atom system:
  esp_atom_idx: [25]     # Valid (< 50)
  substrate_idxs: [0, 1, 2]  # Valid
  # atom index 50 would be INVALID (>= 50)

For more help, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
- README.md: Section 2.2 (Input File Formats)
{'='*60}
""")


def check_index_overlap(
    indices1: List[int],
    indices2: List[int],
    name1: str = "first set",
    name2: str = "second set",
    context: str = ""
) -> None:
    """
    Check if two sets of indices have any overlap.

    Parameters
    ----------
    indices1 : list of int
        First set of atom indices
    indices2 : list of int
        Second set of atom indices
    name1 : str
        Name of first set for error message
    name2 : str
        Name of second set for error message
    context : str, optional
        Additional context for error message

    Raises
    ------
    ValueError
        If there is any overlap between the sets
    """
    overlap = set(indices1) & set(indices2)

    if overlap:
        context_str = f" ({context})" if context else ""

        raise ValueError(f"""
{'='*60}
ERROR: Overlapping atom indices{context_str}
{'='*60}
{name1}: {sorted(indices1)}
{name2}: {sorted(indices2)}
Overlapping indices: {sorted(overlap)}

The same atoms cannot be in both sets.
This would lead to incorrect results.

To fix:
- Ensure the two sets are mutually exclusive
- Review your system partitioning

Example:
  substrate_idxs: [0, 1, 2]      # Active site atoms
  env_idxs: [10, 11, 12, 13]     # Environment (no overlap)

For more information, see:
- /home/gridsan/mmanetsch/pyEF/pyef/ExampleUsage.py
- README.md: Section 3 (Electrostatic Stabilization)
{'='*60}
""")
