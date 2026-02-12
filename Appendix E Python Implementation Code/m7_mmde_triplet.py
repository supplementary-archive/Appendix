# SPDX-License-Identifier: PROPRIETARY
# File: m7_mmde_triplet.py
# Purpose: Module 7 - MMDE Triplet Creation and Sorting for DEMATEL-ISM Integration

"""
MMDE (Maximum Mean De-Entropy) Threshold Calculation - Step 1

This module creates triplets from the Total Relation Matrix (TRM) and sorts them
in descending order by value. This is the first step in calculating the MMDE threshold.

Mathematical Procedure:
-----------------------
For each element t_ij in matrix T (including diagonal elements):
    Create triplet: (value, row_index, column_index)
    Using 1-based indexing for row and column

Sort all triplets in descending order by value to create ordered set T*
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import from Module 1
from m1_data_processing import (
    BARRIER_NAMES,
    OUTPUT_DIR_DEMATEL,
    get_script_directory
)

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================

# Output directory for ISM files
OUTPUT_DIR_ISM = "output/ism/"

# Number of decimal places for values
DECIMAL_PLACES = 4

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def load_trm_from_file(file_path=None):
    """
    Load Total Relation Matrix (TRM) from Module 3 Excel file.

    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the TRM Excel file. If None, uses default path.

    Returns:
    --------
    pandas.DataFrame
        TRM with barrier codes as index and columns
    """
    if file_path is None:
        script_dir = get_script_directory()
        file_path = script_dir / OUTPUT_DIR_DEMATEL / "dematel_trm.xlsx"

    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"TRM file not found: {file_path}\n"
            "Please run Module 3 (m3_dematel_trm.py) first to generate "
            "the Total Relation Matrix."
        )

    try:
        # Load the raw codes sheet (with short barrier codes)
        trm = pd.read_excel(file_path, sheet_name='TRM_Raw_Codes', index_col=0)
    except PermissionError:
        raise PermissionError(
            f"Cannot access file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )

    # Ensure column names and index are lowercase for consistency
    trm.columns = [str(col).lower() for col in trm.columns]
    trm.index = [str(idx).lower() for idx in trm.index]

    return trm


def create_triplets(trm):
    """
    Create triplets from the Total Relation Matrix.

    Mathematical Formula:
    ---------------------
    For each element t_ij in matrix T:
        triplet = (value, row_index, column_index)

    Using 1-based indexing as per the procedure document.

    Parameters:
    -----------
    trm : pandas.DataFrame
        Total Relation Matrix of size n×n

    Returns:
    --------
    list
        List of n² triplets, each as (value, row_index, column_index)
        Row and column indices are 1-based
    """
    n = len(trm)
    triplets = []

    # Get barrier codes for reference
    barriers = list(trm.index)

    # Create triplet for each element (including diagonal)
    # Using 1-based indexing as specified in the procedure
    for i in range(n):
        for j in range(n):
            value = trm.iloc[i, j]
            # 1-based indexing
            row_index = i + 1
            col_index = j + 1
            triplet = (value, row_index, col_index)
            triplets.append(triplet)

    return triplets


def sort_triplets_descending(triplets):
    """
    Sort triplets in descending order by value.

    This creates the ordered set T* where:
    - T*[0] has the highest value
    - T*[n²-1] has the lowest value

    Parameters:
    -----------
    triplets : list
        List of triplets (value, row_index, column_index)

    Returns:
    --------
    list
        Sorted list of triplets in descending order by value
    """
    # Sort by value (first element of triplet) in descending order
    sorted_triplets = sorted(triplets, key=lambda x: x[0], reverse=True)
    return sorted_triplets


def create_triplet_dataframe(sorted_triplets, barriers):
    """
    Create DataFrame for Excel export.

    Parameters:
    -----------
    sorted_triplets : list
        Sorted list of triplets
    barriers : list
        List of barrier codes

    Returns:
    --------
    pandas.DataFrame
        DataFrame with t_ij notation and triplet columns
    """
    data = []

    for triplet in sorted_triplets:
        value, row_idx, col_idx = triplet

        # Create t_ij notation (e.g., t_23 for row 2, col 3)
        t_ij_notation = f"t_{row_idx}{col_idx}"

        # Create barrier pair notation (e.g., B2→B3)
        barrier_from = barriers[row_idx - 1].upper()
        barrier_to = barriers[col_idx - 1].upper()
        barrier_pair = f"{barrier_from}→{barrier_to}"

        # Format triplet as string for display
        triplet_str = f"({value:.{DECIMAL_PLACES}f}, {row_idx}, {col_idx})"

        data.append({
            'Position': len(data) + 1,
            't_ij': t_ij_notation,
            'Barrier_Pair': barrier_pair,
            'Value': round(value, DECIMAL_PLACES),
            'Triplet': triplet_str,
            'Row_Index': row_idx,
            'Col_Index': col_idx
        })

    return pd.DataFrame(data)


def save_triplets(triplet_df, output_dir=None):
    """
    Save triplets to Excel file.

    Output file: mmde_triplet.xlsx
    Columns: Position, t_ij, Barrier_Pair, Value, Triplet, Row_Index, Col_Index

    Parameters:
    -----------
    triplet_df : pandas.DataFrame
        DataFrame with triplet information
    output_dir : str or Path, optional
        Output directory

    Returns:
    --------
    Path
        Path to the saved file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_ISM

    script_dir = get_script_directory()
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "mmde_triplet.xlsx"

    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            triplet_df.to_excel(writer, sheet_name='Sorted_Triplets', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )

    print(f"MMDE Triplets saved to: {file_path}")
    return file_path


def print_triplet_summary(trm, triplets, sorted_triplets, barriers):
    """
    Print summary of triplet creation.

    Parameters:
    -----------
    trm : pandas.DataFrame
        Original TRM
    triplets : list
        Unsorted triplets
    sorted_triplets : list
        Sorted triplets
    barriers : list
        List of barrier codes
    """
    n = len(barriers)

    print("-" * 70)
    print("MMDE TRIPLET CREATION SUMMARY")
    print("-" * 70)
    print(f"Matrix Size: {n} × {n}")
    print(f"Total Triplets Created: {len(triplets)} (n² = {n}² = {n*n})")
    print()

    print("TRIPLET FORMULA:")
    print("  For each element t_ij in TRM:")
    print("    triplet = (value, row_index, column_index)")
    print("  Using 1-based indexing")
    print()

    print("SORTING:")
    print("  Triplets sorted in DESCENDING order by value")
    print("  This creates the ordered set T*")
    print()

    # Show top 5 and bottom 5 triplets
    print("TOP 5 TRIPLETS (Highest Values):")
    for i, triplet in enumerate(sorted_triplets[:5], 1):
        value, row_idx, col_idx = triplet
        barrier_from = barriers[row_idx - 1].upper()
        barrier_to = barriers[col_idx - 1].upper()
        print(f"  {i}. ({value:.{DECIMAL_PLACES}f}, {row_idx}, {col_idx}) "
              f"- {barrier_from}→{barrier_to}")
    print()

    print("BOTTOM 5 TRIPLETS (Lowest Values):")
    for i, triplet in enumerate(sorted_triplets[-5:], len(sorted_triplets) - 4):
        value, row_idx, col_idx = triplet
        barrier_from = barriers[row_idx - 1].upper()
        barrier_to = barriers[col_idx - 1].upper()
        print(f"  {i}. ({value:.{DECIMAL_PLACES}f}, {row_idx}, {col_idx}) "
              f"- {barrier_from}→{barrier_to}")
    print()

    # Value statistics
    values = [t[0] for t in sorted_triplets]
    print("VALUE STATISTICS:")
    print(f"  Maximum: {max(values):.{DECIMAL_PLACES}f}")
    print(f"  Minimum: {min(values):.{DECIMAL_PLACES}f}")
    print(f"  Mean: {np.mean(values):.{DECIMAL_PLACES}f}")
    print(f"  Std Dev: {np.std(values):.{DECIMAL_PLACES}f}")
    print()

    print("-" * 70)


def create_mmde_triplets(trm=None, save=True):
    """
    Main function to create and sort MMDE triplets from TRM.

    This is Step 1 of the MMDE threshold calculation process.

    Parameters:
    -----------
    trm : pandas.DataFrame, optional
        Total Relation Matrix. If None, loads from file.
    save : bool, optional
        Whether to save outputs to Excel file.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'trm': Original Total Relation Matrix
        - 'triplets': Unsorted list of triplets
        - 'sorted_triplets': Sorted list of triplets (descending by value)
        - 'triplet_df': DataFrame with triplet information
        - 'barriers': List of barrier codes
        - 'n': Number of barriers
        - 'output_file': Path to saved file (if save=True)
    """
    print("\n" + "=" * 70)
    print("MODULE 7: MMDE TRIPLET CREATION")
    print("=" * 70 + "\n")

    # Step 1: Load or use provided TRM
    if trm is None:
        print("Loading Total Relation Matrix from Module 3...")
        trm = load_trm_from_file()
        print(f"Loaded TRM: {len(trm)} × {len(trm)} matrix.\n")
    else:
        print(f"Using provided TRM ({len(trm)} × {len(trm)} matrix).\n")

    # Get barriers
    barriers = list(trm.index)
    n = len(barriers)

    print(f"Barriers: {[b.upper() for b in barriers]}\n")

    # Step 2: Create triplets
    print("Creating triplets from TRM...")
    print("  Formula: triplet = (t_ij, row_index, column_index)")
    print("  Using 1-based indexing")
    triplets = create_triplets(trm)
    print(f"  Created {len(triplets)} triplets (n² = {n*n})\n")

    # Step 3: Sort triplets in descending order
    print("Sorting triplets in descending order by value...")
    sorted_triplets = sort_triplets_descending(triplets)
    print("  Triplets sorted. Created ordered set T*\n")

    # Step 4: Create DataFrame for export
    print("Creating DataFrame for export...")
    triplet_df = create_triplet_dataframe(sorted_triplets, barriers)
    print(f"  DataFrame created with {len(triplet_df)} rows\n")

    # Step 5: Print summary
    print_triplet_summary(trm, triplets, sorted_triplets, barriers)

    # Step 6: Save outputs
    output_file = None
    if save:
        print("\nSaving outputs...")
        output_file = save_triplets(triplet_df)

    # Prepare results
    results = {
        'trm': trm,
        'triplets': triplets,
        'sorted_triplets': sorted_triplets,
        'triplet_df': triplet_df,
        'barriers': barriers,
        'n': n,
        'output_file': output_file
    }

    print("\n" + "=" * 70)
    print("MODULE 7 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 7 in standalone mode...")
    results = create_mmde_triplets()

    # Display sample of sorted triplets
    print("\n" + "=" * 70)
    print("SORTED TRIPLETS (T*) - First 20 entries")
    print("=" * 70)
    print(results['triplet_df'].head(20).to_string(index=False))
