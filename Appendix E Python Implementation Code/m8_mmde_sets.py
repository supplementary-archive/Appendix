# SPDX-License-Identifier: PROPRIETARY
# File: m8_mmde_sets.py
# Purpose: Module 8 - MMDE Cumulative Sets (Td and Tr) for DEMATEL-ISM Integration

"""
MMDE (Maximum Mean De-Entropy) Threshold Calculation - Step 2

This module builds cumulative dispatch (Td) and receive (Tr) sets from the
sorted triplets. These sets accumulate row and column indices respectively.

Mathematical Definitions:
-------------------------
- Td (Dispatch Set): Cumulative list of row indices encountered
- Tr (Receive Set): Cumulative list of column indices encountered

Procedure:
----------
Initialize Td = [] and Tr = [] as empty lists

For each position i from 1 to n²:
    Get triplet T*[i] = (value, row, col)
    Append row to Td
    Append col to Tr
    Store Td_i and Tr_i (copies at position i)

IMPORTANT: Keep duplicate values in Td and Tr. Do NOT remove duplicates.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import from Module 1 and Module 7
from m1_data_processing import (
    BARRIER_NAMES,
    get_script_directory
)
from m7_mmde_triplet import (
    OUTPUT_DIR_ISM,
    DECIMAL_PLACES,
    create_mmde_triplets
)

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def build_cumulative_sets(sorted_triplets):
    """
    Build cumulative Td (dispatch) and Tr (receive) sets.

    Mathematical Procedure:
    -----------------------
    Initialize Td = [] and Tr = [] as empty lists

    For each position i from 1 to n²:
        Get triplet T*[i] = (value, row, col)
        Append row to Td
        Append col to Tr
        Store Td_i and Tr_i (copies at position i)

    IMPORTANT: Duplicates are KEPT in both Td and Tr.

    Parameters:
    -----------
    sorted_triplets : list
        Sorted list of triplets (value, row_index, column_index)
        Sorted in descending order by value

    Returns:
    --------
    dict
        Dictionary containing:
        - 'Td_cumulative': List of Td sets at each position
        - 'Tr_cumulative': List of Tr sets at each position
        - 'positions': List of position numbers (1-based)
        - 'triplets_at_position': List of triplets at each position
    """
    n_sq = len(sorted_triplets)

    # Initialize empty lists
    Td = []  # Cumulative dispatch set (row indices)
    Tr = []  # Cumulative receive set (column indices)

    # Storage for cumulative sets at each position
    Td_cumulative = []  # Td_i at each position
    Tr_cumulative = []  # Tr_i at each position
    positions = []
    triplets_at_position = []

    # Build cumulative sets
    for i in range(n_sq):
        # Get triplet at position i (0-indexed in Python, but 1-indexed in output)
        triplet = sorted_triplets[i]
        value, row, col = triplet

        # Append row to Td (dispatch set) - KEEP DUPLICATES
        Td.append(row)

        # Append col to Tr (receive set) - KEEP DUPLICATES
        Tr.append(col)

        # Store copies at this position
        Td_cumulative.append(Td.copy())  # Copy to preserve state at position i
        Tr_cumulative.append(Tr.copy())  # Copy to preserve state at position i
        positions.append(i + 1)  # 1-based position
        triplets_at_position.append(triplet)

    return {
        'Td_cumulative': Td_cumulative,
        'Tr_cumulative': Tr_cumulative,
        'positions': positions,
        'triplets_at_position': triplets_at_position
    }


def format_set_for_display(index_list, max_display=20):
    """
    Format a list of indices for display in Excel.

    Parameters:
    -----------
    index_list : list
        List of indices
    max_display : int
        Maximum number of elements to display before truncating

    Returns:
    --------
    str
        Formatted string representation of the list
    """
    if len(index_list) <= max_display:
        return str(index_list)
    else:
        # Show first few and indicate truncation
        shown = index_list[:max_display]
        return f"{shown}... (+{len(index_list) - max_display} more)"


def create_sets_dataframe(cumulative_data, barriers):
    """
    Create DataFrame for Excel export.

    Output columns:
    - Position (i)
    - Triplet
    - Value
    - Row_Index
    - Col_Index
    - Td (rows accumulated)
    - Tr (columns accumulated)
    - Td_Count
    - Tr_Count

    Parameters:
    -----------
    cumulative_data : dict
        Dictionary from build_cumulative_sets()
    barriers : list
        List of barrier codes

    Returns:
    --------
    pandas.DataFrame
        DataFrame with cumulative set information
    """
    data = []

    positions = cumulative_data['positions']
    triplets = cumulative_data['triplets_at_position']
    Td_cumulative = cumulative_data['Td_cumulative']
    Tr_cumulative = cumulative_data['Tr_cumulative']

    for i in range(len(positions)):
        position = positions[i]
        triplet = triplets[i]
        value, row_idx, col_idx = triplet
        Td_i = Td_cumulative[i]
        Tr_i = Tr_cumulative[i]

        # Create barrier pair notation
        barrier_from = barriers[row_idx - 1].upper()
        barrier_to = barriers[col_idx - 1].upper()

        # Format triplet string
        triplet_str = f"({value:.{DECIMAL_PLACES}f}, {row_idx}, {col_idx})"

        data.append({
            'Position_i': position,
            'Triplet': triplet_str,
            'Value': round(value, DECIMAL_PLACES),
            'Barrier_Pair': f"{barrier_from}→{barrier_to}",
            'Row_Index': row_idx,
            'Col_Index': col_idx,
            'Td_i': str(Td_i),
            'Tr_i': str(Tr_i),
            'Td_Count': len(Td_i),
            'Tr_Count': len(Tr_i)
        })

    return pd.DataFrame(data)


def create_summary_dataframe(cumulative_data, n):
    """
    Create summary statistics DataFrame.

    Parameters:
    -----------
    cumulative_data : dict
        Dictionary from build_cumulative_sets()
    n : int
        Number of barriers

    Returns:
    --------
    pandas.DataFrame
        Summary statistics
    """
    n_sq = n * n
    final_Td = cumulative_data['Td_cumulative'][-1]
    final_Tr = cumulative_data['Tr_cumulative'][-1]

    # Count unique factors at the end
    unique_Td = len(set(final_Td))
    unique_Tr = len(set(final_Tr))

    summary_data = {
        'Parameter': [
            'Number of Barriers (n)',
            'Total Triplets (n²)',
            'Final Td Length',
            'Final Tr Length',
            'Unique Factors in Td',
            'Unique Factors in Tr',
            'Duplicates Kept in Td',
            'Duplicates Kept in Tr'
        ],
        'Value': [
            n,
            n_sq,
            len(final_Td),
            len(final_Tr),
            unique_Td,
            unique_Tr,
            'Yes (as required)',
            'Yes (as required)'
        ]
    }

    return pd.DataFrame(summary_data)


def save_sets(sets_df, summary_df, output_dir=None):
    """
    Save cumulative sets to Excel file.

    Output file: mmde_sets.xlsx
    Sheets:
    - Cumulative_Sets: Main data with positions, triplets, Td, Tr
    - Summary: Summary statistics

    Parameters:
    -----------
    sets_df : pandas.DataFrame
        DataFrame with cumulative set information
    summary_df : pandas.DataFrame
        Summary statistics DataFrame
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

    file_path = output_path / "mmde_sets.xlsx"

    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            sets_df.to_excel(writer, sheet_name='Cumulative_Sets', index=False)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )

    print(f"MMDE Sets saved to: {file_path}")
    return file_path


def print_sets_summary(cumulative_data, barriers):
    """
    Print summary of cumulative sets construction.

    Parameters:
    -----------
    cumulative_data : dict
        Dictionary from build_cumulative_sets()
    barriers : list
        List of barrier codes
    """
    n = len(barriers)
    n_sq = n * n

    Td_cumulative = cumulative_data['Td_cumulative']
    Tr_cumulative = cumulative_data['Tr_cumulative']

    print("-" * 70)
    print("MMDE CUMULATIVE SETS SUMMARY")
    print("-" * 70)
    print(f"Number of Barriers (n): {n}")
    print(f"Total Positions (n²): {n_sq}")
    print()

    print("DEFINITIONS:")
    print("  Td (Dispatch Set): Cumulative list of row indices")
    print("  Tr (Receive Set): Cumulative list of column indices")
    print()

    print("PROCEDURE:")
    print("  For each position i from 1 to n²:")
    print("    - Get triplet T*[i] = (value, row, col)")
    print("    - Append row to Td")
    print("    - Append col to Tr")
    print("    - Store Td_i and Tr_i")
    print()
    print("  IMPORTANT: Duplicates are KEPT (not removed)")
    print()

    # Show first few positions
    print("FIRST 5 POSITIONS:")
    for i in range(min(5, n_sq)):
        triplet = cumulative_data['triplets_at_position'][i]
        value, row, col = triplet
        Td_i = Td_cumulative[i]
        Tr_i = Tr_cumulative[i]
        print(f"  Position {i+1}: triplet=({value:.4f}, {row}, {col})")
        print(f"    Td_{i+1} = {Td_i}")
        print(f"    Tr_{i+1} = {Tr_i}")
    print()

    # Show final state
    final_Td = Td_cumulative[-1]
    final_Tr = Tr_cumulative[-1]
    print(f"FINAL STATE (Position {n_sq}):")
    print(f"  Td length: {len(final_Td)} (with duplicates)")
    print(f"  Tr length: {len(final_Tr)} (with duplicates)")
    print(f"  Unique factors in Td: {len(set(final_Td))}")
    print(f"  Unique factors in Tr: {len(set(final_Tr))}")
    print()

    # Frequency distribution in final sets
    print("FREQUENCY IN FINAL Td (row indices):")
    from collections import Counter
    Td_freq = Counter(final_Td)
    for idx in sorted(Td_freq.keys()):
        barrier = barriers[idx - 1].upper()
        print(f"  {barrier} (index {idx}): {Td_freq[idx]} occurrences")
    print()

    print("FREQUENCY IN FINAL Tr (column indices):")
    Tr_freq = Counter(final_Tr)
    for idx in sorted(Tr_freq.keys()):
        barrier = barriers[idx - 1].upper()
        print(f"  {barrier} (index {idx}): {Tr_freq[idx]} occurrences")
    print()

    print("-" * 70)


def build_mmde_sets(sorted_triplets=None, barriers=None, trm_results=None, save=True):
    """
    Main function to build MMDE cumulative sets.

    This is Step 2 of the MMDE threshold calculation process.

    Parameters:
    -----------
    sorted_triplets : list, optional
        Sorted list of triplets. If None, runs Module 7 first.
    barriers : list, optional
        List of barrier codes. Required if sorted_triplets is provided.
    trm_results : dict, optional
        Results from Module 7. Alternative to providing sorted_triplets directly.
    save : bool, optional
        Whether to save outputs to Excel file.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'cumulative_data': Dictionary with Td and Tr cumulative sets
        - 'sets_df': DataFrame with set information
        - 'summary_df': Summary statistics DataFrame
        - 'sorted_triplets': The sorted triplets used
        - 'barriers': List of barrier codes
        - 'n': Number of barriers
        - 'output_file': Path to saved file (if save=True)
    """
    print("\n" + "=" * 70)
    print("MODULE 8: MMDE CUMULATIVE SETS (Td and Tr)")
    print("=" * 70 + "\n")

    # Step 1: Get sorted triplets
    if sorted_triplets is None:
        if trm_results is not None:
            sorted_triplets = trm_results['sorted_triplets']
            barriers = trm_results['barriers']
            print(f"Using sorted triplets from provided results.\n")
        else:
            print("Running Module 7 to get sorted triplets...")
            m7_results = create_mmde_triplets(save=False)
            sorted_triplets = m7_results['sorted_triplets']
            barriers = m7_results['barriers']
            print()

    if barriers is None:
        raise ValueError("barriers must be provided if sorted_triplets is provided directly")

    n = len(barriers)
    n_sq = len(sorted_triplets)

    print(f"Number of barriers: {n}")
    print(f"Number of triplets: {n_sq}\n")

    # Step 2: Build cumulative sets
    print("Building cumulative Td and Tr sets...")
    print("  - Td: Accumulating row indices (dispatch)")
    print("  - Tr: Accumulating column indices (receive)")
    print("  - Keeping duplicates as required")
    cumulative_data = build_cumulative_sets(sorted_triplets)
    print(f"  Cumulative sets built for {n_sq} positions.\n")

    # Step 3: Create DataFrames
    print("Creating DataFrames for export...")
    sets_df = create_sets_dataframe(cumulative_data, barriers)
    summary_df = create_summary_dataframe(cumulative_data, n)
    print(f"  DataFrames created.\n")

    # Step 4: Print summary
    print_sets_summary(cumulative_data, barriers)

    # Step 5: Save outputs
    output_file = None
    if save:
        print("\nSaving outputs...")
        output_file = save_sets(sets_df, summary_df)

    # Prepare results
    results = {
        'cumulative_data': cumulative_data,
        'sets_df': sets_df,
        'summary_df': summary_df,
        'sorted_triplets': sorted_triplets,
        'barriers': barriers,
        'n': n,
        'output_file': output_file
    }

    print("\n" + "=" * 70)
    print("MODULE 8 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 8 in standalone mode...")
    results = build_mmde_sets()

    # Display sample of cumulative sets
    print("\n" + "=" * 70)
    print("CUMULATIVE SETS - First 10 positions")
    print("=" * 70)
    display_cols = ['Position_i', 'Triplet', 'Td_i', 'Tr_i']
    print(results['sets_df'][display_cols].head(10).to_string(index=False))
