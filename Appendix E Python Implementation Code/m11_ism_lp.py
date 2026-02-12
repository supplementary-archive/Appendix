# SPDX-License-Identifier: PROPRIETARY
# File: m11_ism_lp.py
# Purpose: Module 11 - ISM Level Partitioning for DEMATEL-ISM Integration

"""
ISM Level Partitioning

This module partitions factors into hierarchical levels based on the
Final Reachability Matrix using set theory operations.

Mathematical Definitions:
-------------------------

1. REACHABILITY SET R(i):
   R(i) = {j : r_ij = 1}
   The set of all factors j that factor i can reach (including itself).
   Implementation: All column indices where row i has value 1.

2. ANTECEDENT SET A(i):
   A(i) = {j : r_ji = 1}
   The set of all factors j that can reach factor i (including itself).
   Implementation: All row indices where column i has value 1.

3. INTERSECTION SET C(i):
   C(i) = R(i) ∩ A(i)
   The set of factors common to both reachability and antecedent sets.

4. LEVEL IDENTIFICATION RULE:
   Factor i is at current level IF R(i) = C(i)
   Meaning: All factors that i can reach are also factors that can reach i.
   These are top-level factors that do not influence any remaining factors.

5. LEVEL PARTITIONING ALGORITHM:
   - Level 1 factors are removed first (they are effects/outcomes)
   - Higher level numbers indicate root causes
   - Arrows point from higher levels to lower levels (causes to effects)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import from previous modules
from m1_data_processing import (
    BARRIER_NAMES,
    get_script_directory
)
from m7_mmde_triplet import OUTPUT_DIR_ISM
from m10_ism_frm import create_reachability_matrix

# =============================================================================
# SET CALCULATION FUNCTIONS
# =============================================================================

def get_reachability_set(reachability_matrix, factor_index):
    """
    Calculate Reachability Set R(i) for a factor.

    Formula:
    --------
    R(i) = {j : r_ij = 1}

    All column indices j where row i has value 1.
    Factor i can reach all factors in R(i).

    Parameters:
    -----------
    reachability_matrix : numpy.ndarray
        Final Reachability Matrix
    factor_index : int
        0-based index of the factor

    Returns:
    --------
    set
        Set of factor indices (0-based) that factor i can reach
    """
    row = reachability_matrix[factor_index, :]
    reachability_set = set(np.where(row == 1)[0])
    return reachability_set


def get_antecedent_set(reachability_matrix, factor_index):
    """
    Calculate Antecedent Set A(i) for a factor.

    Formula:
    --------
    A(i) = {j : r_ji = 1}

    All row indices j where column i has value 1.
    All factors in A(i) can reach factor i.

    Parameters:
    -----------
    reachability_matrix : numpy.ndarray
        Final Reachability Matrix
    factor_index : int
        0-based index of the factor

    Returns:
    --------
    set
        Set of factor indices (0-based) that can reach factor i
    """
    col = reachability_matrix[:, factor_index]
    antecedent_set = set(np.where(col == 1)[0])
    return antecedent_set


def get_intersection_set(reachability_set, antecedent_set):
    """
    Calculate Intersection Set C(i).

    Formula:
    --------
    C(i) = R(i) ∩ A(i)

    Parameters:
    -----------
    reachability_set : set
        Reachability set R(i)
    antecedent_set : set
        Antecedent set A(i)

    Returns:
    --------
    set
        Intersection of R(i) and A(i)
    """
    return reachability_set & antecedent_set


# =============================================================================
# LEVEL PARTITIONING ALGORITHM
# =============================================================================

def partition_levels(reachability_matrix):
    """
    Partition factors into hierarchical levels.

    Algorithm:
    ----------
    remaining = {0, 1, ..., n-1}  (all factor indices)
    levels = {}
    level_number = 1

    WHILE remaining is not empty:
        current_level_factors = []

        FOR each factor i in remaining:
            R_i = R(i) ∩ remaining  (only consider remaining factors)
            A_i = A(i) ∩ remaining  (only consider remaining factors)
            C_i = R_i ∩ A_i

            IF R_i == C_i:
                Add i to current_level_factors

        Store: levels[level_number] = current_level_factors
        Remove current_level_factors from remaining
        level_number = level_number + 1

    RETURN levels

    Important Notes:
    ----------------
    - Level 1 factors are removed first (they are effects/outcomes)
    - Higher level numbers indicate root causes
    - When computing R(i) and A(i), only consider factors still in remaining set

    Parameters:
    -----------
    reachability_matrix : numpy.ndarray
        Final Reachability Matrix

    Returns:
    --------
    dict
        Dictionary containing:
        - 'levels': Dict mapping level number to list of factor indices
        - 'factor_levels': Dict mapping factor index to its level
        - 'iterations': List of iteration data for each level
    """
    n = len(reachability_matrix)

    # Initialize
    remaining = set(range(n))  # All factor indices (0-based)
    levels = {}
    factor_levels = {}
    iterations = []
    level_number = 1

    while remaining:
        current_level_factors = []
        iteration_data = {
            'level': level_number,
            'remaining_factors': remaining.copy(),
            'factor_sets': []
        }

        # For each factor in remaining set
        for i in sorted(remaining):
            # Get full sets from the FRM
            full_R_i = get_reachability_set(reachability_matrix, i)
            full_A_i = get_antecedent_set(reachability_matrix, i)

            # Restrict to remaining factors only
            R_i = full_R_i & remaining
            A_i = full_A_i & remaining

            # Calculate intersection
            C_i = get_intersection_set(R_i, A_i)

            # Store data for this factor
            factor_data = {
                'factor_index': i,
                'R_i': R_i.copy(),
                'A_i': A_i.copy(),
                'C_i': C_i.copy(),
                'is_level': R_i == C_i
            }
            iteration_data['factor_sets'].append(factor_data)

            # Check level identification rule: R(i) = C(i)
            if R_i == C_i:
                current_level_factors.append(i)

        # Store level assignment
        levels[level_number] = current_level_factors.copy()
        for factor in current_level_factors:
            factor_levels[factor] = level_number

        iteration_data['level_factors'] = current_level_factors.copy()
        iterations.append(iteration_data)

        # Remove assigned factors from remaining
        remaining -= set(current_level_factors)
        level_number += 1

        # Safety check to prevent infinite loop
        if level_number > n + 1:
            raise RuntimeError("Level partitioning failed to converge")

    return {
        'levels': levels,
        'factor_levels': factor_levels,
        'iterations': iterations,
        'max_level': level_number - 1
    }


# =============================================================================
# DATAFRAME CREATION FUNCTIONS
# =============================================================================

def format_set_for_display(index_set, barriers):
    """
    Format a set of indices for display.

    Parameters:
    -----------
    index_set : set
        Set of factor indices (0-based)
    barriers : list
        List of barrier codes

    Returns:
    --------
    str
        Formatted string representation
    """
    if not index_set:
        return ""

    codes = sorted([barriers[i].upper() for i in index_set])

    return ", ".join(codes)


def create_iteration_dataframe(iteration_data, barriers, barrier_names=None):
    """
    Create DataFrame for a single level partitioning iteration.

    Columns:
    - Barrier (Code:Name)
    - Reachability Set R(i)
    - Antecedent Set A(i)
    - Intersection Set C(i)
    - Level (if assigned at this iteration)

    Parameters:
    -----------
    iteration_data : dict
        Data for this iteration from partition_levels()
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        DataFrame for this iteration
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    level = iteration_data['level']
    level_factors = set(iteration_data['level_factors'])

    data = []
    for factor_data in iteration_data['factor_sets']:
        i = factor_data['factor_index']
        barrier_code = barriers[i]

        # Create barrier label (Code: Name)
        barrier_label = f"{barrier_code.upper()}: {barrier_names.get(barrier_code, barrier_code.upper()).split(': ', 1)[-1]}"

        # Format sets
        R_i_str = format_set_for_display(factor_data['R_i'], barriers)
        A_i_str = format_set_for_display(factor_data['A_i'], barriers)
        C_i_str = format_set_for_display(factor_data['C_i'], barriers)

        # Level assignment
        level_str = str(level) if i in level_factors else ""

        data.append({
            'Barrier': barrier_label,
            'Reachability_Set_R(i)': R_i_str,
            'Antecedent_Set_A(i)': A_i_str,
            'Intersection_C(i)': C_i_str,
            'Level': level_str
        })

    return pd.DataFrame(data)


def create_final_levels_dataframe(partition_result, barriers, barrier_names=None):
    """
    Create final summary DataFrame with all level assignments.

    Parameters:
    -----------
    partition_result : dict
        Result from partition_levels()
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        Summary DataFrame with level assignments
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    factor_levels = partition_result['factor_levels']

    data = []
    for i in range(len(barriers)):
        barrier_code = barriers[i]
        level = factor_levels.get(i, 0)

        data.append({
            'Barrier_Code': barrier_code.upper(),
            'Barrier_Name': barrier_names.get(barrier_code, barrier_code.upper()),
            'Level': level
        })

    # Sort by level, then by barrier code
    df = pd.DataFrame(data)
    df = df.sort_values(['Level', 'Barrier_Code']).reset_index(drop=True)

    return df


def create_comprehensive_final_dataframe(frm, partition_result, barriers, barrier_names=None):
    """
    Create comprehensive final DataFrame with barrier info, all sets from original FRM, and levels.

    This creates a single summary with:
    - Barrier (Code: Name)
    - Reachability Set R(i) from original FRM
    - Antecedent Set A(i) from original FRM
    - Intersection Set C(i) from original FRM
    - Level assignment

    Parameters:
    -----------
    frm : numpy.ndarray
        Final Reachability Matrix (original, before level partitioning reductions)
    partition_result : dict
        Result from partition_levels()
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        Comprehensive summary DataFrame
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    factor_levels = partition_result['factor_levels']

    data = []
    for i in range(len(barriers)):
        barrier_code = barriers[i]
        level = factor_levels.get(i, 0)

        # Get sets from the original FRM (not reduced by iterations)
        R_i = get_reachability_set(frm, i)
        A_i = get_antecedent_set(frm, i)
        C_i = get_intersection_set(R_i, A_i)

        # Format barrier label (Code: Name)
        barrier_label = f"{barrier_code.upper()}: {barrier_names.get(barrier_code, barrier_code.upper()).split(': ', 1)[-1]}"

        # Format sets for display
        R_i_str = format_set_for_display(R_i, barriers)
        A_i_str = format_set_for_display(A_i, barriers)
        C_i_str = format_set_for_display(C_i, barriers)

        data.append({
            'Barrier': barrier_label,
            'Reachability_Set_R(i)': R_i_str,
            'Antecedent_Set_A(i)': A_i_str,
            'Intersection_C(i)': C_i_str,
            'Level': level
        })

    df = pd.DataFrame(data)
    return df


def create_level_summary_dataframe(partition_result, barriers, barrier_names=None):
    """
    Create summary showing factors at each level.

    Parameters:
    -----------
    partition_result : dict
        Result from partition_levels()
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        Level summary DataFrame
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    levels = partition_result['levels']

    data = []
    for level_num in sorted(levels.keys()):
        factors = levels[level_num]
        factor_codes = [barriers[i].upper() for i in factors]
        factor_names = [barrier_names.get(barriers[i], barriers[i].upper()) for i in factors]

        data.append({
            'Level': level_num,
            'Number_of_Factors': len(factors),
            'Factor_Codes': ", ".join(factor_codes),
            'Factor_Names': "; ".join(factor_names)
        })

    return pd.DataFrame(data)


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_level_partitioning(iterations_dfs, final_levels_df, level_summary_df,
                             comprehensive_df=None, output_dir=None):
    """
    Save level partitioning results to Excel files.

    Output files:
    - ism_lp_1.xlsx, ism_lp_2.xlsx, ...: One file per iteration
    - ism_lp_final.xlsx: Final summary with all levels and comprehensive data

    Parameters:
    -----------
    iterations_dfs : list
        List of DataFrames, one per iteration
    final_levels_df : pandas.DataFrame
        Final levels DataFrame
    level_summary_df : pandas.DataFrame
        Level summary DataFrame
    comprehensive_df : pandas.DataFrame, optional
        Comprehensive DataFrame with barrier, sets, and levels
    output_dir : str or Path, optional
        Output directory

    Returns:
    --------
    list
        Paths to saved files
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_ISM

    script_dir = get_script_directory()
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # Save iteration files
    for i, df in enumerate(iterations_dfs, 1):
        file_path = output_path / f"ism_lp_{i}.xlsx"
        try:
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name=f'Level_{i}_Iteration', index=False)
        except PermissionError:
            raise PermissionError(
                f"Cannot write to file: {file_path}\n"
                "Please close the Excel file if it's open and try again."
            )
        saved_files.append(file_path)
        print(f"Level {i} iteration saved to: {file_path}")

    # Save final summary
    final_file = output_path / "ism_lp_final.xlsx"
    try:
        with pd.ExcelWriter(final_file, engine='openpyxl') as writer:
            # Save comprehensive summary as first sheet (main output)
            if comprehensive_df is not None:
                comprehensive_df.to_excel(writer, sheet_name='Comprehensive_Summary', index=False)
            final_levels_df.to_excel(writer, sheet_name='Factor_Levels', index=False)
            level_summary_df.to_excel(writer, sheet_name='Level_Summary', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {final_file}\n"
            "Please close the Excel file if it's open and try again."
        )
    saved_files.append(final_file)
    print(f"Final level summary saved to: {final_file}")

    return saved_files


def print_level_partitioning_summary(partition_result, barriers, barrier_names=None):
    """
    Print summary of level partitioning results.

    Parameters:
    -----------
    partition_result : dict
        Result from partition_levels()
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    levels = partition_result['levels']
    max_level = partition_result['max_level']

    print("-" * 70)
    print("ISM LEVEL PARTITIONING SUMMARY")
    print("-" * 70)
    print(f"Number of Barriers: {len(barriers)}")
    print(f"Number of Levels: {max_level}")
    print()

    print("SET DEFINITIONS:")
    print("  R(i) = Reachability Set = {j : r_ij = 1}")
    print("        (factors that i can reach)")
    print("  A(i) = Antecedent Set = {j : r_ji = 1}")
    print("        (factors that can reach i)")
    print("  C(i) = R(i) ∩ A(i)")
    print("        (intersection)")
    print()

    print("LEVEL IDENTIFICATION RULE:")
    print("  Factor i is at current level IF R(i) = C(i)")
    print()

    print("HIERARCHICAL LEVELS:")
    print("  (Level 1 = Top/Effects, Higher Levels = Root Causes)")
    print()

    for level_num in sorted(levels.keys()):
        factors = levels[level_num]
        print(f"  Level {level_num}:")
        for i in factors:
            barrier_code = barriers[i].upper()
            barrier_name = barrier_names.get(barriers[i], barriers[i].upper())
            print(f"    - {barrier_name}")
        print()

    print("INTERPRETATION:")
    print("  - Level 1 factors are EFFECTS (dependent on other factors)")
    print("  - Highest level factors are ROOT CAUSES (drivers)")
    print("  - Arrows in digraph point from higher to lower levels")
    print()

    print("-" * 70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def perform_level_partitioning(frm=None, barriers=None, frm_results=None,
                                 barrier_names=None, save=True):
    """
    Main function to perform ISM level partitioning.

    Parameters:
    -----------
    frm : numpy.ndarray, optional
        Final Reachability Matrix. If None, runs Module 10 first.
    barriers : list, optional
        List of barrier codes.
    frm_results : dict, optional
        Results from Module 10 (alternative to providing frm directly).
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names.
    save : bool, optional
        Whether to save outputs to Excel files.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'partition_result': Full partition result from partition_levels()
        - 'levels': Dict mapping level number to factor indices
        - 'factor_levels': Dict mapping factor index to level
        - 'max_level': Maximum level number
        - 'iterations_dfs': List of DataFrames for each iteration
        - 'final_levels_df': Final levels DataFrame
        - 'level_summary_df': Level summary DataFrame
        - 'comprehensive_df': Comprehensive DataFrame with all barriers, sets, and levels
        - 'barriers': List of barrier codes
        - 'frm': Final Reachability Matrix used
        - 'output_files': Paths to saved files (if save=True)
    """
    print("\n" + "=" * 70)
    print("MODULE 11: ISM LEVEL PARTITIONING")
    print("=" * 70 + "\n")

    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    # Step 1: Get FRM and barriers
    if frm is None or barriers is None:
        if frm_results is not None:
            frm = frm_results['frm']
            barriers = frm_results['barriers']
            print("Using FRM from provided Module 10 results.\n")
        else:
            print("Running Module 10 to get Final Reachability Matrix...")
            frm_results = create_reachability_matrix(save=False)
            frm = frm_results['frm']
            barriers = frm_results['barriers']
            print()

    n = len(barriers)
    print(f"Number of barriers: {n}")
    print(f"Barriers: {[b.upper() for b in barriers]}\n")

    # Step 2: Perform level partitioning
    print("Performing level partitioning...")
    print("  Using iterative algorithm with R(i) = C(i) rule")
    partition_result = partition_levels(frm)
    print(f"  Partitioning complete. {partition_result['max_level']} levels found.\n")

    # Step 3: Create DataFrames for each iteration
    print("Creating DataFrames for each iteration...")
    iterations_dfs = []
    for iteration_data in partition_result['iterations']:
        df = create_iteration_dataframe(iteration_data, barriers, barrier_names)
        iterations_dfs.append(df)
    print(f"  Created {len(iterations_dfs)} iteration DataFrames.\n")

    # Step 4: Create final summary DataFrames
    print("Creating final summary DataFrames...")
    final_levels_df = create_final_levels_dataframe(partition_result, barriers, barrier_names)
    level_summary_df = create_level_summary_dataframe(partition_result, barriers, barrier_names)
    comprehensive_df = create_comprehensive_final_dataframe(frm, partition_result, barriers, barrier_names)
    print("  Summary DataFrames created.\n")

    # Step 5: Print summary
    print_level_partitioning_summary(partition_result, barriers, barrier_names)

    # Step 6: Save outputs
    output_files = None
    if save:
        print("\nSaving outputs...")
        output_files = save_level_partitioning(iterations_dfs, final_levels_df, level_summary_df, comprehensive_df)

    # Prepare results
    results = {
        'partition_result': partition_result,
        'levels': partition_result['levels'],
        'factor_levels': partition_result['factor_levels'],
        'max_level': partition_result['max_level'],
        'iterations_dfs': iterations_dfs,
        'final_levels_df': final_levels_df,
        'level_summary_df': level_summary_df,
        'comprehensive_df': comprehensive_df,
        'barriers': barriers,
        'barrier_names': barrier_names,
        'frm': frm,
        'n': n,
        'output_files': output_files
    }

    print("\n" + "=" * 70)
    print("MODULE 11 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 11 in standalone mode...")
    results = perform_level_partitioning()

    # Display level summary
    print("\n" + "=" * 70)
    print("LEVEL SUMMARY")
    print("=" * 70)
    print(results['level_summary_df'].to_string(index=False))

    print("\n" + "=" * 70)
    print("FACTOR LEVELS")
    print("=" * 70)
    print(results['final_levels_df'].to_string(index=False))
