# SPDX-License-Identifier: PROPRIETARY
# File: m10_ism_frm.py
# Purpose: Module 10 - ISM Initial and Final Reachability Matrix for DEMATEL-ISM

"""
ISM Reachability Matrix Construction

This module converts the TRM to a binary matrix using the MMDE threshold,
applies transitivity to create the Final Reachability Matrix (FRM),
and calculates driving and dependence power.

Mathematical Procedures:
------------------------

1. BINARY MATRIX CONVERSION:
   k_ij = 1  if t_ij >= λ (threshold)
   k_ij = 0  if t_ij < λ

2. SET DIAGONAL ELEMENTS (Reflexivity):
   k_ii = 1  for all i = 1 to n

3. TRANSITIVITY RULE:
   If k_ij = 1 AND k_jk = 1, then k_ik must = 1

4. BOOLEAN OPERATIONS:
   Boolean OR:  0+0=0, 0+1=1, 1+0=1, 1+1=1
   Boolean AND: 0×0=0, 0×1=0, 1×0=0, 1×1=1

5. BOOLEAN MATRIX MULTIPLICATION:
   C = A ⊗ B
   c_ij = 1 if there exists any k where a_ik = 1 AND b_kj = 1

6. CONVERGENCE ALGORITHM:
   K^(m+1) = K^(m) ⊕ (K^(m) ⊗ K^(m))
   Where ⊕ is Boolean OR (element-wise), ⊗ is Boolean matrix multiplication
   Converges within n-1 iterations for n factors.

7. DRIVING POWER:
   DP(i) = Σ r_ij for j = 1 to n  (sum of row i)

8. DEPENDENCE POWER:
   DEP(i) = Σ r_ji for j = 1 to n  (sum of column i)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import from previous modules
from m1_data_processing import (
    BARRIER_NAMES,
    OUTPUT_DIR_DEMATEL,
    get_script_directory
)
from m7_mmde_triplet import OUTPUT_DIR_ISM, DECIMAL_PLACES, load_trm_from_file
from m9_mmde_final import calculate_mmde_threshold

# =============================================================================
# BINARY MATRIX CONVERSION
# =============================================================================

def apply_threshold(trm, threshold):
    """
    Convert TRM to binary matrix by applying threshold.

    Formula:
    --------
    k_ij = 1  if t_ij >= λ
    k_ij = 0  if t_ij < λ

    Parameters:
    -----------
    trm : pandas.DataFrame or numpy.ndarray
        Total Relation Matrix
    threshold : float
        MMDE threshold value λ

    Returns:
    --------
    numpy.ndarray
        Binary matrix K with values {0, 1}
    """
    if isinstance(trm, pd.DataFrame):
        matrix = trm.values.astype(float)
    else:
        matrix = trm.astype(float)

    # Apply threshold: 1 if >= threshold, else 0
    binary_matrix = np.where(matrix >= threshold, 1, 0)

    return binary_matrix.astype(int)


def set_diagonal_to_one(binary_matrix):
    """
    Set all diagonal elements to 1 (reflexivity).

    Formula:
    --------
    k_ii = 1  for all i = 1 to n

    This represents that each factor can reach itself.

    Parameters:
    -----------
    binary_matrix : numpy.ndarray
        Binary matrix K

    Returns:
    --------
    numpy.ndarray
        Binary matrix with diagonal set to 1
    """
    n = len(binary_matrix)
    result = binary_matrix.copy()

    for i in range(n):
        result[i, i] = 1

    return result


def create_initial_reachability_matrix(trm, threshold):
    """
    Create Initial Reachability Matrix (IRM) from TRM.

    Steps:
    1. Apply threshold to create binary matrix
    2. Set diagonal elements to 1

    Parameters:
    -----------
    trm : pandas.DataFrame
        Total Relation Matrix
    threshold : float
        MMDE threshold value

    Returns:
    --------
    numpy.ndarray
        Initial Reachability Matrix
    """
    # Step 1: Apply threshold
    binary_matrix = apply_threshold(trm, threshold)

    # Step 2: Set diagonal to 1
    irm = set_diagonal_to_one(binary_matrix)

    return irm


# =============================================================================
# BOOLEAN MATRIX OPERATIONS
# =============================================================================

def boolean_or(a, b):
    """
    Element-wise Boolean OR operation.

    Boolean OR:
    0 + 0 = 0
    0 + 1 = 1
    1 + 0 = 1
    1 + 1 = 1

    Parameters:
    -----------
    a, b : numpy.ndarray
        Binary matrices of same shape

    Returns:
    --------
    numpy.ndarray
        Result of element-wise Boolean OR
    """
    return np.logical_or(a, b).astype(int)


def boolean_matrix_multiply(A, B):
    """
    Boolean matrix multiplication.

    Formula:
    --------
    C = A ⊗ B
    c_ij = (a_i1 × b_1j) + (a_i2 × b_2j) + ... + (a_in × b_nj)

    Using Boolean arithmetic:
    c_ij = 1 if there exists any k where a_ik = 1 AND b_kj = 1

    Parameters:
    -----------
    A, B : numpy.ndarray
        Binary matrices of size n×n

    Returns:
    --------
    numpy.ndarray
        Result of Boolean matrix multiplication
    """
    n = len(A)
    C = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            # c_ij = 1 if there exists any k where A[i,k]=1 AND B[k,j]=1
            for k in range(n):
                if A[i, k] == 1 and B[k, j] == 1:
                    C[i, j] = 1
                    break  # No need to continue once we find one

    return C


# =============================================================================
# TRANSITIVITY CHECK
# =============================================================================

def transitivity_check(binary_matrix, max_iterations=None):
    """
    Apply transitivity check to ensure reachability matrix is transitive.

    Convergence Algorithm:
    ----------------------
    K^(m+1) = K^(m) ⊕ (K^(m) ⊗ K^(m))

    Where:
    - ⊕ is Boolean OR (element-wise)
    - ⊗ is Boolean matrix multiplication

    The algorithm converges when K^(m+1) = K^(m).
    Convergence is guaranteed within n-1 iterations.

    Transitivity Rule:
    If k_ij = 1 AND k_jk = 1, then k_ik must = 1

    Parameters:
    -----------
    binary_matrix : numpy.ndarray
        Initial binary matrix K
    max_iterations : int, optional
        Maximum iterations. Defaults to n (number of factors).

    Returns:
    --------
    dict
        Dictionary containing:
        - 'frm': Final Reachability Matrix
        - 'iterations': Number of iterations to converge
        - 'converged': Boolean indicating if converged
        - 'transitivity_entries': List of (i,j) pairs changed to 1
    """
    n = len(binary_matrix)

    if max_iterations is None:
        max_iterations = n  # Guaranteed to converge within n iterations

    current = binary_matrix.copy()
    transitivity_entries = []

    iteration = 0
    converged = False

    while iteration < max_iterations:
        # K^(m+1) = K^(m) ⊕ (K^(m) ⊗ K^(m))
        product = boolean_matrix_multiply(current, current)
        next_matrix = boolean_or(current, product)

        # Check for convergence
        if np.array_equal(next_matrix, current):
            converged = True
            break

        # Track which entries were changed (transitivity entries)
        changed = np.where((next_matrix == 1) & (current == 0))
        for i, j in zip(changed[0], changed[1]):
            transitivity_entries.append((i + 1, j + 1))  # 1-based indexing

        current = next_matrix
        iteration += 1

    return {
        'frm': current,
        'iterations': iteration,
        'converged': converged,
        'transitivity_entries': transitivity_entries
    }


# =============================================================================
# POWER CALCULATIONS
# =============================================================================

def calculate_driving_power(reachability_matrix):
    """
    Calculate Driving Power for each factor.

    Formula:
    --------
    DP(i) = Σ r_ij for j = 1 to n

    Driving power is the sum of all elements in row i.
    It represents the number of factors that factor i can reach (including itself).

    Parameters:
    -----------
    reachability_matrix : numpy.ndarray
        Final Reachability Matrix

    Returns:
    --------
    numpy.ndarray
        Array of driving power values for each factor
    """
    # Sum along rows (axis=1)
    driving_power = np.sum(reachability_matrix, axis=1)
    return driving_power


def calculate_dependence_power(reachability_matrix):
    """
    Calculate Dependence Power for each factor.

    Formula:
    --------
    DEP(i) = Σ r_ji for j = 1 to n

    Dependence power is the sum of all elements in column i.
    It represents the number of factors that can reach factor i (including itself).

    Parameters:
    -----------
    reachability_matrix : numpy.ndarray
        Final Reachability Matrix

    Returns:
    --------
    numpy.ndarray
        Array of dependence power values for each factor
    """
    # Sum along columns (axis=0)
    dependence_power = np.sum(reachability_matrix, axis=0)
    return dependence_power


# =============================================================================
# DATAFRAME CREATION
# =============================================================================

def create_irm_dataframe(irm, barriers, barrier_names=None):
    """
    Create DataFrame for Initial Reachability Matrix export.

    Format:
    - First column (rows): "B1 - BarrierName" format
    - First row (columns): Only barrier codes (B1, B2, etc.)

    Parameters:
    -----------
    irm : numpy.ndarray
        Initial Reachability Matrix
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        IRM with row labels as "B1 - Name" and column labels as "B1"
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    # Row labels: "B1 - BarrierName" format
    row_labels = [f"{b.upper()} - {barrier_names.get(b, b.upper())}" for b in barriers]

    # Column labels: Only barrier codes (B1, B2, etc.)
    col_labels = [b.upper() for b in barriers]

    df = pd.DataFrame(irm, index=row_labels, columns=col_labels)

    return df


def create_frm_dataframe(frm, driving_power, dependence_power, barriers,
                          barrier_names=None, transitivity_entries=None):
    """
    Create DataFrame for Final Reachability Matrix export.

    Format:
    - First column (rows): "B1 - BarrierName" format
    - First row (columns): Only barrier codes (B1, B2, etc.)
    - Last column shows Driving Power for each barrier
    - Last row shows Dependence Power for each barrier
    - Transitivity entries marked with "1*"
    - Corner cell shows total (sum of all driving power = sum of all dependence power)

    Parameters:
    -----------
    frm : numpy.ndarray
        Final Reachability Matrix
    driving_power : numpy.ndarray
        Driving power for each factor
    dependence_power : numpy.ndarray
        Dependence power for each factor
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    transitivity_entries : list, optional
        List of (row, col) tuples (1-based) for entries added by transitivity

    Returns:
    --------
    pandas.DataFrame
        FRM with powers and transitivity marked with "1*"
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    if transitivity_entries is None:
        transitivity_entries = []

    n = len(barriers)

    # Row labels: "B1 - BarrierName" format
    row_labels = [f"{b.upper()} - {barrier_names.get(b, b.upper())}" for b in barriers]

    # Column labels: Only barrier codes (B1, B2, etc.) + Driving Power
    col_labels = [b.upper() for b in barriers] + ['Driving Power']

    # Create set of transitivity positions for quick lookup (convert to 0-based)
    transitivity_set = {(r - 1, c - 1) for r, c in transitivity_entries}

    # Create string matrix to allow "1*" marking
    frm_str = np.empty((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            if frm[i, j] == 1:
                if (i, j) in transitivity_set:
                    frm_str[i, j] = "1*"  # Mark transitivity entry
                else:
                    frm_str[i, j] = "1"
            else:
                frm_str[i, j] = "0"

    # Add driving power column (as integers)
    dp_col = [[str(int(dp))] for dp in driving_power]
    frm_with_dp = np.column_stack([frm_str, [str(int(dp)) for dp in driving_power]])

    # Create DataFrame
    df = pd.DataFrame(frm_with_dp, index=row_labels, columns=col_labels)

    # Add dependence power row
    dep_row = [str(int(dep)) for dep in dependence_power] + [str(int(np.sum(driving_power)))]
    df.loc['Dependence Power'] = dep_row

    return df


def create_transitivity_dataframe(transitivity_entries, barriers, barrier_names=None):
    """
    Create DataFrame showing transitivity entries added.

    Parameters:
    -----------
    transitivity_entries : list
        List of (row, col) tuples for entries changed to 1
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        DataFrame with transitivity entries
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    if len(transitivity_entries) == 0:
        return pd.DataFrame({'Message': ['No transitivity entries needed']})

    data = []
    for row_idx, col_idx in transitivity_entries:
        barrier_from = barriers[row_idx - 1]
        barrier_to = barriers[col_idx - 1]

        data.append({
            'Row_Index': row_idx,
            'Col_Index': col_idx,
            'From_Barrier': barrier_names.get(barrier_from, barrier_from.upper()),
            'To_Barrier': barrier_names.get(barrier_to, barrier_to.upper()),
            'Entry': f"{barrier_from.upper()}→{barrier_to.upper()}"
        })

    return pd.DataFrame(data)


def create_power_summary_dataframe(driving_power, dependence_power, barriers, barrier_names=None):
    """
    Create summary DataFrame with driving and dependence power.

    Parameters:
    -----------
    driving_power : numpy.ndarray
        Driving power for each factor
    dependence_power : numpy.ndarray
        Dependence power for each factor
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        Power summary DataFrame
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    data = []
    for i, b in enumerate(barriers):
        data.append({
            'Barrier_Code': b.upper(),
            'Barrier_Name': barrier_names.get(b, b.upper()),
            'Driving_Power': int(driving_power[i]),
            'Dependence_Power': int(dependence_power[i])
        })

    return pd.DataFrame(data)


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_reachability_matrices(irm_df, frm_df, transitivity_df, power_df,
                                threshold, iterations, output_dir=None):
    """
    Save Initial and Final Reachability Matrices to Excel files.

    Output files:
    1. ism_irm.xlsx: Initial Reachability Matrix (before transitivity)
    2. ism_frm.xlsx: Final Reachability Matrix (with transitivity and powers)

    Parameters:
    -----------
    irm_df : pandas.DataFrame
        Initial Reachability Matrix DataFrame
    frm_df : pandas.DataFrame
        Final Reachability Matrix DataFrame
    transitivity_df : pandas.DataFrame
        Transitivity entries DataFrame
    power_df : pandas.DataFrame
        Power summary DataFrame
    threshold : float
        MMDE threshold used
    iterations : int
        Number of iterations to converge
    output_dir : str or Path, optional
        Output directory

    Returns:
    --------
    tuple
        Paths to saved files
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_ISM

    script_dir = get_script_directory()
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Create info DataFrame
    info_data = {
        'Parameter': [
            'Threshold (λ)',
            'Threshold Rule',
            'Diagonal Rule',
            'Transitivity Iterations',
            'Convergence'
        ],
        'Value': [
            f"{threshold:.{DECIMAL_PLACES}f}",
            f"k_ij = 1 if t_ij >= {threshold:.{DECIMAL_PLACES}f}, else 0",
            "k_ii = 1 for all i (reflexivity)",
            iterations,
            "Converged"
        ]
    }
    info_df = pd.DataFrame(info_data)

    # Save IRM
    irm_file = output_path / "ism_irm.xlsx"
    try:
        with pd.ExcelWriter(irm_file, engine='openpyxl') as writer:
            irm_df.to_excel(writer, sheet_name='Initial_Reachability_Matrix')
            info_df.to_excel(writer, sheet_name='Info', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {irm_file}\n"
            "Please close the Excel file if it's open and try again."
        )
    print(f"Initial Reachability Matrix saved to: {irm_file}")

    # Save FRM
    frm_file = output_path / "ism_frm.xlsx"
    try:
        with pd.ExcelWriter(frm_file, engine='openpyxl') as writer:
            frm_df.to_excel(writer, sheet_name='Final_Reachability_Matrix')
            power_df.to_excel(writer, sheet_name='Power_Summary', index=False)
            transitivity_df.to_excel(writer, sheet_name='Transitivity_Entries', index=False)
            info_df.to_excel(writer, sheet_name='Info', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {frm_file}\n"
            "Please close the Excel file if it's open and try again."
        )
    print(f"Final Reachability Matrix saved to: {frm_file}")

    return irm_file, frm_file


def print_frm_summary(irm, frm, driving_power, dependence_power, transitivity_result,
                       threshold, barriers, barrier_names=None):
    """
    Print summary of reachability matrix construction.

    Parameters:
    -----------
    irm : numpy.ndarray
        Initial Reachability Matrix
    frm : numpy.ndarray
        Final Reachability Matrix
    driving_power : numpy.ndarray
        Driving power values
    dependence_power : numpy.ndarray
        Dependence power values
    transitivity_result : dict
        Result from transitivity_check()
    threshold : float
        MMDE threshold used
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    n = len(barriers)

    print("-" * 70)
    print("ISM REACHABILITY MATRIX SUMMARY")
    print("-" * 70)
    print(f"Number of Barriers: {n}")
    print(f"MMDE Threshold (λ): {threshold:.{DECIMAL_PLACES}f}")
    print()

    print("BINARY MATRIX CONVERSION:")
    print(f"  k_ij = 1  if t_ij >= {threshold:.{DECIMAL_PLACES}f}")
    print(f"  k_ij = 0  if t_ij < {threshold:.{DECIMAL_PLACES}f}")
    print(f"  k_ii = 1  for all i (reflexivity)")
    print()

    # Count 1s in IRM vs TRM comparison
    irm_ones = np.sum(irm)
    print(f"INITIAL REACHABILITY MATRIX (IRM):")
    print(f"  Total 1s: {irm_ones} out of {n*n} elements")
    print(f"  Density: {irm_ones / (n*n) * 100:.1f}%")
    print()

    print("TRANSITIVITY CHECK:")
    print(f"  Formula: K^(m+1) = K^(m) ⊕ (K^(m) ⊗ K^(m))")
    print(f"  Iterations to converge: {transitivity_result['iterations']}")
    print(f"  Entries added by transitivity: {len(transitivity_result['transitivity_entries'])}")
    if len(transitivity_result['transitivity_entries']) > 0:
        print(f"  Transitivity entries (marked with *):")
        for entry in transitivity_result['transitivity_entries'][:10]:
            row_idx, col_idx = entry
            b_from = barriers[row_idx - 1].upper()
            b_to = barriers[col_idx - 1].upper()
            print(f"    {b_from}→{b_to}")
        if len(transitivity_result['transitivity_entries']) > 10:
            print(f"    ... and {len(transitivity_result['transitivity_entries']) - 10} more")
    print()

    frm_ones = np.sum(frm)
    print(f"FINAL REACHABILITY MATRIX (FRM):")
    print(f"  Total 1s: {frm_ones} out of {n*n} elements")
    print(f"  Density: {frm_ones / (n*n) * 100:.1f}%")
    print()

    print("DRIVING AND DEPENDENCE POWER:")
    print(f"  {'Barrier':<8} {'Driving Power':<15} {'Dependence Power':<18}")
    print(f"  {'-'*8} {'-'*15} {'-'*18}")
    for i, b in enumerate(barriers):
        print(f"  {b.upper():<8} {int(driving_power[i]):<15} {int(dependence_power[i]):<18}")
    print()

    print(f"  Total Driving Power: {int(np.sum(driving_power))}")
    print(f"  Total Dependence Power: {int(np.sum(dependence_power))}")
    print(f"  (Both should equal total 1s in FRM: {frm_ones})")
    print()

    print("-" * 70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def create_reachability_matrix(trm=None, threshold=None, mmde_results=None,
                                 barrier_names=None, save=True):
    """
    Main function to create Initial and Final Reachability Matrices.

    Parameters:
    -----------
    trm : pandas.DataFrame, optional
        Total Relation Matrix. If None, loads from file.
    threshold : float, optional
        MMDE threshold. If None, calculates using Module 9.
    mmde_results : dict, optional
        Results from Module 9 (alternative to providing threshold).
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names.
    save : bool, optional
        Whether to save outputs to Excel files.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'irm': Initial Reachability Matrix (numpy array)
        - 'frm': Final Reachability Matrix (numpy array)
        - 'driving_power': Driving power array
        - 'dependence_power': Dependence power array
        - 'threshold': MMDE threshold used
        - 'transitivity_result': Transitivity check results
        - 'irm_df': IRM DataFrame for export
        - 'frm_df': FRM DataFrame for export
        - 'barriers': List of barrier codes
        - 'output_files': Paths to saved files (if save=True)
    """
    print("\n" + "=" * 70)
    print("MODULE 10: ISM REACHABILITY MATRIX")
    print("=" * 70 + "\n")

    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    # Step 1: Get TRM
    if trm is None:
        print("Loading Total Relation Matrix from file...")
        trm = load_trm_from_file()
        print(f"Loaded TRM: {len(trm)} × {len(trm)} matrix.\n")
    else:
        print(f"Using provided TRM ({len(trm)} × {len(trm)} matrix).\n")

    barriers = list(trm.index)
    n = len(barriers)

    # Step 2: Get threshold
    if threshold is None:
        if mmde_results is not None:
            threshold = mmde_results['threshold']
            print(f"Using threshold from provided MMDE results: λ = {threshold:.{DECIMAL_PLACES}f}\n")
        else:
            print("Running Module 9 to calculate MMDE threshold...")
            mmde_results = calculate_mmde_threshold(save=False)
            threshold = mmde_results['threshold']
            print(f"\nMMDE Threshold: λ = {threshold:.{DECIMAL_PLACES}f}\n")
    else:
        print(f"Using provided threshold: λ = {threshold:.{DECIMAL_PLACES}f}\n")

    # Step 3: Create Initial Reachability Matrix
    print("Creating Initial Reachability Matrix (IRM)...")
    print(f"  Applying threshold: k_ij = 1 if t_ij >= {threshold:.{DECIMAL_PLACES}f}")
    print(f"  Setting diagonal to 1 (reflexivity)")
    irm = create_initial_reachability_matrix(trm, threshold)
    print(f"  IRM created: {n}×{n} binary matrix\n")

    # Step 4: Apply transitivity check
    print("Applying transitivity check...")
    print("  Formula: K^(m+1) = K^(m) ⊕ (K^(m) ⊗ K^(m))")
    transitivity_result = transitivity_check(irm)
    frm = transitivity_result['frm']
    print(f"  Converged in {transitivity_result['iterations']} iterations")
    print(f"  Transitivity entries added: {len(transitivity_result['transitivity_entries'])}\n")

    # Step 5: Calculate driving and dependence power
    print("Calculating Driving and Dependence Power...")
    print("  DP(i) = Σ r_ij (sum of row i)")
    print("  DEP(i) = Σ r_ji (sum of column i)")
    driving_power = calculate_driving_power(frm)
    dependence_power = calculate_dependence_power(frm)
    print("  Powers calculated.\n")

    # Step 6: Create DataFrames
    print("Creating DataFrames for export...")
    irm_df = create_irm_dataframe(irm, barriers, barrier_names)
    frm_df = create_frm_dataframe(
        frm, driving_power, dependence_power, barriers, barrier_names,
        transitivity_entries=transitivity_result['transitivity_entries']
    )
    transitivity_df = create_transitivity_dataframe(
        transitivity_result['transitivity_entries'], barriers, barrier_names
    )
    power_df = create_power_summary_dataframe(driving_power, dependence_power, barriers, barrier_names)
    print("  DataFrames created.\n")

    # Step 7: Print summary
    print_frm_summary(irm, frm, driving_power, dependence_power, transitivity_result,
                       threshold, barriers, barrier_names)

    # Step 8: Save outputs
    output_files = None
    if save:
        print("\nSaving outputs...")
        output_files = save_reachability_matrices(
            irm_df, frm_df, transitivity_df, power_df,
            threshold, transitivity_result['iterations']
        )

    # Prepare results
    results = {
        'irm': irm,
        'frm': frm,
        'driving_power': driving_power,
        'dependence_power': dependence_power,
        'threshold': threshold,
        'transitivity_result': transitivity_result,
        'irm_df': irm_df,
        'frm_df': frm_df,
        'power_df': power_df,
        'transitivity_df': transitivity_df,
        'barriers': barriers,
        'barrier_names': barrier_names,
        'trm': trm,
        'n': n,
        'output_files': output_files
    }

    print("\n" + "=" * 70)
    print("MODULE 10 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 10 in standalone mode...")
    results = create_reachability_matrix()

    # Display matrices
    print("\n" + "=" * 70)
    print("INITIAL REACHABILITY MATRIX (IRM)")
    print("=" * 70)
    print(results['irm_df'].to_string())

    print("\n" + "=" * 70)
    print("FINAL REACHABILITY MATRIX (FRM) with Powers")
    print("=" * 70)
    print(results['frm_df'].to_string())
