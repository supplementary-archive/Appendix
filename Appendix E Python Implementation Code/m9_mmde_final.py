# SPDX-License-Identifier: PROPRIETARY
# File: m9_mmde_final.py
# Purpose: Module 9 - MMDE Final Calculations (Entropy, MDE, Threshold) for DEMATEL-ISM

"""
MMDE (Maximum Mean De-Entropy) Threshold Calculation - Step 3 (Final)

This module calculates entropy, de-entropy, mean de-entropy (MDE) for each position,
finds the maximum MDE positions, and determines the final threshold value.

Mathematical Formulas:
----------------------

1. Information Entropy:
   H(p₁, p₂, ..., pₖ) = -Σ pᵢ × ln(pᵢ)  for i = 1 to k
   Where pᵢ is the probability of factor i

2. Probability Calculation:
   pᵢ = count(factor i) / total_count

3. Maximum Entropy (uniform distribution):
   H_max = ln(N)
   Where N is the number of unique factors

4. De-Entropy:
   H^D = H_max - H_actual = ln(N) - H

5. Mean De-Entropy (MDE):
   MDE = H^D / N

Edge Cases:
- If N <= 1 (only one unique factor or empty list): MDE = 0

Threshold Determination:
- Find pos_D where MDE_D is maximum
- Find pos_R where MDE_R is maximum
- T_d_max = triplets from position 1 to pos_D
- T_r_max = triplets from position 1 to pos_R
- T_th = T_d_max ∪ T_r_max
- threshold λ = min(value) in T_th
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# Import from Module 1 and Module 8
from m1_data_processing import (
    BARRIER_NAMES,
    get_script_directory
)
from m7_mmde_triplet import OUTPUT_DIR_ISM, DECIMAL_PLACES
from m8_mmde_sets import build_mmde_sets

# =============================================================================
# MATHEMATICAL FUNCTIONS
# =============================================================================

def calculate_probability(factor_list):
    """
    Calculate probability distribution for factors in a list.

    Formula:
    --------
    pᵢ = count(factor i) / total_count

    Parameters:
    -----------
    factor_list : list
        List of factor indices (with possible duplicates)

    Returns:
    --------
    dict
        Dictionary mapping each unique factor to its probability
    """
    if len(factor_list) == 0:
        return {}

    total_count = len(factor_list)
    counts = Counter(factor_list)

    probabilities = {}
    for factor, count in counts.items():
        probabilities[factor] = count / total_count

    return probabilities


def calculate_entropy(factor_list):
    """
    Calculate Shannon information entropy for a list of factors.

    Formula:
    --------
    H(p₁, p₂, ..., pₖ) = -Σ pᵢ × ln(pᵢ)  for i = 1 to k

    Where:
    - pᵢ = count(factor i) / total_count
    - ln is the natural logarithm
    - If pᵢ = 0, treat pᵢ × ln(pᵢ) = 0

    Parameters:
    -----------
    factor_list : list
        List of factor indices (with possible duplicates)

    Returns:
    --------
    float
        Entropy value H
    """
    if len(factor_list) == 0:
        return 0.0

    probabilities = calculate_probability(factor_list)

    entropy = 0.0
    for p in probabilities.values():
        if p > 0:  # Avoid log(0)
            entropy -= p * np.log(p)  # Natural logarithm

    return entropy


def calculate_max_entropy(n_unique):
    """
    Calculate maximum entropy for N unique factors.

    Formula:
    --------
    H_max = ln(N)

    Maximum entropy occurs when all N factors have equal probability (1/N each).

    Parameters:
    -----------
    n_unique : int
        Number of unique factors

    Returns:
    --------
    float
        Maximum entropy value
    """
    if n_unique <= 1:
        return 0.0

    return np.log(n_unique)  # Natural logarithm


def calculate_de_entropy(h_actual, h_max):
    """
    Calculate de-entropy (deviation from maximum entropy).

    Formula:
    --------
    H^D = H_max - H_actual = ln(N) - H

    De-entropy measures how much the distribution deviates from uniform.

    Parameters:
    -----------
    h_actual : float
        Actual entropy
    h_max : float
        Maximum entropy

    Returns:
    --------
    float
        De-entropy value
    """
    return h_max - h_actual


def calculate_mde(factor_list):
    """
    Calculate Mean De-Entropy (MDE) for a list of factors.

    Complete MDE Calculation:
    -------------------------
    1. N = count of unique factors in the list
    2. If N <= 1: return 0 (edge case)
    3. H = entropy of the list
    4. H_max = ln(N)
    5. H^D = H_max - H
    6. MDE = H^D / N

    Parameters:
    -----------
    factor_list : list
        List of factor indices (with possible duplicates)

    Returns:
    --------
    dict
        Dictionary containing:
        - 'mde': Mean De-Entropy value
        - 'entropy': Actual entropy H
        - 'max_entropy': Maximum entropy H_max
        - 'de_entropy': De-entropy H^D
        - 'n_unique': Number of unique factors N
        - 'total_count': Total count in list
    """
    # Edge case: empty list
    if len(factor_list) == 0:
        return {
            'mde': 0.0,
            'entropy': 0.0,
            'max_entropy': 0.0,
            'de_entropy': 0.0,
            'n_unique': 0,
            'total_count': 0
        }

    # Count unique factors
    n_unique = len(set(factor_list))
    total_count = len(factor_list)

    # Edge case: only one unique factor
    if n_unique <= 1:
        return {
            'mde': 0.0,
            'entropy': 0.0,
            'max_entropy': 0.0,
            'de_entropy': 0.0,
            'n_unique': n_unique,
            'total_count': total_count
        }

    # Calculate entropy
    entropy = calculate_entropy(factor_list)

    # Calculate maximum entropy
    max_entropy = calculate_max_entropy(n_unique)

    # Calculate de-entropy
    de_entropy = calculate_de_entropy(entropy, max_entropy)

    # Calculate MDE
    mde = de_entropy / n_unique

    return {
        'mde': mde,
        'entropy': entropy,
        'max_entropy': max_entropy,
        'de_entropy': de_entropy,
        'n_unique': n_unique,
        'total_count': total_count
    }


# =============================================================================
# MMDE CALCULATION FUNCTIONS
# =============================================================================

def calculate_all_mde(cumulative_data):
    """
    Calculate MDE for all positions for both Td (dispatch) and Tr (receive).

    For each position i:
    - MDE_D_i = MDE of Td_i (dispatch set at position i)
    - MDE_R_i = MDE of Tr_i (receive set at position i)

    Parameters:
    -----------
    cumulative_data : dict
        Dictionary from build_cumulative_sets() containing:
        - 'Td_cumulative': List of Td sets at each position
        - 'Tr_cumulative': List of Tr sets at each position

    Returns:
    --------
    dict
        Dictionary containing:
        - 'mde_d_list': List of MDE_D results at each position
        - 'mde_r_list': List of MDE_R results at each position
        - 'positions': List of position numbers
    """
    Td_cumulative = cumulative_data['Td_cumulative']
    Tr_cumulative = cumulative_data['Tr_cumulative']
    positions = cumulative_data['positions']

    mde_d_list = []
    mde_r_list = []

    for i in range(len(positions)):
        Td_i = Td_cumulative[i]
        Tr_i = Tr_cumulative[i]

        # Calculate MDE for dispatch (Td)
        mde_d_result = calculate_mde(Td_i)
        mde_d_list.append(mde_d_result)

        # Calculate MDE for receive (Tr)
        mde_r_result = calculate_mde(Tr_i)
        mde_r_list.append(mde_r_result)

    return {
        'mde_d_list': mde_d_list,
        'mde_r_list': mde_r_list,
        'positions': positions
    }


def find_max_mde_positions(mde_results):
    """
    Find positions where MDE_D and MDE_R are maximum.

    Parameters:
    -----------
    mde_results : dict
        Dictionary from calculate_all_mde()

    Returns:
    --------
    dict
        Dictionary containing:
        - 'pos_d': Position of maximum MDE_D (1-based)
        - 'max_mde_d': Maximum MDE_D value
        - 'pos_r': Position of maximum MDE_R (1-based)
        - 'max_mde_r': Maximum MDE_R value
    """
    mde_d_list = mde_results['mde_d_list']
    mde_r_list = mde_results['mde_r_list']
    positions = mde_results['positions']

    # Extract MDE values
    mde_d_values = [result['mde'] for result in mde_d_list]
    mde_r_values = [result['mde'] for result in mde_r_list]

    # Find maximum positions (0-indexed)
    max_d_idx = np.argmax(mde_d_values)
    max_r_idx = np.argmax(mde_r_values)

    return {
        'pos_d': positions[max_d_idx],  # 1-based position
        'max_mde_d': mde_d_values[max_d_idx],
        'pos_r': positions[max_r_idx],  # 1-based position
        'max_mde_r': mde_r_values[max_r_idx],
        'max_d_idx': max_d_idx,  # 0-based index for array access
        'max_r_idx': max_r_idx   # 0-based index for array access
    }


def determine_threshold(sorted_triplets, max_positions):
    """
    Determine the final MMDE threshold value.

    Procedure:
    ----------
    1. T_d_max = all triplets from position 1 to pos_D (inclusive)
    2. T_r_max = all triplets from position 1 to pos_R (inclusive)
    3. T_th = T_d_max ∪ T_r_max (union, remove duplicates)
    4. threshold λ = min(value) for all triplets in T_th

    Parameters:
    -----------
    sorted_triplets : list
        Sorted list of triplets (value, row, col)
    max_positions : dict
        Dictionary from find_max_mde_positions()

    Returns:
    --------
    dict
        Dictionary containing:
        - 'threshold': Final threshold value λ
        - 'T_d_max': Triplets from position 1 to pos_D
        - 'T_r_max': Triplets from position 1 to pos_R
        - 'T_th': Union of T_d_max and T_r_max
        - 'pos_d': Position of max MDE_D
        - 'pos_r': Position of max MDE_R
    """
    pos_d = max_positions['pos_d']
    pos_r = max_positions['pos_r']

    # T_d_max = triplets from position 1 to pos_D (inclusive)
    # Since positions are 1-based and Python arrays are 0-indexed,
    # we need triplets from index 0 to pos_d-1, which is [:pos_d]
    T_d_max = sorted_triplets[:pos_d]

    # T_r_max = triplets from position 1 to pos_R (inclusive)
    T_r_max = sorted_triplets[:pos_r]

    # T_th = T_d_max ∪ T_r_max (union)
    # Use set to remove duplicates, then convert back to list
    T_th_set = set(T_d_max) | set(T_r_max)
    T_th = list(T_th_set)

    # threshold = min(value) for all triplets in T_th
    if len(T_th) == 0:
        threshold = 0.0
    else:
        threshold = min(triplet[0] for triplet in T_th)

    return {
        'threshold': threshold,
        'T_d_max': T_d_max,
        'T_r_max': T_r_max,
        'T_th': T_th,
        'pos_d': pos_d,
        'pos_r': pos_r,
        'n_T_d_max': len(T_d_max),
        'n_T_r_max': len(T_r_max),
        'n_T_th': len(T_th)
    }


# =============================================================================
# DATAFRAME CREATION FUNCTIONS
# =============================================================================

def create_mde_dataframe(mde_results, sorted_triplets, barriers):
    """
    Create DataFrame with MDE values for all positions.

    Output columns:
    - Position_i
    - Value (triplet value at this position)
    - MDE_D_i
    - MDE_R_i
    - Additional details for verification

    Parameters:
    -----------
    mde_results : dict
        Dictionary from calculate_all_mde()
    sorted_triplets : list
        Sorted list of triplets
    barriers : list
        List of barrier codes

    Returns:
    --------
    pandas.DataFrame
        DataFrame with MDE values
    """
    positions = mde_results['positions']
    mde_d_list = mde_results['mde_d_list']
    mde_r_list = mde_results['mde_r_list']

    data = []
    for i in range(len(positions)):
        triplet = sorted_triplets[i]
        value, row, col = triplet
        mde_d = mde_d_list[i]
        mde_r = mde_r_list[i]

        data.append({
            'Position_i': positions[i],
            'Triplet_Value': round(value, DECIMAL_PLACES),
            'MDE_D_i': round(mde_d['mde'], 6),
            'MDE_R_i': round(mde_r['mde'], 6),
            'H_D_i': round(mde_d['entropy'], 6),
            'H_R_i': round(mde_r['entropy'], 6),
            'H_max_D_i': round(mde_d['max_entropy'], 6),
            'H_max_R_i': round(mde_r['max_entropy'], 6),
            'HD_D_i': round(mde_d['de_entropy'], 6),
            'HD_R_i': round(mde_r['de_entropy'], 6),
            'N_D_i': mde_d['n_unique'],
            'N_R_i': mde_r['n_unique']
        })

    return pd.DataFrame(data)


def create_final_results_dataframe(max_positions, threshold_data, n, n_sq):
    """
    Create DataFrame with final MMDE results.

    Parameters:
    -----------
    max_positions : dict
        Dictionary from find_max_mde_positions()
    threshold_data : dict
        Dictionary from determine_threshold()
    n : int
        Number of barriers
    n_sq : int
        Total number of triplets (n²)

    Returns:
    --------
    pandas.DataFrame
        Final results DataFrame
    """
    results_data = {
        'Parameter': [
            'Number of Barriers (n)',
            'Total Triplets (n²)',
            '',
            'Maximum MDE_D Position (pos_D)',
            'Maximum MDE_D Value',
            'Triplets in T_d_max',
            '',
            'Maximum MDE_R Position (pos_R)',
            'Maximum MDE_R Value',
            'Triplets in T_r_max',
            '',
            'Triplets in Union (T_th)',
            '',
            'FINAL THRESHOLD (λ)'
        ],
        'Value': [
            n,
            n_sq,
            '',
            max_positions['pos_d'],
            round(max_positions['max_mde_d'], 6),
            threshold_data['n_T_d_max'],
            '',
            max_positions['pos_r'],
            round(max_positions['max_mde_r'], 6),
            threshold_data['n_T_r_max'],
            '',
            threshold_data['n_T_th'],
            '',
            round(threshold_data['threshold'], DECIMAL_PLACES)
        ]
    }

    return pd.DataFrame(results_data)


def create_threshold_triplets_dataframe(threshold_data, barriers):
    """
    Create DataFrame showing triplets in the threshold set T_th.

    Parameters:
    -----------
    threshold_data : dict
        Dictionary from determine_threshold()
    barriers : list
        List of barrier codes

    Returns:
    --------
    pandas.DataFrame
        DataFrame with threshold triplets
    """
    T_th = threshold_data['T_th']

    # Sort by value for display
    T_th_sorted = sorted(T_th, key=lambda x: x[0], reverse=True)

    data = []
    for triplet in T_th_sorted:
        value, row, col = triplet
        barrier_from = barriers[row - 1].upper()
        barrier_to = barriers[col - 1].upper()

        data.append({
            'Value': round(value, DECIMAL_PLACES),
            'Row_Index': row,
            'Col_Index': col,
            'Barrier_Pair': f"{barrier_from}→{barrier_to}",
            'In_T_d_max': triplet in threshold_data['T_d_max'],
            'In_T_r_max': triplet in threshold_data['T_r_max']
        })

    return pd.DataFrame(data)


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_mmde_results(mde_df, final_results_df, threshold_triplets_df, output_dir=None):
    """
    Save MMDE results to Excel files.

    Output files:
    1. mmde_mde.xlsx: Position, MDE_D_i, MDE_R_i and details
    2. mmde_final_results.xlsx: Final threshold and parameters

    Parameters:
    -----------
    mde_df : pandas.DataFrame
        DataFrame with MDE values
    final_results_df : pandas.DataFrame
        Final results DataFrame
    threshold_triplets_df : pandas.DataFrame
        Threshold triplets DataFrame
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

    # Save MDE values
    mde_file = output_path / "mmde_mde.xlsx"
    try:
        with pd.ExcelWriter(mde_file, engine='openpyxl') as writer:
            mde_df.to_excel(writer, sheet_name='MDE_Values', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {mde_file}\n"
            "Please close the Excel file if it's open and try again."
        )
    print(f"MDE values saved to: {mde_file}")

    # Save final results
    final_file = output_path / "mmde_final_results.xlsx"
    try:
        with pd.ExcelWriter(final_file, engine='openpyxl') as writer:
            final_results_df.to_excel(writer, sheet_name='Final_Results', index=False)
            threshold_triplets_df.to_excel(writer, sheet_name='Threshold_Triplets', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {final_file}\n"
            "Please close the Excel file if it's open and try again."
        )
    print(f"Final results saved to: {final_file}")

    return mde_file, final_file


def print_mmde_summary(mde_results, max_positions, threshold_data, n):
    """
    Print comprehensive summary of MMDE calculations.

    Parameters:
    -----------
    mde_results : dict
        Dictionary from calculate_all_mde()
    max_positions : dict
        Dictionary from find_max_mde_positions()
    threshold_data : dict
        Dictionary from determine_threshold()
    n : int
        Number of barriers
    """
    n_sq = n * n

    print("-" * 70)
    print("MMDE CALCULATION SUMMARY")
    print("-" * 70)
    print(f"Number of Barriers (n): {n}")
    print(f"Total Triplets (n²): {n_sq}")
    print()

    print("MATHEMATICAL FORMULAS USED:")
    print("  Probability: pᵢ = count(i) / total")
    print("  Entropy: H = -Σ pᵢ × ln(pᵢ)")
    print("  Max Entropy: H_max = ln(N)")
    print("  De-Entropy: H^D = H_max - H")
    print("  Mean De-Entropy: MDE = H^D / N")
    print()

    print("MAXIMUM MDE POSITIONS:")
    print(f"  Dispatch (MDE_D):")
    print(f"    Position: {max_positions['pos_d']}")
    print(f"    Max MDE_D: {max_positions['max_mde_d']:.6f}")
    print()
    print(f"  Receive (MDE_R):")
    print(f"    Position: {max_positions['pos_r']}")
    print(f"    Max MDE_R: {max_positions['max_mde_r']:.6f}")
    print()

    print("THRESHOLD DETERMINATION:")
    print(f"  T_d_max = triplets from position 1 to {threshold_data['pos_d']}")
    print(f"    Count: {threshold_data['n_T_d_max']} triplets")
    print()
    print(f"  T_r_max = triplets from position 1 to {threshold_data['pos_r']}")
    print(f"    Count: {threshold_data['n_T_r_max']} triplets")
    print()
    print(f"  T_th = T_d_max ∪ T_r_max (union)")
    print(f"    Count: {threshold_data['n_T_th']} unique triplets")
    print()

    print("=" * 70)
    print(f"  FINAL THRESHOLD (λ) = {threshold_data['threshold']:.{DECIMAL_PLACES}f}")
    print("=" * 70)
    print()
    print("  This threshold will be used to convert TRM to binary matrix:")
    print(f"    k_ij = 1  if t_ij >= {threshold_data['threshold']:.{DECIMAL_PLACES}f}")
    print(f"    k_ij = 0  if t_ij < {threshold_data['threshold']:.{DECIMAL_PLACES}f}")
    print()

    print("-" * 70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def calculate_mmde_threshold(cumulative_data=None, sorted_triplets=None, barriers=None,
                              m8_results=None, save=True):
    """
    Main function to calculate MMDE threshold.

    This is the final step of the MMDE threshold calculation process.

    Parameters:
    -----------
    cumulative_data : dict, optional
        Cumulative sets data from Module 8
    sorted_triplets : list, optional
        Sorted triplets from Module 7
    barriers : list, optional
        List of barrier codes
    m8_results : dict, optional
        Complete results from Module 8 (alternative to providing data separately)
    save : bool, optional
        Whether to save outputs to Excel files

    Returns:
    --------
    dict
        Dictionary containing:
        - 'threshold': Final MMDE threshold value λ
        - 'mde_results': MDE values for all positions
        - 'max_positions': Maximum MDE positions
        - 'threshold_data': Threshold determination data
        - 'mde_df': DataFrame with MDE values
        - 'final_results_df': Final results DataFrame
        - 'sorted_triplets': Sorted triplets used
        - 'barriers': List of barrier codes
        - 'n': Number of barriers
        - 'output_files': Paths to saved files (if save=True)
    """
    print("\n" + "=" * 70)
    print("MODULE 9: MMDE FINAL CALCULATIONS")
    print("=" * 70 + "\n")

    # Step 1: Get cumulative data and sorted triplets
    if cumulative_data is None or sorted_triplets is None or barriers is None:
        if m8_results is not None:
            cumulative_data = m8_results['cumulative_data']
            sorted_triplets = m8_results['sorted_triplets']
            barriers = m8_results['barriers']
            print("Using data from provided Module 8 results.\n")
        else:
            print("Running Module 8 to get cumulative sets...")
            m8_results = build_mmde_sets(save=False)
            cumulative_data = m8_results['cumulative_data']
            sorted_triplets = m8_results['sorted_triplets']
            barriers = m8_results['barriers']
            print()

    n = len(barriers)
    n_sq = len(sorted_triplets)

    print(f"Number of barriers: {n}")
    print(f"Number of triplets: {n_sq}\n")

    # Step 2: Calculate MDE for all positions
    print("Calculating MDE for all positions...")
    print("  For each position i:")
    print("    MDE_D_i = MDE of Td_i (dispatch)")
    print("    MDE_R_i = MDE of Tr_i (receive)")
    mde_results = calculate_all_mde(cumulative_data)
    print(f"  MDE calculated for {n_sq} positions.\n")

    # Step 3: Find maximum MDE positions
    print("Finding maximum MDE positions...")
    max_positions = find_max_mde_positions(mde_results)
    print(f"  pos_D (max MDE_D): {max_positions['pos_d']}")
    print(f"  pos_R (max MDE_R): {max_positions['pos_r']}\n")

    # Step 4: Determine threshold
    print("Determining threshold...")
    print("  T_th = T_d_max ∪ T_r_max")
    print("  threshold λ = min(value) in T_th")
    threshold_data = determine_threshold(sorted_triplets, max_positions)
    print(f"  Threshold λ = {threshold_data['threshold']:.{DECIMAL_PLACES}f}\n")

    # Step 5: Create DataFrames
    print("Creating DataFrames for export...")
    mde_df = create_mde_dataframe(mde_results, sorted_triplets, barriers)
    final_results_df = create_final_results_dataframe(max_positions, threshold_data, n, n_sq)
    threshold_triplets_df = create_threshold_triplets_dataframe(threshold_data, barriers)
    print("  DataFrames created.\n")

    # Step 6: Print summary
    print_mmde_summary(mde_results, max_positions, threshold_data, n)

    # Step 7: Save outputs
    output_files = None
    if save:
        print("\nSaving outputs...")
        output_files = save_mmde_results(mde_df, final_results_df, threshold_triplets_df)

    # Prepare results
    results = {
        'threshold': threshold_data['threshold'],
        'mde_results': mde_results,
        'max_positions': max_positions,
        'threshold_data': threshold_data,
        'mde_df': mde_df,
        'final_results_df': final_results_df,
        'threshold_triplets_df': threshold_triplets_df,
        'sorted_triplets': sorted_triplets,
        'cumulative_data': cumulative_data,
        'barriers': barriers,
        'n': n,
        'output_files': output_files
    }

    print("\n" + "=" * 70)
    print("MODULE 9 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 9 in standalone mode...")
    results = calculate_mmde_threshold()

    # Display key results
    print("\n" + "=" * 70)
    print("MDE VALUES - First 15 positions")
    print("=" * 70)
    display_cols = ['Position_i', 'Triplet_Value', 'MDE_D_i', 'MDE_R_i']
    print(results['mde_df'][display_cols].head(15).to_string(index=False))

    print("\n" + "=" * 70)
    print(f"FINAL THRESHOLD: λ = {results['threshold']:.{DECIMAL_PLACES}f}")
    print("=" * 70)
