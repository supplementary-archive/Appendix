# SPDX-License-Identifier: PROPRIETARY
# File: m5_dematel_pro_relation.py
# Purpose: Module 5 - Calculate Prominence (D+R) and Relation (D-R) for DEMATEL cause-effect analysis

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

# Number of decimal places for Prominence and Relation values
DECIMAL_PLACES = 4

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def load_d_and_r_from_file(file_path=None):
    """
    Load D (Dispatch) and R (Receive) values from Module 4 Excel file.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the TRM D&R Excel file. If None, uses default path.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with D and R values per barrier
    """
    if file_path is None:
        script_dir = get_script_directory()
        file_path = script_dir / OUTPUT_DIR_DEMATEL / "dematel_trm_dar.xlsx"
    
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"TRM D&R file not found: {file_path}\n"
            "Please run Module 4 (m4_dematel_trm_dar.py) first to generate "
            "the D and R values."
        )
    
    try:
        # Load the D_R_Summary sheet which has D and R values per barrier
        d_r_df = pd.read_excel(file_path, sheet_name='D_R_Summary')
    except PermissionError:
        raise PermissionError(
            f"Cannot access file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    # Remove the TOTAL row if present
    d_r_df = d_r_df[d_r_df['Barrier'] != 'TOTAL'].copy()
    
    return d_r_df


def extract_d_and_r_values(d_r_df):
    """
    Extract D and R values from the loaded DataFrame.
    
    Parameters:
    -----------
    d_r_df : pandas.DataFrame
        DataFrame with D and R values from Module 11
        
    Returns:
    --------
    tuple
        - barriers: List of barrier codes (lowercase)
        - barrier_names: List of full barrier names
        - d_values: pandas.Series with D values
        - r_values: pandas.Series with R values
    """
    # Get barrier codes and names
    barrier_codes = d_r_df['Code'].str.lower().tolist()
    barrier_names_list = d_r_df['Barrier'].tolist()
    
    # Get D and R values
    # Handle different possible column names from Module 11
    d_col = 'D (Dispatch/Cause)' if 'D (Dispatch/Cause)' in d_r_df.columns else 'D'
    r_col = 'R (Receive/Effect)' if 'R (Receive/Effect)' in d_r_df.columns else 'R'
    
    d_values = pd.Series(d_r_df[d_col].values, index=barrier_codes, name='D')
    r_values = pd.Series(d_r_df[r_col].values, index=barrier_codes, name='R')
    
    return barrier_codes, barrier_names_list, d_values, r_values


def calculate_prominence(d_values, r_values, decimal_places=None):
    """
    Calculate Prominence (D+R) for each barrier.
    
    Prominence represents the total involvement/importance of a barrier.
    Higher D+R means the barrier is more important in the system.
    
    Formula: Prominence_i = D_i + R_i
    
    Parameters:
    -----------
    d_values : pandas.Series
        D (Dispatch) values for each barrier
    r_values : pandas.Series
        R (Receive) values for each barrier
    decimal_places : int, optional
        Number of decimal places for rounding
        
    Returns:
    --------
    pandas.Series
        Prominence (D+R) values indexed by barrier code
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    prominence = (d_values + r_values).round(decimal_places)
    prominence.name = 'D+R'
    
    return prominence


def calculate_relation(d_values, r_values, decimal_places=None):
    """
    Calculate Relation (D-R) for each barrier.
    
    Relation represents the net cause/effect role of a barrier.
    - D-R > 0: Barrier is a NET CAUSE (influences more than it is influenced)
    - D-R < 0: Barrier is a NET EFFECT (influenced more than it influences)
    - D-R ≈ 0: Barrier is neutral (balanced influence)
    
    Formula: Relation_i = D_i - R_i
    
    Parameters:
    -----------
    d_values : pandas.Series
        D (Dispatch) values for each barrier
    r_values : pandas.Series
        R (Receive) values for each barrier
    decimal_places : int, optional
        Number of decimal places for rounding
        
    Returns:
    --------
    pandas.Series
        Relation (D-R) values indexed by barrier code
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    relation = (d_values - r_values).round(decimal_places)
    relation.name = 'D-R'
    
    return relation


def classify_cause_effect(relation_value):
    """
    Classify a barrier as Cause or Effect based on D-R value.
    
    Parameters:
    -----------
    relation_value : float
        The D-R value for a barrier
        
    Returns:
    --------
    str
        'Cause' if D-R > 0, 'Effect' if D-R < 0, 'Neutral' if D-R = 0
    """
    if relation_value > 0:
        return 'Cause'
    elif relation_value < 0:
        return 'Effect'
    else:
        return 'Neutral'


def create_prominence_relation_df(barrier_names_list, d_values, r_values, 
                                   prominence, relation, decimal_places=None):
    """
    Create the main Prominence and Relation DataFrame for export.
    
    Parameters:
    -----------
    barrier_names_list : list
        List of full barrier names
    d_values : pandas.Series
        D values for each barrier
    r_values : pandas.Series
        R values for each barrier
    prominence : pandas.Series
        D+R values for each barrier
    relation : pandas.Series
        D-R values for each barrier
    decimal_places : int, optional
        Number of decimal places
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: Barrier, D, R, D-R, D+R
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    barrier_codes = list(d_values.index)
    
    rows = []
    for i, code in enumerate(barrier_codes):
        rows.append({
            'Barrier': barrier_names_list[i],
            'D': round(d_values[code], decimal_places),
            'R': round(r_values[code], decimal_places),
            'D-R': round(relation[code], decimal_places),
            'D+R': round(prominence[code], decimal_places)
        })
    
    return pd.DataFrame(rows)


def create_extended_analysis_df(barrier_names_list, d_values, r_values,
                                 prominence, relation, decimal_places=None):
    """
    Create extended analysis DataFrame with cause/effect classification.
    
    Parameters:
    -----------
    barrier_names_list : list
        List of full barrier names
    d_values : pandas.Series
        D values for each barrier
    r_values : pandas.Series
        R values for each barrier
    prominence : pandas.Series
        D+R values for each barrier
    relation : pandas.Series
        D-R values for each barrier
    decimal_places : int, optional
        Number of decimal places
        
    Returns:
    --------
    pandas.DataFrame
        Extended analysis with cause/effect classification and rankings
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    barrier_codes = list(d_values.index)
    
    rows = []
    for i, code in enumerate(barrier_codes):
        d_val = d_values[code]
        r_val = r_values[code]
        prom_val = prominence[code]
        rel_val = relation[code]
        
        rows.append({
            'Barrier': barrier_names_list[i],
            'Code': code.upper(),
            'D': round(d_val, decimal_places),
            'R': round(r_val, decimal_places),
            'D-R (Relation)': round(rel_val, decimal_places),
            'D+R (Prominence)': round(prom_val, decimal_places),
            'Role': classify_cause_effect(rel_val)
        })
    
    df = pd.DataFrame(rows)
    
    # Add rankings
    df['Prominence Rank'] = df['D+R (Prominence)'].rank(ascending=False).astype(int)
    df['|D-R| Rank'] = df['D-R (Relation)'].abs().rank(ascending=False).astype(int)
    
    return df


def create_metadata_df(d_values, r_values, prominence, relation, decimal_places=None):
    """
    Create metadata DataFrame with summary statistics.
    
    Parameters:
    -----------
    d_values : pandas.Series
        D values
    r_values : pandas.Series
        R values
    prominence : pandas.Series
        D+R values
    relation : pandas.Series
        D-R values
    decimal_places : int, optional
        Number of decimal places
        
    Returns:
    --------
    pandas.DataFrame
        Metadata information
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    n = len(d_values)
    n_causes = sum(1 for v in relation if v > 0)
    n_effects = sum(1 for v in relation if v < 0)
    n_neutral = sum(1 for v in relation if v == 0)
    
    # Find key barriers
    max_prom_code = prominence.idxmax()
    min_prom_code = prominence.idxmin()
    max_rel_code = relation.idxmax()
    min_rel_code = relation.idxmin()
    
    metadata = {
        'Parameter': [
            'Number of Barriers',
            'Prominence Formula',
            'Relation Formula',
            'Number of Causes (D-R > 0)',
            'Number of Effects (D-R < 0)',
            'Number of Neutral (D-R = 0)',
            'Mean Prominence (D+R)',
            'Max Prominence',
            'Min Prominence',
            'Mean Relation (D-R)',
            'Max Relation (strongest cause)',
            'Min Relation (strongest effect)',
            'Barrier with Highest Prominence',
            'Barrier with Lowest Prominence',
            'Strongest Cause Barrier',
            'Strongest Effect Barrier',
            'Decimal Places'
        ],
        'Value': [
            n,
            'D + R',
            'D - R',
            n_causes,
            n_effects,
            n_neutral,
            round(prominence.mean(), decimal_places),
            round(prominence.max(), decimal_places),
            round(prominence.min(), decimal_places),
            round(relation.mean(), decimal_places),
            round(relation.max(), decimal_places),
            round(relation.min(), decimal_places),
            max_prom_code.upper(),
            min_prom_code.upper(),
            max_rel_code.upper(),
            min_rel_code.upper(),
            decimal_places
        ]
    }
    
    return pd.DataFrame(metadata)


def save_prominence_relation(main_df, extended_df, metadata_df, output_dir=None):
    """
    Save Prominence and Relation analysis to Excel file.
    
    Parameters:
    -----------
    main_df : pandas.DataFrame
        Main DataFrame with Barrier, D, R, D-R, D+R
    extended_df : pandas.DataFrame
        Extended analysis DataFrame
    metadata_df : pandas.DataFrame
        Metadata information
    output_dir : str or Path, optional
        Output directory. If None, uses OUTPUT_DIR_DEMATEL.
        
    Returns:
    --------
    Path
        Path to the saved Excel file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_DEMATEL
    
    script_dir = get_script_directory()
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    file_path = output_path / "dematel_pro_relation.xlsx"
    
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            main_df.to_excel(writer, sheet_name='Prominence_Relation', index=False)
            extended_df.to_excel(writer, sheet_name='Extended_Analysis', index=False)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    print(f"Prominence and Relation analysis saved to: {file_path}")
    
    return file_path


def print_prominence_relation_summary(barrier_names_list, d_values, r_values,
                                       prominence, relation, decimal_places=None):
    """
    Print summary of Prominence and Relation analysis.
    
    Parameters:
    -----------
    barrier_names_list : list
        List of full barrier names
    d_values : pandas.Series
        D values
    r_values : pandas.Series
        R values
    prominence : pandas.Series
        D+R values
    relation : pandas.Series
        D-R values
    decimal_places : int, optional
        Number of decimal places
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    barrier_codes = list(d_values.index)
    n = len(barrier_codes)
    
    print("-" * 90)
    print("PROMINENCE (D+R) AND RELATION (D-R) ANALYSIS")
    print("-" * 90)
    print(f"Number of Barriers: {n}")
    print()
    
    print("FORMULAS:")
    print("  Prominence (D+R) = D + R")
    print("    → Measures total involvement/importance of barrier in the system")
    print("    → Higher value = more important barrier")
    print()
    print("  Relation (D-R) = D - R")
    print("    → Measures net cause/effect role of barrier")
    print("    → D-R > 0: NET CAUSE (influences more than influenced)")
    print("    → D-R < 0: NET EFFECT (influenced more than influences)")
    print()
    
    print("PROMINENCE AND RELATION VALUES:")
    print("-" * 90)
    print(f"{'Barrier':<45} {'D':<10} {'R':<10} {'D-R':<12} {'D+R':<12} {'Role':<10}")
    print("-" * 90)
    
    for i, code in enumerate(barrier_codes):
        name = barrier_names_list[i]
        # Truncate name if too long
        display_name = name[:42] + "..." if len(name) > 42 else name
        
        d_val = d_values[code]
        r_val = r_values[code]
        prom_val = prominence[code]
        rel_val = relation[code]
        role = classify_cause_effect(rel_val)
        
        print(f"{display_name:<45} {d_val:<10.{decimal_places}f} {r_val:<10.{decimal_places}f} "
              f"{rel_val:<12.{decimal_places}f} {prom_val:<12.{decimal_places}f} {role:<10}")
    
    print("-" * 90)
    
    # Summary statistics
    n_causes = sum(1 for v in relation if v > 0)
    n_effects = sum(1 for v in relation if v < 0)
    
    print()
    print("CAUSE-EFFECT CLASSIFICATION:")
    print(f"  Causes (D-R > 0):  {n_causes} barriers")
    print(f"  Effects (D-R < 0): {n_effects} barriers")
    print()
    
    # Key findings
    max_prom_idx = prominence.idxmax()
    min_prom_idx = prominence.idxmin()
    max_rel_idx = relation.idxmax()
    min_rel_idx = relation.idxmin()
    
    # Get barrier names by index
    max_prom_name = barrier_names_list[barrier_codes.index(max_prom_idx)]
    min_prom_name = barrier_names_list[barrier_codes.index(min_prom_idx)]
    max_rel_name = barrier_names_list[barrier_codes.index(max_rel_idx)]
    min_rel_name = barrier_names_list[barrier_codes.index(min_rel_idx)]
    
    print("KEY FINDINGS:")
    print()
    print("  PROMINENCE (Importance):")
    print(f"    Highest: {max_prom_name}")
    print(f"             D+R = {prominence[max_prom_idx]:.{decimal_places}f}")
    print(f"    Lowest:  {min_prom_name}")
    print(f"             D+R = {prominence[min_prom_idx]:.{decimal_places}f}")
    print()
    print("  RELATION (Cause-Effect Role):")
    print(f"    Strongest CAUSE:  {max_rel_name}")
    print(f"                      D-R = {relation[max_rel_idx]:.{decimal_places}f}")
    print(f"    Strongest EFFECT: {min_rel_name}")
    print(f"                      D-R = {relation[min_rel_idx]:.{decimal_places}f}")
    print("-" * 90)


def calculate_prominence_and_relation(d_r_data=None, decimal_places=None, save=True):
    """
    Main function to calculate Prominence (D+R) and Relation (D-R) for DEMATEL.
    
    Prominence (D+R): Total involvement/importance of barrier
    Relation (D-R): Net cause/effect role of barrier
    
    Parameters:
    -----------
    d_r_data : dict, optional
        Dictionary containing d_values and r_values from Module 4.
        If None, loads from dematel_trm_dar.xlsx
    decimal_places : int, optional
        Number of decimal places for calculated values.
        If None, uses DECIMAL_PLACES (default: 4).
    save : bool, optional
        Whether to save outputs to Excel file.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'main_df': Main DataFrame (Barrier, D, R, D-R, D+R)
        - 'extended_df': Extended analysis DataFrame
        - 'd_values': D values Series
        - 'r_values': R values Series
        - 'prominence': D+R values Series
        - 'relation': D-R values Series
        - 'barrier_codes': List of barrier codes
        - 'barrier_names': List of barrier names
        - 'n_causes': Number of cause barriers
        - 'n_effects': Number of effect barriers
        - 'output_file': Path to saved file (if save=True)
    """
    print("\n" + "=" * 90)
    print("MODULE 5: DEMATEL - PROMINENCE (D+R) AND RELATION (D-R)")
    print("=" * 90 + "\n")
    
    # Step 1: Load or use provided D and R data
    if d_r_data is None:
        print("Loading D and R values from Module 4...")
        d_r_df = load_d_and_r_from_file()
        barrier_codes, barrier_names_list, d_values, r_values = extract_d_and_r_values(d_r_df)
        print(f"Loaded D and R values for {len(d_values)} barriers.\n")
    else:
        # Use provided data
        d_values = d_r_data['d_values']
        r_values = d_r_data['r_values']
        barrier_codes = list(d_values.index)
        barrier_names_list = [d_r_data['barrier_to_name'].get(code, code.upper()) 
                              for code in barrier_codes]
        print(f"Using provided D and R values ({len(d_values)} barriers).\n")
    
    print(f"Barriers: {[b.upper() for b in barrier_codes]}\n")
    
    # Step 2: Set decimal places
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    # Step 3: Calculate Prominence (D+R)
    print("Calculating Prominence (D+R) = D + R...")
    prominence = calculate_prominence(d_values, r_values, decimal_places)
    print(f"Prominence calculated for {len(prominence)} barriers.\n")
    
    # Step 4: Calculate Relation (D-R)
    print("Calculating Relation (D-R) = D - R...")
    relation = calculate_relation(d_values, r_values, decimal_places)
    print(f"Relation calculated for {len(relation)} barriers.\n")
    
    # Step 5: Count causes and effects
    n_causes = sum(1 for v in relation if v > 0)
    n_effects = sum(1 for v in relation if v < 0)
    print(f"Classification: {n_causes} Causes, {n_effects} Effects\n")
    
    # Step 6: Print summary
    print_prominence_relation_summary(
        barrier_names_list, d_values, r_values, prominence, relation, decimal_places
    )
    
    # Step 7: Create main DataFrame (as specified by user)
    main_df = create_prominence_relation_df(
        barrier_names_list, d_values, r_values, prominence, relation, decimal_places
    )
    
    # Step 8: Create extended analysis DataFrame
    extended_df = create_extended_analysis_df(
        barrier_names_list, d_values, r_values, prominence, relation, decimal_places
    )
    
    # Step 9: Create metadata DataFrame
    metadata_df = create_metadata_df(d_values, r_values, prominence, relation, decimal_places)
    
    # Step 10: Save outputs
    output_file = None
    if save:
        print("\nSaving outputs...")
        output_file = save_prominence_relation(main_df, extended_df, metadata_df)
    
    # Prepare results dictionary
    results = {
        'main_df': main_df,
        'extended_df': extended_df,
        'd_values': d_values,
        'r_values': r_values,
        'prominence': prominence,
        'relation': relation,
        'barrier_codes': barrier_codes,
        'barrier_names': barrier_names_list,
        'n_causes': n_causes,
        'n_effects': n_effects,
        'output_file': output_file
    }
    
    print("\n" + "=" * 90)
    print("MODULE 5 COMPLETED SUCCESSFULLY")
    print("=" * 90 + "\n")
    
    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 5 in standalone mode...")
    results = calculate_prominence_and_relation()
    
    # Display the main DataFrame
    print("\n" + "=" * 90)
    print("PROMINENCE AND RELATION TABLE")
    print("=" * 90)
    print("Columns: Barrier | D | R | D-R (Relation) | D+R (Prominence)")
    print()
    print(results['main_df'].to_string(index=False))
    
    # Display cause-effect summary
    print("\n" + "=" * 90)
    print("CAUSE-EFFECT SUMMARY")
    print("=" * 90)
    print(f"Total Barriers: {len(results['barrier_codes'])}")
    print(f"Causes (D-R > 0): {results['n_causes']}")
    print(f"Effects (D-R < 0): {results['n_effects']}")

