# SPDX-License-Identifier: PROPRIETARY
# File: m4_dematel_trm_dar.py
# Purpose: Module 4 - Calculate Dispatch (D) and Receive (R) from Total Relation Matrix for DEMATEL analysis

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

# Number of decimal places for D and R values
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
        Total Relation Matrix with barrier codes as index and columns
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


def extract_barrier_mapping(trm, barrier_names=None):
    """
    Extract barrier code to full name mapping.
    
    Parameters:
    -----------
    trm : pandas.DataFrame
        Total Relation Matrix with barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names.
        If None, uses BARRIER_NAMES from Module 1.
        
    Returns:
    --------
    dict
        Mapping from barrier codes (lowercase) to full names
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES
    
    # Create mapping for all barriers in the matrix
    barrier_to_name = {}
    for code in trm.index:
        barrier_to_name[code] = barrier_names.get(code, code.upper())
    
    return barrier_to_name


def calculate_d_dispatch(trm, decimal_places=None):
    """
    Calculate D (Dispatch/Cause) for each barrier.
    
    D represents the total influence that barrier i dispatches/causes to all other barriers.
    Formula: D_i = Σⱼ t_ij (sum of row i)
    
    Parameters:
    -----------
    trm : pandas.DataFrame
        Total Relation Matrix
    decimal_places : int, optional
        Number of decimal places for rounding
        
    Returns:
    --------
    pandas.Series
        D values indexed by barrier code
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    # D = sum of each row (how much each barrier dispatches/influences others)
    d_values = trm.sum(axis=1).round(decimal_places)
    d_values.name = 'D'
    
    return d_values


def calculate_r_receive(trm, decimal_places=None):
    """
    Calculate R (Receive/Effect) for each barrier.
    
    R represents the total influence that barrier i receives from all other barriers.
    Formula: R_i = Σⱼ t_ji (sum of column i)
    
    Parameters:
    -----------
    trm : pandas.DataFrame
        Total Relation Matrix
    decimal_places : int, optional
        Number of decimal places for rounding
        
    Returns:
    --------
    pandas.Series
        R values indexed by barrier code
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    # R = sum of each column (how much each barrier receives/is influenced by others)
    r_values = trm.sum(axis=0).round(decimal_places)
    r_values.name = 'R'
    
    return r_values


def create_trm_with_d_and_r(trm, d_values, r_values, decimal_places=None):
    """
    Create TRM matrix with D column and R row appended.
    
    Parameters:
    -----------
    trm : pandas.DataFrame
        Total Relation Matrix
    d_values : pandas.Series
        D (Dispatch) values for each barrier
    r_values : pandas.Series
        R (Receive) values for each barrier
    decimal_places : int, optional
        Number of decimal places for rounding
        
    Returns:
    --------
    pandas.DataFrame
        TRM with D column (row sums) and R row (column sums) appended
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    # Create a copy of TRM
    trm_dar = trm.copy()
    
    # Add D column (row sums) at the end
    trm_dar['D'] = d_values
    
    # Create R row (column sums)
    r_row = pd.Series(index=trm_dar.columns, dtype=float)
    
    # Fill R values for each barrier column
    for col in trm.columns:
        r_row[col] = r_values[col]
    
    # Calculate total sum for D column (sum of all D values = sum of all R values)
    total_sum = round(d_values.sum(), decimal_places)
    r_row['D'] = total_sum
    
    # Add R row at the bottom
    trm_dar.loc['R'] = r_row
    
    return trm_dar


def create_trm_dar_for_export(trm_dar, barrier_to_name):
    """
    Create TRM with D and R with full barrier names for Excel export.
    
    Parameters:
    -----------
    trm_dar : pandas.DataFrame
        TRM with D column and R row
    barrier_to_name : dict
        Mapping from barrier codes to full names
        
    Returns:
    --------
    pandas.DataFrame
        TRM with D and R, full names as row index, short codes as columns
    """
    export_df = trm_dar.copy()
    
    # Replace row index with full names
    new_index = []
    for code in export_df.index:
        if code == 'R':
            new_index.append('R (Receive/Effect)')
        else:
            new_index.append(barrier_to_name.get(code, code.upper()))
    
    export_df.index = new_index
    
    # Uppercase column headers (B1, B2, etc.) and rename D column
    new_columns = []
    for col in export_df.columns:
        if col == 'D':
            new_columns.append('D (Dispatch/Cause)')
        else:
            new_columns.append(col.upper())
    
    export_df.columns = new_columns
    
    return export_df


def create_d_r_summary_df(d_values, r_values, barrier_to_name, decimal_places=None):
    """
    Create summary DataFrame with D and R values per barrier.
    
    Parameters:
    -----------
    d_values : pandas.Series
        D (Dispatch) values
    r_values : pandas.Series
        R (Receive) values
    barrier_to_name : dict
        Mapping from codes to names
    decimal_places : int, optional
        Number of decimal places
        
    Returns:
    --------
    pandas.DataFrame
        Summary table with columns: Barrier, Code, D, R
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    rows = []
    for code in d_values.index:
        full_name = barrier_to_name.get(code, code.upper())
        rows.append({
            'Barrier': full_name,
            'Code': code.upper(),
            'D (Dispatch/Cause)': round(d_values[code], decimal_places),
            'R (Receive/Effect)': round(r_values[code], decimal_places)
        })
    
    # Add totals row
    rows.append({
        'Barrier': 'TOTAL',
        'Code': '-',
        'D (Dispatch/Cause)': round(d_values.sum(), decimal_places),
        'R (Receive/Effect)': round(r_values.sum(), decimal_places)
    })
    
    return pd.DataFrame(rows)


def create_metadata_df(trm, d_values, r_values, decimal_places=None):
    """
    Create metadata DataFrame with calculation information.
    
    Parameters:
    -----------
    trm : pandas.DataFrame
        Original TRM
    d_values : pandas.Series
        D values
    r_values : pandas.Series
        R values
    decimal_places : int, optional
        Number of decimal places used
        
    Returns:
    --------
    pandas.DataFrame
        Metadata information
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    n = len(trm)
    
    metadata = {
        'Parameter': [
            'Matrix Size',
            'D Formula',
            'R Formula',
            'D Interpretation',
            'R Interpretation',
            'Total D (Sum)',
            'Total R (Sum)',
            'Min D Value',
            'Max D Value',
            'Min R Value',
            'Max R Value',
            'Decimal Places'
        ],
        'Value': [
            f'{n} × {n}',
            'D_i = Σⱼ t_ij (sum of row i)',
            'R_i = Σᵢ t_ij (sum of column i)',
            'Total influence barrier i dispatches/causes to others',
            'Total influence barrier i receives from others',
            round(d_values.sum(), decimal_places),
            round(r_values.sum(), decimal_places),
            round(d_values.min(), decimal_places),
            round(d_values.max(), decimal_places),
            round(r_values.min(), decimal_places),
            round(r_values.max(), decimal_places),
            decimal_places
        ]
    }
    
    return pd.DataFrame(metadata)


def save_trm_dar(trm_dar_export, trm_dar_codes, d_r_summary, metadata_df, output_dir=None):
    """
    Save TRM with D and R to Excel file.
    
    Parameters:
    -----------
    trm_dar_export : pandas.DataFrame
        TRM with D and R (full names for export)
    trm_dar_codes : pandas.DataFrame
        TRM with D and R (short codes)
    d_r_summary : pandas.DataFrame
        D and R summary per barrier
    metadata_df : pandas.DataFrame
        Calculation metadata
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
    
    file_path = output_path / "dematel_trm_dar.xlsx"
    
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            trm_dar_export.to_excel(writer, sheet_name='TRM_with_D_and_R')
            trm_dar_codes.to_excel(writer, sheet_name='TRM_DAR_Raw_Codes')
            d_r_summary.to_excel(writer, sheet_name='D_R_Summary', index=False)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    print(f"TRM with D and R saved to: {file_path}")
    
    return file_path


def print_d_r_summary(trm, d_values, r_values, barrier_to_name, decimal_places=None):
    """
    Print summary of D and R calculation.
    
    Parameters:
    -----------
    trm : pandas.DataFrame
        Original TRM
    d_values : pandas.Series
        D values
    r_values : pandas.Series
        R values
    barrier_to_name : dict
        Barrier code to name mapping
    decimal_places : int, optional
        Number of decimal places
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    n = len(trm)
    
    print("-" * 80)
    print("DISPATCH (D) AND RECEIVE (R) CALCULATION SUMMARY")
    print("-" * 80)
    print(f"Matrix Size: {n} × {n}")
    print()
    
    print("FORMULAS:")
    print("  D (Dispatch/Cause) for factor i = Σⱼ t_ij (sum of row i)")
    print("  R (Receive/Effect) for factor i = Σᵢ t_ij (sum of column i)")
    print()
    
    print("INTERPRETATION:")
    print("  D = Total influence that barrier i dispatches/causes to all other barriers")
    print("  R = Total influence that barrier i receives from all other barriers")
    print("  Higher D → Barrier is a stronger cause/driver")
    print("  Higher R → Barrier is more affected/dependent")
    print()
    
    print("D AND R VALUES PER BARRIER:")
    print("-" * 80)
    print(f"{'Barrier':<50} {'D':<12} {'R':<12}")
    print("-" * 80)
    
    for code in d_values.index:
        name = barrier_to_name.get(code, code.upper())
        # Truncate name if too long
        display_name = name[:47] + "..." if len(name) > 47 else name
        d_val = d_values[code]
        r_val = r_values[code]
        print(f"{display_name:<50} {d_val:<12.{decimal_places}f} {r_val:<12.{decimal_places}f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<50} {d_values.sum():<12.{decimal_places}f} {r_values.sum():<12.{decimal_places}f}")
    print("-" * 80)
    print()
    
    # Identify highest D and R
    max_d_code = d_values.idxmax()
    max_r_code = r_values.idxmax()
    min_d_code = d_values.idxmin()
    min_r_code = r_values.idxmin()
    
    print("KEY FINDINGS:")
    print(f"  Highest D (strongest cause):  {barrier_to_name.get(max_d_code, max_d_code.upper())}")
    print(f"                                D = {d_values[max_d_code]:.{decimal_places}f}")
    print(f"  Lowest D (weakest cause):     {barrier_to_name.get(min_d_code, min_d_code.upper())}")
    print(f"                                D = {d_values[min_d_code]:.{decimal_places}f}")
    print()
    print(f"  Highest R (most affected):    {barrier_to_name.get(max_r_code, max_r_code.upper())}")
    print(f"                                R = {r_values[max_r_code]:.{decimal_places}f}")
    print(f"  Lowest R (least affected):    {barrier_to_name.get(min_r_code, min_r_code.upper())}")
    print(f"                                R = {r_values[min_r_code]:.{decimal_places}f}")
    print("-" * 80)


def calculate_d_and_r(trm=None, barrier_names=None, decimal_places=None, save=True):
    """
    Main function to calculate D (Dispatch) and R (Receive) from TRM.
    
    D (Dispatch/Cause) = Sum of row i = Total influence barrier i causes
    R (Receive/Effect) = Sum of column i = Total influence barrier i receives
    
    Parameters:
    -----------
    trm : pandas.DataFrame, optional
        Total Relation Matrix from Module 3.
        If None, loads from dematel_trm.xlsx
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names.
        If None, uses BARRIER_NAMES from Module 1.
    decimal_places : int, optional
        Number of decimal places for D and R values.
        If None, uses DECIMAL_PLACES (default: 4).
    save : bool, optional
        Whether to save outputs to Excel file.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'trm': Original TRM DataFrame
        - 'd_values': D (Dispatch) values Series
        - 'r_values': R (Receive) values Series
        - 'trm_dar': TRM with D column and R row (short codes)
        - 'trm_dar_export': TRM with D and R (full names for export)
        - 'd_r_summary': Summary DataFrame with D and R per barrier
        - 'barriers': List of barrier codes
        - 'barrier_to_name': Mapping of codes to names
        - 'output_file': Path to saved file (if save=True)
    """
    print("\n" + "=" * 80)
    print("MODULE 4: DEMATEL - DISPATCH (D) AND RECEIVE (R) CALCULATION")
    print("=" * 80 + "\n")
    
    # Step 1: Load or use provided TRM
    if trm is None:
        print("Loading Total Relation Matrix from Module 3...")
        trm = load_trm_from_file()
        print(f"Loaded TRM: {len(trm)} × {len(trm)} matrix.\n")
    else:
        print(f"Using provided TRM ({len(trm)} × {len(trm)} matrix).\n")
    
    # Step 2: Extract barriers and mapping
    barriers = list(trm.index)
    barrier_to_name = extract_barrier_mapping(trm, barrier_names)
    print(f"Barriers: {[b.upper() for b in barriers]}\n")
    
    # Step 3: Set decimal places
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    # Step 4: Calculate D (Dispatch/Cause) - row sums
    print("Calculating D (Dispatch/Cause) = sum of each row...")
    d_values = calculate_d_dispatch(trm, decimal_places)
    print(f"D values calculated for {len(d_values)} barriers.\n")
    
    # Step 5: Calculate R (Receive/Effect) - column sums
    print("Calculating R (Receive/Effect) = sum of each column...")
    r_values = calculate_r_receive(trm, decimal_places)
    print(f"R values calculated for {len(r_values)} barriers.\n")
    
    # Step 6: Create TRM with D and R
    print("Creating TRM with D column and R row...")
    trm_dar = create_trm_with_d_and_r(trm, d_values, r_values, decimal_places)
    print(f"Extended matrix created: {len(trm_dar)} × {len(trm_dar.columns)}\n")
    
    # Step 7: Print summary
    print_d_r_summary(trm, d_values, r_values, barrier_to_name, decimal_places)
    
    # Step 8: Create export version with full names
    trm_dar_export = create_trm_dar_for_export(trm_dar, barrier_to_name)
    
    # Step 9: Create D and R summary DataFrame
    d_r_summary = create_d_r_summary_df(d_values, r_values, barrier_to_name, decimal_places)
    
    # Step 10: Create metadata DataFrame
    metadata_df = create_metadata_df(trm, d_values, r_values, decimal_places)
    
    # Step 11: Save outputs
    output_file = None
    if save:
        print("\nSaving outputs...")
        output_file = save_trm_dar(trm_dar_export, trm_dar, d_r_summary, metadata_df)
    
    # Prepare results dictionary
    results = {
        'trm': trm,
        'd_values': d_values,
        'r_values': r_values,
        'trm_dar': trm_dar,
        'trm_dar_export': trm_dar_export,
        'd_r_summary': d_r_summary,
        'barriers': barriers,
        'barrier_to_name': barrier_to_name,
        'output_file': output_file
    }
    
    print("\n" + "=" * 80)
    print("MODULE 4 COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")
    
    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 4 in standalone mode...")
    results = calculate_d_and_r()
    
    # Display the TRM with D and R
    print("\n" + "=" * 80)
    print("TOTAL RELATION MATRIX WITH D AND R")
    print("=" * 80)
    print("D (Dispatch/Cause) = Row sum = Total influence dispatched")
    print("R (Receive/Effect) = Column sum = Total influence received")
    print()
    print(results['trm_dar'].to_string())
    
    print("\n" + "=" * 80)
    print("TRM WITH D AND R (For Export)")
    print("=" * 80)
    print(results['trm_dar_export'].to_string())

