# SPDX-License-Identifier: PROPRIETARY
# File: m2_dematel_normalized_drm.py
# Purpose: Module 2 - Create Normalized Direct Relation Matrix (Normalized DRM) for DEMATEL analysis

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

# Number of decimal places for normalized values
DECIMAL_PLACES = 4

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def load_drm_from_file(file_path=None):
    """
    Load Direct Relation Matrix (DRM) from Module 1 Excel file.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the DRM Excel file. If None, uses default path.
        
    Returns:
    --------
    pandas.DataFrame
        Direct Relation Matrix with barrier names as index and short codes as columns
    """
    if file_path is None:
        script_dir = get_script_directory()
        file_path = script_dir / OUTPUT_DIR_DEMATEL / "dematel_drm.xlsx"
    
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"DRM file not found: {file_path}\n"
            "Please run Module 1 (m1_data_processing.py) first to generate the Direct Relation Matrix."
        )
    
    try:
        # Load the DRM (has full names as index, short codes as columns)
        drm = pd.read_excel(file_path, sheet_name='Direct_Relation_Matrix', index_col=0)
    except PermissionError:
        raise PermissionError(
            f"Cannot access file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    return drm


def extract_barrier_mapping(drm):
    """
    Extract barrier code to full name mapping from DRM.
    
    Parameters:
    -----------
    drm : pandas.DataFrame
        Direct Relation Matrix with full names as index
        
    Returns:
    --------
    tuple
        - barriers: List of barrier codes (lowercase)
        - barrier_to_name: Dict mapping code to full name
    """
    # Columns are short codes (B1, B2, etc.)
    barriers = [col.lower() for col in drm.columns]
    
    # Index contains full names
    barrier_to_name = {}
    for i, full_name in enumerate(drm.index):
        code = barriers[i]
        barrier_to_name[code] = full_name
    
    return barriers, barrier_to_name


def create_drm_with_codes(drm):
    """
    Create DRM with short codes as both index and columns for calculations.
    
    Parameters:
    -----------
    drm : pandas.DataFrame
        Original DRM with full names as index
        
    Returns:
    --------
    pandas.DataFrame
        DRM with short codes (lowercase) as both index and columns
    """
    # Get short codes from columns
    codes = [col.lower() for col in drm.columns]
    
    # Create new DataFrame with codes
    drm_codes = drm.copy()
    drm_codes.index = codes
    drm_codes.columns = codes
    
    return drm_codes


def calculate_row_sums(drm):
    """
    Calculate the sum of each row in the matrix.
    
    Parameters:
    -----------
    drm : pandas.DataFrame
        Direct Relation Matrix
        
    Returns:
    --------
    pandas.Series
        Row sums indexed by barrier code
    """
    return drm.sum(axis=1)


def calculate_column_sums(drm):
    """
    Calculate the sum of each column in the matrix.
    
    Parameters:
    -----------
    drm : pandas.DataFrame
        Direct Relation Matrix
        
    Returns:
    --------
    pandas.Series
        Column sums indexed by barrier code
    """
    return drm.sum(axis=0)


def calculate_normalization_factor(drm):
    """
    Calculate the normalization factor S.
    
    Formula: S = max(max_row_sum, max_col_sum)
    
    Where:
    - max_row_sum = maximum of all row sums
    - max_col_sum = maximum of all column sums
    
    Parameters:
    -----------
    drm : pandas.DataFrame
        Direct Relation Matrix
        
    Returns:
    --------
    tuple
        - s: Normalization factor
        - row_sums: Series of row sums
        - col_sums: Series of column sums
        - max_row_sum: Maximum row sum value
        - max_col_sum: Maximum column sum value
    """
    row_sums = calculate_row_sums(drm)
    col_sums = calculate_column_sums(drm)
    
    max_row_sum = row_sums.max()
    max_col_sum = col_sums.max()
    
    # S is the larger of the two maximums
    s = max(max_row_sum, max_col_sum)
    
    return s, row_sums, col_sums, max_row_sum, max_col_sum


def normalize_matrix(drm, s, decimal_places=None):
    """
    Normalize the Direct Relation Matrix.
    
    Formula: D = A / S
    
    Parameters:
    -----------
    drm : pandas.DataFrame
        Direct Relation Matrix (A)
    s : float
        Normalization factor (S)
    decimal_places : int, optional
        Number of decimal places for rounding. If None, uses DECIMAL_PLACES.
        
    Returns:
    --------
    pandas.DataFrame
        Normalized Direct Relation Matrix (D) with values in [0, 1]
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    # Element-wise division
    normalized = drm / s
    
    # Round to specified decimal places
    normalized = normalized.round(decimal_places)
    
    return normalized


def validate_normalization(normalized_drm):
    """
    Validate that all normalized values are in the range [0, 1].
    
    Parameters:
    -----------
    normalized_drm : pandas.DataFrame
        Normalized Direct Relation Matrix
        
    Returns:
    --------
    dict
        Validation results with keys:
        - 'valid': bool indicating if all values are valid
        - 'min_value': Minimum value in matrix
        - 'max_value': Maximum value in matrix
        - 'issues': List of any issues found
    """
    min_val = normalized_drm.min().min()
    max_val = normalized_drm.max().max()
    
    issues = []
    
    if min_val < 0:
        issues.append(f"Found negative values (min: {min_val})")
    
    if max_val > 1:
        issues.append(f"Found values > 1 (max: {max_val})")
    
    return {
        'valid': len(issues) == 0,
        'min_value': min_val,
        'max_value': max_val,
        'issues': issues
    }


def create_normalized_drm_for_export(normalized_drm, barrier_to_name):
    """
    Create Normalized DRM with full barrier names for Excel export.
    
    Parameters:
    -----------
    normalized_drm : pandas.DataFrame
        Normalized DRM with short codes
    barrier_to_name : dict
        Mapping from barrier codes to full names
        
    Returns:
    --------
    pandas.DataFrame
        Normalized DRM with full names as row index and short codes as columns
    """
    export_df = normalized_drm.copy()
    
    # Replace row index with full names
    new_index = [barrier_to_name.get(code, code.upper()) for code in export_df.index]
    export_df.index = new_index
    
    # Uppercase column headers (B1, B2, etc.)
    export_df.columns = [col.upper() for col in export_df.columns]
    
    return export_df


def create_normalization_info_df(s, row_sums, col_sums, max_row_sum, max_col_sum, 
                                  barrier_to_name, validation):
    """
    Create DataFrame with normalization information for export.
    
    Parameters:
    -----------
    s : float
        Normalization factor
    row_sums : pandas.Series
        Row sums for each barrier
    col_sums : pandas.Series
        Column sums for each barrier
    max_row_sum : float
        Maximum row sum
    max_col_sum : float
        Maximum column sum
    barrier_to_name : dict
        Mapping from codes to names
    validation : dict
        Validation results
        
    Returns:
    --------
    tuple
        - info_df: DataFrame with summary information
        - sums_df: DataFrame with row and column sums per barrier
    """
    # Summary information
    info_data = {
        'Parameter': [
            'Normalization Factor (S)',
            'Maximum Row Sum',
            'Maximum Column Sum',
            'S Determined By',
            'Normalized Min Value',
            'Normalized Max Value',
            'Validation Status'
        ],
        'Value': [
            round(s, DECIMAL_PLACES),
            round(max_row_sum, DECIMAL_PLACES),
            round(max_col_sum, DECIMAL_PLACES),
            'Row Sum' if max_row_sum >= max_col_sum else 'Column Sum',
            round(validation['min_value'], DECIMAL_PLACES),
            round(validation['max_value'], DECIMAL_PLACES),
            'Valid' if validation['valid'] else 'Issues Found'
        ]
    }
    info_df = pd.DataFrame(info_data)
    
    # Per-barrier sums
    sums_data = []
    for code in row_sums.index:
        full_name = barrier_to_name.get(code, code.upper())
        sums_data.append({
            'Barrier': full_name,
            'Code': code.upper(),
            'Row Sum': round(row_sums[code], DECIMAL_PLACES),
            'Column Sum': round(col_sums[code], DECIMAL_PLACES)
        })
    
    sums_df = pd.DataFrame(sums_data)
    
    return info_df, sums_df


def save_normalized_drm(normalized_drm_export, normalized_drm_codes, info_df, sums_df, 
                        output_dir=None):
    """
    Save Normalized DRM and related information to Excel file.
    
    Parameters:
    -----------
    normalized_drm_export : pandas.DataFrame
        Normalized DRM with full names (for export)
    normalized_drm_codes : pandas.DataFrame
        Normalized DRM with short codes (for chaining)
    info_df : pandas.DataFrame
        Normalization summary information
    sums_df : pandas.DataFrame
        Row and column sums per barrier
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
    
    file_path = output_path / "dematel_normalized_drm.xlsx"
    
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            normalized_drm_export.to_excel(writer, sheet_name='Normalized_DRM')
            normalized_drm_codes.to_excel(writer, sheet_name='NDRM_Raw_Codes')
            info_df.to_excel(writer, sheet_name='Normalization_Info', index=False)
            sums_df.to_excel(writer, sheet_name='Row_Column_Sums', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    print(f"Normalized DRM saved to: {file_path}")
    
    return file_path


def print_normalization_summary(drm, normalized_drm, s, row_sums, col_sums, 
                                 max_row_sum, max_col_sum, validation, barrier_to_name):
    """
    Print summary of the normalization process.
    
    Parameters:
    -----------
    drm : pandas.DataFrame
        Original Direct Relation Matrix
    normalized_drm : pandas.DataFrame
        Normalized Direct Relation Matrix
    s : float
        Normalization factor
    row_sums : pandas.Series
        Row sums
    col_sums : pandas.Series
        Column sums
    max_row_sum : float
        Maximum row sum
    max_col_sum : float
        Maximum column sum
    validation : dict
        Validation results
    barrier_to_name : dict
        Barrier code to name mapping
    """
    print("-" * 70)
    print("NORMALIZATION SUMMARY")
    print("-" * 70)
    print(f"Matrix Size: {len(drm)} x {len(drm)}")
    print()
    
    # Row and column sums
    print("ROW SUMS (Σⱼ aᵢⱼ for each row i):")
    for code in row_sums.index:
        name = barrier_to_name.get(code, code.upper())
        # Truncate name if too long
        display_name = name[:45] + "..." if len(name) > 45 else name
        print(f"  {display_name:<48} {row_sums[code]:.4f}")
    print(f"  {'Maximum Row Sum:':<48} {max_row_sum:.4f}")
    print()
    
    print("COLUMN SUMS (Σᵢ aᵢⱼ for each column j):")
    for code in col_sums.index:
        name = barrier_to_name.get(code, code.upper())
        display_name = name[:45] + "..." if len(name) > 45 else name
        print(f"  {display_name:<48} {col_sums[code]:.4f}")
    print(f"  {'Maximum Column Sum:':<48} {max_col_sum:.4f}")
    print()
    
    print("NORMALIZATION FACTOR (S):")
    print(f"  S = max(max_row_sum, max_col_sum)")
    print(f"  S = max({max_row_sum:.4f}, {max_col_sum:.4f})")
    print(f"  S = {s:.4f}")
    print()
    
    determining_factor = "Row Sum" if max_row_sum >= max_col_sum else "Column Sum"
    print(f"  → S determined by: {determining_factor}")
    print()
    
    print("NORMALIZATION FORMULA:")
    print(f"  D = A / S")
    print(f"  Each element dᵢⱼ = aᵢⱼ / {s:.4f}")
    print()
    
    print("VALIDATION:")
    print(f"  Min normalized value: {validation['min_value']:.4f}")
    print(f"  Max normalized value: {validation['max_value']:.4f}")
    if validation['valid']:
        print("  ✓ All values are in valid range [0, 1]")
    else:
        print("  ✗ Validation issues found:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    
    print("-" * 70)


def create_normalized_drm(drm=None, barrier_names=None, decimal_places=None, save=True):
    """
    Main function to create Normalized Direct Relation Matrix for DEMATEL.
    
    Formula: D = A / S
    Where S = max(max_row_sum, max_col_sum)
    
    Parameters:
    -----------
    drm : pandas.DataFrame, optional
        Direct Relation Matrix from Module 1.
        If None, loads from dematel_drm.xlsx
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names.
        If None, extracted from DRM or uses BARRIER_NAMES.
    decimal_places : int, optional
        Number of decimal places for normalized values.
        If None, uses DECIMAL_PLACES (default: 4).
    save : bool, optional
        Whether to save outputs to Excel file.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'normalized_drm': Normalized DRM with short codes
        - 'normalized_drm_export': Normalized DRM with full names
        - 'original_drm': Original DRM (with codes)
        - 's': Normalization factor
        - 'row_sums': Row sums for each barrier
        - 'col_sums': Column sums for each barrier
        - 'max_row_sum': Maximum row sum
        - 'max_col_sum': Maximum column sum
        - 'validation': Validation results
        - 'barriers': List of barrier codes
        - 'barrier_to_name': Mapping of codes to names
        - 'output_file': Path to saved file (if save=True)
    """
    print("\n" + "=" * 70)
    print("MODULE 2: DEMATEL - NORMALIZED DIRECT RELATION MATRIX")
    print("=" * 70 + "\n")
    
    # Step 1: Load or use provided DRM
    if drm is None:
        print("Loading Direct Relation Matrix from Module 1...")
        drm = load_drm_from_file()
        print(f"Loaded DRM: {len(drm)} x {len(drm)} matrix.\n")
    else:
        print(f"Using provided DRM ({len(drm)} x {len(drm)} matrix).\n")
    
    # Step 2: Extract barrier mapping
    barriers, barrier_to_name = extract_barrier_mapping(drm)
    print(f"Barriers: {[b.upper() for b in barriers]}\n")
    
    # Override with provided barrier_names if available
    if barrier_names is not None:
        barrier_to_name = {**barrier_to_name, **{k.lower(): v for k, v in barrier_names.items()}}
    
    # Step 3: Create DRM with codes for calculations
    drm_codes = create_drm_with_codes(drm)
    
    # Step 4: Calculate normalization factor S
    print("Calculating normalization factor S...")
    s, row_sums, col_sums, max_row_sum, max_col_sum = calculate_normalization_factor(drm_codes)
    print(f"S = max({max_row_sum:.4f}, {max_col_sum:.4f}) = {s:.4f}\n")
    
    # Step 5: Normalize the matrix
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    print(f"Normalizing matrix (D = A / S)...")
    normalized_drm = normalize_matrix(drm_codes, s, decimal_places)
    print(f"Normalization complete (values rounded to {decimal_places} decimal places).\n")
    
    # Step 6: Validate normalization
    print("Validating normalized values...")
    validation = validate_normalization(normalized_drm)
    if validation['valid']:
        print("✓ All values are in valid range [0, 1].\n")
    else:
        print("✗ Validation issues found!")
        for issue in validation['issues']:
            print(f"  - {issue}")
        print()
    
    # Step 7: Print summary
    print_normalization_summary(
        drm_codes, normalized_drm, s, row_sums, col_sums,
        max_row_sum, max_col_sum, validation, barrier_to_name
    )
    
    # Step 8: Create export version with full names
    normalized_drm_export = create_normalized_drm_for_export(normalized_drm, barrier_to_name)
    
    # Step 9: Create info DataFrames
    info_df, sums_df = create_normalization_info_df(
        s, row_sums, col_sums, max_row_sum, max_col_sum, barrier_to_name, validation
    )
    
    # Step 10: Save outputs
    output_file = None
    if save:
        print("\nSaving outputs...")
        output_file = save_normalized_drm(
            normalized_drm_export, normalized_drm, info_df, sums_df
        )
    
    # Prepare results dictionary
    results = {
        'normalized_drm': normalized_drm,
        'normalized_drm_export': normalized_drm_export,
        'original_drm': drm_codes,
        's': s,
        'row_sums': row_sums,
        'col_sums': col_sums,
        'max_row_sum': max_row_sum,
        'max_col_sum': max_col_sum,
        'validation': validation,
        'barriers': barriers,
        'barrier_to_name': barrier_to_name,
        'output_file': output_file
    }
    
    print("\n" + "=" * 70)
    print("MODULE 2 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")
    
    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 2 in standalone mode...")
    results = create_normalized_drm()
    
    # Display the Normalized DRM
    print("\n" + "=" * 70)
    print("NORMALIZED DIRECT RELATION MATRIX (D = A / S)")
    print("=" * 70)
    print(f"Normalization Factor S = {results['s']:.4f}")
    print()
    print(results['normalized_drm'].to_string())
    
    print("\n" + "=" * 70)
    print("NORMALIZED DRM (For Export)")
    print("=" * 70)
    print(results['normalized_drm_export'].to_string())

