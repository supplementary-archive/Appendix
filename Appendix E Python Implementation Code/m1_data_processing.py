# SPDX-License-Identifier: PROPRIETARY
# File: m1_data_processing.py
# Purpose: Module 1 - Data loading, filtering, and processing for DEMATEL analysis

import pandas as pd
import numpy as np
import re
from pathlib import Path

# =============================================================================
# CUSTOMIZABLE PARAMETERS - MODIFY THESE AS NEEDED
# =============================================================================

# Data source file path (relative to this script or absolute path)
DATA_FILE = "data_projectconnector_2025-12-07_13-58.xlsx"

# Barrier names - customize these for your study
# Keys must match the barrier codes in your data (b1, b2, etc.)
BARRIER_NAMES = {
    'b1': 'B1: Lack of time',
    'b2': 'B2: Lack of financial and human resources',
    'b3': 'B3: Lack of trust',
    'b4': 'B4: Organizational policies not prioritizing KM',
    'b5': 'B5: Language and geographical barriers',
    'b6': 'B6: Poor digital platform integration and usability',
    'b7': 'B7: Lack of systematic KM processes',
    'b8': 'B8: Diverse stakeholders and understanding gaps',
}

# Role filter configuration
# Set to None for all data, or specify role number (1, 2, 3, or 4)
# 1 = Academia/Research
# 2 = Industry/Private Sector
# 3 = Public Authority/Government
# 4 = Civil Society/Community citizen
ROLE_FILTER = None

# Role names mapping
ROLE_NAMES = {
    1: 'Academia/Research',
    2: 'Industry/Private Sector',
    3: 'Public Authority/Government',
    4: 'Civil Society/Community citizen'
}

# Output directory
OUTPUT_DIR_DEMATEL = "output/dematel/"

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def get_script_directory():
    """Get the directory where this script is located."""
    return Path(__file__).parent.resolve()


def load_data(data_file=None):
    """
    Load data from Excel file with row 2 as column headers.
    
    Parameters:
    -----------
    data_file : str, optional
        Path to the Excel file. If None, uses DATA_FILE from configuration.
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data with proper column headers
    """
    if data_file is None:
        data_file = DATA_FILE
    
    # Resolve path relative to script directory
    script_dir = get_script_directory()
    file_path = script_dir / data_file
    
    # Load Excel with row 2 (index 1) as header
    data = pd.read_excel(file_path, header=1)
    
    # Convert column names to lowercase for consistency
    data.columns = data.columns.str.lower()
    
    return data


def identify_columns(data):
    """
    Identify influence columns and rating columns from the data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The loaded data
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'influence_cols': list of influence columns (bi_bj pattern)
        - 'rating_cols': list of rating columns (bi_bj_rating pattern)
        - 'barriers': sorted list of unique barrier codes
        - 'role_col': name of role column
    """
    influence_cols = []
    rating_cols = []
    barriers = set()
    
    # Pattern for influence columns: bi_bj (e.g., b1_b2)
    influence_pattern = re.compile(r'^b(\d+)_b(\d+)$')
    
    # Pattern for rating columns: bi_bj_rating (e.g., b1_b2_rating)
    rating_pattern = re.compile(r'^b(\d+)_b(\d+)_rating$')
    
    for col in data.columns:
        col_lower = col.lower()
        
        # Check for influence columns
        influence_match = influence_pattern.match(col_lower)
        if influence_match:
            influence_cols.append(col)
            barriers.add(f'b{influence_match.group(1)}')
            barriers.add(f'b{influence_match.group(2)}')
        
        # Check for rating columns
        rating_match = rating_pattern.match(col_lower)
        if rating_match:
            rating_cols.append(col)
    
    # Sort barriers numerically (b1, b2, ..., b8)
    barriers = sorted(list(barriers), key=lambda x: int(x[1:]))
    
    # Identify role column (check multiple possible names)
    role_col = None
    for possible_name in ['a902', 'role', 'Role', 'ROLE']:
        if possible_name in data.columns:
            role_col = possible_name
            break
    
    return {
        'influence_cols': sorted(influence_cols),
        'rating_cols': sorted(rating_cols),
        'barriers': barriers,
        'role_col': role_col
    }


def print_data_summary(data, columns_info, barrier_names=None):
    """
    Print a summary of the loaded data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The loaded data
    columns_info : dict
        Dictionary from identify_columns()
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES
    
    print("=" * 80)
    print("DATA SUMMARY")
    print("=" * 80)
    print(f"Total Barriers: {len(columns_info['barriers'])}")
    print(f"Total Influence Pairs: {len(columns_info['influence_cols'])}")
    print(f"Total Stakeholders: {len(data)}")
    print("-" * 80)
    
    # Print barrier names
    print("Barrier Names:")
    for barrier in columns_info['barriers']:
        name = barrier_names.get(barrier, f'{barrier.upper()}: (name not defined)')
        print(f"  {name}")
    print("-" * 80)
    
    # Print role distribution if role column exists
    if columns_info['role_col']:
        print("Role Distribution:")
        role_counts = data[columns_info['role_col']].value_counts().sort_index()
        for role_num, count in role_counts.items():
            role_name = ROLE_NAMES.get(int(role_num), f'Unknown Role ({role_num})')
            print(f"  {role_name}: {count}")
    print("=" * 80)


def filter_by_role(data, role_filter=None, role_col='a902'):
    """
    Filter data by stakeholder role.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The loaded data
    role_filter : int or None
        Role number to filter by (1-4), or None for all data
    role_col : str
        Name of the role column
        
    Returns:
    --------
    pandas.DataFrame
        Filtered data
    """
    if role_filter is None:
        print(f"Selected Data: All stakeholders ({len(data)} rows)")
        return data.copy()
    
    if role_col not in data.columns:
        print(f"Warning: Role column '{role_col}' not found. Using all data.")
        return data.copy()
    
    filtered_data = data[data[role_col] == role_filter].copy()
    role_name = ROLE_NAMES.get(role_filter, f'Role {role_filter}')
    print(f"Selected Data: {role_name} ({len(filtered_data)} rows)")
    
    return filtered_data


def calculate_dematel_drm(data, rating_cols, barriers, barrier_names=None):
    """
    Calculate DEMATEL Direct Relation Matrix using arithmetic mean.
    
    Formula: A = (1/K) * Σ(Ak)
    Where:
    - K = number of stakeholders
    - Ak = response matrix from stakeholder k
    
    Rating values:
    - 1-4: Direct influence rating
    - NA or -9: Treated as 0 (no influence)
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The filtered data
    rating_cols : list
        List of rating column names
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
        
    Returns:
    --------
    pandas.DataFrame
        Direct Relation Matrix with barriers as rows and columns
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES
    
    n_barriers = len(barriers)
    n_stakeholders = len(data)
    
    # Create full barrier names for row index (e.g., "B1: Lack of time")
    full_names = [barrier_names.get(b, b.upper()) for b in barriers]
    
    # Create short codes for column headers (e.g., "B1", "B2")
    short_codes = [b.upper() for b in barriers]
    
    # Initialize DRM with zeros (full names for rows, short codes for columns)
    drm = pd.DataFrame(
        np.zeros((n_barriers, n_barriers)),
        index=full_names,
        columns=short_codes
    )
    
    # Create mapping from barrier code to index
    barrier_to_idx = {b: i for i, b in enumerate(barriers)}
    
    # Process each rating column
    for col in rating_cols:
        # Extract barrier codes from column name
        match = re.match(r'^(b\d+)_(b\d+)_rating$', col.lower())
        if not match:
            continue
        
        b1, b2 = match.groups()
        
        if b1 not in barrier_to_idx or b2 not in barrier_to_idx:
            continue
        
        # Get ratings, replace NA and -9 with 0
        ratings = data[col].copy()
        ratings = ratings.fillna(0)
        ratings = ratings.replace(-9, 0)
        
        # Calculate arithmetic mean
        arithmetic_mean = ratings.sum() / n_stakeholders
        
        # Get full name for row and short code for column
        row_name = barrier_names.get(b1, b1.upper())
        col_code = b2.upper()
        
        # Set value in DRM
        drm.loc[row_name, col_code] = round(arithmetic_mean, 4)
    
    # Ensure diagonal is 0
    for i, name in enumerate(full_names):
        drm.loc[name, short_codes[i]] = 0
    
    return drm


def save_outputs(dematel_drm, output_dir_dematel=None):
    """
    Save DEMATEL Direct Relation Matrix to Excel file.

    Parameters:
    -----------
    dematel_drm : pandas.DataFrame
        DEMATEL Direct Relation Matrix
    output_dir_dematel : str, optional
        Output directory for DEMATEL files
    """
    if output_dir_dematel is None:
        output_dir_dematel = OUTPUT_DIR_DEMATEL

    # Get script directory for relative paths
    script_dir = get_script_directory()

    # Create output directory if it doesn't exist
    dematel_path = script_dir / output_dir_dematel
    dematel_path.mkdir(parents=True, exist_ok=True)

    # Save DEMATEL DRM
    dematel_file = dematel_path / "dematel_drm.xlsx"
    dematel_drm.to_excel(dematel_file, sheet_name='Direct_Relation_Matrix')
    print(f"DEMATEL DRM saved to: {dematel_file}")


def process_data(data_file=None, role_filter=None, barrier_names=None, save=True):
    """
    Main function to process data for DEMATEL analysis.

    Parameters:
    -----------
    data_file : str, optional
        Path to the Excel file
    role_filter : int or None, optional
        Role to filter by (1-4) or None for all data
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    save : bool, optional
        Whether to save outputs to Excel files

    Returns:
    --------
    dict
        Dictionary containing all results:
        - 'barriers': List of barrier codes
        - 'barrier_names': Barrier name mapping
        - 'n_barriers': Number of barriers
        - 'n_influences': Number of influence pairs
        - 'n_stakeholders': Total stakeholders
        - 'n_selected': Stakeholders after filtering
        - 'dematel_drm': DEMATEL Direct Relation Matrix DataFrame
        - 'role_filter': Applied role filter
    """
    # Use defaults if not provided
    if role_filter is None:
        role_filter = ROLE_FILTER
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    print("\n" + "=" * 80)
    print("MODULE 1: DATA PROCESSING")
    print("=" * 80 + "\n")

    # Step 1: Load data
    print("Loading data...")
    data = load_data(data_file)
    print(f"Data loaded successfully: {len(data)} rows\n")

    # Step 2: Identify columns
    columns_info = identify_columns(data)

    # Step 3: Print summary
    print_data_summary(data, columns_info, barrier_names)

    # Step 4: Filter by role
    print()
    filtered_data = filter_by_role(data, role_filter, columns_info['role_col'])
    print()

    # Step 5: Calculate DEMATEL DRM
    print("Calculating DEMATEL Direct Relation Matrix...")
    dematel_drm = calculate_dematel_drm(
        filtered_data,
        columns_info['rating_cols'],
        columns_info['barriers'],
        barrier_names
    )
    print(f"DEMATEL DRM calculated: {len(dematel_drm)}x{len(dematel_drm)} matrix\n")

    # Step 6: Save outputs
    if save:
        print("Saving outputs...")
        save_outputs(dematel_drm)
        print()

    # Prepare results dictionary
    results = {
        'barriers': columns_info['barriers'],
        'barrier_names': barrier_names,
        'n_barriers': len(columns_info['barriers']),
        'n_influences': len(columns_info['influence_cols']),
        'n_stakeholders': len(data),
        'n_selected': len(filtered_data),
        'dematel_drm': dematel_drm,
        'role_filter': role_filter,
        'filtered_data': filtered_data,
        'columns_info': columns_info
    }

    print("=" * 80)
    print("MODULE 1 COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run with default configuration
    results = process_data()

    # Display sample outputs
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUTS")
    print("=" * 80)

    print("\n--- DEMATEL Direct Relation Matrix ---")
    print(results['dematel_drm'].to_string())

