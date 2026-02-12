# SPDX-License-Identifier: PROPRIETARY
# File: m3_dematel_trm.py
# Purpose: Module 3 - Calculate Total Relation Matrix (TRM) for DEMATEL analysis

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

# Number of decimal places for TRM values
DECIMAL_PLACES = 4

# Convergence check - verify matrix series converges
CHECK_CONVERGENCE = True

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def load_normalized_drm_from_file(file_path=None):
    """
    Load Normalized Direct Relation Matrix (D) from Module 2 Excel file.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the Normalized DRM Excel file. If None, uses default path.
        
    Returns:
    --------
    pandas.DataFrame
        Normalized DRM with barrier codes as index and columns
    """
    if file_path is None:
        script_dir = get_script_directory()
        file_path = script_dir / OUTPUT_DIR_DEMATEL / "dematel_normalized_drm.xlsx"
    
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Normalized DRM file not found: {file_path}\n"
            "Please run Module 2 (m2_dematel_normalized_drm.py) first to generate "
            "the Normalized Direct Relation Matrix."
        )
    
    try:
        # Load the raw codes sheet (with short barrier codes)
        normalized_drm = pd.read_excel(file_path, sheet_name='NDRM_Raw_Codes', index_col=0)
    except PermissionError:
        raise PermissionError(
            f"Cannot access file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    # Ensure column names and index are lowercase for consistency
    normalized_drm.columns = [str(col).lower() for col in normalized_drm.columns]
    normalized_drm.index = [str(idx).lower() for idx in normalized_drm.index]
    
    return normalized_drm


def extract_barrier_mapping(normalized_drm, barrier_names=None):
    """
    Extract barrier code to full name mapping.
    
    Parameters:
    -----------
    normalized_drm : pandas.DataFrame
        Normalized DRM with barrier codes
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
    for code in normalized_drm.index:
        barrier_to_name[code] = barrier_names.get(code, code.upper())
    
    return barrier_to_name


def create_identity_matrix(n):
    """
    Create an identity matrix of size n×n.
    
    Parameters:
    -----------
    n : int
        Size of the identity matrix
        
    Returns:
    --------
    numpy.ndarray
        Identity matrix I
    """
    return np.eye(n)


def calculate_i_minus_d(identity_matrix, d_matrix):
    """
    Calculate (I - D) matrix.
    
    Parameters:
    -----------
    identity_matrix : numpy.ndarray
        Identity matrix I
    d_matrix : numpy.ndarray
        Normalized Direct Relation Matrix D
        
    Returns:
    --------
    numpy.ndarray
        Result of (I - D)
    """
    return identity_matrix - d_matrix


def calculate_matrix_inverse(matrix):
    """
    Calculate the inverse of a matrix.
    
    Parameters:
    -----------
    matrix : numpy.ndarray
        Square matrix to invert
        
    Returns:
    --------
    tuple
        - inverse: numpy.ndarray of the inverted matrix
        - determinant: float, determinant of original matrix
        - is_singular: bool, whether matrix is singular
    """
    try:
        determinant = np.linalg.det(matrix)
        
        # Check if matrix is singular (determinant close to zero)
        if np.abs(determinant) < 1e-10:
            return None, determinant, True
        
        inverse = np.linalg.inv(matrix)
        return inverse, determinant, False
        
    except np.linalg.LinAlgError:
        return None, 0.0, True


def calculate_trm(d_matrix, i_minus_d_inverse, decimal_places=None):
    """
    Calculate Total Relation Matrix: T = D × (I - D)⁻¹
    
    Parameters:
    -----------
    d_matrix : numpy.ndarray
        Normalized Direct Relation Matrix D
    i_minus_d_inverse : numpy.ndarray
        Inverse of (I - D)
    decimal_places : int, optional
        Number of decimal places for rounding
        
    Returns:
    --------
    numpy.ndarray
        Total Relation Matrix T
    """
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    # Matrix multiplication: T = D × (I - D)⁻¹
    trm = np.matmul(d_matrix, i_minus_d_inverse)
    
    # Round to specified decimal places
    trm = np.round(trm, decimal_places)
    
    return trm


def verify_convergence(d_matrix):
    """
    Verify that the matrix series converges (spectral radius < 1).
    
    For the series D + D² + D³ + ... to converge, the spectral radius
    (largest absolute eigenvalue) of D must be less than 1.
    
    Parameters:
    -----------
    d_matrix : numpy.ndarray
        Normalized Direct Relation Matrix D
        
    Returns:
    --------
    dict
        Convergence information:
        - 'converges': bool
        - 'spectral_radius': float
        - 'eigenvalues': array of eigenvalues
    """
    eigenvalues = np.linalg.eigvals(d_matrix)
    spectral_radius = np.max(np.abs(eigenvalues))
    
    return {
        'converges': spectral_radius < 1,
        'spectral_radius': spectral_radius,
        'eigenvalues': eigenvalues
    }


def validate_trm(trm):
    """
    Validate the Total Relation Matrix.
    
    Parameters:
    -----------
    trm : numpy.ndarray
        Total Relation Matrix
        
    Returns:
    --------
    dict
        Validation results:
        - 'valid': bool
        - 'min_value': float
        - 'max_value': float
        - 'has_negative': bool
        - 'issues': list of issues found
    """
    min_val = np.min(trm)
    max_val = np.max(trm)
    has_negative = min_val < 0
    
    issues = []
    
    if has_negative:
        issues.append(f"Matrix contains negative values (min: {min_val:.4f})")
    
    if np.isnan(trm).any():
        issues.append("Matrix contains NaN values")
    
    if np.isinf(trm).any():
        issues.append("Matrix contains infinite values")
    
    return {
        'valid': len(issues) == 0,
        'min_value': min_val,
        'max_value': max_val,
        'has_negative': has_negative,
        'issues': issues
    }


def create_trm_dataframe(trm_array, barriers):
    """
    Convert TRM numpy array to pandas DataFrame with barrier codes.
    
    Parameters:
    -----------
    trm_array : numpy.ndarray
        Total Relation Matrix as numpy array
    barriers : list
        List of barrier codes
        
    Returns:
    --------
    pandas.DataFrame
        TRM with barrier codes as index and columns
    """
    return pd.DataFrame(trm_array, index=barriers, columns=barriers)


def create_trm_for_export(trm_df, barrier_to_name):
    """
    Create TRM with full barrier names for Excel export.
    
    Parameters:
    -----------
    trm_df : pandas.DataFrame
        TRM with short barrier codes
    barrier_to_name : dict
        Mapping from barrier codes to full names
        
    Returns:
    --------
    pandas.DataFrame
        TRM with full names as row index and short codes as columns
    """
    export_df = trm_df.copy()
    
    # Replace row index with full names
    new_index = [barrier_to_name.get(code, code.upper()) for code in export_df.index]
    export_df.index = new_index
    
    # Uppercase column headers (B1, B2, etc.)
    export_df.columns = [col.upper() for col in export_df.columns]
    
    return export_df


def create_intermediate_matrices_df(identity_matrix, i_minus_d, i_minus_d_inverse, barriers):
    """
    Create DataFrames for intermediate matrices (for verification).
    
    Parameters:
    -----------
    identity_matrix : numpy.ndarray
        Identity matrix I
    i_minus_d : numpy.ndarray
        (I - D) matrix
    i_minus_d_inverse : numpy.ndarray
        (I - D)⁻¹ matrix
    barriers : list
        List of barrier codes
        
    Returns:
    --------
    dict
        Dictionary with intermediate matrices as DataFrames
    """
    barriers_upper = [b.upper() for b in barriers]
    
    return {
        'identity': pd.DataFrame(identity_matrix, index=barriers_upper, columns=barriers_upper),
        'i_minus_d': pd.DataFrame(np.round(i_minus_d, DECIMAL_PLACES), 
                                   index=barriers_upper, columns=barriers_upper),
        'i_minus_d_inverse': pd.DataFrame(np.round(i_minus_d_inverse, DECIMAL_PLACES), 
                                           index=barriers_upper, columns=barriers_upper)
    }


def create_calculation_info_df(n, determinant, convergence_info, validation):
    """
    Create DataFrame with calculation information.
    
    Parameters:
    -----------
    n : int
        Matrix size
    determinant : float
        Determinant of (I - D)
    convergence_info : dict
        Convergence verification results
    validation : dict
        TRM validation results
        
    Returns:
    --------
    pandas.DataFrame
        Calculation information
    """
    info_data = {
        'Parameter': [
            'Matrix Size',
            'Formula',
            'Determinant of (I-D)',
            'Matrix Invertible',
            'Spectral Radius of D',
            'Series Converges',
            'TRM Min Value',
            'TRM Max Value',
            'Validation Status'
        ],
        'Value': [
            f"{n} × {n}",
            'T = D × (I - D)⁻¹',
            f"{determinant:.6f}",
            'Yes' if np.abs(determinant) >= 1e-10 else 'No',
            f"{convergence_info['spectral_radius']:.6f}",
            'Yes' if convergence_info['converges'] else 'No',
            f"{validation['min_value']:.{DECIMAL_PLACES}f}",
            f"{validation['max_value']:.{DECIMAL_PLACES}f}",
            'Valid' if validation['valid'] else 'Issues Found'
        ]
    }
    
    return pd.DataFrame(info_data)


def save_trm(trm_export, trm_codes, calculation_info, intermediate_matrices, output_dir=None):
    """
    Save Total Relation Matrix and related information to Excel file.
    
    Parameters:
    -----------
    trm_export : pandas.DataFrame
        TRM with full names (for export)
    trm_codes : pandas.DataFrame
        TRM with short codes (for chaining)
    calculation_info : pandas.DataFrame
        Calculation information
    intermediate_matrices : dict
        Dictionary with intermediate matrices
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
    
    file_path = output_path / "dematel_trm.xlsx"
    
    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            trm_export.to_excel(writer, sheet_name='Total_Relation_Matrix')
            trm_codes.to_excel(writer, sheet_name='TRM_Raw_Codes')
            calculation_info.to_excel(writer, sheet_name='Calculation_Info', index=False)
            intermediate_matrices['i_minus_d'].to_excel(writer, sheet_name='I_minus_D')
            intermediate_matrices['i_minus_d_inverse'].to_excel(writer, sheet_name='I_minus_D_Inverse')
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    print(f"Total Relation Matrix saved to: {file_path}")
    
    return file_path


def print_trm_summary(d_matrix, trm, determinant, convergence_info, validation, 
                       barriers, barrier_to_name):
    """
    Print summary of the TRM calculation process.
    
    Parameters:
    -----------
    d_matrix : numpy.ndarray
        Normalized Direct Relation Matrix
    trm : numpy.ndarray
        Total Relation Matrix
    determinant : float
        Determinant of (I - D)
    convergence_info : dict
        Convergence verification results
    validation : dict
        TRM validation results
    barriers : list
        List of barrier codes
    barrier_to_name : dict
        Barrier code to name mapping
    """
    n = len(barriers)
    
    print("-" * 70)
    print("TOTAL RELATION MATRIX (TRM) CALCULATION SUMMARY")
    print("-" * 70)
    print(f"Matrix Size: {n} × {n}")
    print()
    
    print("FORMULA:")
    print("  T = D × (I - D)⁻¹")
    print()
    print("  Where:")
    print("    D = Normalized Direct Relation Matrix (from Module 2)")
    print("    I = Identity Matrix")
    print("    T = Total Relation Matrix (captures direct + indirect effects)")
    print()
    
    print("MATRIX INVERSION:")
    print(f"  Determinant of (I - D): {determinant:.6f}")
    if np.abs(determinant) >= 1e-10:
        print("  ✓ Matrix (I - D) is invertible")
    else:
        print("  ✗ Matrix (I - D) is singular (not invertible)")
    print()
    
    print("CONVERGENCE CHECK:")
    print(f"  Spectral Radius of D: {convergence_info['spectral_radius']:.6f}")
    if convergence_info['converges']:
        print("  ✓ Spectral radius < 1, infinite series converges")
    else:
        print("  ✗ Spectral radius >= 1, convergence not guaranteed")
    print()
    
    print("VALIDATION:")
    print(f"  TRM Min Value: {validation['min_value']:.{DECIMAL_PLACES}f}")
    print(f"  TRM Max Value: {validation['max_value']:.{DECIMAL_PLACES}f}")
    if validation['valid']:
        print("  ✓ All values are valid")
    else:
        print("  ✗ Validation issues found:")
        for issue in validation['issues']:
            print(f"    - {issue}")
    print()
    
    print("TRM INTERPRETATION:")
    print("  • Each element tᵢⱼ represents the total (direct + indirect)")
    print("    influence that barrier i exerts on barrier j")
    print("  • Higher values indicate stronger total influence relationships")
    print("  • Diagonal elements represent self-reinforcing effects")
    print()
    
    print("-" * 70)


def create_total_relation_matrix(normalized_drm=None, barrier_names=None, 
                                  decimal_places=None, save=True):
    """
    Main function to calculate Total Relation Matrix (TRM) for DEMATEL.
    
    Formula: T = D × (I - D)⁻¹
    
    Where:
    - D = Normalized Direct Relation Matrix (from Module 2)
    - I = Identity Matrix
    - T = Total Relation Matrix (direct + indirect effects)
    
    Parameters:
    -----------
    normalized_drm : pandas.DataFrame, optional
        Normalized DRM from Module 2.
        If None, loads from dematel_normalized_drm.xlsx
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names.
        If None, uses BARRIER_NAMES from Module 1.
    decimal_places : int, optional
        Number of decimal places for TRM values.
        If None, uses DECIMAL_PLACES (default: 4).
    save : bool, optional
        Whether to save outputs to Excel file.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'trm': Total Relation Matrix DataFrame (short codes)
        - 'trm_export': TRM DataFrame (full names for export)
        - 'trm_array': TRM as numpy array
        - 'normalized_drm': Input normalized DRM
        - 'identity_matrix': Identity matrix used
        - 'i_minus_d': (I - D) matrix
        - 'i_minus_d_inverse': (I - D)⁻¹ matrix
        - 'determinant': Determinant of (I - D)
        - 'convergence_info': Convergence verification results
        - 'validation': TRM validation results
        - 'barriers': List of barrier codes
        - 'barrier_to_name': Mapping of codes to names
        - 'output_file': Path to saved file (if save=True)
    """
    print("\n" + "=" * 70)
    print("MODULE 3: DEMATEL - TOTAL RELATION MATRIX (TRM)")
    print("=" * 70 + "\n")
    
    # Step 1: Load or use provided Normalized DRM
    if normalized_drm is None:
        print("Loading Normalized Direct Relation Matrix from Module 2...")
        normalized_drm = load_normalized_drm_from_file()
        print(f"Loaded Normalized DRM: {len(normalized_drm)} × {len(normalized_drm)} matrix.\n")
    else:
        print(f"Using provided Normalized DRM ({len(normalized_drm)} × {len(normalized_drm)} matrix).\n")
    
    # Step 2: Extract barriers and mapping
    barriers = list(normalized_drm.index)
    barrier_to_name = extract_barrier_mapping(normalized_drm, barrier_names)
    n = len(barriers)
    print(f"Barriers: {[b.upper() for b in barriers]}\n")
    
    # Step 3: Convert to numpy array for calculations
    d_matrix = normalized_drm.values.astype(float)
    
    # Step 4: Create Identity Matrix
    print("Creating Identity Matrix (I)...")
    identity_matrix = create_identity_matrix(n)
    print(f"Identity Matrix created: {n} × {n}\n")
    
    # Step 5: Calculate (I - D)
    print("Calculating (I - D)...")
    i_minus_d = calculate_i_minus_d(identity_matrix, d_matrix)
    print("(I - D) calculated.\n")
    
    # Step 6: Calculate (I - D)⁻¹
    print("Calculating (I - D)⁻¹ (matrix inverse)...")
    i_minus_d_inverse, determinant, is_singular = calculate_matrix_inverse(i_minus_d)
    
    if is_singular:
        raise ValueError(
            f"Matrix (I - D) is singular and cannot be inverted.\n"
            f"Determinant: {determinant}\n"
            "This may indicate issues with the normalized DRM values."
        )
    
    print(f"Matrix inverse calculated. Determinant: {determinant:.6f}\n")
    
    # Step 7: Verify convergence (optional)
    convergence_info = {'converges': True, 'spectral_radius': 0.0, 'eigenvalues': []}
    if CHECK_CONVERGENCE:
        print("Verifying convergence (spectral radius check)...")
        convergence_info = verify_convergence(d_matrix)
        if convergence_info['converges']:
            print(f"✓ Spectral radius = {convergence_info['spectral_radius']:.6f} < 1 (converges)\n")
        else:
            print(f"⚠ Spectral radius = {convergence_info['spectral_radius']:.6f} >= 1\n")
    
    # Step 8: Calculate TRM: T = D × (I - D)⁻¹
    if decimal_places is None:
        decimal_places = DECIMAL_PLACES
    
    print("Calculating Total Relation Matrix: T = D × (I - D)⁻¹...")
    trm_array = calculate_trm(d_matrix, i_minus_d_inverse, decimal_places)
    print(f"TRM calculated (values rounded to {decimal_places} decimal places).\n")
    
    # Step 9: Validate TRM
    print("Validating Total Relation Matrix...")
    validation = validate_trm(trm_array)
    if validation['valid']:
        print("✓ TRM validation passed.\n")
    else:
        print("⚠ TRM validation issues:")
        for issue in validation['issues']:
            print(f"  - {issue}")
        print()
    
    # Step 10: Print summary
    print_trm_summary(d_matrix, trm_array, determinant, convergence_info, 
                       validation, barriers, barrier_to_name)
    
    # Step 11: Create DataFrames
    trm_df = create_trm_dataframe(trm_array, barriers)
    trm_export = create_trm_for_export(trm_df, barrier_to_name)
    
    # Step 12: Create intermediate matrices DataFrames
    intermediate_matrices = create_intermediate_matrices_df(
        identity_matrix, i_minus_d, i_minus_d_inverse, barriers
    )
    
    # Step 13: Create calculation info DataFrame
    calculation_info = create_calculation_info_df(n, determinant, convergence_info, validation)
    
    # Step 14: Save outputs
    output_file = None
    if save:
        print("\nSaving outputs...")
        output_file = save_trm(trm_export, trm_df, calculation_info, intermediate_matrices)
    
    # Prepare results dictionary
    results = {
        'trm': trm_df,
        'trm_export': trm_export,
        'trm_array': trm_array,
        'normalized_drm': normalized_drm,
        'identity_matrix': identity_matrix,
        'i_minus_d': i_minus_d,
        'i_minus_d_inverse': i_minus_d_inverse,
        'determinant': determinant,
        'convergence_info': convergence_info,
        'validation': validation,
        'barriers': barriers,
        'barrier_to_name': barrier_to_name,
        'output_file': output_file
    }
    
    print("\n" + "=" * 70)
    print("MODULE 3 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")
    
    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 3 in standalone mode...")
    results = create_total_relation_matrix()
    
    # Display the Total Relation Matrix
    print("\n" + "=" * 70)
    print("TOTAL RELATION MATRIX (T = D × (I - D)⁻¹)")
    print("=" * 70)
    print(f"Formula: T = D × (I - D)⁻¹")
    print(f"Determinant of (I-D): {results['determinant']:.6f}")
    print()
    print(results['trm'].to_string())
    
    print("\n" + "=" * 70)
    print("TOTAL RELATION MATRIX (For Export)")
    print("=" * 70)
    print(results['trm_export'].to_string())

