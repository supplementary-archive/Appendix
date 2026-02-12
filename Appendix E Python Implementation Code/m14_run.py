# SPDX-License-Identifier: PROPRIETARY
# File: m14_run.py
# Purpose: Module 14 - Complete DEMATEL-ISM Integration Pipeline Runner

"""
DEMATEL-ISM Integration Complete Pipeline

This module runs the complete analysis pipeline from M1 to M13:

DEMATEL Analysis (M1-M6):
    M1: Data Processing → Direct Relation Matrix (DRM)
    M2: Normalization → Normalized DRM
    M3: Total Relation Matrix (TRM) calculation
    M4: D+R and D-R analysis
    M5: Prominence-Relation analysis
    M6: Cause-Effect classification

DEMATEL-ISM Integration (M7-M13):
    M7: MMDE Triplet Creation
    M8: MMDE Cumulative Sets (Td, Tr)
    M9: MMDE Threshold Calculation (Entropy, MDE)
    M10: ISM Reachability Matrix (Binary conversion, Transitivity)
    M11: ISM Level Partitioning
    M12: ISM Digraph Visualization
    M13: MICMAC Analysis

Process Flow:
    TRM → MMDE Threshold → Binary Matrix → Transitivity Check →
    Reachability Matrix → Level Partitioning → Digraph → MICMAC Analysis
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Set to True to run DEMATEL modules (M1-M6) first
# Set to False to start from existing TRM file
RUN_DEMATEL_MODULES = True

# Set to True to save all intermediate outputs
SAVE_INTERMEDIATE = True

# Set to True for verbose output
VERBOSE = True

# =============================================================================
# IMPORT MODULES
# =============================================================================

def import_modules():
    """Import all required modules with error handling."""
    modules = {}

    try:
        # DEMATEL Modules (M1-M6)
        from m1_data_processing import process_data, BARRIER_NAMES, get_script_directory
        modules['m1'] = process_data
        modules['barrier_names'] = BARRIER_NAMES
        modules['get_script_directory'] = get_script_directory

        from m2_dematel_normalized_drm import create_normalized_drm
        modules['m2'] = create_normalized_drm

        from m3_dematel_trm import create_total_relation_matrix
        modules['m3'] = create_total_relation_matrix

        from m4_dematel_trm_dar import calculate_d_and_r
        modules['m4'] = calculate_d_and_r

        from m5_dematel_pro_relation import calculate_prominence_and_relation
        modules['m5'] = calculate_prominence_and_relation

        from m6_dematel_cause_effect import create_cause_effect_analysis
        modules['m6'] = create_cause_effect_analysis

        # ISM Integration Modules (M7-M13)
        from m7_mmde_triplet import create_mmde_triplets
        modules['m7'] = create_mmde_triplets

        from m8_mmde_sets import build_mmde_sets
        modules['m8'] = build_mmde_sets

        from m9_mmde_final import calculate_mmde_threshold
        modules['m9'] = calculate_mmde_threshold

        from m10_ism_frm import create_reachability_matrix
        modules['m10'] = create_reachability_matrix

        from m11_ism_lp import perform_level_partitioning
        modules['m11'] = perform_level_partitioning

        from m12_ism_diagraph import create_ism_digraph
        modules['m12'] = create_ism_digraph

        from m13_ism_micmac import perform_micmac_analysis
        modules['m13'] = perform_micmac_analysis

        print("All modules imported successfully.\n")
        return modules

    except ImportError as e:
        print(f"Error importing module: {e}")
        print("Please ensure all module files (m1-m13) are in the same directory.")
        sys.exit(1)


# =============================================================================
# PIPELINE EXECUTION
# =============================================================================

def run_dematel_pipeline(modules, save=True):
    """
    Run DEMATEL analysis pipeline (M1-M6).

    Parameters:
    -----------
    modules : dict
        Dictionary of imported module functions
    save : bool
        Whether to save intermediate outputs

    Returns:
    --------
    dict
        Results from all DEMATEL modules
    """
    results = {}

    print("\n" + "=" * 80)
    print("PHASE 1: DEMATEL ANALYSIS (M1-M6)")
    print("=" * 80)

    # M1: Data Processing
    print("\n[1/6] Running M1: Data Processing...")
    start = time.time()
    results['m1'] = modules['m1'](save=save)
    print(f"      Completed in {time.time() - start:.2f}s")

    # M2: Normalization
    # create_normalized_drm expects 'drm' parameter
    print("\n[2/6] Running M2: Normalization...")
    start = time.time()
    results['m2'] = modules['m2'](drm=results['m1']['dematel_drm'], save=save)
    print(f"      Completed in {time.time() - start:.2f}s")

    # M3: Total Relation Matrix
    print("\n[3/6] Running M3: Total Relation Matrix...")
    start = time.time()
    results['m3'] = modules['m3'](normalized_drm=results['m2']['normalized_drm'], save=save)
    print(f"      Completed in {time.time() - start:.2f}s")

    # M4: D and R Calculation
    # calculate_d_and_r expects 'trm' parameter
    print("\n[4/6] Running M4: D and R Calculation...")
    start = time.time()
    results['m4'] = modules['m4'](trm=results['m3']['trm'], save=save)
    print(f"      Completed in {time.time() - start:.2f}s")

    # M5: Prominence-Relation Analysis
    # calculate_prominence_and_relation expects 'd_r_data' dict with d_values, r_values, barrier_to_name
    print("\n[5/6] Running M5: Prominence-Relation Analysis...")
    start = time.time()
    results['m5'] = modules['m5'](d_r_data=results['m4'], save=save)
    print(f"      Completed in {time.time() - start:.2f}s")

    # M6: Cause-Effect Diagram
    # create_cause_effect_analysis expects 'pro_rel_data' DataFrame or None (loads from file)
    print("\n[6/6] Running M6: Cause-Effect Diagram...")
    start = time.time()
    results['m6'] = modules['m6'](pro_rel_data=results['m5']['main_df'], save=save, show=False)
    print(f"      Completed in {time.time() - start:.2f}s")

    print("\n" + "-" * 80)
    print("DEMATEL ANALYSIS COMPLETE")
    print("-" * 80)

    return results


def run_ism_pipeline(modules, trm=None, dematel_results=None, save=True):
    """
    Run DEMATEL-ISM Integration pipeline (M7-M13).

    Parameters:
    -----------
    modules : dict
        Dictionary of imported module functions
    trm : pandas.DataFrame, optional
        Total Relation Matrix. If None, loads from file.
    dematel_results : dict, optional
        Results from DEMATEL pipeline
    save : bool
        Whether to save intermediate outputs

    Returns:
    --------
    dict
        Results from all ISM modules
    """
    results = {}

    print("\n" + "=" * 80)
    print("PHASE 2: DEMATEL-ISM INTEGRATION (M7-M13)")
    print("=" * 80)

    # Get TRM from DEMATEL results if available
    if trm is None and dematel_results is not None:
        trm = dematel_results['m3']['trm']

    # M7: MMDE Triplet Creation
    print("\n[1/7] Running M7: MMDE Triplet Creation...")
    start = time.time()
    results['m7'] = modules['m7'](trm=trm, save=save)
    print(f"      Created {len(results['m7']['sorted_triplets'])} triplets")
    print(f"      Completed in {time.time() - start:.2f}s")

    # M8: MMDE Cumulative Sets
    print("\n[2/7] Running M8: MMDE Cumulative Sets...")
    start = time.time()
    results['m8'] = modules['m8'](
        sorted_triplets=results['m7']['sorted_triplets'],
        barriers=results['m7']['barriers'],
        save=save
    )
    print(f"      Built Td and Tr sets for {len(results['m8']['cumulative_data']['positions'])} positions")
    print(f"      Completed in {time.time() - start:.2f}s")

    # M9: MMDE Threshold Calculation
    print("\n[3/7] Running M9: MMDE Threshold Calculation...")
    start = time.time()
    results['m9'] = modules['m9'](m8_results=results['m8'], save=save)
    threshold = results['m9']['threshold']
    print(f"      MMDE Threshold (lambda) = {threshold:.4f}")
    print(f"      Completed in {time.time() - start:.2f}s")

    # M10: ISM Reachability Matrix
    print("\n[4/7] Running M10: ISM Reachability Matrix...")
    start = time.time()
    results['m10'] = modules['m10'](
        trm=trm,
        threshold=results['m9']['threshold'],
        save=save
    )
    print(f"      Transitivity entries added: {len(results['m10']['transitivity_result']['transitivity_entries'])}")
    print(f"      Completed in {time.time() - start:.2f}s")

    # M11: ISM Level Partitioning
    print("\n[5/7] Running M11: ISM Level Partitioning...")
    start = time.time()
    results['m11'] = modules['m11'](frm_results=results['m10'], save=save)
    print(f"      Identified {results['m11']['max_level']} hierarchical levels")
    print(f"      Completed in {time.time() - start:.2f}s")

    # M12: ISM Digraph
    print("\n[6/7] Running M12: ISM Digraph Visualization...")
    start = time.time()
    results['m12'] = modules['m12'](
        irm=results['m10']['irm'],
        lp_results=results['m11'],
        frm_results=results['m10'],
        save=save
    )
    print(f"      Created digraph with {len(results['m12']['edges'])} edges")
    print(f"      Completed in {time.time() - start:.2f}s")

    # M13: MICMAC Analysis
    print("\n[7/7] Running M13: MICMAC Analysis...")
    start = time.time()
    results['m13'] = modules['m13'](frm_results=results['m10'], save=save)
    cluster_counts = {}
    for cluster in results['m13']['clusters'].values():
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    print(f"      Classification: {cluster_counts}")
    print(f"      Completed in {time.time() - start:.2f}s")

    print("\n" + "-" * 80)
    print("DEMATEL-ISM INTEGRATION COMPLETE")
    print("-" * 80)

    return results


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def print_final_summary(dematel_results, ism_results, total_time):
    """
    Print comprehensive final summary of all analyses.

    Parameters:
    -----------
    dematel_results : dict
        Results from DEMATEL pipeline
    ism_results : dict
        Results from ISM pipeline
    total_time : float
        Total execution time in seconds
    """
    print("\n")
    print("=" * 80)
    print("                    FINAL SUMMARY REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print()

    # Basic Info
    barriers = ism_results['m7']['barriers']
    n = len(barriers)
    print(f"Number of Barriers Analyzed: {n}")
    print(f"Barriers: {', '.join([b.upper() for b in barriers])}")
    print()

    # MMDE Threshold
    print("-" * 80)
    print("MMDE THRESHOLD CALCULATION")
    print("-" * 80)
    threshold = ism_results['m9']['threshold']
    max_positions = ism_results['m9']['max_positions']
    print(f"  Max MDE_D Position: {max_positions['pos_d']}")
    print(f"  Max MDE_D Value: {max_positions['max_mde_d']:.6f}")
    print(f"  Max MDE_R Position: {max_positions['pos_r']}")
    print(f"  Max MDE_R Value: {max_positions['max_mde_r']:.6f}")
    print(f"  FINAL THRESHOLD (lambda): {threshold:.4f}")
    print()

    # Reachability Matrix
    print("-" * 80)
    print("ISM REACHABILITY MATRIX")
    print("-" * 80)
    irm = ism_results['m10']['irm']
    frm = ism_results['m10']['frm']
    transitivity_count = len(ism_results['m10']['transitivity_result']['transitivity_entries'])
    print(f"  Initial 1s (IRM): {irm.sum()}")
    print(f"  Final 1s (FRM): {frm.sum()}")
    print(f"  Transitivity Entries Added: {transitivity_count}")
    print()

    # Driving and Dependence Power
    print("-" * 80)
    print("DRIVING AND DEPENDENCE POWER")
    print("-" * 80)
    dp = ism_results['m10']['driving_power']
    dep = ism_results['m10']['dependence_power']
    barrier_names = ism_results['m13']['barrier_names']

    print(f"  {'Barrier':<40} {'DP':>5} {'DEP':>5}")
    print(f"  {'-'*40} {'-'*5} {'-'*5}")
    for i, b in enumerate(barriers):
        name = barrier_names.get(b, b.upper())
        print(f"  {name:<40} {int(dp[i]):>5} {int(dep[i]):>5}")
    print()

    # Level Partitioning
    print("-" * 80)
    print("ISM HIERARCHICAL LEVELS")
    print("-" * 80)
    levels = ism_results['m11']['levels']
    max_level = ism_results['m11']['max_level']
    print(f"  Total Levels: {max_level}")
    print()
    print("  Level 1 = Effects/Outcomes (Top of hierarchy)")
    print(f"  Level {max_level} = Root Causes (Bottom of hierarchy)")
    print()

    for level_num in sorted(levels.keys()):
        factors = levels[level_num]
        codes = [barriers[i].upper() for i in factors]
        print(f"  Level {level_num}: {', '.join(codes)}")
    print()

    # MICMAC Classification
    print("-" * 80)
    print("MICMAC CLASSIFICATION")
    print("-" * 80)
    clusters = ism_results['m13']['clusters']
    midpoints = ism_results['m13']['midpoints']

    print(f"  Midpoints: DP_mid = {midpoints['dp_mid']:.2f}, DEP_mid = {midpoints['dep_mid']:.2f}")
    print()

    # Group by cluster
    cluster_groups = {'Independent': [], 'Linkage': [], 'Dependent': [], 'Autonomous': []}
    for i, cluster in clusters.items():
        cluster_groups[cluster].append(barriers[i].upper())

    cluster_descriptions = {
        'Independent': 'ROOT CAUSES (Strong Driver, Weak Dependent)',
        'Linkage': 'UNSTABLE (Strong Driver, Strong Dependent)',
        'Dependent': 'EFFECTS (Weak Driver, Strong Dependent)',
        'Autonomous': 'DISCONNECTED (Weak Driver, Weak Dependent)'
    }

    for cluster_name in ['Independent', 'Linkage', 'Dependent', 'Autonomous']:
        factors = cluster_groups[cluster_name]
        desc = cluster_descriptions[cluster_name]
        print(f"  {cluster_name.upper()} - {desc}")
        if factors:
            for code in factors:
                idx = [i for i, b in enumerate(barriers) if b.upper() == code][0]
                name = barrier_names.get(barriers[idx], code)
                print(f"    - {name}")
        else:
            print(f"    (None)")
        print()

    # Output Files
    print("-" * 80)
    print("OUTPUT FILES GENERATED")
    print("-" * 80)
    print("  DEMATEL Outputs (output/dematel/):")
    print("    - dematel_drm.xlsx")
    print("    - dematel_normalized_drm.xlsx")
    print("    - dematel_trm.xlsx")
    print("    - dematel_trm_dar.xlsx")
    print("    - dematel_pro_relation.xlsx")
    print("    - dematel_cause_effect.xlsx")
    print()
    print("  ISM Outputs (output/ism/):")
    print("    - mmde_triplet.xlsx")
    print("    - mmde_sets.xlsx")
    print("    - mmde_mde.xlsx")
    print("    - mmde_final_results.xlsx")
    print("    - ism_irm.xlsx")
    print("    - ism_frm.xlsx")
    print("    - ism_lp_1.xlsx ... ism_lp_final.xlsx")
    print("    - ism_diagraph.png")
    print("    - ism_diagraph_detailed.png")
    print("    - ism_edge_list.xlsx")
    print("    - ism_micmac.xlsx")
    print("    - ism_micmac.png")
    print()

    print("=" * 80)
    print("                    ANALYSIS COMPLETE")
    print("=" * 80)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_complete_pipeline(run_dematel=True, save=True, verbose=True):
    """
    Run the complete DEMATEL-ISM integration pipeline.

    Parameters:
    -----------
    run_dematel : bool
        If True, runs DEMATEL modules M1-M6 first.
        If False, loads TRM from existing file.
    save : bool
        Whether to save all outputs to files.
    verbose : bool
        Whether to print detailed progress.

    Returns:
    --------
    dict
        Dictionary containing all results:
        - 'dematel': Results from M1-M6 (if run_dematel=True)
        - 'ism': Results from M7-M13
        - 'total_time': Total execution time
    """
    total_start = time.time()

    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + "DEMATEL-ISM INTEGRATION ANALYSIS".center(78) + "*")
    print("*" + "Complete Pipeline Execution".center(78) + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print()
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run DEMATEL Modules: {run_dematel}")
    print(f"Save Outputs: {save}")
    print()

    # Import all modules
    print("Importing modules...")
    modules = import_modules()

    # Run DEMATEL pipeline (M1-M6)
    dematel_results = None
    if run_dematel:
        dematel_results = run_dematel_pipeline(modules, save=save)

    # Run ISM pipeline (M7-M13)
    ism_results = run_ism_pipeline(
        modules,
        dematel_results=dematel_results,
        save=save
    )

    total_time = time.time() - total_start

    # Print final summary
    print_final_summary(dematel_results, ism_results, total_time)

    return {
        'dematel': dematel_results,
        'ism': ism_results,
        'total_time': total_time
    }


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("DEMATEL-ISM INTEGRATION - COMPLETE PIPELINE")
    print("=" * 80)
    print("\nThis script runs the complete analysis from M1 to M13.")
    print("All output files will be saved to output/dematel/ and output/ism/")
    print()

    # Run the complete pipeline
    results = run_complete_pipeline(
        run_dematel=RUN_DEMATEL_MODULES,
        save=SAVE_INTERMEDIATE,
        verbose=VERBOSE
    )

    print("\nPipeline execution completed successfully!")
    print(f"Total time: {results['total_time']:.2f} seconds")
