# SPDX-License-Identifier: PROPRIETARY
# File: m13_ism_micmac.py
# Purpose: Module 13 - MICMAC Analysis for DEMATEL-ISM Integration

"""
MICMAC Analysis

MICMAC (Matriced Impacts Croises Multiplication Appliquee a un Classement)
classifies factors into four clusters based on their driving and dependence power.

Mathematical Formulas:
----------------------

1. BOUNDARY CALCULATIONS:
   DP_max = max(DP(i)) for all i
   DP_min = min(DP(i)) for all i
   DEP_max = max(DEP(i)) for all i
   DEP_min = min(DEP(i)) for all i

2. MIDPOINT CALCULATIONS (Standard Academic Approach):
   Reference: Warfield (1974), Mandal & Deshmukh (1994), Ravi & Shankar (2005)
   
   DP_mid = n / 2   (where n = number of factors)
   DEP_mid = n / 2
   
   Rationale: Maximum possible power = n (influence all factors)
              Minimum possible power = 1 (self-influence only)
              Theoretical range midpoint ≈ n / 2

3. FOUR QUADRANTS CLASSIFICATION:
   | Quadrant | Name       | Condition                       | Characteristic                  |
   |----------|------------|----------------------------------|--------------------------------|
   | I        | Autonomous | DP < DP_mid AND DEP < DEP_mid   | Weak driver, weak dependent    |
   | II       | Dependent  | DP < DP_mid AND DEP >= DEP_mid  | Weak driver, strong dependent  |
   | III      | Linkage    | DP >= DP_mid AND DEP >= DEP_mid | Strong driver, strong dependent|
   | IV       | Independent| DP >= DP_mid AND DEP < DEP_mid  | Strong driver, weak dependent  |

4. MICMAC SCATTER PLOT:
   - X-axis: Dependence Power (DEP)
   - Y-axis: Driving Power (DP)
   - Vertical line at x = DEP_mid
   - Horizontal line at y = DP_mid
"""

import pandas as pd
import numpy as np
import textwrap
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. MICMAC visualization will be limited.")

# Import from previous modules
from m1_data_processing import (
    BARRIER_NAMES,
    get_script_directory
)
from m7_mmde_triplet import OUTPUT_DIR_ISM
from m10_ism_frm import create_reachability_matrix

# =============================================================================
# CONFIGURABLE PARAMETERS - Academic Publication Standards
# =============================================================================

# Figure settings
FIGURE_WIDTH = 12                # Figure width in inches
FIGURE_HEIGHT = 9                # Figure height in inches
FIGURE_DPI = 300                 # Output resolution (publication standard)
FIGURE_BG_COLOR = '#FFFFFF'      # Pure white background
PLOT_BG_COLOR = '#FFFFFF'        # Pure white plot area

# Marker/point settings
MARKER_SIZE = 200                # Base marker size
MARKER_EDGE_COLOR = 'white'      # Marker edge color
MARKER_EDGE_WIDTH = 2            # Marker edge line width
MARKER_ZORDER = 5                # Draw order for markers

# Point colors by cluster (Okabe-Ito colorblind-safe palette)
# Warm colors for drivers/causes, cool colors for effects
POINT_COLORS = {
    'Autonomous': '#009E73',     # Bluish Green - disconnected (Wong palette)
    'Dependent': '#0072B2',      # Blue - effects (cool, matches m6 effect color)
    'Linkage': '#D55E00',        # Vermillion - unstable (warm)
    'Independent': '#E69F00'     # Orange - root causes (warm, matches m6 cause color)
}

# Quadrant background settings
SHOW_QUADRANT_BACKGROUNDS = False  # Set to False for clean academic look
QUADRANT_BG_ALPHA = 0.08         # Very subtle if enabled
QUADRANT_COLORS = {
    'Autonomous': '#C5EBE0',     # Teal (bottom-left) - matches m12 Level 2
    'Dependent': '#D4EBF7',      # Sky Blue (bottom-right) - matches m12 Level 1
    'Linkage': '#FADDD0',        # Pink (top-right) - matches m12 Level 3
    'Independent': '#F9E5B5'     # Yellow/Orange (top-left) - matches m12 Level 4
}

# Midpoint line settings
MIDLINE_COLOR = '#666666'        # Gray for threshold lines
MIDLINE_STYLE = '--'             # Dashed line style
MIDLINE_WIDTH = 1.5              # Line width

# Grid settings (matching m6 style)
SHOW_GRID = True                 # Light grid aids data reading
GRID_COLOR = '#E0E0E0'           # Very light gray (matches m6)
GRID_ALPHA = 0.7                 # Grid transparency (matches m6)
GRID_LINESTYLE = '-'             # Solid thin lines (matches m6)
GRID_LINEWIDTH = 0.5             # Thin grid lines (matches m6)

# Font settings - Academic publication standards (Arial)
FONT_FAMILY = 'Arial'            # Nature/Elsevier standard
AXIS_LABEL_SIZE = 15             # Axis label font size
AXIS_LABEL_WEIGHT = 'bold'       # Axis label weight
TICK_LABEL_SIZE = 10             # Tick label font size
AXIS_LABEL_PAD = 8              # Spacing between axis label and tick labels

# Quadrant label settings (matching m6 cause-effect style)
QUADRANT_LABEL_SIZE = 15         # Quadrant label font size (matches m6)
QUADRANT_LABEL_STYLE = 'italic'  # Italic for distinction (like m6)
QUADRANT_LABEL_COLOR = '#888888' # Subtle gray color (like m6)
QUADRANT_LABEL_ALPHA = 0.85      # Slight transparency (like m6)

# Barrier label settings (external labels with full names)
BARRIER_LABEL_SIZE = 14          # Barrier label font size
BARRIER_LABEL_WEIGHT = 'normal'  # Barrier label weight
BARRIER_LABEL_COLOR = '#000000'  # Barrier label text color
BARRIER_LABEL_BG_COLOR = 'white' # Label box background color
BARRIER_LABEL_BG_ALPHA = 0.92    # Label background transparency
BARRIER_LABEL_EDGE_COLOR = '#CCCCCC'  # Label box border color
BARRIER_LABEL_EDGE_WIDTH = 0.5   # Label box border width
BARRIER_LABEL_MAX_WIDTH = 18     # Max characters per line for text wrapping
BARRIER_LABEL_PAD = 0.3          # Padding inside label box

# Leader line settings (connecting bubbles to labels)
LEADER_LINE_COLOR = '#888888'    # Leader line color
LEADER_LINE_WIDTH = 0.8          # Leader line width
LEADER_LINE_SHRINK_A = 0         # Shrink at label end
LEADER_LINE_SHRINK_B = 6         # Shrink at bubble end (to not overlap)

# Manual label position overrides for specific barriers
# Format: 'barrier_code': (offset_x, offset_y, ha, va)
# Use this to fine-tune label positions that overlap with quadrant labels or each other
MANUAL_LABEL_OVERRIDES = {
    'B2': (20, -15, 'left', 'top'),        # Move down-right to avoid "IV. Independent" label
    'B3': (5, 25, 'left', 'bottom'),       # Up-right at ~78° angle (same as B5)
    'B4': (-25, 30, 'right', 'bottom'),    # Move up-left (B4 and B6 at same point)
    'B5': (13, 35, 'left', 'bottom'),      # Up-right at ~70° angle (same as B3)
    'B6': (25, 15, 'left', 'bottom'),      # Move up-RIGHT to stay in quadrant IV, separate from B4
    'B7': (20, 20, 'left', 'bottom'),      # Move right-up to stay within quadrant II
    'B8': (-60, 15, 'right', 'bottom'),    # Move left-up to stay within quadrant II
}

# Legend settings
SHOW_LEGEND = True               # Show legend for quadrant classification
LEGEND_LOCATION = 'center right'   # Legend position (away from data points)
LEGEND_FONT_SIZE = 12             # Legend font size

# Midpoint annotation settings
SHOW_MIDPOINT_ANNOTATION = False # Disabled - threshold values belong in figure caption
MIDPOINT_FONT_SIZE = 9           # Midpoint annotation font size (if enabled)
MIDPOINT_BG_COLOR = 'wheat'      # Midpoint box background (if enabled)
MIDPOINT_BG_ALPHA = 0.8          # Midpoint box transparency (if enabled)

# Title settings
SHOW_TITLE = False               # Disabled for academic use (use figure caption instead)
TITLE_SIZE = 14                  # Title font size (if enabled)
TITLE_WEIGHT = 'bold'            # Title weight (if enabled)

# =============================================================================
# MICMAC CALCULATIONS
# =============================================================================

def calculate_boundaries(driving_power, dependence_power):
    """
    Calculate min and max boundaries for driving and dependence power.

    Formulas:
    ---------
    DP_max = max(DP(i)) for all i
    DP_min = min(DP(i)) for all i
    DEP_max = max(DEP(i)) for all i
    DEP_min = min(DEP(i)) for all i

    Parameters:
    -----------
    driving_power : numpy.ndarray
        Driving power values for each factor
    dependence_power : numpy.ndarray
        Dependence power values for each factor

    Returns:
    --------
    dict
        Dictionary with boundary values
    """
    return {
        'dp_max': np.max(driving_power),
        'dp_min': np.min(driving_power),
        'dep_max': np.max(dependence_power),
        'dep_min': np.min(dependence_power)
    }


def calculate_midpoints(boundaries, n_factors=None):
    """
    Calculate midpoints for driving and dependence power.

    Standard Academic Practice (Warfield 1974, Mandal & Deshmukh 1994):
    -------------------------------------------------------------------
    The threshold for MICMAC quadrant classification uses n/2 where n is
    the number of factors. This is because:
    - Maximum possible driving/dependence power = n (influence all factors)
    - Minimum possible power = 1 (self-influence in reachability matrix)
    - Theoretical range midpoint = (n + 1) / 2 ≈ n / 2

    Formula:
    --------
    DP_mid = DEP_mid = n / 2

    Parameters:
    -----------
    boundaries : dict
        Dictionary with boundary values from calculate_boundaries()
    n_factors : int, optional
        Number of factors. If provided, uses standard n/2 threshold.
        If None, falls back to (max + min) / 2.

    Returns:
    --------
    dict
        Dictionary with midpoint values
    """
    if n_factors is not None:
        # Standard academic approach: n/2 threshold
        # This provides consistent, theoretically-grounded classification
        dp_mid = n_factors / 2
        dep_mid = n_factors / 2
    else:
        # Fallback: use actual data range (less preferred)
        dp_mid = (boundaries['dp_max'] + boundaries['dp_min']) / 2
        dep_mid = (boundaries['dep_max'] + boundaries['dep_min']) / 2

    return {
        'dp_mid': dp_mid,
        'dep_mid': dep_mid
    }


def classify_factors(driving_power, dependence_power, midpoints):
    """
    Classify factors into MICMAC quadrants.

    Classification Rules:
    ---------------------
    Quadrant I   (Autonomous):  DP < DP_mid AND DEP < DEP_mid
    Quadrant II  (Dependent):   DP < DP_mid AND DEP >= DEP_mid
    Quadrant III (Linkage):     DP >= DP_mid AND DEP >= DEP_mid
    Quadrant IV  (Independent): DP >= DP_mid AND DEP < DEP_mid

    Parameters:
    -----------
    driving_power : numpy.ndarray
        Driving power values
    dependence_power : numpy.ndarray
        Dependence power values
    midpoints : dict
        Dictionary with midpoint values

    Returns:
    --------
    dict
        Dictionary mapping factor index to cluster name
    """
    dp_mid = midpoints['dp_mid']
    dep_mid = midpoints['dep_mid']

    clusters = {}
    n = len(driving_power)

    for i in range(n):
        dp = driving_power[i]
        dep = dependence_power[i]

        if dp < dp_mid and dep < dep_mid:
            clusters[i] = 'Autonomous'
        elif dp < dp_mid and dep >= dep_mid:
            clusters[i] = 'Dependent'
        elif dp >= dp_mid and dep >= dep_mid:
            clusters[i] = 'Linkage'
        else:  # dp >= dp_mid and dep < dep_mid
            clusters[i] = 'Independent'

    return clusters


def get_cluster_characteristics():
    """
    Get characteristics description for each cluster.

    Returns:
    --------
    dict
        Dictionary with cluster descriptions
    """
    return {
        'Autonomous': {
            'quadrant': 'I',
            'position': 'Bottom-Left',
            'driving': 'Weak',
            'dependence': 'Weak',
            'description': 'These factors are relatively disconnected from the system. '
                           'They have weak driving power and weak dependence on other factors.',
            'implication': 'Low priority factors that neither significantly drive nor are '
                           'significantly driven by other factors.'
        },
        'Dependent': {
            'quadrant': 'II',
            'position': 'Bottom-Right',
            'driving': 'Weak',
            'dependence': 'Strong',
            'description': 'These factors are highly dependent on other factors but have '
                           'little driving power themselves.',
            'implication': 'These are EFFECT factors. Addressing root causes will impact these.'
        },
        'Linkage': {
            'quadrant': 'III',
            'position': 'Top-Right',
            'driving': 'Strong',
            'dependence': 'Strong',
            'description': 'These factors are unstable. They have strong driving power but are '
                           'also strongly dependent on other factors.',
            'implication': 'Handle with care. Any action will have cascading effects through the system.'
        },
        'Independent': {
            'quadrant': 'IV',
            'position': 'Top-Left',
            'driving': 'Strong',
            'dependence': 'Weak',
            'description': 'These factors are strong drivers with low dependence. '
                           'They influence other factors but are not influenced much.',
            'implication': 'These are ROOT CAUSES. Priority targets for intervention.'
        }
    }


# =============================================================================
# MICMAC VISUALIZATION
# =============================================================================

def format_barrier_label(barrier_code, barrier_names, max_width=None):
    """
    Format barrier label with code and name, with text wrapping.

    Parameters:
    -----------
    barrier_code : str
        Barrier code (e.g., 'b1')
    barrier_names : dict
        Dictionary mapping barrier codes to full names
    max_width : int, optional
        Maximum characters per line. If None, uses BARRIER_LABEL_MAX_WIDTH.

    Returns:
    --------
    str
        Formatted label text with line breaks if needed
    """
    if max_width is None:
        max_width = BARRIER_LABEL_MAX_WIDTH

    # Get full name from barrier_names dict (try both cases)
    full_name = barrier_names.get(barrier_code.lower(),
                barrier_names.get(barrier_code.upper(), barrier_code.upper()))

    # Wrap text if too long
    if len(full_name) > max_width:
        lines = textwrap.wrap(full_name, width=max_width)
        return '\n'.join(lines)

    return full_name


def calculate_label_positions(dependence_power, driving_power, midpoints, clusters, n, barriers=None):
    """
    Calculate smart label positions for MICMAC scatter plot.

    Positions labels based on quadrant location to avoid overlaps:
    - Autonomous (bottom-left): labels to left/below
    - Dependent (bottom-right): labels to right/below
    - Linkage (top-right): labels to right/above
    - Independent (top-left): labels to left/above

    Handles overlapping points by spreading their labels in different directions.
    Manual overrides can be specified in MANUAL_LABEL_OVERRIDES for fine-tuning.

    Parameters:
    -----------
    dependence_power : numpy.ndarray
        Dependence power values (x-axis)
    driving_power : numpy.ndarray
        Driving power values (y-axis)
    midpoints : dict
        Dictionary with midpoint values
    clusters : dict
        Factor classifications
    n : int
        Number of factors
    barriers : list, optional
        List of barrier codes for manual override lookup

    Returns:
    --------
    list of tuples
        List of (offset_x, offset_y, ha, va) for each label
    """
    dp_mid = midpoints['dp_mid']
    dep_mid = midpoints['dep_mid']

    # Find overlapping points (same coordinates)
    coord_groups = {}
    for i in range(n):
        coord = (dependence_power[i], driving_power[i])
        if coord not in coord_groups:
            coord_groups[coord] = []
        coord_groups[coord].append(i)

    positions = [None] * n

    for coord, indices in coord_groups.items():
        dep, dp = coord
        cluster = clusters[indices[0]]  # All points at same coord have same cluster
        num_overlapping = len(indices)

        # Base offset in points
        base_offset = 15

        # Define spread angles for overlapping points at same coordinate
        # Spread labels in different directions based on how many overlap
        if num_overlapping == 1:
            # Single point - standard positioning based on quadrant
            if cluster == 'Autonomous':
                spread_offsets = [(-base_offset - 5, -base_offset - 10, 'right', 'top')]
            elif cluster == 'Dependent':
                spread_offsets = [(base_offset + 5, -base_offset - 10, 'left', 'top')]
            elif cluster == 'Linkage':
                spread_offsets = [(base_offset + 5, base_offset + 5, 'left', 'bottom')]
            else:  # Independent
                spread_offsets = [(-base_offset - 5, base_offset + 5, 'right', 'bottom')]
        elif num_overlapping == 2:
            # Two overlapping points - spread vertically
            if cluster == 'Autonomous':
                spread_offsets = [
                    (-base_offset - 5, base_offset + 15, 'right', 'bottom'),   # Above
                    (-base_offset - 5, -base_offset - 25, 'right', 'top'),     # Below
                ]
            elif cluster == 'Dependent':
                spread_offsets = [
                    (base_offset + 5, base_offset + 15, 'left', 'bottom'),     # Above
                    (base_offset + 5, -base_offset - 25, 'left', 'top'),       # Below
                ]
            elif cluster == 'Linkage':
                spread_offsets = [
                    (base_offset + 5, base_offset + 30, 'left', 'bottom'),     # Upper
                    (base_offset + 5, base_offset, 'left', 'bottom'),          # Lower
                ]
            else:  # Independent
                spread_offsets = [
                    (-base_offset - 5, base_offset + 30, 'right', 'bottom'),   # Upper
                    (-base_offset - 5, base_offset, 'right', 'bottom'),        # Lower
                ]
        else:
            # Three or more - spread in multiple directions
            angles = np.linspace(0, 2 * np.pi, num_overlapping, endpoint=False)
            spread_offsets = []
            for angle in angles:
                ox = int(np.cos(angle) * (base_offset + 15))
                oy = int(np.sin(angle) * (base_offset + 15))
                ha = 'left' if ox >= 0 else 'right'
                va = 'bottom' if oy >= 0 else 'top'
                spread_offsets.append((ox, oy, ha, va))

        # Assign positions to each point at this coordinate
        for idx, i in enumerate(indices):
            # Check for manual override first
            if barriers is not None:
                barrier_code = barriers[i].upper()
                if barrier_code in MANUAL_LABEL_OVERRIDES:
                    positions[i] = MANUAL_LABEL_OVERRIDES[barrier_code]
                    continue

            offset_x, offset_y, ha, va = spread_offsets[idx]

            # Fine-tune based on position within quadrant
            # If close to midline, adjust to avoid overlap with line
            if abs(dep - dep_mid) < 0.8:  # Close to vertical midline
                offset_x = int(offset_x * 1.3)
            if abs(dp - dp_mid) < 0.8:  # Close to horizontal midline
                offset_y = int(offset_y * 1.3)

            positions[i] = (offset_x, offset_y, ha, va)

    return positions


def create_micmac_plot(driving_power, dependence_power, clusters, midpoints,
                        boundaries, barriers, barrier_names=None, figsize=None):
    """
    Create MICMAC scatter plot with four quadrants.

    Plot Elements:
    --------------
    - X-axis: Dependence Power (DEP)
    - Y-axis: Driving Power (DP)
    - Vertical line at x = DEP_mid
    - Horizontal line at y = DP_mid
    - Points for each factor labeled with barrier code

    Parameters:
    -----------
    driving_power : numpy.ndarray
        Driving power values
    dependence_power : numpy.ndarray
        Dependence power values
    clusters : dict
        Factor classifications
    midpoints : dict
        Midpoint values
    boundaries : dict
        Boundary values
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    figsize : tuple, optional
        Figure size. If None, uses (FIGURE_WIDTH, FIGURE_HEIGHT)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Cannot create MICMAC plot.")
        return None

    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    if figsize is None:
        figsize = (FIGURE_WIDTH, FIGURE_HEIGHT)

    # Create figure with academic-standard white background
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(FIGURE_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    dp_mid = midpoints['dp_mid']
    dep_mid = midpoints['dep_mid']

    # Calculate axis limits with padding
    # Use 0.5 as minimum to ensure tick at 1 is visible, end at n+0.2 for slight padding
    n_barriers = len(barriers)
    x_min = 0.5
    x_max = n_barriers #+ 0.2  # 8.2 for 8 barriers
    y_min = 0.5
    y_max = n_barriers #+ 0.2  # 8.2 for 8 barriers

    # Draw quadrant backgrounds (only if enabled)
    if SHOW_QUADRANT_BACKGROUNDS:
        # Quadrant I: Autonomous (bottom-left)
        rect1 = plt.Rectangle((x_min, y_min), dep_mid - x_min, dp_mid - y_min,
                               facecolor=QUADRANT_COLORS['Autonomous'], alpha=QUADRANT_BG_ALPHA)
        ax.add_patch(rect1)

        # Quadrant II: Dependent (bottom-right)
        rect2 = plt.Rectangle((dep_mid, y_min), x_max - dep_mid, dp_mid - y_min,
                               facecolor=QUADRANT_COLORS['Dependent'], alpha=QUADRANT_BG_ALPHA)
        ax.add_patch(rect2)

        # Quadrant III: Linkage (top-right)
        rect3 = plt.Rectangle((dep_mid, dp_mid), x_max - dep_mid, y_max - dp_mid,
                               facecolor=QUADRANT_COLORS['Linkage'], alpha=QUADRANT_BG_ALPHA)
        ax.add_patch(rect3)

        # Quadrant IV: Independent (top-left)
        rect4 = plt.Rectangle((x_min, dp_mid), dep_mid - x_min, y_max - dp_mid,
                               facecolor=QUADRANT_COLORS['Independent'], alpha=QUADRANT_BG_ALPHA)
        ax.add_patch(rect4)

    # Add grid (behind everything) if enabled
    if SHOW_GRID:
        ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA,
                linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, zorder=0)

    # Draw midpoint lines (threshold lines)
    ax.axhline(y=dp_mid, color=MIDLINE_COLOR, linestyle=MIDLINE_STYLE,
               linewidth=MIDLINE_WIDTH, zorder=1)
    ax.axvline(x=dep_mid, color=MIDLINE_COLOR, linestyle=MIDLINE_STYLE,
               linewidth=MIDLINE_WIDTH, zorder=1)

    # Calculate label positions for all barriers
    n = len(barriers)
    label_positions = calculate_label_positions(
        dependence_power, driving_power, midpoints, clusters, n, barriers
    )

    # Plot factors (points only, labels added separately)
    for i in range(n):
        dep = dependence_power[i]
        dp = driving_power[i]
        cluster = clusters[i]

        # Plot point
        ax.scatter(dep, dp, c=POINT_COLORS[cluster], s=MARKER_SIZE,
                   edgecolors=MARKER_EDGE_COLOR, linewidths=MARKER_EDGE_WIDTH,
                   zorder=MARKER_ZORDER)

    # Add labels with leader lines (after all points to ensure proper layering)
    for i in range(n):
        dep = dependence_power[i]
        dp = driving_power[i]
        barrier_code = barriers[i]

        # Get formatted label text with full barrier name
        label_text = format_barrier_label(barrier_code, barrier_names)

        # Get calculated position for this label
        offset_x, offset_y, ha, va = label_positions[i]

        # Add annotation with leader line connecting to bubble
        ax.annotate(
            label_text,
            (dep, dp),  # Point to the bubble center
            xytext=(offset_x, offset_y),
            textcoords='offset points',
            fontsize=BARRIER_LABEL_SIZE,
            fontweight=BARRIER_LABEL_WEIGHT,
            fontfamily=FONT_FAMILY,
            color=BARRIER_LABEL_COLOR,
            ha=ha,
            va=va,
            zorder=10,
            bbox=dict(
                boxstyle=f'round,pad={BARRIER_LABEL_PAD}',
                facecolor=BARRIER_LABEL_BG_COLOR,
                edgecolor=BARRIER_LABEL_EDGE_COLOR,
                alpha=BARRIER_LABEL_BG_ALPHA,
                linewidth=BARRIER_LABEL_EDGE_WIDTH
            ),
            arrowprops=dict(
                arrowstyle='-',  # Simple line (no arrowhead)
                color=LEADER_LINE_COLOR,
                linewidth=LEADER_LINE_WIDTH,
                shrinkA=LEADER_LINE_SHRINK_A,
                shrinkB=LEADER_LINE_SHRINK_B,
                connectionstyle='arc3,rad=0'  # Straight line
            )
        )

    # Add quadrant labels (m6 cause-effect style: positioned in corners)
    # Calculate margins - keep labels close to axes/corners to avoid overlap with data labels
    x_range = x_max - x_min
    y_range = y_max - y_min
    label_margin_x = x_range * 0.01  # Reduced to stay closer to axis
    label_margin_y = y_range * 0.01  # Reduced to stay closer to axis

    # I. Autonomous (bottom-left corner)
    ax.text(x_min + label_margin_x, y_min + label_margin_y,
            'I. Autonomous',
            ha='left', va='bottom', fontsize=QUADRANT_LABEL_SIZE,
            fontstyle=QUADRANT_LABEL_STYLE, fontfamily=FONT_FAMILY,
            color=QUADRANT_LABEL_COLOR, alpha=QUADRANT_LABEL_ALPHA)

    # II. Dependent (bottom-right corner)
    ax.text(x_max - label_margin_x, y_min + label_margin_y,
            'II. Dependent',
            ha='right', va='bottom', fontsize=QUADRANT_LABEL_SIZE,
            fontstyle=QUADRANT_LABEL_STYLE, fontfamily=FONT_FAMILY,
            color=QUADRANT_LABEL_COLOR, alpha=QUADRANT_LABEL_ALPHA)

    # III. Linkage (top-right corner)
    ax.text(x_max - label_margin_x, y_max - label_margin_y,
            'III. Linkage',
            ha='right', va='top', fontsize=QUADRANT_LABEL_SIZE,
            fontstyle=QUADRANT_LABEL_STYLE, fontfamily=FONT_FAMILY,
            color=QUADRANT_LABEL_COLOR, alpha=QUADRANT_LABEL_ALPHA)

    # IV. Independent (top-left corner)
    ax.text(x_min + label_margin_x, y_max - label_margin_y,
            'IV. Independent',
            ha='left', va='top', fontsize=QUADRANT_LABEL_SIZE,
            fontstyle=QUADRANT_LABEL_STYLE, fontfamily=FONT_FAMILY,
            color=QUADRANT_LABEL_COLOR, alpha=QUADRANT_LABEL_ALPHA)

    # Set axis labels (no title - use figure caption instead for academic publications)
    ax.set_xlabel('Dependence Power', fontsize=AXIS_LABEL_SIZE,
                  fontweight=AXIS_LABEL_WEIGHT, fontfamily=FONT_FAMILY,
                  labelpad=AXIS_LABEL_PAD)
    ax.set_ylabel('Driving Power', fontsize=AXIS_LABEL_SIZE,
                  fontweight=AXIS_LABEL_WEIGHT, fontfamily=FONT_FAMILY,
                  labelpad=AXIS_LABEL_PAD)

    # Set title only if enabled (disabled by default for academic use)
    if SHOW_TITLE:
        ax.set_title('MICMAC Analysis\n(Driving Power vs Dependence Power)',
                      fontsize=TITLE_SIZE, fontweight=TITLE_WEIGHT,
                      fontfamily=FONT_FAMILY, pad=20)

    # Set axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set tick parameters with Arial font
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily(FONT_FAMILY)

    # Set integer ticks - explicitly show all values from 1 to n (number of barriers)
    # This ensures all 8 barriers are represented on both axes
    n_barriers = len(barriers)
    ax.set_xticks(range(1, n_barriers + 1))
    ax.set_yticks(range(1, n_barriers + 1))

    # Add legend if enabled
    if SHOW_LEGEND:
        # Only include quadrants that have barriers (exclude Linkage if empty)
        legend_elements = [
            mpatches.Patch(facecolor=POINT_COLORS['Independent'], edgecolor='none',
                           label='Root Causes'),
            mpatches.Patch(facecolor=POINT_COLORS['Dependent'], edgecolor='none',
                           label='Effects'),
            mpatches.Patch(facecolor=POINT_COLORS['Autonomous'], edgecolor='none',
                           label='Disconnected'),
        ]
        # Position legend in Linkage quadrant (top-right), below the quadrant label
        legend = ax.legend(handles=legend_elements,
                           loc='upper right',
                           bbox_to_anchor=(0.98, 0.88),  # Below "III. Linkage" label
                           fontsize=LEGEND_FONT_SIZE, frameon=True,
                           fancybox=False, edgecolor='#CCCCCC', framealpha=0.95)
        legend.get_frame().set_linewidth(0.5)
        # Set legend font family
        for text in legend.get_texts():
            text.set_fontfamily(FONT_FAMILY)

    # Add midpoint annotation if enabled
    if SHOW_MIDPOINT_ANNOTATION:
        midpoint_text = f'Thresholds:\nDP_mid = {dp_mid:.2f}\nDEP_mid = {dep_mid:.2f}'
        ax.text(0.98, 0.98, midpoint_text, transform=ax.transAxes,
                fontsize=MIDPOINT_FONT_SIZE, fontfamily=FONT_FAMILY,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=MIDPOINT_BG_COLOR,
                          alpha=MIDPOINT_BG_ALPHA))

    plt.tight_layout()

    # Add abbreviation note below x-axis as figure footnote
    fig.text(0.05, 0.02, 'KM = Knowledge Management',
             fontsize=9, fontfamily=FONT_FAMILY,
             fontstyle='italic', color='#666666',
             verticalalignment='bottom', horizontalalignment='left')

    return fig


# =============================================================================
# DATAFRAME CREATION
# =============================================================================

def create_micmac_dataframe(driving_power, dependence_power, clusters,
                             barriers, barrier_names=None):
    """
    Create MICMAC classification DataFrame.

    Parameters:
    -----------
    driving_power : numpy.ndarray
        Driving power values
    dependence_power : numpy.ndarray
        Dependence power values
    clusters : dict
        Factor classifications
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        MICMAC classification DataFrame
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    characteristics = get_cluster_characteristics()

    data = []
    for i in range(len(barriers)):
        barrier_code = barriers[i]
        cluster = clusters[i]
        char = characteristics[cluster]

        data.append({
            'Barrier_Code': barrier_code.upper(),
            'Barrier_Name': barrier_names.get(barrier_code, barrier_code.upper()),
            'Driving_Power': int(driving_power[i]),
            'Dependence_Power': int(dependence_power[i]),
            'Cluster': cluster,
            'Quadrant': char['quadrant'],
            'Position': char['position'],
            'Driver_Strength': char['driving'],
            'Dependence_Strength': char['dependence']
        })

    df = pd.DataFrame(data)

    # Sort by cluster then by driving power
    cluster_order = {'Independent': 0, 'Linkage': 1, 'Dependent': 2, 'Autonomous': 3}
    df['_sort_order'] = df['Cluster'].map(cluster_order)
    df = df.sort_values(['_sort_order', 'Driving_Power'], ascending=[True, False])
    df = df.drop('_sort_order', axis=1).reset_index(drop=True)

    return df


def create_cluster_summary_dataframe(clusters, barriers, barrier_names=None):
    """
    Create cluster summary DataFrame.

    Parameters:
    -----------
    clusters : dict
        Factor classifications
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        Cluster summary DataFrame
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    characteristics = get_cluster_characteristics()

    # Group factors by cluster
    cluster_factors = {'Autonomous': [], 'Dependent': [], 'Linkage': [], 'Independent': []}
    for i, cluster in clusters.items():
        cluster_factors[cluster].append(barriers[i].upper())

    data = []
    for cluster_name in ['Independent', 'Linkage', 'Dependent', 'Autonomous']:
        char = characteristics[cluster_name]
        factors = cluster_factors[cluster_name]

        data.append({
            'Cluster': cluster_name,
            'Quadrant': char['quadrant'],
            'Position': char['position'],
            'Driver': char['driving'],
            'Dependence': char['dependence'],
            'Count': len(factors),
            'Factors': ', '.join(factors) if factors else 'None',
            'Implication': char['implication']
        })

    return pd.DataFrame(data)


def create_calculation_info_dataframe(boundaries, midpoints, n_factors=None):
    """
    Create calculation information DataFrame.

    Parameters:
    -----------
    boundaries : dict
        Boundary values
    midpoints : dict
        Midpoint values
    n_factors : int, optional
        Number of factors (for showing n/2 formula)

    Returns:
    --------
    pandas.DataFrame
        Calculation info DataFrame
    """
    # Build midpoint formula description
    if n_factors is not None:
        dp_formula = f'DP_mid (formula: n/2 = {n_factors}/2)'
        dep_formula = f'DEP_mid (formula: n/2 = {n_factors}/2)'
    else:
        dp_formula = 'DP_mid (formula: (DP_max + DP_min) / 2)'
        dep_formula = 'DEP_mid (formula: (DEP_max + DEP_min) / 2)'
    
    data = {
        'Parameter': [
            'n (number of factors)',
            'DP_max',
            'DP_min',
            'DEP_max',
            'DEP_min',
            '',
            dp_formula,
            dep_formula,
            '',
            'Reference: Warfield (1974), Mandal & Deshmukh (1994)'
        ],
        'Value': [
            n_factors if n_factors else '',
            boundaries['dp_max'],
            boundaries['dp_min'],
            boundaries['dep_max'],
            boundaries['dep_min'],
            '',
            midpoints['dp_mid'],
            midpoints['dep_mid'],
            '',
            'Standard n/2 threshold for MICMAC quadrant classification'
        ]
    }

    return pd.DataFrame(data)


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_micmac_results(micmac_df, summary_df, calc_info_df, fig, output_dir=None):
    """
    Save MICMAC results to files.

    Output files:
    - ism_micmac.xlsx: Classification and summary
    - ism_micmac.png: Scatter plot visualization

    Parameters:
    -----------
    micmac_df : pandas.DataFrame
        MICMAC classification DataFrame
    summary_df : pandas.DataFrame
        Cluster summary DataFrame
    calc_info_df : pandas.DataFrame
        Calculation info DataFrame
    fig : matplotlib.figure.Figure
        MICMAC scatter plot
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

    # Save Excel file
    excel_file = output_path / "ism_micmac.xlsx"
    try:
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            micmac_df.to_excel(writer, sheet_name='MICMAC_Classification', index=False)
            summary_df.to_excel(writer, sheet_name='Cluster_Summary', index=False)
            calc_info_df.to_excel(writer, sheet_name='Calculation_Info', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {excel_file}\n"
            "Please close the Excel file if it's open and try again."
        )
    saved_files.append(excel_file)
    print(f"MICMAC results saved to: {excel_file}")

    # Save PNG file
    if fig is not None:
        png_file = output_path / "ism_micmac.png"
        try:
            fig.savefig(png_file, dpi=FIGURE_DPI, bbox_inches='tight',
                        facecolor=FIGURE_BG_COLOR, edgecolor='none')
            saved_files.append(png_file)
            print(f"MICMAC plot saved to: {png_file}")
        except Exception as e:
            print(f"Error saving MICMAC plot: {e}")
        plt.close(fig)

    return saved_files


def print_micmac_summary(driving_power, dependence_power, clusters, midpoints,
                          boundaries, barriers, barrier_names=None):
    """
    Print MICMAC analysis summary.

    Parameters:
    -----------
    driving_power : numpy.ndarray
        Driving power values
    dependence_power : numpy.ndarray
        Dependence power values
    clusters : dict
        Factor classifications
    midpoints : dict
        Midpoint values
    boundaries : dict
        Boundary values
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    print("-" * 70)
    print("MICMAC ANALYSIS SUMMARY")
    print("-" * 70)
    print(f"Number of Barriers: {len(barriers)}")
    print()

    print("BOUNDARY CALCULATIONS:")
    print(f"  DP_max = {boundaries['dp_max']}")
    print(f"  DP_min = {boundaries['dp_min']}")
    print(f"  DEP_max = {boundaries['dep_max']}")
    print(f"  DEP_min = {boundaries['dep_min']}")
    print()

    print("MIDPOINT CALCULATIONS (Standard Academic Approach: n/2):")
    n = len(barriers)
    print(f"  DP_mid = n/2 = {n}/2 = {midpoints['dp_mid']:.2f}")
    print(f"  DEP_mid = n/2 = {n}/2 = {midpoints['dep_mid']:.2f}")
    print()

    print("CLASSIFICATION RULES:")
    print("  I.   Autonomous:  DP < DP_mid AND DEP < DEP_mid")
    print("  II.  Dependent:   DP < DP_mid AND DEP >= DEP_mid")
    print("  III. Linkage:     DP >= DP_mid AND DEP >= DEP_mid")
    print("  IV.  Independent: DP >= DP_mid AND DEP < DEP_mid")
    print()

    # Group factors by cluster
    cluster_factors = {'Autonomous': [], 'Dependent': [], 'Linkage': [], 'Independent': []}
    for i, cluster in clusters.items():
        cluster_factors[cluster].append(barriers[i].upper())

    print("CLASSIFICATION RESULTS:")
    for cluster_name in ['Independent', 'Linkage', 'Dependent', 'Autonomous']:
        factors = cluster_factors[cluster_name]
        print(f"\n  {cluster_name.upper()} ({len(factors)} factors):")
        if factors:
            for code in factors:
                idx = [i for i, b in enumerate(barriers) if b.upper() == code][0]
                name = barrier_names.get(barriers[idx], code)
                print(f"    - {name}")
                print(f"      DP={int(driving_power[idx])}, DEP={int(dependence_power[idx])}")
        else:
            print("    (None)")
    print()

    print("INTERPRETATION:")
    print("  INDEPENDENT (Root Causes): Priority targets for intervention")
    print("  LINKAGE (Unstable): Handle carefully - cascading effects")
    print("  DEPENDENT (Effects): Will improve when root causes addressed")
    print("  AUTONOMOUS (Disconnected): Low priority")
    print()

    print("-" * 70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def perform_micmac_analysis(driving_power=None, dependence_power=None, barriers=None,
                             frm_results=None, barrier_names=None, save=True):
    """
    Main function to perform MICMAC analysis.

    Parameters:
    -----------
    driving_power : numpy.ndarray, optional
        Driving power values. If None, runs Module 10 first.
    dependence_power : numpy.ndarray, optional
        Dependence power values.
    barriers : list, optional
        List of barrier codes.
    frm_results : dict, optional
        Results from Module 10.
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names.
    save : bool, optional
        Whether to save outputs.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'clusters': Factor classifications
        - 'boundaries': Boundary values
        - 'midpoints': Midpoint values
        - 'micmac_df': Classification DataFrame
        - 'summary_df': Cluster summary DataFrame
        - 'fig': MICMAC scatter plot figure
        - 'output_files': Paths to saved files
    """
    print("\n" + "=" * 70)
    print("MODULE 13: MICMAC ANALYSIS")
    print("=" * 70 + "\n")

    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    # Step 1: Get required data
    if driving_power is None or dependence_power is None or barriers is None:
        if frm_results is not None:
            driving_power = frm_results['driving_power']
            dependence_power = frm_results['dependence_power']
            barriers = frm_results['barriers']
            print("Using power data from provided Module 10 results.\n")
        else:
            print("Running Module 10 to get driving and dependence power...")
            frm_results = create_reachability_matrix(save=False)
            driving_power = frm_results['driving_power']
            dependence_power = frm_results['dependence_power']
            barriers = frm_results['barriers']
            print()

    n = len(barriers)
    print(f"Number of barriers: {n}")
    print(f"Barriers: {[b.upper() for b in barriers]}\n")

    # Step 2: Calculate boundaries
    print("Calculating boundaries...")
    boundaries = calculate_boundaries(driving_power, dependence_power)
    print(f"  DP: min={boundaries['dp_min']}, max={boundaries['dp_max']}")
    print(f"  DEP: min={boundaries['dep_min']}, max={boundaries['dep_max']}\n")

    # Step 3: Calculate midpoints using standard n/2 approach
    print("Calculating midpoints (standard academic approach: n/2)...")
    midpoints = calculate_midpoints(boundaries, n_factors=n)
    print(f"  DP_mid = n/2 = {n}/2 = {midpoints['dp_mid']:.2f}")
    print(f"  DEP_mid = n/2 = {n}/2 = {midpoints['dep_mid']:.2f}\n")

    # Step 4: Classify factors
    print("Classifying factors into quadrants...")
    clusters = classify_factors(driving_power, dependence_power, midpoints)
    cluster_counts = {}
    for cluster in clusters.values():
        cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1
    for cluster, count in cluster_counts.items():
        print(f"  {cluster}: {count} factors")
    print()

    # Step 5: Print summary
    print_micmac_summary(driving_power, dependence_power, clusters, midpoints,
                          boundaries, barriers, barrier_names)

    # Step 6: Create DataFrames
    print("\nCreating DataFrames...")
    micmac_df = create_micmac_dataframe(driving_power, dependence_power, clusters,
                                         barriers, barrier_names)
    summary_df = create_cluster_summary_dataframe(clusters, barriers, barrier_names)
    calc_info_df = create_calculation_info_dataframe(boundaries, midpoints, n_factors=n)
    print("  DataFrames created.\n")

    # Step 7: Create visualization
    fig = None
    if MATPLOTLIB_AVAILABLE:
        print("Creating MICMAC scatter plot...")
        fig = create_micmac_plot(driving_power, dependence_power, clusters, midpoints,
                                  boundaries, barriers, barrier_names)
        print("  Plot created.\n")
    else:
        print("matplotlib not available. Skipping visualization.\n")

    # Step 8: Save outputs
    output_files = None
    if save:
        print("Saving outputs...")
        output_files = save_micmac_results(micmac_df, summary_df, calc_info_df, fig)

    # Prepare results
    results = {
        'clusters': clusters,
        'boundaries': boundaries,
        'midpoints': midpoints,
        'driving_power': driving_power,
        'dependence_power': dependence_power,
        'micmac_df': micmac_df,
        'summary_df': summary_df,
        'calc_info_df': calc_info_df,
        'fig': fig,
        'barriers': barriers,
        'barrier_names': barrier_names,
        'n': n,
        'output_files': output_files
    }

    print("\n" + "=" * 70)
    print("MODULE 13 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 13 in standalone mode...")
    results = perform_micmac_analysis()

    # Display MICMAC classification
    print("\n" + "=" * 70)
    print("MICMAC CLASSIFICATION")
    print("=" * 70)
    print(results['micmac_df'].to_string(index=False))

    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY")
    print("=" * 70)
    print(results['summary_df'][['Cluster', 'Count', 'Factors', 'Implication']].to_string(index=False))
