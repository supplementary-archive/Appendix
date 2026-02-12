# SPDX-License-Identifier: PROPRIETARY
# File: m12_ism_diagraph.py
# Purpose: Module 12 - ISM Digraph (Directed Graph) Visualization for DEMATEL-ISM

"""
ISM Digraph Construction

This module creates a hierarchical directed graph (digraph) visualization
showing the relationships between factors based on their ISM levels.

Mathematical Procedures:
------------------------

1. IDENTIFY DIRECT RELATIONSHIPS:
   Direct edge from i to j exists IF:
   - k_ij = 1 (in initial binary matrix K, BEFORE transitivity)
   - AND i ≠ j (not diagonal)

2. CREATE EDGE LIST:
   edges = []
   FOR i from 1 to n:
       FOR j from 1 to n:
           IF i != j AND K[i][j] == 1:
               edges.append((i, j))

3. HIERARCHICAL LAYOUT:
   - Level 1 factors at the TOP of diagram (effects/outcomes)
   - Higher numbered levels at the BOTTOM (root causes)
   - Factors within same level spaced horizontally

4. ARROW DIRECTION:
   - Arrows point UPWARD from root causes to effects
   - From higher level numbers to lower level numbers
   - Example: Level 4 → Level 3 → Level 2 → Level 1 (top)
"""

import pandas as pd
import numpy as np
import textwrap
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Digraph visualization will be limited.")

# Import from previous modules
from m1_data_processing import (
    BARRIER_NAMES,
    get_script_directory
)
from m7_mmde_triplet import OUTPUT_DIR_ISM
from m10_ism_frm import create_reachability_matrix
from m11_ism_lp import perform_level_partitioning

# =============================================================================
# CONFIGURABLE PARAMETERS - Academic Publication Standards (Refined)
# =============================================================================

# Figure settings
FIGURE_WIDTH = 14                # Figure width in inches
FIGURE_HEIGHT = 12               # Figure height in inches
FIGURE_DPI = 300                 # Output resolution (publication standard)
FIGURE_BG_COLOR = '#FFFFFF'      # Pure white background

# Node settings - refined for elegance
NODE_WIDTH = 3.8                 # Node width in data units (slightly wider for breathing room)
NODE_HEIGHT = 1.8                # Node height in data units
NODE_CORNER_RADIUS = 0.12        # Corner rounding size (subtler corners)
NODE_EDGE_WIDTH = 1.2            # Node border line width (thinner, more refined)
NODE_EDGE_COLOR = '#4A4A4A'      # Node border color (softer gray)
NODE_TEXT_WRAP_WIDTH = 18        # Maximum characters per line for text wrapping

# Layout settings - improved spacing
HORIZONTAL_SPACING = 4.5         # Horizontal spacing between nodes at same level (more breathing room)
VERTICAL_SPACING = 3.2           # Vertical spacing between levels (more separation)
LEVEL_BAND_HEIGHT = 3.0          # Height of level background bands
LEVEL_BAND_ALPHA = 0.20          # Transparency of level bands (subtler, more elegant)

# Arrow settings - refined for elegance and distinctiveness
ARROW_STYLE = '-|>'              # Arrow head style (clean triangular)
ARROW_MUTATION_SCALE = 18        # Arrow head size (smaller, more refined)
ARROW_HEAD_LENGTH = 0.4          # Arrow head length in data units (proportionate)
ARROW_HEAD_WIDTH = 0.25          # Arrow head width in data units (narrower, elegant)
UPWARD_ARROW_COLOR = '#2171B5'   # Influence arrows - refined blue (academic standard)
UPWARD_ARROW_WIDTH = 1.4         # Line width for upward arrows (thinner, cleaner)
UPWARD_ARROW_ALPHA = 0.85        # Transparency for upward arrows
DOWNWARD_ARROW_COLOR = '#D94801' # Feedback arrows - refined orange
DOWNWARD_ARROW_WIDTH = 1.3       # Line width for downward arrows
DOWNWARD_ARROW_ALPHA = 0.80      # Transparency for downward arrows
SAME_LEVEL_ARROW_COLOR = '#7B3294'  # Same level arrows - refined purple
SAME_LEVEL_ARROW_WIDTH = 1.4     # Line width for same level arrows
SAME_LEVEL_ARROW_ALPHA = 0.85    # Transparency for same level arrows
BIDIRECTIONAL_ARROW_COLOR = '#7B3294'  # Reciprocal arrows - purple (mutual influence)
BIDIRECTIONAL_ARROW_WIDTH = 1.5  # Line width for bidirectional arrows
BIDIRECTIONAL_ARROW_ALPHA = 0.88 # Transparency for bidirectional arrows
BIDIRECTIONAL_ARROW_STYLE = '<|-|>'  # Double-headed arrow style

# Arrow routing settings - improved for distinctiveness
ARROW_CURVE_RAD_BASE = 0.12      # Base curvature radius for inter-level arrows
ARROW_CURVE_RAD_INCREMENT = 0.08 # Increment for separating parallel arrows
SAME_LEVEL_ARC_RAD = -0.35       # Arc radius for same-level arrows (negative = arc upward)
ARROW_EDGE_BUFFER = 0.10         # Buffer from node edge (as fraction of node dimension)
ARROW_SPREAD_FACTOR = 0.6        # Factor for spreading arrows along node edges

# Custom arrow routing overrides for specific edges
# Format: (from_barrier_code, to_barrier_code): curvature_radius
# Positive = curve right, Negative = curve left, 0 = straight
# Use this to manually route specific arrows through desired paths
CUSTOM_ARROW_ROUTES = {
    ('b2', 'b7'): 0.20,  # Route B2→B7 through gap between B4 and B6 (slight right curve)
    ('b2', 'b8'): 0.35,  # Route B2→B8 with stronger right curve to avoid passing below B6
    ('b2', 'b5'): -0.20, # Route B2→B5 with stronger left curve
}

# Swap origin points for pairs of edges from the same source
# Format: List of tuples, each containing two edges to swap: ((from1, to1), (from2, to2))
# Both edges must share the same source (from) barrier
EDGE_ORIGIN_SWAPS = [
    #(('b2', 'b5'), ('b2', 'b7')),  # Swap origin points of B2→B5 and B2→B7
]

# Font settings - Nature journal standards (refined for elegance)
FONT_FAMILY = 'Arial'            # Nature requirement: Arial or Helvetica
NODE_FONT_SIZE = 14              # Node text size (balanced for multi-line barrier names)
NODE_FONT_WEIGHT = 'semibold'    # Font weight for barrier names (semibold for elegance)
NODE_FONT_COLOR = '#2D2D2D'      # Font color (refined dark gray)
LEVEL_LABEL_FONT_SIZE = 15       # Level labels (increased for visibility)
LEVEL_LABEL_FONT_WEIGHT = 'medium'  # Font weight for level labels (medium for subtlety)
LEVEL_LABEL_FONT_COLOR = '#606060'  # Font color for level labels (softer gray)
LEGEND_FONT_SIZE = 12            # Regular text (increased for visibility)
NOTE_FONT_SIZE = 8               # Regular text (slightly larger for readability)

# Level label settings
LEVEL_LABEL_POSITION = 'left'   # Position of level labels ('left', 'right', 'both')
LEVEL_LABEL_OFFSET = 0.6         # Offset from plot edge (slightly more for breathing room)

# Legend settings
SHOW_LEGEND = True               # Whether to show legend
LEGEND_LOCATION = 'lower right'   # Legend position

# Interpretation note settings
SHOW_INTERPRETATION_NOTE = False  # Whether to show interpretation note
NOTE_POSITION = (0.02, 0.02)     # Position (x, y) in axes coordinates

# Level colors - Okabe-Ito colorblind-safe palette (refined, slightly desaturated)
# Reference: Wong, B. (2011) Nature Methods 8:441
# Level 1 (Top) = Effects/Outcomes (cool colors)
# Higher levels (Bottom) = Root Causes (warm colors)
LEVEL_COLORS = {
    1: '#D4EBF7',  # Sky Blue - Effects (softer, more elegant)
    2: '#C5EBE0',  # Teal - Transition (refined)
    3: '#FADDD0',  # Vermillion - Transition (gentle)
    4: '#F9E5B5',  # Orange - Root Causes (softer warmth)
    5: '#EDCEE3',  # Red-Purple (muted)
    6: '#F9F3C0',  # Yellow (softer)
    7: '#CEDDED',  # Blue (gentle)
    8: '#E0E0E0',  # Grey (light)
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def int_to_roman(num):
    """
    Convert an integer to a Roman numeral string.

    Parameters:
    -----------
    num : int
        Integer to convert (1-3999)

    Returns:
    --------
    str
        Roman numeral representation
    """
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    syms = ['M', 'CM', 'D', 'CD', 'C', 'XC', 'L', 'XL', 'X', 'IX', 'V', 'IV', 'I']
    roman_num = ''
    for i, v in enumerate(val):
        while num >= v:
            roman_num += syms[i]
            num -= v
    return roman_num


# =============================================================================
# EDGE EXTRACTION
# =============================================================================

def extract_edges(binary_matrix):
    """
    Extract direct relationship edges from binary matrix.

    Formula:
    --------
    Direct edge from i to j exists IF:
    - k_ij = 1 (in binary matrix)
    - AND i ≠ j (not diagonal)

    Note: Uses the INITIAL binary matrix (IRM), NOT the FRM with transitivity.
    This ensures we only show direct relationships, not transitive ones.

    Parameters:
    -----------
    binary_matrix : numpy.ndarray
        Initial Reachability Matrix (before transitivity)

    Returns:
    --------
    list
        List of tuples (from_index, to_index) representing edges
        Indices are 0-based
    """
    n = len(binary_matrix)
    edges = []

    for i in range(n):
        for j in range(n):
            # Direct edge exists if k_ij = 1 and i != j
            if i != j and binary_matrix[i, j] == 1:
                edges.append((i, j))

    return edges


def filter_edges_by_levels(edges, factor_levels):
    """
    Categorize edges based on level relationships.

    Parameters:
    -----------
    edges : list
        List of (from_index, to_index) tuples
    factor_levels : dict
        Dict mapping factor index to level

    Returns:
    --------
    dict
        Dictionary with categorized edges:
        - 'inter_level': Edges between different levels
        - 'intra_level': Edges within same level
        - 'upward': Edges pointing to higher level (cause to effect)
        - 'downward': Edges pointing to lower level
    """
    inter_level = []
    intra_level = []
    upward = []  # From higher level number to lower (cause to effect)
    downward = []

    for from_idx, to_idx in edges:
        from_level = factor_levels.get(from_idx, 0)
        to_level = factor_levels.get(to_idx, 0)

        if from_level == to_level:
            intra_level.append((from_idx, to_idx))
        else:
            inter_level.append((from_idx, to_idx))
            if from_level > to_level:
                upward.append((from_idx, to_idx))  # Cause → Effect
            else:
                downward.append((from_idx, to_idx))

    return {
        'inter_level': inter_level,
        'intra_level': intra_level,
        'upward': upward,
        'downward': downward,
        'all': edges
    }


def identify_reciprocal_edges(edges):
    """
    Identify reciprocal (bidirectional) edges and separate them from unidirectional edges.

    Reciprocal edges occur when both (i, j) and (j, i) exist in the edge list.
    These should be drawn as a single double-headed arrow.

    Parameters:
    -----------
    edges : list
        List of (from_index, to_index) tuples

    Returns:
    --------
    dict
        Dictionary containing:
        - 'reciprocal': List of tuples (min_idx, max_idx) for bidirectional edges
        - 'unidirectional': List of tuples (from_idx, to_idx) for one-way edges
    """
    edge_set = set(edges)
    reciprocal = []
    unidirectional = []
    processed = set()

    for from_idx, to_idx in edges:
        # Skip if already processed as part of a reciprocal pair
        if (from_idx, to_idx) in processed or (to_idx, from_idx) in processed:
            continue

        # Check if reverse edge exists
        if (to_idx, from_idx) in edge_set:
            # Reciprocal edge - use consistent ordering (lower index first)
            reciprocal.append((min(from_idx, to_idx), max(from_idx, to_idx)))
            processed.add((from_idx, to_idx))
            processed.add((to_idx, from_idx))
        else:
            # Unidirectional edge
            unidirectional.append((from_idx, to_idx))
            processed.add((from_idx, to_idx))

    return {
        'reciprocal': reciprocal,
        'unidirectional': unidirectional
    }


# =============================================================================
# LAYOUT CALCULATION
# =============================================================================

def calculate_hierarchical_positions(levels, barriers, max_level):
    """
    Calculate node positions for hierarchical layout.

    Layout Convention:
    ------------------
    - Level 1 at TOP (y = max_level)
    - Higher level numbers at BOTTOM (y = 1 for max level)
    - Factors within level spaced horizontally

    Parameters:
    -----------
    levels : dict
        Dict mapping level number to list of factor indices
    barriers : list
        List of barrier codes
    max_level : int
        Maximum level number

    Returns:
    --------
    dict
        Dictionary mapping factor index to (x, y) position
    """
    positions = {}

    for level_num, factors in levels.items():
        # Y position: Level 1 at top, higher levels at bottom
        # Invert so Level 1 is at max_level height
        y_position = (max_level - level_num + 1) * VERTICAL_SPACING

        # X positions: center the factors at this level
        num_factors = len(factors)
        if num_factors == 1:
            x_positions = [0]
        else:
            total_width = (num_factors - 1) * HORIZONTAL_SPACING
            start_x = -total_width / 2
            x_positions = [start_x + i * HORIZONTAL_SPACING for i in range(num_factors)]

        for idx, factor in enumerate(sorted(factors)):
            positions[factor] = (x_positions[idx], y_position)

    return positions


# =============================================================================
# ARROW CONNECTION POINT CALCULATION
# =============================================================================

def calculate_arrow_endpoints(x1, y1, x2, y2, from_level, to_level,
                               source_edge_index=0, source_total_edges=1,
                               target_edge_index=0, target_total_edges=1,
                               node_width=None, node_height=None):
    """
    Calculate arrow start and end points on node edges (not centers).
    
    Arrows are distributed along the node edge so multiple arrows don't
    converge at the same point. Each arrow gets a unique position on the edge.
    
    Parameters:
    -----------
    x1, y1 : float
        Center coordinates of source node
    x2, y2 : float
        Center coordinates of target node
    from_level, to_level : int
        Level numbers of source and target nodes
    source_edge_index : int
        Index of this arrow among arrows leaving the source node
    source_total_edges : int
        Total arrows leaving the source node (in same direction)
    target_edge_index : int
        Index of this arrow among arrows arriving at the target node
    target_total_edges : int
        Total arrows arriving at the target node
    node_width, node_height : float, optional
        Node dimensions. If None, uses configured defaults.
    
    Returns:
    --------
    tuple
        (start_x, start_y, end_x, end_y) - Arrow endpoint coordinates
    """
    if node_width is None:
        node_width = NODE_WIDTH
    if node_height is None:
        node_height = NODE_HEIGHT
    
    half_width = node_width / 2
    half_height = node_height / 2

    # CRITICAL: Padding to keep arrowheads OUTSIDE nodes
    # FancyArrowPatch draws arrowhead AT the endpoint, extending beyond it
    # So endpoint must be pulled back by arrowhead length to prevent penetration
    # Arrowhead actual size ≈ ARROW_HEAD_LENGTH * ARROW_MUTATION_SCALE / 10
    arrowhead_size = ARROW_HEAD_LENGTH * ARROW_MUTATION_SCALE / 10
    target_padding = arrowhead_size + 0.08  # Extra gap for clean separation
    source_padding = 0.05  # Small gap at source (no arrowhead there)

    # Calculate spread offset for source (distribute arrows along edge)
    usable_width = half_width * 2 * ARROW_SPREAD_FACTOR
    if source_total_edges > 1:
        source_spread = (source_edge_index - (source_total_edges - 1) / 2) * (usable_width / max(source_total_edges - 1, 1))
    else:
        source_spread = 0

    # Calculate spread offset for target (distribute arrows along edge)
    if target_total_edges > 1:
        target_spread = (target_edge_index - (target_total_edges - 1) / 2) * (usable_width / max(target_total_edges - 1, 1))
    else:
        target_spread = 0

    # Determine connection points - arrows start/end OUTSIDE node boundaries
    if from_level > to_level:
        # Upward arrow: source below, target above
        # Start just outside TOP edge of source
        start_y = y1 + half_height + source_padding
        # End with padding for arrowhead OUTSIDE BOTTOM edge of target
        end_y = y2 - half_height - target_padding
        start_x = x1 + source_spread
        end_x = x2 + target_spread

    elif from_level < to_level:
        # Downward arrow (feedback): source above, target below
        start_y = y1 - half_height - source_padding
        end_y = y2 + half_height + target_padding
        start_x = x1 + source_spread
        end_x = x2 + target_spread

    else:
        # Same level: horizontal connection from side edges
        start_y = y1
        end_y = y2
        dx = x2 - x1
        if dx > 0:
            # Target is to the right
            start_x = x1 + half_width + source_padding
            end_x = x2 - half_width - target_padding
        else:
            # Target is to the left
            start_x = x1 - half_width - source_padding
            end_x = x2 + half_width + target_padding

    return start_x, start_y, end_x, end_y


def get_connection_style(from_level, to_level, x1, x2, y1=0, y2=0,
                          edge_index=0, total_edges=1, from_idx=0, to_idx=0,
                          from_barrier=None, to_barrier=None):
    """
    Determine the appropriate connection style for an arrow based on relationship type.

    Enhanced algorithm for distinct, non-overlapping arrows:
    - Uses unique curvature based on source-target pair
    - Spreads parallel arrows with varying arc radii
    - Considers both horizontal and vertical positioning
    - Supports custom overrides for specific barrier pairs

    Parameters:
    -----------
    from_level, to_level : int
        Level numbers of source and target
    x1, x2 : float
        X coordinates of source and target
    y1, y2 : float
        Y coordinates of source and target
    edge_index : int
        Index of this edge among edges to same target (for spreading)
    total_edges : int
        Total number of edges going to the same target
    from_idx, to_idx : int
        Indices of source and target nodes (for unique curve calculation)
    from_barrier, to_barrier : str, optional
        Barrier codes for custom routing overrides

    Returns:
    --------
    str
        Matplotlib connectionstyle string
    """
    # Check for custom route override
    if from_barrier and to_barrier:
        custom_key = (from_barrier.lower(), to_barrier.lower())
        if custom_key in CUSTOM_ARROW_ROUTES:
            custom_rad = CUSTOM_ARROW_ROUTES[custom_key]
            return f"arc3,rad={custom_rad}"

    if from_level == to_level:
        # Same level: use arc that goes ABOVE the nodes
        # Vary slightly based on node indices for distinctiveness
        arc_variation = (from_idx - to_idx) * 0.03
        arc_rad = SAME_LEVEL_ARC_RAD + arc_variation
        return f"arc3,rad={arc_rad}"
    else:
        # Inter-level: use distinct curves for each arrow
        dx = x2 - x1
        level_diff = abs(from_level - to_level)

        # Base curvature depends on horizontal offset
        if abs(dx) < 0.5:
            # Nearly vertical: use small curve, direction based on node indices
            base_rad = 0.06 * (1 if (from_idx + to_idx) % 2 == 0 else -1)
        else:
            # Horizontal offset: curve in natural direction
            base_rad = ARROW_CURVE_RAD_BASE if dx > 0 else -ARROW_CURVE_RAD_BASE

        # Add unique offset based on source-target pair to separate overlapping arrows
        # This ensures arrows between different pairs have distinct curvatures
        pair_signature = (from_idx * 10 + to_idx) % 7  # Generate variety
        pair_offset = (pair_signature - 3) * 0.025

        # Spread multiple arrows going to same target
        if total_edges > 1:
            spread_offset = (edge_index - (total_edges - 1) / 2) * ARROW_CURVE_RAD_INCREMENT
        else:
            spread_offset = 0

        # Combine all offsets
        final_rad = base_rad + pair_offset + spread_offset

        # Clamp to reasonable range to prevent extreme curves
        final_rad = max(-0.35, min(0.35, final_rad))

        return f"arc3,rad={final_rad}"


# =============================================================================
# DIGRAPH VISUALIZATION
# =============================================================================

def create_digraph(edges, positions, levels, factor_levels, barriers,
                    barrier_names=None, figsize=None, title=None):
    """
    Create hierarchical digraph visualization.

    Parameters:
    -----------
    edges : list
        List of (from_index, to_index) tuples
    positions : dict
        Dict mapping factor index to (x, y) position
    levels : dict
        Dict mapping level number to list of factor indices
    factor_levels : dict
        Dict mapping factor index to level
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    figsize : tuple, optional
        Figure size (width, height). If None, uses configured defaults.
    title : str, optional
        Title for the digraph (not used - title removed for academic style)

    Returns:
    --------
    matplotlib.figure.Figure
        The created figure object
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available. Cannot create digraph.")
        return None

    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    if figsize is None:
        figsize = (FIGURE_WIDTH, FIGURE_HEIGHT)

    # Create figure with white background
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_facecolor(FIGURE_BG_COLOR)
    ax.set_facecolor(FIGURE_BG_COLOR)

    max_level = max(levels.keys())

    # Calculate plot boundaries with padding for level labels
    x_min = min(pos[0] for pos in positions.values()) - NODE_WIDTH - 0.5
    x_max = max(pos[0] for pos in positions.values()) + NODE_WIDTH + 0.5

    # Draw level bands (background)
    for level_num in levels.keys():
        y = (max_level - level_num + 1) * VERTICAL_SPACING
        color = LEVEL_COLORS.get(level_num, '#F5F5F5')
        band_y_start = y - LEVEL_BAND_HEIGHT / 2
        rect = plt.Rectangle(
            (x_min, band_y_start),
            x_max - x_min,
            LEVEL_BAND_HEIGHT,
            facecolor=color,
            alpha=LEVEL_BAND_ALPHA,
            edgecolor='none',
            zorder=0
        )
        ax.add_patch(rect)

        # Add level label
        # Determine annotation text for special levels
        annotation = ""
        if level_num == max_level:
            annotation = "\n\n(Independent)"
        elif level_num == 1:
            annotation = "\n\n(Dependent)"

        if LEVEL_LABEL_POSITION in ['right', 'both']:
            ax.text(
                x_max + LEVEL_LABEL_OFFSET, y,
                f'Level {int_to_roman(level_num)}{annotation}',
                fontsize=LEVEL_LABEL_FONT_SIZE,
                fontweight=LEVEL_LABEL_FONT_WEIGHT,
                fontfamily=FONT_FAMILY,
                va='center',
                ha='left',
                color=LEVEL_LABEL_FONT_COLOR
            )
        if LEVEL_LABEL_POSITION in ['left', 'both']:
            ax.text(
                x_min - LEVEL_LABEL_OFFSET, y,
                f'Level {int_to_roman(level_num)}{annotation}',
                fontsize=LEVEL_LABEL_FONT_SIZE,
                fontweight=LEVEL_LABEL_FONT_WEIGHT,
                fontfamily=FONT_FAMILY,
                va='center',
                ha='right',
                color=LEVEL_LABEL_FONT_COLOR
            )

    # Identify reciprocal (bidirectional) vs unidirectional edges
    edge_types = identify_reciprocal_edges(edges)
    reciprocal_edges = edge_types['reciprocal']
    unidirectional_edges = edge_types['unidirectional']

    # Arrow endpoint padding - arrowhead TIP is drawn AT the endpoint
    # Small padding ensures clean visual separation without large gaps
    target_padding = 0.05  # Small gap so arrowhead tip just touches node edge
    source_padding = 0.05  # Small gap at source

    # Pre-process unidirectional edges for spreading
    edges_to_target = {}
    edges_from_source = {}
    for from_idx, to_idx in unidirectional_edges:
        if to_idx not in edges_to_target:
            edges_to_target[to_idx] = []
        edges_to_target[to_idx].append(from_idx)
        if from_idx not in edges_from_source:
            edges_from_source[from_idx] = []
        edges_from_source[from_idx].append(to_idx)

    for target_idx in edges_to_target:
        edges_to_target[target_idx].sort(key=lambda src: positions[src][0])
    for source_idx in edges_from_source:
        edges_from_source[source_idx].sort(key=lambda tgt: positions[tgt][0])

    target_edge_counter = {to_idx: 0 for to_idx in edges_to_target}
    source_edge_counter = {from_idx: 0 for from_idx in edges_from_source}

    half_width = NODE_WIDTH / 2
    half_height = NODE_HEIGHT / 2

    # =========================================================================
    # DRAW BIDIRECTIONAL (reciprocal) arrows - double-headed
    # =========================================================================
    for idx1, idx2 in reciprocal_edges:
        x1, y1 = positions[idx1]
        x2, y2 = positions[idx2]
        level1 = factor_levels[idx1]
        level2 = factor_levels[idx2]

        # Both ends need small padding (double-headed arrow)
        bidir_padding = 0.05  # Small gap so arrowhead tips just touch node edges

        if level1 == level2:
            # Same level: horizontal connection
            if x1 < x2:
                start_x = x1 + half_width + bidir_padding
                end_x = x2 - half_width - bidir_padding
            else:
                start_x = x1 - half_width - bidir_padding
                end_x = x2 + half_width + bidir_padding
            start_y, end_y = y1, y2
            conn_style = f"arc3,rad={SAME_LEVEL_ARC_RAD}"
        else:
            # Different levels: vertical connection
            if y1 < y2:
                start_y = y1 + half_height + bidir_padding
                end_y = y2 - half_height - bidir_padding
            else:
                start_y = y1 - half_height - bidir_padding
                end_y = y2 + half_height + bidir_padding
            start_x, end_x = x1, x2
            dx = x2 - x1
            rad = ARROW_CURVE_RAD_BASE if dx > 0 else -ARROW_CURVE_RAD_BASE
            conn_style = f"arc3,rad={rad}"

        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle=f'{BIDIRECTIONAL_ARROW_STYLE},head_length={ARROW_HEAD_LENGTH},head_width={ARROW_HEAD_WIDTH}',
            mutation_scale=ARROW_MUTATION_SCALE,
            connectionstyle=conn_style,
            color=BIDIRECTIONAL_ARROW_COLOR,
            linewidth=BIDIRECTIONAL_ARROW_WIDTH,
            alpha=BIDIRECTIONAL_ARROW_ALPHA,
            zorder=1
        )
        ax.add_patch(arrow)

    # =========================================================================
    # DRAW UNIDIRECTIONAL arrows - single-headed
    # =========================================================================
    for from_idx, to_idx in unidirectional_edges:
        x1, y1 = positions[from_idx]
        x2, y2 = positions[to_idx]
        from_level = factor_levels[from_idx]
        to_level = factor_levels[to_idx]

        # Determine color based on direction
        if from_level > to_level:
            color = UPWARD_ARROW_COLOR
            linewidth = UPWARD_ARROW_WIDTH
            alpha = UPWARD_ARROW_ALPHA
        elif from_level < to_level:
            color = DOWNWARD_ARROW_COLOR
            linewidth = DOWNWARD_ARROW_WIDTH
            alpha = DOWNWARD_ARROW_ALPHA
        else:
            color = SAME_LEVEL_ARROW_COLOR
            linewidth = SAME_LEVEL_ARROW_WIDTH
            alpha = SAME_LEVEL_ARROW_ALPHA

        # Get spreading indices
        source_idx_in_group = source_edge_counter.get(from_idx, 0)
        source_total = len(edges_from_source.get(from_idx, [from_idx]))
        if from_idx in source_edge_counter:
            source_edge_counter[from_idx] += 1
        target_idx_in_group = target_edge_counter.get(to_idx, 0)
        target_total = len(edges_to_target.get(to_idx, [to_idx]))
        if to_idx in target_edge_counter:
            target_edge_counter[to_idx] += 1

        # Calculate spread
        usable_width = half_width * 2 * ARROW_SPREAD_FACTOR

        # Get this edge's position in the sorted source list
        source_targets = edges_from_source.get(from_idx, [])
        if to_idx in source_targets:
            source_position = source_targets.index(to_idx)
        else:
            source_position = source_idx_in_group

        # Check for origin point swaps
        from_barrier = barriers[from_idx].lower()
        to_barrier = barriers[to_idx].lower()
        for swap_pair in EDGE_ORIGIN_SWAPS:
            edge1, edge2 = swap_pair
            current_edge = (from_barrier, to_barrier)
            if current_edge == edge1:
                swap_to_barrier = edge2[1]
                swap_to_idx = next((i for i, b in enumerate(barriers) if b.lower() == swap_to_barrier), None)
                if swap_to_idx is not None and swap_to_idx in source_targets:
                    source_position = source_targets.index(swap_to_idx)
                break
            elif current_edge == edge2:
                swap_to_barrier = edge1[1]
                swap_to_idx = next((i for i, b in enumerate(barriers) if b.lower() == swap_to_barrier), None)
                if swap_to_idx is not None and swap_to_idx in source_targets:
                    source_position = source_targets.index(swap_to_idx)
                break

        # Calculate spread using the (possibly swapped) position
        if source_total > 1:
            source_spread = (source_position - (source_total - 1) / 2) * (usable_width / max(source_total - 1, 1))
        else:
            source_spread = 0
        if target_total > 1:
            target_spread = (target_idx_in_group - (target_total - 1) / 2) * (usable_width / max(target_total - 1, 1))
        else:
            target_spread = 0

        # Calculate endpoints with proper padding
        if from_level > to_level:
            # Upward arrow
            start_x = x1 + source_spread
            start_y = y1 + half_height + source_padding
            end_x = x2 + target_spread
            end_y = y2 - half_height - target_padding
        elif from_level < to_level:
            # Downward arrow
            start_x = x1 + source_spread
            start_y = y1 - half_height - source_padding
            end_x = x2 + target_spread
            end_y = y2 + half_height + target_padding
        else:
            # Same level: horizontal
            start_y = y1
            end_y = y2
            dx = x2 - x1
            if dx > 0:
                start_x = x1 + half_width + source_padding
                end_x = x2 - half_width - target_padding
            else:
                start_x = x1 - half_width - source_padding
                end_x = x2 + half_width + target_padding

        conn_style = get_connection_style(
            from_level, to_level, x1, x2, y1, y2,
            edge_index=target_idx_in_group, total_edges=target_total,
            from_idx=from_idx, to_idx=to_idx,
            from_barrier=barriers[from_idx], to_barrier=barriers[to_idx]
        )

        arrow = FancyArrowPatch(
            (start_x, start_y), (end_x, end_y),
            arrowstyle=f'{ARROW_STYLE},head_length={ARROW_HEAD_LENGTH},head_width={ARROW_HEAD_WIDTH}',
            mutation_scale=ARROW_MUTATION_SCALE,
            connectionstyle=conn_style,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1
        )
        ax.add_patch(arrow)

    # Draw nodes (factors)
    for factor_idx, (x, y) in positions.items():
        barrier_code = barriers[factor_idx]
        level = factor_levels[factor_idx]

        # Node color based on level
        node_color = LEVEL_COLORS.get(level, '#FFFFFF')

        # Create node rectangle with refined styling
        rect = FancyBboxPatch(
            (x - NODE_WIDTH / 2, y - NODE_HEIGHT / 2),
            NODE_WIDTH,
            NODE_HEIGHT,
            boxstyle=f"round,pad=0.05,rounding_size={NODE_CORNER_RADIUS}",
            facecolor=node_color,
            edgecolor=NODE_EDGE_COLOR,
            linewidth=NODE_EDGE_WIDTH,
            zorder=2
        )
        ax.add_patch(rect)

        # Get full barrier name and wrap text for display
        full_name = barrier_names.get(barrier_code, barrier_code.upper())
        wrapped_text = textwrap.fill(full_name, width=NODE_TEXT_WRAP_WIDTH)

        # Add barrier name text with refined font
        ax.text(
            x, y,
            wrapped_text,
            fontsize=NODE_FONT_SIZE,
            fontweight=NODE_FONT_WEIGHT,
            fontfamily=FONT_FAMILY,
            ha='center',
            va='center',
            color=NODE_FONT_COLOR,
            zorder=3
        )

    # Set axis properties
    plot_x_min = x_min - 0.3
    plot_x_max = x_max + 1.8  # Extra space for level labels
    y_min = min(pos[1] for pos in positions.values()) - VERTICAL_SPACING * 0.6
    y_max = max(pos[1] for pos in positions.values()) + VERTICAL_SPACING * 0.6

    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    ax.axis('off')

    # Add legend (positioned on level band)
    if SHOW_LEGEND:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=UPWARD_ARROW_COLOR, linewidth=2.5,
                   alpha=UPWARD_ARROW_ALPHA, label='Influence (Cause → Effect)',
                   marker='>', markersize=8, markeredgewidth=0),
            Line2D([0], [0], color=BIDIRECTIONAL_ARROW_COLOR, linewidth=2.5,
                   alpha=BIDIRECTIONAL_ARROW_ALPHA, label='Reciprocal',
                   marker='d', markersize=8, markeredgewidth=0),
        ]
        # Position legend on Level 4 band (bottom level, lower part)
        # Calculate y position for max level (Level 4)
        legend_level = max_level  # Change this to position on different level
        legend_y = (max_level - legend_level + 1) * VERTICAL_SPACING - LEVEL_BAND_HEIGHT * 0.25
        legend_x = x_max - 3  # Right side of the plot

        legend = ax.legend(
            handles=legend_elements,
            loc='center',
            bbox_to_anchor=(legend_x, legend_y),
            bbox_transform=ax.transData,
            fontsize=LEGEND_FONT_SIZE,
            frameon=True,
            fancybox=False,
            edgecolor='#D0D0D0',
            framealpha=0.92,
            labelspacing=0.8
        )
        legend.get_frame().set_linewidth(0.6)

    # Add "KM =" abbreviation label in lower left
    # Position: below level 4 (max_level) band but close to it
    km_label_y = (max_level - max_level + 1) * VERTICAL_SPACING - LEVEL_BAND_HEIGHT * 0.60
    km_label_x = x_min + 0.3
    ax.text(
        km_label_x, km_label_y,
        'KM = Knowledge Management',
        fontsize=LEVEL_LABEL_FONT_SIZE - 2,
        fontfamily=FONT_FAMILY,
        fontweight='normal',
        va='top',
        ha='left',
        color=LEVEL_LABEL_FONT_COLOR,
        fontstyle='italic'
    )

    # Add interpretation note (refined styling)
    if SHOW_INTERPRETATION_NOTE:
        note_text = "Level 1 (Top) = Effects/Outcomes\nHigher Levels (Bottom) = Root Causes"
        ax.text(
            NOTE_POSITION[0], NOTE_POSITION[1],
            note_text,
            transform=ax.transAxes,
            fontsize=NOTE_FONT_SIZE,
            fontfamily=FONT_FAMILY,
            verticalalignment='bottom',
            bbox=dict(
                boxstyle='round,pad=0.4',
                facecolor='#FAFAFA',
                edgecolor='#D0D0D0',
                alpha=0.90,
                linewidth=0.6
            )
        )

    plt.tight_layout()

    return fig


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================

def save_digraph(fig, filename='ism_diagraph.png', output_dir=None, dpi=None):
    """
    Save digraph to PNG file.

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Output filename
    output_dir : str or Path, optional
        Output directory
    dpi : int, optional
        Resolution in dots per inch. If None, uses FIGURE_DPI.

    Returns:
    --------
    Path
        Path to saved file
    """
    if fig is None:
        print("No figure to save.")
        return None

    if output_dir is None:
        output_dir = OUTPUT_DIR_ISM

    if dpi is None:
        dpi = FIGURE_DPI

    script_dir = get_script_directory()
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / filename

    try:
        fig.savefig(
            file_path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor=FIGURE_BG_COLOR,
            edgecolor='none'
        )
        print(f"Digraph saved to: {file_path}")
    except Exception as e:
        print(f"Error saving digraph: {e}")
        return None

    plt.close(fig)

    return file_path


def create_edge_list_dataframe(edges, barriers, factor_levels, barrier_names=None):
    """
    Create DataFrame with edge list information.

    Parameters:
    -----------
    edges : list
        List of (from_index, to_index) tuples
    barriers : list
        List of barrier codes
    factor_levels : dict
        Dict mapping factor index to level
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names

    Returns:
    --------
    pandas.DataFrame
        Edge list DataFrame
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    data = []
    for from_idx, to_idx in edges:
        from_code = barriers[from_idx].upper()
        to_code = barriers[to_idx].upper()
        from_level = factor_levels[from_idx]
        to_level = factor_levels[to_idx]

        # Determine relationship type
        if from_level > to_level:
            rel_type = "Influence (Cause→Effect)"
        elif from_level < to_level:
            rel_type = "Feedback"
        else:
            rel_type = "Same Level"

        data.append({
            'From_Code': from_code,
            'From_Name': barrier_names.get(barriers[from_idx], from_code),
            'From_Level': from_level,
            'To_Code': to_code,
            'To_Name': barrier_names.get(barriers[to_idx], to_code),
            'To_Level': to_level,
            'Relationship_Type': rel_type
        })

    return pd.DataFrame(data)


def save_edge_list(edge_df, output_dir=None):
    """
    Save edge list to Excel file.

    Parameters:
    -----------
    edge_df : pandas.DataFrame
        Edge list DataFrame
    output_dir : str or Path, optional
        Output directory

    Returns:
    --------
    Path
        Path to saved file
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_ISM

    script_dir = get_script_directory()
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "ism_edge_list.xlsx"

    try:
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            edge_df.to_excel(writer, sheet_name='Edge_List', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )

    print(f"Edge list saved to: {file_path}")
    return file_path


def print_digraph_summary(edges, levels, factor_levels, barriers, barrier_names=None):
    """
    Print summary of digraph construction.

    Parameters:
    -----------
    edges : list
        List of edges
    levels : dict
        Dict mapping level to factors
    factor_levels : dict
        Dict mapping factor to level
    barriers : list
        List of barrier codes
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names
    """
    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    edge_categories = filter_edges_by_levels(edges, factor_levels)

    print("-" * 70)
    print("ISM DIGRAPH CONSTRUCTION SUMMARY")
    print("-" * 70)
    print(f"Number of Barriers: {len(barriers)}")
    print(f"Number of Levels: {len(levels)}")
    print(f"Total Edges: {len(edges)}")
    print()

    print("EDGE EXTRACTION RULE:")
    print("  Direct edge from i to j exists IF:")
    print("    - k_ij = 1 (in initial binary matrix)")
    print("    - AND i ≠ j (not diagonal)")
    print()

    print("EDGE CATEGORIES:")
    print(f"  Influence (Cause→Effect): {len(edge_categories['upward'])}")
    print(f"  Feedback: {len(edge_categories['downward'])}")
    print(f"  Same Level: {len(edge_categories['intra_level'])}")
    print()

    print("LAYOUT:")
    print("  Level 1 at TOP (Effects/Outcomes)")
    print("  Higher levels at BOTTOM (Root Causes)")
    print("  Arrows point from causes to effects")
    print()

    print("NODES BY LEVEL:")
    for level_num in sorted(levels.keys()):
        factors = levels[level_num]
        codes = [barriers[i].upper() for i in factors]
        print(f"  Level {int_to_roman(level_num)}: {', '.join(codes)}")
    print()

    print("-" * 70)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def create_ism_digraph(irm=None, levels=None, factor_levels=None, barriers=None,
                        frm_results=None, lp_results=None, barrier_names=None, save=True):
    """
    Main function to create ISM digraph visualization.

    Parameters:
    -----------
    irm : numpy.ndarray, optional
        Initial Reachability Matrix (before transitivity).
    levels : dict, optional
        Dict mapping level number to list of factor indices.
    factor_levels : dict, optional
        Dict mapping factor index to level.
    barriers : list, optional
        List of barrier codes.
    frm_results : dict, optional
        Results from Module 10.
    lp_results : dict, optional
        Results from Module 11.
    barrier_names : dict, optional
        Dictionary mapping barrier codes to full names.
    save : bool, optional
        Whether to save outputs.

    Returns:
    --------
    dict
        Dictionary containing:
        - 'edges': List of edges
        - 'positions': Node positions
        - 'fig': Matplotlib figure
        - 'edge_df': Edge list DataFrame
        - 'output_files': Paths to saved files
    """
    print("\n" + "=" * 70)
    print("MODULE 12: ISM DIGRAPH CONSTRUCTION")
    print("=" * 70 + "\n")

    if barrier_names is None:
        barrier_names = BARRIER_NAMES

    # Step 1: Get required data
    if irm is None or levels is None or factor_levels is None or barriers is None:
        if lp_results is not None:
            levels = lp_results['levels']
            factor_levels = lp_results['factor_levels']
            barriers = lp_results['barriers']
            print("Using level data from provided Module 11 results.\n")
        else:
            print("Running Module 11 to get level partitioning...")
            lp_results = perform_level_partitioning(save=False)
            levels = lp_results['levels']
            factor_levels = lp_results['factor_levels']
            barriers = lp_results['barriers']
            print()

        if frm_results is None:
            print("Running Module 10 to get IRM...")
            frm_results = create_reachability_matrix(save=False)

        irm = frm_results['irm']

    max_level = max(levels.keys())
    n = len(barriers)

    print(f"Number of barriers: {n}")
    print(f"Number of levels: {max_level}")
    print(f"Barriers: {[b.upper() for b in barriers]}\n")

    # Step 2: Extract edges from IRM
    print("Extracting edges from Initial Reachability Matrix...")
    print("  Using IRM (before transitivity) for direct relationships only")
    edges = extract_edges(irm)
    print(f"  Found {len(edges)} direct edges.\n")

    # Step 3: Calculate positions
    print("Calculating hierarchical layout positions...")
    positions = calculate_hierarchical_positions(levels, barriers, max_level)
    print("  Positions calculated.\n")

    # Step 4: Print summary
    print_digraph_summary(edges, levels, factor_levels, barriers, barrier_names)

    # Step 5: Create visualization
    fig = None

    if MATPLOTLIB_AVAILABLE:
        print("\nCreating digraph visualization...")
        fig = create_digraph(
            edges, positions, levels, factor_levels, barriers,
            barrier_names, figsize=(FIGURE_WIDTH, FIGURE_HEIGHT)
        )
        print("  Visualization created.\n")
    else:
        print("\nMatplotlib not available. Skipping visualization.\n")

    # Step 6: Create edge list DataFrame
    edge_df = create_edge_list_dataframe(edges, barriers, factor_levels, barrier_names)

    # Step 7: Save outputs
    output_files = []
    if save:
        print("Saving outputs...")

        if fig is not None:
            path1 = save_digraph(fig, 'ism_diagraph.png')
            if path1:
                output_files.append(path1)

        path2 = save_edge_list(edge_df)
        output_files.append(path2)

    # Prepare results
    results = {
        'edges': edges,
        'positions': positions,
        'levels': levels,
        'factor_levels': factor_levels,
        'max_level': max_level,
        'fig': fig,
        'edge_df': edge_df,
        'barriers': barriers,
        'barrier_names': barrier_names,
        'irm': irm,
        'n': n,
        'output_files': output_files
    }

    print("\n" + "=" * 70)
    print("MODULE 12 COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 12 in standalone mode...")
    results = create_ism_digraph()

    # Display edge list
    print("\n" + "=" * 70)
    print("EDGE LIST")
    print("=" * 70)
    print(results['edge_df'].to_string(index=False))
