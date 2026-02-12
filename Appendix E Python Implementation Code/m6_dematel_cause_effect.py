# SPDX-License-Identifier: PROPRIETARY
# File: m6_dematel_cause_effect.py
# Purpose: Module 6 - Generate DEMATEL Cause-Effect Diagram visualization with prominence-based bubble sizing

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import textwrap

# Import from Module 1
from m1_data_processing import (
    OUTPUT_DIR_DEMATEL,
    get_script_directory
)

# =============================================================================
# CONFIGURABLE PARAMETERS - Academic Publication Standards
# =============================================================================

# Bubble sizing parameters
MIN_BUBBLE_SIZE = 400           # Minimum marker size (for smallest D+R)
MAX_BUBBLE_SIZE = 2000          # Maximum marker size (for largest D+R)
SIZE_BASED_ON = "D+R"           # Column to base size on ("D+R" for prominence)

# Color scheme - Colorblind-safe academic palette
# Blue for causes (positive D-R), Orange for effects (negative D-R)
CAUSE_COLOR = '#D55E00'    # Vermillion/Orange (colorblind-safe)
EFFECT_COLOR = '#0072B2'     # Blue (colorblind-safe)
BUBBLE_ALPHA = 0.75             # Transparency for better overlap visibility
BUBBLE_EDGE_COLOR = '#333333'   # Dark gray edge for clarity
BUBBLE_EDGE_WIDTH = 1.0         # Edge line width

# Reference line (y=0, separating causes from effects)
SHOW_ZERO_LINE = True           # Show horizontal line at D-R = 0
ZERO_LINE_COLOR = '#666666'     # Neutral gray
ZERO_LINE_WIDTH = 1.5           # Line width
ZERO_LINE_STYLE = '--'          # Dashed line for academic clarity

# Vertical threshold line (x=mean(D+R), separating high/low prominence)
SHOW_PROMINENCE_LINE = True     # Show vertical line at mean(D+R)
PROMINENCE_LINE_COLOR = '#666666'  # Same neutral gray as horizontal line
PROMINENCE_LINE_WIDTH = 1.5     # Line width
PROMINENCE_LINE_STYLE = '--'    # Dashed line for consistency

# Four-quadrant classification labels
SHOW_QUADRANT_LABELS = True     # Show quadrant classification names
QUADRANT_FONT_SIZE = 15  # Smaller than barrier labels (LABEL_FONT_SIZE=11)
QUADRANT_FONT_COLOR = '#888888' # Subtle gray color
QUADRANT_FONT_STYLE = 'italic'  # Italic for distinction from barrier names
QUADRANT_ALPHA = 0.85           # Slight transparency

# Quadrant names (academic standard terminology)
QUADRANT_NAMES = {
    'I': 'I. Core Factors',           # High prominence, Cause (D-R > 0)
    'II': 'II. Driving Factors',       # Low prominence, Cause (D-R > 0)
    'III': 'III. Independent Factors',  # Low prominence, Effect (D-R < 0)
    'IV': 'IV. Impact Factors'         # High prominence, Effect (D-R < 0)
}

# Axis configuration
X_AXIS_LABEL = 'Prominence (D+R)'   # X-axis label with full description
Y_AXIS_LABEL = 'Relation (D-R)'     # Y-axis label with full description
AXIS_PADDING = 0.5                  # Padding around data points

# Label configuration
SHOW_LABELS = True              # Show barrier name labels outside bubbles
LABEL_FONT_SIZE = 12            # Font size for labels (increased for readability)
LABEL_FONT_FAMILY = 'Arial'  # Arial font for academic publications
LABEL_OFFSET_X = 2.0            # X offset from bubble center (base value, scaled dynamically)
LABEL_OFFSET_Y = 2.0            # Y offset from bubble center (base value, scaled dynamically)
LABEL_MAX_WIDTH = 20            # Max characters before wrapping (wider for better flow)

# Barrier code label configuration (inside bubbles)
SHOW_BARRIER_CODES = True       # Show barrier codes (B1, B2, etc.) inside bubbles
CODE_FONT_SIZE_MIN = 13        # Minimum font size for barrier codes (for smallest bubbles)
CODE_FONT_SIZE_MAX = 26         # Maximum font size for barrier codes (for largest bubbles)
CODE_FONT_COLOR = '#FFFFFF'     # White text for visibility on colored bubbles
CODE_FONT_WEIGHT = 'bold'       # Bold for readability

# Figure settings - Academic publication standards
FIGURE_WIDTH = 12               # Figure width in inches (fits journal column)
FIGURE_HEIGHT = 9               # Figure height in inches
FIGURE_DPI = 300                # Output resolution (publication standard)
FIGURE_BG_COLOR = '#FFFFFF'     # Pure white background (academic standard)
PLOT_BG_COLOR = '#FFFFFF'       # Pure white plot area

# Frame/border settings
SHOW_FRAME = True               # Show border around plot
FRAME_COLOR = '#000000'         # Black frame (professional)
FRAME_WIDTH = 1.0               # Thin frame line width

# Font settings - Sans-serif for academic publications
AXIS_LABEL_SIZE = 15           # Axis label font size
TICK_LABEL_SIZE = 10            # Tick label font size

# Grid settings
SHOW_GRID = True                # Light grid aids data reading
GRID_COLOR = '#E0E0E0'          # Very light gray
GRID_ALPHA = 0.7                # Grid transparency
GRID_LINESTYLE = '-'            # Solid thin lines
GRID_LINEWIDTH = 0.5            # Thin grid lines

# Legend settings
SHOW_LEGEND = True              # Show legend for cause/effect
LEGEND_LOCATION = 'center left'  # Legend position

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def load_prominence_relation_from_file(file_path=None):
    """
    Load Prominence (D+R) and Relation (D-R) data from Module 5 Excel file.
    
    Parameters:
    -----------
    file_path : str or Path, optional
        Path to the prominence/relation Excel file. If None, uses default path.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: Barrier, D, R, D-R, D+R
    """
    if file_path is None:
        script_dir = get_script_directory()
        file_path = script_dir / OUTPUT_DIR_DEMATEL / "dematel_pro_relation.xlsx"
    
    if not Path(file_path).exists():
        raise FileNotFoundError(
            f"Prominence/Relation file not found: {file_path}\n"
            "Please run Module 5 (m5_dematel_pro_relation.py) first to generate "
            "the Prominence and Relation data."
        )
    
    try:
        # Load the main Prominence_Relation sheet
        pro_rel_df = pd.read_excel(file_path, sheet_name='Prominence_Relation')
    except PermissionError:
        raise PermissionError(
            f"Cannot access file: {file_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    return pro_rel_df


def extract_plot_data(pro_rel_df):
    """
    Extract data needed for plotting from the DataFrame.
    
    Parameters:
    -----------
    pro_rel_df : pandas.DataFrame
        DataFrame from Module 12 with Barrier, D, R, D-R, D+R columns
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'barriers': List of barrier names
        - 'x_values': D+R (Prominence) values
        - 'y_values': D-R (Relation) values
        - 'd_values': D values
        - 'r_values': R values
    """
    barriers = pro_rel_df['Barrier'].tolist()
    
    # Handle different possible column names
    x_col = 'D+R' if 'D+R' in pro_rel_df.columns else 'D+R (Prominence)'
    y_col = 'D-R' if 'D-R' in pro_rel_df.columns else 'D-R (Relation)'
    
    x_values = pro_rel_df[x_col].values
    y_values = pro_rel_df[y_col].values
    d_values = pro_rel_df['D'].values
    r_values = pro_rel_df['R'].values
    
    return {
        'barriers': barriers,
        'x_values': x_values,
        'y_values': y_values,
        'd_values': d_values,
        'r_values': r_values
    }


def calculate_bubble_sizes(values, min_size=None, max_size=None):
    """
    Calculate bubble sizes based on values (normalized to size range).
    
    Larger values (more prominent) get larger bubbles.
    
    Parameters:
    -----------
    values : array-like
        Values to base sizes on (typically D+R)
    min_size : float, optional
        Minimum bubble size. If None, uses MIN_BUBBLE_SIZE.
    max_size : float, optional
        Maximum bubble size. If None, uses MAX_BUBBLE_SIZE.
        
    Returns:
    --------
    numpy.ndarray
        Array of bubble sizes
    """
    if min_size is None:
        min_size = MIN_BUBBLE_SIZE
    if max_size is None:
        max_size = MAX_BUBBLE_SIZE
    
    values = np.array(values)
    
    # Normalize values to [0, 1] range
    val_min = values.min()
    val_max = values.max()
    
    if val_max == val_min:
        # All values are the same, use middle size
        return np.full_like(values, (min_size + max_size) / 2, dtype=float)
    
    normalized = (values - val_min) / (val_max - val_min)
    
    # Scale to size range
    sizes = min_size + normalized * (max_size - min_size)
    
    return sizes


def format_label_text(text, max_width=None):
    """
    Format barrier name for display as label.
    
    Parameters:
    -----------
    text : str
        Full barrier name (e.g., "B1: Lack of time")
    max_width : int, optional
        Maximum characters per line. If None, uses LABEL_MAX_WIDTH.
        
    Returns:
    --------
    str
        Formatted text with line breaks if needed
    """
    if max_width is None:
        max_width = LABEL_MAX_WIDTH
    
    # Remove barrier code prefix for cleaner labels
    if ':' in text:
        # Keep just the description part
        text = text.split(':', 1)[1].strip()
    
    # Wrap text if too long
    if len(text) > max_width:
        lines = textwrap.wrap(text, width=max_width)
        return '\n'.join(lines)
    
    return text


def get_quadrant(x, y, x_threshold):
    """
    Determine which quadrant a point belongs to.

    Parameters:
    -----------
    x : float
        X coordinate (D+R, prominence)
    y : float
        Y coordinate (D-R, relation)
    x_threshold : float
        Vertical threshold (mean of D+R)

    Returns:
    --------
    str
        Quadrant number ('I', 'II', 'III', or 'IV')
    """
    if y > 0:  # Cause (above horizontal line)
        if x >= x_threshold:  # High prominence (right of vertical line)
            return 'I'   # Core Factors
        else:  # Low prominence (left of vertical line)
            return 'II'  # Driving Factors
    else:  # Effect (below horizontal line)
        if x < x_threshold:  # Low prominence (left of vertical line)
            return 'III'  # Independent Factors
        else:  # High prominence (right of vertical line)
            return 'IV'   # Impact Factors


def calculate_label_positions(x_values, y_values, sizes, barriers):
    """
    Calculate smart label positions that avoid overlapping ANY bubble or other labels.

    Uses a radial positioning algorithm that:
    1. Converts bubble sizes to data-coordinate radii
    2. Tests positions at various angles and distances around each bubble
    3. Selects the first position that doesn't overlap any bubble or label
    4. Uses leader lines to connect labels to bubbles

    Parameters:
    -----------
    x_values : array-like
        X coordinates (D+R)
    y_values : array-like
        Y coordinates (D-R)
    sizes : array-like
        Bubble sizes (matplotlib scatter size units)
    barriers : list
        Barrier names

    Returns:
    --------
    list of tuples
        List of (label_x, label_y, ha, va) in data coordinates for each label
    """
    n = len(x_values)
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    sizes = np.array(sizes)

    # Calculate data ranges
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Convert bubble sizes to approximate data-coordinate radii
    # Scatter size is in points^2, so radius in points = sqrt(size)/2
    # We need to convert to data coordinates
    fig_width_pts = FIGURE_WIDTH * 72  # inches to points
    fig_height_pts = FIGURE_HEIGHT * 72

    # Approximate conversion: data units per point (accounting for plot area ~70% of figure)
    x_pts_to_data = x_range / (fig_width_pts * 0.70)
    y_pts_to_data = y_range / (fig_height_pts * 0.70)

    # Calculate bubble radii in data coordinates
    bubble_radii_pts = np.sqrt(sizes) / 2
    bubble_radii_x = bubble_radii_pts * x_pts_to_data
    bubble_radii_y = bubble_radii_pts * y_pts_to_data

    # Estimate label dimensions in data coordinates
    # Balance between avoiding overlaps and keeping labels close
    label_width = 0.16 * x_range   # Approximate label width
    label_height = 0.14 * y_range  # Approximate label height (increased for multi-line labels)

    # Track placed labels: list of (center_x, center_y, width, height)
    placed_labels = []

    def rectangles_overlap(x1, y1, w1, h1, x2, y2, w2, h2):
        """Check if two rectangles overlap."""
        return not (x1 + w1/2 < x2 - w2/2 or  # rect1 is left of rect2
                    x1 - w1/2 > x2 + w2/2 or  # rect1 is right of rect2
                    y1 + h1/2 < y2 - h2/2 or  # rect1 is below rect2
                    y1 - h1/2 > y2 + h2/2)    # rect1 is above rect2

    def label_overlaps_bubble(lx, ly, lw, lh, exclude_idx):
        """Check if a label rectangle overlaps any bubble."""
        for j in range(n):
            if j == exclude_idx:
                continue
            bx, by = x_values[j], y_values[j]
            # Treat bubble as rectangle for simpler collision detection
            bw = bubble_radii_x[j] * 2.5  # Add padding
            bh = bubble_radii_y[j] * 2.5
            if rectangles_overlap(lx, ly, lw, lh, bx, by, bw, bh):
                return True
        return False

    def label_overlaps_labels(lx, ly, lw, lh):
        """Check if a label overlaps any already-placed label."""
        for (px, py, pw, ph) in placed_labels:
            if rectangles_overlap(lx, ly, lw, lh, px, py, pw, ph):
                return True
        return False

    def get_alignment(angle):
        """Get text alignment based on angle from bubble center."""
        # angle in degrees, 0 = right, 90 = up, 180 = left, 270 = down
        if -45 <= angle < 45:
            return 'left', 'center'
        elif 45 <= angle < 135:
            return 'center', 'bottom'
        elif angle >= 135 or angle < -135:
            return 'right', 'center'
        else:  # -135 <= angle < -45
            return 'center', 'top'

    def find_label_position(idx, angles, distances):
        """Find a non-overlapping position for label at various angles and distances."""
        x, y = x_values[idx], y_values[idx]
        bubble_rx, bubble_ry = bubble_radii_x[idx], bubble_radii_y[idx]

        for dist_mult in distances:
            for angle_deg in angles:
                angle_rad = np.radians(angle_deg)
                # Calculate label center position
                # Distance from bubble edge, scaled by bubble radius
                dist_x = (bubble_rx + label_width/2) * dist_mult
                dist_y = (bubble_ry + label_height/2) * dist_mult

                lx = x + np.cos(angle_rad) * dist_x
                ly = y + np.sin(angle_rad) * dist_y

                # Check bounds (keep labels within plot area with margin)
                margin_x = 0.05 * x_range
                margin_y = 0.08 * y_range
                if (lx - label_width/2 < x_min - margin_x or
                    lx + label_width/2 > x_max + margin_x * 3 or
                    ly - label_height/2 < y_min - margin_y or
                    ly + label_height/2 > y_max + margin_y):
                    continue

                # Avoid legend area (upper right corner)
                legend_x_threshold = x_max - 0.25 * x_range
                legend_y_threshold = y_max - 0.35 * y_range
                if lx > legend_x_threshold and ly > legend_y_threshold:
                    continue

                # Check for overlaps
                if (not label_overlaps_bubble(lx, ly, label_width, label_height, idx) and
                    not label_overlaps_labels(lx, ly, label_width, label_height)):
                    ha, va = get_alignment(angle_deg)
                    placed_labels.append((lx, ly, label_width, label_height))
                    return (lx, ly, ha, va)

        # Fallback: place far away in a safe direction
        # Use angle based on position relative to center of plot
        fallback_angle = np.degrees(np.arctan2(y - np.mean(y_values), x - np.mean(x_values)))
        angle_rad = np.radians(fallback_angle)
        lx = x + np.cos(angle_rad) * label_width * 3
        ly = y + np.sin(angle_rad) * label_height * 3
        ha, va = get_alignment(fallback_angle)
        placed_labels.append((lx, ly, label_width, label_height))
        return (lx, ly, ha, va)

    # Process barriers: prioritize isolated ones first, then clustered
    # Calculate "crowding score" for each barrier
    def crowding_score(idx):
        x, y = x_values[idx], y_values[idx]
        score = 0
        for j in range(n):
            if j != idx:
                dx = abs(x_values[j] - x) / x_range
                dy = abs(y_values[j] - y) / y_range
                dist = np.sqrt(dx**2 + dy**2)
                if dist < 0.3:  # Nearby bubbles increase score
                    score += (0.3 - dist)
        return score

    # Sort by crowding (least crowded first)
    processing_order = sorted(range(n), key=lambda i: (crowding_score(i), -y_values[i]))

    # Define preferred angles based on bubble position
    # Spread angles more for better distribution
    label_positions = [None] * n

    # Calculate the actual quadrant thresholds
    x_mean = np.mean(x_values)  # Vertical threshold (D+R mean)
    y_threshold = 0  # Horizontal threshold (D-R = 0)

    for idx in processing_order:
        x, y = x_values[idx], y_values[idx]

        # Determine preferred angles based on ACTUAL QUADRANT (not normalized position)
        # Use D-R = 0 as the horizontal boundary and mean(D+R) as vertical boundary
        is_cause = y > y_threshold  # Above D-R=0 line = Cause
        is_high_prominence = x > x_mean  # Right of mean = High prominence

        # Generate angles that keep labels in same quadrant as their bubble
        # Also consider position WITHIN quadrant to spread labels
        x_norm_in_quadrant = (x - x_min) / x_range  # 0=left, 1=right
        y_norm_in_quadrant = (y - y_min) / y_range  # 0=bottom, 1=top

        base_angles = []
        if is_cause:  # Upper half (Quadrants I and II)
            if is_high_prominence:  # Quadrant I - Core Factors (upper-right)
                # Spread labels based on position within quadrant
                # B4: y_norm=0.689, B6: y_norm=0.557, B7: y_norm=0.469
                if y_norm_in_quadrant > 0.65:  # Top bubbles like B4 (highest D-R)
                    # Prefer upward/rightward angles
                    base_angles = [60, 45, 75, 30, 90, 50, 40, 70, 20, 80]
                elif y_norm_in_quadrant < 0.50:  # Lower bubbles like B7
                    # Prefer rightward angles (space to the right)
                    base_angles = [30, 15, 45, 0, 60, -15, 20, 40, 10, 50]
                else:
                    # Middle positions like B6 - prefer leftward to avoid B4 above
                    base_angles = [150, 165, 135, 180, 170, 120, 160, 175]
            else:  # Quadrant II - Driving Factors (upper-left)
                # Prefer upward and leftward angles to stay in Quadrant II
                base_angles = [135, 120, 150, 105, 165, 90, 180, 140, 130, 160, 110, 170]
        else:  # Lower half (Quadrants III and IV) - Effects
            if is_high_prominence:  # Quadrant IV - Impact Factors (lower-right)
                # Spread labels based on position within quadrant
                if x_norm_in_quadrant > 0.7:  # Rightmost bubbles (like B1)
                    # Prefer rightward angles
                    base_angles = [0, -15, 15, -30, 30, -45, 45, -60]
                elif y_norm_in_quadrant < 0.3:  # Bottom bubbles (like B8)
                    # Prefer downward angles to go below
                    base_angles = [-90, -75, -105, -60, -120, -45, -135, -80]
                else:
                    # Default for middle positions
                    base_angles = [-45, -30, -60, -15, -75, 0, -90, -50, -40, -70, -20, -80]
            else:  # Quadrant III - Independent Factors (lower-left)
                # Prefer downward and leftward angles to stay in Quadrant III
                base_angles = [-135, -120, -150, -105, -165, -90, 180, -140, -130, -160, -110, -170]

        # Add more angles for fallback (still prefer same-quadrant angles first)
        all_angles = base_angles + [a for a in range(0, 360, 15) if a not in base_angles]

        # Try increasing distances - use larger distances for Quadrant IV to avoid overlaps
        if not is_cause and is_high_prominence:  # Quadrant IV
            distances = [1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.2]
        else:
            distances = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4]

        pos = find_label_position(idx, all_angles, distances)
        label_positions[idx] = pos

    return label_positions


def calculate_code_font_size(bubble_size):
    """
    Calculate appropriate font size for barrier code based on bubble size.

    Ensures the code text fits within the bubble without overlapping borders.

    Parameters:
    -----------
    bubble_size : float
        Bubble size in matplotlib scatter units (points squared)

    Returns:
    --------
    float
        Font size in points
    """
    # Bubble radius in points = sqrt(size) / 2
    # For text to fit comfortably, font size should be ~40-50% of diameter
    radius_pts = np.sqrt(bubble_size) / 2

    # Calculate font size as fraction of diameter (0.45 works well for 2-char codes)
    calculated_size = radius_pts * 0.9

    # Clamp to min/max range
    font_size = max(CODE_FONT_SIZE_MIN, min(CODE_FONT_SIZE_MAX, calculated_size))

    return font_size


def draw_bubble(ax, x, y, size, is_cause=True, barrier_code=None, alpha=None, edge_color=None, edge_width=None):
    """
    Draw a clean, academic-standard bubble marker with optional barrier code.

    Uses colorblind-safe colors to distinguish causes from effects.
    Single-layer design following academic visualization best practices.
    Font size for barrier codes is dynamically calculated based on bubble size.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to draw on
    x, y : float
        Center coordinates
    size : float
        Bubble size (matplotlib scatter size units)
    is_cause : bool
        True for cause (D-R > 0), False for effect (D-R < 0)
    barrier_code : str, optional
        Barrier code (e.g., "B1", "B2") to display inside the bubble
    alpha : float, optional
        Transparency
    edge_color : str, optional
        Edge outline color
    edge_width : float, optional
        Edge line width
    """
    if alpha is None:
        alpha = BUBBLE_ALPHA
    if edge_color is None:
        edge_color = BUBBLE_EDGE_COLOR
    if edge_width is None:
        edge_width = BUBBLE_EDGE_WIDTH

    # Select color based on cause/effect classification
    fill_color = CAUSE_COLOR if is_cause else EFFECT_COLOR

    # Single clean scatter point - academic standard (no 3D effects)
    ax.scatter(x, y, s=size, c=fill_color, alpha=alpha,
               edgecolors=edge_color, linewidths=edge_width, zorder=3)

    # Add barrier code inside the bubble if provided
    if barrier_code and SHOW_BARRIER_CODES:
        # Calculate dynamic font size based on bubble size
        code_font_size = calculate_code_font_size(size)
        ax.text(x, y, barrier_code, fontsize=code_font_size, fontweight=CODE_FONT_WEIGHT,
                color=CODE_FONT_COLOR, ha='center', va='center', zorder=4,
                fontfamily=LABEL_FONT_FAMILY)


def create_cause_effect_diagram(plot_data, bubble_sizes):
    """
    Create the DEMATEL Cause-Effect Diagram figure with four-quadrant classification.

    Follows academic publication standards:
    - White background
    - Colorblind-safe palette (blue=cause, orange=effect)
    - Sans-serif fonts
    - Clean design without decorative elements
    - Four quadrants with classification labels
    - Clear legend

    Parameters:
    -----------
    plot_data : dict
        Dictionary with barriers, x_values, y_values, etc.
    bubble_sizes : numpy.ndarray
        Calculated bubble sizes for each barrier

    Returns:
    --------
    matplotlib.figure.Figure
        The cause-effect diagram figure
    """
    barriers = plot_data['barriers']
    x_values = plot_data['x_values']
    y_values = plot_data['y_values']
    n = len(barriers)

    # Create figure with academic-standard white background
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    # Set white background (academic standard)
    fig.patch.set_facecolor(FIGURE_BG_COLOR)
    ax.set_facecolor(PLOT_BG_COLOR)

    # Calculate axis limits with padding
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Calculate mean of D+R for vertical threshold line
    x_mean = np.mean(x_values)

    # Add padding for labels (increased padding to prevent label cutoff)
    x_pad = max(AXIS_PADDING, x_range * 0.22)
    y_pad = max(AXIS_PADDING, y_range * 0.28)

    # Set axis limits
    plot_x_min = x_min - x_pad
    plot_x_max = x_max + x_pad
    plot_y_min = y_min - y_pad
    plot_y_max = y_max + y_pad

    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)

    # Draw grid first (behind everything) if enabled
    if SHOW_GRID:
        ax.grid(True, color=GRID_COLOR, alpha=GRID_ALPHA,
                linestyle=GRID_LINESTYLE, linewidth=GRID_LINEWIDTH, zorder=0)

    # Draw reference line at y=0 (separating causes from effects)
    if SHOW_ZERO_LINE:
        ax.axhline(y=0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH,
                   linestyle=ZERO_LINE_STYLE, zorder=1)

    # Draw vertical reference line at x=mean(D+R) (separating high/low prominence)
    if SHOW_PROMINENCE_LINE:
        ax.axvline(x=x_mean, color=PROMINENCE_LINE_COLOR, linewidth=PROMINENCE_LINE_WIDTH,
                   linestyle=PROMINENCE_LINE_STYLE, zorder=1)

    # Add quadrant classification labels
    if SHOW_QUADRANT_LABELS:
        # Calculate positions for quadrant labels (in corners of each quadrant)
        # Use small offset from the quadrant boundaries
        label_margin_x = x_range * 0.03
        label_margin_y = y_range * 0.05

        # Quadrant I: Core Factors (top-right)
        ax.text(plot_x_max - label_margin_x, plot_y_max - label_margin_y,
                QUADRANT_NAMES['I'], fontsize=QUADRANT_FONT_SIZE,
                fontfamily=LABEL_FONT_FAMILY, fontstyle=QUADRANT_FONT_STYLE,
                color=QUADRANT_FONT_COLOR, ha='right', va='top',
                alpha=QUADRANT_ALPHA, zorder=2)

        # Quadrant II: Driving Factors (top-left)
        ax.text(plot_x_min + label_margin_x, plot_y_max - label_margin_y,
                QUADRANT_NAMES['II'], fontsize=QUADRANT_FONT_SIZE,
                fontfamily=LABEL_FONT_FAMILY, fontstyle=QUADRANT_FONT_STYLE,
                color=QUADRANT_FONT_COLOR, ha='left', va='top',
                alpha=QUADRANT_ALPHA, zorder=2)

        # Quadrant III: Independent Factors (bottom-left)
        ax.text(plot_x_min + label_margin_x, plot_y_min + label_margin_y,
                QUADRANT_NAMES['III'], fontsize=QUADRANT_FONT_SIZE,
                fontfamily=LABEL_FONT_FAMILY, fontstyle=QUADRANT_FONT_STYLE,
                color=QUADRANT_FONT_COLOR, ha='left', va='bottom',
                alpha=QUADRANT_ALPHA, zorder=2)

        # Quadrant IV: Impact Factors (bottom-right)
        ax.text(plot_x_max - label_margin_x, plot_y_min + label_margin_y,
                QUADRANT_NAMES['IV'], fontsize=QUADRANT_FONT_SIZE,
                fontfamily=LABEL_FONT_FAMILY, fontstyle=QUADRANT_FONT_STYLE,
                color=QUADRANT_FONT_COLOR, ha='right', va='bottom',
                alpha=QUADRANT_ALPHA, zorder=2)

    # Draw bubbles with cause/effect color distinction and barrier codes
    for i in range(n):
        is_cause = y_values[i] > 0  # Positive D-R = Cause
        # Extract barrier code (e.g., "B1" from "B1: Lack of time")
        barrier_code = barriers[i].split(':')[0].strip() if ':' in barriers[i] else f"B{i+1}"
        draw_bubble(ax, x_values[i], y_values[i], bubble_sizes[i], is_cause=is_cause, barrier_code=barrier_code)

    # Add labels with improved positioning and leader lines
    if SHOW_LABELS:
        label_positions = calculate_label_positions(x_values, y_values, bubble_sizes, barriers)

        for i in range(n):
            bubble_x, bubble_y = x_values[i], y_values[i]
            label_x, label_y, ha, va = label_positions[i]

            label_text = format_label_text(barriers[i])

            # Add annotation with leader line connecting label to bubble
            ax.annotate(
                label_text,
                (bubble_x, bubble_y),  # Point to the bubble center
                xytext=(label_x, label_y),  # Label position in data coordinates
                textcoords='data',
                fontsize=LABEL_FONT_SIZE,
                fontfamily=LABEL_FONT_FAMILY,
                ha=ha,
                va=va,
                color='#000000',
                zorder=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.9, linewidth=0.5),
                arrowprops=dict(
                    arrowstyle='-',  # Simple line (no arrowhead)
                    color='#888888',
                    linewidth=0.8,
                    shrinkA=0,  # Don't shrink at label end
                    shrinkB=8,  # Shrink at bubble end to not overlap bubble
                    connectionstyle='arc3,rad=0'  # Straight line
                )
            )

    # Set axis labels with academic formatting
    ax.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE, fontfamily=LABEL_FONT_FAMILY,
                  fontweight='bold', labelpad=10)
    ax.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_SIZE, fontfamily=LABEL_FONT_FAMILY,
                  fontweight='bold', labelpad=10)

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

    # Configure frame/border (clean black border)
    if SHOW_FRAME:
        for spine in ax.spines.values():
            spine.set_edgecolor(FRAME_COLOR)
            spine.set_linewidth(FRAME_WIDTH)

    # Add legend for cause/effect distinction and bubble size explanation
    if SHOW_LEGEND:
        # Create custom legend handles
        cause_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=CAUSE_COLOR,
                              markersize=12, label='Cause (D-R > 0)', markeredgecolor=BUBBLE_EDGE_COLOR,
                              markeredgewidth=0.5)
        effect_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=EFFECT_COLOR,
                               markersize=12, label='Effect (D-R < 0)', markeredgecolor=BUBBLE_EDGE_COLOR,
                               markeredgewidth=0.5)
        threshold_handle = Line2D([0], [0], color=ZERO_LINE_COLOR, linestyle=ZERO_LINE_STYLE,
                                  linewidth=ZERO_LINE_WIDTH,
                                  label=f'Thresholds (D-R=0, D+R={x_mean:.2f})')

        # Bubble size legend - showing relative relationship
        size_large = Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                            markersize=18, markeredgecolor=BUBBLE_EDGE_COLOR,
                            markeredgewidth=1.0, label='↑ Higher Prominence (D+R)')
        size_small = Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                            markersize=8, markeredgecolor=BUBBLE_EDGE_COLOR,
                            markeredgewidth=1.0, label='↓ Lower Prominence (D+R)')

        legend = ax.legend(handles=[cause_handle, effect_handle, threshold_handle,
                                    size_large, size_small],
                           loc=LEGEND_LOCATION, fontsize=TICK_LABEL_SIZE,
                           frameon=True, fancybox=False, edgecolor='#CCCCCC',
                           framealpha=0.95)
        legend.get_frame().set_linewidth(0.5)

    # Title removed for cleaner academic appearance
    # (Title can be added in the paper/thesis caption instead)

    plt.tight_layout()

    # Add abbreviation note below x-axis as figure footnote
    fig.text(0.05, 0.02, 'KM = Knowledge Management',
             fontsize=9, fontfamily=LABEL_FONT_FAMILY,
             fontstyle='italic', color='#666666',
             verticalalignment='bottom', horizontalalignment='left')

    return fig


def create_plot_data_df(plot_data, bubble_sizes):
    """
    Create DataFrame with plot data for export.

    Parameters:
    -----------
    plot_data : dict
        Dictionary with barriers, x_values, y_values, etc.
    bubble_sizes : numpy.ndarray
        Calculated bubble sizes

    Returns:
    --------
    pandas.DataFrame
        DataFrame with plot coordinates and sizes
    """
    x_values = plot_data['x_values']
    y_values = plot_data['y_values']
    x_mean = np.mean(x_values)

    # Determine quadrant for each barrier
    quadrants = []
    quadrant_names = []
    for x, y in zip(x_values, y_values):
        q = get_quadrant(x, y, x_mean)
        quadrants.append(q)
        quadrant_names.append(QUADRANT_NAMES[q])

    return pd.DataFrame({
        'Barrier': plot_data['barriers'],
        'D': plot_data['d_values'],
        'R': plot_data['r_values'],
        'D+R (X-axis)': x_values,
        'D-R (Y-axis)': y_values,
        'Bubble_Size': bubble_sizes,
        'Role': ['Cause' if y > 0 else 'Effect' if y < 0 else 'Neutral'
                 for y in y_values],
        'Quadrant': quadrants,
        'Classification': quadrant_names
    })


def create_metadata_df(plot_data, bubble_sizes):
    """
    Create metadata DataFrame with visualization parameters.

    Parameters:
    -----------
    plot_data : dict
        Plot data dictionary
    bubble_sizes : numpy.ndarray
        Bubble sizes

    Returns:
    --------
    pandas.DataFrame
        Metadata information
    """
    x_values = plot_data['x_values']
    y_values = plot_data['y_values']
    n = len(plot_data['barriers'])
    x_mean = np.mean(x_values)

    n_causes = sum(1 for y in y_values if y > 0)
    n_effects = sum(1 for y in y_values if y < 0)

    # Count barriers in each quadrant
    quadrant_counts = {'I': 0, 'II': 0, 'III': 0, 'IV': 0}
    for x, y in zip(x_values, y_values):
        q = get_quadrant(x, y, x_mean)
        quadrant_counts[q] += 1

    # Find key barriers
    max_prom_idx = np.argmax(x_values)
    min_prom_idx = np.argmin(x_values)
    max_rel_idx = np.argmax(y_values)
    min_rel_idx = np.argmin(y_values)

    metadata = {
        'Parameter': [
            'Number of Barriers',
            'X-Axis',
            'Y-Axis',
            'Bubble Size Based On',
            'Horizontal Threshold (D-R)',
            'Vertical Threshold (D+R mean)',
            'Number of Causes (D-R > 0)',
            'Number of Effects (D-R < 0)',
            'Quadrant I (Core Factors)',
            'Quadrant II (Driving Factors)',
            'Quadrant III (Independent Factors)',
            'Quadrant IV (Impact Factors)',
            'Min D+R (Prominence)',
            'Max D+R (Prominence)',
            'Min D-R (Relation)',
            'Max D-R (Relation)',
            'Most Prominent Barrier',
            'Least Prominent Barrier',
            'Strongest Cause',
            'Strongest Effect',
            'Min Bubble Size',
            'Max Bubble Size'
        ],
        'Value': [
            n,
            'D+R (Prominence)',
            'D-R (Relation)',
            SIZE_BASED_ON,
            0,
            round(x_mean, 4),
            n_causes,
            n_effects,
            quadrant_counts['I'],
            quadrant_counts['II'],
            quadrant_counts['III'],
            quadrant_counts['IV'],
            round(min(x_values), 4),
            round(max(x_values), 4),
            round(min(y_values), 4),
            round(max(y_values), 4),
            plot_data['barriers'][max_prom_idx],
            plot_data['barriers'][min_prom_idx],
            plot_data['barriers'][max_rel_idx],
            plot_data['barriers'][min_rel_idx],
            MIN_BUBBLE_SIZE,
            MAX_BUBBLE_SIZE
        ]
    }

    return pd.DataFrame(metadata)


def save_outputs(fig, plot_df, metadata_df, output_dir=None):
    """
    Save figure and data to files.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The cause-effect diagram figure
    plot_df : pandas.DataFrame
        Plot data
    metadata_df : pandas.DataFrame
        Metadata
    output_dir : str or Path, optional
        Output directory
        
    Returns:
    --------
    tuple
        (image_path, excel_path)
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR_DEMATEL
    
    script_dir = get_script_directory()
    output_path = script_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    image_path = output_path / "dematel_cause_effect.png"
    fig.savefig(image_path, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor=FIGURE_BG_COLOR, edgecolor='none')
    print(f"Cause-Effect Diagram saved to: {image_path}")
    
    # Save Excel data
    excel_path = output_path / "dematel_cause_effect.xlsx"
    
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            plot_df.to_excel(writer, sheet_name='Plot_Data', index=False)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
    except PermissionError:
        raise PermissionError(
            f"Cannot write to file: {excel_path}\n"
            "Please close the Excel file if it's open and try again."
        )
    
    print(f"Plot data saved to: {excel_path}")
    
    return image_path, excel_path


def print_diagram_summary(plot_data, bubble_sizes):
    """
    Print summary of the cause-effect diagram with four-quadrant classification.

    Parameters:
    -----------
    plot_data : dict
        Plot data dictionary
    bubble_sizes : numpy.ndarray
        Bubble sizes
    """
    barriers = plot_data['barriers']
    x_values = plot_data['x_values']
    y_values = plot_data['y_values']
    n = len(barriers)
    x_mean = np.mean(x_values)

    print("-" * 90)
    print("DEMATEL CAUSE-EFFECT DIAGRAM SUMMARY (Four-Quadrant Classification)")
    print("-" * 90)
    print(f"Total Barriers: {n}")
    print(f"X-Axis: D+R (Prominence) - measures total involvement")
    print(f"Y-Axis: D-R (Relation) - separates causes (+) from effects (-)")
    print(f"Bubble Size: Based on {SIZE_BASED_ON} (larger = more significant)")
    print()
    print(f"THRESHOLD VALUES:")
    print(f"  Horizontal threshold (D-R):  0")
    print(f"  Vertical threshold (D+R):    {x_mean:.4f} (mean of D+R)")
    print()

    # Count barriers in each quadrant
    quadrant_barriers = {'I': [], 'II': [], 'III': [], 'IV': []}
    for i, (x, y) in enumerate(zip(x_values, y_values)):
        q = get_quadrant(x, y, x_mean)
        quadrant_barriers[q].append(barriers[i])

    print("FOUR-QUADRANT DISTRIBUTION:")
    print(f"  Quadrant I  (Core Factors - high prominence, cause):        {len(quadrant_barriers['I'])} barriers")
    print(f"  Quadrant II (Driving Factors - low prominence, cause):      {len(quadrant_barriers['II'])} barriers")
    print(f"  Quadrant III (Independent Factors - low prominence, effect): {len(quadrant_barriers['III'])} barriers")
    print(f"  Quadrant IV (Impact Factors - high prominence, effect):     {len(quadrant_barriers['IV'])} barriers")
    print()

    print("BARRIER POSITIONS:")
    print("-" * 90)
    print(f"{'Barrier':<40} {'D+R':<9} {'D-R':<9} {'Quadrant':<12} {'Classification':<20}")
    print("-" * 90)

    # Sort by quadrant (I, II, III, IV) then by D-R within quadrant
    def sort_key(i):
        q = get_quadrant(x_values[i], y_values[i], x_mean)
        q_order = {'I': 0, 'II': 1, 'III': 2, 'IV': 3}
        return (q_order[q], -y_values[i])

    sorted_indices = sorted(range(n), key=sort_key)

    for i in sorted_indices:
        name = barriers[i]
        display_name = name[:37] + "..." if len(name) > 37 else name
        q = get_quadrant(x_values[i], y_values[i], x_mean)
        classification = QUADRANT_NAMES[q]

        print(f"{display_name:<40} {x_values[i]:<9.4f} {y_values[i]:<9.4f} {q:<12} {classification:<20}")

    print("-" * 90)

    # Key findings
    max_prom_idx = np.argmax(x_values)
    min_prom_idx = np.argmin(x_values)
    max_rel_idx = np.argmax(y_values)
    min_rel_idx = np.argmin(y_values)

    print()
    print("KEY FINDINGS:")
    print(f"  Most Prominent (highest D+R):   {barriers[max_prom_idx]}")
    print(f"                                  D+R = {x_values[max_prom_idx]:.4f}")
    print(f"  Least Prominent (lowest D+R):   {barriers[min_prom_idx]}")
    print(f"                                  D+R = {x_values[min_prom_idx]:.4f}")
    print()
    print(f"  Strongest Cause (highest D-R):  {barriers[max_rel_idx]}")
    print(f"                                  D-R = {y_values[max_rel_idx]:.4f}")
    print(f"  Strongest Effect (lowest D-R):  {barriers[min_rel_idx]}")
    print(f"                                  D-R = {y_values[min_rel_idx]:.4f}")
    print("-" * 90)


def create_cause_effect_analysis(pro_rel_data=None, save=True, show=False):
    """
    Main function to create DEMATEL Cause-Effect Diagram.
    
    Parameters:
    -----------
    pro_rel_data : pandas.DataFrame, optional
        Prominence/Relation data from Module 5.
        If None, loads from dematel_pro_relation.xlsx
    save : bool, optional
        Whether to save outputs to files.
    show : bool, optional
        Whether to display the figure interactively.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'figure': matplotlib Figure object
        - 'plot_data': Dictionary with x_values, y_values, barriers
        - 'bubble_sizes': Array of bubble sizes
        - 'plot_df': DataFrame with plot coordinates
        - 'metadata_df': Metadata DataFrame
        - 'n_causes': Number of cause barriers
        - 'n_effects': Number of effect barriers
        - 'image_path': Path to saved image (if save=True)
        - 'excel_path': Path to saved Excel (if save=True)
    """
    print("\n" + "=" * 80)
    print("MODULE 6: DEMATEL - CAUSE-EFFECT DIAGRAM")
    print("=" * 80 + "\n")
    
    # Step 1: Load or use provided data
    if pro_rel_data is None:
        print("Loading Prominence and Relation data from Module 5...")
        pro_rel_df = load_prominence_relation_from_file()
        print(f"Loaded data for {len(pro_rel_df)} barriers.\n")
    else:
        pro_rel_df = pro_rel_data
        print(f"Using provided data ({len(pro_rel_df)} barriers).\n")
    
    # Step 2: Extract plot data
    print("Extracting plot coordinates...")
    plot_data = extract_plot_data(pro_rel_df)
    n = len(plot_data['barriers'])
    print(f"Barriers: {n}")
    print(f"X-axis range (D+R): {min(plot_data['x_values']):.4f} to {max(plot_data['x_values']):.4f}")
    print(f"Y-axis range (D-R): {min(plot_data['y_values']):.4f} to {max(plot_data['y_values']):.4f}\n")
    
    # Step 3: Calculate bubble sizes based on prominence
    print(f"Calculating bubble sizes based on {SIZE_BASED_ON}...")
    bubble_sizes = calculate_bubble_sizes(plot_data['x_values'])
    print(f"Size range: {min(bubble_sizes):.0f} to {max(bubble_sizes):.0f}\n")
    
    # Step 4: Print summary
    print_diagram_summary(plot_data, bubble_sizes)
    
    # Step 5: Create figure
    print("\nGenerating Cause-Effect Diagram...")
    fig = create_cause_effect_diagram(plot_data, bubble_sizes)
    print("Figure generated.\n")
    
    # Step 6: Create data DataFrames
    plot_df = create_plot_data_df(plot_data, bubble_sizes)
    metadata_df = create_metadata_df(plot_data, bubble_sizes)
    
    # Step 7: Save outputs
    image_path = None
    excel_path = None
    if save:
        print("Saving outputs...")
        image_path, excel_path = save_outputs(fig, plot_df, metadata_df)
    
    # Step 8: Show if requested
    if show:
        plt.show()
    
    # Count causes and effects
    n_causes = sum(1 for y in plot_data['y_values'] if y > 0)
    n_effects = sum(1 for y in plot_data['y_values'] if y < 0)
    
    # Prepare results
    results = {
        'figure': fig,
        'plot_data': plot_data,
        'bubble_sizes': bubble_sizes,
        'plot_df': plot_df,
        'metadata_df': metadata_df,
        'n_causes': n_causes,
        'n_effects': n_effects,
        'image_path': image_path,
        'excel_path': excel_path
    }
    
    print("\n" + "=" * 80)
    print("MODULE 6 COMPLETED SUCCESSFULLY")
    print("=" * 80 + "\n")
    
    return results


# =============================================================================
# STANDALONE EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Running Module 6 in standalone mode...")
    results = create_cause_effect_analysis(save=True, show=True)
    
    print(f"\nOutput files:")
    print(f"  Image: {results['image_path']}")
    print(f"  Excel: {results['excel_path']}")
    
    print(f"\nSummary:")
    print(f"  Total barriers: {len(results['plot_data']['barriers'])}")
    print(f"  Causes (D-R > 0): {results['n_causes']}")
    print(f"  Effects (D-R < 0): {results['n_effects']}")

