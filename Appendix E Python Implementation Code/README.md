# DEMATEL-ISM-MICMAC Integration Analysis

## Computational Appendix for Research Question 2

### Master Thesis: Knowledge Sharing Barriers in Energy Communities Development: A Cause-Effect Analysis

This repository accompanies the master thesis investigation into barriers that hinder effective knowledge sharing among stakeholders involved in European energy community project development. The analytical pipeline presented here operationalises an integrated DEMATEL-ISM-MICMAC methodology, coupling the Decision Making Trial and Evaluation Laboratory (DEMATEL) with Interpretive Structural Modelling (ISM) through a scientifically derived threshold. The code was written to ensure full transparency and reproducibility of every numerical step reported in the thesis.


## Data Collection

Eight knowledge-sharing barriers were identified through a systematic review of academic and grey literature, spanning individual (e.g., lack of time), organisational (e.g., limited resources) and technical (e.g., data management) dimensions. These barriers were then validated and assessed through semi-structured interviews with 26 stakeholders representing the Quadruple Helix model:

| Role Code | Stakeholder Group |
|-----------|-------------------|
| 1 | Academia / Research |
| 2 | Industry / Private Sector |
| 3 | Public Authority / Government |
| 4 | Civil Society / Community Citizen |

Survey data were collected using the SOSCI survey platform. Each respondent evaluated the pairwise influence between all barrier combinations on a scale from 0 (no influence) to 4 (very high influence).

---

## Knowledge-Sharing Barriers Under Investigation

| Code | Barrier |
|------|---------|
| B1 | Lack of time |
| B2 | Lack of financial and human resources |
| B3 | Lack of trust |
| B4 | Organisational policies not prioritising knowledge management |
| B5 | Language and geographical barriers |
| B6 | Poor digital platform integration and usability |
| B7 | Lack of systematic knowledge management processes |
| B8 | Diverse stakeholders and understanding gaps |

---

## Methodological Pipeline

The computation proceeds in two distinct phases comprising thirteen analytical modules, each encapsulated in its own Python script. A fourteenth script orchestrates the sequential execution of the entire pipeline.

### Phase 1: DEMATEL Analysis (Modules 1 through 6)

**Module 1 (`m1_data_processing.py`)**
Loads the raw survey responses collected via the SOSCI survey platform from the source spreadsheet, identifies influence-pair columns and rating columns through pattern matching, and computes the Direct Relation Matrix (DRM) as the arithmetic mean across all K respondents: A = (1/K) * sum of individual response matrices. Ratings recorded as NA or coded as -9 are treated as zero influence. The module also supports optional filtering by Quadruple Helix stakeholder role.

**Module 2 (`m2_dematel_normalized_drm.py`)**
Normalises the DRM so that all cell values fall within the closed interval [0, 1]. The normalisation scalar S is taken as the larger of the maximum row sum and the maximum column sum of the DRM, following the canonical DEMATEL procedure: D = A / S.

**Module 3 (`m3_dematel_trm.py`)**
Derives the Total Relation Matrix (TRM), which captures both direct and indirect influence pathways, using the geometric series formula T = D * (I - D)^(-1). A spectral radius check confirms that the Neumann series converges before the inversion is performed.

**Module 4 (`m4_dematel_trm_dar.py`)**
Computes the Dispatch vector (D, row sums of the TRM) and the Receive vector (R, column sums of the TRM) for each barrier. These vectors quantify, respectively, how much total influence a barrier exerts on the system and how much it absorbs from other barriers.

**Module 5 (`m5_dematel_pro_relation.py`)**
Calculates the Prominence index (D+R), which gauges the overall involvement of each barrier in the system, and the Relation index (D-R), which separates net causes (positive values) from net effects (negative values).

**Module 6 (`m6_dematel_cause_effect.py`)**
Generates a publication-quality bubble chart that plots every barrier on a Prominence vs. Relation coordinate plane. The diagram is partitioned into four quadrants (Core, Driving, Independent and Impact factors) using the mean of D+R as the vertical threshold and D-R = 0 as the horizontal threshold. Bubble diameter scales with prominence; the colour palette follows the Okabe-Ito scheme for colorblind accessibility.

### Phase 2: DEMATEL-ISM-MICMAC Integration (Modules 7 through 13)

**Module 7 (`m7_mmde_triplet.py`)**
Converts the TRM into a flat list of triplets (cell value, row index, column index) using one-based indexing, then sorts these triplets in descending order of cell value to produce the ordered set T*.

**Module 8 (`m8_mmde_sets.py`)**
Constructs two cumulative multisets at each position in the sorted sequence. The dispatch set T_d collects row indices (with repetition permitted), and the receive set T_r collects column indices. These growing sets form the input to the entropy calculations that follow.

**Module 9 (`m9_mmde_final.py`)**
Applies Shannon entropy to each cumulative set, derives the de-entropy (the gap between maximum possible entropy and observed entropy), and computes the Mean De-Entropy (MDE) at every position. The positions of peak MDE for the dispatch and receive sets (pos_D and pos_R) are identified, and the MMDE threshold lambda is determined as the minimum TRM value within the union of those two peak sets.

**Module 10 (`m10_ism_frm.py`)**
Binarises the TRM against the MMDE threshold: cells meeting or exceeding lambda become 1 and all others become 0, with the diagonal forced to 1 to satisfy the reflexivity axiom. A Boolean matrix power iteration then enforces the transitivity property, yielding the Final Reachability Matrix (FRM). Entries introduced solely through transitivity are flagged with an asterisk notation (1*). Driving power (row sums) and dependence power (column sums) are recorded for later classification.

**Module 11 (`m11_ism_lp.py`)**
Performs iterative level partitioning on the FRM. For each remaining barrier, the reachability set R(i), the antecedent set A(i), and their intersection C(i) are determined. Barriers satisfying R(i) = C(i) are assigned to the current level and removed; the process repeats until every barrier has been placed. Level 1 sits at the top of the hierarchy (most dependent outcomes) while the highest-numbered level contains the foundational root causes.

**Module 12 (`m12_ism_diagraph.py`)**
Constructs a hierarchical directed graph (digraph) from the level partitioning results and the Initial Reachability Matrix. Edges are drawn only for direct relationships (before transitivity), distinguishing influence arrows, feedback arrows and reciprocal (bidirectional) arrows through colour and style. The layout places Level I at the top and the deepest level at the bottom, in line with standard ISM conventions. All visual parameters follow Nature journal typographic specifications and use the Okabe-Ito colorblind-safe palette.

**Module 13 (`m13_ism_micmac.py`)**
Carries out a MICMAC (Cross-Impact Matrix Multiplication Applied to Classification) analysis, plotting each barrier on a Driving Power vs. Dependence Power scatter diagram. The plot area is divided at the n/2 threshold into four clusters following the original formulation by Warfield (1974): Autonomous (weak driver, weak dependent), Dependent (weak driver, strong dependent), Linkage (strong driver, strong dependent) and Independent (strong driver, weak dependent).

### Pipeline Runner

**Module 14 (`m14_run.py`)**
Orchestrates the end-to-end execution of Modules 1 through 13 in proper sequence, passing intermediate results between stages. It prints a comprehensive summary report upon completion, listing the MMDE threshold, hierarchical levels, MICMAC classification and all generated output files.

---

## Repository Structure

```
Appendix/
|
|-- m1_data_processing.py            Module 1:  Survey data loading and DRM computation
|-- m2_dematel_normalized_drm.py     Module 2:  DRM normalisation
|-- m3_dematel_trm.py                Module 3:  Total Relation Matrix derivation
|-- m4_dematel_trm_dar.py            Module 4:  Dispatch and Receive vectors
|-- m5_dematel_pro_relation.py       Module 5:  Prominence and Relation indices
|-- m6_dematel_cause_effect.py       Module 6:  Cause-effect bubble chart
|-- m7_mmde_triplet.py               Module 7:  MMDE triplet creation and sorting
|-- m8_mmde_sets.py                  Module 8:  Cumulative dispatch and receive sets
|-- m9_mmde_final.py                 Module 9:  Entropy, MDE and threshold calculation
|-- m10_ism_frm.py                   Module 10: Binary conversion, transitivity, FRM
|-- m11_ism_lp.py                    Module 11: Iterative level partitioning
|-- m12_ism_diagraph.py              Module 12: Hierarchical digraph visualisation
|-- m13_ism_micmac.py                Module 13: MICMAC driving-dependence classification
|-- m14_run.py                       Pipeline runner (Modules 1 through 13)
|-- requirements.txt                 Python package dependencies
```

---

## Installation and Execution

### Prerequisites

Python 3.9 or newer is required. All third-party packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

### Running the Complete Pipeline

To reproduce every result reported in the thesis, execute the pipeline runner from the repository root:

```bash
python m14_run.py
```

This will process the survey data, carry out both the DEMATEL and ISM analyses, generate all figures and tables, and save the outputs to the `output/dematel/` and `output/ism/` directories.

### Running Individual Modules

Each module can also be executed independently. For instance, to regenerate only the cause-effect diagram:

```bash
python m6_dematel_cause_effect.py
```

Standalone execution of any module beyond Module 1 requires that its upstream dependencies have already produced the necessary output files.

### Running the Appendix Script

For a minimal, visualisation-free reproduction of all numerical results:

```bash
python appendix_dematel_ism.py
```

---

## Output Artefacts

### DEMATEL Outputs (`output/dematel/`)

| File | Contents |
|------|----------|
| `dematel_drm.xlsx` | Direct Relation Matrix |
| `dematel_normalized_drm.xlsx` | Normalised DRM with validation metrics |
| `dematel_trm.xlsx` | Total Relation Matrix and intermediate matrices |
| `dematel_trm_dar.xlsx` | TRM augmented with D and R vectors |
| `dematel_pro_relation.xlsx` | Prominence (D+R) and Relation (D-R) table |
| `dematel_cause_effect.xlsx` | Plot coordinates and quadrant classification data |
| `dematel_cause_effect.png` | Cause-effect bubble chart (300 dpi) |

### ISM Outputs (`output/ism/`)

| File | Contents |
|------|----------|
| `mmde_triplet.xlsx` | Sorted triplet set T* |
| `mmde_sets.xlsx` | Cumulative T_d and T_r multisets |
| `mmde_mde.xlsx` | Entropy, de-entropy and MDE at each position |
| `mmde_final_results.xlsx` | Peak MDE positions and threshold value |
| `ism_irm.xlsx` | Initial Reachability Matrix (before transitivity) |
| `ism_frm.xlsx` | Final Reachability Matrix with driving/dependence power |
| `ism_lp_1.xlsx` ... `ism_lp_final.xlsx` | Level partitioning iterations and final assignment |
| `ism_diagraph.png` | Hierarchical digraph visualisation (300 dpi) |
| `ism_edge_list.xlsx` | Classified edge list for the digraph |
| `ism_micmac.xlsx` | MICMAC cluster assignments |
| `ism_micmac.png` | MICMAC scatter plot (300 dpi) |

---

## Configuration

All user-adjustable parameters are concentrated at the top of each module file, clearly separated from the computational logic. Notable configuration points include:

- **Data source**: Set `DATA_FILE` in `m1_data_processing.py` to point at the survey spreadsheet.
- **Role filtering**: Set `ROLE_FILTER` in the same module to `None` for the full sample or to an integer (1 through 4) to restrict the analysis to a single Quadruple Helix stakeholder group (1 = Academia/Research, 2 = Industry/Private Sector, 3 = Public Authority/Government, 4 = Civil Society/Community Citizen).
- **Barrier definitions**: Modify the `BARRIER_NAMES` dictionary to adapt the pipeline to a different set of factors.
- **Visualisation aesthetics**: Font sizes, colour palettes, figure dimensions and other graphical settings are exposed as named constants in Modules 6, 12 and 13.

---

## Key Mathematical Foundations

For a comprehensive treatment of all formulae, consult the accompanying `DEMATEL-ISM Integration procedure document.md`. The principal equations implemented in the codebase are summarised below.

| Concept | Formula |
|---------|---------|
| Direct Relation Matrix | A = (1/K) * sum(A_k) for k = 1 to K |
| Normalisation scalar | S = max(max row sum, max column sum) |
| Normalised DRM | D = A / S |
| Total Relation Matrix | T = D * (I - D)^(-1) |
| Dispatch (D) and Receive (R) | D_i = sum of row i of T; R_j = sum of column j of T |
| Prominence and Relation | (D+R), (D-R) |
| Shannon entropy | H = - sum(p_i * ln(p_i)) |
| De-entropy | H^D = ln(N) - H |
| Mean De-Entropy | MDE = H^D / N |
| MMDE threshold | lambda = min value in T_d(pos_D) union T_r(pos_R) |
| Binary conversion | k_ij = 1 if t_ij >= lambda, else 0; diagonal = 1 |
| Transitivity | K* = K OR (K AND K), iterated to convergence |
| Level partitioning | Assign level when R(i) = R(i) intersect A(i) |
| MICMAC threshold | n/2 for both driving and dependence power axes |

---

## Accessibility and Visualisation Standards

All figures produced by this pipeline adhere to the following conventions to maximise legibility and inclusivity:

- Colour palettes follow the Okabe-Ito scheme (Wong, 2011, Nature Methods 8:441), which is discernible under the most prevalent forms of colour vision deficiency.
- Fonts are set to Arial at sizes that satisfy Nature journal typographic guidelines.
- Output resolution is fixed at 300 dpi, meeting the minimum requirement of most academic publishers.
- Bubble charts and scatter plots employ distinct marker shapes and textual labels alongside colour cues, ensuring that information is never communicated through colour alone.

---

## Licence

This code is proprietary and forms part of the master thesis "Knowledge Sharing Barriers in Energy Communities Development: A Cause-Effect Analysis." It is shared here solely for the purposes of academic examination and peer review. Redistribution or commercial use without explicit written permission from the author is not permitted.
