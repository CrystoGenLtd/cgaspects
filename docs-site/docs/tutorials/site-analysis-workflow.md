# Site Analysis Walkthrough

This tutorial shows how to analyse crystallization events at individual lattice sites using CGAspects.

---

## Prerequisites

- A CrystoGen simulation folder containing:
  - Crystallization event files
  - Population/occupation files
  - Site energy or coordinate data
- CGAspects is open with the folder imported

---

## Step 1 — Load the Data

1. Click **Import XYZ Files** and select the folder
2. If crystallization event files are detected, the **Calculate Site Analysis** button becomes active

!!! tip
    If the button is greyed out, the expected event files were not found. Check that the folder contains the CrystoGen output files (not just the XYZ shape files).

---

## Step 2 — Start Site Analysis

Click **Calculate Site Analysis**. The analysis runs in the background — a progress bar appears for large datasets.

---

## Step 3 — Explore Results in the Plotting Dialog

When complete, the plotting dialog opens with site analysis data.

### Key Variables to Explore

| Variable | Description |
|----------|-------------|
| energy | Site interaction energy — lower = more stable |
| occupation | Fraction of time the site was occupied |
| coordination | Number of occupied neighbours at crystallization |
| total_events | Total crystallization/dissolution event count |
| tile_type | Face / edge / corner classification |

Start with a scatter plot of **energy** vs **occupation** to see the relationship between site stability and how often each site is occupied.

---

## Step 4 — Highlight Sites in the 3D Viewport

Click a point in the scatter plot — the corresponding site is highlighted in the 3D viewport. This connects data features to spatial positions in the crystal.

To highlight multiple sites:

1. Open **View → Highlight Sites** (`Ctrl+Shift+S`)
2. Enter site numbers from the plot (e.g., `1-50` for the lowest-energy sites)
3. Choose a highlight colour
4. Click **Apply All**

The selected sites light up in the 3D viewport in your chosen colour.

---

## Step 5 — Filter the Data

Focus on a specific subset of sites using the Data Filter:

1. Click **Filter** in the plotting dialog toolbar
2. In the **Data Filters** tab, add a condition: e.g., `energy < -100`
3. Click **Apply**

The plot updates to show only sites matching the filter. The status bar shows how many sites passed the filter.

### Interaction Filters

Switch to the **Interaction Filters** tab to filter sites by the frequency of specific neighbour interaction types. This can isolate sites with particular crystallographic bonding environments.

---

## Step 6 — Hierarchical Clustering

To group sites by interaction pattern similarity:

1. In the plotting dialog, enable **Hierarchical Clustering**
2. The sites are grouped automatically and colour-coded by cluster membership
3. Plot **energy** vs **occupation** again — clusters of similar bonding environments appear as colour-grouped regions

This reveals structurally distinct site populations (e.g., terraces, steps, kinks).

---

## Step 7 — Export

- **Save plot**: Click **Save** in the navigation toolbar
- **CSV data**: `aspects_output/site_analysis.csv` and `interaction_data.csv`
- **Filtered export**: Apply a filter, then export only the filtered rows from the plotting dialog

---

## Interpreting the Results

| Site Property | High Value Means | Low Value Means |
|--------------|-----------------|-----------------|
| Energy | Weakly bound, prone to dissolution | Strongly bound, stable |
| Occupation | Site is frequently occupied | Site crystallizes rarely |
| Coordination | Well-surrounded (bulk-like) | Surface site (kink/edge) |
| Total events | High exchange rate | Stable once occupied |

Sites with high energy and low occupation are the most soluble (typically surface steps or kinks). Sites with low energy and high occupation are the most stable face sites.

For more background, see [Shape Metrics](../science/shape-metrics.md) and [Zingg Classification](../science/zingg-classification.md).
