# Growth Rate Analysis Walkthrough

This tutorial demonstrates how to analyse how a crystal grows over time using CrystoGen simulation output.

---

## Prerequisites

- A folder of CrystoGen output containing:
  - Multiple XYZ files (one per time step, or a `size.csv` file)
  - A simulation parameters file listing crystallographic directions
- CGAspects is open

---

## Step 1 — Load the Data

1. Click **Import XYZ Files** and select your simulation folder
2. CGAspects scans for size files and detects available directions
3. If a size file is found, the **Calculate Growth Rates** button becomes active

!!! note
    If only XYZ files are present (no `size.csv`), CGAspects can still estimate sizes from the point cloud extents. The analysis options will reflect what data is available.

---

## Step 2 — Start Growth Rate Analysis

1. Click **Calculate Growth Rates** in the left panel
2. The Growth Rate configuration dialog opens

---

## Step 3 — Configure the Analysis

### Select Directions

Check the directions you want to analyse from the scrollable list. These correspond to the crystallographic directions defined in your CrystoGen simulation.

For a first look, check all available directions.

### Choose X-Axis Mode

| Mode | When to Use |
|------|-------------|
| Auto | Default — uses time column if available, otherwise row index |
| Force time | If you know a time column is present and want an error if missing |
| Row index | To use frame number rather than simulation time |

Leave as **Auto** for most cases.

### Click OK

The analysis starts. A progress bar shows computation progress.

---

## Step 4 — View Results in the Plotting Dialog

When analysis completes, the plotting dialog opens with growth rate results.

- **X axis**: Time (or row index)
- **Y axis**: Crystal dimension along each direction
- Each direction is plotted as a separate series

### Add a Trendline

1. Click **Trendline** in the plotting toolbar
2. Select **Linear fit** or **Polynomial fit**
3. The fitted line and equation appear on the plot

The slope of the linear fit gives the average growth rate (Å/time unit).

---

## Step 5 — Apply Smoothing

If your data is noisy:

1. Click **Smoothing** in the plotting toolbar
2. The Smoothing dialog opens — one tab per data series
3. For each series, choose:
   - **Method**: Savitzky-Golay is a good default (preserves peaks)
   - **Window size**: Larger window = more smoothing (try 11–21)
4. Click **Apply**

The smoothed data overlays the original. Use the **Legend Display Mode** to show both or just the processed series.

---

## Step 6 — Compare Directions

To compare growth along different directions:

1. In the plotting dialog, use **Colour by** to map direction names to different colours
2. Or generate separate plots for each direction using the permutation mode

Directions that grow faster have steeper slopes on the size-vs-time plot.

---

## Step 7 — Export

- **Save plot**: Click **Save** in the navigation toolbar
- **CSV data**: `aspects_output/growth_rates.csv` contains all size vs time data
- **Render**: Press `Ctrl+R` to save a 3D viewport snapshot of any frame

---

## Interpreting the Results

| Observation | Interpretation |
|-------------|---------------|
| Steep slope | Fast growth along that direction |
| Shallow slope | Slow growth (kinetically limited) |
| Plateau | Growth stopped (equilibrium reached) |
| Nonlinear curve | Growth rate changes over time (e.g., due to supersaturation depletion) |

For the scientific background on how growth rates relate to crystal morphology, see [Zingg Classification](../science/zingg-classification.md).
