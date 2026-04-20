# Basic Workflow

This tutorial walks through the typical end-to-end workflow in CGAspects: importing simulation data, visualising the crystal, running an analysis, and exporting results.

---

## Step 1 — Import XYZ Data

1. Launch CGAspects (`cgaspects`)
2. Click **Import XYZ Files** in the left panel, or use **File → Import** (`Ctrl+I`)
3. Browse to the folder containing your CrystoGen XYZ output files
4. Click **Open**

CGAspects scans the folder and loads all XYZ files. The 3D viewport updates to show the first crystal frame.

!!! tip
    If your folder contains a CrystoGen structure file, lattice parameters are loaded automatically.

---

## Step 2 — Explore the 3D Visualisation

- **Orbit**: Left-click and drag to rotate the crystal
- **Zoom**: Scroll the mouse wheel
- Press `R` to reset the view at any time

The **Crystal Info** panel on the right shows the aspect ratios, shape, volume, and surface area of the displayed frame.

If multiple XYZ frames were loaded, use the **frame slider** at the bottom to navigate between them, or press **Play** to animate the trajectory.

---

## Step 3 — (Optional) Set Lattice Parameters

If you want to use Miller indices for planes or fractional indices for directions:

1. Open **Crystallography → Set Lattice Parameters**
2. Enter your unit cell parameters (a, b, c, α, β, γ), or click **Load CIF** to import from a CIF file
3. Click **OK**

---

## Step 4 — (Optional) Add Crystallographic Planes and Directions

### Add a Plane

1. Open **Crystallography → Add Planes** (`Ctrl+Shift+L`)
2. Select **Miller** mode and enter h, k, l values (e.g., `1 0 0`)
3. Click **Add Plane**

The plane appears as a semi-transparent quad over the crystal.

### Add a Direction

1. Open **Crystallography → Add Directions** (`Ctrl+Shift+D`)
2. Select **Fractional** mode and enter [u v w] values (e.g., `1 0 0`)
3. Choose a rendering style (Arrow is recommended)
4. Click **Add Direction**

---

## Step 5 — Run an Analysis

Choose the analysis appropriate for your data:

| Button | When to Use |
|--------|-------------|
| Calculate Aspect Ratio | You want shape metrics and Zingg classification |
| Calculate Growth Rates | You have time-series size data |
| Calculate Site Analysis | You have crystallization event files |

Click the button. A configuration dialog opens — accept defaults or adjust options. Click **OK**.

The analysis runs in the background. A progress bar appears in the status bar.

---

## Step 6 — Explore Results in the Plotting Dialog

When the analysis completes, the [Plotting Dialog](../plotting.md) opens automatically with the results.

- Use the **X / Y / Colour by** dropdowns to choose what to plot
- Add a **trendline** if needed
- Use the **Filter** button to focus on a subset of data
- Add custom **Labels** for the title and axes

---

## Step 7 — Export

### Export Plot
Click **Save** in the plot navigation toolbar. Choose PNG, PDF, or EPS. Plots are also auto-saved to `aspects_output/CGPlots/`.

### Export Analysis Data
CSV files are automatically saved in `aspects_output/` when the analysis completes.

### Render the 3D Viewport
Press `Ctrl+R` or use **File → Render** to save the current 3D view as a PNG.

### Export Point Cloud
Use **File → Export XYZ** (`Ctrl+Shift+E`) to save the current point cloud as an XYZ file.

---

## Summary

```
Import XYZ folder
     ↓
Explore 3D viewport
     ↓
(Optional) Set lattice parameters
(Optional) Add planes and directions
     ↓
Run analysis (Aspect Ratio / Growth Rate / Site Analysis)
     ↓
Explore results in Plotting Dialog
     ↓
Export CSV, plots, or rendered image
```
