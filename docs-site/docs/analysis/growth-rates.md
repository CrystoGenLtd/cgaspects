# Growth Rate Analysis

Access via the **Calculate Growth Rates** button in the left panel (enabled when size files are present in the loaded folder).

Growth rate analysis measures how the crystal grows along selected crystallographic directions over time, using size data from the CrystoGen simulation output.

---

## Input Data

The analysis reads from these files (auto-detected in the loaded folder):

| File | Contents |
|------|---------|
| `size.csv` | Crystal dimension along each direction at each time step |
| Simulation parameters file | Lists available crystallographic directions |
| Summary file (optional) | Supersaturation and thermodynamic metadata |

---

## Configuration Dialog

When you click **Calculate Growth Rates**, a dialog opens with:

### Direction Selection

A scrollable list of all available crystallographic directions from the simulation. Check any directions you want to include in the analysis.

### X-Axis Mode

| Mode | Behaviour |
|------|---------|
| Auto (time if available) | Use the `time` column if present; fall back to row index |
| Force time column | Use `time`; error if column is missing |
| Use row index | Always use integer row index, ignore time |

---

## Output

Results are saved to `aspects_output/` and include:

| File | Contents |
|------|---------|
| `growth_rates.csv` | Size vs time for each selected direction |
| Growth rate plots | Scatter/line plots per direction |

---

## Smoothing and Interpolation

After the initial analysis, a **Smoothing** dialog lets you post-process each data series:

### Smoothing Methods

| Method | Parameters | Description |
|--------|-----------|-------------|
| None | — | Raw data |
| Moving Average | Window size (3–51, odd) | Rolling mean |
| Savitzky-Golay | Window, polynomial order | Polynomial fit in rolling window |
| Gaussian | Window size | Gaussian-weighted kernel |
| LOWESS | Window size | Locally weighted regression |

### Interpolation Methods

| Method | Extra Points | Description |
|--------|-------------|-------------|
| None | — | No interpolation |
| Linear | N points | Linear interpolation between data points |
| Cubic Spline | N points | Smooth cubic spline |
| Polynomial | N points | Polynomial fit |

### Extrapolation Methods

| Method | Extra Points | Description |
|--------|-------------|-------------|
| None | — | No extrapolation |
| Polynomial | N points | Extrapolate beyond data range |
| Linear | N points | Linear extrapolation |

### Legend Display Mode

| Mode | Shows |
|------|-------|
| Original only | Unprocessed data series |
| Processed only | Smoothed/interpolated series |
| Both | Both the original and processed series overlaid |

---

## Plotting Results

Growth rate results are automatically displayed in the [Plotting Dialog](../plotting.md) after analysis. The X-axis is time (or row index), and the Y-axis is the crystal dimension along each direction.

Trendlines (linear or polynomial fit) can be added in the plotting dialog.

---

## Notes

- Multiple directions can be compared on the same plot
- Smoothing settings are per-series, so different directions can use different filters
- If the `time` column is present but non-monotonic, results may be unexpected — use "row index" mode in that case
- Analysis can be cancelled from **Tools → Active Threads**. Cancellation takes effect at the next checkpoint (typically the start of each iteration), so there may be a short delay before the task stops.
